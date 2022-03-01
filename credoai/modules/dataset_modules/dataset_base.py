import numpy as np
import pandas as pd
from credoai.modules.credo_module import CredoModule
from credoai.utils.common import NotRunError
from credoai.utils.dataset_utils import ColumnTransformerUtil
from credoai.utils.model_utils import get_gradient_boost_model
from itertools import combinations
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from typing import List, Optional


class DatasetModule(CredoModule):
    """Dataset module for Credo AI.

    This module takes in features and labels and provides functionality to perform dataset assessment

    Parameters
    ----------
    data : pd.DataFrame
        Dataset dataframe that includes all features and labels
    sensitive_feature_key : str
        Name of the sensitive feature column, like 'race' or 'gender'
    label_key : str
        Name of the label column
    categorical_features_keys : list[str], optional
        Names of the categorical features (including the sensitive feature, if applicable)
    """    
    def __init__(self,
                data: pd.DataFrame,
                sensitive_feature_key: str,
                label_key: str,
                categorical_features_keys: Optional[List[str]]=None):

        self.data = data
        self.sensitive_feature_key = sensitive_feature_key
        self.label_key = label_key
        self.X = data.drop(columns=[sensitive_feature_key, label_key])

        if categorical_features_keys:
            self.categorical_features_keys = categorical_features_keys
        else:
            self.categorical_features_keys = self._find_categorical_features()
    
    def run(self):
        """Runs the assessment process

        Returns
        -------
        dict, nested
            Key: assessment category
            Values: detailed results associated with each category
        """        
        sensitive_feature_prediction_results = self._run_cv()
        group_differences = self._group_differences()
        normalized_mutual_information = self._calculate_mutual_information()
        balance_metrics = self._assess_balance_metrics()
        self.results = {**sensitive_feature_prediction_results,
                        'standardized_group_diffs': group_differences,
                        'normalized_mutual_information': normalized_mutual_information,
                        **balance_metrics
                        }  
        return self  
    
    def prepare_results(self):
        if self.results is not None:
            return self.results
        else:
            raise NotRunError(
                "Results not created yet. Call 'run' to create results"
            )
    
    def _group_differences(self):
        """Calculates standardized mean differences

        It is performed for all numeric features and all possible group pairs combinations present in the sensitive feature.

        Returns
        -------
        dict, nested
            Key: sensitive feature groups pair
            Values: dict
                Key: name of feature
                Value: standardized mean difference
        """
        group_means = self.X.groupby(self.data[self.sensitive_feature_key]).mean()
        std = self.X.std(numeric_only=True)
        diffs = {}
        for group1, group2 in combinations(group_means.index, 2):
            diff = (group_means.loc[group1]-group_means.loc[group2])/std
            diffs[f'{group1}-{group2}'] = diff.to_dict()

        return diffs
    
    def _run_cv(self):
        """Determines cross-validated ROC-AUC score
        
        A model is trained on the features to predict the sensitive attribute.
        The score quantifies the performance of this prediction. 
        A high score means the data collectively serves as a proxy.

        Parameters
        ----------
        pipe : sklearn.pipeline
            Pipeline of transforms

        Returns
        -------
        ndarray
            Cross-validation score
        """
        results = {}
        pipe = self._make_pipe()
        sensitive_features = self.data[self.sensitive_feature_key].cat.codes
        scorer = make_scorer(roc_auc_score, 
                             needs_proba=True,
                             multi_class='ovo')
        cv_results = cross_val_score(pipe, self.X, sensitive_features,
                             cv = StratifiedKFold(5),
                             scoring = scorer,
                             error_score='raise')

        # Get feature importances by running once
        pipe.fit(self.X, sensitive_features)
        model = pipe['model']
        preprocessor = pipe['preprocessor']
        col_names = ColumnTransformerUtil.get_ct_feature_names(preprocessor)
        feature_importances = pd.Series(model.feature_importances_, 
            index=col_names).sort_values(ascending=False)
        results['sensitive_feature_prediction_score'] = cv_results.mean()
        results['sensitive_feature_prediction_feature_importances'] = feature_importances

        return results
    
    def _make_pipe(self):
        """Makes a pipeline

        Returns
        -------
        sklearn.pipeline
            Pipeline of scaler and model transforms
        """
        categorical_features = self.categorical_features_keys.copy()
        if self.sensitive_feature_key in categorical_features:
            categorical_features.remove(self.sensitive_feature_key)

        all_features = list(self.data.drop(columns=[self.label_key, self.sensitive_feature_key]).columns)
        numeric_features = [x for x in all_features if x not in categorical_features]

        # Define features tansformers
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")
        
        numeric_transformer = Pipeline(
            steps=[("scaler", StandardScaler())]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        model = get_gradient_boost_model()

        pipe = Pipeline(
            steps=[("preprocessor", preprocessor), ("model", model)]
        )
        
        return pipe

    
    def _find_categorical_features(self, threshold=0.05):
        """Identifies categorical features

        Logic: If type is not float and there are relatively few unique values for a feature, the feature is likely categorical.
        The results are estimates and are not guaranteed to be correct.

        Parameters
        ----------
        threshold : float
            The threshold (number of the unique values over the total number of values)

        Returns
        -------
        list
            Names of categorical features
        """
        float_cols = list(self.data.select_dtypes(include=[np.float]).columns)
        cat_cols = []
        for name, column in self.data.iteritems():
            if name not in float_cols and name != self.label_key:
                unique_count = column.unique().shape[0]
                total_count = column.shape[0]
                if unique_count / total_count < threshold:
                    cat_cols.append(name)

        return cat_cols

    def _calculate_mutual_information(self, normalize=True):
        """Calculates normalized mutual information between sensitive feature and other features

        Mutual information is the "amount of information" obtained about the sensitive feature by observing another feature.
        Mutual information is useful to proxy detection purposes.

        Parameters
        ----------
        normalize : bool, optional
            If True, calculated mutual information values are normalized
            Normalization is done via dividing by the mutual information between the sensitive feature and itself.

        Returns
        -------
        dict, nested
            Key: feature name
            Value: mutual information and considered feature type (categorical/continuous)
        """
        df = self.data.copy().drop(columns=[self.label_key])
        
        # Encode categorical features
        for col in self.categorical_features_keys:
            df[col] = df[col].astype("category").cat.codes

        discrete_features = [
            True if col in self.categorical_features_keys else False for col in df.columns
        ]
        
        # Use the right mutual information methods based on the feature type of the sensitive attribute
        if self.sensitive_feature_key in self.categorical_features_keys:
            mi = mutual_info_classif(
                df,
                df[self.sensitive_feature_key],
                discrete_features=discrete_features,
                random_state=42,
            )
        else:
            mi = mutual_info_regression(
                df,
                df[self.sensitive_feature_key],
                discrete_features=discrete_features,
                random_state=42,
            )

        # Normalize the mutual information values, if requested
        mi = pd.Series(mi, index=df.columns)
        if normalize:
            mi = mi / mi.max()

        # Create the results
        mi = mi.sort_index().to_dict()
        mutual_information_results = {}
        for k, v in mi.items():
            if k in self.categorical_features_keys:
                mutual_information_results[k] = {"value": v, "feature_type": "categorical"}
            else:
                mutual_information_results[k] = {"value": v, "feature_type": "continuous"}

        mutual_information_results[self.sensitive_feature_key]["feature_type"] = mutual_information_results[self.sensitive_feature_key]["feature_type"] + '_reference'

        return mutual_information_results
    
    def _assess_balance_metrics(self):
        """Calculate dataset balance statistics and metrics 

        Returns
        -------
        dict
            'sample_balance': distribution of samples across groups
            'label_balance': distribution of labels across groups
            'metrics': demographic parity difference and ratio between groups for all preferred label value possibilities 
        """
        balance_results = {}

        # Distribution of samples across groups
        sample_balance = (
            self.data.groupby([self.sensitive_feature_key])
            .agg(
                count=(self.label_key, len),
                percentage=(self.label_key, lambda x: 100.0 * len(x) / len(self.data)),
            )
            .reset_index()
            .to_dict(orient="records")
        )
        balance_results["sample_balance"] = sample_balance

        # Distribution of samples across groups
        label_balance = (
            self.data.groupby([self.sensitive_feature_key, self.label_key])
            .size()
            .unstack(fill_value=0)
            .stack()
            .reset_index(name="count")
            .to_dict(orient="records")
        )
        balance_results["label_balance"] = label_balance

        # Fairness metrics
        r = self.data.groupby([self.sensitive_feature_key, self.label_key])\
                        .agg({self.label_key: 'count'})\
                        .groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))\
                        .rename({self.label_key:'ratio'}, inplace=False, axis=1)\
                        .reset_index(inplace=False)

        # Compute the maximum difference between any two pairs of groups
        demographic_parity_difference = r.groupby(self.label_key)['ratio'].apply(lambda x: np.max(x)-np.min(x)).reset_index(name='value').to_dict(orient='records')

        # Compute the minimum ratio between any two pairs of groups
        demographic_parity_ratio = r.groupby(self.label_key)['ratio'].apply(lambda x: np.min(x)/np.max(x)).reset_index(name='value').to_dict(orient='records')
        
        balance_results['demographic_parity_difference'] = demographic_parity_difference
        balance_results['demographic_parity_ratio'] = demographic_parity_ratio
        return balance_results
