import numpy as np
import pandas as pd
from credoai.modules.credo_module import CredoModule
from credoai.utils.common import NotRunError, is_categorical
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


class DatasetFairness(CredoModule):
    """Dataset module for Credo AI.

    This module takes in features and labels and provides functionality to perform dataset assessment

    Parameters
    ----------
    X : pandas.DataFrame
        The features
    y : pandas.Series
        The outcome labels
    sensitive_features : pandas.Series
        A series of the sensitive feature labels (e.g., "male", "female") which should be used to create subgroups
    categorical_features_keys : list[str], optional
        Names of the categorical features
    categorical_threshold : float
        Parameter for automatically identifying categorical columns. See
        `credoai.utils.common.is_categorical`
    """    
    def __init__(self,
                X,
                y,
                sensitive_features: pd.Series,
                categorical_features_keys: Optional[List[str]]=None,
                categorical_threshold: float=0.05):

        self.data = pd.concat([X, y], axis=1)
        self.sensitive_features = sensitive_features
        self.X = X
        self.y = y

        # set up categorical features
        if categorical_features_keys:
            self.categorical_features_keys = categorical_features_keys.copy()
            if self.sensitive_features.name in self.categorical_features_keys:
                self.sensitive_features = self.sensitive_features.astype('category')
                self.categorical_features_keys.remove(self.sensitive_features.name)
        else:
            self.categorical_features_keys = self._find_categorical_features(categorical_threshold)
    
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
        self.results = {**balance_metrics,
                        **sensitive_feature_prediction_results,
                        'standardized_group_diffs': group_differences,
                        'normalized_mutual_information': normalized_mutual_information,
                        }  
        return self  
    
    def prepare_results(self):
        """Prepares results for export to Credo AI's Governance App

        Structures a subset of results for export as a dataframe with appropriate structure
        for exporting. See credoai.modules.credo_module.

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        NotRunError
            If results have not been run, raise
        """
        if self.results is not None:
            metric_types = ['sensitive_feature_prediction_score',
           'demographic_parity_difference',
           'demographic_parity_ratio']
            index = []
            prepared_arr = []
            for metric_type in metric_types:
                val = self.results[metric_type]
                # if multiple values were calculated for metric_type
                # add them all. Assumes each element of list is a dictionary with a "value" key,
                # and other optional keys as metricmetadata
                if isinstance(val, list):
                    for l in val:
                        index.append(metric_type)
                        prepared_arr.append(l)
                else:
                    # assumes the dictionary has a "value" key, along with other optional keys
                    # as metric metadata
                    if isinstance(val, dict):
                        tmp = val
                    elif isinstance(val, (int, float)):
                        tmp = {'value': val}
                    index.append(metric_type)
                    prepared_arr.append(tmp)
            return pd.DataFrame(prepared_arr, index=index).rename_axis(index='metric_type')
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
        group_means = self.X.groupby(self.sensitive_features).mean()
        std = self.X.std(numeric_only=True)
        diffs = {}
        for group1, group2 in combinations(group_means.index, 2):
            diff = (group_means.loc[group1]-group_means.loc[group2])/std
            diffs[f'{group1}-{group2}'] = diff.to_dict()
        return diffs
    
    def _run_cv(self):
        """Determines redundant encoding

        A model is trained on the features to predict the sensitive attribute.
        The score is cross-validated ROC-AUC score.
        It quantifies the performance of this prediction. 
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
        if is_categorical(self.sensitive_features):
            sensitive_features = self.sensitive_features.cat.codes
        else:
            sensitive_features = self.sensitive_features
        
        pipe = self._make_pipe()
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
        results['sensitive_feature_prediction_feature_importances'] = feature_importances.to_dict()

        return results
    
    def _make_pipe(self):
        """Makes a pipeline

        Returns
        -------
        sklearn.pipeline
            Pipeline of scaler and model transforms
        """
        categorical_features = self.categorical_features_keys.copy()
        numeric_features = [x for x in self.X.columns if x not in categorical_features]

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

    def _find_categorical_features(self, threshold):
        """Identifies categorical features

        Returns
        -------
        list
            Names of categorical features
        """
        if is_categorical(self.sensitive_features, threshold=threshold):
            self.sensitive_features = self.sensitive_features.astype('category')
        cat_cols = []
        for name, column in self.X.iteritems():
            if is_categorical(column, threshold=threshold):
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
        # Encode categorical features
        for col in self.categorical_features_keys:
            self.X[col] = self.X[col].astype("category").cat.codes

        discrete_features = [
            True if col in self.categorical_features_keys else False for col in self.X.columns
        ]
        
        # Use the right mutual information methods based on the feature type of the sensitive attribute
        if is_categorical(self.sensitive_features):
            sensitive_feature = self.sensitive_features.cat.codes
            mi = mutual_info_classif(
                self.X,
                self.sensitive_features.cat.codes,
                discrete_features=discrete_features,
                random_state=42,
            )
            ref = mutual_info_classif(sensitive_feature.values[:,None], sensitive_feature, 
                                        discrete_features=[True], random_state=42)[0]
        else:
            mi = mutual_info_regression(
                self.X,
                self.sensitive_features,
                discrete_features=discrete_features,
                random_state=42,
            )
            ref = mutual_info_regression(self.sensitive_feature.values[:,None], 
                                         self.sensitive_features,
                                         random_state=42)[0]

        # Normalize the mutual information values, if requested
        mi = pd.Series(mi, index=self.X.columns)
        if normalize:
            mi = mi / ref

        # Create the results
        mi = mi.sort_index().to_dict()
        mutual_information_results = {}
        for k, v in mi.items():
            if k in self.categorical_features_keys:
                mutual_information_results[k] = {"value": v, "feature_type": "categorical"}
            else:
                mutual_information_results[k] = {"value": v, "feature_type": "continuous"}

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
            self.y.groupby(self.sensitive_features)
            .agg(
                count=(len),
                percentage=(lambda x: 100.0 * len(x) / len(self.y)),
            )
            .reset_index()
            .to_dict(orient="records")
        )
        balance_results["sample_balance"] = sample_balance

        # Distribution of samples across groups
        label_balance = (
            self.data.groupby([self.sensitive_features, self.y.name])
            .size()
            .unstack(fill_value=0)
            .stack()
            .reset_index(name="count")
            .to_dict(orient="records")
        )
        balance_results["label_balance"] = label_balance

        # Fairness metrics
        r = self.data.groupby([self.sensitive_features, self.y.name])\
                        .agg({self.y.name: 'count'})\
                        .groupby(level=0).apply(lambda x: x / float(x.sum()))\
                        .rename({self.y.name:'ratio'}, inplace=False, axis=1)\
                        .reset_index(inplace=False)

        # Compute the maximum difference between any two pairs of groups
        demographic_parity_difference = r.groupby(self.y.name)['ratio'].apply(lambda x: np.max(x)-np.min(x)).reset_index(name='value').to_dict(orient='records')

        # Compute the minimum ratio between any two pairs of groups
        demographic_parity_ratio = r.groupby(self.y.name)['ratio'].apply(lambda x: np.min(x)/np.max(x)).reset_index(name='value').to_dict(orient='records')
        
        balance_results['demographic_parity_difference'] = demographic_parity_difference
        balance_results['demographic_parity_ratio'] = demographic_parity_ratio
        return balance_results
