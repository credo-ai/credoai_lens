import credoai.lens as cl
from credoai.modules.credo_module import CredoModule
from credoai.utils.common import NotRunError
from credoai.utils.model_utils import get_gradient_boost_model
from itertools import combinations
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

import pandas as pd

class DatasetModule(CredoModule):
    """Dataset module for Credo AI. 

    This module takes in features and labels and provides functionality to:
    - perform proxy analysis of features
    - generate fairness-oriented descriptive information 

    Parameters
    ----------
    X : (List, pandas.Series, numpy.ndarray)
        The features
    y : (List, pandas.Series, numpy.ndarray)
        The labels
    sensitive_features : (List, pandas.Series, numpy.ndarray)
        The sensitive features which should be used to create subgroups
    """    
    def __init__(self,
                 X,
                 y,
                 sensitive_features):                
        self.X = pd.DataFrame(X)
        self.y = y
        self.sensitive_features = sensitive_features
    
    def run(self):
        pipe = self._make_pipe()
        cv_results = self._run_cv(pipe)
        group_differences = self._group_differences()
        normalized_mutual_information = self._calculate_mutual_information()
        balance_metrics = self._assess_balance_metrics()
        self.results = {'overall_proxy_score': cv_results.mean(),
                        'group_diffs': group_differences,
                        'normalized_mutual_information': normalized_mutual_information,
                        'balance_metrics': balance_metrics}  
        return self  
    
    def prepare_results(self):
        if self.results is not None:
            return self.results
        else:
            raise NotRunError(
                "Results not created yet. Call 'run' to create results"
            )
    
    def _group_differences(self):
        group_means = self.X.groupby(self.sensitive_features).mean()
        std = self.X.std(numeric_only=True)
        diffs = {}
        for group1, group2 in combinations(group_means.index, 2):
            diff = (group_means.loc[group1]-group_means.loc[group2])/std
            diffs[f'{group1}-{group2}'] = diff.to_dict()

        return diffs
    
    def _run_cv(self, pipe):
        scorer = make_scorer(roc_auc_score, 
                             needs_proba=True,
                             multi_class='ovo')
        results = cross_val_score(pipe, self.X, self.y,
                             cv = StratifiedKFold(5),
                             scoring = scorer)
        return results
    
    def _make_pipe(self):
        model = get_gradient_boost_model()
        
        pipe = Pipeline(steps = 
               [('scaler', StandardScaler()),
                ('model', model)])
        return pipe

    
    def _find_categorical_features(self, df, threshold=0.10):
        """Identifies categorical features
        Logic: If type is not float and there are relatively few unique values for a feature, the feature is likely categorical.

        Parameters
        ----------
        df : pandas.dataframe
            A dataframe

        threshold : float
            The unique-count over total-count threshold

        Returns
        -------
        list
            Names of categorical features
        """        
        cols = df.columns
        float_cols = df.select_dtypes(include=[np.float]).columns
        cat_cols = set(cols) - set(float_cols)

        for name, column in df.iteritems():
            unique_count = column.unique().shape[0]
            total_count = column.shape[0]
            if unique_count / total_count < threshold:
                cat_cols.add(name)

        return list(cat_cols)

    def _calculate_mutual_information(self, categorical_features=None, normalize=True):
        """Calculates normalized mutual information between sensitive feature and other features
        Mutual information is the "amount of information" obtained about the sensitive feature by observing the other feature.
        It can therefore be used to proxy detection purposes.

        Parameters
        ----------
        categorical_features : [str], optional
            List of the categorical features (including the sensitive attribute) in the dataset, by default None
            If not provided, all non-float-type features are considered as categorical features

        Returns
        -------
        dict
            Normalized mutual information between sensitive feature and features and their considered feature type (categorical/continuous)
            Normalized mutual information between sensitive feature and itself is always 1, but is included to report its considered type
        """

        # Create a pandas dataframe of features and sensitive feature
        if isinstance(self.sensitive_features, pd.Series):
            df = pd.concat([self.X, self.sensitive_features], axis=1)
            sensitive_feature_name = self.sensitive_features.name
        else:
            df = self.X.copy()
            df["sensitive_feature"] = self.sensitive_features
            sensitive_feature_name = "sensitive_feature"

        # Estimate categorical features, if not provided
        if not categorical_features:
            categorical_features = self._find_categorical_features(df)

        # Encode categorical features
        for col in categorical_features:
            df[col] = df[col].astype("category").cat.codes

        discrete_features = [
            True if col in categorical_features else False for col in df.columns
        ]

        # Use the right mutual information methods based on the feature type of the sensitive attribute
        if sensitive_feature_name in categorical_features:
            mi = mutual_info_classif(
                df,
                df[sensitive_feature_name],
                discrete_features=discrete_features,
                random_state=42,
            )
        else:
            mi = mutual_info_regression(
                df,
                df[sensitive_feature_name],
                discrete_features=discrete_features,
                random_state=42,
            )

        # Normalize the mutual information values, if requested
        mi = pd.Series(mi, index=df.columns)
        if normalize:
            mi = mi / mi.max()

        # Create the results
        mi = mi.sort_index().to_dict()
        results = {}
        for k, v in mi.items():
            if k in categorical_features:
                results[k] = {"value": v, "feature_type": "categorical"}
            else:
                results[k] = {"value": v, "feature_type": "continuous"}

        results[sensitive_feature_name]["feature_type"] = results[sensitive_feature_name]["feature_type"] + '_reference'

        return results
    
    def _assess_balance_metrics(self):
        """Calculate dataset balance statistics and metrics 

        Returns
        -------
        dict
            'sample_balance': distribution of samples across groups
            'label_balance': distribution of labels across groups
            'metrics': maximum statistical parity and maximum disparate impact between groups for all preferred label value possibilities 
        """
        df, sensitive_feature_name, label_name = cl.CredoData._concat_features_label_to_dataframe(
            X=self.X, y=self.y, sensitive_features=self.sensitive_features
            )
        results = {}

        # Distribution of samples across groups
        sample_balance = (
            df.groupby([sensitive_feature_name])
            .agg(
                count=(label_name, len),
                percentage=(label_name, lambda x: 100.0 * len(x) / len(df)),
            )
            .reset_index()
            .to_dict(orient="records")
        )
        results["sample_balance"] = sample_balance

        # Distribution of samples across groups
        label_balance = (
            df.groupby([sensitive_feature_name, label_name])
            .size()
            .unstack(fill_value=0)
            .stack()
            .reset_index(name="count")
            .to_dict(orient="records")
        )
        results["label_balance"] = label_balance

        # Fairness metrics
        r = df.groupby([sensitive_feature_name, label_name]).agg({label_name: 'count'})
        r = r.groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))
        r.rename({label_name:'ratio'}, inplace=True, axis=1)
        r.reset_index(inplace=True)

        # Compute the maximum difference between any two pairs of groups
        demographic_parity_difference = r.groupby(label_name)['ratio'].apply(lambda x: np.max(x)-np.min(x)).reset_index(name='value').to_dict(orient='records')

        # Compute the maximum ratio between any two pairs of groups
        demographic_parity_ratio = r.groupby(label_name)['ratio'].apply(lambda x: np.max(x)/np.min(x)).reset_index(name='value').to_dict(orient='records')
        
        results['metrics'] = {'demographic_parity_difference': demographic_parity_difference, 'demographic_parity_ratio': demographic_parity_ratio}

        return results
