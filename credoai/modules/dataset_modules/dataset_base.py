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
        self.results = {'sensitive_feature_roc_auc': cv_results.mean(),
                        'group_diffs': group_differences,
                        'normalized_mutual_information': normalized_mutual_information}  
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
            diffs[f'{group1}-{group2}'] = diff
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

    
    def _find_categorical_features(self, df):
        """Estimates categorical features
        Logic: all non-float-type columns are categorical

        Parameters
        ----------
        df : pandas.dataframe
            A dataframe

        Returns
        -------
        list
            Names of categorical features
        """        
        cols = df.columns
        float_cols = df.select_dtypes(include=[np.float]).columns
        cat_cols = list(set(cols) - set(float_cols))
        return cat_cols

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
            df['sensitive_feature'] = sensitive_features
            sensitive_feature_name = 'sensitive_feature'

        # Estimate categorical features, if not provided
        if not categorical_features:
            categorical_features = self._find_categorical_features(df)

        # Encode categorical features
        for col in categorical_features:
            df[col] = df[col].astype('category').cat.codes

        discrete_features = [True if col in categorical_features else False for col in df.columns]

        # Use the right mutual information methods based on the feature type of the sensitive attribute
        if sensitive_feature_name in categorical_features:
            mi = mutual_info_classif(df, df[sensitive_feature_name], discrete_features=discrete_features, random_state=42)
        else:
            mi = mutual_info_regression(df, df[sensitive_feature_name], discrete_features=discrete_features, random_state=42)

        # Normalize the mutual information values, if requested
        mi = pd.Series(mi, index=df.columns)
        if normalize:
            mi = mi/mi.max()

        # Create the results
        mi = mi.sort_index().to_dict() 
        results = {}
        for k, v in mi.items():
            if k in categorical_features:
                results[k] = {'value':v, 'feature_type': 'categorical'}
            else:
                results[k] = {'value':v, 'feature_type': 'continuous'}

        return results
    

    
        
