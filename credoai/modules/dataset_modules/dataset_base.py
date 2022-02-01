from credoai.modules.credo_module import CredoModule
from credoai.utils.common import NotRunError
from credoai.utils.model_utils import get_gradient_boost_model
from itertools import combinations
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, make_scorer, normalized_mutual_info_score

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
        feature_space_proxy_scores = self._assess_proxy_in_feature_space()
        self.results = {'sensitive_feature_roc_auc': cv_results.mean(),
                        'group_diffs': group_differences,
                        'feature_space_proxy_scores': feature_space_proxy_scores}  
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

    def _assess_proxy_in_feature_space(
        self, excluded_columns=None, method="mutual_information"
    ):
        """Assesses features based on how much they serve as a proxy for a sensitive attribute
        This method relies on features information only (label information is not used)

        Parameters
        ----------
        excluded_columns : list[str]
            List of columns to exclude from the assessment
        method : str, optional
            Method to use for quantifying the magnitude of proxy, by default 'mutual_information'
            Choices:
                'mutual_information' (the "amount of information" in normalized nats obtained about a feature by observing another feature)

        Returns
        -------
        dict
            Keys are feature names and values are their proxy scores
            Score between 0.0 and 1.0
        """
        proxy_scores = {}
        
        if method == "mutual_information":
            # Exclude features of type float
            float_cols = list(self.X.select_dtypes(include=["float16", "float32", "float64"]))
            if excluded_columns:
                excluded_columns.extend(float_cols)
            else:
                excluded_columns = float_cols

            for column in self.X.drop(excluded_columns, axis=1):
                proxy_scores[column] = normalized_mutual_info_score(
                    self.X[column], self.sensitive_features
                )

        return proxy_scores

    
        
