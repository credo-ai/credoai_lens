from credoai.modules.credo_module import CredoModule
from credoai.utils.model_utils import get_gradient_boost_model
from itertools import combinations
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, make_scorer

class DatasetModule(CredoModule):
    def __init__(self,
                 X,
                 y,
                 sensitive_features):
        self.X = X
        self.y = y
        self.sensitive_features = sensitive_features
    
    def run(self):
        pipe = self._make_pipe()
        cv_results = self._run_cv(pipe)
        group_differences = self._group_differences()
        return {'sensitive_feature_roc_auc': cv_results.mean(),
                'group_diffs': group_differences}    
    
    def prepare_results(self):
        return self.run()
    
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
        
