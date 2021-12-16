"""
Module containing all CredoAssessmsents
"""


from credoai.assessment.assessment import CredoAssessment, AssessmentRequirements
import credoai.modules as mod

class FairnessAssessment(CredoAssessment):
    def __init__(self):
        super().__init__(
            'FairnessBase', 
            mod.FairnessModule,
            AssessmentRequirements(
                model_requirements=[('prob_fun', 'pred_fun')],
                data_requirements=['X', 'y', 'sensitive_features']
            )
        )
    
    def init_module(self, scope, model, data, additional_metrics=None, replace=False):
        bounds = scope.get('bounds', {})
        y_pred = None
        y_prob = None
        if getattr(model, 'pred_fun'):
            y_pred = model.pred_fun(data.X)
        if getattr(model, 'prob_fun'):
            y_prob = model.prob_fun(data.X)
        module = self.module(
            scope['metrics'],
            data.sensitive_features,
            data.y,
            y_pred,
            y_prob,
            bounds,
            bounds)
        if additional_metrics:
            module.update_metrics(additional_metrics, replace)
        self.initialized_module = module
            
class NLPEmbeddingAssessment(CredoAssessment):
    def __init__(self):    
        super().__init__(
            'NLPEmbeddingFairness', 
            mod.NLPEmbeddingAnalyzer,
            AssessmentRequirements(
                model_requirements=['embedding_fun'])
            )
        
    def init_module(self, scope, model, data=None, 
              group_embeddings=None, 
              comparison_categories=None, 
              include_default=True):
        module = self.module(model.embedding_fun)
        if group_embeddings:
            module.set_group_embeddings(group_embeddings)
        if comparison_categories:
            module.set_comparison_categories(include_default, comparison_categories)
        self.initialized_module = module
            
class DatasetAssessment(CredoAssessment):
    def __init__(self):
        super().__init__(
            'Dataset', 
            mod.DatasetModule,
            AssessmentRequirements(
                data_requirements=['X', 'y', 'sensitive_features']
            )
        )

    def init_module(self, *, scope, data, model=None):
        self.initialized_module = self.module(
            data.X, 
            data.y,
            data.sensitive_features)
        
import sys, inspect
def list_classes():
    return inspect.getmembers(sys.modules[__name__], 
                              lambda member: inspect.isclass(member) and member.__module__ == __name__)