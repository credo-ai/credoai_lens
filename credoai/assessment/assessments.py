"""
Module containing all CredoAssessmsents
"""

from credoai.assessment.credo_assessment import CredoAssessment, AssessmentRequirements
from credoai.data.utils import get_data_path
import credoai.modules as mod
import sys, inspect

class FairnessBaseAssessment(CredoAssessment):
    def __init__(self):
        super().__init__(
            'FairnessBase', 
            mod.FairnessModule,
            AssessmentRequirements(
                model_requirements=[('prob_fun', 'pred_fun')],
                data_requirements=['X', 'y', 'sensitive_features']
            )
        )
    
    def init_module(self, *, manifest, model, data, additional_metrics=None, replace=False):
        """ Initializes the assessment module

        Transforms the manifest, CredoModel and CredoData into the proper form
        to create a runnable assessment.

        See the lens_customization notebook for examples

        Parameters
        ------------
        manifest : dict
            dictionary containing kwargs for the module defined by the manifest
        model : CredoModel, optional
        data : CredoData, optional
        additional_metrics : list-like, optional
            passed to mod.FairnessModule.update_metrics
        replace : bool, optional
            passed to mod.FAirnessModule.update_metrics

        Example:
        def build(self, ...):
            y_pred = CredoModel.pred_fun(CredoData.X)
            y = CredoData.y
            self.initialized_module = self.module(y_pred, y)

        """
        bounds = manifest.get('bounds', {})
        y_pred = None
        y_prob = None
        if getattr(model, 'pred_fun'):
            y_pred = model.pred_fun(data.X)
        if getattr(model, 'prob_fun'):
            y_prob = model.prob_fun(data.X)
        module = self.module(
            manifest['metrics'],
            data.sensitive_features,
            data.y,
            y_pred,
            y_prob,
            bounds,
            bounds)
        if additional_metrics:
            module.update_metrics(additional_metrics, replace)
        self.initialized_module = module
            
class NLPEmbeddingBiasAssessment(CredoAssessment):
    def __init__(self):    
        super().__init__(
            'NLPEmbeddingBias', 
            mod.NLPEmbeddingAnalyzer,
            AssessmentRequirements(
                model_requirements=['embedding_fun'])
            )
        
    def init_module(self, manifest, model, data=None, 
              group_embeddings=None, 
              comparison_categories=None, 
              include_default=True):
        module = self.module(model.embedding_fun)
        if group_embeddings:
            module.set_group_embeddings(group_embeddings)
        if comparison_categories:
            module.set_comparison_categories(include_default, comparison_categories)
        self.initialized_module = module

class NLPGeneratorAssessment(CredoAssessment):
    def __init__(self):    
        super().__init__(
            'NLPGenerator', 
            mod.NLPGeneratorAnalyzer,
            AssessmentRequirements(
                model_requirements=['generation_fun'])
            )
        
    def init_module(self, scope, model, data=None, 
              toxicity_fun=None):
        module = self.module(
            model.generation_fun,
            toxicity_fun=toxicity_fun)

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

    def init_module(self, *, manifest, data, model=None):
        self.initialized_module = self.module(
            data.X, 
            data.y,
            data.sensitive_features)
        
def list_assessments_exhaustive():
    """List all defined assessments"""
    return inspect.getmembers(sys.modules[__name__], 
                              lambda member: inspect.isclass(member) and member.__module__ == __name__)

def list_assessments():
    """List subset of all defined assessments where the module is importable"""
    assessments = list_assessments_exhaustive()
    usable_assessments = []
    for assessment in assessments:
        try:
            _ = assessment[1]()
            usable_assessments.append(assessment)
        except AttributeError:
            pass
    return usable_assessments
