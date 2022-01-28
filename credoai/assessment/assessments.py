"""
Module containing all CredoAssessmsents
"""

from credoai.assessment.credo_assessment import CredoAssessment, AssessmentRequirements
import credoai.assessment.nlp_utils as nlp_utils
from credoai.data.utils import get_data_path
import credoai.modules as mod
from credoai.reporting import FairnessReport, NLPGeneratorAnalyzerReport
from sklearn.utils.multiclass import type_of_target
import sys, inspect

class FairnessBaseAssessment(CredoAssessment):
    """
    FairnessBase Assessment
    
    Runs the FairnessModule.
    
    Requirements
    ------------
    Requires that the CredoModel defines either `pred_fun` or `prob_fun` (or both).
    - `pred_fun` should return the model's predictions.
    - `prob_fun` should return probabilities associated with the predictions (like scikit-learn's `predict_proba`)
       Only applicable in classification scenarios.
    """
    def __init__(self):
        super().__init__(
            'FairnessBase', 
            mod.FairnessModule,
            AssessmentRequirements(
                model_requirements=[('prob_fun', 'pred_fun')],
                data_requirements=['X', 'y', 'sensitive_features']
            )
        )
    
    def init_module(self, *, model, data, metrics):
        """ Initializes the assessment module

        Transforms the spec, CredoModel and CredoData into the proper form
        to create a runnable assessment.

        See the lens_customization notebook for examples

        Parameters
        ------------
        spec : dict
            assessment spec: dictionary containing kwargs 
            for the module defined by the spec. Other kwargs
            can be passed at run time.
        model : CredoModel, optional
        data : CredoData, optional

        Example
        ---------
        def build(self, ...):
            y_pred = CredoModel.pred_fun(CredoData.X)
            y = CredoData.y
            self.initialized_module = self.module(y_pred, y)

        """
        try:
            y_pred = model.pred_fun(data.X)
        except AttributeError:
            y_pred = None
        try:
            y_prob = model.prob_fun(data.X)
        except AttributeError:
            y_prob = None
            
        module = self.module(
            metrics,
            data.sensitive_features,
            data.y,
            y_pred,
            y_prob)
        self.initialized_module = module
    
    def create_report(self, filename=None, include_fairness=True, include_disaggregation=True):
        """Creates a fairness report 
        
        Currently only supports binary classification models

        Parameters
        ----------
        filename : string, optional
            If given, the location where the generated pdf report will be saved, by default None
        include_fairness : bool, optional
            Whether to include fairness plots, by default True
        include_disaggregation : bool, optional
            Whether to include performance plots broken down by
            subgroup. Overall performance are always reported, by default True
        """        
        if type_of_target(self.initialized_module.y_true) == 'binary':
            self.report = FairnessReport(self.initialized_module)
            return self.report.create_report(filename, include_fairness, include_disaggregation)
            
class NLPEmbeddingBiasAssessment(CredoAssessment):
    """
    NLP Embedding-Bias Assessments
    
    Runs the NLPEmbeddingAnalyzer module.
    """
    def __init__(self):    
        super().__init__(
            'NLPEmbeddingBias', 
            mod.NLPEmbeddingAnalyzer,
            AssessmentRequirements(
                model_requirements=['embedding_fun'])
            )
        
    def init_module(self, model, data=None, 
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
    """
    NLP Generator Assessment
    
    Runs the NLPGenerator module.
    """
    def __init__(self):    
        super().__init__(
            'NLPGenerator', 
            mod.NLPGeneratorAnalyzer,
            AssessmentRequirements(
                model_requirements=['generator_fun'])
            )
        
    def init_module(self, *, model, data=None,
                   prompts='bold_religious_ideology',
                   assessment_functions=None,
                   comparison_models='gpt'):
        # set up deafult assessments
        if assessment_functions is None:
            assessment_functions = nlp_utils.get_default_nlp_assessments()
            
        # set up generation functions
        generation_functions = {model.name: model.generator_fun}
        # extract generation functions from comparisons
        if comparison_models == 'gpt':
            generation_functions['gpt2_comparison'] = \
                nlp_utils.gpt2_text_generator
        else:
            generation_functions.update(comparison_models)
            
        module = self.module(
            prompts,
            generation_functions,
            assessment_functions)

        self.initialized_module = module
        
    def create_report(self, filename=None):
        """Creates a report for nlp generator

        Parameters
        ----------
        filename : string, optional
            If given, the location where the generated pdf report will be saved, by default None
    """        
        self.report = NLPGeneratorAnalyzerReport(self.initialized_module)
        return self.report.create_report(filename)
            
class DatasetAssessment(CredoAssessment):
    """
    Dataset Assessment
    
    Runs the Dataset module.
    """
    def __init__(self):
        super().__init__(
            'Dataset', 
            mod.DatasetModule,
            AssessmentRequirements(
                data_requirements=['X', 'y', 'sensitive_features']
            )
        )

    def init_module(self, *, data, model=None):
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
