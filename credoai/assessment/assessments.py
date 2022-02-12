"""
Module containing all CredoAssessmsents
"""

from credoai.assessment.credo_assessment import CredoAssessment, AssessmentRequirements
from credoai.data.utils import get_data_path
from credoai.reporting import FairnessReport, NLPGeneratorAnalyzerReport, DatasetModuleReport
from sklearn.utils.multiclass import type_of_target

from credoai.utils import InstallationError
import credoai.utils as cutils
import credoai.modules as mod
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

        Transforms CredoModel and CredoData into the proper form
        to create a runnable assessment.

        See the lens_customization notebook for examples

        Parameters
        ------------
        model : CredoModel, optional
        data : CredoData, optional
        metrics : List-like
            list of metric names as string or list of FairnessFunctions.
            Metric strings should in list returned by credoai.utils.list_metrics.
            Note for performance parity metrics like 
            "false negative rate parity" just list "false negative rate". Partiy metrics
            are calculated automatically.

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
        
    def init_module(self, model, 
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
    
    Requirements
    ------------
    Requires that the CredoModel defines a "generator_fun"
    - `generator_fun` should take in text as input and return text. 
        See `credoai.utils.nlp_utils.gpt1_text_generator` as an example
    """
    def __init__(self):    
        super().__init__(
            'NLPGenerator', 
            mod.NLPGeneratorAnalyzer,
            AssessmentRequirements(
                model_requirements=['generator_fun'])
            )
        
    def init_module(self, *, model, 
                   prompts='bold_religious_ideology',
                   assessment_functions=None,
                   comparison_models=None,
                   perspective_config=None):
        """ Initializes the assessment module

        Transforms CredoModel into the proper form
        to create a runnable assessment.

        Parameters
        ------------
        model : CredoModel
        prompts : str
            choices are builtin datasets, which include:
                'bold_gender', 'bold_political_ideology', 'bold_profession', 
                'bold_race', 'bold_religious_ideology' (Dhamala et al. 2021)
                'realtoxicityprompts_1000', 'realtoxicityprompts_challenging_20', 
                'realtoxicityprompts_challenging_100', 'realtoxicityprompts_challenging' (Gehman et al. 2020)
            or path of your own prompts csv file with columns 'group', 'subgroup', 'prompt'
        assessment_functions : dict
            keys are names of the assessment functions and values could be custom callable assessment functions 
            or name of builtin assessment functions. 
            Current choices, all using Perspective API include:
                    'perspective_toxicity', 'perspective_severe_toxicity', 
                    'perspective_identify_attack', 'perspective_insult', 
                    'perspective_profanity', 'perspective_threat'
        comparison_models : dict, optional
            Dictionary of other generator functions to use. Will assess these as well against
            the prompt dataset to use for comparison. If None, gpt2 will be used. To
            specify no comparison_models, supply the empty dictionary {}
        perspective_config : dict
            if Perspective API is to be used, this must be passed with the following:
                'api_key': your Perspective API key
                'rpm_limit': request per minute limit of your Perspective API account

        """
        # set up default assessments
        if assessment_functions is None:
            try:
                assessment_functions = cutils.nlp_utils.get_default_nlp_assessments()
            except AttributeError:
                raise InstallationError("To use default assessment functions requires installing credoai-lens[extras]")
            
        # set up generation functions
        generation_functions = {model.name: model.generator_fun}
        # extract generation functions from comparisons
        if comparison_models is None:
            try:
                generation_functions['gpt2_comparison'] = \
                    cutils.nlp_utils.gpt2_text_generator
            except AttributeError:
                raise InstallationError("To use the default comparison model requires installing credoai-lens[extras]")
        else:
            generation_functions.update(comparison_models)
            
        module = self.module(
            prompts,
            generation_functions,
            assessment_functions,
            perspective_config)

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

    def init_module(self, *, data):
        self.initialized_module = self.module(
            data.X, 
            data.y,
            data.sensitive_features)

    def create_report(self, filename=None):
        """Creates a report for dataset assessment

        Parameters
        ----------
        filename : string, optional
            If given, the location where the generated pdf report will be saved, by default None
        """        
        self.report = DatasetModuleReport(self.initialized_module)
        return self.report.create_report(filename)
        
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
