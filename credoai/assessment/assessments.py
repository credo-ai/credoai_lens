"""
Module containing all CredoAssessments
"""

from credoai.assessment.credo_assessment import CredoAssessment, AssessmentRequirements
from credoai.data.utils import get_data_path
from credoai.reporting import (FairnessReporter, BinaryClassificationReporter,
                               NLPGeneratorAnalyzerReporter, DatasetFairnessReporter, RegressionReporter)
from sklearn.utils.multiclass import type_of_target
from credoai.reporting.dataset_profiling import DatasetProfilingReporter

from credoai.utils import InstallationError
import credoai.utils as cutils
import credoai.modules as mod
import sys
import inspect

# *******************
# Model Assessments
# *******************

class FairnessAssessment(CredoAssessment):
    """Basic evaluation of the fairness of ML models

    Runs fairness analysis on models with well-defined
    objective functions. Examples include:

    * binary classification
    * regression
    * recommendation systems

    Modules:

    * credoai.modules.fairness_base

    Requirements
    ------------
    Requires that the CredoModel defines either `predict` or `predict_proba` (or both).
    - `predict` should return the model's predictions.
    - `predict_proba` should return probabilities associated with the predictions (like scikit-learn's `predict_proba`)
       Only applicable in classification scenarios.
    """

    def __init__(self):
        super().__init__(
            'Fairness',
            mod.FairnessModule,
            AssessmentRequirements(
                model_requirements=[('predict_proba', 'predict')],
                data_requirements=['X', 'y', 'sensitive_features']
            )
        )

    def init_module(self, *, model, data, metrics):
        """Initializes the assessment module

        Parameters
        ------------
        model : CredoModel
        data : CredoData
        metrics : List-like
            list of metric names as string or list of Metrics (credoai.metrics.Metric).
            Metric strings should in list returned by credoai.metrics.list_metrics.
            Note for performance parity metrics like 
            "false negative rate parity" just list "false negative rate". Parity metrics
            are calculated automatically if the performance metric is supplied

        Example
        ---------
        def build(self, ...):
            y_pred = CredoModel.predict(CredoData.X)
            y = CredoData.y
            self.initialized_module = self.module(y_pred, y)

        """
        super().init_module(model=model, data=data)
        try:
            y_pred = model.predict(data.X)
        except AttributeError:
            y_pred = None
        try:
            y_prob = model.predict_proba(data.X)
        except AttributeError:
            y_prob = None

        module = self.module(
            metrics,
            data.sensitive_features,
            data.y,
            y_pred,
            y_prob)
        self.initialized_module = module

    def get_reporter(self):
        if type_of_target(self.initialized_module.y_true) == 'binary':
            return BinaryClassificationReporter(self)
        elif type_of_target(self.initialized_module.y_true) == 'continuous':
            return RegressionReporter(self)
        else:
            return FairnessReporter(self)


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
        super().init_module(model=model)
        module = self.module(model.embedding_fun)
        if group_embeddings:
            module.set_group_embeddings(group_embeddings)
        if comparison_categories:
            module.set_comparison_categories(
                include_default, comparison_categories)
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
                    assessment_functions,
                    prompts='bold_religious_ideology',
                    comparison_models=None,
                    perspective_config=None):
        """ Initializes the assessment module

        Parameters
        ------------
        model : CredoModel
        assessment_functions : dict
            keys are names of the assessment functions and values could be custom callable assessment functions 
            or name of builtin assessment functions. 
            Current choices, all using Perspective API include:
                    'perspective_toxicity', 'perspective_severe_toxicity', 
                    'perspective_identify_attack', 'perspective_insult', 
                    'perspective_profanity', 'perspective_threat'
        prompts : str
            choices are builtin datasets, which include:
                'bold_gender', 'bold_political_ideology', 'bold_profession', 
                'bold_race', 'bold_religious_ideology' (Dhamala et al. 2021)
                'realtoxicityprompts_1000', 'realtoxicityprompts_challenging_20', 
                'realtoxicityprompts_challenging_100', 'realtoxicityprompts_challenging' (Gehman et al. 2020)
            or path of your own prompts csv file with columns 'group', 'subgroup', 'prompt'
        comparison_models : dict, optional
            Dictionary of other generator functions to use. Will assess these as well against
            the prompt dataset to use for comparison. If None, gpt2 will be used. To
            specify no comparison_models, supply the empty dictionary {}
        perspective_config : dict
            if Perspective API is to be used, this must be passed with the following:
                'api_key': your Perspective API key
                'rpm_limit': request per minute limit of your Perspective API account

        """
        super().init_module(model=model)
        # set up default assessments
        if assessment_functions is None:
            try:
                assessment_functions = cutils.nlp_utils.get_default_nlp_assessments()
            except AttributeError:
                raise InstallationError(
                    "To use default assessment functions requires installing credoai-lens[extras]")

        # set up generation functions
        generation_functions = {model.name: model.generator_fun}
        # extract generation functions from comparisons
        if comparison_models is None:
            try:
                generation_functions['gpt2_comparison'] = \
                    cutils.nlp_utils.gpt2_text_generator
            except AttributeError:
                raise InstallationError(
                    "To use the default comparison model requires installing credoai-lens[extras]")
        else:
            generation_functions.update(comparison_models)

        module = self.module(
            prompts,
            generation_functions,
            assessment_functions,
            perspective_config)

        self.initialized_module = module

    def get_reporter(self):
        return NLPGeneratorAnalyzerReporter(self)


class PerformanceAssessment(CredoAssessment):
    """Basic evaluation of the performance of ML models

    Runs performance analysis on models with well-defined
    objective functions. Examples include:

    * binary classification
    * regression
    * recommendation systems

    Modules:

    * credoai.modules.fairness_base

    Requirements
    ------------
    Requires that the CredoModel defines either `predict` or `predict_proba` (or both).
    - `predict` should return the model's predictions.
    - `predict_proba` should return probabilities associated with the predictions (like scikit-learn's `predict_proba`)
       Only applicable in classification scenarios.
    """

    def __init__(self):
        super().__init__(
            'Performance',
            mod.PerformanceModule,
            AssessmentRequirements(
                model_requirements=[('predict_proba', 'predict')],
                data_requirements=['X', 'y']
            )
        )

    def init_module(self, *, model, data, metrics, ignore_sensitive=True):
        """Initializes the performance module

        Parameters
        ------------
        model : CredoModel, optional
        data : CredoData, optional
        metrics : List-like
            list of metric names as string or list of Metrics (credoai.metrics.Metric).
            Metric strings should in list returned by credoai.metrics.list_metrics.
            Note for performance parity metrics like 
            "false negative rate parity" just list "false negative rate". Parity metrics
            are calculated automatically if the performance metric is supplied
        ignore_sensitive : bool
            Whether to ignore the sensitive_feature of CredoData (thus preventing calculation
            of disaggregated performance). Generally used when Lens is also running 
            Fairness Assessment, which also calculates disaggregated performance.

        Example
        ---------
        def build(self, ...):
            y_pred = CredoModel.predict(CredoData.X)
            y = CredoData.y
            self.initialized_module = self.module(y_pred, y)

        """
        super().init_module(model=model, data=data)
        try:
            y_pred = model.predict(data.X)
        except AttributeError:
            y_pred = None
        try:
            y_prob = model.predict_proba(data.X)
        except AttributeError:
            y_prob = None
        
        sensitive_features = None if ignore_sensitive else data.sensitive_features
        module = self.module(
            metrics,
            data.y,
            y_pred,
            y_prob,
            sensitive_features)
        self.initialized_module = module

# *******************
# Dataset Assessments
# *******************


class DatasetFairnessAssessment(CredoAssessment):
    """
    Dataset Assessment

    Runs fairness assessment on a CredoDataset. This
    includes:

    * Distributional assessment of dataset
    * Proxy detection
    * Demographic Parity of outcomes

    Note: this assessment runs on the the scrubbed data (see CredoData.get_scrubbed_data).

    Modules:

    * credoai.modules.dataset_fairness

    """

    def __init__(self):
        super().__init__(
            'DatasetFairness',
            mod.DatasetFairness,
            AssessmentRequirements(
                data_requirements=['X', 'y', 'sensitive_features']
            )
        )

    def init_module(self, *, data):
        super().init_module(data=data)
        scrubbed_data = data.get_scrubbed_data()
        self.initialized_module = self.module(
            scrubbed_data['X'],
            scrubbed_data['y'],
            scrubbed_data['sensitive_features'],
            data.categorical_features_keys)

    def get_reporter(self):
        return DatasetFairnessReporter(self)


class DatasetProfilingAssessment(CredoAssessment):
    """
    Dataset Profiling

    Generate profile reports 

    Modules
    -------
    * credoai.modules.dataset_profiling

    """

    def __init__(self):
        super().__init__(
            'DatasetProfiling',
            mod.DatasetProfiling,
            AssessmentRequirements(
                data_requirements=['X', 'y']
            )
        )

    def init_module(self, *, data):
        super().init_module(data=data)
        self.initialized_module = self.module(
            data.X,
            data.y)

    def get_reporter(self):
        return DatasetProfilingReporter(self)


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




class DummyAssessment(CredoAssessment):
    """
    Dummy assessment

    Generate dummy reports 

    Modules
    -------
    * credoai.modules.dummy_module

    """
    def __init__(self):
        super().__init__(
            'DummyModule', 
            mod.DummyModule,
            AssessmentRequirements(
                model_requirements=[('predict_proba', 'predict')],
                data_requirements=['X', 'y'],
                training_data_requirements=['X', 'y']
            )
        )
    
    def init_module(self, *, model, data, training_data):
        super().init_module(
            model=model,
            data=data,
            training_data=training_data
        )

        module = self.module(
            model,
            training_data.X,
            training_data.y,
            data.X,
            data.y
        )
            
        self.initialized_module = module
