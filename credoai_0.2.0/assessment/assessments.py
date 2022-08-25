"""
Module containing all CredoAssessments
"""

import inspect
import sys

import credoai.modules as mod
import credoai.utils as cutils
import pandas as pd
from credoai.assessment.credo_assessment import AssessmentRequirements, CredoAssessment
from credoai.data.utils import get_data_path
from credoai.modules.utils import init_sensitive_feature_module
from credoai.reporting import (
    BinaryClassificationReporter,
    DatasetFairnessReporter,
    EquityReporter,
    FairnessReporter,
    NLPGeneratorAnalyzerReporter,
    RegressionReporter,
)
from credoai.reporting.dataset_profiling import DatasetProfilingReporter
from credoai.utils.model_utils import get_default_metrics
from sklearn.utils.multiclass import type_of_target

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
            "Fairness",
            mod.FairnessModule,
            AssessmentRequirements(
                model_requirements=[("predict_proba", "predict")],
                data_requirements=["X", "y", "sensitive_features"],
            ),
        )

    def init_module(self, *, model, data, metrics=None, **module_kwargs):
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
        module_kwargs : dict
            Optional keyword arguments to pass to FairnessModule

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
        metrics = get_default_metrics(model.model) if metrics is None else metrics
        if metrics is None:
            raise cutils.ValidationError(
                "Metrics are not defined for 'Fairness' Assessment in the assessment plan"
            )
        module = init_sensitive_feature_module(
            self.module,
            data.sensitive_features,
            metrics=metrics,
            y_true=data.y,
            y_pred=y_pred,
            y_prob=y_prob,
            **module_kwargs,
        )
        self.initialized_module = module

    def init_reporters(self):
        for module in self.initialized_module.modules.values():
            if type_of_target(module.y_true) == "binary":
                self.reporters = BinaryClassificationReporter(module)
            elif type_of_target(module.y_true) == "continuous":
                self.reporters = RegressionReporter(module)
            else:
                self.reporters = FairnessReporter(module)


class ModelEquityAssessment(CredoAssessment):
    """Evaluation of the equity of model outcomes"""

    def __init__(self):
        super().__init__(
            "ModelEquity",
            mod.EquityModule,
            AssessmentRequirements(
                model_requirements=["predict"], data_requirements=["sensitive_features"]
            ),
        )

    def init_module(self, *, model, data, **module_kwargs):
        """Initializes the assessment module

        Parameters
        ------------
        model : CredoModel
        data : CredoData
        module_kwargs : dict
            Optional keyword arguments to pass to FairnessModule
        """
        super().init_module(model=model, data=data)
        y = pd.Series(model.predict(data.X))
        try:
            y.name = f"predicted {data.y.name}"
        except:
            y.name = "predicted outcome"

        module = init_sensitive_feature_module(
            self.module, data.sensitive_features, y=y
        )
        self.initialized_module = module

    def init_reporters(self):
        reporters = [
            EquityReporter(m) for m in self.initialized_module.modules.values()
        ]
        self.reporters = reporters


class NLPEmbeddingBiasAssessment(CredoAssessment):
    """
    NLP Embedding-Bias Assessments

    Runs the NLPEmbeddingAnalyzer module.
    """

    def __init__(self):
        super().__init__(
            "NLPEmbeddingBias",
            mod.NLPEmbeddingAnalyzer,
            AssessmentRequirements(model_requirements=["embedding_fun"]),
        )

    def init_module(
        self,
        model,
        group_embeddings=None,
        comparison_categories=None,
        include_default=True,
    ):
        super().init_module(model=model)
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
            "NLPGenerator",
            mod.NLPGeneratorAnalyzer,
            AssessmentRequirements(model_requirements=["generator_fun"]),
        )

    def init_module(
        self,
        *,
        model,
        assessment_functions,
        prompts="bold_religious_ideology",
        comparison_models=None,
        perspective_config=None,
    ):
        """Initializes the assessment module

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
                raise cutils.InstallationError(
                    "To use default assessment functions requires installing credoai-lens[full]"
                )

        # set up generation functions
        generation_functions = {model.name: model.generator_fun}
        # extract generation functions from comparisons
        if comparison_models is None:
            try:
                generation_functions[
                    "gpt2_comparison"
                ] = cutils.nlp_utils.gpt2_text_generator
            except AttributeError:
                raise cutils.InstallationError(
                    "To use the default comparison model requires installing credoai-lens[full]"
                )
        else:
            generation_functions.update(comparison_models)

        module = self.module(
            prompts, generation_functions, assessment_functions, perspective_config
        )

        self.initialized_module = module

    def init_reporters(self):
        self.reporters = NLPGeneratorAnalyzerReporter(self.initialized_module)


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
            "Performance",
            mod.PerformanceModule,
            AssessmentRequirements(
                model_requirements=[("predict_proba", "predict")],
                data_requirements=["X", "y"],
            ),
        )

    def init_module(self, *, model, data, metrics=None, ignore_sensitive=True):
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

        metrics = get_default_metrics(model.model) if metrics is None else metrics
        if metrics is None:
            raise cutils.ValidationError(
                "Metrics are not defined for 'Performance' Assessment in the assessment plan"
            )

        sensitive_features = None if ignore_sensitive else data.sensitive_features
        if sensitive_features is not None:
            module = init_sensitive_feature_module(
                self.module,
                data.sensitive_features,
                metrics=metrics,
                y_true=data.y,
                y_pred=y_pred,
                y_prob=y_prob,
            )
        else:
            module = self.module(metrics, data.y, y_pred, y_prob, sensitive_features)
        self.initialized_module = module

    def init_reporters(self):
        try:
            modules = getattr(self.initialized_module, "modules")
            for module in modules.values():
                if type_of_target(module.y_true) == "binary":
                    self.reporters = BinaryClassificationReporter(
                        self.initialized_module
                    )
        except AttributeError:
            if type_of_target(self.initialized_module.y_true) == "binary":
                self.reporters = BinaryClassificationReporter(self.initialized_module)


class PrivacyAssessment(CredoAssessment):
    """Basic evaluation of the privacy of ML models

    Runs privacy analysis on models with well-defined
    objective functions. Currently includes:

    * Multi-class classification

    Supports models from any platform that have a `predict` function that returns
        predicted classes for a given feature vectors as a one-dimensional array.

    Modules:

    * credoai.modules.model_modules.privacy

    Requirements
    ------------
    Requires that the CredoModel defines it as "CLASSIFIER"
    """

    def __init__(self):
        super().__init__(
            "Privacy",
            mod.PrivacyModule,
            AssessmentRequirements(
                model_requirements=[("predict")],
                data_requirements=["X", "y"],
                training_data_requirements=["X", "y"],
                model_types=["CLASSIFIER"],
            ),
        )

    def init_module(self, *, model, data, training_data):
        """Initializes the assessment module

        Transforms CredoModel and CredoData into the proper form
        to create a runnable assessment.

        See the lens_customization notebook for examples

        Parameters
        ------------
        model : CredoModel
        data : CredoData
        training_data: CredoData
        metrics : List-like
            list of metric names as string or list of Metrics (credoai.metrics.Metric).
            Metric strings should in list returned by credoai.metrics.list_metrics.
            Note for performance parity metrics like
            "false negative rate parity" just list "false negative rate". Parity metrics
            are calculated automatically if the performance metric is supplied

        Example
        ---------
        def build(self, ...):
            y_pred = CredoModel.pred_fun(CredoData.X)
            y = CredoData.y
            self.initialized_module = self.module(model, data)

        """
        super().init_module(model=model, data=data, training_data=training_data)

        module = self.module(model, training_data.X, training_data.y, data.X, data.y)

        self.initialized_module = module


class SecurityAssessment(CredoAssessment):
    """Basic evaluation of the security of ML models

    Runs security analysis on models with well-defined
    objective functions. Currently includes:

    * Multi-class classification

    Supports models  from any platform that have a `predict` function that returns
        predicted classes for a given feature vectors as a one-dimensional array.

    Modules:

    * credoai.modules.model_modules.security

    Requirements
    ------------
    Requires that the CredoModel defines it as "CLASSIFIER"
    """

    def __init__(self):
        super().__init__(
            "Security",
            mod.SecurityModule,
            AssessmentRequirements(
                model_requirements=[("predict")],
                data_requirements=["X", "y"],
                training_data_requirements=["X", "y"],
                model_types=["CLASSIFIER"],
            ),
        )

    def init_module(self, *, model, data, training_data):
        """Initializes the assessment module

        Transforms CredoModel and CredoData into the proper form
        to create a runnable assessment.

        See the lens_customization notebook for examples

        Parameters
        ------------
        model : CredoModel
        data : CredoData
        training_data: CredoData

        """
        super().init_module(model=model, data=data, training_data=training_data)

        module = self.module(model, training_data.X, training_data.y, data.X, data.y)

        self.initialized_module = module


# *******************
# Dataset Assessments
# *******************


class DatasetEquityAssessment(CredoAssessment):
    """Evaluation of the equity of model outcomes"""

    def __init__(self):
        super().__init__(
            "DatasetEquity",
            mod.EquityModule,
            AssessmentRequirements(data_requirements=["y", "sensitive_features"]),
        )

    def init_module(self, *, data):
        """Initializes the assessment module

        Parameters
        ------------
        model : CredoModel
        data : CredoData
        module_kwargs : dict
            Optional keyword arguments to pass to FairnessModule
        """
        super().init_module(data=data)
        y = data.y
        module = init_sensitive_feature_module(
            self.module, data.sensitive_features, y=y
        )
        self.initialized_module = module

    def init_reporters(self):
        reporters = [
            EquityReporter(m) for m in self.initialized_module.modules.values()
        ]
        self.reporters = reporters


class DatasetFairnessAssessment(CredoAssessment):
    """
    Dataset Assessment

    Runs fairness assessment on a CredoDataset. This
    includes:

    * Distributional assessment of dataset
    * Proxy detection
    * Demographic Parity of outcomes

    Note: this assessment scrubs the data first (see utils.dataset_utils.scrub_data).

    Modules:

    * credoai.modules.dataset_fairness

    """

    def __init__(self):
        super().__init__(
            "DatasetFairness",
            mod.DatasetFairness,
            AssessmentRequirements(data_requirements=["X", "y", "sensitive_features"]),
        )

    def init_module(self, *, data, nan_strategy="ignore"):
        """Initializes the assessment module

        Transforms CredoModel and CredoData into the proper form
        to create a runnable assessment.

        See the lens_customization notebook for examples

        Parameters
        ------------
        data : CredoData
        nan_strategy : str or callable, optional
            The strategy for dealing with NaNs, passed to credoai.utils.scrub_data.
            In general, recommend you deal with NaNs before passing your data to Lens.

            -- If "ignore" do nothing,
            -- If "drop" drop any rows with any NaNs. X must be a pd.DataFrame
            -- If any other string, pass to the "strategy" argument of `Simple Imputer <https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html>`_.

            You can also supply your own imputer with
            the same API as `SimpleImputer <https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html>`_.

        """
        super().init_module(data=data)
        X, y, sensitive_features = cutils.scrub_data(data, nan_strategy)
        module = init_sensitive_feature_module(
            self.module,
            sensitive_features,
            X=X,
            y=y,
            categorical_features_keys=data.categorical_features_keys,
        )
        self.initialized_module = module

    def init_reporters(self):
        reporters = [
            DatasetFairnessReporter(m) for m in self.initialized_module.modules.values()
        ]
        self.reporters = reporters


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
            "DatasetProfiling",
            mod.DatasetProfiling,
            AssessmentRequirements(data_requirements=["X", "y"]),
        )

    def init_module(self, *, data):
        super().init_module(data=data)
        self.initialized_module = self.module(data.X, data.y)

    def init_reporters(self):
        self.reporters = DatasetProfilingReporter(self.initialized_module)


def list_assessments_exhaustive():
    """List all defined assessments"""
    return inspect.getmembers(
        sys.modules[__name__],
        lambda member: inspect.isclass(member) and member.__module__ == __name__,
    )


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
