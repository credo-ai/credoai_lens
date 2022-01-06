from credoai.assessment.credo_assessment import CredoAssessment
from credoai.assessment import get_usable_assessments
from credoai.utils.common import ValidationError
from credoai.utils.credo_api_utils import get_aligned_metrics, patch_metrics
from credoai import __version__
from dataclasses import dataclass
from os import makedirs, path
from sklearn.utils import check_consistent_length
from typing import List, Union

import credoai.integration as ci

BASE_CONFIGS = ('sklearn')

# *********************************
# Overview
# *********************************
# CredoLens relies on four classes
# - CredoGovernance
# - CredoModel
# - CredoData
# - CredoAssessment

# CredoGovernance contains the information needed
# to connect CredoLens with the Credo AI governance platform

# CredoModel follows an `adapter pattern` to convert
# any model into an interface CredoLens can work with.
# The functionality contained in CredoModel determines
# what assessments can be run.

# CredoData is a lightweight wrapper that stores data

# CredoAssessment is the interface between a CredoModel,
# CredoData and a module, which performs some assessment.


@dataclass
class CredoGovernance:
    """ Class to store governance data.

    This information is used to interact with the CredoAI
    Governance Platform. Artifacts (AI solutions, models, and datasets)
    are identified by a unique ID which can be found on the platform.

    To make use of the governance platform a .credo_config file must
    also be set up (see README)
    """
    ai_solution_id: str = None
    model_id: str = None
    data_id: str = None

class CredoModel:
    """Class wrapper around model-to-be-assessed

    CredoModel serves as an adapter between arbitrary models
    and the assessments in CredoLens. Assessments depend
    on CredoModel instantiating certain methods. In turn,
    the methods an instance of CredoModel defines informs
    Lens which assessment can be automatically run.

    An assessment's required CredoModel functionality can be accessed
    using the `get_requirements` function of an assessment instance.

    The simplest way to interact with CredoModel is to pass a model_config:
    a dictionary where the key/value pairs reflect functions. 
    E.g. {'prob_fun': model.predict}

    The model_config can also be inferred automatically, well-known packages
    (call CredoModel.supported_frameworks for a list.) If supported, a model
    can be passed directly to CredoModel's "model" argument and a model_config
    will be inferred.
    
    Note a model or model_config *must* be passed. If both are passed, any 
    functionality specified in the model_config will overwrite and inferences
    made from the model itself.

    See the quickstart and lens_customization notebooks for examples.


    Parameters
    ----------
    name : str
        Label of the model
    model : model, optional
        A model from a supported framework. Note functionality will be limited
        by CredoModel's automated inference. model_config is a more
        flexible and reliable method of interaction, by default None
    model_config : dict, optional
        dictionary containing mappings between CredoModel function names (e.g., "prob_fun")
        and functions (e.g., "model.predict"), by default None
    metadata: dict, optional
        Arbitrary additional data that will be associated with the model
    """ 

    def __init__(
        self,
        name: str,
        model=None,
        model_config : dict = None,
        metadata=None
    ):
        self.name = name
        self.config = {}
        assert model is not None or model_config is not None
        if model is not None:
            self._init_config(model)
        if model_config is not None:
            self.config.update(model_config)
        self._build_functionality()

    @staticmethod
    def supported_frameworks():
        return BASE_CONFIGS

    def _build_functionality(self):
        for key, val in self.config.items():
            if val is not None:
                self.__dict__[key] = val

    def _init_config(self, model):
        framework = self._get_model_type(model)
        if framework == 'sklearn':
            config = self._init_sklearn(model)
        self.config = config

    def _init_sklearn(self, model):
        config = {
            'pred_fun': model.predict,
            'prob_fun': model.predict_proba
        }
        return config

    def _get_model_type(self, model):
        try:
            framework = model.__module__.split('.')[0]
        except AttributeError:
            framework = None
        if framework in BASE_CONFIGS:
            return framework


@dataclass
class CredoData:
    """ Class to store assessment data. 

    Lightweight wrapper to hold data for analysis or to pass to 
    a model. 

    Passed to Lens for certain assessments. Either will be used
    by a CredoModel to make predictions or analyzed itself. 
    
    Parameters
    -------------
    name : str
        Label of the dataset
    X : data for model-input
        Features passed to a model. Should be dataframe-like in the case of tabular data
    y: data analogous model-output
        Ground-truth labels
    sensitive_features: array-like, optional
        Array of sensitive-feature labels. E.g., protected attributes like race or gender
        or categorical labels for important performance dimensions (["bright_lighting", "dark_lighting"])
    metadata: dict, optional
        Arbitrary additional data that will be associated with the dataset
    """
    name: str
    X: "model-input"
    y: "model-output"
    sensitive_features: 'array-like' = None
    metadata: dict = None

    def __post_init__(self):
        self.metadata = self.metadata or {}
        self._validate_data()

    def _validate_data(self):
        try:
            check_consistent_length(self.X, self.y)
        except ValueError:
            raise ValidationError("X and y don't have the same length")
        try:
            check_consistent_length(self.X, self.sensitive_features)
        except ValueError:
            raise ValidationError(f"X and sensitive features don't have the same index")


class Lens:
    def __init__(        
        self,
        governance: CredoGovernance = None,
        spec: dict = None,
        assessments: Union[List[CredoAssessment], str] = 'auto',
        model: CredoModel = None,
        data: CredoData = None,
        user_id: str = None,
        init_assessment_kwargs = None
    ):
        """Lens runs a suite of assessments on AI Models and Data for AI Governance

        Lens is the assessment framework component of the broader CredoAI suite.
        It is usable as a standalone gateway to a suite of assessments or in 
        combination with CredoAI's Governance Platform. 
        
        If the latter, Lens handles connecting Governance Alignment
        to assessments as well as exporting assessment results back to the Governance
        Platform.

        Parameters
        ----------
        governance : CredoGovernance, optional
            CredoGovernance object connecting
            Lens with Governance platform, by default None
        spec : dict
            key word arguments passed to each assessments `init_module` 
            function using `Lens.init_module`. Each key must correspond to
            an assessment name (CredoAssessment.name), with each value
            being a dictionary of kwargs. Passed to the init_module function
            as kwargs for each assessment
        assessments : Union[List[CredoAssessment], str], optional
            List of assessments to run. If "auto", runs all assessments
            CredoModel and CredoData support from the standard
            set of CredoLens assessments (defined by credoai.lens.ASSESSMENTS), by default 'auto'
        model : CredoModel, optional
            CredoModel to assess, by default None
        data : CredoData, optional
            CredoData to assess, or be used by CredoModel for assessment, by default None
        user_id : str, optional
            Label for user running assessments, by default None
        """    

        self.gov = governance
        self.model = model
        self.data = data
        self.spec = {}

        if assessments == 'auto':
            assessments = self._select_assessments()
        else:
            self._validate_assessments(assessments)
            assessments = assessments
        self.assessments = {a.name: a for a in assessments}
        self.user_id = user_id

        # if governance is defined, pull down spec for
        # solution / model
        if self.gov:
            self.spec = self.get_spec_from_gov()
        if spec:
            self.spec.update(spec)
            
        # initialize
        self._init_assessments()
            
    def run_assessments(self, export=False, assessment_kwargs=None):
        """Runs assessments on the associated model and/or data

        Parameters
        ----------
        export : bool or str, optional
            If a boolean, and true, export to Credo AI Governance Platform.
            If a string, save as a json to the output_directory indicated by the string.
            If False, do not export, by default False
        assessment_kwargs : dict, optional
            key word arguments passed to each assessments `run` or 
            `prepare_results` function. Each key must correspond to
            an assessment name (CredoAssessment.name). The assessments
            loaded by an instance of Lens can be accessed by calling
            `get_assessments`. 
            
            Example: {'FairnessBase': {'method': 'to_overall'}}
            by default None

        Returns
        -------
        [type]
            [description]
        """        
        assessment_kwargs = assessment_kwargs or {}
        assessment_results = {}
        for name, assessment in self.assessments.items():
            kwargs = assessment_kwargs.get(name, {})
            assessment_results[name] = assessment.run(**kwargs)
            if export:
                prepared_results = self._prepare_results(assessment, **kwargs)
                if type(export) == str:
                    self._export_to_file(prepared_results, export)
                else:
                    self._export_to_credo(prepared_results, to_model=True)
        return assessment_results

    def get_spec_from_gov(self):
        spec = {}
        metrics = self._get_aligned_metrics()
        spec['metrics'] = list(metrics.keys())
        spec['bounds'] = metrics
        return spec

    def get_assessments(self):
        return self.assessments
    
    def _to_record(self, results):
        return ci.record_metrics(results)
            
    def _export_to_credo(self, results, to_model=True):
        metric_records = self._to_record(results)
        destination_id = self.gov.model_id if to_model else self.gov.data_id
        ci.export_to_credo(metric_records, destination_id)
    
    def _export_to_file(self, results, output_directory):
        if not path.exists(output_directory):
            makedirs(output_directory, exist_ok=False)
        metric_records = self._to_record(results)
        assessment_name = results.assessment.unique()[0]
        output_file = path.join(output_directory, f'{assessment_name}.json')
        ci.export_to_file(metric_records, output_file)

    def _gather_meta(self, assessment_name):
        model_name = self.model.name if self.model else 'NA'
        data_name = self.data.name if self.data else 'NA'
        return {'model_label': model_name,
                'dataset_label': data_name,
                'user_id': self.user_id,
                'assessment': assessment_name,
                'lens_version': f'CredoAILens_v{__version__}'}

    def _get_aligned_metrics(self):
        """Get aligned metrics from credoai.s Governance Platform

        Returns
        -------
        dict
            The aligned metrics for one Model contained in the AI solution.
            Format: {"Metric1": (lower_bound, upper_bound), ...}
        """
        aligned_metrics = get_aligned_metrics(self.gov.ai_solution_id)[
            self.gov.model_id]
        return aligned_metrics

    def _init_assessments(self):
        """Initializes modules in each assessment
        
        Parameters
        ----------
        assessment_kwargs : dict, optional
            key word arguments passed to each assessments `init_module` 
            function. Each key must correspond to
            an assessment name (CredoAssessment.name). The assessments
            loaded by an instance of Lens can be accessed by calling
            `get_assessments`. 

        Example: {'FairnessBase': {'method': 'to_overall'}}
        by default None
        """
        for assessment in self.assessments.values():
            kwargs = self.spec.get(assessment.name, {})
            assessment.init_module(model=self.model,
                                   data=self.data,
                                   **kwargs)
            
    def _prepare_results(self, assessment, **kwargs):
        metadata = self._gather_meta(assessment.name)
        return assessment.prepare_results(metadata, **kwargs)

    def _select_assessments(self):
        return list(get_usable_assessments(self.model, self.data).values())

    def _validate_assessments(self, assessments):
        for assessment in assessments:
            if not assessment.check_requirements(self.model, self.data):
                raise ValidationError(
                    f"Model or Data does not conform to {assessment.name} assessment's requirements")


