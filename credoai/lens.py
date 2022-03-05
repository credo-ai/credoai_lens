"""Credo AI's AI Assessment Framework"""

from absl import logging
from copy import deepcopy
from credoai.assessment.credo_assessment import CredoAssessment
from credoai.assessment import get_usable_assessments
from credoai.reporting.reports import MainReport
from credoai.utils.common import (
    IntegrationError, NotRunError, ValidationError, raise_or_warn)
from credoai.utils.credo_api_utils import (get_dataset_by_name, 
                                           get_model_by_name,
                                           get_model_project_by_name, 
                                           get_use_case_by_name)
from credoai import __version__
from dataclasses import dataclass, field
from datetime import datetime
from os import listdir, makedirs, path
from sklearn.utils import check_consistent_length
from typing import List, Union, Optional

import collections.abc
import credoai.integration as ci
import pandas as pd
import shutil
import tempfile

BASE_CONFIGS = ('sklearn', 'xgboost')

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
    Governance Platform. Artifacts (Use Cases, model projects,
    models, and datasets) are identified by a unique ID which 
    can be found on the platform.

    To make use of the governance platform a .credo_config file must
    also be set up (see README)
    """

    def __init__(self,
                 use_case_id: str = None,
                 model_project_id: str = None,
                 model_id: str = None,
                 dataset_id: str = None,
                 warning_level=1):
        """[summary]

        Parameters
        ----------
        use_case_id : str, optional
            ID of Use Case on Credo AI Governance Platform, by default None
        model_project_id : str, optional
            ID of model project on Credo AI Governance Platform, by default None
        model_id : str, optional
            ID of model on Credo AI Governance Platform, by default None
        dataset_id : str, optional
            ID of dataset on Credo AI Governance Platform, by default None
        warning_level : int
            warning level. 
                0: warnings are off
                1: warnings are raised (default)
                2: warnings are raised as exceptions.
        """
        self.use_case_id = use_case_id
        self.model_project_id = model_project_id
        self.model_id = model_id
        self.dataset_id = dataset_id
        self.assessment_spec = {}
        self.warning_level = warning_level

    def get_assessment_spec(self):
        """Get assessment spec
        
        Return the assessment spec for the model defined
        by model_id.
        
        If not retrieved yet, attempt to retrieve the spec first
        from the AI Governance platform. 
        """
        if not self.assessment_spec:
            self.retrieve_assessment_spec()
        spec = {}
        metrics = self.assessment_spec
        if self.model_id in metrics.keys():
            spec['metrics'] = list(metrics[self.model_id].keys())
        return {"FairnessBase": spec}

    def get_info(self):
        """Return Credo AI Governance IDs"""
        to_return = self.__dict__.copy()
        del to_return['assessment_spec']
        del to_return['warning_level']
        return to_return

    def get_defined_ids(self):
        """Return IDS that have been defined"""
        return [k for k, v in self.get_info().items() if v]

    def set_governance_info_by_name(self,
                                    *,
                                    use_case_name=None,
                                    model_name=None,
                                    dataset_name=None,
                                    model_project_name=None):
        """Sets governance info by name

        Sets model_id, model_project_id and/or dataset_id
        using names

        Parameters
        ----------
        use_case_name : str
            name of a use_case
        model_name : str
            name of a model
        model_name : str
            name of a dataset
        model_name : str
            name of a model project
        """
        if use_case_name:
            ids = get_use_case_by_name(use_case_name)
            if ids is not None:
                self.use_case_id = ids['use_case_id']
        if model_name:
            ids = get_model_by_name(model_name)
            if ids is not None:
                self.model_id = ids['model_id']
                self.model_project_id = ids['model_project_id']
        if dataset_name:
            ids = get_dataset_by_name(dataset_name)
            if ids is not None:
                self.dataset_id = ids['dataset_id']
        if model_project_name and not model_name:
            ids = get_model_project_by_name(model_project_name)
            if ids is not None:
                self.model_project_id = ids['id']

    def retrieve_assessment_spec(self, spec_path=None):
        """Retrieve assessment spec

        Retrieve assessment spec, either from Credo AI's 
        governance platform or a json file. This spec will be
        for a use-case, and may apply to multiple models.
        get_assessment_spec returns the spec associated with 
        `model_id`.

        Parameters
        __________
        spec_path : string, optional
            The file location for the technical spec json downloaded from
            the technical requirements of an Use Case on Credo AI's
            Governance Platform. If no spec_path is provided,
            will use the Use Case ID. Default None

        Returns
        -------
        dict
            The assessment spec for one Model contained in the Use Case.
            Format: {"Metric1": (lower_bound, upper_bound), ...}
        """
        assessment_spec = {}
        if self.use_case_id is not None:
            assessment_spec = ci.get_assessment_spec(
                self.use_case_id, spec_path)
        self.assessment_spec = assessment_spec
        return self.assessment_spec

    def register_dataset(self, dataset_name):
        """Registers a dataset

        If a project has not been registered, a new project will be created to
        register the dataset under.
        """
        try:
            ids = ci.register_dataset(dataset_name=dataset_name)
            self.model_id = ids['dataset_id']
        except IntegrationError:
            self.set_governance_info_by_name(dataset_name=dataset_name)
            raise_or_warn(IntegrationError,
                          f"The dataset ({dataset_name}) is already registered.",
                          f"The dataset ({dataset_name}) is already registered. Using registered dataset",
                          self.warning_level)

    def register_model(self, model_name):
        """Registers a model

        If a project has not been registered, a new project will be created to
        register the model under.

        If an AI solution has been set, the model will be registered to that
        solution.
        """
        try:
            ids = ci.register_model(model_name=model_name,
                                    model_project_id=self.model_project_id)
            self.model_id = ids['model_id']
            self.model_project_id = ids['model_project_id']
        except IntegrationError:
            self.set_governance_info_by_name(model_name=model_name)
            raise_or_warn(IntegrationError,
                          f"The model ({model_name}) is already registered.",
                          f"The model ({model_name}) is already registered. Using registered model",
                          self.warning_level)
        if self.use_case_id:
            ci.register_model_to_use_case(self.use_case_id, self.model_id)

    def register_project(self, model_project_name):
        """Registers a model project"""
        if self.model_project_id is not None:
            raise ValidationError("Trying to register a project when a project ID ",
                                  "was already provided to CredoGovernance.")
        try:
            ids = ci.register_project(model_project_name)
            self.model_project_id = ids['model_project_id']
        except IntegrationError:
            self.set_governance_info_by_name(
                model_project_name=model_project_name)
            raise_or_warn(IntegrationError,
                          f"The model project ({model_project_name}) is already registered.",
                          f"The model project ({model_project_name}) is already registered. Using registered model project",
                          self.warning_level)


class CredoModel:
    """Class wrapper around model-to-be-assessed

    CredoModel serves as an adapter between arbitrary models
    and the assessments in CredoLens. Assessments depend
    on CredoModel instantiating certain methods. In turn,
    the methods an instance of CredoModel defines informs
    Lens which assessment can be automatically run.

    An assessment's required CredoModel functionality can be accessed
    using the `get_requirements` function of an assessment instance.

    The most generic way to interact with CredoModel is to pass a model_config:
    a dictionary where the key/value pairs reflect functions. This method is
    agnostic to framework. As long as the functions serve the needs of the
    assessments, they'll work.

    E.g. {'prob_fun': model.predict}

    The model_config can also be inferred automatically, from well-known packages
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
    """

    def __init__(
        self,
        name: str,
        model=None,
        model_config: dict = None,
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
        config = {}
        framework = self._get_model_type(model)
        if framework == 'sklearn':
            config = self._init_sklearn(model)
        elif framework == 'xgboost':
            config = self._init_xgboost(model)
        self.config = config

    def _init_sklearn(self, model):
        return self._sklearn_style_config(model)

    def _init_xgboost(self, model):
        return self._sklearn_style_config(model)

    def _sklearn_style_config(self, model):
        # if binary classification, only return
        # the positive classes probabilities by default
        if len(model.classes_) == 2:
            def prob_fun(X): return model.predict_proba(X)[:, 1]
        else:
            prob_fun = model.predict_proba

        config = {
            'pred_fun': model.predict,
            'prob_fun': prob_fun
        }
        return config

    def _get_model_type(self, model):
        try:
            framework = model.__module__.split('.')[0]
        except AttributeError:
            framework = None
        if framework in BASE_CONFIGS:
            return framework


class CredoData:
    """Class wrapper around data-to-be-assessed

    CredoData serves as an adapter between tabular datasets
    and the assessments in CredoLens.

    Passed to Lens for certain assessments. Either will be used
    by a CredoModel to make predictions or analyzed itself. 

    Parameters
    -------------
    name : str
        Label of the dataset
    data : pd.DataFrame
        Dataset dataframe that includes all features and labels
    sensitive_feature_key : str
        Name of the sensitive feature column, like 'race' or 'gender'
    label_key : str
        Name of the label column
    categorical_features_keys : list[str], optional
        Names of categorical features. If the sensitive feature is categorical, include it in this list.
        Note - ordinal features should not be included. 
    unused_features_keys : list[str], optional
        Names of the features to ignore when performing prediction.
        Include all the features in the data that were not used during model training
    drop_sensitive_feature : bool, optional
        If True, automatically adds sensitive_feature_key to the list of 
        unused_features_keys. If you do not explicitly use the sensitive feature
        in your model, this argument should be True. Otherwise, set to False.
        Default, True

    """

    def __init__(self,
                 name: str,
                 data: pd.DataFrame,
                 sensitive_feature_key: str,
                 label_key: str,
                 categorical_features_keys: Optional[List[str]] = None,
                 unused_features_keys: Optional[List[str]] = None,
                 drop_sensitive_feature: bool = True
                 ):

        self.name = name
        self.data = data
        self.sensitive_feature_key = sensitive_feature_key
        self.label_key = label_key
        self.categorical_features_keys = categorical_features_keys
        self.unused_features_keys = unused_features_keys
        self.drop_sensitive_feature = drop_sensitive_feature

        self.X = None
        self.y = None
        self.sensitive_features = None
        self._process_data(self.data)

    def __post_init__(self):
        self.metadata = self.metadata or {}
        self._validate_data()

    def _process_data(self, data):
        # set up sensitive features, y and X
        self.sensitive_features = data[self.sensitive_feature_key]
        self.y = data[self.label_key]

        # drop columns from X
        to_drop = [self.label_key]
        if self.unused_features_keys:
            to_drop += self.unused_features_keys
        if self.drop_sensitive_feature:
            to_drop.append(self.sensitive_feature_key)
        X = data.drop(columns=to_drop, axis=1)
        self.X = X

    def _validate_data(self):
        # Validate the types
        if not isinstance(self.data, pd.DataFrame):
            raise ValidationError(
                "The provided data type is " + self.data.__class__.__name__ +
                " but the required type is pd.DataFrame"
            )
        if not isinstance(self.sensitive_feature_key, str):
            raise ValidationError(
                "The provided sensitive_feature_key type is " +
                self.sensitive_feature_key.__class__.__name__ + " but the required type is str"
            )
        if not isinstance(self.label_key, str):
            raise ValidationError(
                "The provided label_key type is " +
                self.label_key.__class__.__name__ + " but the required type is str"
            )
        if self.categorical_features_keys and not isinstance(self.categorical_features_keys, list):
            raise ValidationError(
                "The provided label_key type is " +
                self.label_key.__class__.__name__ + " but the required type is list"
            )
        # Validate that the data column names are unique
        if len(self.data. columns) != len(set(self.data. columns)):
            raise ValidationError(
                "The provided data contains duplicate column names"
            )
        # Validate that the data contains the provided sensitive feature and label keys
        col_names = list(self.data.columns)
        if self.sensitive_feature_key not in col_names:
            raise ValidationError(
                "The provided sensitive_feature_key " + self.sensitive_feature_key +
                " does not exist in the provided data"
            )
        if self.label_key not in col_names:
            raise ValidationError(
                "The provided label_key " + self.label_key +
                " does not exist in the provided data"
            )

    def dev_mode(self, frac=0.1):
        """Samples data down for faster assessment and iteration

        Sampling will be stratified across the sensitive feature

        Parameters
        ----------
        frac : float
            The fraction of data to use
        """
        data = self.data.groupby(self.sensitive_features,
                                 group_keys=False).apply(lambda x: x.sample(frac=frac))
        self._process_data(data)


class Lens:
    def __init__(
        self,
        governance: CredoGovernance = None,
        spec: dict = None,
        assessments: Union[List[CredoAssessment], str] = 'auto',
        model: CredoModel = None,
        data: CredoData = None,
        user_id: str = None,
        dev_mode: Union[bool, float] = False,
        warning_level=1
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
        assessments : CredoAssessment or str, optional
            List of assessments to run. If "auto", runs all assessments
            CredoModel and CredoData support from the standard
            set of CredoLens assessments (defined by credoai.lens.ASSESSMENTS), by default 'auto'
        model : CredoModel, optional
            CredoModel to assess, by default None
        data : CredoData, optional
            CredoData to assess, or be used by CredoModel for assessment, by default None
        user_id : str, optional
            Label for user running assessments, by default None
        dev_mode : bool or float, optional
            If True, the passed CredoData will be reduced in size to speed up development. 
            A float<1 can also be provided which will determine the fraction of data to retain.
            Defaults to 0.1 when dev_mode is set to True.
            Default, False
        warning_level : int
            warning level. 
                0: warnings are off
                1: warnings are raised (default)
                2: warnings are raised as exceptions.
        """

        self.gov = governance or CredoGovernance(warning_level=warning_level)
        self.model = model
        self.data = data
        self.user_id = user_id
        self.spec = {}
        self.warning_level = warning_level
        self.dev_mode = dev_mode
        self.run_time = False

        # set up assessments
        if assessments == 'auto':
            assessments = self._select_assessments()
        else:
            self._validate_assessments(assessments)
            assessments = assessments
        self.assessments = {a.name: a for a in assessments}

        # if data is defined and dev mode, convert data
        if self.data and self.dev_mode:
            if self.dev_mode == True:
                self.dev_mode = 0.1
            self.data.dev_mode(self.dev_mode)

        # if governance is defined, pull down spec for
        # use_case / model
        if self.gov:
            self.spec = self.gov.get_assessment_spec()
        if spec:
            self._update_spec(self.spec, spec)

        # initialize
        self._init_assessments()

    def run_assessments(self, assessment_kwargs=None):
        """Runs assessments on the associated model and/or data

        Parameters
        ----------
        assessment_kwargs : dict, optional
            key word arguments passed to each assessments `run` or 
            `prepare_results` function. Each key must correspond to
            an assessment name (CredoAssessment.name). The assessments
            loaded by an instance of Lens can be accessed by calling
            `get_assessments`. 

        Returns
        -------
        assessment_results
        """
        assessment_kwargs = assessment_kwargs or {}
        assessment_results = {}
        for name, assessment in self.assessments.items():
            logging.info(f"Running assessment-{name}")
            kwargs = assessment_kwargs.get(name, {})
            assessment_results[name] = assessment.run(**kwargs).get_results()
        self.run_time = datetime.now().isoformat()
        return assessment_results

    def create_reports(self, report_name, export=False, display_results=False):
        """Creates notebook reports

        Creates jupyter notebook reports for every assessment that
        has reporting functionality defined. It then concatenates the reports into
        one final report. The reports are then able to be saved to file (if report_directory
        is defined), or exported to Credo AI's governance platform (if export=True).

        Note: to export to Credo AI's Governance Platform, CredoGovernance must be passed
        to Lens with a defined "use_case_id". "model_id" is also required, but if no "model_id" 
        is explicitly provided, a model will be registered and used.

        Parameters
        ----------
        report_name : str
            Title of the final report
        export : bool or str, optional
            If false, do not export. If a string, pass to export_assessments
        display_results : bool
            If True, display results. Calls credo_reporter.plot_results and 
            credo_reporter.display_table_results

        Returns
        -------
        reporters : dict
            dictionary of reporters (credoai.reporting.credo_reporter). Each reporter is
            responsible for creating visualizations and reports for a particular assessment
        final_report : credoai.reports.MainReport
            The final report. This object is responsible for managing notebook report creation.
        """
        if self.run_time == False:
            raise NotRunError(
                "Results not created yet. Call 'run_assessments' first"
            )
        reporters = {}
        for name, assessment in self.assessments.items():
            reporter = assessment.get_reporter()
            if reporter is not None:
                logging.info(
                    f"Reporter creating notebook for assessment-{name}")
                reporter.create_notebook()
                reporters[name] = reporter
                if display_results:
                    reporter.display_results_tables()
                    reporter.plot_results()
            else:
                logging.info(f"No reporter found for assessment-{name}")
        final_report = MainReport(f"{ci.RISK} Report", reporters.values())
        final_report.create_report(self)
        # exporting
        if export:
                self.export_asessments(export, report=final_report)
        return reporters, final_report

    def export_assessments(self, export="credoai", report=None):
        """_summary_

        Parameters
        ----------
        export : str
            If the special string "credoai", Credo AI Governance Platform.
            If a string, save assessment json to the output_directory indicated by the string.
            If False, do not export, by default "credoai""
        report : credoai.reports.Report, optional
            a report to include with the export. by default None
        """        
        prepared_results = []
        report = None
        for name, assessment in self.assessments.items():
            logging.info(f"** Exporting assessment-{name}")
            prepared_results.append(self._prepare_results(assessment))
        payload = ci.prepare_assessment_payload(prepared_results, report=report, assessed_at=self.run_time)

        if export == 'credoai':
            model_id = self._get_credo_destination()
            defined_ids = self.gov.get_defined_ids()
            if len({'model_id', 'use_case_id'}.intersection(defined_ids)) == 2:
                ci.post_assessment(self.gov.use_case_id, self.gov.model_id, payload)
                logging.info(
                    f"Exporting assessments to Credo AI's Governance Platform")
            else:
                logging.warning("Couldn't upload assessment to Credo AI's Governance Platform. "
                                "Ensure use_case_id is defined in CredoGovernance")
        else:
            if not path.exists(export):
                makedirs(export, exist_ok=False)
            names = self.get_artifact_names()
            name_for_save = f"{ci.RISK}_model-{names['model']}_data-{names['dataset']}.json"
            output_file = path.join(export, name_for_save)
            with open(output_file, 'w') as f:
                f.write(payload)

    def get_assessments(self):
        return self.assessments

    def get_governance(self):
        return self.gov

    def get_results(self):
        return {name: a.get_results() for name, a in self.assessments.items()}

    def _gather_meta(self, assessment_name):
        names = self.get_artifact_names()
        return {'process': f'Lens-{assessment_name}',
                'model_label': names['model'],
                'dataset_label': names['dataset'],
                'user_id': self.user_id,
                'assessment': assessment_name,
                'lens_version': f'Lens-v{__version__}'}

    def _get_credo_destination(self, to_model=True):
        if self.gov.model_id is None and to_model:
            raise_or_warn(ValidationError,
                          "No model_id supplied to export to Credo AI.")
            logging.info(f"**** Registering model ({self.model.name})")
            self.gov.register_model(self.model.name)
        if self.gov.model_id is None and not to_model:
            raise_or_warn(ValidationError,
                          "No dataset_id supplied to export to Credo AI.")
            logging.info(f"**** Registering dataset ({self.dataset.name})")
            self.gov.register_dataset(self.dataset.name)
        destination = self.gov.model_id if to_model else self.gov.dataset_id
        label = 'model' if to_model else 'dataset'
        logging.info(f"**** Destination for export: {label} id-{destination}")
        return destination

    def get_artifact_names(self):
        model_name = self.model.name if self.model else 'NA'
        data_name = self.data.name if self.data else 'NA'
        return {'model': model_name, 'dataset': data_name}

    def _init_assessments(self):
        """Initializes modules in each assessment"""
        for assessment in self.assessments.values():
            kwargs = deepcopy(self.spec.get(assessment.name, {}))
            reqs = assessment.get_requirements()
            if reqs['model_requirements']:
                kwargs['model'] = self.model
            if reqs['data_requirements']:
                kwargs['data'] = self.data
            assessment.init_module(**kwargs)

    def _prepare_results(self, assessment, **kwargs):
        metadata = self._gather_meta(assessment.name)
        return assessment.prepare_results(metadata, **kwargs)

    def _update_spec(self, d, u):
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = self._update_spec(d.get(k, {}), v)
            elif isinstance(v, list):
                d[k] = v + d.get(k, [])
            else:
                d[k] = v
        return d

    def _select_assessments(self):
        return list(get_usable_assessments(self.model, self.data).values())

    def _validate_assessments(self, assessments):
        for assessment in assessments:
            if not assessment.check_requirements(self.model, self.data):
                raise ValidationError(
                    f"Model or Data does not conform to {assessment.name} assessment's requirements")
