"""Credo AI's AI Assessment Framework"""

from absl import logging
from copy import deepcopy
from credoai.assessment.credo_assessment import CredoAssessment
from credoai.assessment import get_usable_assessments
from credoai.utils.common import (
    IntegrationError, ValidationError, raise_or_warn)
from credoai.utils.credo_api_utils import (get_dataset_by_name, get_model_by_name,
                                           get_model_project_by_name, patch_metrics)
from credoai import __version__
from dataclasses import dataclass, field
from os import listdir, makedirs, path
from sklearn.utils import check_consistent_length
from typing import List, Union

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

        If not retrieved yet, attempt to retrieve them from AI Governance platform
        """
        if not self.assessment_spec:
            self.retrieve_assessment_spec()
        spec = {}
        metrics = self.assessment_spec
        if metrics:
            spec['metrics'] = list(metrics.keys())
        return {"FairnessBase": spec}

    def get_info(self):
        """Returns Credo AI Governance IDs"""
        to_return = self.__dict__.copy()
        del to_return['assessment_spec']
        del to_return['warning_level']
        return to_return

    def set_governance_info_by_name(self,
                                    *,
                                    model_name=None,
                                    dataset_name=None,
                                    model_project_name=None):
        """Sets governance info by name

        Sets model_id, model_project_id and/or dataset_id
        using names

        Parameters
        ----------
        model_name : str
            name of a model
        model_name : str
            name of a dataset
        model_name : str
            name of a model project
        """
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

        Either from Credo AI's governance platform or a json file

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
        if self.model_id and self.model_id in assessment_spec:
            assessment_spec = assessment_spec[self.model_id]
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
    metadata: dict, optional
        Arbitrary additional data that will be associated with the model
    """

    def __init__(
        self,
        name: str,
        model=None,
        model_config: dict = None,
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
        if model.n_classes_ == 2:
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
            raise ValidationError(
                f"X and sensitive features don't have the same index")

    @staticmethod
    def _concat_features_label_to_dataframe(X, y, sensitive_features):
        """A utility method that concatenates all features and labels into a single dataframe

        Returns
        -------
        pandas.dataframe, str, str
            Full dataset dataframe, sensitive feature name, label name
        """
        if isinstance(sensitive_features, pd.Series):
            df = pd.concat([X, sensitive_features], axis=1)
            sensitive_feature_name = sensitive_features.name
        else:
            df = X.copy()
            df['sensitive_feature'] = sensitive_features
            sensitive_feature_name = 'sensitive_feature'

        if isinstance(y, pd.Series):
            df = pd.concat([df, y], axis=1)
            label_name = y.name
        else:
            label_name = 'label'
            df[label_name] = sensitive_features

        return df, sensitive_feature_name, label_name


class Lens:
    def __init__(
        self,
        governance: CredoGovernance = None,
        spec: dict = None,
        assessments: Union[List[CredoAssessment], str] = 'auto',
        model: CredoModel = None,
        data: CredoData = None,
        user_id: str = None,
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
        warning_level : int
            warning level. 
                0: warnings are off
                1: warnings are raised (default)
                2: warnings are raised as exceptions.
        """

        self.gov = governance or CredoGovernance(warning_level=warning_level)
        self.model = model
        self.data = data
        self.spec = {}
        self.warning_level = warning_level

        if assessments == 'auto':
            assessments = self._select_assessments()
        else:
            self._validate_assessments(assessments)
            assessments = assessments
        self.assessments = {a.name: a for a in assessments}
        self.user_id = user_id

        # if governance is defined, pull down spec for
        # use_case / model
        if self.gov:
            self.spec = self.gov.get_assessment_spec()
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
            if export:
                logging.info(f"** Exporting assessment-{name}")
                prepared_results = self._prepare_results(assessment, **kwargs)
                if type(export) == str:
                    self._export_results_to_file(prepared_results, export)
                else:
                    self._export_results_to_credo(
                        prepared_results, to_model=True)
        return assessment_results

    def create_reports(self, export=False,
                       report_directory=None,
                       report_kwargs=None):
        """Create reports for assessments that have reports

        Parameters
        ----------
        export : bool or str, optional
            If a boolean, and true, export to Credo AI Governance Platform.
            If a string, reports to output_directory indicated by the string.
            If False, do not export, by default False
        report_kwargs : dict, optional
            key word arguments passed to each assessments `create_report`
            Each key must correspond to  an assessment name (CredoAssessment.name). 
            The assessments loaded by an instance of Lens can be accessed by calling
            `get_assessments`. 

        Returns
        -------
        reports
        """
        report_kwargs = report_kwargs or {}
        reports = {}
        for name, assessment in self.assessments.items():
            logging.info(f"Creating report for assessment-{name}")
            kwargs = report_kwargs.get(name, {})
            reports[name] = assessment.create_report(**kwargs)
        if export:
            logging.info(f"** Exporting report for assessment-{name}")
            self._export_reports(export)
        return reports

    def get_assessments(self):
        return self.assessments

    def get_governance(self):
        return self.gov

    def _export_reports(self, export=False):
        tmpdir = tempfile.mkdtemp()
        report_records = []
        for name, assessment in self.assessments.items():
            report = assessment.report
            # get filename
            meta = self._gather_meta(name)
            names = self._get_names()
            report_name = f"AssessmentReport_assessment-{name}_model-{names['model']}_data-{names['data']}"
            filename = path.join(tmpdir, report_name)
            # create report recodr
            report.export_report(filename)
            if export is True:
                report_record = ci.Figure(
                    name=f"{report_name}.pdf",
                    figure=f"{filename}.pdf",
                    **meta)
                self._export_report_to_credo(report_record)
        if type(export) == str:
            # move to final location
            allfiles = listdir(tmpdir)
            for f in allfiles:
                shutil.move(path.join(tmpdir, f),
                            path.join(export, f))
        shutil.rmtree(tmpdir)

    def _export_results_to_credo(self, results, to_model=True):
        metric_records = ci.record_metrics(results)
        destination_id = self._get_credo_destination(to_model)
        ci.export_to_credo(metric_records, destination_id)

    def _export_results_to_file(self, results, output_directory):
        if not path.exists(output_directory):
            makedirs(output_directory, exist_ok=False)
        metric_records = ci.record_metrics(results)
        # determine save directory
        assessment_name = results.assessment.unique()[0]
        names = self._get_names()
        results_file = (f"AssessmentResults_assessment-{assessment_name}_"
                        f"model-{names['model']}_data-{names['data']}.json")
        output_file = path.join(output_directory, results_file)
        ci.export_to_file(metric_records, output_file)

    def _export_report_to_credo(self, report_record, to_model=True):
        destination_id = self._get_credo_destination(to_model)
        ci.export_figure_to_credo(report_record, destination_id)

    def _gather_meta(self, assessment_name):
        names = self._get_names()
        return {'process': f'Lens-{assessment_name}',
                'model_label': names['model'],
                'dataset_label': names['data'],
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

    def _get_names(self):
        model_name = self.model.name if self.model else 'NA'
        data_name = self.data.name if self.data else 'NA'
        return {'model': model_name, 'data': data_name}

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

    def _select_assessments(self):
        return list(get_usable_assessments(self.model, self.data).values())

    def _validate_assessments(self, assessments):
        for assessment in assessments:
            if not assessment.check_requirements(self.model, self.data):
                raise ValidationError(
                    f"Model or Data does not conform to {assessment.name} assessment's requirements")
