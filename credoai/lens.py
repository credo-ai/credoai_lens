"""Credo AI's AI Assessment Framework"""

from absl import logging
from copy import deepcopy
from credoai.artifacts import CredoGovernance, CredoModel, CredoData
from credoai.assessment.credo_assessment import CredoAssessment
from credoai.assessment import get_usable_assessments
from credoai.reporting.reports import MainReport
from credoai.utils.common import (
    json_dumps, wrap_list,
    NotRunError, ValidationError, raise_or_warn)
from credoai import __version__
from datetime import datetime
from os import listdir, makedirs, path
from sklearn.utils import check_consistent_length
from typing import List, Union
import collections.abc
import credoai.integration as ci
import shutil


class Lens:
    def __init__(
        self,
        governance: Union[CredoGovernance, str] = None,
        spec: dict = None,
        assessments: List[CredoAssessment] = None,
        model: CredoModel = None,
        data: Union[CredoData, List[CredoData]] = None,
        training_data: CredoData = None,
        user_id: str = None,
        dev_mode: Union[bool, float] = False,
        logging_level: Union[str, int] = 'info',
        warning_level=1
    ):
        """Lens runs a suite of assessments on AI Models and Data for AI Governance

        Lens is the assessment framework component of the broader CredoAI suite.
        It is usable as a standalone gateway to a suite of assessments or in 
        combination with CredoAI's Governance App. 

        If the latter, Lens handles connecting Governance Alignment
        to assessments as well as exporting assessment results back to the Governance
        App.

        Parameters
        ----------
        governance : CredoGovernance or string, optional
            If CredoGovernance, object connecting
            Lens with Governance App. If string, interpreted as 
            use-case ID on the Governance App. A CredoGovernance object
            will be created with the string as use_case_id, by default None
        spec : dict
            key word arguments passed to each assessments `init_module` 
            function using `Lens.init_module`. Each key must correspond to
            an assessment name (CredoAssessment.name), with each value
            being a dictionary of kwargs. Passed to the init_module function
            as kwargs for each assessment
        assessments : liar of CredoAssessment, optional
            List of assessments to select from. If None, selects from all eligible assessments.
            Assessments are ultimately selected from the ones that the model and 
            assessment datasets and/or training dataset support.
            Assessments must be selected from a set of CredoLens assessments 
            (defined by credoai.lens.ASSESSMENTS), by default None
        model : CredoModel, optional
            CredoModel to assess, by default None
        data : CredoData or list of CredoData, optional
            CredoData used to assess the model 
            (and/or assessed itself), by default None
        training_data : CredoData, optional
            CredoData object containing the training data used for the model. Will not be
            used to assess the model, but will be assessed itself if provided,
            by default None
        user_id : str, optional
            Label for user running assessments, by default None
        dev_mode : bool or float, optional
            If True, the passed CredoData will be reduced in size to speed up development. 
            A float<1 can also be provided which will determine the fraction of data to retain.
            Defaults to 0.1 when dev_mode is set to True.
            Default, False
        logging_level : int or str
            Sets logging and verbosity. Calls lens.set_logging_level. Options include:
            * 'info'
            * 'warning'
            * 'error'
        warning_level : int
            warning level. 
                0: warnings are off
                1: warnings are raised (default)
                2: warnings are raised as exceptions.
        """
        if isinstance(governance, str):
            self.gov = CredoGovernance(
                use_case_id=governance, warning_level=warning_level)
        else:
            self.gov = governance or CredoGovernance(
                warning_level=warning_level)
        self.model = model
        self.assessment_dataset = data
        self.training_dataset = training_data
        self.user_id = user_id
        self.spec = {}
        set_logging_level(logging_level)
        self.warning_level = warning_level
        self.dev_mode = dev_mode
        self.run_time = False

        # set up assessments
        self.assessments = self._select_assessments(assessments)

        # set up reporter objects
        self.report = None
        self.reporters = {}

        # if data is defined and dev mode, convert data
        self._apply_dev_mode(self.dev_mode)

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
        self
        """
        assessment_kwargs = assessment_kwargs or {}
        for assessment in self.get_assessments():
            logging.info(f"Running assessment-{assessment.get_id()}")
            kwargs = assessment_kwargs.get(assessment.name, {})
            assessment.run(**kwargs).get_results()
        self.run_time = datetime.now().isoformat()
        return self

    def create_report(self, display_results=False):
        """Creates notebook report

        Creates jupyter notebook reports for every assessment that
        has reporting functionality defined. It then concatenates the reports into
        one final report. 

        Parameters
        ----------
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
        for assessment in self.get_assessments():
            name = assessment.get_id()
            reporter = assessment.get_reporter()
            if reporter is not None:
                logging.info(
                    f"Reporter creating notebook for assessment-{name}")
                reporter.create_notebook()
                self.reporters[name] = reporter
                if display_results:
                    reporter.display_results_tables()
                    reporter.plot_results()
            else:
                logging.info(f"No reporter found for assessment-{name}")
        self.report = MainReport(f"Assessment Report", self.reporters.values())
        self.report.create_report(self)
        return self

    def export(self, destination="credoai"):
        """Exports assessments to file or Credo AI's governance app

        Note: to export to Credo AI's Governance App, CredoGovernance must be passed
        to Lens with a defined "use_case_id". "model_id" is also required, but if no "model_id" 
        is explicitly provided, a model will be registered and used.

        Parameters
        ----------
        destination : str
            Where to send the report
            -- "credoai", a special string to send to Credo AI Governance App.
            -- Any other string, save assessment json to the output_directory indicated by the string.
        """
        prepared_results = []
        for assessment in self.get_assessments():
            try:
                logging.info(f"** Exporting assessment-{assessment.get_id()}")
                prepared_results.append(self._prepare_results(assessment))
            except:
                raise Exception(
                    f"Assessment ({assessment.get_id()}) failed preparation")
        if self.report is None:
            logging.warning(
                "No report is included. To include a report, run create_reports first")
        payload = ci.prepare_assessment_payload(
            prepared_results, report=self.report, assessed_at=self.run_time)

        if destination == 'credoai':
            model_id = self._get_credo_destination()
            defined_ids = self.gov.get_defined_ids()
            if len({'model_id', 'use_case_id'}.intersection(defined_ids)) == 2:
                logging.info(
                    f"Exporting assessments to Credo AI's Governance App")
                return ci.post_assessment(self.gov.use_case_id, self.gov.model_id, payload)
            else:
                logging.warning("Couldn't upload assessment to Credo AI's Governance App. "
                                "Ensure use_case_id is defined in CredoGovernance")
        else:
            if not path.exists(destination):
                makedirs(destination, exist_ok=False)
            name_for_save = f"assessment_run-{self.run_time}.json"
            output_file = path.join(destination, name_for_save)
            with open(output_file, 'w') as f:
                f.write(json_dumps(payload))

    def get_assessments(self, dataset=None, assessment_name=None):
        """Return assessments defined

        Parameters
        ----------
        dataset : CredoData, optional
            If provided, only return assessments associated with the corresponding dataset
        assessment_name : str, optional
            If provided, only return assessments with the corresponding name,
            e.g. "Performance", by default None

        Returns
        -------
        list
            list of assessments
        """
        all_assessments = []
        for assessment_dataset, assessments in self.assessments.items():
            if dataset is not None and assessment_dataset != dataset.name:
                continue
            if assessment_name:
                all_assessments += [a for a in assessments if a.name == assessment_name]
            else:
                all_assessments += assessments
        return all_assessments

    def get_datasets(self):
        datasets = []
        if self.assessment_dataset is not None:
            datasets.append(self.assessment_dataset)
        if self.training_dataset is not None:
            datasets.append(self.training_dataset)
        return datasets

    def get_governance(self):
        return self.gov

    def get_report(self):
        return self.report

    def get_results(self):
        return {a.get_id(): a.get_results() for a in self.get_assessments()}

    def _apply_dev_mode(self, dev_mode):
        if dev_mode:
            if dev_mode == True:
                dev_mode = 0.1
            if self.assessment_dataset:
                self.assessment_dataset.dev_mode(self.dev_mode)
            if self.training_dataset:
                self.training_dataset.dev_mode(self.dev_mode)

    def _gather_meta(self, assessment):
        if assessment.data_name == self.assessment_dataset.name:
            dataset_id = self.gov.dataset_id
        elif assessment.data_name == self.training_dataset.name:
            dataset_id = self.gov.training_dataset_id
        return {'process': f'Lens-{assessment.name}',
                'model_label': assessment.model_name,
                'dataset_label': assessment.data_name,
                'dataset_id': dataset_id,
                'user_id': self.user_id,
                'assessment': assessment.name,
                'lens_version': f'Lens-v{__version__}'}

    def _get_credo_destination(self, to_model=True):
        """Get destination for export and ensure all artifacts are registered"""
        to_register = {}
        if self.gov.model_id is None and self.model:
            raise_or_warn(ValidationError,
                          "No model_id supplied to export to Credo AI.")
            logging.info(f"**** Registering model ({self.model.name})")
            to_register['model_name'] = self.model.name
        if self.gov.dataset_id is None and self.assessment_dataset:
            raise_or_warn(ValidationError,
                          "No dataset_id supplied to export to Credo AI.")
            logging.info(
                f"**** Registering assessment dataset ({self.assessment_dataset.name})")
            to_register['dataset_name'] = self.assessment_dataset.name
        if self.gov.training_dataset_id is None and self.training_dataset:
            raise_or_warn(ValidationError,
                          "No training dataset_id supplied to export to Credo AI.")
            logging.info(
                f"**** Registering training dataset ({self.training_dataset.name})")
            to_register['training_dataset_name'] = self.training_dataset.name
        if to_register:
            self.gov.register(**to_register)
        destination = self.gov.model_id if to_model else self.gov.dataset_id
        label = 'model' if to_model else 'dataset'
        logging.info(f"**** Destination for export: {label} id-{destination}")
        return destination

    def _init_assessments(self):
        """Initializes modules in each assessment"""
        for dataset in self.get_datasets():
            logging.info(
                f"Initializing assessments for dataset: {dataset.name}")
            assessments = self.assessments[dataset.name]
            for assessment in assessments:
                kwargs = deepcopy(self.spec.get(assessment.name, {}))
                reqs = assessment.get_requirements()
                if reqs['model_requirements']:
                    kwargs['model'] = self.model
                if reqs['data_requirements']:
                    kwargs['data'] = dataset
                try:
                    assessment.init_module(**kwargs)
                except:
                    raise ValidationError(f"Assessment ({assessment.get_id()}) could not be initialized."
                                          "Ensure the assessment spec is passing the required parameters"
                                          )

    def _prepare_results(self, assessment, **kwargs):
        metadata = self._gather_meta(assessment)
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

    def _select_assessments(self, candidate_assessments=None):
        selected_assessments = {}
        # get assesments for each assessment dataset
        for dataset in self.get_datasets():
            if dataset == self.training_dataset:
                model = None
            else:
                model = self.model
            usable_assessments = get_usable_assessments(model, dataset)
            assessment_text = f"Automatically Selected Assessments for dataset: {dataset.name}\n--" + \
                '\n--'.join(usable_assessments.keys())
            logging.info(assessment_text)
            selected_assessments[dataset.name] = list(usable_assessments.values())
        return selected_assessments


def set_logging_level(logging_level):
    """Alias for absl.logging.set_verbosity"""
    logging.set_verbosity(logging_level)
