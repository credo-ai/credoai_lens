"""Credo AI's AI Assessment Framework"""

from absl import logging
from copy import deepcopy
from credoai.artifacts import CredoGovernance, CredoModel, CredoData
from credoai.assessment.credo_assessment import CredoAssessment
from credoai.assessment import AssessmentBunch
from credoai.reporting.reports import MainReport
from credoai.utils.common import (
    raise_or_warn, update_dictionary, wrap_list,
    NotRunError, ValidationError)
from credoai.utils.policy_utils import PolicyChecklist
from credoai import __version__
from datetime import datetime
from os import listdir, makedirs, path
from sklearn.utils import check_consistent_length
from typing import List, Union
import collections.abc
import credoai.integration as ci
import shutil
from collections import namedtuple
from itertools import combinations


class Lens:
    def __init__(
        self,
        governance: Union[CredoGovernance, str] = None,
        assessment_plan: dict = None,
        assessments: List[CredoAssessment] = None,
        model: CredoModel = None,
        data: CredoData = None,
        training_data: CredoData = None,
        display_policy_checklist: bool = True,
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
            a spec_destination to be passed to a CredoGovernance object, 
            by default None
        assessment_plan : dict
            key word arguments passed to each assessments `init_module` 
            function using `Lens.init_module`. Each key must correspond to
            an assessment name (CredoAssessment.name), with each value
            being a dictionary of kwargs. Passed to the init_module function
            as kwargs for each assessment
        assessments : list of CredoAssessment, optional
            List of assessments to select from. If None, selects from all eligible assessments.
            Assessments are ultimately selected from the ones that the model and 
            assessment datasets and/or training dataset support. If a list of assessments
            are provided, but aren't supported by the supplied credomodel/credodata a warning
            will be logged.
            Assessments must be selected from a set of CredoLens assessments 
            (defined by credoai.lens.ASSESSMENTS), by default None
        model : CredoModel, optional
            CredoModel to assess, by default None
        data : CredoData, optional
            CredoData used to assess the model 
            (and/or assessed itself). Called the "validation" dataset, by default None
        training_data : CredoData, optional
            CredoData object containing the training data used for the model. Will not be
            used to assess the model, but will be assessed itself if provided. Called
            the "training" dataset, by default None
        display_policy_checklist : bool, optional
            If True, and governance is defined, a policy checklist will be displayed in the 
            jupyter notebook (this only works where Ipython displays work). The policy checklist
            is composed of yes/no controls relevant to model development defined in the Governance
            App.
            Default, False
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
        self.model = model
        self.assessment_dataset = data
        self.training_dataset = training_data
        if self.assessment_dataset == self.training_dataset:
            raise ValidationError("Assessment dataset and training dataset should not be the same!")
        self.assessment_plan = {}
        set_logging_level(logging_level)
        self.warning_level = warning_level
        self.dev_mode = dev_mode
        self.run_time = False

        # set up governance
        self.gov = None
        if governance:
            if isinstance(governance, str):
                self.gov = CredoGovernance(
                    spec_destination=governance, warning_level=warning_level)
            else:
                self.gov = governance
            self._register_artifacts()

        # set up assessments
        self.assessments = self._select_assessments(assessments)

        # set up reporter objects
        self.report = None
        self.reporters = []

        # if data is defined and dev mode, convert data
        self._apply_dev_mode(self.dev_mode)

        # if governance is defined, use its assessment plan for
        # use_case / model
        if self.gov:
            self.assessment_plan = self.gov.get_assessment_plan()
        if assessment_plan:
            update_dictionary(self.assessment_plan, assessment_plan)

        # initialize
        self._init_assessments()

        # display checklist
        if display_policy_checklist and self.gov:
            checklist = self.gov.get_policy_checklist()
            if checklist:
                self.checklist = PolicyChecklist(checklist)
                self.checklist.create_checklist()

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
        for assessment in self.get_assessments(flatten=True):
            logging.info(f"Running assessment-{assessment.get_name()}")
            kwargs = assessment_kwargs.get(assessment.name, {})
            assessment.run(**kwargs).get_results()
        self.run_time = datetime.now().isoformat()
        return self

    def create_report(self):
        """Creates notebook report

        Creates jupyter notebook reports for every assessment that
        has reporting functionality defined. It then concatenates the reports into
        one final report. 

        Returns
        -------
        self
        """
        if self.run_time == False:
            raise NotRunError(
                "Results not created yet. Call 'run_assessments' first"
            )
        if not self.reporters:
            self._create_reporters()
        self.report = MainReport(f"Assessment Report", self.reporters)
        self.report.create_report(self)
        return self

    def export(self, destination="credoai"):
        """Exports assessments to file or Credo AI's governance app

        Note: to export to Credo AI's Governance App, CredoGovernance must be defined.

        Parameters
        ----------
        destination : str
            Where to send the report
            -- "credoai", a special string to send to Credo AI Governance App.
            -- Any other string, save assessment json to the output_directory indicated by the string.
        """
        if self.gov is None:
            raise ValidationError("CredoGovernance must be defined to export!")
        prepared_results = []
        for assessment in self.get_assessments(flatten=True):
            try:
                logging.info(
                    f"** Exporting assessment-{assessment.get_name()}")
                prepared_assessment = self._prepare_results(assessment)
                if prepared_assessment is not None:
                    prepared_results.append(prepared_assessment)
            except:
                raise Exception(
                    f"Assessment ({assessment.get_name()}) failed preparation")
        if self.report is None:
            logging.warning(
                "No report is included. To include a report, run create_reports first")
        self.gov.export_assessment_results(prepared_results, destination, self.report, self.run_time)

    def get_assessments(self, flatten=False):
        """Return assessments defined

        Parameters
        ----------
        flatten: bool, optional
            If True, return one list of assessments. Otherwise return dictionary
            of the form {dataset: [list of assessments]}, default to False

        Returns
        -------
        list or dict
            list or dict of assessments
        """
        if flatten:
            all_assessments = []
            for bunch in self.assessments:
                all_assessments += bunch.assessments.values()
            return all_assessments
        return self.assessments

    def get_datasets(self):
        datasets = {}
        if self.assessment_dataset is not None:
            datasets['validation'] = self.assessment_dataset
        if self.training_dataset is not None:
            datasets['training'] = self.training_dataset
        return datasets

    def get_artifacts(self):
        artifacts = {}
        if self.assessment_dataset is not None:
            artifacts['validation'] = self.assessment_dataset
        if self.training_dataset is not None:
            artifacts['training'] = self.training_dataset
        if self.model is not None:
            artifacts['model'] = self.model
        return artifacts

    def get_governance(self):
        return self.gov

    def get_report(self):
        return self.report

    def get_results(self):
        """Return results of assessments"""
        return {bunch.name: {a.get_name(): a.get_results() for a in bunch.assessments.values()}
                for bunch in self.get_assessments()}

    def display_results(self, assessments=None):
        """Display results from all assessment reporters
        
        Parameters
        ----------
        assessments : str or list of assessment names, optional
            List of assessments to display results from. If None, display
            all assessments. Assessments should be taken from the list
            of assessments returned by lens.get_assessments
        """
        assessments = wrap_list(assessments)
        if not self.reporters:
            self._create_reporters()
        for reporter in self.reporters:
            name = reporter.assessment.name
            if assessments and name not in assessments:
                continue
            reporter.display_results_tables()
            reporter.plot_results()
        return self

    def _apply_dev_mode(self, dev_mode):
        if dev_mode:
            if dev_mode == True:
                dev_mode = 0.1
            if self.assessment_dataset:
                self.assessment_dataset.dev_mode(self.dev_mode)
            if self.training_dataset:
                self.training_dataset.dev_mode(self.dev_mode)

    def _create_reporters(self):
        for assessment in self.get_assessments(flatten=True):
            name = assessment.get_name()
            reporter = assessment.get_reporter()
            if reporter is not None:
                logging.info(
                    f"Reporter creating notebook for assessment-{name}")
                reporter.create_notebook()
                self.reporters.append(reporter)
            else:
                logging.info(f"No reporter found for assessment-{name}")

    def _gather_meta(self, assessment):
        if assessment.data_name == self.assessment_dataset.name:
            dataset_id = self.gov.dataset_id
        elif assessment.data_name == self.training_dataset.name:
            dataset_id = self.gov.training_dataset_id
        return {'process': f'Lens-v{__version__}_{assessment.name}',
                'dataset_id': dataset_id}

    def _get_credo_destination(self, to_model=True):
        """Get destination for export and ensure all artifacts are registered"""
        self._register_artifacts()
        destination = self.gov.model_id if to_model else self.gov.dataset_id
        label = 'model' if to_model else 'dataset'
        logging.info(f"**** Destination for export: {label} id-{destination}")
        return destination

    def _init_assessments(self):
        for bunch in self.assessments:
            for assessment in bunch.assessments.values():
                kwargs = deepcopy(self.assessment_plan.get(assessment.name, {}))
                reqs = assessment.get_requirements()
                if reqs['model_requirements']:
                    kwargs['model'] = bunch.model
                if reqs['data_requirements']:
                    kwargs['data'] = bunch.primary_dataset
                if reqs['training_data_requirements']:
                    kwargs['training_data'] = bunch.secondary_dataset
                try:
                    assessment.init_module(**kwargs)
                except:
                    raise ValidationError(f"Assessment ({assessment.get_name()}) could not be initialized."
                                          " Ensure the assessment plan is passing the required parameters"
                                          )

    def _prepare_results(self, assessment, **kwargs):
        metadata = self._gather_meta(assessment)
        return assessment.prepare_results(metadata, **kwargs)

    def _register_artifacts(self):
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

    def _select_assessments(self, candidate_assessments=None):        
        # generate all possible artifacts combinations
        artifacts = self.get_artifacts()

        artifacts_combinations = []
        for i in range(1,len(artifacts)+1):
            artifacts_combinations.extend(list(map(dict, combinations(artifacts.items(), i))))

        # filter undesirable combinations
        filtered_keys = [{'training', 'model'}]
        artifacts_combinations = [c for c in artifacts_combinations 
                                  if set(c.keys()) not in filtered_keys]

        # create bunches
        assessment_bunches = []
        for af_comb in artifacts_combinations:
            bunch_name = '_'.join(list(af_comb.keys()))
            primary = af_comb.get('validation') or af_comb.get('training')
            secondary = af_comb.get('training') if af_comb.get('validation') else None
            assessment_bunch = AssessmentBunch(bunch_name, af_comb.get('model'), primary, secondary)
            assessment_bunch.set_usable_assessments(candidate_assessments)
            if assessment_bunch.assessments:
                assessment_bunches.append(assessment_bunch)        
        return assessment_bunches


def set_logging_level(logging_level):
    """Alias for absl.logging.set_verbosity"""
    logging.set_verbosity(logging_level)


