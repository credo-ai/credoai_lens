"""Credo AI's AI Assessment Framework"""

from absl import logging
from copy import deepcopy
from credoai.artifacts import CredoGovernance, CredoModel, CredoData
from credoai.assessment.credo_assessment import CredoAssessment
from credoai.assessment import get_usable_assessments
from credoai.reporting.reports import MainReport
from credoai.utils.common import (
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
        if isinstance(governance, str):
            self.gov = CredoGovernance(use_case_id=governance, warning_level=warning_level)
        else:
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
        self
        """
        assessment_kwargs = assessment_kwargs or {}
        assessment_results = {}
        for name, assessment in self.assessments.items():
            logging.info(f"Running assessment-{name}")
            kwargs = assessment_kwargs.get(name, {})
            assessment_results[name] = assessment.run(**kwargs).get_results()
        self.run_time = datetime.now().isoformat()
        return self

    def create_reports(self, export=False, display_results=False):
        """Creates notebook reports

        Creates jupyter notebook reports for every assessment that
        has reporting functionality defined. It then concatenates the reports into
        one final report. The reports are then able to be saved to file (if report_directory
        is defined), or exported to Credo AI's Governance App (if export=True).

        Note: to export to Credo AI's Governance App, CredoGovernance must be passed
        to Lens with a defined "use_case_id". "model_id" is also required, but if no "model_id" 
        is explicitly provided, a model will be registered and used.

        Parameters
        ----------
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
        final_report = MainReport(f"{ci.RISK.title()} Report", reporters.values())
        final_report.create_report(self)
        # exporting
        if export:
                self.export_assessments(export, report=final_report)
        return reporters, final_report

    def export_assessments(self, export="credoai", report=None):
        """Exports assessments to file or Credo AI's governance app

        Parameters
        ----------
        export : str
            If the special string "credoai", Credo AI Governance App.
            If a string, save assessment json to the output_directory indicated by the string.
            If False, do not export, by default "credoai""
        report : credoai.reports.Report, optional
            a report to include with the export. by default None
        """        
        prepared_results = []
        for name, assessment in self.assessments.items():
            logging.info(f"** Exporting assessment-{name}")
            prepared_results.append(self._prepare_results(assessment))
        payload = ci.prepare_assessment_payload(prepared_results, report=report, assessed_at=self.run_time)

        if export == 'credoai':
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
