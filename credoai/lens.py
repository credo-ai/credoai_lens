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
from typing import List, Union, Optional
import collections.abc
import credoai.integration as ci
import shutil


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
        self.spec = {}
        self.warning_level = warning_level
        self.dev_mode = dev_mode
        self.run = False

        if assessments == 'auto':
            assessments = self._select_assessments()
        else:
            self._validate_assessments(assessments)
            assessments = assessments
        self.assessments = {a.name: a for a in assessments}
        self.user_id = user_id

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
        self.run = True
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
            If a boolean, and true, export to Credo AI Governance Platform.
            If a string, save notebook to the output_directory indicated by the string.
            If False, do not export, by default False
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
        if self.run == False:
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
        final_report = MainReport(report_name, reporters.values())
        final_report.create_report(self)
        # exporting
        names = self.get_artifact_names()
        name_for_save = f"{report_name}_model-{names['model']}_data-{names['dataset']}.html"
        if isinstance(export, str):
            final_report.write_notebook(
                path.join(export, name_for_save), as_html=True)
        elif export:
            model_id = self._get_credo_destination()
            defined_ids = self.gov.get_defined_ids()
            if len({'model_id', 'use_case_id'}.intersection(defined_ids)) == 2:
                final_report.send_to_credo(
                    self.gov.use_case_id, self.gov.model_id)
                logging.info(
                    f"Exporting complete report to Credo AI's Governance Platform")
            else:
                logging.warning("Couldn't upload report to Credo AI's Governance Platform. "
                                "Ensure use_case_id is defined in CredoGovernance")
        return reporters, final_report

    def get_assessments(self):
        return self.assessments

    def get_governance(self):
        return self.gov

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
        names = self.get_artifact_names()
        results_file = (f"AssessmentResults_assessment-{assessment_name}_"
                        f"model-{names['model']}_data-{names['dataset']}.json")
        output_file = path.join(output_directory, results_file)
        ci.export_to_file(metric_records, output_file)

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

    def _select_assessments(self):
        return list(get_usable_assessments(self.model, self.data).values())

    def _validate_assessments(self, assessments):
        for assessment in assessments:
            if not assessment.check_requirements(self.model, self.data):
                raise ValidationError(
                    f"Model or Data does not conform to {assessment.name} assessment's requirements")
