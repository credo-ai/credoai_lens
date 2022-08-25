"""Artifacts used by Lens to structure governance, models and data"""
# CredoLens relies on four classes
# - CredoGovernance
# - CredoModel
# - CredoData
# - CredoAssessment

# CredoGovernance contains the information needed
# to connect CredoLens with the Credo AI Governance App

# CredoModel follows an `adapter pattern` to convert
# any model into an interface CredoLens can work with.
# The functionality contained in CredoModel determines
# what assessments can be run.

# CredoData is a lightweight wrapper that stores data

# CredoAssessment is the interface between a CredoModel,
# CredoData and a module, which performs some assessment.

import os
from collections import defaultdict
from datetime import datetime

from absl import logging
from credoai.metrics.metrics import find_metrics
from credoai.utils.common import json_dumps, update_dictionary
from credoai.utils.constants import RISK_ISSUE_MAPPING

from .credo_api import CredoApi
from .credo_api_client import CredoApiClient
from .utils import prepare_assessment_payload, process_assessment_spec


class CredoGovernance:
    """Class to store governance data.

    This information is used to interact with the CredoAI
    Governance App.

    At least one of the credo_url or spec_path must be provided! If both
    are provided, the spec_path takes precedence.

    To make use of Governance App a .credo_config file must
    also be set up (see README)

    Parameters
    ----------
    spec_destination: str
        Where to find the assessment spec. Two possibilities. Either:
        * end point to retrieve assessment spec from credo AI's governance platform
        * The file location for the assessment spec json downloaded from
        the assessment requirements of an Use Case on Credo AI's
        Governance App
    """

    def __init__(self, spec_destination: str = None):
        self.assessment_spec = {}
        self.use_case_id = None
        self.model_id = None
        self.dataset_id = None
        self.training_dataset_id = None

        client = CredoApiClient()
        self._api = CredoApi(client=client)

        # set up assessment spec
        if spec_destination:
            self._process_spec(spec_destination)

    def get_assessment_plan(self):
        """Get assessment plan

        Return the assessment plan for the model defined
        by model_id.

        If not retrieved yet, attempt to retrieve the plan first
        from the AI Governance app.
        """
        assessment_plan = defaultdict(dict)
        missing_metrics = []
        for risk_issue, risk_plan in self.assessment_spec.get(
            "assessment_plan", {}
        ).items():
            metrics = [m["type"] for m in risk_plan]
            passed_metrics = []
            for m in metrics:
                found = bool(find_metrics(m))
                if not found:
                    missing_metrics.append(m)
                else:
                    passed_metrics.append(m)
            if risk_issue in RISK_ISSUE_MAPPING:
                update_dictionary(
                    assessment_plan[RISK_ISSUE_MAPPING[risk_issue]],
                    {"metrics": passed_metrics},
                )
        # alert about missing metrics
        for m in missing_metrics:
            logging.warning(
                f"Metric ({m}) is defined in the assessment plan but is not defined by Credo AI.\n"
                "Ensure you create a custom Metric (credoai.metrics.Metric) and add it to the\n"
                "assessment plan passed to lens"
            )
        # remove repeated metrics
        for plan in assessment_plan.values():
            plan["metrics"] = list(set(plan["metrics"]))
        return assessment_plan

    def get_policy_checklist(self):
        return self.assessment_spec.get("policy_questions")

    def get_info(self):
        """Return Credo AI Governance IDs"""
        to_return = self.__dict__.copy()
        del to_return["assessment_spec"]
        return to_return

    def get_defined_ids(self):
        """Return IDS that have been defined"""
        return [k for k, v in self.get_info().items() if v]

    def register(
        self,
        model_name=None,
        dataset_name=None,
        training_dataset_name=None,
        assessment_template=None,
    ):
        """Registers artifacts to Credo AI Governance App

        Convenience function to register multiple artifacts at once

        Parameters
        ----------
        model_name : str
            name of a model
        dataset_name : str
            name of a dataset used to assess the model
        training_dataset_name : str
            name of a dataset used to train the model
        assessment_template : str
            name of an assessment template that already exists on the governance platform,
            which will be applied to the model

        """
        if model_name:
            self._register_model(model_name)
        if dataset_name:
            self._register_dataset(dataset_name)
        if training_dataset_name:
            self._register_dataset(training_dataset_name, register_as_training=True)
        if assessment_template and self.model_id:
            self._api.apply_assessment_template(
                assessment_template, self.use_case_id, self.model_id
            )
        # reset assessment spec if assessment_plan not defined yet
        if not self.assessment_spec.get("assessment_plan", {}):
            self._process_spec(
                f"use_cases/{self.use_case_id}/models/{self.model_id}/assessment_spec",
                set_ids=False,
            )
            if self.assessment_spec.get("assessment_plan", {}):
                logging.info("Assessment plan downloaded after artifact registration")

    def export_assessment_results(
        self,
        assessment_results,
        reporter_assets=None,
        destination="credoai",
        assessed_at=None,
    ):
        """Export assessment json to file or credo

        Parameters
        ----------
        assessment_results : dict or list
            dictionary of metrics or
            list of prepared_results from credo_assessments. See lens.export for example
        reporter_assets : list, optional
            list of assets from a CredoReporter, by default None
        destination : str
            Where to send the report
            -- "credoai", a special string to send to Credo AI Governance App.
            -- Any other string, save assessment json to the output_directory indicated by the string.
        assessed_at : str, optional
            date when assessments were created, by default None
        """
        assessed_at = assessed_at or datetime.utcnow().isoformat()
        payload = ci.prepare_assessment_payload(
            assessment_results, reporter_assets=reporter_assets, assessed_at=assessed_at
        )
        if destination == "credoai":
            if self.use_case_id and self.model_id:
                self._api.create_assessment(self.use_case_id, self.model_id, payload)
                logging.info(
                    f"Successfully exported assessments to Credo AI's Governance App"
                )
            else:
                logging.error(
                    "Couldn't upload assessment to Credo AI's Governance App. "
                    "Ensure use_case_id is defined in CredoGovernance"
                )
        else:
            if not os.path.exists(destination):
                os.makedirs(destination, exist_ok=False)
            name_for_save = f"assessment_run-{assessed_at}.json"
            # change name in case of windows
            if os.name == "nt":
                name_for_save = name_for_save.replace(":", "-")
            output_file = os.path.join(destination, name_for_save)
            with open(output_file, "w") as f:
                f.write(json_dumps(payload))

    def _process_spec(self, spec_destination, set_ids=True):
        self.assessment_spec = ci.process_assessment_spec(spec_destination, self._api)
        if set_ids:
            self.use_case_id = self.assessment_spec["use_case_id"]
            self.model_id = self.assessment_spec["model_id"]
            self.dataset_id = self.assessment_spec["validation_dataset_id"]
            self.training_dataset_id = self.assessment_spec["training_dataset_id"]

    def _register_dataset(self, dataset_name, register_as_training=False):
        """Registers a dataset

        Parameters
        ----------
        dataset_name : str
            name of a dataset
        register_as_training : bool
            If True and model_id is defined, register dataset to model as training data,
            default False
        """
        prefix = ""
        if register_as_training:
            prefix = "training_"
        dataset_id = self._api.register_dataset(name=dataset_name)["id"]
        setattr(self, f"{prefix}dataset_id", dataset_id)

        if not register_as_training and self.model_id and self.use_case_id:
            self._api.register_dataset_to_model_usecase(
                use_case_id=self.use_case_id,
                model_id=self.model_id,
                dataset_id=self.dataset_id,
            )
        if register_as_training and self.model_id and self.training_dataset_id:
            logging.info(
                f"Registering dataset ({dataset_name}) to model ({self.model_id})"
            )
            self._api.register_dataset_to_model(self.model_id, self.training_dataset_id)

    def _register_model(self, model_name):
        """Registers a model

        If a project has not been registered, a new project will be created to
        register the model under.

        If an AI solution has been set, the model will be registered to that
        solution.
        """
        self.model_id = self._api.register_model(name=model_name)["id"]

        if self.use_case_id:
            logging.info(
                f"Registering model ({model_name}) to Use Case ({self.use_case_id})"
            )
            self._api.register_model_to_usecase(self.use_case_id, self.model_id)
