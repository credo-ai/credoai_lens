"""
Credo API functions
"""
from collections import defaultdict

import requests
from absl import logging
from credoai.utils.common import IntegrationError
from credoai.utils.credo_api_client import CredoApiClient


def _process_policies(policies):
    """Returns list of binary questions"""
    policies = sorted(policies, key=lambda x: x["stage_key"])
    question_list = defaultdict(list)
    for policy in policies:
        for control in policy["controls"]:
            label = policy["stage_key"]
            questions = control["questions"]
            filtered_questions = [
                f"{control['key']}: {q['question']}"
                for q in questions
                if q.get("options") == ["Yes", "No"]
            ]
            if filtered_questions:
                question_list[label] += filtered_questions
    return question_list


class CredoApi:
    """
    CredoApi holds Credo API functions
    """

    def __init__(self, client: CredoApiClient = None):
        self._client = client

    def set_client(self, client: CredoApiClient):
        """
        sets Credo Api Client
        """
        self._client = client

    # Not used
    def get_assessment(self, assessment_id: str):
        """
        get assessment
        """
        return self._client.get(f"use_case_assessments/{assessment_id}")

    def apply_assessment_template(self, template_name, use_case_id, model_id):
        """Applies an assessment template to a model under a use case

        The assessment templates are drawn from the list of assessment templates saved
        in the Governance Platform

        Parameters
        ----------
        template_name : str
            The name of an assessment template.
        use_case_id : string
            Identifier for Use Case on Credo AI Governance App
        model_id : string
            Identifier for Model on Credo AI Governance App
        """
        # get template
        templates = self._client.get("assessment_plan_templates")
        filtered = [t for t in templates if t["name"] == template_name]
        if len(filtered) > 1:
            raise IntegrationError(
                f"More than one template found with the name: {template_name}"
            )
        elif not filtered:
            raise IntegrationError(f"No template found with the name: {template_name}")
        template = filtered[0]

        # get assessment plan
        plan = self._client.get(
            f"use_cases/{use_case_id}/models/{model_id}/assessment_plans/draft"
        )

        # apply template
        data = {"assessment_plan_id": plan["id"], "$type": "string"}
        self._client.post(f"assessment_plan_templates/{template['id']}/apply", data)
        self._client.post(f"assessment_plans/{plan['id']}/publish", data)

    def create_assessment(self, use_case_id: str, model_id, data: str):
        """
        create assessment
        """
        endpoint = f"use_cases/{use_case_id}/models/{model_id}/assessments"
        return self._client.post(endpoint, data)

    def get_assessment_spec(self, spec_url: str):
        """
        Get assessment spec
        """
        try:
            downloaded_spec = self._client.get(spec_url)
            assessment_spec = {k: v for k, v in downloaded_spec.items() if "_id" in k}
            assessment_spec["assessment_plan"] = downloaded_spec["assessment_plan"]
            assessment_spec["policy_questions"] = _process_policies(
                downloaded_spec["policies"]
            )
        except requests.exceptions.HTTPError:
            raise IntegrationError(
                "Failed to retrieve assessment spec. Check that the url is correct"
            )
        return assessment_spec

    def get_dataset_by_name(self, name: str):
        """
        Get dataset by name
        """
        params = {"filter[name]": name}
        datasets = self._client.get("datasets", params=params)
        if len(datasets) > 0:
            return datasets[0]

        return None

    def get_model_by_name(self, name):
        """
        Get model by name
        """
        params = {"filter[name]": name}
        models = self._client.get("models", params=params)
        if len(models) > 0:
            return models[0]

        return None

    def get_use_case_by_name(self, name: str):
        """
        Get use_case by name
        """
        params = {"filter[name]": name}
        use_cases = self._client.get("use_cases", params=params)
        if len(use_cases) > 0:
            return use_cases[0]

        return None

    def register_dataset(self, name: str):
        """
        Find a dataset by name, if it does not exist create one.
        """
        dataset = self.get_dataset_by_name(name)
        if dataset:
            logging.info(f"Found dataset ({name}) registered on platform")
            return dataset

        logging.info(f"Registering dataset: ({name})")
        data = {"name": name, "$type": "datasets"}
        return self._client.post("datasets", data)

    def register_model(self, name: str, version: str = "1.0"):
        """
        Find a model by name, if it does not exist create one.
        """
        model = self.get_model_by_name(name)
        if model:
            logging.info(f"Found model ({name}) registered on platform")
            return model

        logging.info(f"Registering model: ({name})")
        data = {"name": name, "version": version, "$type": "models"}
        return self._client.post("models", data)

    def register_model_to_usecase(self, use_case_id: str, model_id: str):
        """
        Register a model to use_case.
        """
        endpoint = f"use_cases/{use_case_id}/relationships/models"
        data = [{"id": model_id, "$type": "models"}]
        self._client.post(endpoint, data)

    def register_dataset_to_model(self, model_id: str, dataset_id: str):
        """
        Register a dataset to model.
        """
        endpoint = f"models/{model_id}/relationships/dataset"
        data = {"id": dataset_id, "$type": "datasets"}
        self._client.patch(endpoint, data)

    def register_dataset_to_model_usecase(
        self, use_case_id: str, model_id: str, dataset_id: str
    ):
        """
        Register a dataset to use_case model.
        """
        endpoint = f"use_cases/{use_case_id}/models/{model_id}/config"
        data = {
            "dataset_id": dataset_id,
            "$type": "use_case_model_configs",
            "id": "unknown",
        }
        self._client.patch(endpoint, data)
