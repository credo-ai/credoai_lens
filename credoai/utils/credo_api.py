"""
Credo API functions
"""

from collections import defaultdict
import requests
from credoai.utils.credo_api_client import CredoApiClient
from credoai.utils.common import IntegrationError


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
    def get_assessment(self, assessment_id):
        """
        get assessment
        """
        return self._client.get(f"use_case_assessments/{assessment_id}")

    def create_assessment(self, use_case_id, model_id, data):
        """
        create assessment
        """
        endpoint = f"use_cases/{use_case_id}/models/{model_id}/assessments"
        return self._client.post(endpoint, data)

    def get_assessment_spec(self, spec_url):
        """
        Get assessment spec
        """
        try:
            downloaded_spec = self._client.get(spec_url)
            assessment_spec = {k: v for k,
                               v in downloaded_spec.items() if "_id" in k}
            assessment_spec["assessment_plan"] = downloaded_spec["assessment_plan"]
            assessment_spec["policy_questions"] = _process_policies(
                downloaded_spec["policies"]
            )
        except requests.exceptions.HTTPError:
            raise IntegrationError(
                "Failed to retrieve assessment spec. Check that the url is correct"
            )
        return assessment_spec

    def get_dataset_by_name(self, name):
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

    def get_use_case_by_name(self, name):
        """
        Get use_case by name
        """
        params = {"filter[name]": name}
        use_cases = self._client.get("use_cases", params=params)
        if len(use_cases) > 0:
            return use_cases[0]

        return None

    def register_dataset(self, name):
        """
        Find a dataset by name, if it does not exist create one.
        """
        dataset = self.get_dataset_by_name(name)
        if dataset:
            return dataset

        data = {"name": name, "$type": "datasets"}
        return self._client.post("datasets", data)

    def register_model(self, name):
        """
        Find a model by name, if it does not exist create one.
        """
        model = self.get_model_by_name(name)
        if model:
            return model

        data = {"name": name, "version": "1.0", "$type": "models"}
        return self._client.post("models", data)

    def register_model_to_usecase(self, use_case_id, model_id):
        """
        Register a model to use_case.
        """
        endpoint = f"use_cases/{use_case_id}/relationships/models"
        data = [{"id": model_id, "$type": "models"}]
        self._client.post(endpoint, data)

    def register_dataset_to_model(self, model_id, dataset_id):
        """
        Register a dataset to model.
        """
        endpoint = f"models/{model_id}/relationships/dataset"
        data = {"id": dataset_id, "$type": "datasets"}
        self._client.patch(endpoint, data)

    def register_dataset_to_model_usecase(self, use_case_id, model_id, dataset_id):
        """
        Register a dataset to use_case model.
        """
        endpoint = f"use_cases/{use_case_id}/models/{model_id}/config"
        data = {
            "dataset_id": dataset_id,
            "$type": "use_case_model_configs",
            "id": "unknown"
        }
        self._client.patch(endpoint, data)
