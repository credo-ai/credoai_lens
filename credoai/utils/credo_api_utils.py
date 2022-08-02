from collections import defaultdict
import requests
from credoai.utils.common import IntegrationError
from credoai.utils.credo_api_client import CredoApiClient

client = CredoApiClient()


def get_assessment(assessment_id):
    return client.get(f"use_case_assessments/{assessment_id}")


def get_associated_models(use_case_id):
    return client.get(f"use_cases/{use_case_id}?include=models")


def get_assessment_spec(assessment_spec_url):
    """Get the assessment spec

    The assessment spec includes all information needed to assess a model and integrate
    with the Credo AI Governance Platform. This includes the necessary IDs, as well as
    the assessment plan
    """
    try:
        downloaded_spec = client.get(assessment_spec_url)
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


def get_dataset_name(dataset_id):
    """Get dataset name form a dataset ID from Credo AI Governance App

    Parameters
    ----------
    dataset_id : string
        Identifier for Model on Credo AI Governance App

    Returns
    -------
    str
        The name of the Model
    """
    return _get_name(dataset_id, "datasets")


def get_model_name(model_id):
    """Get model name form a model ID from Credo AI Governance App

    Parameters
    ----------
    model_id : string
        Identifier for Model on Credo AI Governance App

    Returns
    -------
    str
        The name of the Model
    """
    return _get_name(model_id, "models")


def get_use_case_name(use_case_id):
    """Get use_case name form a use_case ID from Credo AI Governance App

    Parameters
    ----------
    use_case_id : string
        Identifier for Model on Credo AI Governance App

    Returns
    -------
    str
        The name of the Model
    """
    return _get_name(use_case_id, "use_cases")


def get_dataset_by_name(dataset_name):
    """Returns governance info (ids) for dataset using its name"""
    returned = _get_by_name(dataset_name, "datasets")
    if returned:
        return {"name": dataset_name, "dataset_id": returned["id"]}
    return None


def get_model_by_name(model_name):
    """Returns governance info (ids) for model using its name"""
    returned = _get_by_name(model_name, "models")
    if returned:
        return {"name": model_name, "model_id": returned["id"]}
    return None


def get_use_case_by_name(use_case_nmae):
    """Returns governance info (ids) for use case using its name"""
    returned = _get_by_name(use_case_nmae, "use_cases")
    if returned:
        return {"name": use_case_nmae, "use_case_id": returned["id"]}
    return None


def post_assessment(use_case_id, model_id, data):
    response = client.post(
        f"use_cases/{use_case_id}/models/{model_id}/assessments", data)
    assessment_id = response["id"]
    return get_assessment(assessment_id)


def register_dataset(dataset_name):
    """Registers a dataset on Credo AI's Governance App

    Parameters
    ----------
    dataset_name : string
        Name for dataset on Credo AI's Governance App

    Returns
    --------
    dict : str
        Dictionary with Identifiers for dataset
        on Credo AI's Governance App
    """
    project = {"name": dataset_name, "$type": "string"}
    response = _register_artifact(project, "datasets")
    return {"name": dataset_name, "dataset_id": response["id"]}


def register_model(model_name):
    """Registers a model  on Credo AI's Governance App

    Parameters
    ----------
    model_name : string
        Name for Model on Credo AI's Governance App

    Returns
    --------
    dict : str
        Dictionary with Identifiers for Model and Project
        on Credo AI's Governance App
    """
    model = {"name": model_name, "version": "1.0", "$type": "string"}
    response = _register_artifact(model, "models")
    return {"name": model_name, "model_id": response["id"]}


def register_model_to_usecase(use_case_id, model_id):
    data = [{"id": model_id, "$type": "models"}]
    client.post(f"use_cases/{use_case_id}/relationships/models", data)


def register_dataset_to_model(model_id, dataset_id):
    data = {"id": dataset_id, "$type": "datasets"}
    client.patch(f"models/{model_id}/relationships/dataset", data)


def register_dataset_to_model_usecase(use_case_id, model_id, dataset_id):
    data = {"dataset_id": dataset_id, "$type": "string", "id": "resource-id"}
    client.patch(f"use_cases/{use_case_id}/models/{model_id}/config", data)


def _register_artifact(data, end_point):
    try:
        return client.post(end_point, data)
    except requests.exceptions.HTTPError as err:
        if err.response.status_code == 422:
            raise IntegrationError(
                "Failed to register artifact. Ensure that the name is unique"
            )
        else:
            raise


def _get_by_name(name, endpoint):
    """Given a name, return the id"""
    params = {"filter[name][value]": name, "filter[name][type]": "match"}
    returned = client.get(endpoint, params=params)
    if len(returned) == 1:
        return returned[0]
    return None


def _get_name(artifact_id, artifact_type):
    """Given an ID, return the name"""
    try:
        response = client.get(f"{artifact_type}/{artifact_id}")
        return response["name"]
    except requests.exceptions.HTTPError as err:
        if err.response.status_code == 400:
            raise IntegrationError(
                f"No {artifact_type} found with id: {artifact_id}")


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
