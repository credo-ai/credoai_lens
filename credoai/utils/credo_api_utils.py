import json
import os
import requests
import time
from collections import defaultdict
from credoai.utils.constants import CREDO_URL
from credoai.utils.common import get_project_root, json_dumps, IntegrationError
from dotenv import dotenv_values
from json_api_doc import deserialize, serialize


def read_config():
    config_file = os.path.join(os.path.expanduser('~'),
                               '.credoconfig')
    if not os.path.exists(config_file):
        # return example
        config = {'API_KEY': 'empty',
                  'TENANT': 'empty'}
    else:
        config = dotenv_values(config_file)
    config['API_URL'] = os.path.join(config.get(
        'CREDO_URL', CREDO_URL), f"api/v1/{config['TENANT']}")
    return config


def exchange_token():
    data = {"api_token": CONFIG['API_KEY'], "tenant": CONFIG['TENANT']}
    headers = {'content-type': 'application/json', 'charset': 'utf-8'}
    auth_url = os.path.join(CONFIG.get(
        'CREDO_URL', CREDO_URL), 'auth', 'exchange')
    r = requests.post(auth_url, json=data, headers=headers)
    return f"Bearer {r.json()['access_token']}"


def refresh_token():
    global HEADERS
    key = exchange_token()
    HEADERS['Authorization'] = key
    SESSION.headers.update(HEADERS)

def renew_access_token(func):
    def wrapper(*args, **kwargs):
        response = func(*args, **kwargs)
        if response.status_code == 401:
            refresh_token()
            response = func(*args, **kwargs)
        response.raise_for_status()
        return response
    return wrapper


CONFIG = read_config()
HEADERS = {"Authorization": None, "accept": "application/vnd.api+json"}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def get_end_point(end_point):
    return os.path.join(CONFIG['API_URL'], end_point)


@renew_access_token
def submit_request(request, end_point, **kwargs):
    response = SESSION.request(request, end_point, **kwargs)
    return response


def get_assessment(assessment_id):
    end_point = get_end_point(f"use_case_assessments/{assessment_id}")
    return deserialize(submit_request('get', end_point).json())


def get_associated_models(use_case_id):
    end_point = get_end_point(f"use_cases/{use_case_id}?include=models")
    return deserialize(submit_request('get', end_point).json())['models']


def get_assessment_spec(assessment_spec_url):
    """Get the assessment spec
    
    The assessment spec includes all information needed to assess a model and integrate
    with the Credo AI Governance Platform. This includes the necessary IDs, as well as 
    the assessment plan
    """
    try:
        end_point = get_end_point(assessment_spec_url)
        downloaded_spec = deserialize(submit_request('get', end_point).json())
        assessment_spec = {k: v for k,
                           v in downloaded_spec.items() if '_id' in k}
        assessment_spec['assessment_plan'] = downloaded_spec['assessment_plan']
        assessment_spec['policy_questions'] = _process_policies(downloaded_spec['policies'])
    except requests.exceptions.HTTPError:
        raise IntegrationError("Failed to retrieve assessment spec. Check that the url is correct")
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
    return _get_name(dataset_id, 'datasets')


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
    return _get_name(model_id, 'models')


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
    return _get_name(use_case_id, 'use_cases')


def get_dataset_by_name(dataset_name):
    """Returns governance info (ids) for dataset using its name"""
    returned = _get_by_name(dataset_name, 'datasets')
    if returned:
        return {'name': dataset_name,
                'dataset_id': returned['id']}
    return None


def get_model_by_name(model_name):
    """Returns governance info (ids) for model using its name"""
    returned = _get_by_name(model_name, 'models')
    if returned:
        return {'name': model_name,
                'model_id': returned['id']}
    return None


def get_use_case_by_name(use_case_nmae):
    """Returns governance info (ids) for use case using its name"""
    returned = _get_by_name(use_case_nmae, 'use_cases')
    if returned:
        return {'name': use_case_nmae,
                'use_case_id': returned['id']}
    return None


def post_assessment(use_case_id, model_id, data):
    end_point = get_end_point(
        f"use_cases/{use_case_id}/models/{model_id}/assessments")
    request = submit_request('post', end_point, data=json_dumps(serialize(data)),
                             headers={"content-type": "application/vnd.api+json"})
    assessment_id = deserialize(request.json())['id']
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
    end_point = get_end_point(f"datasets")
    project = {"name": dataset_name, "$type": "string"}
    data = json.dumps(serialize(project))
    response = _register_artifact(data, end_point)
    return {'name': dataset_name, 'dataset_id': response['id']}


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
    end_point = get_end_point(f"models")
    model = {"name": model_name,
             "version": "1.0",
             "$type": "string"}
    data = json.dumps(serialize(model))
    response = _register_artifact(data, end_point)
    return {'name': model_name, 'model_id': response['id']}


def register_model_to_usecase(use_case_id, model_id):
    data = {"data": [{"id": model_id, "type": "models"}]}
    end_point = get_end_point(f"use_cases/{use_case_id}/relationships/models")
    submit_request('post', end_point, data=json.dumps(data), headers={
                   "content-type": "application/vnd.api+json"})


def register_dataset_to_model(model_id, dataset_id):
    data = {"data": {"id": dataset_id, "type": "datasets"}}
    end_point = get_end_point(f"models/{model_id}/relationships/dataset")
    submit_request('patch', end_point, data=json.dumps(data), headers={
                   "content-type": "application/vnd.api+json"})


def register_dataset_to_model_usecase(use_case_id, model_id, dataset_id):
    data = serialize(
        {"dataset_id": dataset_id, '$type': 'string', 'id': 'resource-id'})
    end_point = get_end_point(
        f"use_cases/{use_case_id}/models/{model_id}/config")
    submit_request(
        "patch",
        end_point,
        data=json.dumps(data),
        headers={"content-type": "application/vnd.api+json"},
    )


def _register_artifact(data, end_point):
    try:
        response = submit_request('post', end_point, data=data,
                                  headers={"content-type": "application/vnd.api+json"})
    except requests.exceptions.HTTPError as err:
        if err.response.status_code == 422:
            raise IntegrationError(
                "Failed to register artifact. Ensure that the name is unique")
        else:
            raise
    return deserialize(response.json())


def _get_by_name(name, endpoint):
    """Given a name, return the id"""
    end_point = get_end_point(endpoint)
    params = {"filter[name][value]": name,
              "filter[name][type]": "match"}
    returned = deserialize(submit_request(
        'get', end_point, params=params).json())
    if len(returned) == 1:
        return returned[0]
    return None


def _get_name(artifact_id, artifact_type):
    """Given an ID, return the name"""
    try:
        end_point = get_end_point(f"{artifact_type}/{artifact_id}")
        return deserialize(submit_request('get', end_point).json())['name']
    except requests.exceptions.HTTPError as err:
        if err.response.status_code == 400:
            raise IntegrationError(
                f"No {artifact_type} found with id: {artifact_id}")

def _process_policies(policies):
    """Returns list of binary questions"""
    policies = sorted(policies, key = lambda x: x['stage_key'])
    question_list = defaultdict(list)
    for policy in policies:
        for control in policy['controls']:
            label = policy['stage_key']
            questions = control['questions']
            filtered_questions = [f"{control['key']}: {q['question']}" for q in questions 
                                  if q.get('options') == ['Yes', 'No']]
            if filtered_questions:
                question_list[label] += filtered_questions
    return question_list