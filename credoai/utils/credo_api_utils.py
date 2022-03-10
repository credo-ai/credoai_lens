import json
import os
import pandas as pd
import requests
import time
from credoai.utils.common import get_project_root, IntegrationError
from dotenv import dotenv_values
from json_api_doc import deserialize, serialize

CREDO_URL = "https://api.credo.ai"

def read_config():
    config_file = os.path.join(os.path.expanduser('~'),
                               '.credoconfig')
    if not os.path.exists(config_file):
        # return example
        config = {'API_KEY': 'empty',
                  'TENANT': 'empty'}
    else:
        config = dotenv_values(config_file)
    config['API_URL'] = os.path.join(config.get('CREDO_URL', CREDO_URL), "api/v1/credoai")
    return config


def exchange_token():
    data = {"api_token": CONFIG['API_KEY'], "tenant": CONFIG['TENANT']}
    headers = {'content-type': 'application/json', 'charset': 'utf-8'}
    auth_url = os.path.join(CONFIG.get('CREDO_URL', CREDO_URL), 'auth', 'exchange')
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
    end_point = get_end_point(f"assessments/{assessment_id}")
    return deserialize(submit_request('get', end_point).json())

def get_associated_models(use_case_id):
    end_point = get_end_point(f"use_cases/{use_case_id}?include=models")
    return deserialize(submit_request('get', end_point).json())['models']

def get_technical_spec(use_case_id, version='latest'):
    """Get technical specifications for an Use Case from Credo AI Governance App

    Parameters
    ----------
    use_case_id : string
        identifier for Use Case on Credo AI Governance App
    version : str
        "latest", for latest published spec, or "draft". If "latest"
        cannot be found, will look for a draft.

    Returns
    -------
    dict
        The spec for the Use Case
    """
    base_end_point = get_end_point(f"use_cases/{use_case_id}/scopes")
    if version is not None:
        end_point = os.path.join(base_end_point, version)
    try:
        return deserialize(submit_request('get', end_point).json())
    except requests.exceptions.HTTPError:
        try: 
            end_point = os.path.join(base_end_point, 'draft')
            return deserialize(submit_request('get', end_point).json())
        except requests.exceptions.HTTPError:
            raise IntegrationError("Failed to download technical spec. Check that your Use Case ID exists")


def get_survey_results(use_case_id, survey_key='FAIR'):
    survey_end_point = get_end_point(f"use_cases/{use_case_id}/surveys")
    answer_end_point = get_end_point(f"use_cases/{use_case_id}"
                                     "/scopes/draft/final_survey_answers")
    all_surveys = deserialize(submit_request('get', survey_end_point).json())
    all_answers = deserialize(submit_request('get', answer_end_point).json())
    # filter
    survey = [s for s in all_surveys if s['id'] == survey_key][0]['questions']
    answers = [a for a in all_answers if a['survey_key']
               == survey_key][0]['answers']
    # combine
    survey = pd.DataFrame(survey).set_index('id')
    survey = pd.concat([survey, pd.Series(answers, name='answer')], axis=1)
    return survey


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
    end_point = get_end_point(f"models/{model_id}")
    return deserialize(submit_request('get', end_point).json())['name']


def get_dataset_by_name(dataset_name):
    """Returns governance info (ids) for dataset using its name"""
    returned = _get_by_name(dataset_name, 'models')
    if returned:
        return {'name': dataset_name,
                'dataset_id': returned['id']}
    return None


def get_model_by_name(model_name):
    """Returns governance info (ids) for model using its name"""
    returned = _get_by_name(model_name, 'models')
    if returned:
        return {'name': model_name,
                'model_id': returned['id'],
                'model_project_id': returned['model_project_id']}
    return None


def get_model_project_by_name(project_name):
    """Returns governance info (ids) for model_project using its name"""
    returned = _get_by_name(project_name, 'models')
    if returned:
        return {'name': project_name,
                'dataset_id': returned['id']}
    return None

def get_use_case_by_name(use_case_nmae):
    """Returns governance info (ids) for model_project using its name"""
    returned = _get_by_name(use_case_nmae, 'use_cases')
    if returned:
        return {'name': use_case_nmae,
                'use_case_id': returned['id']}
    return None

def post_assessment(use_case_id, model_id, data):
    end_point = get_end_point(f"use_cases/{use_case_id}/models/{model_id}/assessments")
    request =  submit_request('post', end_point, data=data, 
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


def register_model(model_name, model_project_id=None):
    """Registers a model  on Credo AI's Governance App

    Parameters
    ----------
    model_name : string
        Name for Model on Credo AI's Governance App
    model_project_id : string
        Identifier for Project on Credo AI's Governance App.
        If not provided, a Project will automatically be created
        with the name {model_name} project.

    Returns
    --------
    dict : str
        Dictionary with Identifiers for Model and Project
        on Credo AI's Governance App
    """
    if model_project_id is None:
        model_project_id = register_project(f'{model_name} project')[
            'model_project_id']
    end_point = get_end_point(f"models")
    model = {"name": model_name,
             "version": "1.0",
             "model_project_id": model_project_id,
             "$type": "string"}
    data = json.dumps(serialize(model))
    response = _register_artifact(data, end_point)
    return {'name': model_name, 'model_id': response['id'], 'model_project_id': model_project_id}


def register_project(project_name):
    """Registers a model project on Credo AI's Governance App

    Parameters
    ----------
    project_name : string
        Name for Project on Credo AI's Governance App

    Returns
    --------
    dict : str
        Dictionary with Identifiers for Project
        on Credo AI's Governance App
    """
    end_point = get_end_point(f"model_projects")
    project = {"name": project_name, "$type": "string"}
    data = json.dumps(serialize(project))
    response = _register_artifact(data, end_point)
    return {'name': project_name, 'model_project_id': response['id']}
    
def register_model_to_use_case(use_case_id, model_id):
    data = {"data": [{"id": model_id, "type": "models"}]}
    end_point = get_end_point(f"use_cases/{use_case_id}/relationships/models")
    submit_request('post', end_point, data=json.dumps(data), headers={"content-type": "application/vnd.api+json"})

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
    end_point = get_end_point(endpoint)
    params = {"filter[name][value]": name,
              "filter[name][type]": "match"}
    returned = deserialize(submit_request(
        'get', end_point, params=params).json())
    if len(returned) == 1:
        return returned[0]
    return None
