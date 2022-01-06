import os
import requests
import time
from collections import defaultdict
from credoai.utils.common import get_project_root
from dotenv import dotenv_values
from json_api_doc import deserialize

def read_config():
    config_file = os.path.join(os.path.expanduser('~'), 
                               '.credoconfig')
    if not os.path.exists(config_file):
        # return example
        config = {
            'TENANT': 'empty',
            'CREDO_URL': 'empty',
            'API_KEY': 'empty'
        }
    else:
        config = dotenv_values(config_file)
    config['API_URL'] = os.path.join(config['CREDO_URL'], "api/v1/credoai")
    return config

def exchange_token():
    data = {"api_token": CONFIG['API_KEY'], "tenant": CONFIG['TENANT']}
    headers = {'content-type': 'application/json', 'charset': 'utf-8'}
    auth_url = os.path.join(CONFIG['CREDO_URL'], 'auth', 'exchange')
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

def get_alignment_spec(ai_solution_id, version='latest'):
    """Get solution spec for an AI solution from credoai.Governance Platform
    
    Parameters
    ----------
    ai_solution_id : string
        identifier for AI solution on Credo AI Governance Platform
        
    Returns
    -------
    dict
        The spec for the AI solution
    """
    end_point = get_end_point(f"ai_solutions/{ai_solution_id}/scopes")
    if version is not None:
        end_point = os.path.join(end_point, version)
    return deserialize(submit_request('get', end_point).json())

def get_model_name(model_id):
    """Get model name form a model ID from credoai.Governance Platform
    
    Parameters
    ----------
    model_id : string
        Identifier for Model on Credo AI Governance Platform
    
    Returns
    -------
    str
        The name of the Model
    """
    end_point = get_end_point(f"models/{model_id}")
    return deserialize(submit_request('get', end_point).json())['name']

def get_aligned_metrics(ai_solution_id, version='latest'):
    """Get aligned metrics frmo Credo's Governance Platform
    
    Parameters
    ----------
    ai_solution_id : string
        Identifier for AI solution on Credo AI Governance Platform
        
    Returns
    -------
    dict
        The aligned metrics for each model contained in the AI solution.
        Format: {"Model": {"Metric1": (lower_bound, upper_bound), ...}}
    """
    spec = get_alignment_spec(ai_solution_id, version=version)
    try:
        models = spec['model_ids']
    except KeyError:
        return spec
    metric_dict = {m: defaultdict(dict) for m in models}
    metrics = spec['metrics']
    for metric in metrics:
        bounds = (metric['lower_threshold'], metric['upper_threshold'])
        metric_dict[metric['model_id']][metric['type']] = bounds
    return metric_dict


def patch_metrics(model_id, model_record):
    """Send a model record object to Credo's Governance Platform
    
    Parameters
    ----------
    model_id : string
        Identifier for Model on Credo AI Governance Platform
    model_record : Record
        Model Record object, see credo.integration.MutliRecord
    """
    end_point = get_end_point(f"models/{model_id}/relationships/metrics")
    return submit_request('patch', end_point, data=model_record.credoify(), headers={"content-type": "application/vnd.api+json"})
    
    
def post_figure(model_id, figure_record):
    """Send a figure record object to Credo's Governance Platform
    
    Parameters
    ----------
    model_id : string
        Identifier for Model on Credo AI Governance Platform
    figure record : Record
        Figure Record object, see credo.integration.FigureRecord
    """
    end_point = get_end_point(f"models/{model_id}/model_assets")
    return submit_request('post', end_point, data=figure_record.credoify(), headers={"content-type": "application/vnd.api+json"})