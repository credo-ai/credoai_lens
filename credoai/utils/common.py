import json
import numpy as np
import pandas as pd
import os
import requests
from pathlib import Path

class ValidationError(Exception):
    pass

class SupressSettingWithCopyWarning:
    def __enter__(self):
        pd.options.mode.chained_assignment = None

    def __exit__(self, *args):
        pd.options.mode.chained_assignment = 'warn'
    
def get_project_root() -> Path:
    return Path(__file__).parent.parent

def wrap_list(obj):
    if type(obj) == str:
        obj = [obj]
    try:
        iter(obj)
    except TypeError:
        obj = [obj]
    return obj

def remove_suffix(text, suffix):
    return text[:-len(suffix)] if text.endswith(suffix) and len(suffix) != 0 else text

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
# Saves data to json file
def save_to_json(data, filename):
    with open(filename, 'w') as fp:
        json.dump(data, fp)
        
# Turns JSON encoded data into Python object
def load_from_json(file):
    with open(file, 'r', encoding="utf-8") as fp:
        return json.load(fp)

# Given a JSON results file, returns the file transformed 
# into a pandas dataframe
def results_to_df(save_file, query=None):
    df = pd.DataFrame.from_dict(load_from_json(save_file), orient="index")
    if query:
        df = df.query(query)
    return df

def load_data():
    # The file to load the results from. 
    results_file = os.path.join(str(get_project_root()), 'credo', 'data', 'fairface.json')
    #Since FairFace does not provide bounding boxes for the images, 
    # there is ambiguity in cases where Rekognition detects multiple faces, so only images where one face is detected are use
    query = 'Num_faces == 1'
    results_df = results_to_df(results_file, query=query)
    
    results_df = results_df.replace({'Male': 1, 'Female': 0})
    results_df['Confidence'] /= 100
    results_df['image_id'] = [int(i.split('/')[-1][:-4]) for i in results_df['Filename']]
    results_df.reset_index(drop=True, inplace=True)
    results_df.drop(['Filename', 'Num_faces'], axis=1, inplace=True)
    return results_df