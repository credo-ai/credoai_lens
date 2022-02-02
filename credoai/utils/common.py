import json
import hashlib
import numpy as np
import pandas as pd
import os
import requests
from pathlib import Path
from typing import Dict, Any

class NotRunError(Exception):
    pass

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
    
def dict_hash(dictionary: Dict[str, Any]) -> str:
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()

def to_array(lst):
    """
    Converts list-like object to array
    Parameters
    ----------
    lst :  (List, pandas.Series, numpy.ndarray)
        The list-like to be converted
    """
    if type(lst) == pd.Series:
        return lst.values
    elif type(lst) == list:
        return np.array(lst)
    elif type(lst) == np.ndarray:
        return lst
    else:
        raise TypeError