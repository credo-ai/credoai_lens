import collections
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import requests
import sklearn.utils as skutils
from absl import logging


class NotRunError(Exception):
    pass


class ValidationError(Exception):
    pass


class InstallationError(Exception):
    pass


class IntegrationError(Exception):
    pass


class SupressSettingWithCopyWarning:
    def __enter__(self):
        pd.options.mode.chained_assignment = None

    def __exit__(self, *args):
        pd.options.mode.chained_assignment = "warn"


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def flatten_list(lst):
    return [item for sublist in lst for item in sublist]


def update_dictionary(d, u):
    """Recursively updates a dictionary"""
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dictionary(d.get(k, {}), v)
        elif isinstance(v, list):
            d[k] = v + d.get(k, [])
        else:
            d[k] = v
    return d


def wrap_list(obj):
    if type(obj) == str:
        obj = [obj]
    elif obj is None:
        return None
    try:
        iter(obj)
    except TypeError:
        obj = [obj]
    return obj


def remove_suffix(text, suffix):
    return text[: -len(suffix)] if text.endswith(suffix) and len(suffix) != 0 else text


def humanize_label(s):
    return " ".join(s.split("_")).title()


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def json_dumps(obj):
    """Custom json dumps with encoder"""
    return json.dumps(obj, cls=NumpyEncoder)


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


def is_categorical(series, threshold=0.05):
    """Identifies whether a series is categorical or not

    Logic: If there are relatively few unique values for a feature, the feature is likely categorical.
    The results are estimates and are not guaranteed to be correct.

    Parameters
    ----------
    series : pd.Series
        Series to evaluate
    threshold : float
        The threshold (number of the unique values over the total number of values)


    Returns
    -------
    bool
        Whether the series is categorical or not
    """

    if series.dtype.name in ["category", "object"]:
        return True
    # float columns are assumed not-categorical
    elif len(series.unique()) / len(series) < threshold:
        return True
    else:
        return False
