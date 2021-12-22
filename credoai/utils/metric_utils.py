import re
from credoai.utils.common import remove_suffix
from credoai.utils.metric_constants import (
    BINARY_CLASSIFICATION_METRICS, FAIRNESS_METRICS, 
    METRIC_EQUIVALENTS, STANDARD_CONVERSIONS
)

def standardize_metric_name(metric):
    # standardize
    # lower, remove spaces, replace delimiters with underscores
    standard =  '_'.join(re.split('[- \s _]', 
                         re.sub('\s\s+', ' ', metric.lower())))
    standard = remove_suffix(remove_suffix(standard, '_difference'), '_parity')
    return STANDARD_CONVERSIONS.get(standard, standard)

def list_metrics():
    metrics = {'performance': list(BINARY_CLASSIFICATION_METRICS.keys()) ,
               'fairness': list(FAIRNESS_METRICS.keys())}
    for key in metrics.keys():
        names = metrics[key]
        for val in names:
            equivalents = METRIC_EQUIVALENTS.get(val)
            if equivalents is not None:
                metrics[key] += equivalents
    metrics = {k: sorted(v) for k, v in metrics.items()}
    return metrics