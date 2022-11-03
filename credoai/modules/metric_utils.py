import textwrap
from collections import defaultdict

import numpy as np
from credoai.modules.metrics import (
    ALL_METRICS,
    METRIC_CATEGORIES,
    METRIC_NAMES,
    MODEL_METRIC_CATEGORIES,
)
from scipy.stats import norm
from sklearn.utils import resample


def list_metrics(verbose=True):
    metrics = defaultdict(set)
    for metric in ALL_METRICS:
        if metric.metric_category in MODEL_METRIC_CATEGORIES:
            metrics[metric.metric_category] |= metric.equivalent_names
    if verbose:
        for key, val in metrics.items():
            metric_str = textwrap.fill(
                ", ".join(sorted(list(val))),
                width=50,
                initial_indent="\t",
                subsequent_indent="\t",
            )
            print(key)
            print(metric_str)
            print("")
    return metrics
