"""Shared functions across the evidence modules"""

from credoai.modules.threshold_metric_constants import THRESHOLD_PROBABILITY_FUNCTIONS
import pandas as pd


def tuple_metric_to_DataFrame(metric_as_tuple, sensitive_feat=None, sensitive_val=None):
    metric_dict = {
        THRESHOLD_PROBABILITY_FUNCTIONS[metric_as_tuple.threshold_metric][
            i
        ]: metric_as_tuple.value[i]
        for i in range(len(metric_as_tuple.value))
    }
    metric_as_df = pd.DataFrame.from_dict(metric_dict)
    metric_as_df.name = metric_as_tuple.threshold_metric
    if sensitive_feat:
        metric_as_df["sensitive_feat"] = sensitive_val
    return metric_as_df
