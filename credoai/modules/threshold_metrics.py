import re
from dataclasses import dataclass
from typing import Any, Callable, List, Optional
from credoai.modules.threshold_metric_constants import *
from credoai.utils.common import ValidationError, humanize_label


@dataclass
class ThresholdMetric:
    """Class to define threshold-varying metrics

    Threshold-varying metrics are assumed to be used exclusively with binary classification
    problems (multi-class metrics not supported).

    credoai_lens assumes that the metric signature corresponds to an scikit-learn function
    which performs the work of evaluating the assessment data and returning the resulting performance metric.

    Threshold-varying metrics differ from Metrics in that performance varies according to the
    selected decision boundary. In general, these will lead to curves (ROC, PR, etc.) rather than
    scalar values.

    Parameters
    ----------
    name : str
        The primary name of the metric
    fun : callable, optional
        The function definition of the metric. If none, the metric cannot be used and is only
        defined for documentation purposes
    takes_prob : bool, optional
        Whether the function takes the decision probabilities
        instead of the predicted class, as for ROC AUC. Similar to `needs_proba` used by
        `sklearn <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html>`_
        by default False
    """

    name: str
    # metric_category: str
    # I think this guy is no longer needed -- all are essentially the same category
    fun: Optional[Callable[[Any], Any]] = None
    takes_prob: Optional[bool] = True
    equivalent_names: Optional[list[str]] = None
    # this is not really used right now -- each metric has only 1 valid name
    # keeping for consistency

    def __post_init__(self):
        if self.equivalent_names is None:
            self.equivalent_names = {self.name}
        else:
            self.equivalent_names = set(self.equivalent_names + [self.name])
        self.humanized_type = humanize_label(self.name)

    def __call__(self, **kwargs):
        self.fun(**kwargs)

    def get_fun_doc(self):
        if self.fun:
            return self.fun.__doc__

    def print_fun_doc(self):
        print(self.get_fun_doc())

    def is_metric(self, metric_name: str):
        metric_name = self.standardize_metric_name(metric_name)
        if self.equivalent_names:
            name_match = metric_name in self.equivalent_names
        return name_match

    def standardize_metric_name(self, metric):
        # standardize
        # lower, remove spaces, replace delimiters with underscores
        standard = "_".join(re.split("[- \s _]", re.sub("\s\s+", " ", metric.lower())))
        return standard


def metrics_from_dict(dict, probability_functions, metric_equivalents):
    # Convert to metric objects
    metrics = {}
    for metric_name, fun in dict.items():
        equivalents = metric_equivalents.get(metric_name, [])  # get equivalent names
        # whether the metric takes probabities instead of predictions
        takes_prob = metric_name in probability_functions
        metric = ThresholdMetric(metric_name, fun, takes_prob, equivalents)
        metrics[metric_name] = metric
    return metrics


def find_metrics(metric_name):
    """Find metric by name

    Parameters
    ----------
    metric_name : str
        metric name to search for

    Returns
    -------
    list
        list of Metrics
    """
    matched_metrics = [i for i in ALL_METRICS if i.is_metric(metric_name)]
    return matched_metrics


THRESHOLD_VARYING_METRICS = metrics_from_dict(
    BINARY_CLASSIFICATION_CURVE_FUNCTIONS,
    PROBABILITY_FUNCTIONS,
    METRIC_EQUIVALENTS,
)


METRIC_NAMES = list(THRESHOLD_VARYING_METRICS.keys())

ALL_METRICS = list(THRESHOLD_VARYING_METRICS.values())
