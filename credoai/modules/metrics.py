import re
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from credoai.modules.constants_metrics import *
from credoai.modules.constants_threshold_metrics import *
from credoai.utils.common import ValidationError, humanize_label


@dataclass
class Metric:
    """Class to define metrics

    Metric categories determine what kind of use the metric is designed for. Credo AI assumes
    that metric signatures either correspond with scikit-learn or fairlearn method signatures,
    in the case of binary/multiclass classification, regression, clustering and fairness metrics.

    Dataset metrics are used as documentation placeholders and define no function.
    See DATASET_METRICS for examples. CUSTOM metrics have no expectations and will
    not be automatically used by Lens modules.

    Metric Categories:

    * {BINARY|MULTICLASS}_CLASSIFICATION: metrics like `scikit-learn's classification metrics <https://scikit-learn.org/stable/modules/model_evaluation.html>`_
    * REGRESSION: metrics like `scikit-learn's regression metrics <https://scikit-learn.org/stable/modules/model_evaluation.html>`_
    * CLUSTERING: metrics like `scikit-learn's clustering metrics <https://scikit-learn.org/stable/modules/model_evaluation.html>`_
    * FAIRNESS: metrics like `fairlearn's equalized odds metric <https://fairlearn.org/v0.5.0/api_reference/fairlearn.metrics.html
    * DATASET: metrics intended
    * CUSTOM: No expectations for fun

    Parameters
    ----------
    name : str
        The primary name of the metric
    metric_category : str
        defined to be one of the METRIC_CATEGORIES, above
    fun : callable, optional
        The function definition of the metric. If none, the metric cannot be used and is only
        defined for documentation purposes
    takes_prob : bool, optional
        Whether the function takes the decision probabilities
        instead of the predicted class, as for ROC AUC. Similar to `needs_proba` used by
        `sklearn <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html>`_
        by default False
    equivalent_names : list
        list of other names for metric
    """

    name: str
    metric_category: str
    fun: Optional[Callable[[Any], Any]] = None
    takes_prob: Optional[bool] = False
    equivalent_names: Optional[List[str]] = None

    def __post_init__(self):
        if self.equivalent_names is None:
            self.equivalent_names = {self.name}
        else:
            self.equivalent_names = set(self.equivalent_names + [self.name])
        self.metric_category = self.metric_category.upper()
        if self.metric_category not in METRIC_CATEGORIES:
            raise ValidationError(f"metric type ({self.metric_category}) isn't valid")
        self.humanized_type = humanize_label(self.name)

    def __call__(self, **kwargs):
        self.fun(**kwargs)

    def get_fun_doc(self):
        if self.fun:
            return self.fun.__doc__

    def print_fun_doc(self):
        print(self.get_fun_doc())

    def is_metric(
        self, metric_name: str, metric_categories: Optional[List[str]] = None
    ):
        metric_name = self.standardize_metric_name(metric_name)
        if self.equivalent_names:
            name_match = metric_name in self.equivalent_names
        if metric_categories is not None:
            return name_match and self.metric_category in metric_categories
        return name_match

    def standardize_metric_name(self, metric):
        # standardize
        # lower, remove spaces, replace delimiters with underscores
        standard = "_".join(re.split("[- \s _]", re.sub("\s\s+", " ", metric.lower())))
        return standard


def metrics_from_dict(dict, metric_category, probability_functions, metric_equivalents):
    # Convert to metric objects
    metrics = {}
    for metric_name, fun in dict.items():
        equivalents = metric_equivalents.get(metric_name, [])  # get equivalent names
        # whether the metric takes probabities instead of predictions
        takes_prob = metric_name in probability_functions
        metric = Metric(metric_name, metric_category, fun, takes_prob, equivalents)
        metrics[metric_name] = metric
    return metrics


def find_metrics(metric_name, metric_category=None):
    """Find metric by name and metric category

    Parameters
    ----------
    metric_name : str
        metric name to search for
    metric_category : str or list, optional
        category or list of categories to constrain search to, by default None

    Returns
    -------
    list
        list of Metrics
    """
    if isinstance(metric_category, str):
        metric_category = [metric_category]
    matched_metrics = [
        i for i in ALL_METRICS if i.is_metric(metric_name, metric_category)
    ]
    return matched_metrics


# Convert To List of Metrics
BINARY_CLASSIFICATION_METRICS = metrics_from_dict(
    BINARY_CLASSIFICATION_FUNCTIONS,
    "binary_classification",
    PROBABILITY_FUNCTIONS,
    METRIC_EQUIVALENTS,
)

MULTICLASS_CLASSIFICATION_METRICS = metrics_from_dict(
    MULTICLASS_CLASSIFICATION_FUNCTIONS,
    "MULTICLASS_CLASSIFICATION",
    PROBABILITY_FUNCTIONS,
    METRIC_EQUIVALENTS,
)

THRESHOLD_VARYING_METRICS = metrics_from_dict(
    THRESHOLD_PROBABILITY_FUNCTIONS,
    "BINARY_CLASSIFICATION_THRESHOLD",
    THRESHOLD_PROBABILITY_FUNCTIONS,
    THRESHOLD_METRIC_EQUIVALENTS,
)

REGRESSION_METRICS = metrics_from_dict(
    REGRESSION_FUNCTIONS, "REGRESSION", PROBABILITY_FUNCTIONS, METRIC_EQUIVALENTS
)

FAIRNESS_METRICS = metrics_from_dict(
    FAIRNESS_FUNCTIONS, "FAIRNESS", PROBABILITY_FUNCTIONS, METRIC_EQUIVALENTS
)

DATASET_METRICS = {m: Metric(m, "DATASET", None, False) for m in DATASET_METRIC_TYPES}

PRIVACY_METRICS = {m: Metric(m, "PRIVACY", None, False) for m in PRIVACY_METRIC_TYPES}

SECURITY_METRICS = {
    m: Metric(m, "SECURITY", None, False) for m in SECURITY_METRIC_TYPES
}


METRIC_NAMES = (
    list(BINARY_CLASSIFICATION_METRICS.keys())
    + list(THRESHOLD_VARYING_METRICS.keys())
    + list(FAIRNESS_METRICS.keys())
    + list(DATASET_METRICS.keys())
    + list(PRIVACY_METRICS.keys())
    + list(SECURITY_METRICS.keys())
    + list(REGRESSION_METRICS.keys())
)

ALL_METRICS = (
    list(BINARY_CLASSIFICATION_METRICS.values())
    + list(MULTICLASS_CLASSIFICATION_METRICS.values())
    + list(THRESHOLD_VARYING_METRICS.values())
    + list(FAIRNESS_METRICS.values())
    + list(DATASET_METRICS.values())
    + list(PRIVACY_METRICS.values())
    + list(SECURITY_METRICS.values())
    + list(REGRESSION_METRICS.values())
)
