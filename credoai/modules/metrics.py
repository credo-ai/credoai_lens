import re
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from credoai.artifacts.model.constants_model import MODEL_TYPES
from credoai.modules.constants_metrics import *
from credoai.modules.constants_threshold_metrics import *
from credoai.utils.common import ValidationError, humanize_label, wrap_list


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
        category or list of metric categories to constrain search to. The list
        of metric categories is stored in modules.constants_metrics.METRIC_CATEGORIES,
        by default None

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


def find_single_metric(metric_name, metric_category=None):
    """As find_metrics, but enforce expectation that a single metric is returned"""
    matched_metric = find_metrics(metric_name, metric_category)
    if len(matched_metric) == 1:
        matched_metric = matched_metric[0]
    elif len(matched_metric) == 0:
        raise Exception(
            f"Returned no metrics when searching using the provided metric name <{metric_name}> with metric category <{metric_category}>. Expected to find one matching metric."
        )
    else:
        raise Exception(
            f"Returned multiple metrics when searching using the provided metric name <{metric_name}> "
            f"with metric category <{metric_category}>. Expected to find only one matching metric. "
            "Try being more specific with the metric categories passed or using find_metrics if "
            "multiple metrics are desired."
        )
    return matched_metric


def process_metrics(metrics, metric_categories=None):
    """Converts a list of metrics or strings into a standardized form

    The standardized form is a dictionary of str: Metric, where the str represent
    a metric name.

    Parameters
    ----------
    metrics: list
        List of strings or Metrics
    metric_categories: str or list
        One or more metric categories to use to constrain string-based metric search
        (see modules.metrics.find_single_metric). The list
        of metric categories is stored in modules.constants_metrics.METRIC_CATEGORIES

    Returns
    -------
    processed_metrics: dict
        Standardized dictionary of metrics. Generally used to pass to
        evaluators.utils.fairlearn.setup_metric_frames
    fairness_metrics: dict
        Standardized dictionary of fairness metrics. Used for certain evaluator functions
    """
    processed_metrics = {}
    fairness_metrics = {}
    metric_categories_to_include = MODEL_METRIC_CATEGORIES.copy()
    if metric_categories is not None:
        metric_categories_to_include += wrap_list(metric_categories)
    else:
        metric_categories_to_include += MODEL_TYPES

    for metric in metrics:
        if isinstance(metric, str):
            metric_name = metric
            metric = find_single_metric(metric, metric_categories_to_include)
        else:
            metric_name = metric.name
        if not isinstance(metric, Metric):
            raise ValidationError(
                "Specified metric is not of type credoai.metric.Metric"
            )
        if metric.metric_category == "FAIRNESS":
            fairness_metrics[metric_name] = metric
        else:
            processed_metrics[metric_name] = metric
    return processed_metrics, fairness_metrics


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
