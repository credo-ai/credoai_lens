"""Constants for threshold metrics

Define relationships between metric names (strings) and
threshold metric functions,
as well as alternate names for each metric name
"""

from credoai.modules.metrics_credoai import (
    credo_det_curve,
    credo_pr_curve,
    credo_roc_curve,
    credo_gain_chart,
)

"""
Current outputting functionality in evaluators (e.g. Performance) relies on
the assumption that threshold metric functions return DataFrames, with columns labeled.

Other return types are possible, in principle. These may require further wrangling on the
evaluator side before converting to Evidence to ensure that the underlying data structure
can easily be read by the Credo AI Platform.
"""

# MODEL METRICS
THRESHOLD_PROBABILITY_FUNCTIONS = {
    "det_curve": credo_det_curve,
    "gain_chart": credo_gain_chart,
    "precision_recall_curve": credo_pr_curve,
    "roc_curve": credo_roc_curve,
}

# Included for consistency relative to Metric and constants_metrics.py
# Empty because there are no supported equivalent names for threshold-varying metric functions
THRESHOLD_METRIC_EQUIVALENTS = {
    "precision_recall_curve": ["pr_curve"],
    "det_curve": ["detection_error_tradeoff"],
}
