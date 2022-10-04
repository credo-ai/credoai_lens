from sklearn import metrics as sk_metrics
from credoai.modules.credoai_metrics import (
    credo_pr_curve,
    credo_roc_curve,
    credo_det_curve,
)

"""
Current outputting functionality in evaluator modules (e.g. performance.py) relies on
the assumption that threshold metric functions return DataFrames, with columns labeled.

Other return types are possible, in principle. These may require further wrangling on the
evaluator side before converting to Evidence to ensure that the underlying data structure
can easily be read by the Credo AI Platform.
"""

# MODEL METRICS
THRESHOLD_PROBABILITY_FUNCTIONS = {
    "roc_curve": credo_roc_curve,
    "precision_recall_curve": credo_pr_curve,
    "det_curve": credo_det_curve,
}

# Included for consistency relative to Metric and metric_constants.py
# Empty because there are no supported equivalent names for threshold-varying metric functions
THRESHOLD_METRIC_EQUIVALENTS = {
    "precision_recall_curve": ["pr_curve"],
    "det_curve": ["detection_error_tradeoff"],
}
