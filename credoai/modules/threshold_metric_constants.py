from sklearn import metrics as sk_metrics
from credoai.modules.credoai_metrics import (
    credo_pr_curve,
    credo_roc_curve,
    credo_det_curve,
)


# MODEL METRICS
BINARY_CLASSIFICATION_CURVE_FUNCTIONS = {
    "roc_curve": credo_roc_curve,
    "precision_recall_curve": credo_pr_curve,
    "det_curve": credo_det_curve,
}

THRESHOLD_PROBABILITY_FUNCTIONS = {"roc_curve", "precision_recall_curve", "det_curve"}

# Included for consistency relative to Metric and metric_constants.py
# Empty because there are no supported equivalent names for threshold-varying metric functions
THRESHOLD_METRIC_EQUIVALENTS = {
    "precision_recall_curve": ["pr_curve"],
    "det_curve": ["detection_error_tradeoff"],
}
