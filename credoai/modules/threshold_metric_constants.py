from sklearn import metrics as sk_metrics

# Not needed for now...everything is binary classification
# Only reason to extend is if we supported multi-class or something...
# METRIC_CATEGORIES = [
#     "BINARY_CLASSIFICATION",
# ]

# MODEL METRICS
BINARY_CLASSIFICATION_CURVE_FUNCTIONS = {
    "roc_curve": sk_metrics.roc_curve,
    "precision_recall_curve": sk_metrics.precision_recall_curve,
    "det_curve": sk_metrics.det_curve,
}

PROBABILITY_FUNCTIONS = {
    "roc_curve",
    "precision_recall_curve",
    "det_curve",
}

# Included for consistency relative to Metric and metric_constants.py
# Empty because there are no supported equivalent names for threshold-varying metric functions
METRIC_EQUIVALENTS = {}
