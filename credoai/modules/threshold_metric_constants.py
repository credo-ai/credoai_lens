from sklearn import metrics as sk_metrics


# MODEL METRICS
BINARY_CLASSIFICATION_CURVE_FUNCTIONS = {
    "roc_curve": sk_metrics.roc_curve,
    "precision_recall_curve": sk_metrics.precision_recall_curve,
    "det_curve": sk_metrics.det_curve,
}

THRESHOLD_PROBABILITY_FUNCTIONS = {
    "roc_curve": ["fpr", "tpr", "thresholds"],
    "precision_recall_curve": ["precision", "recall", "thresholds"],
    "det_curve": ["fpr", "fnr", "thresholds"],
}

# Included for consistency relative to Metric and metric_constants.py
# Empty because there are no supported equivalent names for threshold-varying metric functions
THRESHOLD_METRIC_EQUIVALENTS = {
    "precision_recall_curve": ["pr_curve"],
    "det_curve": ["detection_error_tradeoff"],
}
