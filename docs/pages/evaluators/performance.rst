
Performance
===========


Performance evaluator for Credo AI.

This evaluator calculates overall performance metrics.
Handles any metric that can be calculated on a set of ground truth labels and predictions,
e.g., binary classification, multi class classification, regression.

This module takes in a set of metrics and provides functionality to:

- calculate the metrics
- create disaggregated metrics

Parameters
----------
metrics : List-like
    list of metric names as strings or list of Metric objects (credoai.modules.metrics.Metric).
    Metric strings should in list returned by credoai.modules.metric_utils.list_metrics().
    Note for performance parity metrics like
    "false negative rate parity" just list "false negative rate". Parity metrics
    are calculated automatically if the performance metric is supplied
y_true : (List, pandas.Series, numpy.ndarray)
    The ground-truth labels (for classification) or target values (for regression).
y_pred : (List, pandas.Series, numpy.ndarray)
    The predicted labels for classification
y_prob : (List, pandas.Series, numpy.ndarray), optional
    The unthresholded predictions, confidence values or probabilities.
