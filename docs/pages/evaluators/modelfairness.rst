
Model fairness
==============


Model Fairness evaluator for Credo AI.

This evaluator calculates performance metrics disaggregated by a sensitive feature, as
well as evaluating the parity of those metrics.

Handles any metric that can be calculated on a set of ground truth labels and predictions,
e.g., binary classification, multiclass classification, regression.


Parameters
----------
metrics : List-like
    list of metric names as string or list of Metrics (credoai.metrics.Metric).
    Metric strings should in list returned by credoai.modules.list_metrics.
    Note for performance parity metrics like
    "false negative rate parity" just list "false negative rate". Parity metrics
    are calculated automatically if the performance metric is supplied
sensitive_features :  pandas.DataFrame
    The segmentation feature(s) which should be used to create subgroups to analyze.
y_true : (List, pandas.Series, numpy.ndarray)
    The ground-truth labels (for classification) or target values (for regression).
y_pred : (List, pandas.Series, numpy.ndarray)
    The predicted labels for classification
y_prob : (List, pandas.Series, numpy.ndarray), optional
    The unthresholded predictions, confidence values or probabilities.
method : str, optional
    How to compute the differences: "between_groups" or "to_overall".
    See fairlearn.metrics.MetricFrame.difference
    for details, by default 'between_groups'
