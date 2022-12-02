
Model Fairness
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
method : str, optional
    How to compute the differences: "between_groups" or "to_overall".
    See fairlearn.metrics.MetricFrame.difference
    for details, by default 'between_groups'
