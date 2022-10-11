from credoai.utils import ValidationError

from fairlearn.metrics import MetricFrame

########### General functions shared across evaluators ###########


def create_metric_frame(metrics, y_pred, y_true, sensitive_features):
    """Creates metric frame from dictionary of key:Metric"""
    metrics = {name: metric.fun for name, metric in metrics.items()}
    return MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )


def setup_metric_frames(
    performance_metrics,
    prob_metrics,
    thresh_metrics,
    y_pred,
    y_prob,
    y_true,
    sensitive_features,
):
    metric_frames = {}
    if y_pred is not None and performance_metrics:
        metric_frames["pred"] = create_metric_frame(
            performance_metrics,
            y_pred,
            y_true,
            sensitive_features=sensitive_features,
        )

    if y_prob is not None and prob_metrics:
        metric_frames["prob"] = create_metric_frame(
            prob_metrics,
            y_prob,
            y_true,
            sensitive_features=sensitive_features,
        )

    if y_prob is not None and thresh_metrics:
        metric_frames["thresh"] = create_metric_frame(
            thresh_metrics,
            y_prob,
            y_true,
            sensitive_features=sensitive_features,
        )
    return metric_frames
