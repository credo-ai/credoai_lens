from credoai.utils import ValidationError
from credoai.utils import global_logger

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

    if prob_metrics:
        if y_prob is not None:
            metric_frames["prob"] = create_metric_frame(
                prob_metrics,
                y_prob,
                y_true,
                sensitive_features=sensitive_features,
            )
        else:
            global_logger.warn(f"Metrics ({list(prob_metrics.keys())}) requested, but no y_prob available")

    if thresh_metrics:
        if y_prob is not None:
            metric_frames["thresh"] = create_metric_frame(
                thresh_metrics,
                y_prob,
                y_true,
                sensitive_features=sensitive_features,
            )
        else:
            global_logger.warn(f"Metrics ({list(thresh_metrics.keys())}) requested, but no y_prob available")
    return metric_frames
