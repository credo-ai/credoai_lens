from fairlearn.metrics import MetricFrame
from credoai.utils import ValidationError

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
    performance_metrics, prob_metrics, y_pred, y_prob, y_true, sensitive_features
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
        if all(
            # sklearn probability metric functions expect a 1d array
            # predict_proba returns 2d array for binary outcome
            # No current support for multi-output probabilistic metrics
            [
                "BINARY" in prob_metrics[key].metric_category
                for key in prob_metrics.keys()
            ]
        ):
            metric_frames["prob"] = create_metric_frame(
                prob_metrics,
                y_prob[:, 1],
                y_true,
                sensitive_features=sensitive_features,
            )
        else:
            raise ValidationError(
                "Specified non-binary metric with probabilistic model. Multi-output probabilistic metrics not currently supported."
            )
    return metric_frames
