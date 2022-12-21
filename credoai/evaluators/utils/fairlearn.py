from fairlearn.metrics import MetricFrame

from credoai.modules.constants_metrics import THRESHOLD_METRIC_CATEGORIES
from credoai.utils import global_logger, wrap_list

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


def filter_processed_metrics(
    processed_metrics, metric_categories=None, Xmetric_categories=None, takes_prob=None
):
    """
    Filters processed metrics

    If any argument is None, it will be ignored for filtering

    Parameters
    ----------
    metric_categories: dict
        Dictionary of metrics (dict of str: Metric)
    metric_categories: str or list
        Positive metric categories to filter metrics. Each metric must have a metric_category
        within this list. The list of metric categories is stored in modules.constants_metrics.METRIC_CATEGORIES
    Xmetric_categories: str or list
        Negative metric categories to filter metrics. Each metric must have a metric_category
        NOT within this list. The list of metric categories is stored in modules.constants_metrics.METRIC_CATEGORIES
    takes_prob: bool
        Whether the metric takes probabilities
    """
    metric_categories = wrap_list(metric_categories)
    return {
        name: metric
        for name, metric in processed_metrics.items()
        if (metric_categories is None or metric.metric_category in metric_categories)
        and (
            Xmetric_categories is None
            or metric.metric_category not in Xmetric_categories
        )
        and (takes_prob is None or metric.takes_prob == takes_prob)
    }


def setup_metric_frames(
    processed_metrics,
    y_pred,
    y_prob,
    y_true,
    sensitive_features,
):
    metric_frames = {}

    # tuple structure: (metric frame name, y_input, dictionary of metrics)
    metric_frame_tuples = [
        ("pred", y_pred, filter_processed_metrics(processed_metrics, takes_prob=False)),
        (
            "prob",
            y_prob,
            filter_processed_metrics(
                processed_metrics,
                Xmetric_categories=THRESHOLD_METRIC_CATEGORIES,
                takes_prob=True,
            ),
        ),
        (
            "thresh",
            y_prob,
            filter_processed_metrics(
                processed_metrics,
                metric_categories=THRESHOLD_METRIC_CATEGORIES,
                takes_prob=True,
            ),
        ),
    ]

    for name, y, metrics in metric_frame_tuples:
        if metrics:
            if y is not None:
                metric_frames[name] = create_metric_frame(
                    metrics, y, y_true, sensitive_features
                )
            else:
                global_logger.warn(
                    f"Metrics ({list(metrics.keys())}) requested for {name} metric frame, but no appropriate y available"
                )

    return metric_frames
