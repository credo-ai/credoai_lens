from collections import defaultdict
from credoai.metrics.metrics import (
    ALL_METRICS, METRIC_NAMES, METRIC_CATEGORIES
)

def list_metrics():
    metrics = defaultdict(set)
    for metric in ALL_METRICS:
        metrics[metric.metric_category] |= metric.equivalent_names
    return metrics