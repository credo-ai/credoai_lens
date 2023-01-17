import pytest

from credoai.modules.metrics import process_metrics
from credoai.utils import check_subset


@pytest.mark.parametrize(
    "superset,subset,expected",
    [
        ({"a": 1, "b": 2, "c": [3, 4], "d": 5}, {"a": 1}, True),
        ({"a": 1, "b": 2, "c": [3, 4], "d": 5}, {"a": 1, "c": [3]}, True),
        ({"a": 1, "b": 2, "c": [3, 4], "d": 5}, {"a": 1, "c": [3, 5]}, False),
        ({"a": 1, "b": 2, "c": [3, 4], "d": 5}, {"a": 1, "e": 5}, False),
        ({"a": 1, "b": 2, "c": [3, 4], "d": 5}, {}, True),
        ([], {}, False),
        (
            {"a": 1, "b": 2, "c": [3, 4], "d": {"e": [5, 6, 7], "f": 8}},
            {"a": 1, "d": {"e": [5, 6]}},
            True,
        ),
    ],
)
def test_check_subset(superset, subset, expected):
    assert check_subset(subset, superset) == expected


@pytest.mark.parametrize(
    "metrics,metric_categories,process_expected,fairness_expected",
    [
        (
            ["equal_opportunity", "precision_score"],
            "BINARY_CLASSIFICATION",
            ["precision_score"],
            ["equal_opportunity"],
        ),
        (
            ["precision", "average_precision"],
            "BINARY_CLASSIFICATION",
            ["precision", "average_precision"],
            [],
        ),
    ],
)
def test_process_metrics(
    metrics, metric_categories, process_expected, fairness_expected
):
    processed_metrics, fairness_metrics = process_metrics(metrics, metric_categories)
    process_equal = list(processed_metrics.keys()) == process_expected
    fairness_equal = list(fairness_metrics.keys()) == fairness_expected
    assert process_equal and fairness_equal
