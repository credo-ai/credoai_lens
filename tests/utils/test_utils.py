import pytest

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
