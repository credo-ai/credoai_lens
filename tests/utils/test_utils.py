from credoai.utils import check_subset


def test_check_subset():
    superset = {"a": 1, "b": 2, "c": [3, 4], "d": 5}
    subset = {"a": 1}
    assert check_subset(subset, superset) == True

    subset = {"a": 1, "c": [3]}
    assert check_subset(subset, superset) == True

    subset = {"a": 1, "c": [3, 5]}
    assert check_subset(subset, superset) == False

    subset = {"a": 1, "e": 5}
    assert check_subset(subset, superset) == False

    subset = {}
    assert check_subset(subset, superset) == True

    assert check_subset({}, []) == False

    superset = {"a": 1, "b": 2, "c": [3, 4], "d": {"e": [5, 6, 7], "f": 8}}
    subset = {"a": 1, "d": {"e": [5, 6]}}
    assert check_subset(subset, superset) == True
