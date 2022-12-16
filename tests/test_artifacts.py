"""
Testing for specific credoai.artifacts.
"""
import pytest

from sklearn.linear_model import LogisticRegression

from credoai.artifacts import TabularData
from credoai.artifacts import DummyClassifier
from credoai.utils import ValidationError


def test_base_data_property(credit_data):
    """
    tests call to Base.data property
    """
    X_test = credit_data["test"]["X"]
    y_test = credit_data["test"]["y"]

    credo_data = TabularData(name="test_data_property", X=X_test, y=y_test)
    pytest.assume(credo_data.data is not None)


def test_tabular_data_array_inputs(credit_data):
    X_test = credit_data["test"]["X"]
    y_test = credit_data["test"]["y"]

    credo_data = TabularData(
        name="test_numpy_inputs", X=X_test.to_numpy(), y=y_test.to_numpy()
    )
    pytest.assume(credo_data.data is not None)


def test_tabular_data_mismatched_X_y(credit_data):
    """
    tests TabularData._validate_y() function
    """
    X_test = credit_data["test"]["X"]
    y_test = credit_data["test"]["y"]

    with pytest.raises(Exception) as e_info:
        TabularData(name="test_mismatched_X_y", X=X_test, y=y_test.iloc[:-1])

    pytest.assume(type(e_info.value) == ValidationError)


def test_tabular_sensitive_intersections(binary_data):
    """
    tests use of sensitive interactions functionality
    """
    X = binary_data["test"]["X"]
    y = binary_data["test"]["y"]
    sensitive_features = binary_data["test"]["sensitive_features"]

    credo_data = TabularData(
        "test_sens_feat_interactions",
        X=X,
        y=y,
        sensitive_features=sensitive_features,
        sensitive_intersections=True,
    )
    assert "race_gender" in credo_data.sensitive_features.columns


def test_dummy_classifier(binary_data):
    """
    tests all functions of DummyClassifier
    """
    model = LogisticRegression(random_state=0)
    model.fit(binary_data["train"]["X"], binary_data["train"]["y"])
    predictions = model.predict(binary_data["test"]["X"])
    probs = model.predict_proba(binary_data["test"]["X"])
    # light wrapping to create the dummy model
    dummy_model = DummyClassifier(
        name="test_dummy", predict_output=predictions, predict_proba_output=probs
    )
    pytest.assume(dummy_model.predict() is not None)
    pytest.assume(dummy_model.predict_proba() is not None)
