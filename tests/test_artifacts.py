"""
Testing protocols for the Lens package. Tested functionalities:
    1. General inits
    2. Individual evaluator runs
"""

from abc import ABC

import pytest
import pandas as pd
import numpy as np

from credoai.artifacts import ComparisonData, ComparisonModel, TabularData
from credoai.artifacts import DummyClassifier

from credoai.datasets import fetch_censusincome, fetch_credit_model

from credoai.lens import Lens

from credoai.utils import ValidationError


def test_base_data_property():
    """
    tests call to Base.data property
    """
    X_test, y_test, _, _ = fetch_credit_model()

    credo_data = TabularData(name="test_data_property", X=X_test, y=y_test)
    pytest.assume(not print(credo_data.data))


def test_tabular_data_array_inputs():
    X_test, y_test, _, _ = fetch_credit_model()

    credo_data = TabularData(
        name="test_numpy_inputs", X=X_test.to_numpy(), y=y_test.to_numpy()
    )
    pytest.assume(not print(credo_data.data))


def test_tabular_data_mismatched_X_y():
    """
    tests TabularData._validate_y() function
    """
    X_test, y_test, _, _ = fetch_credit_model()

    with pytest.raises(Exception) as e_info:
        TabularData(name="test_mismatched_X_y", X=X_test, y=y_test.iloc[:-1])

    pytest.assume(type(e_info.value) == ValidationError)


def test_tabular_sensitive_intersections():
    """
    tests use of sensitive interactions functionality
    """
    X, y, sensitive_features, model = fetch_credit_model()
    sens = pd.concat([sensitive_features, X["MARRIAGE"]], axis=1)
    X = X.drop("MARRIAGE", axis=1)

    credo_data = TabularData(
        "test_sens_feat_interactions",
        X=X,
        y=y,
        sensitive_features=sens,
        sensitive_intersections=True,
    )
    test_intersections = credo_data.sensitive_features

    with pytest.raises(Exception) as e_info:
        pd.testing.assert_series_equal(credo_data.sensitive_features, sens)
    assert type(e_info.value) == AssertionError


def test_dummy_classifier():
    """
    tests all functions of DummyClassifier
    """
    X_test, y_test, sensitive_features, model = fetch_credit_model()
    predictions = model.predict(X_test)
    probs = model.predict_proba(X_test)
    # light wrapping to create the dummy model
    dummy_model = DummyClassifier(
        name="test_dummy", predict_output=predictions, predict_proba_output=probs
    )
    pytest.assume(dummy_model.predict() is not None)
    pytest.assume(dummy_model.predict_proba() is not None)
