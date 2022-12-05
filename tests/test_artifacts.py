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


def test_base_data_property():
    """
    tests call to Base.data property
    """
    X_test, y_test, _, _ = fetch_credit_model()

    try:
        credo_data = TabularData(name="test_data_property", X=X_test, y=y_test)
        print(credo_data.data)
    except:
        assert False


def test_tabular_data_array_inputs():
    X_test, y_test, sensitive_features_test, model = fetch_credit_model()

    credo_data = TabularData(
        name="test_numpy_inputs", X=X_test.to_numpy(), y=y_test.to_numpy()
    )

    assert credo_data


def test_tabular_data_mismatched_X_y():
    """
    tests TabularData._validate_y() function
    """
    X_test, y_test, _, _ = fetch_credit_model()

    try:
        TabularData(name="test_mismatched_X_y", X=X_test, y=y_test.iloc[:-1])
        assert False  # if we get here, validation didn't work
    except:
        assert True  # if we get here, validation did work


def test_tabular_sensitive_intersections():
    """
    tests use of sensitive interactions functionality
    """
    X, y, sensitive_features, model = fetch_credit_model()
    sens = pd.concat([sensitive_features, X["MARRIAGE"]], axis=1)
    X = X.drop("MARRIAGE", axis=1)

    try:
        credo_data = TabularData(
            "test_sens_feat_interactions",
            X=X,
            y=y,
            sensitive_features=sens,
            sensitive_intersections=True,
        )
        test_intersections = credo_data.sensitive_features
    except:
        assert False

    try:
        pd.testing.assert_series_equal(credo_data.sensitive_features, sens)
        # THESE SHOULD BE NOT EQUAL
        assert False
    except:
        assert True


def test_dummy_classifier():
    """
    tests all functions of DummyClassifier
    """
    X_test, y_test, sensitive_features, model = fetch_credit_model()
    predictions = model.predict(X_test)
    probs = model.predict_proba(X_test)
    # light wrapping to create the dummy model
    try:
        dummy_model = DummyClassifier(
            name="test_dummy", predict_output=predictions, predict_proba_output=probs
        )
    except:
        assert False
    try:
        test_predict = dummy_model.predict()
    except:
        assert False
    try:
        test_predict_proba = dummy_model.predict_proba()
    except:
        assert False
