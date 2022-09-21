from random import random
from typing import Union
from pytest import fixture
from credoai.data._fetch_testdata import fetch_testdata
from sklearn.model_selection import train_test_split
from credoai.artifacts import TabularData, ClassificationModel
from sklearn.linear_model import LogisticRegression
from pandas import Series, DataFrame


@fixture(scope="session")
def data():
    train_data, test_data = fetch_testdata(False, 1, 1)
    return {"train": train_data, "test": test_data}


@fixture(scope="session")
def credo_model(data):
    model = LogisticRegression(random_state=0)
    model.fit(data["train"]["X"], data["train"]["y"])
    credo_model = ClassificationModel("income_classifier", model)
    return credo_model


@fixture(scope="session")
def assessment_data(data):
    test = data["test"]
    return TabularData(
        name="assessment_data",
        X=test["X"],
        y=test["y"],
        sensitive_features=test["sensitive_features"],
    )


@fixture(scope="session")
def train_data(data):
    train = data["train"]
    return TabularData(
        name="training_data",
        X=train["X"],
        y=train["y"],
        sensitive_features=train["sensitive_features"],
    )
