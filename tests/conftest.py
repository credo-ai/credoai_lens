from credoai.artifacts import ClassificationModel, RegressionModel, TabularData
from credoai.datasets import fetch_testdata
from pandas import DataFrame, Series
from pytest import fixture
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split


@fixture(scope="session")
def binary_data():
    train_data, test_data = fetch_testdata(False, 1, 1)
    return {"train": train_data, "test": test_data}


@fixture(scope="session")
def continuous_data():
    train_data, test_data = fetch_testdata(False, 1, 1, "continuous")
    return {"train": train_data, "test": test_data}


@fixture(scope="session")
def classification_model(binary_data):
    model = LogisticRegression(random_state=0)
    model.fit(binary_data["train"]["X"], binary_data["train"]["y"])
    credo_model = ClassificationModel("hire_classifier", model)
    return credo_model


@fixture(scope="session")
def classification_assessment_data(binary_data):
    test = binary_data["test"]
    return TabularData(
        name="assessment_data",
        X=test["X"],
        y=test["y"],
        sensitive_features=test["sensitive_features"],
    )


@fixture(scope="session")
def classification_train_data(binary_data):
    train = binary_data["train"]
    return TabularData(
        name="training_data",
        X=train["X"],
        y=train["y"],
        sensitive_features=train["sensitive_features"],
    )


@fixture(scope="session")
def regression_model(continuous_data):
    model = LinearRegression()
    model.fit(continuous_data["train"]["X"], continuous_data["train"]["y"])
    credo_model = RegressionModel("skill_regressor", model)
    return credo_model


@fixture(scope="session")
def regression_assessment_data(continuous_data):
    test = continuous_data["test"]
    return TabularData(
        name="assessment_data",
        X=test["X"],
        y=test["y"],
        sensitive_features=test["sensitive_features"],
    )


@fixture(scope="session")
def regression_train_data(continuous_data):
    train = continuous_data["train"]
    return TabularData(
        name="assessment_data",
        X=train["X"],
        y=train["y"],
        sensitive_features=train["sensitive_features"],
    )
