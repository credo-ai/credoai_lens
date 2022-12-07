from pytest import fixture
from credoai.artifacts.model.comparison_model import DummyComparisonModel
from credoai.artifacts import (
    TabularData,
    ComparisonData,
    ClassificationModel,
    RegressionModel,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier


### Identity verification data/model artifacts ###


@fixture(scope="session")
def identity_verification_model():
    similarity_scores = [31.5, 16.7, 20.8, 84.4, 12.0, 15.2, 45.8, 23.5, 28.5, 44.5]

    credo_model = DummyComparisonModel(
        name="face-compare", compare_output=similarity_scores
    )
    return credo_model


@fixture(scope="session")
def identity_verification_comparison_data(identity_verification_data):
    pairs, subjects_sensitive_features = identity_verification_data
    credo_data = ComparisonData(
        name="face-data",
        pairs=pairs,
        subjects_sensitive_features=subjects_sensitive_features,
    )
    return credo_data


### Ranking fairness data artifacts ############


@fixture(scope="session")
def ranking_fairness_assessment_data(ranking_fairness_data):
    df = ranking_fairness_data
    data = TabularData(
        name="ranks",
        y=df[["rankings", "scores"]],
        sensitive_features=df[["sensitive_features"]],
    )
    return data


### Classification model/data artifacts ########


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


### Regression model/data artifacts ############


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


### Credit model/data artifacts ################


@fixture(scope="session")
def credit_classification_model(credit_data):
    model = RandomForestClassifier()
    model.fit(credit_data["train"]["X"], credit_data["train"]["y"])
    credo_model = ClassificationModel("credit_default_classifier", model)
    return credo_model


@fixture(scope="session")
def credit_assessment_data(credit_data):
    test = credit_data["test"]
    assessment_data = TabularData(
        name="UCI-credit-default-test",
        X=test["X"],
        y=test["y"],
    )
    return assessment_data


@fixture(scope="session")
def credit_training_data(credit_data):
    train = credit_data["train"]
    training_data = TabularData(
        name="UCI-credit-default-test",
        X=train["X"],
        y=train["y"],
    )
    return training_data


### Multiclass model/data artifacts ############


@fixture(scope="session")
def multiclass_model(multiclass_data):
    model = DecisionTreeClassifier(max_depth=2)
    model.fit(multiclass_data["train"]["X"], multiclass_data["train"]["y"])
    credo_model = ClassificationModel("credit_default_classifier", model)
    return credo_model


@fixture(scope="session")
def multiclass_assessment_data(multiclass_data):
    test = multiclass_data["test"]
    assessment_data = TabularData(
        name="UCI-credit-default-test",
        X=test["X"],
        y=test["y"],
        sensitive_features=test["sens_features"],
    )
    return assessment_data


@fixture(scope="session")
def multiclass_training_data(multiclass_data):
    train = multiclass_data["train"]
    training_data = TabularData(
        name="UCI-credit-default-test",
        X=train["X"],
        y=train["y"],
        sensitive_features=train["sens_features"],
    )
    return training_data
