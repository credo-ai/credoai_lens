from credoai.artifacts import ClassificationModel, RegressionModel, TabularData
from credoai.datasets import fetch_testdata
from pytest import fixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from credoai.datasets import fetch_creditdefault
import pickle


@fixture(scope="session")
def binary_data():
    train_data, test_data = fetch_testdata(False, 1, 1)
    return {"train": train_data, "test": test_data}


@fixture(scope="session")
def continuous_data():
    train_data, test_data = fetch_testdata(False, 1, 1, "continuous")
    return {"train": train_data, "test": test_data}


@fixture(scope="session")
def credit_data():
    data = fetch_creditdefault()
    X = data["data"].iloc[0:100]
    X = X.drop(columns=["SEX"])
    y = data["target"].iloc[0:100].astype(int)
    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(X, y, random_state=42)
    train_data = {"X": X_train, "y": y_train}
    test_data = {"X": X_test, "y": y_test}
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


# with open(
#         "tests/frozen_ml_tests/frozen_results/binary/pipeline_info.pkl", "rb"
#     ) as f:
#         pipeline_info = pickle.load(f)

#     metrics = pipeline_info["metrics"]
#     assessments = pipeline_info["assessments"]

#     pipeline = []
#     for assessment in assessments:
#         pipeline.append(
#             (
#                 string2evaluator(assessment)(metrics),
#                 assessment + " Assessment",
#             )
#         )


@fixture(scope="session")
def frozen_classifier():
    # Load frozen classifier and wrap as a Credo Model
    with open("tests/frozen_ml_tests/frozen_models/loan_binary_clf.pkl", "rb") as f:
        clf = pickle.load(f)

    return ClassificationModel("binary_clf", clf)


@fixture(scope="session")
def frozen_validation_data():
    # Load frozen validation data and wrap as Credo Data
    with open(
        "tests/frozen_ml_tests/frozen_data/binary/loan_validation.pkl", "rb"
    ) as f:
        val_data = pickle.load(f)

    return TabularData(
        name=val_data["name"],
        X=val_data["val_features"],
        y=val_data["val_labels"],
        sensitive_features=val_data["sensitive_features"],
    )


# @fixture(scope="session")
# def frozen_training_data():
#     # Load frozen training data and wrap as Credo Data
#     with open("tests/frozen_ml_tests/frozen_data/binary/loan_train.pkl", "rb") as f:
#         train_data = pickle.load(f)

#     return TabularData(
#         name=train_data["name"],
#         X=train_data["train_features"],
#         y=train_data["train_labels"],
#         sensitive_features=train_data["sensitive_features"],
#     )
