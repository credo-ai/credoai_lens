import pickle

from pytest import fixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

from credoai.artifacts import ClassificationModel, RegressionModel, TabularData
from credoai.datasets import fetch_creditdefault, fetch_testdata

import pytest
from pandas import DataFrame

from credoai.artifacts import ComparisonData, TabularData
from credoai.artifacts.model.comparison_model import DummyComparisonModel

from credoai.lens import Lens

from connect.governance import Governance

################################################
############ Lens init #########################
################################################


@pytest.fixture(scope="function")
def temp_file(tmp_path):
    d = tmp_path / "test.json"
    d.touch()
    return d


@pytest.fixture(scope="function")
def init_lens_classification(
    classification_model,
    classification_assessment_data,
    classification_train_data,
    temp_file,
):
    gov = Governance()
    my_pipeline = Lens(
        model=classification_model,
        assessment_data=classification_assessment_data,
        training_data=classification_train_data,
        governance=gov,
    )
    return my_pipeline, temp_file, gov


@pytest.fixture(scope="function")
def init_lens_credit(
    credit_classification_model,
    credit_assessment_data,
    credit_training_data,
    temp_file,
):
    gov = Governance()
    my_pipeline = Lens(
        model=credit_classification_model,
        assessment_data=credit_assessment_data,
        training_data=credit_training_data,
        governance=gov,
    )
    return my_pipeline, temp_file, gov


@pytest.fixture(scope="function")
def init_lens_fairness(
    ranking_fairness_assessment_data,
    temp_file,
):
    expected_results = DataFrame(
        {
            "value": [0.11, 0.20, 0.90, 0.67, 0.98, 0.65, 0.59],
            "type": [
                "skew_parity_difference",
                "ndkl",
                "demographic_parity_ratio",
                "balance_ratio",
                "score_parity_ratio",
                "score_balance_ratio",
                "relevance_parity_ratio",
            ],
            "subtype": ["score"] * 7,
        }
    )
    gov = Governance()
    pipeline = Lens(assessment_data=ranking_fairness_assessment_data, governance=gov)

    return pipeline, temp_file, gov, expected_results


@pytest.fixture(scope="function")
def init_lens_identityverification(
    identity_verification_model,
    identity_verification_comparison_data,
    temp_file,
):
    expected_results_perf = DataFrame(
        {
            "value": [0.33, 1.00],
            "type": ["false_match_rate", "false_non_match_rate"],
            "subtype": ["score"] * 2,
        }
    )

    expected_results_fair = DataFrame(
        {
            "gender": ["female", "male", "female", "male"],
            "type": [
                "false_match_rate",
                "false_match_rate",
                "false_non_match_rate",
                "false_non_match_rate",
            ],
            "value": [0, 0, 0, 1],
        }
    )
    expected_results = {"fair": expected_results_fair, "perf": expected_results_perf}

    gov = Governance()
    pipeline = Lens(
        model=identity_verification_model,
        assessment_data=identity_verification_comparison_data,
        governance=gov,
    )

    return pipeline, temp_file, gov, expected_results


################################################
############ Artifacts fixtures ################
################################################

### Datasets definition ########################


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
def ranking_fairness_data():
    df = DataFrame(
        {
            "rankings": [1, 2, 3, 4, 5, 6, 7, 8],
            "scores": [10, 8, 7, 6, 2, 2, 1, 1],
            "sensitive_features": ["f", "f", "m", "m", "f", "m", "f", "f"],
        }
    )
    return df


@fixture(scope="session")
def identity_verification_data():
    source_subject_id = 4 * ["s0"] + 5 * ["s1"] + ["s2"]
    source_subject_data_sample = 4 * ["s00"] + 3 * ["s10"] + 2 * ["s11"] + ["s20"]
    target_subject_id = ["s1"] + ["s1", "s2", "s3"] * 2 + ["s2", "s3", "s3"]
    target_subject_data_sample = (
        ["s10"] + ["s11", "s20", "s30"] * 2 + ["s20"] + ["s30"] * 2
    )
    pairs = DataFrame(
        {
            "source-subject-id": source_subject_id,
            "source-subject-data-sample": source_subject_data_sample,
            "target-subject-id": target_subject_id,
            "target-subject-data-sample": target_subject_data_sample,
        }
    )

    subjects_sensitive_features = DataFrame(
        {
            "subject-id": ["s0", "s1", "s2", "s3"],
            "gender": ["female", "male", "female", "female"],
        }
    )
    return pairs, subjects_sensitive_features


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


################################################
############ Frozen fixtures ###################
################################################

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


@fixture(scope="session")
def config_path_in():
    return  # TODO!!! FILL IN WITH GITHUB SECRET FOR CONFIG FILE (API TOKEN)
    # return ""


@fixture(scope="session")
def assessment_plan_url_in():
    return  # TODO!!! FILL IN WITH GITHUB SECRET FOR ASSESSMENT PLAN (API LINK)
    # return ""
