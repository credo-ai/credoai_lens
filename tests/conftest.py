import pickle

import pytest
from connect.governance import Governance
from pandas import DataFrame
from pytest import fixture

from credoai.artifacts import ClassificationModel, TabularData
from credoai.lens import Lens


@pytest.fixture(scope="function")
def temp_file(tmp_path):
    d = tmp_path / "test.json"
    d.touch()
    return d


################################################
############ Lens init #########################
################################################


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


@pytest.fixture(scope="function")
def init_lens_regression(
    regression_model,
    regression_assessment_data,
    regression_train_data,
    temp_file,
):
    gov = Governance()
    my_pipeline = Lens(
        model=regression_model,
        assessment_data=regression_assessment_data,
        training_data=regression_train_data,
        governance=gov,
    )

    return my_pipeline, temp_file, gov


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
