"""
Contains any fixture relevant for frozen tests.

At the moment frozen tests are semi-experimental, when they are finalized,
these can be potentially moved to artifacts/datasets modules.
"""

import pickle
from pytest import fixture
from credoai.artifacts import ClassificationModel, TabularData

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
