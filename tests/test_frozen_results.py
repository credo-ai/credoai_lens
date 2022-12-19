"""
EXPERIMENTAL! Testing results of evaluator runs against
pre-saved results.

Currently skipped.
"""
import os
import pickle
from abc import ABC

import pytest
from pandas import testing

from credoai.artifacts import ClassificationModel, TabularData
from credoai.evaluators import (
    DataEquity,
    DataFairness,
    DataProfiler,
    ModelEquity,
    ModelFairness,
    Performance,
    Privacy,
    Security,
    evaluator,
)
from credoai.evaluators.utils.utils import name2evaluator
from credoai.lens import Lens

SUPORTED_EVALUATORS = ["Performance", "ModelFairness"]
FROZEN_METRICS = ["false_negative_rate", "average_precision_score"]
TEST_EVALUATOR_IDS = ["performance", "model_fairness"]

"""
Frozen Results

These tests load frozen validation data, model, pipeline info, and results from past run of Lens
Runs Lens (in current form) with frozen model, data, and pipeline info and compares current 
results to the frozen results.

Tests are not designed to reveal fine-grain issues. Targeted at 'broad strokes' evaluation
of possible breaking changes.

Frozen artifacts (including results) generated in credoai_lens/tests/frozen_ml_tests/generation_notebooks

Tests of this type require a substantial degree of hard-coding. We make assumptions about:
    Locations of pickle files for data, models, etc.
    Evaluator names; fixed in the SRING2EVALUATOR dictionary at the top of this file
    Structure of frozen artifacts
        e.g. If, down the road, we change how Lens wraps results from a dictionary to some other structure,
        then this test will break. That is a feature of this test type.

Devising a viable way to generalize/standardize this test type (see, for instance, the parametrization
    decorators utilized by other tests in this file) remains an open problem. It is not necessarily
    clear, however, that such standardization/generalization is desirable for a frozen results test.
"""


@pytest.fixture(scope="class")
def init_lens(
    frozen_classifier,
    frozen_validation_data,
    # frozen_training_data,
    request,
):
    my_pipeline = Lens(
        model=frozen_classifier,
        assessment_data=frozen_validation_data,
        # training_data=frozen_training_data,
    )
    request.cls.pipeline = my_pipeline


@pytest.mark.usefixtures("init_lens")
class Base_Frozen_Test(ABC):
    """
    Base evaluator class

    This takes in the initialized lens fixture and defines standardized tests
    for each evaluator.
    """

    ...


# class TestFrozenBinaryCLF(Base_Frozen_Test):
"""
TestFrozenBinaryCLF

Tests Lens against frozen results for a binary classifier.
Model in freeze: sklearn.LogisticRegression
Assessments in freeze: Performance and ModelFairness
Metrics in freeze: false_negative_rate, average_precision_score

Data info: Data are derived from a Kaggle dataset on loan worthiness:
https://www.kaggle.com/datasets/vikasukani/loan-eligible-dataset?select=loan-train.csv
Data were cleaned by Amin Rasekh (amin@credo.ai) here:
https://github.com/credo-ai/customer_demos/blob/prod/prod/d3_loan_approval/data_preparation.ipynb
Neither CSV file is pushed to GitHub;
Test relies on pickled version in tests/frozen_ml_tests/frozen_data/binary/loan_processed.pkl and the derived
validation data in tests/frozen_ml_tests/frozen_data/binary/loan_validation.pkl
"""


@pytest.mark.skip(
    reason="Currently not correctly working for multiple sensitive features"
)
@pytest.mark.parametrize("evaluator", SUPORTED_EVALUATORS, ids=TEST_EVALUATOR_IDS)
def test_frozen_binary_clf_results(
    frozen_classifier, frozen_validation_data, evaluator
):
    """
    Test checks each component of the results for equality with frozen results
    Frozen results is dictionary with "[Evaluator] Assessment" as keys and lists
    of DataFrames as values.
    Current freeze (10/20/22) uses Performance and ModelFairness evaluators.
        Performance list contains 1 DataFrame: performance results
        ModelFairness list contains 2 DataFrames: parity results, disaggregated performance results
    """
    lens = Lens(
        model=frozen_classifier,
        assessment_data=frozen_validation_data,
        # training_data=frozen_training_data,
    )
    eval = name2evaluator(evaluator)(FROZEN_METRICS)
    lens.add(eval)
    lens.run()

    with open(
        "tests/frozen_ml_tests/frozen_results/binary/binary_clf_"
        + evaluator
        + "_results.pkl",
        "rb",
    ) as f:
        frozen_results = pickle.load(f)

    test_results = lens.get_results()[0]["results"]
    for idx, result in enumerate(test_results):
        current_result = result.reset_index(drop=True)
        testing.assert_frame_equal(current_result, frozen_results[idx])
