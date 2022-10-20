"""
Testing protocols for the Lens package. Tested functionalities:
    1. General inits
    2. Individual evaluator runs
"""

from abc import ABC, abstractmethod

import pytest
from credoai.artifacts import TabularData, ClassificationModel
from credoai.evaluators import (
    DataEquity,
    DataFairness,
    DataProfiling,
    ModelEquity,
    ModelFairness,
    Performance,
    Privacy,
    Security,
    evaluator,
)
from credoai.evaluators.ranking_fairness import RankingFairness
from credoai.lens import Lens
from pandas import DataFrame
import pickle
import os

TEST_METRICS = [
    ["false_negative_rate"],
    ["average_precision_score"],
    ["false_negative_rate", "average_precision_score"],
]
TEST_METRICS_IDS = ["binary_metric", "probability_metric", "both"]

STRING2EVALUATOR = {
    "DataEquity": DataEquity,
    "DataFairness": DataFairness,
    "DataProfiling": DataProfiling,
    "ModelEquity": ModelEquity,
    "ModelFairness": ModelFairness,
    "Performance": Performance,
    "Privacy": Privacy,
    "Security": Security,
    "evaluator": evaluator,
}


@pytest.fixture(scope="class")
def init_lens(
    classification_model,
    classification_assessment_data,
    classification_train_data,
    request,
):
    my_pipeline = Lens(
        model=classification_model,
        assessment_data=classification_assessment_data,
        training_data=classification_train_data,
    )
    request.cls.pipeline = my_pipeline


@pytest.mark.usefixtures("init_lens")
class Base_Evaluator_Test(ABC):
    """
    Base evaluator class

    This takes in the initialized lens fixture and defines standardized tests
    for each evaluator.
    """

    ...


class TestModelFairness(Base_Evaluator_Test):
    @pytest.mark.parametrize("metrics", TEST_METRICS, ids=TEST_METRICS_IDS)
    def test_full_run(self, metrics):
        evaluator = ModelFairness(metrics)
        self.pipeline.add(evaluator)
        self.pipeline.run()
        assert len(self.pipeline.pipeline) == 4
        assert self.pipeline.get_results()
        self.pipeline.pipeline = {}


def test_privacy(
    credit_classification_model, credit_assessment_data, credit_training_data
):
    lens = Lens(
        model=credit_classification_model,
        assessment_data=credit_assessment_data,
        training_data=credit_training_data,
    )
    lens.add(Privacy(attack_feature="MARRIAGE"))
    lens.run()
    assert lens.get_results()


class TestDataFairness(Base_Evaluator_Test):
    evaluator = DataFairness()

    def test_add(self):
        self.pipeline.add(self.evaluator)
        assert len(self.pipeline.pipeline) == 4

    def test_run(self):
        self.pipeline.get_results()
        assert True


class TestDataProfiling(Base_Evaluator_Test):
    evaluator = DataProfiling()

    def test_add(self):
        self.pipeline.add(self.evaluator)
        assert len(self.pipeline.pipeline) == 2

    def test_run(self):
        self.pipeline.get_results()
        assert True


class TestModelEquity(Base_Evaluator_Test):
    evaluator = ModelEquity()

    def test_add(self):
        self.pipeline.add(self.evaluator)
        assert len(self.pipeline.pipeline) == 2

    def test_run(self):
        self.pipeline.run()
        assert self.pipeline.get_results()


class TestDataEquity(Base_Evaluator_Test):
    evaluator = DataEquity()

    def test_add(self):
        self.pipeline.add(self.evaluator)
        assert len(self.pipeline.pipeline) == 4

    def test_run(self):
        self.pipeline.run()
        assert self.pipeline.get_results()


class TestSecurity(Base_Evaluator_Test):
    evaluator = Security()

    def test_add(self):
        self.pipeline.add(self.evaluator)
        assert len(self.pipeline.pipeline) == 1

    def test_run(self):
        self.pipeline.run()
        assert self.pipeline.get_results()


class TestPerformance(Base_Evaluator_Test):
    @pytest.mark.parametrize("metrics", TEST_METRICS, ids=TEST_METRICS_IDS)
    def test_full_run(self, metrics):
        evaluator = Performance(metrics)
        self.pipeline.add(evaluator)
        self.pipeline.run()
        assert len(self.pipeline.pipeline) == 1
        assert self.pipeline.get_results()
        self.pipeline.pipeline = {}


class TestThresholdPerformance(Base_Evaluator_Test):
    evaluator = Performance(["roc_curve"])

    def test_add(self):
        self.pipeline.add(self.evaluator)
        assert len(self.pipeline.pipeline) == 1

    def test_run(self):
        self.pipeline.run()
        assert self.pipeline.get_results()


class TestThresholdPerformanceMultiple(Base_Evaluator_Test):
    evaluator = Performance(["roc_curve", "precision_recall_curve"])

    def test_add(self):
        self.pipeline.add(self.evaluator)
        assert len(self.pipeline.pipeline) == 1

    def test_run(self):
        self.pipeline.run()
        assert self.pipeline.get_results()


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


class TestFrozenBinaryCLF:
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

    # Load pipeline information: evaluators and metrics to run
    with open(
        "tests/frozen_ml_tests/frozen_results/binary/pipeline_info.pkl", "rb"
    ) as f:
        pipeline_info = pickle.load(f)

    pipeline = [
        (
            STRING2EVALUATOR[assessment](pipeline_info["metrics"]),
            assessment + " Assessment",
        )
        for assessment in pipeline_info["assessments"]
    ]

    # Load frozen classifier and wrap as a Credo Model
    with open("tests/frozen_ml_tests/frozen_models/loan_binary_clf.pkl", "rb") as f:
        clf = pickle.load(f)

    test_model = ClassificationModel("binary_clf", clf)

    # Load frozen validation data and wrap as Credo Data
    with open(
        "tests/frozen_ml_tests/frozen_data/binary/loan_validation.pkl", "rb"
    ) as f:
        val_data = pickle.load(f)

    val_data_credo = TabularData(
        name=val_data["name"],
        X=val_data["val_features"],
        y=val_data["val_labels"],
        sensitive_features=val_data["sensitive_features"],
    )

    # Load frozen results
    with open(
        "tests/frozen_ml_tests/frozen_results/binary/binary_clf_results.pkl", "rb"
    ) as f:
        frozen_results = pickle.load(f)

    # Create lens object
    lens = Lens(model=test_model, assessment_data=val_data_credo, pipeline=pipeline)

    def test_results(self):
        """
        Test checks each component of the results for equality with frozen results
        Frozen results is dictionary with "[Evaluator] Assessment" as keys and lists
        of DataFrames as values.
        Current freeze (10/20/22) uses Performance and ModelFairness evaluators.
            Performance list contains 1 DataFrame: performance results
            ModelFairness list contains 2 DataFrames: parity results, disaggregated performance results
        """
        self.lens.run()
        test_results = self.lens.get_results()
        assert len(test_results) == len(self.frozen_results)
        for assessment, assessment_results in test_results.items():
            for idx, result in enumerate(assessment_results):
                current_result = result.reset_index(drop=True)
                assert current_result.equals(self.frozen_results[assessment][idx])


class TestRankingFairnes:
    evaluator = RankingFairness()

    df = DataFrame(
        {
            "rankings": [1, 2, 3, 4, 5, 6, 7],
            "sensitive_features": ["f", "f", "m", "m", "f", "m", "f"],
        }
    )
    data = TabularData(
        name="ranks", y=df[["rankings"]], sensitive_features=df[["sensitive_features"]]
    )
    expected_results = DataFrame(
        {
            "value": [0.86, 1.14, 0.32],
            "type": ["minimum_skew", "maximum_skew", "NDKL"],
            "subtype": ["score"] * 3,
        }
    )
    pipeline = Lens(assessment_data=data)

    def test_add(self):
        self.pipeline.add(self.evaluator, "dummy")
        assert len(self.pipeline.pipeline) == 1

    def test_run(self):
        self.pipeline.run()
        assert self.pipeline.get_results()

    def test_results(self):
        results = self.pipeline.get_results()["dummy"][0].round(2)
        results = results.reset_index(drop=True)
        assert results.equals(self.expected_results)


def test_bulk_pipeline_run(
    classification_model, classification_assessment_data, classification_train_data
):
    """
    Testing the passing of the list of evaluator works
    and the pipeline is running.
    """
    pipe_structure = [
        (Security(), "Security assessment"),
        (DataProfiling(), "Profiling test data"),
        (DataFairness(), "Test data Fairness"),
    ]
    my_pipeline = Lens(
        model=classification_model,
        assessment_data=classification_assessment_data,
        training_data=classification_train_data,
        pipeline=pipe_structure,
    )
    my_pipeline.run()
    assert my_pipeline.get_results()


@pytest.mark.xfail(raises=RuntimeError)
def test_empty_pipeline_run(
    classification_model, classification_assessment_data, classification_train_data
):
    my_pipeline = Lens(
        model=classification_model,
        assessment_data=classification_assessment_data,
        training_data=classification_train_data,
    )
    my_pipeline.run()
