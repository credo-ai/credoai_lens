"""
Testing protocols for the Lens package. Tested functionalities:
    1. General inits
    2. Individual evaluator runs
"""

from credoai.lens import Lens
import pytest
from credoai.evaluators import ModelFairness


@pytest.fixture(scope="module")
def lens(credo_model, assessment_data, train_data):
    my_pipeline = Lens(
        model=credo_model, assessment_data=assessment_data, training_data=train_data
    )
    return my_pipeline


def test_lens_init(lens):
    assert not lens.pipeline  # Pipeline holder is there


@pytest.fixture(scope="module")
def model_fairness_pipeline(lens):
    pipeline = lens
    pipeline.add(ModelFairness(metrics=["false positive rate"]))
    return pipeline


def test_run(model_fairness_pipeline):
    model_fairness_pipeline.run()
    # self.pipeline = pipeline
    assert True
