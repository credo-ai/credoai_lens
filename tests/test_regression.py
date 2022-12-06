"""
Testing protocols for the Lens package. Tested functionalities:
    1. General inits
    2. Individual evaluator runs
"""


import pytest


from credoai.evaluators import (
    DataEquity,
    DataProfiler,
    ModelEquity,
    ModelFairness,
    Performance,
)


TEST_METRICS = [["r2_score"]]
TEST_METRICS_IDS = ["regression_metric"]


# class TestModelFairness(Base_Evaluator_Test):
#     @pytest.mark.parametrize("metrics", TEST_METRICS, ids=TEST_METRICS_IDS)
#     def test_full_run(self, metrics):
#         evaluator = ModelFairness(metrics)
#         self.pipeline.add(evaluator)
#         self.pipeline.run()
#         assert len(self.pipeline.pipeline) == 4
#         assert self.pipeline.get_results()
#         self.pipeline.pipeline = {}


@pytest.mark.parametrize(
    "evaluator",
    [DataProfiler, ModelEquity, DataEquity],
    ids=["DataProfiler", "ModelEquity", "DataEquity"],
)
def test_generic_evaluator(init_lens_regression, evaluator):
    """
    Any evaluator not requiring specific treatment can be tested here
    """
    lens, temp_file, gov = init_lens_regression
    lens.add(evaluator())
    lens.run()
    pytest.assume(lens.get_results())
    pytest.assume(lens.get_evidence())
    pytest.assume(lens.send_to_governance())
    pytest.assume(not gov._file_export(temp_file))


# class TestPerformance(Base_Evaluator_Test):
#     @pytest.mark.parametrize("metrics", TEST_METRICS, ids=TEST_METRICS_IDS)
#     def test_full_run(self, metrics):
#         evaluator = Performance(metrics)
#         self.pipeline.add(evaluator)
#         self.pipeline.run()
#         assert len(self.pipeline.pipeline) == 1
#         assert self.pipeline.get_results()
#         self.pipeline.pipeline = {}
