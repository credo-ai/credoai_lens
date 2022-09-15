# reset style after pandas profiler
import matplotlib
import pandas as pd
from credoai.artifacts.data.tabular_data import TabularData
from credoai.evaluators import Evaluator
from credoai.utils.common import ValidationError

backend = matplotlib.get_backend()
# load pands profiler, which sets backend to Agg
from pandas_profiling import ProfileReport

matplotlib.use(backend)


class DataProfiling(Evaluator):
    """Dataset profiling module for Credo AI.

    This module takes in features and labels and provides functionality to generate profile reports

    Parameters
    ----------
    X : pandas.DataFrame
        The features
    y : pandas.Series
        The outcome labels
    profile_kwargs
        Passed to pandas_profiling.ProfileReport
    """

    name = "DataProfiler"
    required_artifacts = ["data"]

    def __init__(self, dataset_name=None, **profile_kwargs):
        self.profile_kwargs = profile_kwargs
        self.dataset_name = dataset_name
        self.results = {}

    def _setup(self):
        self.data_to_eval = self.data

        self.data = pd.concat([self.data_to_eval.X, self.data_to_eval.y], axis=1)
        return self

    def _validate_arguments(self):
        if not isinstance(self.data, TabularData):
            raise ValidationError("Data under evaluation is not of type TabularData.")

        if self.data.sensitive_features is None:
            raise ValidationError(
                f"Step: {self.name} ->  No sensitive feature were found in the dataset"
            )

        return self

    def get_html_report(self):
        return self._create_reporter().to_html()

    def profile_data(self):
        return self._create_reporter().to_notebook_iframe()

    def evaluate(self):
        """Generates data profile reports"""
        self.results = self._create_reporter().get_description()
        return self

    def _prepare_results(self):
        return self

    def _create_reporter(self):
        default_kwargs = {"title": "Dataset", "minimal": True}
        default_kwargs.update(self.profile_kwargs)
        return ProfileReport(self.data, **default_kwargs)
