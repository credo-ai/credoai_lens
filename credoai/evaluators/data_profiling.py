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

    def __init__(self, dataset_name=None, **profile_kwargs):
        self.profile_kwargs = profile_kwargs
        self.dataset_name = dataset_name
        self.results = {}

    def _setup(self, assessment_data, training_data):
        if self.dataset_name is None:
            self.data_to_eval = self.assessment_data
        else:
            self.data_to_eval = [
                x
                for x in [self.assessment_data, self.self.training_data]
                if x and x.name == self.dataset_name
            ]
            if len(self.data_to_eval) > 1:
                raise ValidationError(
                    f"More then 1 dataset named {self.dataset_name} were found."
                )
            self.data_to_eval = self.data_to_eval[0]  # Pick the only member

        self.data = pd.concat([self.data_to_eval.X, self.data_to_eval.y], axis=1)
        return self

    def _validate_arguments(self):
        if not isinstance(self.data_to_eval, TabularData):
            raise ValidationError("Data under evaluation is not of type TabularData.")

        if self.data_to_eval.sensitive_features is None:
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
