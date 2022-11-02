# reset style after pandas profiler

import matplotlib
import pandas as pd
from credoai.artifacts.data.tabular_data import TabularData
from credoai.evaluators import Evaluator
from credoai.evaluators.utils.validation import check_data_instance
from credoai.evidence import ProfilerContainer
from credoai.utils.common import ValidationError

backend = matplotlib.get_backend()
# load pands profiler, which sets backend to Agg
from pandas_profiling import ProfileReport

matplotlib.use(backend)


class DataProfiler(Evaluator):
    """Data profiling module for Credo AI.

    This evaluator runs the pandas profiler on a data. Pandas profiler calculates a number
    of descriptive statistics about the data.

    Parameters
    ----------
    X : pandas.DataFrame
        The features
    y : pandas.Series
        The outcome labels
    profile_kwargs
        Passed to pandas_profiling.ProfileReport
    """

    required_artifacts = {"data"}

    def __init__(self, dataset_name=None, **profile_kwargs):
        self.profile_kwargs = profile_kwargs
        self.dataset_name = dataset_name
        super().__init__()

    def _setup(self):
        self.data_to_profile = pd.concat([self.data.X, self.data.y], axis=1)
        return self

    def _validate_arguments(self):
        check_data_instance(self.data, TabularData)
        return self

    def get_html_report(self):
        return self._create_reporter().to_html()

    def profile_data(self):
        return self._create_reporter().to_notebook_iframe()

    def evaluate(self):
        """Generates data profile reports"""
        results = self._create_reporter().get_description()
        results = pd.DataFrame({"results": results})
        results = ProfilerContainer(results, **self.get_container_info())
        self.results = [results]
        return self

    def _create_reporter(self):
        default_kwargs = {"title": "Dataset", "minimal": True}
        default_kwargs.update(self.profile_kwargs)
        return ProfileReport(self.data_to_profile, **default_kwargs)
