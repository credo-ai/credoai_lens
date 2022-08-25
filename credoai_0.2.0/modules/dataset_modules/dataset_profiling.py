# reset style after pandas profiler
import matplotlib
import pandas as pd
from credoai.modules.credo_module import CredoModule

backend = matplotlib.get_backend()
# load pands profiler, which sets backend to Agg
from pandas_profiling import ProfileReport

matplotlib.use(backend)


class DatasetProfiling(CredoModule):
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

    def __init__(self, X: pd.DataFrame, y: pd.Series, **profile_kwargs):
        self.profile_kwargs = profile_kwargs
        self.data = pd.concat([X, y], axis=1)
        self.results = {}

    def get_html_report(self):
        return self._create_reporter().to_html()

    def profile_data(self):
        return self._create_reporter().to_notebook_iframe()

    def run(self):
        """Generates data profile reports"""
        self.results = self._create_reporter().get_description()
        return self

    def prepare_results(self):
        return None

    def _create_reporter(self):
        default_kwargs = {"title": "Dataset", "minimal": True}
        default_kwargs.update(self.profile_kwargs)
        return ProfileReport(self.data, **default_kwargs)
