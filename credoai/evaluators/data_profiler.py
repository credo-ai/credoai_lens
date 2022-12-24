# reset style after pandas profiler

import matplotlib
import pandas as pd
from connect.evidence.lens_evidence import DataProfilerContainer

from credoai.artifacts.data.base_data import Data
from credoai.evaluators import Evaluator
from credoai.evaluators.utils import check_data_instance
from credoai.utils.common import ValidationError, check_pandas

backend = matplotlib.get_backend()
# load pands profiler, which sets backend to Agg
from pandas_profiling import ProfileReport

matplotlib.use(backend)


class DataProfiler(Evaluator):
    """
    Data profiling evaluator for Credo AI

    This evaluator runs the pandas profiler on a data. Pandas profiler calculates a number
    of descriptive statistics about the data. The DataProfiler can only be run on parts of the
    data that are pandas objects (dataframes or series). E.g., if X is a multi-dimensional
    array, X will NOT be profiled.

    Parameters
    ----------
    profile_kwargs
        Potential arguments to be passed to pandas_profiling.ProfileReport
    """

    required_artifacts = {"data"}

    def __init__(self, **profile_kwargs):
        self.profile_kwargs = profile_kwargs
        super().__init__()

    def _validate_arguments(self):
        check_data_instance(self.data, Data)
        return self

    def _setup(self):
        data_subsets = [self.data.X, self.data.y, self.data.sensitive_features]
        self.data_to_profile = list(filter(check_pandas, data_subsets))
        if not self.data_to_profile:
            raise ValidationError(
                "At least one of X, y or sensitive features must exist and be a pandas object"
            )
        self.data_to_profile = pd.concat(self.data_to_profile, axis=1)
        return self

    def evaluate(self):
        """Generates data profile reports"""
        profile = create_report(self.data_to_profile, **self.profile_kwargs)
        metadata = self.get_column_meta()
        results = DataProfilerContainer(profile, **self.get_info(metadata=metadata))
        self.results = [results]
        return self

    def get_column_meta(self):
        metadata = {}
        if check_pandas(self.data.X):
            metadata["model_features"] = self.data.X.columns.tolist()
        if check_pandas(self.data.sensitive_features):
            metadata[
                "sensitive_features"
            ] = self.data.sensitive_features.columns.tolist()
        if isinstance(self.data.y, pd.Series):
            metadata["target"] = self.data.y.name
        elif isinstance(self.data.y, pd.DataFrame):
            metadata["targets"] = self.data.y.columns.tolist()
        return metadata


def create_report(data, **profile_kwargs):
    """Creates a pandas profiler report"""
    default_kwargs = {"title": "Dataset", "minimal": True}
    default_kwargs.update(profile_kwargs)
    return ProfileReport(data, **default_kwargs)
