# reset style after pandas profiler

import matplotlib
import pandas as pd
from connect.evidence import TableContainer
from connect.evidence.lens_evidence import DataProfilerContainer

from credoai.artifacts.data.base_data import Data
from credoai.evaluators.evaluator import Evaluator
from credoai.evaluators.utils import check_data_instance
from credoai.utils.common import ValidationError, check_pandas

backend = matplotlib.get_backend()
# load ydata profiler, which sets backend to Agg
from ydata_profiling import ProfileReport

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
        Potential arguments to be passed to ydata_profiling.ProfileReport

    Required Artifacts
    ------------------
    Generally artifacts are passed directly to :class:`credoai.lens.Lens`, which
    handles evaluator setup. However, if you are using the evaluator directly, you
    will need to pass the following artifacts when instantiating the evaluator:

    data : TabularData
        The data to evaluate, which must include a sensitive feature
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
        metadata = self._get_column_meta()
        results = DataProfilerContainer(profile, **self.get_info(metadata=metadata))
        self.results = [results] + self._wrap_sensitive_counts()
        return self

    def _get_column_meta(self):
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

    def _wrap_sensitive_counts(self):
        counts = sensitive_feature_counts(self.data)
        if counts:
            return [TableContainer(count) for count in counts]


def create_report(df, **profile_kwargs):
    """Creates a pandas profiler report"""
    default_kwargs = {"title": "Dataset", "minimal": True}
    default_kwargs.update(profile_kwargs)
    return ProfileReport(df, **default_kwargs)


def sensitive_feature_counts(data):
    """Returns the sensitive feature distributions of a Data object"""
    if data.sensitive_features is None:
        return
    sensitive_feature_distributions = []
    for name, col in data.sensitive_features.items():
        df = pd.concat([col.value_counts(), col.value_counts(normalize=True)], axis=1)
        df.columns = ["Count", "Proportion"]
        df.name = f"{name} Distribution"
        sensitive_feature_distributions.append(df)
    return sensitive_feature_distributions
