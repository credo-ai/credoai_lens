"""
Generic containers for evaluator results

The containers accept raw data from the evaluators and convert it into
suitable evidences.
"""
from abc import ABC, abstractmethod

import pandas as pd
from credoai.utils import ValidationError

from .evidence import (
    DataProfilerEvidence,
    MetricEvidence,
    ModelProfilerEvidence,
    TableEvidence,
)


class EvidenceContainer(ABC):
    def __init__(self, evidence_class, df, labels=None, metadata=None):
        """Abstract Class defining Evidence Containers

        Evidence Containers are light wrappers around dataframes that
        validate their format for the purpose of evidence export. They
        define a "to_evidence" function which transforms the
        dataframe into a particular evidence format

        Parameters
        ----------
        evidence_class : Evidence
            An Evidence class
        df : pd.DataFrame
            The dataframe, formatted appropriately for the evidence type
        labels : dict
            Additional labels to pass to underlying evidence
        metadata : dict
            Metadata to pass to underlying evidence
        """
        self.evidence_class = evidence_class
        if not isinstance(df, pd.DataFrame):
            raise ValidationError("'df' must be a dataframe")
        self._validate(df)
        self._df = df
        self.labels = labels
        self.metadata = metadata or {}

    @property
    def df(self):
        return self._df

    @abstractmethod
    def to_evidence(self):
        pass

    @abstractmethod
    def _validate(self, df):
        pass


class MetricContainer(EvidenceContainer):
    """Containers for all Metric type evidence"""

    def __init__(self, df: pd.DataFrame, labels: dict = None, metadata: dict = None):
        super().__init__(MetricEvidence, df, labels, metadata)

    def to_evidence(self, **metadata):
        evidence = []
        for _, data in self._df.iterrows():
            evidence.append(
                self.evidence_class(
                    additional_labels=self.labels, **data, **self.metadata, **metadata
                )
            )
        return evidence

    def _validate(self, df):
        required_columns = {"type", "value"}
        column_overlap = df.columns.intersection(required_columns)
        if len(column_overlap) != len(required_columns):
            raise ValidationError(f"Must have columns: {required_columns}")


class TableContainer(EvidenceContainer):
    """Container for all Table type evidence"""

    def __init__(self, df: pd.DataFrame, labels: dict = None, metadata: dict = None):
        super().__init__(TableEvidence, df, labels, metadata)

    def to_evidence(self, **metadata):
        return [
            self.evidence_class(
                self._df.name, self._df, self.labels, **self.metadata, **metadata
            )
        ]

    def _validate(self, df):
        try:
            df.name
        except AttributeError:
            raise ValidationError("DataFrame must have a 'name' attribute")


class ProfilerContainer(EvidenceContainer):
    """Container for al profiler type evidence"""

    def __init__(self, data, labels: dict = None, metadata: dict = None):
        super().__init__(DataProfilerEvidence, data, labels, metadata)

    def to_evidence(self, **metadata):
        return [self.evidence_class(self._df, self.labels, **self.metadata, **metadata)]

    def _validate(self, df):
        if list(df.columns) != ["results"]:
            raise ValidationError("Profiler data must only have one column: 'results'")


class ModelProfilerContainer(EvidenceContainer):
    """Container for Model Profiler type evidence"""

    def __init__(self, df, labels=None, metadata=None):
        super().__init__(ModelProfilerEvidence, df, labels, metadata)

    def to_evidence(self, **metadata):
        return [self.evidence_class(self._df, self.labels, **self.metadata, **metadata)]

    def _validate(self, df):
        necessary_index = ["parameters", "feature_names", "model_name"]
        if list(df.columns) != ["results"]:
            raise ValidationError(
                "Model profiler data must only have one column: 'results'"
            )
        if sum(df.index.isin(necessary_index)) != 3:
            raise ValidationError(f"Model profiler data must contain {necessary_index}")
