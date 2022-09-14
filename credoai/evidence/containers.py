from abc import ABC, abstractmethod

import pandas as pd
from credoai.utils import ValidationError

from .evidence import Metric, Table


class EvidenceContainer(ABC):
    def __init__(self, evidence_class, df, labels=None):
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
        """
        self.evidence_class = evidence_class
        if not isinstance(df, pd.DataFrame):
            raise ValidationError("'df' must be a dataframe")
        self._validate(df)
        self._df = df
        self.labels = labels

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
    def __init__(self, df: pd.DataFrame, labels: dict = None):
        super().__init__(Metric, df, labels)

    def to_evidence(self, **metadata):
        evidence = []
        for _, data in self._df.iterrows():
            evidence.append(
                self.evidence_class(additional_labels=self.labels, **data, **metadata)
            )
        return evidence

    def _validate(self, df):
        required_columns = {"type", "value", "subtype"}
        column_overlap = df.columns.intersection(required_columns)
        if len(column_overlap) != len(required_columns):
            raise ValidationError(f"Must have columns: {required_columns}")


class TableContainer(EvidenceContainer):
    def __init__(self, df: pd.DataFrame, labels: dict = None):
        super().__init__(Table, df, labels)

    def to_evidence(self, **metadata):
        return [self.evidence_class(self._df.name, self._df, self.labels, **metadata)]

    def _validate(self, df):
        try:
            df.name
        except AttributeError:
            raise ValidationError("DataFrame must have a 'name' attribute")
