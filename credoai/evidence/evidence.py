import pprint
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Tuple

from credoai.utils import ValidationError
from pandas import DataFrame, Series


class Evidence(ABC):
    def __init__(self, type: str, additional_labels: dict = None, **metadata):
        self.type = type
        self.additional_labels = additional_labels or {}
        self.metadata = metadata
        self.creation_time: str = datetime.utcnow().isoformat()
        self._validate()

    def __str__(self):
        return pprint.pformat(self.struct())

    def struct(self):
        """Structure of evidence"""
        # set labels, additional_labels prioritized
        labels = self.label() | self.additional_labels
        structure = {
            "type": self.type,
            "label": labels,
            "data": self.data(),
            "generated_at": self.creation_time,
            "metadata": self.metadata,
        }
        return structure

    @property
    @abstractmethod
    def label(self):
        """
        Adds evidence type specific label
        """
        return {}

    @property
    @abstractmethod
    def data(self):
        """
        Adds data reflecting additional structure of child classes
        """
        return {}

    def _validate(self):
        pass


class Metric(Evidence):
    """
    Metric Evidence

    Parameters
    ----------
    type : string
        short identifier for metric.
    value : float
        metric value
    confidence_interval : [float, float]
        [lower, upper] confidence interval
    confidence_level : int
        Level of confidence for the confidence interval (e.g., 95%)
    metadata : dict, optional
        Arbitrary keyword arguments to append to metric as metadata. These will be
        displayed in the governance app
    """

    def __init__(
        self,
        type: str,
        value: float,
        confidence_interval: Tuple[float, float] = None,
        confidence_level: int = None,
        additional_labels=None,
        **metadata
    ):
        self.metric_type = type
        self.value = value
        self.confidence_interval = confidence_interval
        self.confidence_level = confidence_level
        super().__init__("metric", additional_labels, **metadata)

    def label(self):
        label = {"metric_type": self.metric_type}
        return label

    def data(self):
        return {
            "value": self.value,
            "confidence_interval": self.confidence_interval,
            "confidence_level": self.confidence_level,
        }

    def _validate(self):
        if self.confidence_interval and not self.confidence_level:
            raise ValidationError


class Table(Evidence):
    """
    Table Evidence


    Parameters
    ----------
    data : str
        a pandas DataFrame to use as evidence
    metadata : dict, optional
        Arbitrary keyword arguments to append to metric as metadata. These will be
        displayed in the governance app
    """

    def __init__(self, name: str, data: DataFrame, additional_labels=None, **metadata):
        self.name = name
        self._data = data
        super().__init__("table", additional_labels, **metadata)

    def data(self):
        return {
            "column": self.data.columns.tolist(),
            "value": self.data.values.tolist(),
        }

    def label(self):
        label = {"data_type": self.name}
        return label


class Profiler(Evidence):
    """
    Place holder for Profiler Evidence
    """

    def __init__(self, data: dict, additional_labels: dict = None, **metadata):
        self._data = data
        super().__init__("profiler", additional_labels, **metadata)

    def data(self):
        return self._data.to_csv(index=False)

    def label(self):
        return {"profiler_info": "placeholder"}
