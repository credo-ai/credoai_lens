import pprint
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Tuple

from credoai.utils import ValidationError
from pandas import DataFrame, Series


class Evidence(ABC):
    def __init__(self, type: str, **metadata):
        self.type = type
        self.metadata = metadata
        self.creation_time: str = datetime.utcnow().isoformat()
        self._validate()

    def struct(self):
        """Structure of evidence"""
        structure = {
            "type": self.type,
            "label": self.label(),
            "data": self.data(),
            "creation_time": self.creation_time,
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

    def __str__(self):
        return pprint.pformat(self.struct())


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
        **metadata
    ):
        self.type = type
        self.value = value
        self.confidence_interval = confidence_interval
        self.confidence_level = confidence_level
        super().__init__("metric", **metadata)

    def label(self):
        label = {"metric_type": self.type}
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

    def __init__(self, name: str, data: DataFrame, **metadata):
        self.name = name
        self.data = data
        super().__init__("table", **metadata)

    def data(self):
        return {"csv": self.data.to_csv(index=False)}

    def label(self):
        label = {"table_name": self.name}
        return label
