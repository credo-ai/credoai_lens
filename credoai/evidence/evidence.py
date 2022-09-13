import pprint
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Tuple

from credoai.utils import ValidationError
from pandas import DataFrame, Series


class Evidence(ABC):
    def __init__(
        self,
        type: str,
        metadata: Optional[dict] = None,
    ):
        self.type = type
        self.metadata = metadata if metadata else {}
        self.creation_time: str = datetime.utcnow().isoformat()
        self._validate()

    def struct(self):
        """Structure of evidence"""
        structure = {
            "type": self.type,
            "label": self.label(),
            "creation_time": self.creation_time,
            # "metadata": self.metadata,
        } | self._struct()
        return structure

    def _struct(self):
        """Function to reflect additional structure of child classes"""
        return {}

    @property
    @abstractmethod
    def label(self):
        """
        Adds evidence type specific label
        """
        label = {}
        return label

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
    model_name : str, optional
        Name of Model, default None
    data_name : str, optional
        Name of Data, default None
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
        self.metadata = metadata
        super().__init__("metric", self.metadata)

    def label(self):
        label = {
            "metric_type": self.type,
        } | self.metadata
        return label

    def _struct(self):
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
        self.metadata = metadata

        super().__init__("table", self.metadata)

    def _struct(self):
        return {"data": self.data.to_csv()}

    def label(self):
        label = {"table_name": self.name} | self.metadata
        return label
