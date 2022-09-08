import pprint
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Tuple

import pandas as pd
from credoai.utils.common import ValidationError


class Evidence(ABC):
    def __init__(self, type: str, label: dict, metadata: dict = None):
        self.type = type
        self.label = label
        self.metadata = metadata | {}
        self.creation_time = datetime.utcnow().isoformat()
        self._validate()

    def struct(self):
        structure = {
            "type": self.type,
            "label": self.label,
            "metadata": self.metadata,
            "creation_time": self.creation_time,
        }
        additional = self._struct()
        if additional:
            structure.update(additional)
        return structure

    @abstractmethod
    def _struct(self):
        pass

    def _validate(self):
        pass

    def __str__(self):
        return pprint.pformat(self.struct())


class Metric(Evidence):
    """
    Metric Evidence


    Parameters
    ----------
    label : dict
        key-value pairs, used as identifier for metric.
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
        label: dict,
        value: float,
        confidence_interval: Tuple[float, float] = None,
        confidence_level: int = None,
        model_name: str = None,
        data_name: str = None,
        **metadata
    ):
        self.value = value
        self.confidence_interval = confidence_interval or []
        self.confidence_level = confidence_level

        super().__init__("metric", label, metadata)
        self.metadata.update({"model": model_name, "dataset": data_name})

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
    label : dict
        key-value pairs, used as identifier for table.
    data : pd.DataFrame
        a pandas DataFrame to use as evidence
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
        label: dict,
        data: pd.DataFrame,
        model_name: str = None,
        data_name: str = None,
        **metadata
    ):
        self.data = data

        super().__init__("table", label, metadata)
        self.metadata.update({"model": model_name, "dataset": data_name})

    def _struct(self):
        return {"data": self.data}
