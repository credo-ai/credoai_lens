import pprint
from abc import ABC, abstractmethod
from datetime import datetime
from turtle import update
from typing import Optional, Tuple

from pandas import Series, DataFrame


class Evidence(ABC):
    def __init__(
        self,
        id: str,
        type: str,
        metadata: Optional[dict] = None,
    ):
        self.id = id
        self.type = type
        self.metadata = metadata if metadata else {}
        self.creation_time: str = datetime.utcnow().isoformat()
        self._validate()

    def struct(self):
        structure = {
            "id": self.id,
            "type": self.type,
            "label": self._label(),
            # "metadata": self.metadata,
            "data": self._data(),
            "creation_time": self.creation_time,
        } | self._update_struct()
        return structure

    def _update_struct(self):
        return {}

    @property
    @abstractmethod
    def _data(self):
        """
        Adds evidence type specific data
        """
        data = {}
        return data

    @property
    @abstractmethod
    def _label(self):
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
    """

    def __init__(
        self,
        id: str,
        data: Series,
        confidence_interval: Tuple[float, float] = None,
        confidence_level: int = None,
        **metadata
    ):
        self.confidence_interval = confidence_interval
        self.confidence_level = confidence_level
        self.data = data
        self.metadata = metadata

        super().__init__(id, "metric", self.metadata)

    def _data(self):
        value_type = [x for x in self.data.index if x not in ["type", "subtype"]]
        return {
            "value": self.data[value_type].to_dict(),
        }

    def _label(self):
        label = {
            "metric_type": self.data.type,
            "calculation": self.data.subtype,
        } | self.metadata
        return label

    def _update_struct(self):
        return {
            "confidence_interval": self.confidence_interval,
            "confidence_level": self.confidence_level,
        }


class Table(Evidence):
    """
    Table Evidence


    Parameters
    ----------
    label : string
        short identifier for metric.
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

    def __init__(self, id: str, data: DataFrame, **metadata):
        self.data = data
        self.metadata = metadata

        super().__init__(id, "table", self.metadata)

    def _data(self):
        value_type = [x for x in self.data.columns if x not in ["subtype"]]
        return {
            "value": self.data[value_type].to_dict(orient="split"),
        }

    def _label(self):
        label = {"calculation": "-".join(list(set(self.data.subtype)))} | self.metadata

        return label
