import pprint
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from pandas import Series, DataFrame
from credoai.utils.common import ValidationError


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
            "label": self._add_label(),
            "metadata": self.metadata,
            "data": self._add_data(),
            "creation_time": self.creation_time,
        }
        return structure

    @abstractmethod
    def _add_data(self):
        """
        Adds evidence type specific data
        """
        data = {}
        return data

    @abstractmethod
    def _add_label(self):
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

    def __init__(self, id: str, data: Series, **metadata):
        self.data = data
        self.metadata = metadata

        super().__init__(id, "metric", self.metadata)

    def _add_data(self):
        value_type = [x for x in self.data.index if x not in ["type", "subtype"]]
        return {
            "value": self.data[value_type].to_dict(),
        }

    def _add_label(self):
        label = {"metric_type": self.data.type, "subtype": self.data.subtype}
        return label


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

    def __init__(
        self,
        label: str,
        data: DataFrame,
        model_name: str = None,
        data_name: str = None,
        **metadata
    ):
        self.data = data
        self.label = label

        super().__init__("table", label, metadata)
        self.metadata.update({"model": model_name, "dataset": data_name})

    def _struct(self):
        return {"data": self.data}
