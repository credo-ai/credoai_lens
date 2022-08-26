import pprint
from abc import ABC, abstractmethod
from datetime import datetime


class Evidence(ABC):
    def __init__(self, type, label, metadata=None):
        self.type = type
        self.label = label
        self.metadata = metadata | {}
        self.creation_time = datetime.utcnow().isoformat()

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

    def __str__(self):
        return pprint.pformat(self.struct())


class Metric(Evidence):
    """
    A metric record

    Record of a metric

    Parameters
    ----------
    label : string
        short identifier for metric.
    value : float
        metric value
    model_name : str, optional
        Name of Model, default None
    data_name : str, optional
        Name of Data, default None
    metadata : dict, optional
        Arbitrary keyword arguments to append to metric as metadata. These will be
        displayed in the governance app
    """

    def __init__(
        self, label, value, subtype="base", model_name=None, data_name=None, **metadata
    ):
        self.value = value
        super().__init__("metric", label, metadata)
        self.metadata.update({"model": model_name, "dataset": data_name})

    def _struct(self):
        return {
            "value": self.value,
        }
