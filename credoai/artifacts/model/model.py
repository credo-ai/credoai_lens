from abc import ABC, abstractmethod

from credoai.utils.model_utils import get_model_info


class Model(ABC):
    """Base class for all models in Lens.

    Parameters
    ----------
    name : str
        Label of the model
    type : str, optional
        Type of the model
    model_like : model_like
        A model or pipeline.

    """

    def __init__(self, type, name, model_like):
        self.type = type
        self.name = name
        self.model_like = model_like

        info = get_model_info(model_like)
        self.framework = info["framework"]
        self._validate(self.model_like)

    @abstractmethod
    def _build(self):
        pass

    def _add_functionality(self, key, model_like):
        func = getattr(model_like, key, None)
        if func:
            self.__dict__[key] = func

    def _validate(self):
        pass
