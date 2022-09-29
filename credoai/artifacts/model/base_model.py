from abc import ABC

from credoai.utils import ValidationError
from credoai.utils.model_utils import get_model_info
from typing import List


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

    def __init__(
        self,
        type: str,
        possible_functions: List[str],
        necessary_functions: List[str],
        name: str,
        model_like,
    ):

        self.type = type
        self.name = name
        self.model_like = model_like

        info = get_model_info(model_like)
        self.framework = info["framework"]
        self._validate(necessary_functions)
        self._build(possible_functions)

    def _build(self, function_names):
        for key in function_names:
            self._add_functionality(key)

    def _validate(self, function_names):
        for key in function_names:
            validated = getattr(self.model_like, key, False)
            if not validated:
                raise ValidationError(f"Model-like must have a {key} function")

    def _add_functionality(self, key):
        """Adds functionality from model_like, if it exists"""
        func = getattr(self.model_like, key, None)
        if func:
            self.__dict__[key] = func
