"""Abstract class for model artifacts used by `Lens`"""
from abc import ABC

from credoai.utils import ValidationError
from credoai.utils.model_utils import get_model_info
from typing import List


class Model(ABC):
    """Base class for all models in Lens.

    Parameters
    ----------
    type : str, optional
        Type of the model
    possible_functions: List[str]
        List of possible methods that can be used by a model
    necessary_functions: List[str]
        List of necessary methods for the model type
    name: str
        Class name.
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

    def _build(self, function_names: List[str]):
        """
        Makes the necessary methods available in the class

        Parameters
        ----------
        function_names : List[str]
            List of possible methods to be imported from model_like
        """
        for key in function_names:
            self._add_functionality(key)

    def _validate(self, function_names: List[str]):
        """
        Checks the the necessary methods are available in model_like

        Parameters
        ----------
        function_names : List[str]
            List of necessary functions

        Raises
        ------
        ValidationError
            If a necessary method is missing from model_like
        """
        for key in function_names:
            validated = getattr(self.model_like, key, False)
            if not validated:
                raise ValidationError(f"Model-like must have a {key} function")

    def _add_functionality(self, key: str):
        """Adds functionality from model_like, if it exists"""
        func = getattr(self.model_like, key, None)
        if func:
            self.__dict__[key] = func