"""Abstract class for model artifacts used by `Lens`"""
from abc import ABC
from typing import List, Optional

from credoai.utils import ValidationError
from credoai.utils.model_utils import get_model_info


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
    tags : dictionary, optional
        Additional metadata to add to model
        E.g., {'model_type': 'binary_classification'}

    """

    def __init__(
        self,
        type: str,
        possible_functions: List[str],
        necessary_functions: List[str],
        name: str,
        model_like,
        tags: Optional[dict] = None,
    ):
        self.type = type
        self.name = name
        self.model_like = model_like
        self.tags = tags or {}
        self._process_model(model_like, necessary_functions, possible_functions)
        self.__post_init__()

    def __post_init__(self):
        """Optional custom functionality to call after Base Model init"""
        pass

    @property
    def tags(self):
        return self._tags

    @tags.setter
    def tags(self, value):
        if not isinstance(value, dict) and value is not None:
            raise ValidationError("Tags must be of type dictionary")
        self._tags = value

    def _process_model(self, model_like, necessary_functions, possible_functions):
        self.model_info = get_model_info(model_like)
        self._validate_framework()
        self._validate_callables(necessary_functions)
        self._build(possible_functions)
        return self

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

    def _validate_framework(self):
        """
        Optional check to determine whether model is from supported framework.
        WARNS if unsupported framework (does not RAISE).
        """
        pass

    def _validate_callables(self, function_names: List[str]):
        """
        Checks that the necessary methods are available in model_like

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
