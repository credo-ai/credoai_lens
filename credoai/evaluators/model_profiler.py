from typing import Optional
from pandas import DataFrame
from credoai.evaluators import Evaluator
from credoai.evaluators.utils.validation import check_requirements_existence
from credoai.evidence.containers import ModelProfilerContainer
from credoai.utils.common import ValidationError

USER_INFO_TEMPLATE = {
    "developed_by": None,
    "shared_by": None,
    "model_type": None,
    "intended_use": None,
    "downstream_use": None,
    "out_of_scope_use": None,
    "language": None,
    "related_models": None,
    "license": None,
    "resources_for_more_info": None,
    "input_description": None,
    "output_description": None,
    "performance_evaluated_on": None,
    "limitations": None,
}

PROTECTED_KEYS = [
    "model_name",
    "python_model_type",
    "library",
    "model_architecture",
    "feature_names",
]


class ModelProfiler(Evaluator):
    """
    Model profiling evaluator.

    This evaluator builds a model card the purpose of which is to characterize
    a fitted model.

    Parameters
    ----------
    model_info : Optional[dict], optional
        Information provided by the user that cannot be inferred by
        the model itself. The dictionary con contain any number of elements,
        a template can be provided by running the generate_template() method.

        The only restrictions are checked in a validation step:
        1. Some keys are protected because they are used internally
        2. Only basic python types are accepted as values

    """

    name = "ModelProfiler"
    required_artifacts = {"model"}

    def __init__(self, model_info: Optional[dict] = None):
        """
        _summary_

        Parameters
        ----------
        model_info : Optional[dict], optional
            _description_, by default None
        """
        super().__init__()
        self.usr_model_info = model_info
        if not self.usr_model_info:
            self.usr_model_info = {}
        self._validate_usr_model_info()

    def _setup(self):
        self.model_name = self.model.name
        self.model = self.model.model_like
        self.model_type = type(self.model)

    def _validate_arguments(self):
        check_requirements_existence(self)

    def evaluate(self):
        # Collate info
        basic = self._get_basic_info()
        res = self._get_model_params()
        self.usr_model_info = {k: v for k, v in self.usr_model_info.items() if v}
        res = {**basic, **res, **self.usr_model_info}
        # Format
        res = DataFrame.from_dict(res, orient="index")
        res.columns = ["results"]
        # Package into evidence
        self.results = [ModelProfilerContainer(res, **self.get_container_info())]
        return self.results

    def _get_basic_info(self) -> dict:
        """
        Collect basic information directly from the model artifact.

        Returns
        -------
        dict
            Dictionary containing name, full class identifier
        """
        return {
            "model_name": self.model_name,
            "python_model_type": str(self.model_type).split("'")[1],
        }

    def _get_model_params(self) -> dict:
        """
        Select which parameter structure to utilize based on library/model used.

        Returns
        -------
        dict
            Dictionary of model info
        """
        if "sklearn" in str(self.model_type):
            return self._get_sklearn_model_params()

    def _get_sklearn_model_params(self) -> dict:
        """
        Get info from sklearn like models

        Returns
        -------
        dict
            Dictionary of info about the model
        """
        parameters = self.model.get_params()
        model_architecture = self.model_type.__name__
        library = "sklearn"
        feature_names = list(self.model.feature_names_in_)
        return {
            "library": library,
            "model_architecture": model_architecture,
            "parameters": parameters,
            "feature_names": feature_names,
        }

    def _validate_usr_model_info(self):
        """
        Validate user model info.

        Any key is valid unless it's already in use internally.

        """
        protected = [k for k in self.usr_model_info.keys() if k in PROTECTED_KEYS]
        if protected:
            message = f"Found {protected} in model_info.keys(), these keys are already in use. Please rename/remove them."
            raise ValidationError(message)

        accepted_formats = (list, int, float, dict, str)
        non_accepted = [
            k
            for k, v in self.usr_model_info.items()
            if not isinstance(v, accepted_formats) and v is not None
        ]
        if non_accepted:
            message = f"The items {non_accepted} in model info are not of types: (list, int, float, dict, str)"
            raise ValidationError(message)

    @staticmethod
    def generate_template() -> dict:
        """
        Passes a template for model related info that the user
        can populate and customize.

        Loosely based on:
        https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/modelcard_template.md#model-details
        https://modelcards.withgoogle.com/model-reports

        Returns
        -------
        dict
            Dictionary of keys working as bookmarks for the user info
        """
        return USER_INFO_TEMPLATE
