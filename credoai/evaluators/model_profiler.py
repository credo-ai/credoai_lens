import re
from typing import Optional

import numpy as np
from connect.evidence import TableContainer
from connect.evidence.lens_evidence import ModelProfilerContainer
from pandas import DataFrame, concat

from credoai.evaluators.evaluator import Evaluator
from credoai.evaluators.utils.validation import check_existence
from credoai.utils import ValidationError, global_logger

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
    "model_library",
    "feature_names",
    "parameters",
    "data_sample",
]


class ModelProfiler(Evaluator):
    """
    Model profiling evaluator (Experimental)

    This evaluator builds a model card the purpose of which is to characterize
    a fitted model.

    The overall strategy is:

    1. Extract all potentially useful info from the model itself in an
       automatic fashion.
    2. Allow the user to personalize the model card freely.

    The method generate_template() provides a dictionary with several entries the
    user could be interested in filling up.

    Required Artifacts
    ------------------
        **Required Artifacts**

        Generally artifacts are passed directly to :class:`credoai.lens.Lens`, which
        handles evaluator setup. However, if you are using the evaluator directly, you
        will need to pass the following artifacts when instantiating the evaluator:

        - model: :class:`credoai.artifacts.Model`

    Parameters
    ----------
    model_info : Optional[dict]
        Information provided by the user that cannot be inferred by
        the model itself. The dictionary con contain any number of elements,
        a template can be provided by running the generate_template() method.

        The only restrictions are checked in a validation step:

        1. Some keys are protected because they are used internally
        2. Only basic python types are accepted as values

    """

    required_artifacts = {"model", "assessment_data"}

    def __init__(self, model_info: Optional[dict] = None):
        super().__init__()
        self.usr_model_info = model_info
        if not self.usr_model_info:
            self.usr_model_info = {}
        self._validate_usr_model_info()
        self.logger = global_logger

    def _setup(self):
        self.model_name = self.model.name
        self.model_internal = self.model.model_like
        self.model_type = type(self.model_internal)

    def _validate_arguments(self):
        check_existence(self.model, "model")

    def evaluate(self):
        basic = self._get_basic_info()
        res = self._get_model_params()
        if self.model.model_info["framework"] == "keras":
            self.results = [
                TableContainer(
                    self._generate_keras_results_table(res, basic),
                    **self.get_info(),
                )
            ]
            return self
        # Add user generated info
        self.usr_model_info = {k: v for k, v in self.usr_model_info.items() if v}
        # Get a sample of the data
        data_sample = self._get_dataset_sample()
        # Collate info
        res = {**basic, **res, **self.usr_model_info, **data_sample}
        # Format
        res, labels = self._add_entries_labeling(res)

        # Package into evidence
        self.results = [ModelProfilerContainer(res, **self.get_info(labels=labels))]
        return self

    @staticmethod
    def _generate_keras_results_table(res, basic):
        basic = DataFrame(basic, index=[0]).T
        opt_info = DataFrame(res["parameters"]["optimizer_info"], index=[0])
        opt_info.columns = [f"optimizer.{x}" for x in opt_info.columns]
        opt_info = opt_info.T
        choose = ["total_parameters", "trainable_parameters"]
        chosen = DataFrame(
            {k: v for k, v in res["parameters"].items() if k in choose}, index=[0]
        ).T

        output = concat([basic, chosen, opt_info])
        output = output.reset_index()
        output.columns = ["parameters", "values"]
        output.name = "model profile"
        return output

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

    def _get_dataset_sample(self) -> dict:
        """
        If assessment data is available get a sample of it.
        """
        try:
            data_sample = {
                "data_sample": self.assessment_data.X.sample(
                    3, random_state=42
                ).to_dict(orient="list")
            }
            return data_sample

        except:
            message = "No data found -> a sample of the data won't be included in the model card"
            self.logger.info(message)
            return {}

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
        if "keras" in str(self.model_type):
            return self._get_keras_model_params()
        self.logger.info(
            "Automatic model parameter inference not available for this model type."
        )
        return {}

    def _get_keras_model_params(self) -> dict:
        trainable_parameters = int(
            np.sum(
                [np.prod(v.get_shape()) for v in self.model_internal.trainable_weights]
            )
        )
        non_trainable_parameters = int(
            np.sum(
                [
                    np.prod(v.get_shape())
                    for v in self.model_internal.non_trainable_weights
                ]
            )
        )

        total_parameters = trainable_parameters + non_trainable_parameters

        opt_info = self.model_internal.optimizer.get_config()  # dict

        network_structure = DataFrame(
            [
                (
                    x.name,
                    re.sub(r"[^a-zA-Z]", "", str(type(x)).split(".")[-1]),
                    x.input_shape,
                    x.output_shape,
                    x.count_params(),
                )
                for x in self.model_internal.layers
            ],
            columns=["name", "layer_type", "input_shape", "output_shape", "parameters"],
        )

        return {
            "parameters": {
                "total_parameters": total_parameters,
                "trainable_parameters": trainable_parameters,
                "non_trainable_parameters": non_trainable_parameters,
                "network_architecture": network_structure,
                "optimizer_info": opt_info,
            }
        }

    def _get_sklearn_model_params(self) -> dict:
        """
        Get info from sklearn like models

        Returns
        -------
        dict
            Dictionary of info about the model
        """
        parameters = self.model_internal.get_params()
        model_library = self.model_type.__name__
        library = "sklearn"
        if hasattr(self.model_internal, "feature_names_in_"):
            feature_names = list(self.model_internal.feature_names_in_)
        else:
            feature_names = None
        return {
            "library": library,
            "model_library": model_library,
            "parameters": parameters,
            "feature_names": feature_names,
        }

    def _validate_usr_model_info(self):
        """
        Validate information that the user has inputted manually.

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
    def _add_entries_labeling(results: dict) -> tuple:
        """
        Takes the combined entries and format + create label to distinguish
        user generated ones.

        Parameters
        ----------
        results : dict
            Dictionary of all the entries

        Returns
        -------
        tuple
            DataFrame, dict
        """
        res = DataFrame.from_dict(results, orient="index")
        res.columns = ["results"]
        labels = {"user_generated": list(res.index[~res.index.isin(PROTECTED_KEYS)])}
        return res, labels

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
