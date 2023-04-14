"""Model artifact wrapping any regression model"""
from .base_model import Model

from credoai.utils.model_utils import reg_handle_torch

try:
    import torch
except ImportError:
    print(
        "Torch not loaded. Torch models will not be wrapped properly if supplied to ClassificationModel"
    )


class RegressionModel(Model):
    """Class wrapper around classification model to be assessed

    RegressionModel serves as an adapter between arbitrary regression models and the
    evaluations in Lens. Evaluations depend on
    RegressionModel instantiating `predict`

    Parameters
    ----------
    name : str
        Label of the model
    model_like : model_like
        A continuous output regression model or pipeline.
            Model must have a `predict`-like function which returns an np.ndarray containing predicted outcomes.
            Sklearn, Keras-based models supported natively and Lens uses their predict functions.
            Torch-based models: If the user has defined a `predict` function, Lens uses that. Otherwise,
                Lens will attempt to use the `forward` function.
            All other models must have a `predict` function specified.
    """

    def __init__(self, name: str, model_like=None, tags=None):
        super().__init__(
            "REGRESSION", ["predict", "call", "forward"], [], name, model_like, tags
        )

    def __post_init__(self):
        """Conditionally updates functionality based on framework"""
        # SKlearn and Keras both have built-in predict functions and so we don't need special handling
        if self.model_info["framework"] == "torch":
            if not hasattr(self, "predict"):
                # If user has custom-specified a predict function, we will ignore `forward`
                self.__dict__["predict"] = reg_handle_torch(self.model_like)

        elif self.model_info["framework"] == "credoai":
            # Functionality for DummyRegression
            if self.model_like.model_like is not None:
                self.model_like = self.model_like.model_like
            # If the dummy model has a model_like specified, reassign
            # the model_like attribute to match the dummy's
            # so that downstream evaluators (ModelProfiler) can use it

        # This check is newly necessary, since `predict` is no longer required in the validation step
        # but _a_ predict function is needed by the end of initialization.
        if "predict" not in self.__dict__:
            raise Exception(
                "`predict` function required for custom model {self.name}. None specified."
            )


class DummyRegression:
    """Class wrapper around regression model predictions

    This class can be used when a regression model is not available but its outputs are.
    The output include the array containing the predicted class labels and/or the array
    containing the class labels probabilities.
    Wrap the outputs with this class into a dummy classifier and pass it as
    the model to `RegressionModel`.

    Parameters
    ----------
    predict_output : array
        Array containing the output of a model's "predict" method
    """

    def __init__(self, name: str, predict_output=None, tags=None):
        self.predict_output = predict_output
        self.name = name
        self.tags = tags

    def predict(self, X=None):
        return self.predict_output
