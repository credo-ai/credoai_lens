"""Model artifact wrapping any regression model"""
from .base_model import Model

from .constants_model import (
    SKLEARN_LIKE_FRAMEWORKS,
    MLP_FRAMEWORKS,
    FRAMEWORK_VALIDATION_FUNCTIONS,
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
        A continuous output regression model or pipeline. It must have a
            `predict` function that returns array containing the predicted outcomes for each sample.
    """

    def __init__(self, name: str, model_like=None, tags=None):
        super().__init__(
            "REGRESSION", ["predict", "call", "forward"], [], name, model_like, tags
        )

    def __post_init__(self):
        if self.model_info["framework"] in MLP_FRAMEWORKS:
            pass
            # replace call/forward with predict


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
