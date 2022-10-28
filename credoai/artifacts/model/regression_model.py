"""Model artifact wrapping any regression model"""
from .base_model import Model


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
        A binary or multi-class classification model or pipeline. It must have a
            `predict` function that returns array containing the class labels for each sample.
            It can also optionally have a `predict_proba` function that returns array containing
            the class labels probabilities for each sample.
    """

    def __init__(self, name: str, model_like=None, tags=None):
        super().__init__("Regression", ["predict"], ["predict"], name, model_like, tags)


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

    def __init__(self, predict_output=None):
        self.predict_output = predict_output

    def predict(self, X=None):
        return self.predict_output
