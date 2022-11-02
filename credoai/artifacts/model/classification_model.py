"""Model artifact wrapping any classification model"""
from .base_model import Model


class ClassificationModel(Model):
    """Class wrapper around classification model to be assessed

    ClassificationModel serves as an adapter between arbitrary binary or multi-class
    classification models and the evaluations in Lens. Evaluations depend on
    ClassificationModel instantiating `predict` and (optionally) `predict_proba`

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
        super().__init__(
            "Classification",
            ["predict", "predict_proba"],
            ["predict"],
            name,
            model_like,
            tags,
        )

    def _update_functionality(self):
        """Conditionally updates functionality based on framework"""
        if self.model_info["framework"] == "sklearn":
            func = getattr(self, "predict_proba", None)
            if func and len(self.model_like.classes_) == 2:
                self.__dict__["predict_proba"] = lambda x: func(x)[:, 1]


class DummyClassifier:
    """Class wrapper around classification model predictions

    This class can be used when a classification model is not available but its outputs are.
        The output include the array containing the predicted class labels and/or the array
        containing the class labels probabilities.
        Wrap the outputs with this class into a dummy classifier and pass it as
        the model to `ClassificationModel`.

    Parameters
    ----------
    predict_output : array
        Array containing the output of a model's "predict" method
    predict_proba_output : array
        Array containing the output of a model's "predict_proba" method
    """

    def __init__(self, predict_output=None, predict_proba_output=None):
        self.predict_output = predict_output
        self.predict_proba_output = predict_proba_output

    def predict(self, X=None):
        return self.predict_output

    def predict_proba(self, X=None):
        return self.predict_proba_output
