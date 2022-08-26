from .model import Model


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
        Array containing the predicted class labels for each sample.
    predict_output : array
        Array containing the class labels probabilities for each sample.
    """

    def __init__(self, predict_output=None, predict_proba_output=None):
        self.predict_output = predict_output
        self.predict_proba_output = predict_proba_output

    def predict(self, X):
        return self.prediction_output

    def predict_proba(self, X):
        return self.prediction_proba_output


class ClassificationModel(Model):
    """Class wrapper around classification model to be assessed

    ClassificationModel serves as an adapter between arbitrary binary or multi-class
    classification models and the evaluations in Lens. Evaluations depend on
    ClassificationModel instantiating certain methods.

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

    def __init__(
        self,
        name: str,
        model_like=None,
    ):
        super().__init__("Classification", name, model_like)

    def predict(self):
        try:
            return self.model_like.predict()
        except:
            AttributeError

    def predict_proba(self):
        try:
            return self.model_like.predict_proba()
        except:
            return
