"""Model artifact wrapping Keras classification model"""
from .base_model import Model

from credoai.utils import ValidationError

import numpy as np


class KerasClassificationModel(Model):
    """Class wrapper around Keras-based classification model to be assessed

    KerasClassificationModel serves as an adapter between arbitrary binary or multi-class
    classification models based on the Tensorflow Keras framework and the evaluations in Lens.
    Evaluations depend on ClassificationModel redefining `predict` and `predict_proba` since
    Keras `predict` is probabilistic by default and no thresholded classification exists.

    Parameters
    ----------
    name : str
        Label of the model
    model_like : model_like
        A Keras Sequential model
            It must have a `predict` function that returns an array containing the probability
            estimates for each sample.
    """

    def __init__(self, name: str, model_like=None, tags=None):
        super().__init__(
            "Keras Classification",
            ["predict"],
            ["predict"],
            name,
            model_like,
            tags,
        )
        self._validate_keras()

    def _validate_keras(self):
        if not self.model_like.__class__.__module__ == "keras.engine.sequential":
            raise ValidationError("Expected model from Keras Sequential framework")

    def _update_functionality(self):
        """Conditionally updates functionality based on framework"""
        pred_func = getattr(self, "predict", None)
        if pred_func:
            self.__dict__["predict"] = lambda x: np.argmax(pred_func(x), axis=1)
            if self.model_like.layers[-1].output_shape == (None, 2):
                self.__dict__["predict_proba"] = lambda x: pred_func(x)[:, 1]
            else:
                self.__dict__["predict_proba"] = lambda x: pred_func(x)
