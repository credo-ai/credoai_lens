"""Model artifact wrapping Keras classification model"""
from .base_model import Model

from credoai.utils import ValidationError

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class KerasClassificationModel(Model):
    """Class wrapper around Keras-based classification model to be assessed

    KerasClassificationModel serves as an adapter between arbitrary binary or multi-class
    classification models based on the Tensorflow Keras framework and the evaluations in Lens.
    Evaluations depend on ClassificationModel redefining `predict` and `predict_proba` since
    Keras `predict` is probabilistic by default and no thresholded classification exists.

    Assumes use of tf.models.Sequential linear layer grouping.

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
        self._validate_sequential()
        self._validate_dense()

    def _validate_keras(self):
        if not self.model_info["framework"] == "keras":
            raise ValidationError("Expected model from Keras framework")

    def _validate_sequential(self):
        # This is how Keras checks sequential too: https://github.com/keras-team/keras/blob/master/keras/utils/layer_utils.py#L219
        if not self.model_info["lib_name"] == "Sequential":
            raise ValidationError("Expected model to use Sequential layer grouping")

    def _validate_dense(self):
        if not isinstance(self.model_like.layers[-1], layers.Dense):
            raise ValidationError(
                "Expected output layer to be of type tf.keras.layers.Dense"
            )
        if len(self.model_like.layers[-1].shape) != 2:
            raise ValidationError(
                "Expected output 2D output shape: (batch_size, n_classes) or (None, n_classes)"
            )
        if self.model_like.layers[-1].shape[0] is not None:
            raise ValidationError("Expected output shape to have arbitrary length")
        if self.model_like.layers[-1].shape[1] < 2:
            raise ValidationError(
                "Expected classification output shape (batch_size, n_classes) or (None, n_classes). Continuous output univariate regression not supported"
            )
            # TODO Add support for model-imposed argmax layer
            # https://stackoverflow.com/questions/56704669/keras-output-single-value-through-argmax

    def _update_functionality(self):
        """Conditionally updates functionality based on framework"""
        pred_func = getattr(self, "predict", None)
        if pred_func:
            self.__dict__["predict"] = lambda x: np.argmax(pred_func(x), axis=1)
            if self.model_like.layers[-1].output_shape == (None, 2):
                self.__dict__["predict_proba"] = lambda x: pred_func(x)[:, 1]
            else:
                self.__dict__["predict_proba"] = lambda x: pred_func(x)
