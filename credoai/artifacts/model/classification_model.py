"""Model artifact wrapping any classification model"""
from .base_model import Model

from credoai.utils import global_logger

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np


PREDICT_PROBA_FRAMEWORKS = ["sklearn", "xgboost"]
MLP_FRAMEWORKS = ["keras"]

FRAMEWORK_VALIDATION_FUNCTIONS = {
    "sklearn": validate_sklearn_like,
    "xgboost": validate_sklearn_like,
    "keras": validate_keras_clf,
    # check on tensorflow generic, given validation strictness
}


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
            `predict` function that returns an array containing model outputs for each sample.
            It can also optionally have a `predict_proba` function that returns array containing
            the class label probabilities for each sample.
            If the supplied model_like is from the sklearn or xgboost framework, `predict` is assumed
            to return a column vector with a single value for each sample (i.e. thresholded predictions).
            If the supplied model_like is from the Keras framework, `predict` is assumed to return a matrix with
            probability values (i.e., with softmax applied; without argmax) similar to sklearn.predict_proba.
    """

    def __init__(self, name: str, model_like=None, tags=None):
        super().__init__(
            "Classification",
            ["predict", "predict_proba"],
            ["predict"],
            # TODO this will not work once we incorporate PyTorch
            # PyTorch allows callables and Module.forward()
            # predict not required
            name,
            model_like,
            tags,
        )

    def _validate_framework(self):
        try:
            FRAMEWORK_VALIDATION_FUNCTIONS[self.model_info["framework"]](
                self.model_like, self.model_info
            )
        except:
            message = "Provided model is from unsupported framework. "
            message += (
                "Lens behavior with unsupported modeling frameworks is undefined."
            )
            global_logger.warning(message)

    def _update_functionality(self):
        """Conditionally updates functionality based on framework"""
        # This needs to remain a big if-statement for now if we're going to keep
        # all classifiers in one class since we're making direct assignments to the class object
        if self.model_info["framework"] in PREDICT_PROBA_FRAMEWORKS:
            func = getattr(self, "predict_proba", None)
            if func and len(self.model_like.classes_) == 2:
                self.__dict__["predict_proba"] = lambda x: func(x)[:, 1]

        elif self.model_info["framework"] in MLP_FRAMEWORKS:
            # TODO change this to '__call__' when adding in general TF and PyTorch
            pred_func = getattr(self, "predict", None)
            if pred_func:
                self.__dict__["predict"] = lambda x: np.argmax(pred_func(x), axis=1)
                if self.model_like.layers[-1].output_shape == (None, 2):
                    self.__dict__["predict_proba"] = lambda x: pred_func(x)[:, 1]
                else:
                    self.__dict__["predict_proba"] = lambda x: pred_func(x)


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

    def __init__(
        self, name: str, predict_output=None, predict_proba_output=None, tags=None
    ):
        self.predict_output = predict_output
        self.predict_proba_output = predict_proba_output
        self.name = name
        self.tags = tags

    def predict(self, X=None):
        return self.predict_output

    def predict_proba(self, X=None):
        return self.predict_proba_output


def validate_sklearn_like(model_obj, model_info: dict):
    pass


def validate_keras_clf(model_obj, model_info: dict):
    # This is how Keras checks sequential too: https://github.com/keras-team/keras/blob/master/keras/utils/layer_utils.py#L219
    if not model_info["lib_name"] == "Sequential":
        message = "Only Keras models with Sequential architecture are supported at this time. "
        message += "Using Keras with other architechtures has undefined behavior."
        global_logger.warning(message)

    valid_final_layer = (
        isinstance(model_obj.layers[-1], layers.Dense)
        and model_obj.layers[-1].activation.__name__ == "softmax"
    )
    valid_final_layer = valid_final_layer or isinstance(
        model_obj.layers[-1], layers.Softmax
    )
    if not valid_final_layer:
        message = "Expected output layer to be either: tf.keras.layers.Softmax or "
        message += "tf.keras.layers.Dense with softmax activation."
        global_logger.warning(message)

    if len(model_obj.layers[-1].shape) != 2:
        message = "Expected 2D output shape for Keras.Sequetial model: (batch_size, n_classes) or (None, n_classes)"
        global_logger.warning(message)

    if model_obj.layers[-1].shape[0] is not None:
        message = "Expected output shape of Keras model to have arbitrary length"
        global_logger.warning(message)

    if model_obj.layers[-1].shape[1] < 2:
        message = "Expected classification output shape (batch_size, n_classes) or (None, n_classes). "
        message += "Continuous output univariate regression not supported"
        global_logger.warning(message)
        # TODO Add support for model-imposed argmax layer
        # https://stackoverflow.com/questions/56704669/keras-output-single-value-through-argmax
