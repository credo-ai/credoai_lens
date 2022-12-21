"""Model artifact wrapping any classification model"""
from .base_model import Model

from credoai.utils import global_logger
from credoai.utils.model_utils import validate_sklearn_like, validate_keras_clf

import numpy as np

from .constants_model import SKLEARN_LIKE_FRAMEWORKS

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
            If the supplied model_like is from the Keras framework, the assumed form of `predict` outputs
            depends on the final-layer activation. If softmax, wrapper assumes return is a matrix with
            probability values (i.e., without argmax) similar to sklearn.predict_proba. If sigmoid, wrapper
            assumes return is a column vector with label predictions.
    """

    def __init__(self, name: str, model_like=None, tags=None):
        super().__init__(
            "CLASSIFICATION",
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

    def __post_init__(self):
        """Conditionally updates functionality based on framework"""
        # This needs to remain a big if-statement for now if we're going to keep
        # all classifiers in one class since we're making direct assignments to the class object
        if self.model_info["framework"] in SKLEARN_LIKE_FRAMEWORKS:
            func = getattr(self, "predict_proba", None)
            if len(self.model_like.classes_) == 2:
                self.type = "BINARY_CLASSIFICATION"
                # if binary, replace probability array with one-dimensional vector
                if func:
                    self.__dict__["predict_proba"] = lambda x: func(x)[:, 1]
            else:
                self.type = "MULTICLASS_CLASSIFICATION"

        elif self.model_info["framework"] in MLP_FRAMEWORKS:
            # TODO change this to '__call__' when adding in general TF and PyTorch
            pred_func = getattr(self, "predict", None)
            if pred_func:
                if self.model_like.layers[-1].output_shape == (None, 1):
                    self.__dict__["predict"] = pred_func
                else:
                    self.__dict__["predict"] = lambda x: np.argmax(pred_func(x), axis=1)

                if self.model_like.layers[-1].output_shape == (None, 2):
                    self.__dict__["predict_proba"] = lambda x: pred_func(x)[:, 1]
                elif (
                    len(self.model_like.layers[-1].output_shape) == 2
                    and self.model_like.layers[-1].output_shape[1] > 2
                ):
                    self.__dict__["predict_proba"] = pred_func
                else:
                    pass
                    # predict_proba is not valid (for now)


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
