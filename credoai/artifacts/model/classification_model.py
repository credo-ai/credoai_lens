"""Model artifact wrapping any classification model"""
from .base_model import Model

from credoai.utils import global_logger

import numpy as np

from sklearn.utils import check_array

from .constants_model import (
    SKLEARN_LIKE_FRAMEWORKS,
    MLP_FRAMEWORKS,
    FRAMEWORK_VALIDATION_FUNCTIONS,
)


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
            depends on the final-layer activation.
            If the final layer is softmax, this wrapper assumes the
            return value is a is a matrix with shape (n_samples, n_classes) corresponding to probability
            values (i.e., without argmax), similar to sklearn.predict_proba. The wrapper applies argmax
            where necessary to obtain discrete labels.
            If the final layer is sigmoid, this wrapper assumes the return value is an (n_samples, 1)
            column vector with per-sample probabilities. The wrapper rounds (.5 as default threshold)
            values where necessary to obtain discrete labels.
    tags : optional
        Additional metadata to add to model
        E.g., {'model_type': 'binary_classification'}
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
            message = """Provided model is from unsupported framework. 
            Lens behavior has not been tested or assured with unsupported modeling frameworks."""
            global_logger.warning(message, message)

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
                    # Assumes sigmoid -> probabilities need to be rounded
                    self.__dict__["predict"] = lambda x: pred_func(x).round()
                else:
                    # Assumes softmax -> probabilities need to be argmaxed
                    self.__dict__["predict"] = lambda x: np.argmax(pred_func(x), axis=1)

                if self.model_like.layers[-1].output_shape == (None, 2):
                    self.__dict__["predict_proba"] = lambda x: pred_func(x)[:, 1]
                elif (
                    len(self.model_like.layers[-1].output_shape) == 2
                    and self.model_like.layers[-1].output_shape[1] == 1
                ):
                    # Sigmoid -> needs to be (n_samples, ) to work with sklearn metrics
                    self.__dict__["predict_proba"] = lambda x: np.reshape(
                        pred_func(x), (-1, 1)
                    )
                elif (
                    len(self.model_like.layers[-1].output_shape) == 2
                    and self.model_like.layers[-1].output_shape[1] > 2
                ):
                    self.__dict__["predict_proba"] = pred_func
                else:
                    pass
                    # predict_proba is not valid (for now)

        elif self.model_info["framework"] == "credoai":
            # Functionality for DummyClassifier
            self.model_like = getattr(self.model_like, "model_like", None)
            # If the dummy model has a model_like specified, reassign
            # the classifier's model_like attribute to match the dummy's
            # so that downstream evaluators (ModelProfiler) can use it

            # Predict and Predict_Proba should already be specified


class DummyClassifier:
    """Class wrapper around classification model predictions

    This class can be used when a classification model's outputs have been precomputed.
    The output include the array containing the predicted class labels and/or the array
    containing the class labels probabilities.
    Wrap the outputs with this class into a dummy classifier and pass it as
    the model to `ClassificationModel`.

    Parameters
    ----------
    name : str
        Label of the model
    model_like : model_like, optional
        While predictions are pre-computed, the model object, itself, may be of use for
        some evaluations (e.g. ModelProfiler).
    predict_output : array, optional
        Array containing per-sample class labels
            Corresponds to sklearn-like `predict` output
            For NN frameworks (Keras.predict, tf.__call__, torch.foward, etc.), this input assumes argmax
            has been applied to the outputs so that they are discrete valued labels
    predict_proba_output : array, optional
        Array containing the per-sample class probabilities
            Corresponds to sklearn-like `predict_proba` output
            For NN frameworks (Keras.predict, etc.) this input assumes no post-processing after a
            final-layer softmax (general) or sigmoid (binary only) activation

    """

    def __init__(
        self,
        name: str,
        model_like=None,
        predict_output=None,
        predict_proba_output=None,
        tags=None,
    ):
        self.model_like = model_like
        if predict_output is not None:
            self.predict_output = check_array(
                predict_output, ensure_2d=False, allow_nd=True
            )
        if predict_proba_output is not None:
            self.predict_proba_output = check_array(
                predict_proba_output, ensure_2d=False, allow_nd=True
            )
        self.name = name
        self.tags = tags

    def predict(self, X=None):
        return self.predict_output

    def predict_proba(self, X=None):
        return self.predict_proba_output
