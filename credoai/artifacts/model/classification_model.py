"""Model artifact wrapping any classification model"""
from .base_model import Model

from credoai.utils import global_logger
from credoai.utils.model_utils import clf_handle_keras, clf_handle_torch

import numpy as np
import pandas as pd

from sklearn.utils import check_array


from .constants_model import (
    SKLEARN_LIKE_FRAMEWORKS,
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
            `predict`-like function that returns an array containing model outputs for each sample.
            It can also optionally have a `predict_proba` function that returns array containing
            the class label probabilities for each sample.

            If the supplied model_like is from the sklearn or xgboost framework, `predict` is assumed
            to return a column vector with a single value for each sample (i.e. thresholded predictions).

            If the supplied model_like is from the Keras framework, the assumed form of `predict` outputs
            depends on the final-layer activation. *Only Keras Sequential-type models are supported at this time.*
            If the final layer is softmax, this wrapper assumes the
            return value is a is a matrix with shape (n_samples, n_classes) corresponding to probability
            values (i.e., without argmax), similar to sklearn.predict_proba. The wrapper applies argmax
            where necessary to obtain discrete labels.
            If the final layer is sigmoid, this wrapper assumes the return value is an (n_samples, 1)
            column vector with per-sample probabilities. The wrapper rounds (.5 as default threshold)
            values where necessary to obtain discrete labels.

            If the supplied model_like is a PyTorch model, *the user must specify the expected input shape
            via a class attribute `input_shape` of type tuple, and the activation function for the  final layer
            of the model via a class attribute `output_activation` of type string.* Supported values for
            classification are "softmax" and "sigmoid"
            For exmaple,
            class NeuralNetwork(nn.Module):
                def __init__(self):
                    ...
                    self.linear_relu_stack = nn.Sequential(
                        ...
                        nn.Softmax(),
                    )
                    self.output_activation = "softmax"

                def forward(self, x):
                    ...
                    logits = self.linear_relu_stack(x)
                    return logits
            If no `output_activation` string is specified, Lens assumes "sigmoid" and outputs a warning.
            The model `input_shape` is required for validation and to allow Lens to automatically determine
            the expected output shape (e.g. discrete value predictions vs. probabilities).

            For custom model_like objects, users may optionally specify a `framework_like` attribute
            of type string. framework_like serves as a flag to enable expected functionality to carry over
            from an external framework to Lens. Presently "sklearn", "xgboost", and "keras" are supported.
            The former two serve as a flags to notify Lens that model_like respects sklearn's predict API
            (and the predict_proba API, if relevant). The latter serves as a flag to Lens that model_like
            respects Keras's predict API with either a sigmoid or softmax final layer.

    tags : optional
        Additional metadata to add to model
        E.g., {'model_type': 'binary_classification'}
    """

    def __init__(self, name: str, model_like=None, tags=None):
        super().__init__(
            "CLASSIFICATION",
            ["predict", "predict_proba", "call", "forward"],
            [],
            name,
            model_like,
            tags,
        )

    def _validate_framework(self):
        """Validates the model framework and logs a warning if unsupported."""
        try:
            FRAMEWORK_VALIDATION_FUNCTIONS[self.model_info["framework"]](
                self.model_like, self.model_info
            )
        except:
            message = """Provided model is from unsupported framework. 
            Lens behavior has not been tested or assured with unsupported modeling frameworks."""
            global_logger.warning(message)

    def __post_init__(self):
        """Conditionally updates functionality based on framework"""
        # This needs to remain a big if-statement for now if we're going to keep
        # all classifiers in one class since we're making direct assignments to the class object

        # Handle sklearn-like frameworks
        if self.model_info["framework"] in SKLEARN_LIKE_FRAMEWORKS:
            func = getattr(self, "predict_proba", None)
            if len(self.model_like.classes_) == 2:
                if all(self.model_like.classes_ == [0, 1]):
                    self.type = "BINARY_CLASSIFICATION"
                    # if binary, replace probability array with one-dimensional vector
                    if func:
                        self.__dict__["predict_proba"] = lambda x: func(x)[:, 1]
                else:
                    self.type = "MULTICLASS_CLASSIFICATION"
                    message = f"\nThe model was considered of type {self.type}.\n"
                    message += f"Classes detected: {list(self.model_like.classes_)}\n"
                    message += f"Expected for binary classification: [0, 1]"
                    global_logger.warning(message)
            else:
                self.type = "MULTICLASS_CLASSIFICATION"

        # Handle Keras models
        elif self.model_info["framework"] == "keras":
            (
                self.__dict__["predict"],
                self.__dict__["predict_proba"],
                self.type,
            ) = clf_handle_keras(self.model_like)

        # Handle PyTorch models
        elif self.model_info["framework"] == "torch":
            (
                self.__dict__["predict"],
                self.__dict__["predict_proba"],
                self.type,
            ) = clf_handle_torch(self.model_like)

        # Handle Lens DummyClassifier models
        elif self.model_info["framework"] == "credoai":
            # Functionality for DummyClassifier
            if self.model_like.model_like is not None:
                self.model_like = self.model_like.model_like
            # If the dummy model has a model_like specified, reassign
            # the classifier's model_like attribute to match the dummy's
            # so that downstream evaluators (ModelProfiler) can use it

            self.type = self.model_like.type
            # DummyClassifier model type is set in the constructor based on whether it
            # is binary or multiclass

            # Predict and Predict_Proba should already be specified

        # Ensure predict function is specified for custom models
        if "predict" not in self.__dict__:
            raise Exception(
                "`predict` function required for custom model {self.name}. None specified."
            )


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
    binary_clf : bool, optional, default = True
        Type of classification model.
            Used when wrapping with ClassificationModel.
            If binary == True, ClassificationModel.type will be set to `BINARY_CLASSIFICATION',
            which enables use of binary metrics.
            If binary == False, ClassificationModel.type will be set to 'MULTICLASS_CLASSIFICATION',
            and use those metrics.
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
        binary_clf=True,
        predict_output=None,
        predict_proba_output=None,
        tags=None,
    ):
        self.model_like = model_like
        self._build_functionality("predict", predict_output)
        self._build_functionality("predict_proba", predict_proba_output)
        self.name = name
        self.tags = tags
        self.type = (
            "BINARY_CLASSIFICATION" if binary_clf else "MULTICLASS_CLASSIFICATION"
        )

    def _wrap_array(self, array):
        """Wraps the array into a lambda function that ignores its input and returns the array."""
        return lambda X=None: array
        # Keeping X as an optional argument to maintain potential backward compatibility
        # Some uses of DummyClassifier may use predict() with no argument

    def _build_functionality(self, function_name, array):
        """
        Builds a function with the given name and wraps the provided array using the _wrap_array method.
        The created function will be added to the class instance dictionary.
        """
        if array is not None:
            if isinstance(array, pd.Series):
                if not len(array):
                    raise Exception("Provided series for y_pred or y_prob is empty")
                if array.dtype is np.number and np.isinf(array).any():
                    raise Exception(
                        "Provided series for y_pred or y_prob contains infinite values"
                    )
            else:
                array = check_array(array, ensure_2d=False, allow_nd=True)
            self.__dict__[function_name] = self._wrap_array(array)
