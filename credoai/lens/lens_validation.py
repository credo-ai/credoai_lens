import numpy as np
import pandas as pd

try:
    tf_exists = True
    import tensorflow as tf
except ImportError:
    tf_exists = False

import inspect

from credoai.utils import global_logger
from credoai.utils.common import ValidationError

TOLERANCE = 1e-8

###############################################
# Checking artifact interactions (model + data)
###############################################


def check_model_data_consistency(model, data):
    """
    This validation function serves to check the compatibility of a model and dataset provided to Lens.
    For each outputting function (e.g., `predict`) supported by Lens, this validator checks
    whether the model supports that function. If so, the validator applies the outputting function to
    a small sample of the supplied dataset. The validator ensures the outputting function does not fail
    and, depending on the nature of the outputting function, performs light checking to verify the outputs
    (e.g. predictions) match the expected form, possibly including: data type and output shape.

    Parameters
    ----------
    model : artifacts.Model or a subtype of artifacts.Model
        A trained machine learning model wrapped as a Lens Model object
    data : artifacts.Data or a subtype of artifacts.Data
        The dataset that will be assessed by Lens evaluators, wrapped as a Lens Data object
    """
    # check predict
    # Keras always outputs numpy types (not Tensor or something else)
    if "predict" in model.__dict__.keys() and data.y is not None and data.X is not None:
        try:
            mini_pred, batch_size = check_prediction_model_output(model.predict, data)
            if not mini_pred.size:
                # results for all presently supported models are ndarray results
                raise Exception("Empty return results from predict function.")
            if len(mini_pred.shape) > 1:
                # check that output size per sample matches up
                if isinstance(data.y, np.ndarray) and (
                    mini_pred.shape[1:] != data.y.shape[1:]
                ):
                    raise Exception("Predictions have mismatched shape from provided y")
                elif isinstance(data.y, pd.Series) and (
                    mini_pred.shape[1:] != data.y.head(batch_size).shape[1:]
                ):
                    raise Exception("Predictions have mismatched shape from provided y")
        except Exception as e:
            raise ValidationError(
                "Lens.model predictions do not match expected form implied by provided labels y.",
                e,
            )

    if (
        "predict_proba" in model.__dict__.keys()
        and data.y is not None
        and data.X is not None
    ):
        try:
            mini_pred, batch_size = check_prediction_model_output(
                model.predict_proba, data
            )
            if not mini_pred.size:
                # results for all presently supported models are ndarray results
                raise Exception("Empty return results from predict_proba function.")
            if len(mini_pred.shape) > 1 and mini_pred.shape[1] > 1:
                # Adding tolerance to account for rounding errors.
                if abs(1 - np.sum(mini_pred[0])) >= TOLERANCE:
                    raise Exception(
                        "`predict_proba` outputs invalid. Per-sample outputs should sum to 1."
                    )
            else:
                if mini_pred[0] >= 1:
                    raise Exception(
                        "`predict_proba` outputs invalid. Binary outputs should be <= 1."
                    )
        except Exception as e:
            raise ValidationError(
                "Lens.model outputs do not match expected form implied by provided labels y.",
                e,
            )

    if "compare" in model.__dict__.keys() and data.pairs is not None:
        try:
            comps, batch_size = check_comparison_model_output(model.compare, data)
            if type(comps) != list:
                raise Exception(
                    "Comparison function expected to produce output of type list."
                )
            if not comps:
                # results are expected to be a list
                raise Exception("Empty return results from compare function.")

        except Exception as e:
            raise ValidationError(
                "Lens.model outputs do not match expected form implied by provided labels y.",
                e,
            )


def check_prediction_model_output(fn, data, batch_in: int = 1):
    """
    Helper function for prediction-type models. For use with `check_model_data_consistency`.

    This helper does the work of actually obtaining predictions (from `predict` or `predict_proba`;
    flexible enough for future use with functions that have similar behavior) and verifying that the
    outputs are consistent with expected outputs specified by the ground truth `data.y`.

    Parameters
    ----------
    fn : function object
        The prediction-generating function for the model passed to `check_model_data_consistency`
    data : artifacts.Data or a subtype of artifacts.Data
        The dataset that will be assessed by Lens evaluators, wrapped as a Lens Data object
    batch : an integer
        The size of the sample prediction. We do not perform prediction on the entire `data.X` object
        since this could be large and computationally expensive.
    """
    mini_pred = None
    batch_out = batch_in
    if isinstance(data.X, np.ndarray):
        mini_pred = fn(np.reshape(data.X[0], (1, -1)))
    elif isinstance(data.X, (pd.DataFrame, pd.Series)):
        mini_pred = fn(data.X.head(batch_in))
    elif tf_exists and isinstance(data.X, tf.Tensor):
        mini_pred = fn(data.X)
    elif (
        tf_exists and isinstance(data.X, tf.data.Dataset)
    ) or inspect.isgeneratorfunction(data.X):
        one_batch = next(iter(data.X))
        batch_out = len(one_batch)
        if len(one_batch) >= 2:
            # batch is tuple
            # includes y and possibly weights; X is first
            mini_pred = fn(one_batch[0])
        else:
            # batch only contains X
            mini_pred = fn(one_batch)
    elif tf_exists and isinstance(data.X, tf.keras.utils.Sequence):
        mini_pred = fn(data.X.__getitem__(0))
        batch_out = len(mini_pred)
    else:
        message = "Input X is of unsupported type. Behavior is undefined. Proceed with caution"
        global_logger.warning(message)
        mini_pred = fn(data.X[0])

    return np.array(mini_pred), batch_out


def check_comparison_model_output(fn, data, batch_in=1):
    """
    Helper function for comparison-type models. For use with `check_model_data_consistency`.

    This helper does the work of actually obtaining comparisons (from `compare`; flexible enough
    for future use with functions that have similar behavior) to verify the function does not fail.

    Parameters
    ----------
    fn : function object
        The comparison-generating function for the model passed to `check_model_data_consistency`
    data : artifacts.Data or a subtype of artifacts.Data
        The dataset that will be assessed by Lens evaluators, wrapped as a Lens Data object
    batch : an integer
        The size of the sample prediction. We do not perform prediction on the entire `data.pairs`
        object since this could be large and computationally expensive.
    """
    comps = None
    batch_out = batch_in
    if isinstance(data.pairs, pd.DataFrame):
        # should always pass for ComparisonData, based on checks in that wrapper. Nevertheless...
        comps = fn(data.pairs.head(batch_in))
    else:
        message = "Input pairs are of unsupported type. Behavior is undefined. Proceed with caution"
        global_logger.warning(message)
        comps = fn(data.pairs[:batch_in])

    return comps, batch_out
