import pandas as pd

import numpy as np

import tensorflow as tf

import inspect

from credoai.utils.common import ValidationError
from credoai.utils import global_logger


###############################################
# Checking artifact interactions (model + data)
###############################################


def check_model_data_consistency(model, data):
    # check predict
    # Keras always outputs numpy types (not Tensor or something else)
    if "predict" in model.__dict__.keys():
        try:
            mini_pred, batch_size = check_model_output(model.predict, data)
            if not mini_pred.size:
                # results for all presently supported models are ndarray results
                raise Exception("Empty return results from predict function.")
            if isinstance(data.y, np.ndarray) and (
                mini_pred.shape != data.y[:batch_size].shape
            ):
                raise Exception("Predictions have mismatched shape from provided y")
            if isinstance(data.y, pd.Series) and (
                mini_pred.shape != data.y.head(batch_size).shape
            ):
                raise Exception("Predictions have mismatched shape from provided y")
        except Exception as e:
            raise ValidationError(
                "Lens.model predictions do not match expected form implied by provided labels y.",
                e,
            )

    if "predict_proba" in model.__dict__.keys():
        try:
            mini_pred, batch_size = check_model_output(model.predict_proba, data)
            if not mini_pred.size:
                # results for all presently supported models are ndarray results
                raise Exception("Empty return results from predict function.")
            if len(mini_pred.shape) > 1 and mini_pred.shape[1] > 1:
                if np.sum(mini_pred[0]) != 1:
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
                "Lens.model predictions do not match expected form implied by provided labels y.",
                e,
            )

    if "compare" in model.__dict__.keys():
        pass


def check_model_output(fn, data, batch=1):
    mini_pred = None
    batch_size = batch
    if isinstance(data.X, np.ndarray):
        mini_pred = fn(np.reshape(data.X[0], (1, -1)))
    elif isinstance(data.X, pd.DataFrame):
        mini_pred = fn(data.X.head(1))
    elif isinstance(data.X, tf.Tensor):
        mini_pred = fn(data.X)
    elif isinstance(data.X, tf.data.Dataset) or inspect.isgeneratorfunction(data.X):
        one_batch = next(iter(data.X))
        batch_size = len(one_batch)
        if len(one_batch) >= 2:
            # batch is tuple
            # includes y and possibly weights; X is first
            mini_pred = fn(one_batch[0])
        else:
            # batch only contains X
            mini_pred = fn(one_batch)
    elif isinstance(data.X, tf.keras.utils.Sequence):
        mini_pred = fn(data.X.__getitem__(0))
        batch_size = len(mini_pred)
    else:
        message = "Input X is of unsupported type. Behavior is undefined. Proceed with caution"
        global_logger.warning(message)
        mini_pred = fn(data.X[0])

    return mini_pred, batch_size
