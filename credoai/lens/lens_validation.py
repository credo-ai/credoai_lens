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
            mini_pred = None
            batch_size = 1
            if isinstance(data.X, np.ndarray):
                mini_pred = model.predict(np.reshape(data.X[0], (1, -1)))
            elif isinstance(data.X, pd.DataFrame):
                mini_pred = model.predict(data.X.head(1))
            elif isinstance(data.X, tf.Tensor):
                mini_pred = model.predict(data.X)
            elif isinstance(data.X, tf.data.Dataset) or inspect.isgeneratorfunction(
                data.X
            ):
                one_batch = next(iter(data.X))
                batch_size = len(one_batch)
                if len(one_batch) >= 2:
                    # batch is tuple
                    # includes y and possibly weights; X is first
                    mini_pred = model.predict(one_batch[0])
                else:
                    # batch only contains X
                    mini_pred = model.predict(one_batch)
            elif isinstance(data.X, tf.keras.utils.Sequence):
                mini_pred = model.predict(data.X.__getitem__(0))
                batch_size = len(mini_pred)
            else:
                message = "Input X is of unsupported type. Behavior is undefined. Proceed with caution"
                global_logger.warning(message)
                mini_pred = model.predict(data.X[0])

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
        pass
        # does it make sense to check this?

    if "compare" in model.__dict__.keys():
        pass
