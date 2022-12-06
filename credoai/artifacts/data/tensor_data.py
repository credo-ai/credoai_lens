"""Data artifact wrapping any data in table format"""
from copy import deepcopy
from typing import Union

import numpy as np
import pandas as pd

from credoai.utils.common import ValidationError, check_array_like

from sklearn.utils import check_array

from .base_data import Data

import tensorflow as tf


class TensorData(Data):
    """Class wrapper around tensor data

    TensorData serves as an adapter between tensor datasets (e.g., images)
    and the evaluators in Lens.

    Currently supports in-memory data only.
    No support for generators to read samples from disk

    No support for sensitive feature handling at this time.

    Parameters
    -------------
    name : str
        Label of the dataset
    X : tensor-like with arbitrary dimension (n_samples, .., ..,)
        Dataset. Processed such that underlying structure is converted to a tf.Tensor object.
        First dimension is reserved for sample size.
    y : array-like of shape (n_samples, n_outputs)
        Outcome
    """

    def __init__(self, name: str, X=None, y=None):
        super().__init__("Tensor", name, X, y)

    def copy(self):
        """Returns a deepcopy of the instantiated class"""
        return TensorData(self.name, self.X, self.y)
        # calling copy.deepcopy is insufficient since deepcopy doesn't work on tf objects

    def _validate_X(self):
        """Validation of X inputs"""
        pass

    def _validate_y(self):
        """Validation of y inputs"""
        check_array_like(self.y)
        if self.X is not None and (len(self.X) != len(self.y)):
            raise ValidationError(
                "X and y are not the same length. "
                + f"X Length: {len(self.X)}, y Length: {len(self.y)}"
            )

    def _process_X(self, X):
        """Standardize X data

        Ensures X is a dataframe with string-named columns
        """
        if isinstance(X, tf.Tensor):
            return tf.identity(X)
            # Tensorflow object don't work with copy.deepcopy
        try:
            return tf.convert_to_tensor(deepcopy(X))
        except:
            raise ValidationError("X cannot be converted to tf.Tensor object.")

    def _process_y(self, y):
        """
        Standardize y data

        If y is convertible, convert y to Tensor object
        """
        if isinstance(y, tf.Tensor):
            return tf.identity(y)
            # Tensorflow object don't work with copy.deepcopy
        try:
            return tf.convert_to_tensor(deepcopy(y))
        except:
            raise ValidationError("y cannot be converted to tf.Tensor object.")

    def _validate_processed_X(self):
        """Validate processed X"""
        pass

    def _validate_processed_y(self):
        """Validate processed Y"""
        pass
