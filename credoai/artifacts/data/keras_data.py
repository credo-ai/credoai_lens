"""Data artifact wrapping any data in table format"""
import pandas as pd

from credoai.utils.common import ValidationError, check_array_like

from sklearn.utils import check_array

from .base_data import Data

import numpy as np


class KerasData(Data):
    """Class wrapper around array data expected by Keras

    KerasData serves as an adapter between array datasets and the evaluators in Lens.

    Currently supports in-memory data only.
    No support for generators to read samples from disk

    No support for sensitive feature handling at this time.

    Parameters
    -------------
    name : str
        Label of the dataset
    X : array-like with arbitrary dimension (n_samples, .., ..,)
        Dataset. Processed such that underlying structure is converted to a np.ndarray object.
        First dimension is reserved for sample size.
    y : array-like of shape (n_samples, n_outputs)
        Outcome
    """

    def __init__(self, name: str, X=None, y=None):
        super().__init__("Data", name, X, y)

    def copy(self):
        """Returns a deepcopy of the instantiated class"""
        return KerasData(self.name, self.X, self.y)
        # calling copy.deepcopy is insufficient since deepcopy doesn't work on tf objects

    def _validate_X(self):
        """Validation of X inputs"""
        check_array(self.X, ensure_2d=False, allow_nd=True)

    def _validate_y(self):
        """Validation of y inputs"""
        check_array_like(self.y)
        if self.X is not None and (len(self.X) != len(self.y)):
            raise ValidationError(
                "X and y are not the same length. "
                + f"X Length: {len(self.X)}, y Length: {len(self.y)}"
            )

    def _process_X(self, X):
        """Standardize X data"""
        return X

    def _process_y(self, y):
        """
        Standardize y data
        """
        return pd.Series(y)

    def _validate_processed_X(self):
        """Validate processed X"""
        pass

    def _validate_processed_y(self):
        """Validate processed Y"""
        pass
