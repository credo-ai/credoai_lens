"""Data artifact wrapping any data in table format"""
from copy import deepcopy
from typing import Union

import numpy as np
import pandas as pd
from credoai.utils.common import ValidationError

from .base_data import Data


class TabularData(Data):
    """Class wrapper around tabular data

    TabularData serves as an adapter between tabular datasets
    and the evaluators in Lens.

    Parameters
    -------------
    name : str
        Label of the dataset
    X : array-like of shape (n_samples, n_features)
        Dataset
    y : array-like of shape (n_samples, n_outputs)
        Outcome
    sensitive_features : pd.Series, pd.DataFrame, optional
        Sensitive Features, which will be used for disaggregating performance
        metrics. This can be the feature you want to perform segmentation analysis on, or
        a feature related to fairness like 'race' or 'gender'. Sensitive Features *must*
        be categorical features.
    sensitive_intersections : bool, list
        Whether to add intersections of sensitive features. If True, add all possible
        intersections. If list, only create intersections from specified sensitive features.
        If False, no intersections will be created. Defaults False
    """

    def __init__(
        self,
        name: str,
        X=None,
        y=None,
        sensitive_features=None,
        sensitive_intersections: Union[bool, list] = False,
    ):
        super().__init__(
            "Tabular", name, X, y, sensitive_features, sensitive_intersections
        )

    def copy(self):
        """Returns a deepcopy of the instantiated class"""
        return deepcopy(self)

    def _process_X(self, X):
        """Standardize X data"""
        temp = pd.DataFrame(X)
        # Column names are converted to strings, to avoid mixed types
        temp.columns = temp.columns.astype("str")
        return temp

    def _process_y(self, y):
        """Standardize y data"""
        # if X is pandas object, and y is convertible, convert y to
        # pandas object with X's index
        if isinstance(y, (np.ndarray, list)):
            pd_type = pd.Series
            if isinstance(y, np.ndarray) and y.ndim == 2 and y.shape[1] > 1:
                pd_type = pd.DataFrame
            if self.X is not None:
                y = pd_type(y, index=self.X.index, name="target")
            else:
                y = pd_type(y, name="target")
        return y

    def _validate_y(self):
        if self.X is not None:
            """Basic validation for y"""
            if len(self.X) != len(self.y):
                raise ValidationError(
                    "X and y are not the same length. "
                    + f"X Length: {len(self.X)}, y Length: {len(self.y)}"
                )

    def _validate_processed_y(self):
        if isinstance(self.X, (pd.Series, pd.DataFrame)) and not self.X.index.equals(
            self.y.index
        ):
            raise ValidationError("X and y must have the same index")

    def _validate_X(self):
        """Basic validation for X"""
        # Validate that the data column names are unique
        if len(self.X.columns) != len(set(self.X.columns)):
            raise ValidationError("X contains duplicate column names")
        if len(self.X.index) != len(set(self.X.index)):
            raise ValidationError("X's index cannot contain duplicates")
