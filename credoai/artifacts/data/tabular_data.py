"""Data artifact wrapping any data in table format"""
from copy import deepcopy
from typing import Union

import numpy as np
import pandas as pd

from credoai.utils.common import ValidationError, check_array_like, check_pandas

from .base_data import Data


class TabularData(Data):
    """Class wrapper around tabular data

    TabularData serves as an adapter between tabular datasets
    and the evaluators in Lens. TabularData processes X

    Parameters
    -------------
    name : str
        Label of the dataset
    X : array-like of shape (n_samples, n_features)
        Dataset. Must be able to be transformed into a pandas DataFrame
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
        """Standardize X data

        Ensures X is a dataframe with string-named columns
        """
        temp = pd.DataFrame(X)
        # Column names are converted to strings, to avoid mixed types
        temp.columns = temp.columns.astype("str")
        # if X was not a pandas object, give it the index of sensitive features
        if not check_pandas(X) and self.sensitive_features is not None:
            temp.index = self.sensitive_features.index
        return temp

    def _process_y(self, y):
        """Standardize y data

        If y is convertible, convert y to pandas object with X's index
        """
        if isinstance(y, (pd.DataFrame, pd.Series)):
            return y
        pd_type = pd.Series
        if isinstance(y, np.ndarray) and y.ndim == 2 and y.shape[1] > 1:
            pd_type = pd.DataFrame
        if self.X is not None:
            y = pd_type(y, index=self.X.index)
        else:
            y = pd_type(y)
        y.name = "target"
        return y

    def _validate_X(self):
        check_array_like(self.X)

    def _validate_y(self):
        """Validation of Y inputs"""
        check_array_like(self.y)
        if self.X is not None and (len(self.X) != len(self.y)):
            raise ValidationError(
                "X and y are not the same length. "
                + f"X Length: {len(self.X)}, y Length: {len(self.y)}"
            )

    def _validate_processed_X(self):
        """Validate processed X"""
        if len(self.X.columns) != len(set(self.X.columns)):
            raise ValidationError("X contains duplicate column names")
        if not self.X.index.is_unique:
            raise ValidationError("X's index must be unique")

    def _validate_processed_y(self):
        """Validate processed Y"""
        if isinstance(self.X, (pd.Series, pd.DataFrame)) and not self.X.index.equals(
            self.y.index
        ):
            raise ValidationError("X and y must have the same index")
