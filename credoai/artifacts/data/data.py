"""Data used by Lens"""
# Data is a lightweight wrapper that stores data

import itertools
from abc import ABC, abstractmethod
from tkinter import N
from typing import Union

import pandas as pd
from credoai.utils.common import ValidationError
from sklearn.utils.multiclass import type_of_target


class Data(ABC):
    """Class wrapper around data-to-be-assessed

    Data is passed to Lens for certain assessments.

    Data serves as an adapter between datasets
    and the evaluators in Lens.

    Parameters
    -------------
    type : str
        Type of the dataset
    name : str
        Label of the dataset
    X : to-be-defined by children
        Dataset
    y : to-be-defined by children
        Outcome
    sensitive_features : pd.Series, pd.DataFrame, optional
        Sensitive Features, which will be used for disaggregating performance
        metrics. This can be the columns you want to perform segmentation analysis on, or
        a feature related to fairness like 'race' or 'gender'
    sensitive_intersections : bool, list
        Whether to add intersections of sensitive features. If True, add all possible
        intersections. If list, only create intersections from specified sensitive features.
        If False, no intersections will be created. Defaults False
    """

    def __init__(
        self,
        type: str,
        name: str,
        X=None,
        y=None,
        sensitive_features=None,
        sensitive_intersections: Union[bool, list] = False,
    ):
        self.name = name
        self.X = self._process_X(X)
        self.y = self._process_y(y)
        self.sensitive_features = self._process_sensitive(
            sensitive_features, sensitive_intersections
        )
        self._validate_data()

    def get_data(self):
        data = {"X": self.X, "y": self.y, "sensitive_features": self.sensitive_features}
        return data

    def _determine_y_type(self):
        return type_of_target(self.y) if self.y is not None else None

    def _process_sensitive(self, sensitive_features, sensitive_intersections):
        if sensitive_features is None:
            return
        df = pd.DataFrame(sensitive_features)
        # add intersections if asked for
        features = df.columns
        if sensitive_intersections is False or len(features) == 1:
            return df
        elif sensitive_intersections is True:
            sensitive_intersections = features
        intersections = []
        for i in range(2, len(features) + 1):
            intersections += list(itertools.combinations(sensitive_intersections, i))
        for intersection in intersections:
            tmp = df[intersection[0]]
            for col in intersection[1:]:
                tmp = tmp.str.cat(df[col].astype(str), sep="_")
            label = "_".join(intersection)
            df[label] = tmp
        return df

    def _process_X(self, X):
        return X

    def _process_y(self, y):
        return y

    def _validate_data(self):
        self._validate_sensitive()
        self._validate_X()
        self._validate_y()

    def _validate_sensitive(self):
        # Validate the types
        if self.sensitive_features is not None and not isinstance(
            self.sensitive_features, (pd.DataFrame, pd.Series)
        ):
            raise ValidationError(
                "Sensitive_feature type is '"
                + type(self.sensitive_features).__name__
                + "' but the required type is either pd.DataFrame or pd.Series"
            )
        if self.sensitive_features is not None:
            if len(self.X) != len(self.sensitive_features):
                raise ValidationError(
                    "X and sensitive_features are not the same length. "
                    + f"X Length: {len(self.X)}, sensitive_features Length: {len(self.y)}"
                )
            if isinstance(
                self.X, (pd.Series, pd.DataFrame)
            ) and not self.X.index.equals(self.sensitive_features.index):
                raise ValidationError(
                    "X and sensitive features must have the same index"
                )

    @abstractmethod
    def _validate_X(self):
        pass

    @abstractmethod
    def _validate_y(self):
        pass