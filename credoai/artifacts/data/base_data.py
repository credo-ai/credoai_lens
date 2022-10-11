"""Abstract class for the data artifacts used by `Lens`"""
# Data is a lightweight wrapper that stores data
import itertools
from abc import ABC, abstractmethod
from typing import Optional, Union

import pandas as pd
from credoai.utils.common import ValidationError
from credoai.utils.model_utils import type_of_target


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
        self.X = X
        self.y = y
        self.sensitive_features = sensitive_features
        self._validate_inputs()
        self._process_inputs(sensitive_intersections)
        self._validate_processing()
        self._active_sensitive_feature: Optional[str] = None

    @property
    def active_sens_feat(self):
        """
        Defines which sensitive feature an evaluator will be working on.

        In combination with the property sensitive_feature this effectively creates
        a view of a specific artifact.
        """
        if self._active_sensitive_feature is None:
            self._active_sensitive_feature = self.sensitive_features.columns[0]
        return self._active_sensitive_feature

    @active_sens_feat.setter
    def active_sens_feat(self, value: str):
        """
        Sets the active_sens_feat value.

        Parameters
        ----------
        value : str
            Name of the sensitive feature column an evaluator has to operate on.
        """
        self._active_sensitive_feature = value

    @property
    def sensitive_feature(self):
        """
        Reveals the sensitive feature defined by active_sens_feat.

        This is generally called from within an evaluator, when it is working
        on a single sensitive feature.
        """
        return self.sensitive_features[self.active_sens_feat]

    @property
    def y_type(self):
        return type_of_target(self.y)

    @property
    def data(self):
        data = {"X": self.X, "y": self.y, "sensitive_features": self.sensitive_features}
        return data

    def _process_inputs(self, sensitive_intersections):
        if self.X is not None:
            self.X = self._process_X(self.X)
        if self.y is not None:
            self.y = self._process_y(self.y)
        if self.sensitive_features is not None:
            self.sensitive_features = self._process_sensitive(
                self.sensitive_features, sensitive_intersections
            )

    def _process_sensitive(self, sensitive_features, sensitive_intersections):
        """
        Formats sensitive features

        Parameters
        ----------
        sensitive_features :
            Sensitive features as provided by a user. Any format that can be constrained
            in a dataframe is accepted.
        sensitive_intersections : Bool
            Indicates whether to create intersections among sensitive features.

        Returns
        -------
        _type_
            _description_
        """
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

    def _validate_inputs(self):
        """Basic input validation"""
        if self.X is not None:
            self._validate_X()
        if self.y is not None:
            self._validate_y()
        if self.sensitive_features is not None:
            self._validate_sensitive()

    def _validate_sensitive(self):
        """Sensitive features validation"""
        # Validate the types
        if not isinstance(self.sensitive_features, (pd.Series, pd.DataFrame)):
            raise ValidationError(
                "Sensitive_feature type is '"
                + type(self.sensitive_features).__name__
                + "' but the required type is either pd.DataFrame or pd.Series"
            )
        if self.X is not None:
            if len(self.X) != len(self.sensitive_features):
                raise ValidationError(
                    "X and sensitive_features are not the same length. "
                    + f"X Length: {len(self.X)}, sensitive_features Length: {len(self.y)}"
                )
        if isinstance(self.X, (pd.Series, pd.DataFrame)) and not self.X.index.equals(
            self.sensitive_features.index
        ):
            raise ValidationError("X and sensitive features must have the same index")

        if isinstance(self.sensitive_features, pd.Series):
            if not hasattr(self.sensitive_features, "name"):
                raise ValidationError("Feature Series should have a name attribute")

    @abstractmethod
    def _validate_X(self):
        pass

    @abstractmethod
    def _validate_y(self):
        pass

    def _validate_processing(self):
        """Validation of processed data"""
        if self.X is not None:
            self._validate_processed_X()
        if self.y is not None:
            self._validate_processed_y()
        if self.sensitive_features is not None:
            self._validate_processed_sensitive()

    def _validate_processed_X(self):
        pass

    def _validate_processed_y(self):
        pass

    def _validate_processed_sensitive(self):
        """VAlidation of processed sensitive features"""
        for col_name, col in self.sensitive_features.iteritems():
            unique_values = col.unique()
            if len(unique_values) == 1:
                raise ValidationError(
                    f"Sensitive Feature column {col_name} must have more "
                    f"than one unique value. Only found one value: {unique_values[0]}"
                )
