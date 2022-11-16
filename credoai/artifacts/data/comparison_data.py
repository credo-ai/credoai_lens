"""Data artifact for pair-wise-comparison-based identity verification"""
from copy import deepcopy

import pandas as pd

from credoai.utils.common import ValidationError

from .base_data import Data


class ComparisonData(Data):
    """Class wrapper for pair-wise-comparison-based identity verification

    ComparisonData serves as an adapter between pair-wise-comparison-based identity verification
    and the identity verification evaluator in Lens.

    Parameters
    -------------
    name : str
        Label of the dataset
    pairs : pd.DataFrame of shape (n_pairs, 4)
        Dataframe where each row represents a data sample pair and associated subjects
        Type of data sample is decided by the ComparisonModel's `compare` function, which takes
        data sample pairs and returns their similarity scores. Examples are selfies, fingerprint scans,
        or voices of a person.
        Required columns:
            source-subject-id: unique identifier of the source subject
            source-subject-data-sample: data sample from the source subject
            target-subject-id: unique identifier of the target subject
            target-subject-data-sample: data sample from the target subject
    subjects_sensitive_features : pd.DataFrame of shape (n_subjects, n_sensitive_feature_names), optional
        Sensitive features of all subjects present in pairs dataframe
        If provided, disaggregated performance assessment is also performed.
        This can be the columns you want to perform segmentation analysis on, or
        a feature related to fairness like 'race' or 'gender'
        Required columns:
            subject-id: id of subjects. Must cover all the subjects inlcluded in `pairs` dataframe
            other columns with arbitrary names for sensitive features
    """

    def __init__(self, name: str, pairs=None, subjects_sensitive_features=None):
        super().__init__("ComparisonData", name)
        self.pairs = pairs
        self.subjects_sensitive_features = subjects_sensitive_features
        self._validate_pairs()
        self._validate_subjects_sensitive_features()
        self._preprocess_pairs()
        self._preprocess_subjects_sensitive_features()
        self._validate_pairs_subjects_sensitive_features_match()

    def copy(self):
        """Returns a deepcopy of the instantiated class"""
        return deepcopy(self)

    def _validate_pairs(self):
        """Validate the input `pairs` object"""
        if self.pairs is not None:
            # Basic validation for pairs
            if not isinstance(self.pairs, (pd.DataFrame)):
                raise ValidationError("pairs must be a pd.DataFrame")

            required_columns = [
                "source-subject-id",
                "source-subject-data-sample",
                "target-subject-id",
                "target-subject-data-sample",
            ]
            available_columns = self.pairs.columns
            for c in required_columns:
                if c not in available_columns:
                    raise ValidationError(
                        f"pairs dataframe does not contain the required column '{c}'"
                    )

            if len(available_columns) != 4:
                raise ValidationError(
                    f"pairs dataframe has '{len(available_columns)}' columns. It must have 4."
                )

            if self.pairs.isnull().values.any():
                raise ValidationError(
                    "pairs dataframe contains NaN values. It must not have any."
                )

    def _validate_subjects_sensitive_features(self):
        """Validate the input `subjects_sensitive_features` object"""
        if self.subjects_sensitive_features is not None:
            # Basic validation for subjects_sensitive_features
            if not isinstance(self.subjects_sensitive_features, (pd.DataFrame)):
                raise ValidationError(
                    "subjects_sensitive_features must be a pd.DataFrame"
                )

            available_columns = self.subjects_sensitive_features.columns
            if "subject-id" not in available_columns:
                raise ValidationError(
                    "subjects_sensitive_features dataframe does not contain the required column 'subject-id'"
                )
            if len(available_columns) < 2:
                raise ValidationError(
                    "subjects_sensitive_features dataframe includes 'subject-id' column only. It must include at least one sensitive feature column too."
                )

            if self.subjects_sensitive_features.isnull().values.any():
                raise ValidationError(
                    "subjects_sensitive_features dataframe contains NaN values. It must not have any."
                )

            sensitive_features_names = list(self.subjects_sensitive_features.columns)
            sensitive_features_names.remove("subject-id")
            for sf_name in sensitive_features_names:
                unique_values = self.subjects_sensitive_features[sf_name].unique()
                if len(unique_values) == 1:
                    raise ValidationError(
                        f"Sensitive Feature column {sf_name} must have more "
                        f"than one unique value. Only found one value: {unique_values[0]}"
                    )

    def _preprocess_pairs(self):
        """Preprocess the input `pairs` object"""
        cols = ["source-subject-id", "target-subject-id"]
        self.pairs[cols] = self.pairs[cols].astype(str)

    def _preprocess_subjects_sensitive_features(self):
        """Preprocess the input `subjects_sensitive_features` object"""
        if self.subjects_sensitive_features is not None:
            self.subjects_sensitive_features = self.subjects_sensitive_features.astype(
                str
            )

    def _validate_pairs_subjects_sensitive_features_match(self):
        if self.subjects_sensitive_features is not None:
            subjects_in_pairs = list(
                pd.unique(
                    self.pairs[["source-subject-id", "target-subject-id"]].values.ravel(
                        "K"
                    )
                )
            )
            subjects_in_subjects_sensitive_features = list(
                self.subjects_sensitive_features["subject-id"].unique()
            )
            missing_ids = set(subjects_in_pairs) - set(
                subjects_in_subjects_sensitive_features
            )
            if len(missing_ids) > 0:
                raise ValidationError(
                    f"Some subject-id s that exist in the input `pairs` object do not exist in the input `subjects_sensitive_features` object."
                    f"These inclide {missing_ids}."
                )

    def _validate_X(self):
        pass

    def _validate_y(self):
        pass
