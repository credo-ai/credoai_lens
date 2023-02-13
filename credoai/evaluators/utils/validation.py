############# Validation related functionality ##################

import inspect

import numpy as np
import pandas as pd

try:
    tf_exists = True
    import tensorflow as tf
except ImportError:
    tf_exists = False

from credoai.artifacts.data.tabular_data import TabularData
from credoai.artifacts.model.base_model import Model
from credoai.utils import global_logger
from credoai.utils.common import ValidationError

##############################
# Checking individual artifacts
##############################


def check_instance(obj, inst_type, message=None):
    if not message:
        message = f"Object {obj} should be an instance of {inst_type.__name__}"
    if not isinstance(obj, inst_type):
        raise ValidationError(message)


def check_data_instance(obj, inst_type, name="Data"):
    message = f"{name} should be an instance of {inst_type.__name__}"
    check_instance(obj, inst_type, message)


def check_model_instance(obj, inst_type, name="Model"):
    if isinstance(inst_type, tuple):
        comp_label = " or ".join([x.__name__ for x in inst_type])
    else:
        comp_label = inst_type.__name__
    message = f"{name} should be an instance of {comp_label}"
    check_instance(obj, inst_type, message)


def check_feature_presence(feature_name, df, name):
    if isinstance(df, pd.DataFrame):
        if not feature_name in df.columns:
            message = f"Feature {feature_name} not found in dataframe {name}"
            raise ValidationError(message)
    if isinstance(df, pd.Series):
        if not df.name == feature_name:
            message = f"Feature {feature_name} not found in series {name}"
            raise ValidationError(message)


def check_existence(obj, name=None):
    message = f"Missing object {name}"
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        if obj.empty:
            raise ValidationError(message)
    elif obj is None or not obj:
        raise ValidationError(message)


def check_nulls_by_data_type(data):
    nulls = False
    if isinstance(data, (pd.DataFrame, pd.Series)):
        nulls = data.isnull().to_numpy().any()
    if isinstance(data, np.ndarray):
        nulls = np.isnan(data).any()
    if tf_exists and isinstance(data, tf.Tensor):
        nulls = tf.reduce_any(tf.math.is_nan(data))
    if (
        tf_exists and isinstance(data, (tf.data.Dataset, tf.keras.utils.Sequence))
    ) or inspect.isgeneratorfunction(data):
        message = """
        Evaluator Validation: Checking for nulls in generator-based or mapped data is not currently
        supported. Please be sure to sanitize your data. Downstream errors may arise due to nulls in 
        image or other tensor data.
        """
        global_logger.warning(message)
    return nulls


#################################
# Checking evaluator requirements
#################################


def check_data_for_nulls(obj, name, check_X=True, check_y=True, check_sens=True):
    errors = []
    if check_X and obj.X is not None:
        if check_nulls_by_data_type(obj.X):
            errors.append("X")
    if check_y and obj.y is not None:
        if check_nulls_by_data_type(obj.y):
            errors.append("y")
    if check_sens and obj.sensitive_features is not None:
        if check_nulls_by_data_type(obj.sensitive_features):
            errors.append("sensitive_features")

    if len(errors) > 0:
        message = f"Detected null values in {name}, in attributes: {','.join(errors)}"
        raise ValidationError(message)


def check_requirements_existence(self):
    for required_name in self.required_artifacts:
        check_existence(vars(self)[required_name], required_name)


def check_requirements_deepchecks(self):
    # For case when we require at least one dataset
    # All supplied datasets must be of correct form
    at_least_one_artifact = False
    for required_name in self.required_artifacts:
        if "data" in required_name:
            try:
                check_data_instance(vars(self)[required_name], TabularData)
                at_least_one_artifact = True
            except ValidationError as e:
                if vars(self)[required_name]:
                    # Check if the artifact actually contains anything
                    # If so, raise the same error
                    raise ValidationError(e)
                else:
                    # Do nothing. We're simply not going to have this optional artifact
                    pass
        else:
            # Check model
            try:
                check_model_instance(vars(self)[required_name], Model)
                at_least_one_artifact = True
            except ValidationError as e:
                if vars(self)[required_name]:
                    # Check if model is NoneType
                    raise ValidationError(e)
                else:
                    # Model is NoneType but model is optional for deepchecks
                    pass

    if not at_least_one_artifact:
        raise ValidationError(
            "Expected at least one valid artifact. None provided or all objects passed are otherwise invalid"
        )
