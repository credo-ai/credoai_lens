############# Validation related functionality ##################

from distutils.log import error
from email import message
from credoai.utils.common import ValidationError
from pandas import Series, DataFrame


def check_instance(obj, inst_type, message=None):
    if not message:
        f"Object {obj} should be an instance of {inst_type.__name__}"
    if not isinstance(obj, inst_type):
        raise ValidationError(message)


def check_data_instance(obj, inst_type, name="Data"):
    message = f"{name} should be an instance of {inst_type.__name__}"
    check_instance(obj, inst_type, message)


def check_model_instance(obj, inst_type, name="Model"):
    message = f"{name} should be an instance of {inst_type.__name__}"
    check_instance(obj, inst_type, message)


def check_feature_presence(feature_name, df, name):
    message = f"Feature {feature_name} not found in dataframe {name}"
    if not feature_name in df.columns:
        raise ValidationError(message)


def check_existence(obj, name=None):
    message = f"Missing object {name}"
    if isinstance(obj, DataFrame):
        if obj is None:
            raise ValidationError(message)
        else:
            return
    if not obj:
        raise ValidationError(message)


def check_requirements_existence(self):
    for required_name in self.required_artifacts:
        check_existence(vars(self)[required_name], required_name)


def check_for_nulls(obj, name):
    message = f"Detected nulls in {name}"
    if obj is not None:
        if obj.isnull().values.any():
            raise ValidationError(message)


def check_artifact_for_nulls(obj, name):
    errors = []
    if obj.X is not None:
        if obj.X.isnull().values.any():
            errors.append("X")
    if obj.y is not None:
        if obj.y.isnull().values.any():
            errors.append("y")
    if obj.sensitive_features is not None:
        if obj.sensitive_features.isnull().values.any():
            errors.append("sensitive_features")

    if len(errors) > 0:
        message = f"Detected null values in {name}, in attributes: {','.join(errors)}"
        raise ValidationError(message)
