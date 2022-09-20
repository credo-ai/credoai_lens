############# Validation related functionality ##################

from credoai.utils.common import ValidationError


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
    if not obj:
        raise ValidationError(message)


def check_requirements_existence(self):
    for required_name in self.required_artifacts:
        check_existence(vars(self)[required_name], required_name)
