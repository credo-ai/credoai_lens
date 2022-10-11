import warnings

from sklearn.base import is_classifier, is_regressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import multiclass


def get_generic_classifier():
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        try:
            import xgboost as xgb

            try:
                model = xgb.XGBClassifier(
                    use_label_encoder=False, eval_metric="logloss"
                )
            except xgb.core.XGBoostError:
                model = RandomForestClassifier()
        except ModuleNotFoundError:
            model = RandomForestClassifier()
        return model


def get_model_info(model):
    try:
        framework = model.__class__.__module__.split(".")[0]
    except AttributeError:
        framework = None
    return {"framework": framework}


def get_default_metrics(model):
    if is_classifier(model):
        return ["accuracy_score"]
    elif is_regressor(model):
        return ["r2_score"]
    else:
        return None


def type_of_target(target):
    return multiclass.type_of_target(target) if target is not None else None
