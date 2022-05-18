from sklearn.ensemble import RandomForestClassifier
from sklearn.base import is_classifier, is_regressor
import warnings

def get_generic_classifier():
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        try:
            import xgboost as xgb
            try:
                model = xgb.XGBClassifier(
                    use_label_encoder=False,
                    eval_metric='logloss')
            except xgb.core.XGBoostError:
                model = RandomForestClassifier()
        except ModuleNotFoundError:
            model = RandomForestClassifier()
        return model

def get_default_metrics(model):
    if is_classifier(model):
        return ['accuracy_score']
    elif is_regressor(model):
        return ['r2_score']
    else:
        return None