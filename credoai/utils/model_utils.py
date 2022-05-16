from sklearn.ensemble import RandomForestClassifier
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