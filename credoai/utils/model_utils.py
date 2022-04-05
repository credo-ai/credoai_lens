from sklearn.ensemble import GradientBoostingClassifier
import warnings

def get_gradient_boost_model():
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        try:
            import xgboost as xgb
            try:
                model = xgb.XGBClassifier(
                    use_label_encoder=False,
                    eval_metric='logloss')
            except xgb.core.XGBoostError:
                model = GradientBoostingClassifier()
        except ModuleNotFoundError:
            model = GradientBoostingClassifier()
        return model