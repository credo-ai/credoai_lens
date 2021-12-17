from sklearn.ensemble import GradientBoostingClassifier

def get_gradient_boost_model():
    try:
        import xgboost as xgb
        model = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss')
    except ModuleNotFoundError:
        model = GradientBoostingClassifier()
    return model