from credoai.modules.credo_module import MultiModule


def init_sensitive_feature_module(mod, sensitive_features, **static_kwargs):
    dynamic_kwargs = {k: {'sensitive_features': v}
                      for k, v in sensitive_features.iteritems()}
    return MultiModule(mod, dynamic_kwargs, static_kwargs)
