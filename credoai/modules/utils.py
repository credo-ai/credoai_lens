from credoai.modules.credo_module import MultiModule


def init_sensitive_feature_module(module, sensitive_features, **static_kwargs):
    """Creates a module that uses sensitive features

    Creates a MultiModule using a base module with different sensitive features

    Parameters
    ----------
    module : CredoModule
    sensitive_features : pd.DataFrame
        A dataframe of sensitive features

    Returns
    -------
    MultiModule
    """
    dynamic_kwargs = {
        k: {"sensitive_features": v} for k, v in sensitive_features.iteritems()
    }
    return MultiModule(module, dynamic_kwargs, static_kwargs)
