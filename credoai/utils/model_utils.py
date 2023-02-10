import warnings

from sklearn.base import is_classifier, is_regressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import multiclass

from credoai.utils import global_logger

try:
    from tensorflow.keras import layers
except ImportError:
    pass


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
    """Returns basic information about model info"""
    try:
        framework = getattr(model, "framework_like", None)
        if not framework:
            framework = model.__class__.__module__.split(".")[0]
    except AttributeError:
        framework = None
    try:
        name = model.__class__.__name__
    except AttributeError:
        name = None
    return {"framework": framework, "lib_name": name}


def get_default_metrics(model):
    if is_classifier(model):
        return ["accuracy_score"]
    elif is_regressor(model):
        return ["r2_score"]
    else:
        return None


def type_of_target(target):
    return multiclass.type_of_target(target) if target is not None else None


#############################################
# Validation Functions for Various Model Types
#############################################
def validate_sklearn_like(model_obj, model_info: dict):
    pass


def validate_keras_clf(model_obj, model_info: dict):
    # This is how Keras checks sequential too: https://github.com/keras-team/keras/blob/master/keras/utils/layer_utils.py#L219
    if not model_info["lib_name"] == "Sequential":
        message = "Only Keras models with Sequential architecture are supported at this time. "
        message += "Using Keras with other architechtures has undefined behavior."
        global_logger.warning(message)

    valid_final_layer = (
        isinstance(model_obj.layers[-1], layers.Dense)
        and model_obj.layers[-1].activation.__name__ == "softmax"
    )
    valid_final_layer = valid_final_layer or (
        isinstance(model_obj.layers[-1], layers.Dense)
        and model_obj.layers[-1].activation.__name__ == "sigmoid"
    )
    valid_final_layer = valid_final_layer or isinstance(
        model_obj.layers[-1], layers.Softmax
    )
    if not valid_final_layer:
        message = "Expected output layer to be either: tf.keras.layers.Softmax or "
        message += "tf.keras.layers.Dense with softmax or sigmoid activation."
        global_logger.warning(message)

    if len(model_obj.layers[-1].output.shape) != 2:
        message = "Expected 2D output shape for Keras.Sequetial model: (batch_size, n_classes) or (None, n_classes)"
        global_logger.warning(message)

    if model_obj.layers[-1].output.shape[0] is not None:
        message = "Expected output shape of Keras model to have arbitrary length"
        global_logger.warning(message)

    if (
        model_obj.layers[-1].output.shape[1] < 2
        and model_obj.layers[-1].activation.__name__ != "sigmoid"
    ):
        message = "Expected classification output shape (batch_size, n_classes) or (None, n_classes). "
        message += "Univariate outputs not supported at this time."
        global_logger.warning(message)

    if (
        model_obj.layers[-1].output.shape[1] > 2
        and model_obj.layers[-1].activation.__name__ != "softmax"
        and not isinstance(model_obj.layers[-1], layers.Softmax)
    ):
        message = "Expected multiclass classification to use softmax activation with "
        message += "output shape (batch_size, n_classes) or (None, n_classes). "
        message += "Non-softmax classification not supported at this time."
        global_logger.warning(message)
        # TODO Add support for model-imposed argmax layer
        # https://stackoverflow.com/questions/56704669/keras-output-single-value-through-argmax


def validate_dummy(model_like, _):
    if model_like.model_like:
        tmp_model_info = get_model_info(model_like.model_like)
        if tmp_model_info["framework"] == "keras":
            validate_keras_clf(model_like.model_like, tmp_model_info)
        elif tmp_model_info["framework"] in ("sklearn", "xgboost"):
            validate_sklearn_like(model_like.model_like, tmp_model_info)
        else:
            raise
