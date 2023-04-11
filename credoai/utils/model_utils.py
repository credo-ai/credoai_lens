import warnings

from sklearn.base import is_classifier, is_regressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import multiclass
import numpy as np

from credoai.utils import global_logger

from credoai.utils.common import ValidationError

try:
    from tensorflow.keras import layers
except ImportError:
    pass

try:
    import torch
except ImportError:
    print(
        "Torch not loaded. Torch models will not be wrapped properly if supplied to ClassificationModel"
    )


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
            # Chunk below covers custom model classes.
            # It enables us to cover cases like torch.nn.Module, where the parent class is structured
            # but the user-specified class won't have a proper "framework"
            if model.__class__.__module__ == "__main__":
                framework = type(model).__bases__[0].__module__.split(".")[0]
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


def validate_torch_clf(model_obj, model_info: dict):
    if not issubclass(model_obj.__class__, torch.nn.Module):
        message = "Only torch.nn.Module subclasses are supported at this time. "
        message += "Using other PyTorch model architectures have undefined behavior."
        global_logger.warning(message)

    if not hasattr(model_obj, "input_shape") or not isinstance(
        model_obj.input_shape, tuple
    ):
        message = (
            "Expected PyTorch model to have an `input_shape` attribute of type tuple. "
        )
        message += "Please specify the expected input shape for your PyTorch model."
        raise ValidationError(message)

    if not hasattr(model_obj, "output_activation"):
        message = "Expected PyTorch model to have an `output_activation` attribute. "
        message += "Supported output activations are 'softmax' and 'sigmoid'. "
        message += "Assuming 'sigmoid' as default."
        model_obj.output_activation = "sigmoid"
        global_logger.warning(message)

    if model_obj.output_activation not in ["softmax", "sigmoid"]:
        message = "Unsupported output activation for PyTorch model. "
        message += (
            "Only 'softmax' and 'sigmoid' activations are supported at this time."
        )
        raise ValidationError(message)


def validate_dummy(model_like, _):
    if model_like.model_like:
        tmp_model_info = get_model_info(model_like.model_like)
        if tmp_model_info["framework"] == "keras":
            validate_keras_clf(model_like.model_like, tmp_model_info)
        elif tmp_model_info["framework"] in ("sklearn", "xgboost"):
            validate_sklearn_like(model_like.model_like, tmp_model_info)
        else:
            raise


def reg_handle_torch(model_like):
    pred_func = getattr(model_like, "forward", None)
    if pred_func is None:
        raise ValueError("Model should have a `forward` method to perform predictions.")

    return pred_func


def clf_handle_keras(model_like):
    predict_obj = None
    predict_proba_obj = None
    clf_type = "BINARY_CLASSIFICATION"

    pred_func = getattr(model_like, "predict", getattr(model_like, "call"))

    if model_like.layers[-1].output_shape == (None, 1):
        # Assumes sigmoid -> probabilities need to be rounded
        predict_obj = lambda x: pred_func(x).round()
        # Single-output sigmoid is binary by definition
        clf_type = "BINARY_CLASSIFICATION"
    else:
        # Assumes softmax -> probabilities need to be argmaxed
        predict_obj = lambda x: np.argmax(pred_func(x), axis=1)
        if model_like.layers[-1].output_shape[1] == 2:
            clf_type = "BINARY_CLASSIFICATION"
        else:
            clf_type = "MULTICLASS_CLASSIFICATION"

    if model_like.layers[-1].output_shape == (None, 2):
        predict_proba_obj = lambda x: pred_func(x)[:, 1]
    elif (
        len(model_like.layers[-1].output_shape) == 2
        and model_like.layers[-1].output_shape[1] == 1
    ):
        # Sigmoid -> needs to be (n_samples, ) to work with sklearn metrics
        predict_proba_obj = lambda x: np.reshape(pred_func(x), (-1, 1))
    elif (
        len(model_like.layers[-1].output_shape) == 2
        and model_like.layers[-1].output_shape[1] > 2
    ):
        predict_proba_obj = pred_func
    else:
        pass
        # predict_proba is not valid (for now) --> this would correspond to multi-dimensional outputs or something similarly weird

    return predict_obj, predict_proba_obj, clf_type


def clf_handle_torch(model_like):
    predict_obj = None
    predict_proba_obj = None
    clf_type = "BINARY_CLASSIFICATION"

    pred_func = getattr(model_like, "forward", None)
    if pred_func is None:
        raise ValueError("Model should have a `forward` method to perform predictions.")

    output_activation = getattr(model_like, "output_activation", "sigmoid")
    if output_activation not in {"sigmoid", "softmax"}:
        global_logger.warning(
            "Invalid output activation function provided. Using sigmoid as the default activation function."
        )
        output_activation = "sigmoid"

    with torch.no_grad():
        dummy_input = torch.randn(1, *model_like.input_shape)
        output = model_like(dummy_input)

    output_shape = output.shape

    if output_activation == "sigmoid" and output_shape[-1] == 1:

        def pred(x):
            if isinstance(x, torch.utils.data.dataloader.DataLoader):
                preds = []
                for data, _ in x:
                    preds.append(torch.round(torch.sigmoid(pred_func(data))).numpy())
                return np.concatenate(preds, axis=0)
            else:
                return torch.round(torch.sigmoid(pred_func(torch.tensor(x)))).numpy()

        predict_obj = pred
        clf_type = "BINARY_CLASSIFICATION"
    elif output_activation == "softmax":

        def pred(x):
            if isinstance(x, torch.utils.data.dataloader.DataLoader):
                preds = []
                for data, _ in x:
                    preds.append(
                        torch.argmax(
                            torch.softmax(pred_func(data), dim=-1), dim=-1
                        ).numpy()
                    )
                return np.concatenate(preds, axis=0)
            else:
                return torch.argmax(
                    torch.softmax(pred_func(torch.tensor(x)), dim=-1), dim=-1
                ).numpy()

        predict_obj = pred
        clf_type = (
            "BINARY_CLASSIFICATION"
            if output_shape[-1] == 2
            else "MULTICLASS_CLASSIFICATION"
        )

    if output_activation == "sigmoid" and output_shape[-1] == 1:

        def pred_prob(x):
            if isinstance(x, torch.utils.data.dataloader.DataLoader):
                probs = []
                for data, _ in x:
                    probs.append(torch.sigmoid(pred_func(data)).numpy())
                return np.concatenate(probs, axis=0)
            else:
                return torch.sigmoid(pred_func(torch.tensor(x))).numpy()

        predict_proba_obj = pred_prob
    elif output_activation == "softmax":
        if output_shape[-1] == 2:

            def pred_prob(x):
                if isinstance(x, torch.utils.data.dataloader.DataLoader):
                    probs = []
                    for data, _ in x:
                        probs.append(
                            torch.softmax(pred_func(data), dim=-1)
                            .detach()
                            .numpy()[:, 1]
                        )
                    return np.concatenate(probs, axis=0)
                else:
                    return (
                        torch.softmax(pred_func(torch.tensor(x)), dim=-1)
                        .detach()
                        .numpy()[:, 1]
                    )

            predict_proba_obj = pred_prob
        else:

            def pred_prob(x):
                if isinstance(x, torch.utils.data.dataloader.DataLoader):
                    probs = []
                    for data, _ in x:
                        probs.append(
                            torch.softmax(pred_func(data), dim=-1).detach().numpy()
                        )
                    return np.concatenate(probs, axis=0)
                else:
                    return (
                        torch.softmax(pred_func(torch.tensor(x)), dim=-1)
                        .detach()
                        .numpy()
                    )

            predict_proba_obj = pred_prob

    return predict_obj, predict_proba_obj, clf_type
