from credoai.utils.model_utils import (
    validate_sklearn_like,
    validate_keras_clf,
    validate_dummy,
)

SKLEARN_LIKE_FRAMEWORKS = ["sklearn", "xgboost"]
MODEL_TYPES = [
    "REGRESSION",
    "CLASSIFICATION",
    "BINARY_CLASSIFICATION",
    "MULTICLASS_CLASSIFICATION",
    "COMPARISON",
]

# Multilayer Perceptron
MLP_FRAMEWORKS = ["keras"]

FRAMEWORK_VALIDATION_FUNCTIONS = {
    "sklearn": validate_sklearn_like,
    "xgboost": validate_sklearn_like,
    "keras": validate_keras_clf,
    "credoai": validate_dummy,
    # check on tensorflow generic, given validation strictness
}
