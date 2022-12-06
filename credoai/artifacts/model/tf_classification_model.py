"""Model artifact wrapping any classification model"""
from .base_model import Model


class TF_Keras_ClassificationModel(Model):
    """Class wrapper around classification model to be assessed

    ClassificationModel serves as an adapter between arbitrary binary or multi-class
    classification models and the evaluations in Lens. Evaluations depend on
    ClassificationModel instantiating `predict` and (optionally) `predict_proba`

    Parameters
    ----------
    name : str
        Label of the model
    model_like : model_like
        A binary or multi-class classification model or pipeline. It must have a
            `predict` function that returns array containing the class labels for each sample.
            It can also optionally have a `predict_proba` function that returns array containing
            the class labels probabilities for each sample.
    """

    def __init__(self, name: str, model_like=None, tags=None):
        super().__init__(
            "Classification",
            ["predict", "predict_proba"],
            ["predict"],
            name,
            model_like,
            tags,
        )

    def _update_functionality(self):
        """Conditionally updates functionality based on framework"""
        if self.model_info["framework"] in PREDICT_PROBA_FRAMEWORKS:
            func = getattr(self, "predict_proba", None)
            if func and len(self.model_like.classes_) == 2:
                self.__dict__["predict_proba"] = lambda x: func(x)[:, 1]
