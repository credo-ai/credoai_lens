from credoai.utils.model_utils import get_model_info


class Model:
    """Base class for all models in Lens.

    Parameters
    ----------
    name : str
        Label of the model
    type : str, optional
        Type of the model
    model_like : model_like
        A model or pipeline.

    """

    def __init__(self, name, type, model_like):
        self.name = name
        self.type = type
        self.model_like = model_like

        info = get_model_info(model_like)
        self.framework = info["framework"]
