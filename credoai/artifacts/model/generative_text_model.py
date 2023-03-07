"""Model artifact wrapping a generative text model (e.g. LLMs like GPT-2)"""
from .base_model import Model


class GenText(Model):
    """Class wrapper around generative text model

    GenText is a class wrapper around generative text model (e.g. LLMs like GPT-2).

    Parameters
    ----------
    name : str
        Label of the model
    model_like
        A model conforming that defines a "generate" method.
        See :py:mod:`credoai.artifacts.model.openai_adapters` for an example of a model_like object
        using openai's api.
    """

    def __init__(self, name: str, model_like=None):
        super().__init__(
            "GenText",
            ["generate"],
            ["generate"],
            name,
            model_like,
        )
