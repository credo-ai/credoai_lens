"""Model artifact wrapping any comparison model"""
from .base_model import Model


class ComparisonModel(Model):
    """Class wrapper around comparison model to be assessed

    ComparisonModel serves as an adapter between arbitrary pair-wise comparison models and 
    the identity verification evaluations in Lens. Evaluations depend on ComparisonModel instantiating `compare`

    Parameters
    ----------
    name : str
        Label of the model
    model_like : model_like
        A pair-wise comparison model or pipeline. It must have a
            `compare` function that returns array containing the similarity scores for each pair.
    """

    def __init__(self, name: str, model_like=None):
        super().__init__(
            "Comparison",
            ["compare"],
            ["compare"],
            name,
            model_like,
        )

class DummyComparisonModel:
    """Class wrapper around comparison model predictions

    This class can be used when a comparison model is not available but its outputs are.
        The output include the array containing the predicted similarity scores.
        Wrap the outputs with this class into a dummy comparison model and pass it as
        the model to `ComparisonModel`.

    Parameters
    ----------
    compare_output : array
        Array containing the output of a comparison model's "compare" method
    """

    def __init__(self, compare_output=None):
        self.compare_output = compare_output

    def compare(self, X=None):
        return self.compare_output
