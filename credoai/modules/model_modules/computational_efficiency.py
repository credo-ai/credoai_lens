from credoai.modules.credo_module import CredoModule
from credoai.utils.common import NotRunError
from time import perf_counter
import pandas as pd

class ComputationalEfficiencyModule(CredoModule):
    """
    Computational Efficiency module for Credo AI. 

    Outputs inference time

    Parameters
    ----------
    prediction_fun : callable
        Function that takes in X and outputs some prediction
    X : data
        Data passed to prediction_fun to perform inference. Must instantiate
        __len__
    """
    def __init__(self, prediction_fun, X):
        self.prediction_fun = prediction_fun
        self.X = X

    def run(self):
        """
        Run computational inference module
           
        Returns
        -------
        self
        """
        self.results = {'computational_efficiency_inference': self._inference_efficiency()}
        return self
        
    def prepare_results(self):
        """Prepares results for Credo AI's governance platform

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        NotRunError
            Occurs if self.run is not called yet to generate the raw assessment results
        """
        if self.results is not None:
            return self.results
        else:
            raise NotRunError(
                "Results not created yet. Call 'run' with appropriate arguments before preparing results"
            )

    def _inference_efficiency(self):
        t1 = perf_counter()
        out = self.prediction_fun(self.X)
        t2 = perf_counter()
        time_in_ms = (t2-t1)*1000
        # calculate for batches of 1000
        time_per_batch = time_in_ms/len(self.X)*1000
        return time_per_batch
        