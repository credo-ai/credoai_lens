from abc import ABC, abstractmethod

class CredoModule(ABC):
    """
    Base Class to build other modules off of.
    
    Defines basic functions for interacting with Credo's governance platform
    """
    def __init__(self):
        self.results = None

    @abstractmethod
    def run(self):
        pass
    
    @abstractmethod
    def prepare_results(self):
        """
        Returns
        --------
        pd.DataFrame or pd.Series or dict
            if pd.DataFrame, the index should be
            called "metric" and list the metric,
            and the first column should be called
            "value" which contains the metrics value 
            as a single number
        """
        if self.results is not None:
            # prepare results code
            pass
        else:
            raise Exception(
                "Results not created yet. Call 'run' with appropriate arguments before preparing results"
            )

    def get_results(self):
        if self.results is not None:
            return self.results
        else:
            raise Exception(
                "Results not created yet. Call 'run' to create results"
            )
        
        