from abc import ABC, abstractmethod
from credoai.utils import NotRunError

class CredoModule(ABC):
    """
    Base Class to build other modules off of.
    
    Defines basic functions for interacting with Credo AI's Governance App
    """
    def __init__(self):
        self.results = None

    @abstractmethod
    def run(self):
        """
        Creates self.results object. 
        
        Returns
        -------
        self
        """
        return self
    
    @abstractmethod
    def prepare_results(self):
        """
        Returns
        --------
        pd.DataFrame or pd.Series or dict
            if pd.DataFrame, the index must be "metric_type", with one column called
            "value", which contains the metrics value as a single number. Other columns
            may be included as metadata.

            If a series, the single column should include the "value" column.

            If a dictionary, the key/val pairs should be metric_type/value
        """
        if self.results is not None:
            # prepare results code
            pass
        else:
            raise NotRunError(
                "Results not created yet. Call 'run' with appropriate arguments before preparing results"
            )

    def get_results(self):
        if self.results is not None:
            return self.results
        else:
            raise NotRunError(
                "Results not created yet. Call 'run' to create results"
            )
        
        