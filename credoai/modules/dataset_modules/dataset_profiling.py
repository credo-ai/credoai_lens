import pandas as pd

from credoai.modules.credo_module import CredoModule
from pandas_profiling import ProfileReport

class DatasetProfiling(CredoModule):
    """Dataset profiling module for Credo AI.

    This module takes in features and labels and provides functionality to generate profile reports 

    Parameters
    ----------
    X : pandas.DataFrame
        The features
    y : pandas.Series
        The outcome labels
    sensitive_features : pandas.Series
        A series of the sensitive feature labels (e.g., "male", "female")
    """
    def __init__(self,
                X: pd.DataFrame,
                y: pd.Series,
                sensitive_features: pd.Series):

        self.data = pd.concat([X, sensitive_features, y], axis=1)
    
    def profile_data(self):
        """Generates data profile reports
        """        

        # Initialize the report
        profile = ProfileReport(self.data, title="Dataset", explorative=True)
        
        profile.to_notebook_iframe()

    def run(self):
        return None
    
    def prepare_results(self):
        return None