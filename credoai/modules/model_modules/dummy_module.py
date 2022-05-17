import numpy as np
import pandas as pd

from credoai.modules.credo_module import CredoModule


class DummyModule(CredoModule):
    """Privacy module for Credo AI.
    This module takes in model and data and provides functionality to perform privacy assessment
    Parameters
    ----------
    model : model
        A trained ML model
    x_train : pandas.DataFrame
        The training features
    y_train : pandas.Series
        The training outcome labels
    x_test : pandas.DataFrame
        The test features
    y_test : pandas.Series
        The test outcome labels
    """    
    def __init__(self,
                model,
                x_train,
                y_train,
                x_test,
                y_test,
                attack_train_ratio=0.50):

        self.x_train = x_train.to_numpy()
        self.y_train = y_train.to_numpy()
        self.x_test = x_test.to_numpy()
        self.y_test = y_test.to_numpy()
        self.model = model
        self.attack_train_ratio = attack_train_ratio

    def run(self):

        self.results = {'a': 1}

        return self 
    
    def prepare_results(self):
        if self.results is not None:
            return pd.DataFrame(self.results)
        else:
            raise NotRunError(
                "Results not created yet. Call 'run' to create results"
            )