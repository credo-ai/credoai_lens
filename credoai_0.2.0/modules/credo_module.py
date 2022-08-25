from abc import ABC, abstractmethod

import pandas as pd
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
            raise NotRunError("Results not created yet. Call 'run' to create results")


class MultiModule(CredoModule):
    def __init__(
        self, module: CredoModule, dynamic_kwargs: dict, static_kwargs: dict = None
    ):
        """Helper class to build higher order modules that take in lists of arguments

        Example:
        The below example shows how you can run the EquityModule with two sets of sensitive
        features.

        dynamic_kwargs = {'race': {'sensitive_features': pd.DataFrame([0, 1, 1, 1])},
                          'gender': {'sensitive_features': pd.DataFrame([0, 1, 0, 1])}}
        MultiModule(EquityModule,
                    dynamic_kwargs,
                    static_kwargs={'y': [0, 1, 1, 1]})

        Parameters
        ----------
        module : CredoModule
            the module to repeat
        dynamic_kwargs : dict
            Dictionary of dictionaries. Each value is a set of kwargs
            to pass to module, with the key distinguishing the different
            "submodules"
        static_kwargs : dict
            Dictionary of kwargs to pass to each instance of module
        """
        if static_kwargs is None:
            static_kwargs = {}
        self.static_kwargs = static_kwargs
        self.dynamic_kwargs = dynamic_kwargs
        self.modules = {}
        for name, kwargs in dynamic_kwargs.items():
            self.modules[name] = module(**static_kwargs, **kwargs)

    def run(self):
        for mod in self.modules.values():
            mod.run()
        return self

    def get_results(self):
        return {name: mod.get_results() for name, mod in self.modules.items()}

    def prepare_results(self):
        prepared_results = pd.DataFrame()
        for name, module in self.modules.items():
            prepared_results = pd.concat([prepared_results, module.prepare_results()])
        return prepared_results
