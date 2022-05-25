"""
Defines abstract base class for all CredoReports
"""

from abc import ABC, abstractmethod
from credoai.utils import get_metric_keys, ValidationError
from credoai.reporting.plot_utils import format_label
from IPython.core.display import display, HTML
import os
import pandas as pd
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import textwrap

class CredoReporter(ABC):
    """Abstract base class for all CredoReports"""
    
    def __init__(self, assessment):
        self.assessment = assessment
        self.module = assessment.initialized_module
        self.key_lookup = None
        self.figs = []

    def report(self, plot=True, rerun=False):
        """Reports assets

        Once run, will cache assets unless rerun = True

        Parameters
        ----------
        plot : bool, optional
            If True, plot assets. Defaults True
        rerun : bool, optional
            If True, rerun asset creation. Defaults True
            
        Returns
        -------
        array of dictionaries reflecting assets
        """        
        if not self.figs or rerun:
            self.figs = []
            self._create_assets()
        if plot:
            [display(fig['figure']) for fig in self.figs]
        return self.figs

    def set_key_lookup(self, lens_prepared_results):
        """Sets the lookup dataframe to be used for key matching

        Parameters
        ----------
        lens_prepared_results : DataFrame
            output of lens._prepare_results(assessment)
        """
        self.key_lookup = lens_prepared_results

    def display_results_tables(self):
        results = self.assessment.get_results()
        for key, val in results.items():
            title = format_label(key.upper(), wrap_length=30)
            anchor_name = f'{str(self.assessment)}-{"-".join(title.split())}'
            display(HTML(f'<h3 id="{anchor_name}"><span style="font-size:1em; text-align: left">{title}</span></h3>'))
            try:
                val = pd.DataFrame(val)
                display(val)
            except ValueError:
                try:
                    val = pd.Series(val).to_frame()
                    display(val)
                except:
                    print(val)
            print('\n')

    @abstractmethod
    def _create_assets(self):
        """Creates assets
        
        Example:
        figures = [list of matplotlib figures]
        assets = [_create_chart(f) for f in figures]
        self.figs = assets
        """
        pass

    def _create_chart(self, 
                     figure, 
                     description: str = None, 
                     name: str = 'Figure',
                     module_prepared_results = None):
        keys = []
        if self.key_lookup is not None:
            if module_prepared_results is not None:
                keys = get_metric_keys(module_prepared_results, self.key_lookup)
            else:
                keys = self.key_lookup['metric_key'].tolist()
        return {'name': name, 'figure': figure, 'description': description, 'metric_keys': keys}


