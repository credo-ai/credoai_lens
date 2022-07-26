"""
Defines abstract base class for all CredoReports
"""

import os
import textwrap
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
from credoai.reporting.plot_utils import format_label
from credoai.utils import ValidationError
from IPython.core.display import HTML, display


class CredoReporter(ABC):
    """Abstract base class for all CredoReports

    CredoReporters are associated with a module
    """

    def __init__(self, module):
        self.module = module
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
            [display(fig["figure"]) for fig in self.figs]
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
        results = self.module.get_results()
        for key, val in results.items():
            title = format_label(key.upper(), wrap_length=30)
            anchor_name = f'{"-".join(title.split())}'
            display(
                HTML(
                    f'<h3 id="{anchor_name}"><span style="font-size:1em; text-align: left">{title}</span></h3>'
                )
            )
            try:
                val = pd.DataFrame(val)
                display(val)
            except ValueError:
                try:
                    val = pd.Series(val).to_frame()
                    display(val)
                except:
                    print(val)
            print("\n")

    @abstractmethod
    def _create_assets(self):
        """Creates assets, appending them to self.figs"""
        pass

    def _create_chart(
        self, figure, description: str = None, name: str = "Figure", metric_keys=None
    ):
        # if metric keys is not defined but key_lookup exists
        # set metric_keys to all associated metrics
        if self.key_lookup is not None and metric_keys is None:
            metric_keys = self.key_lookup["metric_key"].tolist()
        return {
            "name": name,
            "figure": figure,
            "description": description,
            "metric_keys": metric_keys or [],
        }

    def _create_html_blob(self, html, name: str = "File", metric_keys=None):
        # if metric keys is not defined but key_lookup exists
        # set metric_keys to all associated metrics
        if self.key_lookup is not None and metric_keys is None:
            metric_keys = self.key_lookup["metric_key"].tolist()
        return {
            "name": name,
            "content": html,
            "content_type": "text/html",
            "metric_keys": metric_keys,
        }
