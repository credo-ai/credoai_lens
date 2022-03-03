"""
Defines abstract base class for all CredoReports
"""

from abc import ABC, abstractmethod
from credoai.reporting.plot_utils import get_table_style, format_label
from credoai.utils.common import ValidationError
from credoai.reporting.plot_utils import get_table_style, format_label
from credoai.reporting.reports import AssessmentReport
from IPython.core.display import display, HTML
import os
import pandas as pd
import matplotlib.backends.backend_pdf
import textwrap

class CredoReporter(ABC):
    """Abstract base class for all CredoReports"""
    
    def __init__(self, assessment):
        self.assessment = assessment
        self.module = assessment.initialized_module
        self.report = None
        self.figs = []

    def create_notebook(self):
        report = AssessmentReport({'reporter': self})
        results_table = [(f"### {self.assessment.name} Result Tables", "markdown"), 
                         ("reporter.display_results_tables()", 'code')]
        cells = [(self._get_description(), 'markdown')] \
            + self._create_report_cells() \
            + results_table
        report.add_cells(cells)
        self.report = report

    def display_results_tables(self):
        results = self.assessment.get_results()
        for key, val in results.items():
            title = format_label(key.upper(), wrap_length=30)
            anchor_name = f'{self.assessment.name}-{"-".join(title.split())}'
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

    def export_report(self, filename):
        pdf = matplotlib.backends.backend_pdf.PdfPages(f"{filename}.pdf")
        for fig in self.figs: ## will open an empty extra figure :(
            pdf.savefig(fig, bbox_inches='tight', pad_inches=1)
        pdf.close()

    def plot_results(self):
        """ Plots results """
        pass

    @abstractmethod
    def _create_report_cells(self):
        """Exports cells required for creating report
        
        Each code cell should reference "reporter" if calling a self method
        """
        pass

    def _get_description(self):
        assessment_description = self.assessment.get_description()
        description = f"""\
        <hr style="border:2px solid #3b07b4"> </hr>
        ## {self.assessment.name} Report
        
        #### Description
        
        {assessment_description['short']}

        {textwrap.indent(assessment_description['long'], ' '*4)}

        ### {self.assessment.name} Results
        """
        return description






