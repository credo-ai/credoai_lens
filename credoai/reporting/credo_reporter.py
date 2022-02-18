"""
Defines abstract base class for all CredoReports
"""

from abc import ABC, abstractmethod
from credoai.utils.common import ValidationError
from credoai.reporting.plot_utils import get_table_style, format_label
from credoai.reporting.reports import AssessmentReport
from IPython.display import display
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

    @abstractmethod
    def create_report(self):
        """ Creates the report """
        pass
    
    def _get_description(self):
        assessment_description = self.assessment.get_description()
        description = f"""\
        <hr style="border:2px solid #3b07b4"> </hr>

        ## {self.assessment.name} Report
        
        #### Description
        
        {assessment_description['short']}

        {textwrap.indent(assessment_description['long'], ' '*4)}

        ### Results
        """
        return description

    def create_notebook(self):
        report = AssessmentReport({'reporter': self})
        results_table = [("### Result Tables", "markdown"), 
                        ("reporter.display_results()", 'code')]
        cells = [(self._get_description(), 'markdown')] \
            + self._create_report_cells() \
            + results_table
        report.add_cells(cells)
        self.report = report

    @abstractmethod
    def _create_report_cells(self):
        """Exports cells required for creating report
        
        Each code cell should reference "reporter" if calling a self method
        """
        pass

    def export_report(self, filename):
        pdf = matplotlib.backends.backend_pdf.PdfPages(f"{filename}.pdf")
        for fig in self.figs: ## will open an empty extra figure :(
            pdf.savefig(fig, bbox_inches='tight', pad_inches=1)
        pdf.close()

    def display_results(self):
        results = self.assessment.get_results()
        styles = get_table_style()
        for key, val in results.items():
            try:
                title = format_label(key.upper(), wrap_length=30)
                val = pd.DataFrame(val)
                to_display=val.style.set_caption(title)\
                                .set_table_styles(styles)
                display(to_display)
            except:
                print(f'{key}: {val}')
            print('\n')


