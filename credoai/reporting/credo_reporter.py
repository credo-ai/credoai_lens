"""
Defines abstract base class for all CredoReports
"""

from abc import ABC, abstractmethod
from credoai.utils.common import ValidationError
from credoai.reporting.reports import AssessmentReport
from IPython.display import display
import os
import pandas as pd
import matplotlib.backends.backend_pdf


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
    
    def export_notebook_report(self, directory):
        report = AssessmentReport()
        # create description
        assessment_description = self.assessment.get_description()
        description = f"""\
    ### {self.assessment.name} Report
    
    #### Description
    
    {assessment_description['short']}

    {assessment_description['long']}

    ### Results
        """
        results_table = ("reporter.display_results()", 'code')
        cells = [(description, 'markdown'), results_table] + self._create_report_cells()
        report.add_cells(cells)
        loc = os.path.join(directory, f'assessment-{self.assessment.name}_report.ipynb')
        report.write_notebook(loc, run=True, reporter=self)
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
        caption_style = {
            'selector': 'caption',
            'props': [
                ('font-weight', 'bold'),
                ('font-size', '16px')
            ]
        }
        for key, val in results.items():
            try:
                to_display=val.style.set_caption(key.upper()).set_table_styles([caption_style])
                display(to_display)
                print('\n\n')
            except:
                pass

