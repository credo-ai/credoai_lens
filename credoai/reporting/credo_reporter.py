"""
Defines abstract base class for all CredoReports
"""

from abc import ABC, abstractmethod
from credoai.reporting.plot_utils import get_table_style, format_label
from credoai.utils.common import ValidationError
from IPython.core.display import display, HTML
import pandas as pd
import matplotlib.backends.backend_pdf
import textwrap


class CredoReporter(ABC):
    """Abstract base class for all CredoReports"""
    
    def __init__(self, assessment):
        self.assessment = assessment
        self.module = assessment.initialized_module
        self.figs = []

    def display_results_tables(self):
        results = self.assessment.get_results()
        for key, val in results.items():
            title = format_label(key.upper(), wrap_length=30)
            display(HTML(f'<h3><span style="font-size:1em; text-align: left">{title}</span></h3>'))
            try:
                val = pd.DataFrame(val)
                display(val)
            except:
                print(val)
            print('\n')

    @abstractmethod
    def create_report(self):
        """ Creates the report """
        pass
    
    def export_report(self, filename):
        pdf = matplotlib.backends.backend_pdf.PdfPages(f"{filename}.pdf")
        for fig in self.figs: ## will open an empty extra figure :(
            pdf.savefig(fig, bbox_inches='tight', pad_inches=1)
        pdf.close()

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

