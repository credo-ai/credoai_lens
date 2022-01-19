"""
Defines abstract base class for all CredoReports
"""

from abc import ABC, abstractmethod
from credoai.utils.common import ValidationError
import pandas as pd
import matplotlib.backends.backend_pdf


class CredoReport(ABC):
    """Abstract base class for all CredoReports"""
    
    def __init__(self):
        self.figs = []

    @abstractmethod
    def create_report(self):
        """ Creates the report """
        pass
    
    def export_report(self, filename):
        pdf = matplotlib.backends.backend_pdf.PdfPages(f"{filename}.pdf")
        for fig in self.figs: ## will open an empty extra figure :(
            pdf.savefig(fig, bbox_inches='tight', pad_inches=1)
        pdf.close()

