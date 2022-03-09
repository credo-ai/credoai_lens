import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns

from credoai.reporting.credo_reporter import CredoReporter
from credoai.reporting import plot_utils


class DatasetProfilingReporter(CredoReporter):
    def __init__(self, assessment, size=5):
        super().__init__(assessment)
        self.size = size

    def _create_report_cells(self):
        cells = [("reporter.profile_data()", "code")]