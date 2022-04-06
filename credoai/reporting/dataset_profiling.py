from credoai.reporting.credo_reporter import CredoReporter


class DatasetProfilingReporter(CredoReporter):
    def __init__(self, assessment, size=5):
        super().__init__(assessment)
        self.size = size

    def plot_results(self):
        self.module.profile_data()

    def display_results_tables(self):
        pass

    def _create_report_cells(self):     
        cells = [("reporter.module.profile_data()", "code")]
        return cells