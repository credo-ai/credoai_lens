from credoai.reporting.credo_reporter import CredoReporter


class DatasetProfilingReporter(CredoReporter):
    def __init__(self, module, size=5):
        super().__init__(module)
        self.size = size

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
            self._create_assets()
        if plot:
            self.module.profile_data()
        return self.figs

    def _create_assets(self):
        self.figs = [
            self._create_html_blob(
                self.module.get_html_report(), name="dataset_profile"
            )
        ]

    def display_results_tables(self):
        pass
