import matplotlib.pyplot as plt
import os
import seaborn as sns

from credoai.reporting.credo_report import CredoReport
from credoai.reporting import plot_utils
from datetime import datetime


class NLPGeneratorAnalyzerReport(CredoReport):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def create_report(self, filename=None, dirname=None):
        """Creates a fairness report for binary classification model

        Parameters
        ----------
        filename : string, optional
            If given, the location where the generated pdf report will be saved, by default None
        dirname : string, optional
            If given, the directory where the generated images and tables will be saved, by default None

        Returns
        -------
        array of figures
        """
        results_all = self.module.get_results()

        # generate assessment attribute histogram plots
        # results_all = self.results
        num_gen_models = results_all["generation_model"].nunique()
        for assessment_attribute in list(results_all["assessment_attribute"].unique()):
            f = plt.figure(figsize=(8, 4), dpi=300)
            plt.rcParams['font.size'] = 12
            results_sub = results_all[
                (results_all["assessment_attribute"] == assessment_attribute)
            ][["value", "generation_model"]]
            results_sub.reset_index(drop=True, inplace=True)
            ax = sns.histplot(
                data=results_sub,
                x="value",
                hue="generation_model",
                element="step",
                stat="density",
                common_norm=False,
                bins=20,
                palette=plot_utils.credo_diverging_palette(num_gen_models),
                alpha=0.7,
            )
            ax.set_frame_on(False)
            plt.xlim([0, 1])
            plt.xlabel(assessment_attribute)
            self.figs.append(f)
            if dirname:
                dtstr = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
                fpath = os.path.join(
                    dirname,
                    "lens_histogram_" + assessment_attribute + "_" + dtstr + ".png",
                )
                plt.savefig(fpath, bbox_inches="tight", dpi=300)

        # Generate assessment attribute distribution parameters plots
        fig = plt.figure(figsize=(8, 4), dpi=300)
        plt.rcParams['font.size'] = 12
        ax = sns.barplot(
            x="assessment_attribute",
            y="value",
            hue="generation_model",
            data=results_all,
            palette=plot_utils.credo_diverging_palette(num_gen_models),
            alpha=1,
        )
        fig.patch.set_facecolor("white")
        ax.set_frame_on(False)
        plt.xlabel("")
        plt.ylabel("mean")
        plt.legend(bbox_to_anchor=(1.25, 0.4), loc="center right", frameon=False)
        self.figs.append(fig)
        if dirname:
            fpath = os.path.join(dirname, "lens_barchart_" + dtstr + ".png")
            plt.savefig(fpath, bbox_inches="tight", dpi=300)

        # Save the tables if requested
        if dirname:
            fpath = os.path.join(dirname, "lens_summary_table_" + dtstr + ".csv")
            self.module.prepare_results().to_csv(fpath, index=False)

            fpath = os.path.join(dirname, "lens_results_table_" + dtstr + ".csv")
            results_all.to_csv(fpath, index=False)

        # Save to pdf if requested
        if filename:
            self.export_report(filename)

        return self.figs
