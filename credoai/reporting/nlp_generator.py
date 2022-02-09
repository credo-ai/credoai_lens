import matplotlib.pyplot as plt
import os
import seaborn as sns

from credoai.reporting.credo_report import CredoReport
from credoai.reporting import plot_utils
from datetime import datetime


class NLPGeneratorAnalyzerReport(CredoReport):
    def __init__(self, module, size=5):
        super().__init__()
        self.module = module
        self.num_gen_models = len(module.generation_functions)
        self.num_assessment_funs = len(module.assessment_functions)
        self.palette = plot_utils.credo_diverging_palette(self.num_gen_models)
        self.size = size

    def create_report(self, filename=None):
        """Creates a fairness report for binary classification model

        Parameters
        ----------
        filename : string, optional
            If given, the location where the generated pdf report will be saved, by default Non

        Returns
        -------
        array of figures
        """
        # Generate assessment attribute distribution parameters plots
        self.figs.append(self._plot_boxplots)

        # Save to pdf if requested
        if filename:
            self.export_report(filename)

        return self.figs

    def _plot_boxplots(self):
        with plot_utils.get_style(figure_ratio = 1/self.num_assessment_funs):
            # Generate assessment attribute distribution parameters plots
            fig = plt.figure()
            sns.boxplot(x="assessment_attribute", y="value",
                        hue="generation_model", dodge=True,
                        data=results_all, palette=self.palette,
                        width=.8, linewidth=2)
            
            sns.despine()
            plt.xlabel("")
            plt.ylabel("Value")
            plt.legend(bbox_to_anchor=(1.05, 0.9))
        return fig

    def _plot_hists(self):
        # generate assessment attribute histogram plots
        results_all = self.module.get_results()
        n_plots = self.num_assessment_funs
        with plot_utils.get_style(figure_ratio = n_plots/2):
            f, axes = plt.subplots(n_plots, 1)
        axes = f.get_axes()
        for i, (assessment_attribute, results_sub) in enumerate(results_all.groupby('assessment_attribute')):
            ax = axes[i]
            n_bins = min(results_sub.shape[0]//4, 20)
            sns.histplot(
                data=results_sub,
                x="value",
                hue="generation_model",
                element="step",
                stat="density",
                common_norm=False,
                bins=n_bins,
                palette=self.palette,
                alpha=0.7,
                ax=ax
            )
            sns.despine()
            plt.xlim([0, 1])
            plt.xlabel(assessment_attribute)
        return fig