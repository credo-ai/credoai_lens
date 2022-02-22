import math
import matplotlib.pyplot as plt
import os
import seaborn as sns

from credoai.reporting.credo_reporter import CredoReporter
from credoai.reporting import plot_utils
from datetime import datetime


class NLPGeneratorAnalyzerReporter(CredoReporter):
    def __init__(self, assessment, size=5):
        super().__init__(assessment)
        self.num_gen_models = len(self.module.generation_functions)
        self.num_assessment_funs = len(self.module.assessment_functions)
        self.size = size

    def plot_results(self, filename=None, include_fairness=True, include_disaggregation=True):
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
        self.figs.append(self._plot_overall_assessment())
        if include_fairness:
            self.figs.append(self._plot_fairness())
        if include_disaggregation:
            self.figs.append(self._plot_disaggregated_assessment())
        #self.figs.append(self._plot_hists())

        # display
        plt.show()
        # Save to pdf if requested
        if filename:
            self.export_report(filename)
        return self.figs

    def _plot_overall_assessment(self, kind='box'):
        """Plots assessment values for each generator as box plots"""        
        results = self.module.get_results()
        palette = plot_utils.credo_converging_palette(self.num_gen_models)
        n_cols = 2
        n_rows = math.ceil(self.num_assessment_funs/n_cols)

        with plot_utils.get_style(figsize=self.size, figure_ratio = n_rows/n_cols):
            # Generate assessment attribute distribution parameters plots
            f, axes = plt.subplots(n_rows, n_cols)
            to_loop = zip(axes.flat, results.groupby('assessment_attribute'))
            for i, (ax, (assessment_attribute, sub)) in enumerate(to_loop):
                if kind == 'box':
                    sns.boxplot(x="value", y="generation_model", 
                                dodge=True, data=sub, palette=palette,
                                width=.8, linewidth=1, fliersize=1, ax=ax)
                elif kind == 'bar':
                    sns.barplot(x="value", y="generation_model", 
                                dodge=True, data=sub, palette=palette,
                                linewidth=1, ax=ax, errwidth=1)

                sns.despine()
                ax.set_xlabel(assessment_attribute)
                ax.set_ylabel("")
                if i%2:
                    ax.tick_params(labelleft=False) 
            plt.subplots_adjust(wspace=.5, hspace=.5)
            plt.suptitle('Overal Assessment of Text Generators', y=1)
        return f

    def _plot_disaggregated_assessment(self, kind='box'):
        """Plots assessment values for each generator and group as box plots"""  
        palette = plot_utils.credo_converging_palette(self.num_gen_models)      
        results = self.module.get_results()
        # if only one group, can't calculate disaggregated
        if len(results['group'].unique())==1:
            return

        n_cols = 2
        n_rows = math.ceil(self.num_assessment_funs/n_cols)

        with plot_utils.get_style(figsize=self.size, figure_ratio = n_rows/n_cols):
            # Generate assessment attribute distribution parameters plots
            f, axes = plt.subplots(n_rows, n_cols)
            to_loop = zip(axes.flat, results.groupby('assessment_attribute'))
            for i, (ax, (assessment_attribute, sub)) in enumerate(to_loop):
                if kind == 'box':
                    sns.boxplot(x="value", y="group",
                                hue="generation_model", dodge=True,
                                data=sub, palette=palette,
                                width=.8, linewidth=1, ax=ax, fliersize=1)
                elif kind == 'bar':
                    sns.barplot(x="value", y="group",
                                hue="generation_model", 
                                data=sub, palette=palette,
                                linewidth=1, ax=ax, errwidth=1)

                sns.despine()
                ax.set_xlabel(assessment_attribute)
                ax.set_ylabel("")
                if i%2:
                    ax.tick_params(labelleft=False) 
                if i == self.num_assessment_funs-1:
                    ax.legend(bbox_to_anchor=(1.05, 0.9), 
                        title='Text Generator', labelcolor='linecolor')
                else:
                    ax.legend().set_visible(False)
            plt.subplots_adjust(wspace=.1, hspace=.5)
            plt.suptitle('Disaggregated Assessment across Groups', y=1)
        return f

    def _plot_fairness(self):
        palette = plot_utils.credo_converging_palette(self.num_gen_models)
        results = self.module.prepare_results()
        # if only one group, can't calculate fairness
        if len(results['group'].unique())==1:
            return
        # create parity metrics
        parity = results.groupby(['generation_model', 'assessment_attribute'])['mean'] \
                    .agg(['max', 'min']).diff(axis=1)['min'].abs().reset_index()
        parity.rename(columns={'min': 'value'}, inplace=True)

        # plot
        with plot_utils.get_style(figsize=self.size, figure_ratio = 1/self.num_assessment_funs):
            f = plt.figure()
            sns.barplot(x='assessment_attribute', y='value', hue='generation_model', data=parity,
                        palette=palette)
            plt.xlabel("Assessment Attribute")
            plt.ylabel("Min/Max Parity")
            plt.title("Parity of Assessment Attributes across Group")
            sns.despine()
            plt.legend(bbox_to_anchor=(1.05, 0.9), title='Text Generator', labelcolor='linecolor')
        return f

    def _plot_hists(self):
        # generate assessment attribute histogram plots
        results = self.module.get_results()
        palette = plot_utils.credo_converging_palette(self.num_gen_models)
        n_plots = self.num_assessment_funs
        with plot_utils.get_style(figsize=self.size, n_rows=n_plots):
            f, axes = plt.subplots(n_plots, 1)
        to_loop = zip(axes.flat, results.groupby('assessment_attribute'))
        for i, (ax, (assessment_attribute, sub)) in enumerate(to_loop):
            ax = axes.flat[i]
            n_bins = min(sub.shape[0]//4, 20)
            sns.histplot(
                data=sub,
                x="value",
                hue="generation_model",
                element="step",
                stat="density",
                common_norm=False,
                bins=n_bins,
                palette=palette,
                alpha=0.7,
                ax=ax
            )
            sns.despine()
            plt.xlim([0, 1])
            plt.xlabel(assessment_attribute)
        return f