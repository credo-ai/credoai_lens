import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns

from credoai.reporting.credo_reporter import CredoReporter
from credoai.reporting import plot_utils


class DatasetModuleReporter(CredoReporter):
    def __init__(self, assessment, size=5):
        super().__init__(assessment)
        self.size = size

    def create_report(self, filename=None):
        """Creates a fairness report for dataset assessment

        Parameters
        ----------
        filename : string, optional
            If given, the location where the generated pdf report will be saved, by default Non
        Returns
        -------
        array of figures
        """
        # Generate data balance charts
        self._plot_balance_metrics()

        # Generate group difference charts
        self._plot_group_diff()

        # Generate mutual information charts
        self._plot_mutual_information()

        # display
        plt.show()
        # Save to pdf if requested
        if filename:
            self.export_report(filename)

        return self.figs

    def _plot_balance_metrics(self):
        """Generates data balance charts

        They include:
        - Data balance across sensitive feature subgroups
        - Data balance across sensitive feature subgroups and label values
        - Demographic parity metrics for different preferred label value possibilities
        """
        with plot_utils.get_style(figsize=self.size, rc={'font.size': self.size*1.5}):
            f, axes = plt.subplots(nrows=3)
            plt.subplots_adjust(hspace=1.8)
            results_all = self.module.get_results()

            # Generate sample balance barplots
            results = results_all["sample_balance"]
            df = pd.DataFrame(results)
            sensitive_feature_name = list(df.drop(["count", "percentage"], axis=1).columns)[
                0
            ]
            ax = sns.barplot(
                x="count",
                y=sensitive_feature_name,
                data=df,
                palette=plot_utils.credo_diverging_palette(1),
                ax=axes[0],
            )
            f.patch.set_facecolor("white")
            sns.despine()
            ax.set_title("Data balance across " + sensitive_feature_name + " subgroups")
            ax.set_xlabel("Number of data samples")
            ax.set_ylabel("")

            # Generate label balance barplots
            results = results_all["label_balance"]
            df = pd.DataFrame(results)
            label_name = list(df.drop([sensitive_feature_name, "count"], axis=1).columns)[0]

            num_classes = df[label_name].nunique()
            ax = sns.barplot(
                x="count",
                y=sensitive_feature_name,
                hue=label_name,
                data=df,
                palette=plot_utils.credo_diverging_palette(num_classes),
                alpha=1,
                ax=axes[1],
            )
            f.patch.set_facecolor("white")
            sns.despine()
            ax.set_title(
                "Data balance across "
                + sensitive_feature_name
                + " subgroups and label values"
            )
            ax.set_xlabel("Number of data samples")
            ax.set_ylabel("")
            ax.get_legend().set_visible(False)

            # Generate parity metrics barplots
            metric_keys = ['demographic_parity_difference',
                        'demographic_parity_ratio']

            lst = []
            for metric in metric_keys:
                temp = pd.DataFrame(results_all[metric])
                temp["metric"] = metric.replace("_", " ")
                lst.append(temp)

            df = pd.concat(lst)
            ax = sns.barplot(
                x="value",
                y="metric",
                hue=label_name,
                data=df,
                palette=plot_utils.credo_diverging_palette(num_classes),
                ax=axes[2],
            )
            f.patch.set_facecolor("white")
            sns.despine()
            plt.title("Parity metrics for different preferred label value possibilities")
            plt.xlabel("Value")
            plt.ylabel("")
            plt.legend(
                bbox_to_anchor=(1.2, 0.5), 
                loc="center",
                frameon=False,
                ncol=num_classes,
                title=label_name
            )
            ax.legend_.set_title(label_name)
        self.figs.append(f)

    def _plot_group_diff(self):
        """Generates group difference barplots"""

        results_all = self.module.get_results()
        results = results_all["standardized_group_diffs"]
        abs_sum = -1
        for k, v in results.items():
            diffs = list(v.values())
            abs_sum_new = sum([abs(x) for x in diffs])
            if abs_sum_new > abs_sum:
                max_pair_key, max_pair_values = k, v
                abs_sum_new = abs_sum

        if not max_pair_values:  # do not plot when standardized_group_diffs is empty, which happens when none of the features are numeric 
            return

        with plot_utils.get_style(figsize=self.size, figure_ratio=0.7):
            f, ax = plt.subplots()
            df = pd.DataFrame(max_pair_values.items(), columns=["feature", "group difference"])
            sns.barplot(
                x="feature",
                y="group difference",
                data=df,
                palette=plot_utils.credo_diverging_palette(1),
                alpha=1,
                dodge=False,
            )
            f.patch.set_facecolor("white")
            ax.axhline(0, color="k")
            sns.despine()
            ax.set_title("Group differences for " + max_pair_key)
            ax.set_xlabel("")
            ax.set_ylabel("Group difference")
            ax.xaxis.set_tick_params(rotation=90)

        self.figs.append(f)

    def _plot_mutual_information(self):
        """Generates normalized mutual information between features and sensitive attribute"""

        results_all = self.module.get_results()
        results = results_all["normalized_mutual_information"]
        df = pd.DataFrame.from_dict(results, orient="index").reset_index()
        df = df.rename(
            columns={
                "index": "feature",
                "value": "mutual information",
                "feature_type": "feature type",
            }
        )
        df.sort_values(
            by=["feature type", "mutual information"],
            inplace=True,
            ascending=[True, False],
        )

        ref = df[df["feature type"].str.contains("reference")]
        df = df[~df["feature type"].str.contains("reference")]
        ref_name = ref.iloc[0]["feature"]
        ref_type = ref.iloc[0]["feature type"].split("_")[0]

        with plot_utils.get_style(figsize=self.size, figure_ratio=0.7) as style:
            f, ax = plt.subplots()
            num_types = 2
            sns.barplot(
                x="feature",
                y="mutual information",
                hue="feature type",
                data=df,
                palette=plot_utils.credo_diverging_palette(num_types),
                alpha=1,
                dodge=False,
            )
            f.patch.set_facecolor("white")
            ax.axhline(0, color="k", lw=self.size/6)
            sns.despine()
            ax.set_title(
                "Normalized mututal information\n with " + ref_type + " feature " + ref_name
            )
            ax.set_xlabel("")
            ax.set_ylabel("Normalized mutual information")
            ax.set_ylim([0, 1])
            ax.xaxis.set_tick_params(rotation=90)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            ax.legend(loc='upper right')

        self.figs.append(f)