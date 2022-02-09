import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from credoai.reporting.credo_report import CredoReport
from credoai.reporting import plot_utils


class DatasetModuleReport(CredoReport):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def create_report(self, filename=None):
        """Creates a fairness report for dataset assessment module

        Parameters
        ----------
        filename : string, optional
            If given, the location where the generated pdf report will be saved, by default Non

        Returns
        -------
        array of figures
        """
        results_all = self.module.get_results()

        # Generate sample balance barplots
        fig, axs = plt.subplots(nrows=2, figsize=(8, 8), dpi=300)
        results = results_all["balance_metrics"]["sample_balance"]
        df = pd.DataFrame(results)
        sensitive_feature_name = list(df.drop(["count", "percentage"], axis=1).columns)[0]

        ax = sns.barplot(
            x=sensitive_feature_name,
            y="count",
            data=df,
            palette=plot_utils.credo_diverging_palette(1),
            alpha=1,
            ax=axs[0],
        )
        fig.patch.set_facecolor("white")
        ax.set_frame_on(False)
        ax.set_title("Data balance across " + sensitive_feature_name + " subgroups")
        ax.set_xlabel("")
        ax.set_ylabel("Number of data samples")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

        # Generate label balance barplots
        results = results_all["balance_metrics"]["label_balance"]
        df = pd.DataFrame(results)
        label_name = list(df.drop([sensitive_feature_name, "count"], axis=1).columns)[0]

        num_classes = df[label_name].nunique()
        ax = sns.barplot(
            x=sensitive_feature_name,
            y="count",
            hue=label_name,
            data=df,
            palette=plot_utils.credo_diverging_palette(num_classes),
            alpha=1,
            ax=axs[1],
        )
        fig.patch.set_facecolor("white")
        ax.set_frame_on(False)
        plt.title(
            "Data balance across "
            + sensitive_feature_name
            + " subgroups and label values"
        )
        plt.xlabel("")
        plt.ylabel("Number of data samples")
        plt.xticks(rotation=90)
        plt.legend(bbox_to_anchor=(1.25, 0.4), loc="center right", frameon=False)
        ax.legend_.set_title(label_name)
        plt.tight_layout()

        self.figs.append(fig)

        # Generate groupd difference barplots
        results = results_all["group_diffs"]
        for k, v in results.items():
            df = pd.DataFrame(v.items(), columns=["feature", "group difference"])
            fig = plt.figure(figsize=(8, 4), dpi=300)
            ax = sns.barplot(
                x="feature",
                y="group difference",
                data=df,
                palette=plot_utils.credo_diverging_palette(1),
                alpha=1,
            )
            fig.patch.set_facecolor("white")
            ax.set_frame_on(False)
            plt.title("Group differences for " + k + " combination across features")
            plt.xlabel("")
            plt.ylabel("Group difference")
            plt.xticks(rotation=90)

            self.figs.append(fig)

        # Generate mutual information barplots
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

        fig = plt.figure(figsize=(8, 4), dpi=300)
        ax = sns.barplot(
            x="feature",
            y="mutual information",
            hue="feature type",
            data=df,
            palette=plot_utils.credo_diverging_palette(2),
            alpha=1,
        )
        fig.patch.set_facecolor("white")
        ax.set_frame_on(False)
        plt.title("Mututal information with " + ref_type + " feature " + ref_name)
        plt.xlabel("")
        plt.ylabel("Mutual information")
        plt.xticks(rotation=90)
        plt.legend(bbox_to_anchor=(1.25, 0.4), loc="center right", frameon=False)

        self.figs.append(fig)

        # Save to pdf if requested
        if filename:
            self.export_report(filename)

        return self.figs
