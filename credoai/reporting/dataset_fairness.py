from turtle import color

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns
from credoai.reporting import plot_utils
from credoai.reporting.credo_reporter import CredoReporter


class DatasetFairnessReporter(CredoReporter):
    def __init__(self, module, size=5):
        super().__init__(module)
        self.size = size
        self.sf_name = self.module.sensitive_features.name

    def _create_assets(self):
        """Creates a fairness dataset assessment assets"""

        # Generate data balance charts
        self._plot_balance_metrics()

        # Generate group difference charts
        self._plot_group_diff()

        # Generate mutual information charts
        self._plot_mutual_information()

    def _plot_balance_metrics(self):
        """Generates data balance charts

        They include:
        - Data balance across sensitive feature subgroups
        - Data balance across sensitive feature subgroups and label values
        - Demographic parity metrics for different preferred label value possibilities
        """

        results_all = self.module.get_results()
        sensitive_features = self.module.sensitive_features
        metric_keys = []

        with plot_utils.get_style(figsize=self.size, rc={"font.size": self.size * 1.5}):
            n_rows = 3 if "label_balance" in results_all else 1
            f, axes = plt.subplots(nrows=n_rows)
            axes = f.get_axes()

            plt.subplots_adjust(hspace=1.8)

            # Generate sample balance barplots
            results = results_all["sample_balance"]
            df = pd.DataFrame(results)
            ax = sns.barplot(
                x="count",
                y=self.sf_name,
                data=df,
                palette=plot_utils.credo_diverging_palette(1),
                ax=axes[0],
            )
            self._add_bar_percentages(ax, self.size * 1.5)

            f.patch.set_facecolor("white")
            sns.despine()
            ax.set_title("Data balance across " + self.sf_name + " subgroups")
            ax.set_xlabel("Number of data samples")
            ax.set_ylabel("")

            # Generate label balance barplots
            if "label_balance" in results_all:
                results = results_all["label_balance"]
                df = pd.DataFrame(results)
                label_name = list(df.drop([self.sf_name, "count"], axis=1).columns)[0]

                num_classes = df[label_name].nunique()
                ax = sns.barplot(
                    x="count",
                    y=self.sf_name,
                    hue=label_name,
                    data=df,
                    palette=plot_utils.credo_diverging_palette(num_classes),
                    alpha=1,
                    ax=axes[1],
                )
                self._add_bar_percentages(ax, self.size * 1.5)
                f.patch.set_facecolor("white")
                sns.despine()
                ax.set_title(
                    f"Data balance across {self.sf_name} subgroups and label values"
                )
                ax.set_xlabel("Number of data samples")
                ax.set_ylabel("")
                ax.get_legend().set_visible(False)

                # Generate parity metrics barplots
                # only using demographic_parity_ratio, ignoring difference
                metric_keys = ["demographic_parity_ratio"]

                lst = []
                for metric in metric_keys:
                    temp = pd.DataFrame(results_all[metric])
                    temp["metric"] = metric
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
                plt.title(
                    "Parity metrics for different preferred label value possibilities"
                )
                plt.xlabel("Value")
                plt.ylabel("")
                plt.legend(
                    bbox_to_anchor=(1.2, 0.5),
                    loc="center",
                    frameon=False,
                    ncol=num_classes,
                    title=label_name,
                )
                ax.legend_.set_title(label_name)

        title = f"Dataset Balance with respect to Sensitive Feature: {self.sf_name}"
        # get metric keys for sensitive feature to append
        if self.key_lookup is not None:
            metric_keys = (
                self.key_lookup.filter(regex="demographic", axis=0)
                .query(f'sensitive_feature=="{self.sf_name}"')["metric_key"]
                .tolist()
            )
        self.figs.append(
            self._create_chart(f, BALANCE_METRICS_DESCRIPTION, title, metric_keys)
        )

    def _plot_group_diff(self):
        """Generates group difference barplots"""

        results_all = self.module.get_results()
        sensitive_features = self.module.sensitive_features
        metric_keys = []
        results = results_all["standardized_group_diffs"]
        abs_sum = -1
        for k, v in results.items():
            diffs = list(v.values())
            abs_sum_new = sum([abs(x) for x in diffs])
            if abs_sum_new > abs_sum:
                max_pair_key, max_pair_values = k, v
                abs_sum = abs_sum_new

        # do not plot when standardized_group_diffs is empty, which happens when none of the features are numeric
        if abs_sum == -1:
            return

        with plot_utils.get_style(figsize=self.size, figure_ratio=0.7):
            f, ax = plt.subplots()
            df = pd.DataFrame(
                max_pair_values.items(), columns=["feature", "group difference"]
            )
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
            title = (
                "Group differences for Sensitive Feature:\n"
                f"{self.sf_name}, ({max_pair_key})"
            )
            ax.set_title(title)
            ax.set_xlabel("")
            ax.set_ylabel("Group difference")
            ax.xaxis.set_tick_params(rotation=90)
        if self.key_lookup is not None:
            metric_keys = (
                self.key_lookup.loc[["sensitive_feature_prediction_score"]]
                .query(f'sensitive_feature=="{self.sf_name}"')["metric_key"]
                .tolist()
            )
        self.figs.append(
            self._create_chart(f, GROUP_DIFF_DESCRIPTION, title, metric_keys)
        )

    def _plot_mutual_information(self):
        """Generates normalized mutual information between features and sensitive attribute"""

        results_all = self.module.get_results()
        sensitive_features = self.module.sensitive_features
        metric_keys = []
        results = results_all["proxy_mutual_information"]
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
            ax.axhline(0, color="k", lw=self.size / 6)
            sns.despine()
            title = "Proxy Detection with Sensitive Feature: " + self.sf_name
            ax.set_title(title)
            ax.set_xlabel("")
            ax.set_ylabel("Normalized\nmutual information")
            ax.set_ylim([0, 1])
            ax.xaxis.set_tick_params(rotation=90)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            ax.legend(loc="upper right")
        if self.key_lookup is not None:
            metric_keys = (
                self.key_lookup.loc[
                    [
                        "sensitive_feature_prediction_score",
                        "max_proxy_mutual_information",
                    ]
                ]
                .query(f'sensitive_feature=="{self.sf_name}"')["metric_key"]
                .tolist()
            )
        self.figs.append(
            self._create_chart(f, MUTUAL_INFO_DESCRIPTION, title, metric_keys)
        )

    def _add_bar_percentages(self, ax, fontsize=10):
        n_containers = len(ax.containers)
        bar_groups = list(zip(*ax.containers))
        totals = [sum([c.get_width() for c in containers]) for containers in bar_groups]
        overall_total = sum(totals)
        if n_containers == 1:
            totals = [overall_total for i in totals]
        for containers in ax.containers:
            widths = [c.get_width() for c in containers]
            percentages = [100 * w / totals[i] for i, w in enumerate(widths)]
            overall_percentages = [
                100 * w / overall_total for i, w in enumerate(widths)
            ]
            percentage_text = [f"{i:.1f}%" for i in percentages]
            if min(overall_percentages) > 10:
                ax.bar_label(
                    containers,
                    labels=percentage_text,
                    color="white",
                    label_type="center",
                    fontsize=fontsize / n_containers,
                )
            else:
                ax.bar_label(
                    containers,
                    labels=percentage_text,
                    color=plot_utils.credo_diverging_palette(1)[0],
                    padding=2,
                    fontsize=fontsize / n_containers,
                )


BALANCE_METRICS_DESCRIPTION = """
Data Balance
------------
The data balance assessment helps us gain insights
into how different subgroups are represented in the 
dataset. Data balance is particularly important for
datasets used to train models, as models generally show some
form of bias towards the most represented group.

For validation datasets it is important that each important subgroup is 
adequately represented, but parity is not necessarily required.
However, if subgroups are imbalanced, it is imperative
that performance measures are disaggregated across subgroups.

Plot Description
----------------
The first plot shows the number of samples for each
subgroup in the dataset.

The second plot shows how the outcome distribution differs
between subgroups.

The third plot summarizes label disparities by calculating
the demographic parity. This metric compares the proportion 
of samples a group is given a particular label to other groups.
Ideally, this value is 1. We calculate this value for each outcome label.
Typically, one is concerned with the demographic parity of outcomes that are either:

    -beneficial vs. the status quo
    -harmful vs. the status quo
    -rarer

That is to say, your AI system probably does something to people. This
plot helps you evaluate whether it is equitable in its actions.

"""

SENSITIVE_FEATURE_DESCRIPTION = """
Redundant Encoding
------------------
The most important thing to check about your dataset is
"does it redundantly code a sensitive feature". Redundant encoding
means that the sensitive feature can be reconstructed from the features 
in your dataset. If it can be reconstructed, this means that your AI system
is implicitly trained on the sensitive feature, even if it isn't explicitly included
in the dataset.

To evaluate this, we train a model that tries to predict the sensitive feature from the
dataset. The score ranges from 0.5 - 1.0. If the score is 0.5, the model is random, and
no information about the sensitive feature is likely contained in the dataset. A value
of 1 means the sensitive feature is able to be perfectly reconstructed.
"""

GROUP_DIFF_DESCRIPTION = """
Feature Balance
---------------
Though potentially less important than balance of the
primary outcome, feature differences are also worth evaluating.

While some differences amongst groups should be expected, large deviations
are problematic. One of the main issues is that they may lead
to your dataset redundantly encoding sensitive features. In other
words, features that differ significantly between groups act as proxies
for the sensitive feature.
"""

MUTUAL_INFO_DESCRIPTION = """
Feature Proxy Detection
-----------------------
Feature Balance serves as a simple descriptive analysis
of sensitive feature parity. A more rigorous method is to calculate
the between the features and the sensitive feature.

Higher values mean there is more information about the sensitive feature
encoded in the feature. We normalize the mutual information by the amount of information
the sensitive feature has to itself. Thus this metric goes from 0-1, where 1 means 
the feature is a perfect proxy of the sensitive feature.

Removing such a feature is advised!
"""
