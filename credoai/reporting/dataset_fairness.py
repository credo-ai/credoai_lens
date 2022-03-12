import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns

from credoai.reporting.credo_reporter import CredoReporter
from credoai.reporting import plot_utils


class DatasetFairnessReporter(CredoReporter):
    def __init__(self, assessment, size=5):
        super().__init__(assessment)
        self.size = size

    def plot_results(self, filename=None):
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

    def _create_report_cells(self):
        # report cells
        cells = [
            self._write_balance_metrics(),
            ("reporter._plot_balance_metrics();", 'code'),
            self._write_sensitive_feature_prediction(),
            self._write_group_diff(),
            ("reporter._plot_group_diff();", 'code'),
            self._write_mutual_information(),
            ("reporter._plot_mutual_information();", 'code'),
        ]
        return cells

    def _write_balance_metrics(self):
        cell = ("""
                #### Data Balance

                <details>
                <summary>Assessment Description:</summary>
                <br>
                <p>The data balance assessment helps us gain insights
                into how different subgroups are represented in the 
                dataset. Data balance is particularly important for
                datasets used to train models, as models generally show some
                form of <a href="https://medium.com/@mrtz/how-big-data-is-unfair-9aa544d739de">bias towards the most represented group</a>. 
                </p>

                <p>For validation datasets
                it is important that each important subgroup is 
                adequately represented, but parity is not necessarily required.
                However, if subgroups <em>are</em> imbalanced, it is imperative
                that performance measures are disaggregated across subgroups.</p>
                </details><br>

                <details>
                <summary>Plot Descriptions:</summary>
                <br>
                <p>The first plot shows the number of samples for each
                subgroup in the dataset.</p>

                <p>The second plot shows how the outcome distribution differs
                between subgroups.</p>

                <p>The third plot summarizes label disparities by calculating
                the <a href="https://afraenkel.github.io/fairness-book/content/05-parity-measures.html#demographic-parity">demographic parity]</a>. This metric compares the proportion 
                of samples a group is given a particular label to other groups.
                Ideally, this value is 1. We calculate this value for each outcome label.
                Typically, one is concerned with the demographic parity of outcomes that are either:</p>

                <ul>
                    <li>beneficial vs. the status quo</li>
                    <li>harmful vs. the status quo</li>
                    <li>rarer</li>
                </ul>

                <p>That is to say, your AI system probably <em>does something</em> to people. This
                plot helps you evaluate whether it is equitable in its actions.</p>
                </details>
                """, 'markdown')
        return cell

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
            # only using demographic_parity_ratio, ignoring difference
            metric_keys = ['demographic_parity_ratio']

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

    def _write_sensitive_feature_prediction(self):
        score = self.module.get_results()['sensitive_feature_prediction_score']
        cell = (f"""
                #### Redundant Encoding
            
                <details open>
                <summary>Assessment Description:</summary>
                <br>
                <p>The most important thing to check about your dataset is
                "does it redundantly code a sensitive feature". Redundant encoding
                means that the sensitive feature can be <em>reconstructed</em> from the features 
                in your dataset. If it can be reconstructed, this means that your AI system
                is implicitly trained on the sensitive feature, <em>even if it isn't explicitly included
                in the dataset</em>.</p>

                <p>To evaluate this, we train a model that tries to predict the sensitive feature from the
                dataset. The score ranges from 0.5 - 1.0. If the score is 0.5, the model is random, and
                no information about the senstive feature is likely contained in the dataset. A value
                of 1 means the sensitive feature is able to be perfectly reconstructed.</p>

                <p>The <a href="#Feature-Balance>Feature Balance</a> and <a href="#Feature-Proxy-Detection>Feature Proxy Detection</a>
                sections each provide additional perspective by diving into whether
                individual features serve as proxies. Note that the overall dataset can be a 
                proxy even if no individual feature is! That's where this score is important.</p>
                
                </details>

                **Overall Proxy Score**: {score:.4f}
                """, 'markdown')
        return cell

    def _write_group_diff(self):
        cell = ("""
                #### Feature Balance
            
                <details>
                <summary>Assessment Description:</summary>
                <br>
                
                <pr>Though potentially less important than balance of the
                primary outcome, feature differences are also worth evaluating.</pr>

                <pr>While some differences amongst groups should be expected, large deviations
                are problematic. One of the main issues is that they may lead
                to your dataset <em>redundantly encoding</em> sensitive features. In other
                words, features that differ significantly between groups act as proxies
                for the sensitive feature.</pr>
                </details>
                """, 'markdown')
        return cell

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

    def _write_mutual_information(self):
        cell = ("""
                #### Feature Proxy Detection
            
                <details open>
                <summary>Assessment Description:</summary>
                <br>
                The previous plot served as a simple descriptive analysis
                of sensitive feature parity. A more rigorous method is to calculate
                the <a href="https://simple.wikipedia.org/wiki/Mutual_information">mutual information</a> between
                the features and the sensitive feature.
                
                </details><br>

                <details>
                <summary>Plot Description:</summary>
                <br>
                <p>Higher values mean there is more information about the sensitive feature
                encoded in the feature. We normalize the mutual information by the amount of information
                the sensitive feature has <em>to itself</em>. Thus this metric goes from 0-1, where 1 means 
                the feature is a perfect proxy of the sensitive feature.</p>

                <p>Removing such features is advised!</p>
                </details>
                """, 'markdown')
        return cell

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
                "Normalized mututal information\n with feature: " + self.module.sensitive_features.name
            )
            ax.set_xlabel("")
            ax.set_ylabel("Normalized mutual information")
            ax.set_ylim([0, 1])
            ax.xaxis.set_tick_params(rotation=90)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            ax.legend(loc='upper right')
        self.figs.append(f)

