import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn.metrics as sk_metrics
from credoai.reporting.credo_reporter import CredoReporter
from credoai.reporting.plot_utils import (
    DEFAULT_COLOR,
    credo_classification_palette,
    credo_diverging_palette,
    format_label,
    get_axis_size,
    get_style,
)
from numpy import pi


class FairnessReporter(CredoReporter):
    def __init__(self, module, size=3):
        super().__init__(module)
        self.size = size

    def _create_assets(self):
        """Creates fairness reporting assets"""
        # plot
        self.plot_fairness()

    def plot_fairness(self):
        """Plots fairness for binary classification
        Creates plots for binary classification model that summarizes
        performance disparities across groups. Individual group
        performance plots are also relevant for fully describing
        performance differences.
        Returns
        -------
        matplotlib figure
        """
        plot_disaggregated = False
        if self.module.metric_frames != {}:
            plot_disaggregated = True
            r = self.module.get_results()["disaggregated_performance"].shape
            ratio = max(r[0] * r[1] / 30, 1)
        else:
            ratio = 1
        # ratio based on number of metrics and sensitive features
        metric_keys = []
        sensitive_features = self.module.sensitive_features
        sf_name = sensitive_features.name

        with get_style(figsize=self.size, figure_ratio=ratio):
            # plot fairness
            if self.key_lookup is not None:
                metric_keys = self.key_lookup.query(
                    f'subtype == "parity" and sensitive_feature == "{sf_name}"'
                )["metric_key"].tolist()
            f = self._plot_fairness_metrics()
            self.figs.append(
                self._create_chart(
                    f,
                    FAIRNESS_DESCRIPTION,
                    f"Fairness metrics for Sensitive Feature: {sf_name.title()}",
                    metric_keys=metric_keys,
                )
            )
            if plot_disaggregated:
                if self.key_lookup is not None:
                    metric_keys = self.key_lookup.query(
                        f'subtype == "disaggregated_performance" and sensitive_feature == "{sf_name}"'
                    )["metric_key"].tolist()
                f = self._plot_disaggregated_metrics()
                self.figs.append(
                    self._create_chart(
                        f,
                        DISAGGREGATED_DESCRIPTION,
                        f"Disaggregated metrics for Sensitive Feature: {sf_name.title()}",
                        metric_keys,
                    )
                )

    def _plot_fairness_metrics(self):
        # create df
        df = self.module.get_fairness_results()
        df.drop(["sensitive_feature"], inplace=True, axis=1)
        # add parity to names
        df.index = [
            i + "_parity" if row["subtype"] == "parity" else i
            for i, row in df.iterrows()
        ]
        df = df["value"]
        df.index.name = "Fairness Metric"
        df.name = "Value"
        # plot
        f, ax = plt.subplots()
        sns.barplot(
            data=df.reset_index(),
            y="Fairness Metric",
            x="Value",
            edgecolor="w",
            color=DEFAULT_COLOR,
            ax=ax,
        )
        self._style_barplot(ax)
        plt.title(
            f"Fairness Metrics for Sensitive Feature: {self.module.sensitive_features.name.title()}",
            fontweight="bold",
        )
        return f

    def _plot_disaggregated_metrics(self):
        # create df
        sf_name = self.module.sensitive_features.name
        df = self.module.get_disaggregated_performance()["disaggregated_performance"]
        df = df.reset_index().melt(
            id_vars=["subtype", sf_name],
            var_name="Performance Metric",
            value_name="Value",
        )
        # plot
        num_cats = len(df[sf_name].unique())
        palette = sns.color_palette("Purples", num_cats)
        palette[-1] = [0.4, 0.4, 0.4]
        f, ax = plt.subplots()
        sns.barplot(
            data=df,
            y="Performance Metric",
            x="Value",
            hue=sf_name,
            palette=palette,
            edgecolor="w",
            ax=ax,
        )
        self._style_barplot(ax)
        plt.legend(bbox_to_anchor=(1.2, 0.5), loc="center")
        plt.title(
            "Disaggregated Performance for Sensitive Feature: " f"{sf_name.title()}",
            fontweight="bold",
        )
        return f

    def _style_barplot(self, ax):
        sns.despine()
        ax.set_xlabel(ax.get_xlabel(), fontweight="bold")
        ax.set_ylabel(ax.get_ylabel(), fontweight="bold")
        # format metric labels
        ax.set_yticklabels(
            [format_label(label.get_text()) for label in ax.get_yticklabels()]
        )


class BinaryClassificationReporter(FairnessReporter):
    def __init__(self, module, infographic_shape=(3, 5), size=3):
        super().__init__(module, size)
        self.infographic_shape = infographic_shape

    def _create_assets(self):
        """Creates fairness reporting assests for binary classification"""

        # plot
        # comparison plots. Will fail for performance assessment
        try:
            self.plot_fairness()
        except:
            pass

        # individual group performance plots
        self.plot_performance_infographics()

    def plot_performance_infographics(self):
        df = self.module.get_df()
        if df["true"].dtype.name == "category":
            df["true"] = df["true"].cat.codes
        metric_keys = []
        if self.key_lookup is not None:
            metric_keys = self.key_lookup.query('subtype=="overall_performance"')[
                "metric_key"
            ].tolist()
        self.figs.append(
            self._plot_performance_infographic(
                df["true"], df["pred"], "Overall", metric_keys
            )
        )
        # plot for individual sensitive groups if they exist
        sf_name = self.module.sensitive_features.name
        if sf_name == "NA":
            return
        for group, sub_df in df.groupby(sf_name):
            if self.key_lookup is not None:
                metric_keys = self.key_lookup[self.key_lookup[sf_name] == group][
                    "metric_key"
                ].tolist()
            self.figs.append(
                self._plot_performance_infographic(
                    sub_df["true"], sub_df["pred"], group, metric_keys
                )
            )

    def _plot_performance_infographic(
        self, y_true, y_pred, label, metric_keys=None, **grid_kwargs
    ):
        """Plots performance for binary classification
        Plots "infographic" depiction of outcomes for ground truth
        and model performance, as well as a confusion matrix.
        Parameters
        ----------
        y_true : (List, pandas.Series, numpy.ndarray)
            The ground-truth labels (for classification) or target values (for regression).
        y_pred : (List, pandas.Series, numpy.ndarray)
            The predicted labels for classification
        label : str
            super title for set of performance plots
        metric_keys : list
            metric_keys to associate with chart
        Returns
        -------
        matplotlib figure
        """
        true_data, pred_data = self._create_data(y_true, y_pred)
        # plotting
        ratio = self.infographic_shape[0] / self.infographic_shape[1]
        with get_style(figsize=self.size, figure_ratio=ratio, n_cols=3):
            f, [true_ax, pred_ax, confusion_ax] = plt.subplots(1, 3)
            self._plot_grid(true_data, true_ax, self.size, **grid_kwargs)
            self._plot_grid(pred_data, pred_ax, self.size, **grid_kwargs)
            self._plot_confusion_matrix(y_true, y_pred, confusion_ax, self.size / 2)
            # add text
            true_ax.set_title("Ground Truth", pad=0)
            pred_ax.set_title("Model Predictions", pad=0)
            confusion_ax.set_title(
                "Confusion Matrix",
            )
            for ax, rate in [(true_ax, y_true.mean()), (pred_ax, y_pred.mean())]:
                ax.text(
                    0.5,
                    0,
                    f"Positive Rate = {rate: .2f}",
                    transform=ax.transAxes,
                    ha="center",
                )

            # other text objects
            text_objects = [
                (0.5, 1, label, {"fontweight": "bold", "fontsize": self.size * 6})
            ]
            text_ax = self._plot_text(f, text_objects)
        return self._create_chart(
            f,
            INFOGRAPHIC_DESCRIPTION,
            f"{label} Binary Classification Infographic",
            metric_keys,
        )

    def _create_data(self, y_true, y_pred):
        n = self.infographic_shape[0] * self.infographic_shape[1]
        true_pos_n = int(np.mean(y_true) * n)
        pred_pos_n = int(np.mean(y_pred) * n)
        true_data = np.reshape(
            [1] * true_pos_n + [0] * (n - true_pos_n), self.infographic_shape
        )
        pred_data = np.reshape(
            [1] * pred_pos_n + [0] * (n - pred_pos_n), self.infographic_shape
        )
        return true_data, pred_data

    def _plot_circles(self, data, ax, colors, marker="o"):
        n_rows, n_cols = data.shape

        # set up point locations
        y, x = np.unravel_index(np.arange(data.size), data.shape)
        y = y + 0.5
        x = x + 0.5

        # set up limits
        # final touches
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlim([-0.5, n_cols + 0.5])
        ax.set_ylim([-0.5, n_rows + 0.5])
        ax.invert_yaxis()

        # set up size
        # radius in data coordinates:
        r = 0.2
        # radius in display coordinates:
        r_ = ax.transData.transform([r, 0])[0] - ax.transData.transform([0, 0])[0]
        # marker size as the area of a circle
        markersize = pi * r_ ** 2

        # plotting
        ax.scatter(x, y, marker=marker, s=markersize, c=colors)
        sns.despine(left=True, bottom=True)

    def _plot_grid(
        self,
        data,
        ax,
        size,
        marker="circle",
        palette=credo_classification_palette(),
        sort="sorted",
    ):
        n_rows, n_cols = data.shape
        # set up positive groups
        if sort == "shuffle":
            np.random.shuffle(data)
        elif sort == "sorted":
            data = np.reshape(np.sort(data.flatten()), data.shape)
        # plot figure
        if marker == "circle":
            colors = np.array([palette[i] for i in data.flatten()], dtype=object)
            self._plot_circles(data, ax, colors)
        else:
            sns.heatmap(data, cbar=False, linewidth=3, cmap=palette, ax=ax)
        # success plotting
        success_colors = [[1, 1, 1, 0], [1, 1, 1, 1]]
        success_colors = np.array(
            [success_colors[i] for i in data.flatten()], dtype=object
        )
        self._plot_circles(data, ax, success_colors, marker="$\checkmark$")

        # failure plotting
        failure_colors = [[0, 0, 0], [1, 1, 1, 0]]
        failure_colors = np.array(
            [failure_colors[i] for i in data.flatten()], dtype=object
        )
        self._plot_circles(data, ax, failure_colors, marker="x")

        ax.set_xticks([])
        ax.set_yticks([])
        return ax

    def _plot_confusion_matrix(self, y_true, y_pred, ax, size):
        mat = sk_metrics.confusion_matrix(y_true, y_pred, normalize="true")
        sns.heatmap(
            mat,
            square=True,
            cbar=False,
            linewidth=size / 2,
            xticklabels=["Negative", "Positive"],
            annot=True,
            fmt=".1%",
            cmap="Purples",
            annot_kws={"fontsize": size * 6},
            ax=ax,
        )
        ax.set_yticklabels(
            ["Negative", "Positive"], va="center", rotation=90, position=(0, 0.28)
        )
        ax.tick_params(labelsize=size * 5, length=0, pad=size / 2)
        # labels
        ax.text(
            -0.2,
            0.5,
            "Ground Truth",
            fontsize=size * 5,
            va="center",
            rotation=90,
            transform=ax.transAxes,
            fontweight="bold",
        )
        ax.text(
            0.5,
            -0.2,
            "Prediction",
            fontsize=size * 5,
            ha="center",
            transform=ax.transAxes,
            fontweight="bold",
        )

        # TPR, FPR, labels
        labels = "TN", "FN", "FP", "TP"
        locations = [(0.5, 0.2), (1.5, 0.2), (0.5, 1.2), (1.5, 1.2)]
        for label, location in zip(labels, locations):
            t = ax.text(
                location[0],
                location[1],
                label,
                ha="center",
                fontsize=size * 3,
                color="k",
            )
            t.set_bbox(
                dict(
                    facecolor="w", alpha=1, edgecolor="white", boxstyle="square,pad=.5"
                )
            )

    def _plot_text(self, f, text_objects):
        ax = f.add_axes([0, 0, 1, 1])
        sns.despine(left=True, bottom=True)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.patch.set_alpha(0.01)
        for x, y, s, kwargs in text_objects:
            ax.text(x, y, s, transform=ax.transAxes, ha="center", **kwargs)
        return ax


class RegressionReporter(FairnessReporter):
    def __init__(self, module, size=3):
        super().__init__(module, size)

    def _create_assets(self):
        """Creates fairness reporting assets for regression"""
        self.plot_fairness()
        self.plot_true_vs_pred_scatter()

    def plot_true_vs_pred_scatter(self, sampling_size=200):
        """generates disaggregated scatter plot

        Parameters
        ----------
        sampling_size : int
            the upper limit on the number of data points to plot by sampling without replacement

        Returns
        -------
        matplotlib figure
        """
        sensitive_feature = self.module.sensitive_features
        df = (
            self.module.get_df()
            .groupby(sensitive_feature)
            .apply(lambda x: x.sample(min(sampling_size, len(x)), random_state=10))
            .reset_index(drop=True)
        )
        y_true, y_pred = df["true"], df["pred"]
        num_cats = len(df[sensitive_feature.name].unique())
        palette = credo_diverging_palette(num_cats)
        with get_style(figsize=self.size, figure_ratio=0.7):
            f, ax = plt.subplots()
            p1 = min(min(y_pred), min(y_true))
            p2 = max(max(y_pred), max(y_true))
            plt.plot([p1, p2], [p1, p2], ":", color=DEFAULT_COLOR, linewidth=0.6)
            sns.scatterplot(
                x="true",
                y="pred",
                hue=sensitive_feature,
                style=sensitive_feature,
                palette=palette,
                data=df,
                alpha=1,
                s=10,
            )
            ax.set_title(
                f"Disaggregated Performance for Sensitive Feature: {sensitive_feature.name.title()}"
            )
            ax.set_xlabel("True Values")
            ax.set_ylabel("Predicted Values")
            ax.legend_.set_title("")
        self.figs.append(
            self._create_chart(f, REGRESSION_DESCRIPTION, "Continuous Value Prediction")
        )


FAIRNESS_DESCRIPTION = """The fairness assessment is divided into two primary metrics: (1) Fairness
metrics, and (2) performance metrics. The former help describe how equitable
your AI system, while the latter describes how performant the system is.
                    
Fairness metrics summarize whether your AI system is performing similarly across all groups.
These metrics may well-known "fairness metrics" like "equal opportunity", 
or performance parity metrics. Performance parity captures the idea that the
AI system should work similarly well for all groups. Some "fairness metrics"
like equal opportunity are actually parity metrics. "Equal opportunity" is simply
the true positive rate parity.
"""

DISAGGREGATED_DESCRIPTION = """Performance metrics describe how performant your system is. It goes without saying
that the AI system should be performing at some minimum acceptable level to be
deployed. This figure disaggregates performance across the 
sensitive feature provided. This ensures that the system is evaluated for 
acceptable performance across groups that are important. Think of it as any
segmentation analysis, where the segments are groups of people.
"""

INFOGRAPHIC_DESCRIPTION = """Positive Rate Infographic
-------------------------

On the left is a visual summary of the AI system's performance on different
subgroups. These plots track the ground truth and predicted positive rate.
Note that the "positive rate" is a function of the dataset's labeling and doesn't
necessarily mean a positive outcome! For instance, "denying bail" could be the positive
label.

Ideally the AI system performs equally well for all subgroups and positively 
classifies each group at similar rates

** Note ** If the ground truth positive rate differs between groups, the AI system cannot
be fair by all definition of fairness.
For instance, the AI system can either accurately reflect the outcome differences
in the data (violating demographic parity if the dataset show disparities) or 
violating performance parity.

Confusion Matrixes
------------------

On the right are confusion matrixes for each group. The confusion matrix plots true positive, 
false positive, true negative and false negative rates. It is rich description of the performance 
of binary classification systems. A well performing system should have
most outcomes along the diagonal (true positives and true negatives)

** Note ** A perfect looking confusion matrix for every group does not guarantee
fairness by all definitions! It just means that the model is accurately reflecting
the dataset. If the dataset has different positive outcome rates
for different groups, the model may be considered unfair.
"""

REGRESSION_DESCRIPTION = """Regression Model Predictions vs Ground Truth

Scatter plot of true vs predicted values is a rich form of data visualization
for regression models.
                
Disaggregated across the demographic groups, this plot also provides visual insights into
how the model may be performing differently across groups.
                
Ideally, all the points should be close to the 45-degree dotted line.
"""
