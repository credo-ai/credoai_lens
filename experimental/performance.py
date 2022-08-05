import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from credoai.metrics.credoai_metrics import wilson_ci
from credoai.metrics.metric_constants import METRICS, PROBABILITY_METRICS
from credoai.reporting.plot_utils import credo_converging_palette, plot_curve
from credoai.utils.common import wrap_list
from IPython.display import display
from sklearn import metrics as sk_metrics


class BinaryClassificationPerformanceToolkit:
    def __init__(self, df, y_true_col, y_pred_col, y_prob_col):
        self.df = df
        self.y_true = self.df[y_true_col]
        self.y_pred = self.df[y_pred_col]
        self.y_prob = self.df[y_prob_col]

    def calc_metrics(self, metrics=None, conf_threshold=0.5):
        """
        Calculates metrics for given dataframe, returning a dict of the results

        Parameters
        ----------
        metrics : list-like, optional
            list of metrics to calculate. Each list element must either be a
            string referencing a sklearn metric (e.g., ['balanced_accuracy', 'f1'],
            or a tuple (str, func) representing
            (metric name, metric function). The function must take two arguments,
            y_true, y_pred.
        conf_threshold : float
            labels below this threshold are set to 0, affects y_pred dependent metrics
        """
        if metrics is None:
            metrics = ["f1", "matthews", "precision", "fpr", "tnr", "fnr"]
        metric_names = [m if type(m) == str else m[0] for m in metrics]
        metric_funs = [METRICS[m] if type(m) == str else m[1] for m in metrics]

        results = {}

        bool_conf = self.y_prob >= conf_threshold
        # calculate performance setting both labels as "positive"
        label_set = sorted(set(self.y_true))
        for pos_class in label_set:
            y_pred = self.y_pred.copy()
            neg_class = label_set[1 - label_set.index(pos_class)]
            y_pred[
                ~bool_conf
            ] = neg_class  # anything below confidence is not labeled successfully
            y_prob = self._rereference_confidence(pos_class)

            for metric_name, metric_fun in zip(metric_names, metric_funs):
                if metric_name in PROBABILITY_METRICS:
                    out = metric_fun(self.y_true == pos_class, y_prob)
                else:
                    out = metric_fun(self.y_true == pos_class, y_pred == pos_class)
                results[(pos_class, metric_name)] = out
        return results

    def metrics_series(self, metrics=None, conf_threshold=0.5):
        series = pd.Series(
            self.calc_metrics(metrics, conf_threshold), name="metric_value"
        )
        series.index.set_names(["class", "metric_name"], inplace=True)
        return series

    def _rereference_confidence(self, pos_label):
        """References all confidence values to positive class"""
        return [
            conf if pred == pos_label else 1.0 - conf
            for pred, conf in zip(self.y_pred, self.y_prob)
        ]

    def FMR(self, n=10):
        FM = [0] * 100 * n
        lower_arr = [0] * 100 * n
        upper_arr = [0] * 100 * n
        total = len(self.y_prob)

        for conf in list(self.y_prob):
            iterr = math.ceil(conf * n)
            for i in range(iterr):
                FM[i] += 1

        FMR = [f / total for f in FM]
        for i, f in enumerate(FM):
            lower_arr[i], upper_arr[i] = wilson_ci(f, total)

        return FMR, lower_arr, upper_arr

    def FMR_threshold(self, thresholds, subgroup_name):
        result = {}
        fmr_arr, low_arr, upp_arr = self.FMR()
        result.update(
            {"Category": subgroup_name, "No. of Imposter Pairs": len(self.y_prob)}
        )
        for thresh in thresholds:
            result.update(
                {f"FMR at {thresh}% Threshold": round(fmr_arr[thresh * 10] * 100, 2)}
            )
        return result

    def FNMR(self, n=10):
        FNM = [0] * 100 * n
        lower_arr = [0] * 100 * n
        upper_arr = [0] * 100 * n
        total = len(self.y_prob)

        for conf in list(self.y_prob):
            iterr = math.ceil(conf * n)
            for i in range(iterr, 100 * n):
                FNM[i] += 1

        FNMR = [f / total for f in FNM]
        for i, f in enumerate(FNM):
            lower_arr[i], upper_arr[i] = wilson_ci(f, total)

        return FNMR, lower_arr, upper_arr

    def FNMR_threshold(self, thresholds, subgroup_name):
        result = {}
        fnmr_arr, low_arr, upp_arr = self.FNMR()
        result.update(
            {"Category": subgroup_name, "No. of Genuine Pairs": len(self.y_prob)}
        )
        for thresh in thresholds:
            result.update(
                {f"FNMR at {thresh}% Threshold": round(fnmr_arr[thresh * 10] * 100, 2)}
            )
        return result

    def _get_performance_curve(self, perf_func, auc_fun, pos_label, **kwargs):
        y_prob = self._rereference_confidence(pos_label)
        perf_metric1, perf_metric2, thresholds = perf_func(
            self.y_true == pos_label, y_prob
        )
        auc = auc_fun(self.y_true == pos_label, y_prob)
        return perf_metric1, perf_metric2, thresholds, auc

    def plot_pr_curve(self, pos_label, ax=None, context="talk", **kwargs):
        prec, reca, thresholds, auc = self._get_performance_curve(
            sk_metrics.precision_recall_curve,
            sk_metrics.average_precision_score,
            pos_label,
        )
        label = f"{pos_label} (AP = {auc:.2f})"
        with sns.plotting_context(context):
            ax = plot_curve(
                reca, prec, label, ax, legend_loc="lower left", context="talk", **kwargs
            )

            ax.set(xlabel="Recall", ylabel="Precision")
            ax.set_title("Precision Recall Curve")

    def plot_roc_curve(self, pos_label, ax=None, context="talk", **kwargs):
        fpr, tpr, thresholds, auc = self._get_performance_curve(
            sk_metrics.roc_curve, sk_metrics.roc_auc_score, pos_label
        )
        label = f"{pos_label} (ROC AUC = {auc:.2f})"
        with sns.plotting_context(context):
            ax = plot_curve(fpr, tpr, label, ax, legend_loc="lower right", **kwargs)
            ax.plot([0, 1], [0, 1], lw=2, ls="--", color="k")
            ax.set(
                xlabel="False Positive Rate (FPR)", ylabel="True Positive Rate (TPR)"
            )
            ax.set_title("ROC")

    def plot_fmr_threshold(self, ax=None, label="", n=10, context="talk", **kwargs):
        fmr, low_arr, upp_arr = self.FMR(n=n)
        color = kwargs.pop("color") if "color" in kwargs else "#152672"
        with sns.plotting_context(context):
            ax = plot_curve(
                [i / n for i in range(100 * n)],
                fmr,
                label,
                ax,
                legend_loc="upper right",
                color=color,
                **kwargs,
            )
            ax.fill_between(
                [i / n for i in range(100 * n)],
                low_arr,
                upp_arr,
                color=color,
                alpha=0.3,
            )
            ax.set(xlabel="Confidence Threshold", ylabel="False Match Rate (FMR)")
            ax.set_title("False Match Rate Over Threshold")

    def plot_fnmr_threshold(self, ax=None, label="", n=10, context="talk", **kwargs):
        fnmr, low_arr, upp_arr = self.FNMR(n=n)
        color = kwargs.pop("color") if "color" in kwargs else "#152672"
        with sns.plotting_context(context):
            ax = plot_curve(
                [i / n for i in range(100 * n)],
                fnmr,
                label,
                ax,
                legend_loc="upper left",
                color=color,
                **kwargs,
            )
            ax.fill_between(
                [i / n for i in range(100 * n)],
                low_arr,
                upp_arr,
                color=color,
                alpha=0.3,
            )
            ax.set(xlabel="Confidence Threshold", ylabel="False Non-Match Rate (FNMR)")
            ax.set_title("False Non-Match Rate Over Threshold")

    def assess_performance(
        self,
        metrics=["precision", "recall", "f1"],
        plots=["roc"],
        size=8,
        context="talk",
    ):
        label_set = set(self.y_true)
        # metrics
        metric_df = self.metrics_series(metrics)
        display(metric_df)
        # plots
        # set up plot
        n_plots = len(plots)
        plot_width = size
        plot_height = size * n_plots * 0.8
        colors = credo_converging_palette(len(label_set))
        if plots:
            with sns.plotting_context(context):
                f, axes = plt.subplots(n_plots, 1, figsize=(plot_width, plot_height))
                axes = wrap_list(axes)

            for plot_name, ax in zip(plots, axes):
                for class_label, color in zip(label_set, colors):
                    title = plot_name.upper() + " Curve"
                    kwargs = {
                        "pos_label": class_label,
                        "ax": ax,
                        "context": context,
                        "color": color,
                    }
                    if plot_name == "roc":
                        self.plot_roc_curve(**kwargs)
                    elif plot_name == "pr":
                        self.plot_pr_curve(**kwargs)
            plt.subplots_adjust(hspace=0.5)
        return metric_df

    def help_tmp(self):
        print(
            "Welcome to Credo AI's metrics toolkit! What metric would you like information about? (When finished, please enter 'Done')\n"
        )

        metric_info = {
            "precision": "The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative. The best value is 1 and the worst value is 0.\n",
            "recall": "The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples. The best value is 1 and the worst value is 0.\n",
        }

        while True:
            metric = str(input("Metric:")).lower()

            if metric == "done":
                break
            if metric not in metric_info:
                print("Sorry, that is not a metric supported by Credo AI's toolkit.\n")
            else:
                print(metric_info[metric])
