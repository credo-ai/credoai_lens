import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from credoai.reporting.credo_reporter import CredoReporter
from credoai.reporting.plot_utils import credo_diverging_palette, get_style
from seaborn.utils import relative_luminance


class EquityReporter(CredoReporter):
    def __init__(self, module, size=3):
        super().__init__(module)
        self.size = size

    def _create_assets(self):
        """Creates equity reporting assets"""
        # plot
        self.plot_outcomes()

    def plot_outcomes(self):
        mod = self.module
        sf_name = mod.sensitive_features.name
        outcome = mod.y.name
        metric_keys = []
        if self.key_lookup is not None:
            metric_keys = self.key_lookup.query(
                f'sensitive_feature=="{self.module.sensitive_features.name}"'
            )["metric_key"].tolist()
        with get_style(figsize=self.size):
            f = plt.figure()
            if mod.type_of_target in ("binary", "multiclass"):
                self._classification_plot()
            else:
                palette = credo_diverging_palette(len(mod.sensitive_features.unique()))
                sns.boxplot(
                    data=mod.df,
                    x=mod.sensitive_features.name,
                    y=mod.y.name,
                    palette=palette,
                )
        self._create_chart(
            f,
            "",
            f"{outcome.title()} Equity for Sensitive Feature: {sf_name.title()}",
            metric_keys,
        )

    def _classification_plot(self):
        sf_name = self.module.sensitive_features.name
        outcome = self.module.y.name
        pivoted_df = (
            self.module.df.groupby([sf_name, outcome])
            .size()
            .reset_index()
            .pivot(sf_name, outcome, 0)
        )
        colors = credo_diverging_palette(pivoted_df.shape[1])
        normalized_df = pivoted_df.divide(pivoted_df.sum(1), 0) * 100

        ax = normalized_df.plot(kind="bar", stacked=True, color=colors)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_ylabel(f"{outcome.title()} Breakdown")
        sns.despine()
        for p in ax.patches:
            lum = relative_luminance(p.get_facecolor())
            text_color = ".15" if lum > 0.208 else "w"

            width, height = p.get_width(), p.get_height()
            x, y = p.get_xy()
            ax.text(
                x + width / 2,
                y + height / 2,
                "{:.0f} %".format(height),
                color=text_color,
                horizontalalignment="center",
                verticalalignment="center",
            )
