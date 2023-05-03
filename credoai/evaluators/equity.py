import numpy as np
import pandas as pd
from connect.evidence import MetricContainer, StatisticTestContainer, TableContainer

from credoai.artifacts import TabularData
from credoai.evaluators.evaluator import Evaluator
from credoai.evaluators.utils.validation import (
    check_data_for_nulls,
    check_data_instance,
    check_existence,
)
from credoai.modules.stats import ChiSquare, OneWayAnova
from credoai.utils.model_utils import type_of_target


class DataEquity(Evaluator):
    """
    Data Equity evaluator for Credo AI (Experimental)

    This evaluator assesses whether outcomes are distributed equally across a sensitive
    feature. Depending on the kind of outcome, different tests will be performed.

    - Discrete: chi-squared contingency tests,
      followed by bonferronni corrected posthoc chi-sq tests
    - Continuous: One-way ANOVA, followed by Tukey HSD posthoc tests
    - Proportion (Bounded [0-1] continuous outcome): outcome is transformed to logits, then
      proceed as normal for continuous

    Required Artifacts
    ------------------
    **Required Artifacts**

    Generally artifacts are passed directly to :class:`credoai.lens.Lens`, which
    handles evaluator setup. However, if you are using the evaluator directly, you
    will need to pass the following artifacts when instantiating the evaluator:

    - data: :class:`credoai.artifacts.TabularData`
        The data to evaluate for equity (based on the outcome variable). Must
        have sensitive feature defined.


    Parameters
    ----------
    p_value : float
        The significance value to evaluate statistical tests
    """

    required_artifacts = {"data", "sensitive_feature"}

    def __init__(self, p_value=0.01):
        self.pvalue = p_value
        super().__init__()

    def _validate_arguments(self):
        check_data_instance(self.data, TabularData)
        check_existence(self.data.sensitive_features, "sensitive_features")
        check_data_for_nulls(self.data, "Data")

    def _setup(self):
        self.sensitive_features = self.data.sensitive_feature
        self.y = self.data.y
        self.type_of_target = self.data.y_type

        self.df = pd.concat([self.sensitive_features, self.y], axis=1)
        self.labels = {
            "sensitive_feature": self.sensitive_features.name,
            "outcome": self.y.name,
        }
        return self

    def evaluate(self):
        summary, parity_results = self._describe()
        outcome_distribution = self._outcome_distributions()
        overall_equity, posthoc_tests = self._get_formatted_stats()

        # Combine
        equity_containers = [
            summary,
            outcome_distribution,
            parity_results,
            overall_equity,
        ]

        # Add posthoc if available
        if posthoc_tests is not None:
            equity_containers.append(posthoc_tests)

        self.results = equity_containers
        return self

    def _describe(self):
        """Create descriptive output"""
        means = self.df.groupby(self.sensitive_features.name).mean()
        results = {"summary": means}

        summary = results["summary"]
        results["sensitive_feature"] = self.sensitive_features.name
        results["highest_group"] = summary[self.y.name].idxmax()
        results["lowest_group"] = summary[self.y.name].idxmin()
        results["demographic_parity_difference"] = (
            summary[self.y.name].max() - summary[self.y.name].min()
        )
        results["demographic_parity_ratio"] = (
            summary[self.y.name].min() / summary[self.y.name].max()
        )

        summary.name = f"Average Outcome Per Group"

        # Format summary results
        summary = TableContainer(
            results["summary"],
            **self.get_info(labels=self.labels),
        )

        # Format parity results
        parity_results = pd.DataFrame(
            [
                {"type": k, "value": v}
                for k, v in results.items()
                if "demographic_parity" in k
            ]
        )
        parity_results = MetricContainer(
            parity_results,
            **self.get_info(labels=self.labels),
        )

        return summary, parity_results

    def _outcome_distributions(self):
        out = TableContainer(
            outcome_distribution(
                self.df, self.sensitive_features.name, self.y.name, self.type_of_target
            ),
            **self.get_info(labels=self.labels),
        )
        return out

    def _get_formatted_stats(self) -> tuple:
        """
        Select statistics based on classification type, add formatting.

        Returns
        -------
        tuple
            Overall equity, posthoc tests
        """
        if self.type_of_target in ("binary", "multiclass"):
            statistics = self.discrete_stats()
        else:
            statistics = self.continuous_stats()

        overall_equity = {
            "statistic_type": statistics["test_type"],
            "test_statistic": statistics["statistic"],
            "p_value": statistics["pvalue"],
            "significance_threshold": self.pvalue,
            "significant": statistics["pvalue"] <= self.pvalue,
        }

        overall_equity = StatisticTestContainer(
            pd.DataFrame(overall_equity, index=[0]), **self.get_info()
        )

        posthoc_tests = None
        if "significant_posthoc_tests" in statistics:
            posthoc_tests = pd.DataFrame(statistics["significant_posthoc_tests"])
            posthoc_tests.name = f"{statistics['test_type']}_posthoc"
            posthoc_tests = TableContainer(posthoc_tests, **self.get_info())

        return overall_equity, posthoc_tests

    def discrete_stats(self):
        """Run statistics on discrete outcomes"""
        test = ChiSquare(self.pvalue)
        return test.run(self.df, self.sensitive_features.name, self.y.name)

    def continuous_stats(self):
        """Run statistics on continuous outcomes"""
        # check for proportional bounding and transform
        if self._check_range(self.y, 0, 1):
            self._proportion_transformation()
        return OneWayAnova(self.pvalue).run(
            self.df, self.sensitive_features.name, self.y.name
        )

    # helper functions
    def _check_range(self, lst, lower_bound, upper_bound):
        return min(lst) >= lower_bound and max(lst) <= upper_bound

    def _proportion_transformation(self):
        """Transforms bounded values between 0-1 into a continuous space"""

        def logit(x):
            eps = 1e-6
            return np.log(x / (1 - x + eps) + eps)

        self.df[self.y.name] = self.df[self.y.name].apply(logit)


class ModelEquity(DataEquity):
    """
    Evaluates the equity of a model's predictions.

    This evaluator assesses whether model predictions are distributed equally across a sensitive
    feature. Depending on the kind of outcome, different tests will be performed:

    - Discrete: chi-squared contingency tests,
      followed by bonferronni corrected posthoc chi-sq tests
    - Continuous: One-way ANOVA, followed by Tukey HSD posthoc tests
    - Proportion (Bounded [0-1] continuous outcome): outcome is transformed to logits, then
      proceed as normal for continuous

    Required Artifacts
    ------------------
        **Required Artifacts**

        Generally artifacts are passed directly to :class:`credoai.lens.Lens`, which
        handles evaluator setup. However, if you are using the evaluator directly, you
        will need to pass the following artifacts when instantiating the evaluator:

        - model: :class:`credoai.artifacts.Model`
        - assessment_data: :class:`credoai.artifacts.TabularData`
            The assessment data to use to create model predictions and evaluate
            the equity of the model. Must have sensitive features.


    Parameters
    ----------
    use_predict_proba : bool, optional
        Defines which predict method will be used, if True predict_proba will be used.
        This methods outputs probabilities rather then class predictions. The availability
        of predict_proba is dependent on the model under assessment. By default False
    p_value : float, optional
        The significance value to evaluate statistical tests, by default 0.01
    """

    required_artifacts = {"model", "assessment_data", "sensitive_feature"}

    def __init__(self, use_predict_proba=False, p_value=0.01):
        self.use_predict_proba = use_predict_proba
        super().__init__(p_value)

    def _setup(self):
        self.sensitive_features = self.assessment_data.sensitive_feature
        fun = self.model.predict_proba if self.use_predict_proba else self.model.predict
        self.y = pd.Series(
            fun(self.assessment_data.X),
            index=self.sensitive_features.index,
        )
        prefix = "predicted probability" if self.use_predict_proba else "predicted"
        try:
            self.y.name = f"{prefix} {self.assessment_data.y.name}"
        except:
            self.y.name = f"{prefix} outcome"

        self.type_of_target = type_of_target(self.y)

        self.df = pd.concat([self.sensitive_features, self.y], axis=1)
        self.labels = {
            "sensitive_feature": self.sensitive_features.name,
            "outcome": self.y.name,
        }
        return self

    def _validate_arguments(self):
        check_data_instance(self.assessment_data, TabularData)
        check_existence(self.assessment_data.sensitive_features, "sensitive_features")
        check_data_for_nulls(
            self.assessment_data, "Data", check_X=True, check_y=True, check_sens=True
        )


############################################
## Evaluation helper functions

## Helper functions create evidences
## to be passed to .evaluate to be wrapped
## by evidence containers
############################################


def outcome_distribution(df, grouping_col, outcome_col, type_of_target, bins=10):
    """Returns outcome distribution over a grouping factor

    For binary/multiclass outcomes, returns the counts for each set of outcomes/grouping.
    For a continuous outcome, bins the outcome and reports the number of records in each bin
    for each group.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with at least two columns for grouping and outcome
    grouping_col : str
        Name of the grouping column, must refer to a categorical column
    outcome_col : str
        Name of the outcome column
    type_of_target : str
        The type of outcome column. Anything besides "binary" and "multiclass" will be treated
        as continuous.
    bins : int
        Number of bins to use in the case of a continuous outcome

    Returns
    -------
    pd.DataFrame
        _description_
    """

    df = df.loc[:, [grouping_col, outcome_col]]
    if type_of_target in ("binary", "multiclass"):
        distribution = df.value_counts().sort_index().reset_index(name="count")
        distribution["proportion"] = distribution["count"] / distribution["count"].sum()
    # histogram binning for continuous
    else:
        distribution = []
        for i, group in df.groupby(grouping_col):
            counts, edges = np.histogram(group[outcome_col], bins=bins)
            bins = edges  # ensure all groups have same bins
            bin_centers = 0.5 * (edges[:-1] + edges[1:])
            tmp = pd.DataFrame(
                {
                    grouping_col: i,
                    outcome_col: bin_centers,
                    "count": counts,
                }
            )
            distribution.append(tmp)
        distribution = pd.concat(distribution, axis=0)
    distribution.name = "Outcome Distributions"

    return distribution
