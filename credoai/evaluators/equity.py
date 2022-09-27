import traceback
from itertools import combinations

import numpy as np
import pandas as pd
from credoai.artifacts import TabularData
from credoai.evaluators import Evaluator
from credoai.evaluators.utils.validation import (
    check_artifact_for_nulls,
    check_data_instance,
    check_existence,
)
from credoai.evidence import MetricContainer, TableContainer
from credoai.utils import NotRunError, global_logger
from scipy.stats import chi2_contingency, f_oneway, tukey_hsd


class DataEquity(Evaluator):
    """
    Dataset Equity module for Credo AI.

    This module assesses whether outcomes are distributed equally across a sensitive
    feature. Depending on the kind of outcome, different tests will be performed.

    * Discrete: chi-squared contingency tests,
      followed by bonferronni corrected posthoc chi-sq tests
    * Continuous: One-way ANOVA, followed by Tukey HSD posthoc tests
    * Proportion (Bounded [0-1] continuous outcome): outcome is transformed to logits, then
        proceed as normal for continuous

    Parameters
    ----------
    sensitive_features :  pandas.Series
        The segmentation feature which should be used to create subgroups to analyze.
    y : (List, pandas.Series, numpy.ndarray)
        Outcomes (e.g., labels for classification or target values for regression),
        either ground-truth or model generated.
    p_value : float
        The significance value to evaluate statistical tests
    """

    name = "DataEquity"
    required_artifacts = {"data", "sensitive_feature"}

    def __init__(self, p_value=0.01):
        self.pvalue = p_value

    def _setup(self):
        self.sensitive_features = self.data.sensitive_feature
        self.y = self.data.y
        self.type_of_target = self.data.y_type

        self.df = pd.concat([self.sensitive_features, self.y], axis=1)
        return self

    def _validate_arguments(self):
        check_data_instance(self.data, TabularData)
        check_existence(self.data.sensitive_features, "sensitive_features")
        check_artifact_for_nulls(self.data, "Data")

    def evaluate(self):
        self._results = {"descriptive": self.describe()}
        if self.type_of_target in ("binary", "multiclass"):
            self.results["statistics"] = self.discrete_stats()
        else:
            self.results["statistics"] = self.continuous_stats()
        self.results = self._prepare_results()
        return self

    def _prepare_results(self):
        if self._results:
            desc = self._results["descriptive"]
            summary = desc["summary"]
            summary["subtype"] = "summary"
            summary.name = "summary"
            summary = TableContainer(
                summary,
                **self.get_container_info(
                    labels={"sensitive_feature": self.sensitive_features.name}
                ),
            )
            desc_metadata = {
                "highest_group": desc["highest_group"],
                "lowest_group": desc["lowest_group"],
            }

            results = pd.DataFrame(
                [
                    {"type": k, "value": v}
                    for k, v in desc.items()
                    if "demographic_parity" in k
                ]
            )
            results = MetricContainer(
                results,
                **self.get_container_info(
                    labels={"sensitive_feature": self.sensitive_features.name}
                ),
            )

            # add statistics
            stats = self.results["statistics"]
            overall_equity = {
                "type": "overall",
                "value": stats["equity_test"]["statistic"],
                "subtype": stats["equity_test"]["test_type"],
                "p_value": stats["equity_test"]["pvalue"],
            }
            overall_equity = MetricContainer(
                pd.DataFrame(overall_equity, index=[0]),
                **self.get_container_info(
                    labels={"sensitive_feature": self.sensitive_features.name}
                ),
            )
            # add posthoc tests if needed
            equity_containers = [summary, results, overall_equity]
            if "significant_posthoc_tests" in stats:
                posthoc_tests = pd.DataFrame(stats["significant_posthoc_tests"])
                posthoc_tests.rename({"test_type": "subtype"}, axis=1, inplace=True)
                posthoc_tests.name = "posthoc"
                equity_containers.append(
                    TableContainer(
                        posthoc_tests,
                        **self.get_container_info(
                            labels={"sensitive_feature": self.sensitive_features.name}
                        ),
                    )
                )

            return equity_containers
        else:
            raise NotRunError(
                "Results not created yet. Call 'run' with appropriate arguments before preparing results"
            )

    def describe(self):
        """Create descriptive output"""
        results = {
            "summary": self.df.groupby(self.sensitive_features.name)[
                self.y.name
            ].describe()
        }
        r = results["summary"]
        results["sensitive_feature"] = self.sensitive_features.name
        results["highest_group"] = r["mean"].idxmax()
        results["lowest_group"] = r["mean"].idxmin()
        results["demographic_parity_difference"] = r["mean"].max() - r["mean"].min()
        results["demographic_parity_ratio"] = r["mean"].min() / r["mean"].max()
        return results

    def discrete_stats(self):
        """Run statistics on discrete outcomes"""
        return self._chisquare_contingency()

    def continuous_stats(self):
        """Run statistics on continuous outcomes"""
        # check for proportion bounding
        if self._check_range(self.y, 0, 1):
            self._proportion_transformation()
            return self._anova_tukey_hsd(f"transformed_{self.y.name}")
        else:
            return self._anova_tukey_hsd(self.y.name)

    def _chisquare_contingency(self):
        """
        Statistical Test: Performs chisquared contingency test

        If chi-squared test is significant, follow up with
        posthoc tests for all pairwise comparisons.
        Multiple comparisons are bonferronni corrected.
        """
        contingency_df = (
            self.df.groupby([self.sensitive_features.name, self.y.name])
            .size()
            .reset_index(name="counts")
            .pivot(self.sensitive_features.name, self.y.name)
        )
        chi2, p, dof, ex = chi2_contingency(contingency_df)
        results = {
            "equity_test": {
                "test_type": "chisquared_contingency",
                "statistic": chi2,
                "pvalue": p,
            }
        }
        # run bonferronni corrected posthoc tests if significant
        if results["equity_test"]["pvalue"] < self.pvalue:
            posthoc_tests = []
            all_combinations = list(combinations(contingency_df.index, 2))
            bonferronni_p = self.pvalue / len(all_combinations)
            for comb in all_combinations:
                # subset df into a dataframe containing only the pair "comb"
                new_df = contingency_df[
                    (contingency_df.index == comb[0])
                    | (contingency_df.index == comb[1])
                ]
                # running chi2 test
                try:
                    chi2, p, dof, ex = chi2_contingency(new_df, correction=False)
                except ValueError as e:
                    global_logger.error(
                        "Chi2 test could not be run, likely due to insufficient"
                        f" outcome frequencies. Error produced below:\n {traceback.print_exc()}"
                    )
                if p < bonferronni_p:
                    posthoc_tests.append(
                        {
                            "test_type": "chisquared_contingency",
                            "comparison": comb,
                            "chi2": chi2,
                            "pvalue": p,
                            "significance_threshold": bonferronni_p,
                        }
                    )
            results["significant_posthoc_tests"] = sorted(
                posthoc_tests, key=lambda x: x["pvalue"]
            )
        return results

    def _anova_tukey_hsd(self, outcome_col):
        """Statistical Test: Performs One way Anova and Tukey HSD Test

        The Tukey HSD test is a posthoc test that is only performed if the
        anova is significant.
        """
        groups = self.df.groupby(self.sensitive_features.name)[outcome_col]
        group_lists = groups.apply(list)
        labels = np.array(group_lists.index)
        overall_test = f_oneway(*group_lists)
        results = {
            "equity_test": {
                "test_type": "oneway_anova",
                "statistic": overall_test.statistic,
                "pvalue": overall_test.pvalue,
            }
        }
        # run posthoc test if significant
        if results["equity_test"]["pvalue"] < self.pvalue:
            posthoc_tests = []
            r = tukey_hsd(*group_lists.values)
            sig_compares = r.pvalue < self.pvalue
            for indices in zip(*np.where(sig_compares)):
                specific_labels = np.take(labels, indices)
                statistic = r.statistic[indices]
                posthoc_tests.append(
                    {
                        "test_type": "tukey_hsd",
                        "comparison": specific_labels,
                        "statistic": statistic,
                        "pvalue": r.pvalue[indices],
                        "significance_threshold": self.pvalue,
                    }
                )
            results["significant_posthoc_tests"] = sorted(
                posthoc_tests, key=lambda x: x["pvalue"]
            )
        return results

    # helper functions
    def _check_range(self, lst, lower_bound, upper_bound):
        return min(lst) >= lower_bound and max(lst) <= upper_bound

    def _normalize_counts(self, f_1, f_2):
        """Normalizes frequencies in f_1 to f_2"""
        f_1 = np.array(f_1)
        f_2 = np.array(f_2)
        return f_1 / f_1.sum() * sum(f_2)

    def _proportion_transformation(self):
        def logit(x):
            eps = 1e-6
            return np.log(x / (1 - x + eps) + eps)

        self.df[f"transformed_{self.y.name}"] = self.df[self.y.name].apply(logit)


class ModelEquity(DataEquity):
    name = "ModelEquity"
    required_artifacts = {"model", "assessment_data", "sensitive_feature"}

    def _setup(self):
        self.sensitive_features = self.assessment_data.sensitive_feature
        self.y = pd.Series(
            self.model.predict(self.assessment_data.X),
            index=self.sensitive_features.index,
        )
        try:
            self.y.name = f"predicted {self.assessment_data.y.name}"
        except:
            self.y.name = "predicted outcome"

        self.type_of_target = self.assessment_data.y_type

        self.df = pd.concat([self.sensitive_features, self.y], axis=1)
        return self

    def _validate_arguments(self):
        check_data_instance(self.assessment_data, TabularData)
        check_existence(self.assessment_data.sensitive_features, "sensitive_features")
        check_artifact_for_nulls(self.assessment_data, "Data")
