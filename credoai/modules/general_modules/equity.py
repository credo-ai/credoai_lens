from itertools import combinations

import numpy as np
import pandas as pd
from credoai.modules.credo_module import CredoModule
from credoai.utils import NotRunError, ValidationError
from scipy.stats import chi2_contingency, chisquare, f_oneway, tukey_hsd
from sklearn.utils.multiclass import type_of_target


class EquityModule(CredoModule):
    """
    Equity module for Credo AI.

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

    def __init__(self, sensitive_features, y, p_value=0.01):
        super().__init__()
        self.sensitive_features = sensitive_features
        self.y = y
        if type(self.y) is not pd.Series:
            self.y = pd.Series(y, name="outcome")
        self.type_of_target = type_of_target(self.y)
        # create df
        self.df = pd.concat([self.sensitive_features, self.y], axis=1)
        # other parameters
        self.pvalue = p_value

    def run(self):
        self.results = {"descriptive": self.describe()}
        if self.type_of_target in ("binary", "multiclass"):
            self.results["statistics"] = self.discrete_stats()
        else:
            self.results["statistics"] = self.continuous_stats()
        return self

    def prepare_results(self):
        if self.results:
            desc = self.results["descriptive"]
            desc_metadata = {
                "group_means": desc["summary"]["mean"].to_dict(),
                "highest_group": desc["highest_group"],
                "lowest_group": desc["lowest_group"],
            }
            results = [
                {"metric_type": k, "value": v, "metadata": desc_metadata}
                for k, v in desc.items()
                if "demographic_parity" in k
            ]
            # add statistics
            stats = self.results["statistics"]
            overall_equity = {
                "metric_type": "equity_test",
                "subtype": "overall_test",
                "value": stats["equity_test"]["pvalue"],
                "test_type": stats["equity_test"]["test_type"],
                "metadata": stats["equity_test"],
            }
            results.append(overall_equity)
            # add posthoc tests if needed
            if "significant_posthoc_tests" in stats:
                for test in stats["significant_posthoc_tests"]:
                    results.append(
                        {
                            "metric_type": "equity_test",
                            "subtype": "posthoc_test",
                            "value": test["pvalue"],
                            "test_type": test["test_type"],
                            "comparison_groups": list(test["comparison"]),
                            "metadata": test,
                        }
                    )
            results = pd.DataFrame(results).set_index("metric_type")
            results["sensitive_feature"] = self.sensitive_features.name
            return results
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
                chi2, p, dof, ex = chi2_contingency(new_df, correction=False)
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
