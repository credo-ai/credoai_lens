import traceback
from itertools import combinations, product

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.stats import chi2_contingency, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from credoai.modules.stats_utils import columns_from_formula
from credoai.utils import global_logger


class CoxPH:
    def __init__(self, **kwargs):
        self.name = "Cox Proportional Hazard"
        self.cph = CoxPHFitter(**kwargs)
        self.fit_kwargs = {}
        self.data = None

    def fit(self, data, **fit_kwargs):
        self.cph.fit(data, **fit_kwargs)
        self.fit_kwargs = fit_kwargs
        self.data = data
        if "formula" in fit_kwargs:
            self.name += f" (formula: {fit_kwargs['formula']})"
        return self

    def summary(self):
        s = self.cph.summary
        s.name = f"{self.name} Stat Summary"
        return s

    def expected_survival(self):
        prediction_data = self._get_prediction_data()
        expected_predictions = self.cph.predict_expectation(prediction_data)
        expected_predictions.name = "E(time survive)"
        final = pd.concat([prediction_data, expected_predictions], axis=1)
        final.name = f"{self.name} Expected Survival"
        return final

    def survival_curves(self):
        prediction_data = self._get_prediction_data()
        survival_curves = self.cph.predict_survival_function(prediction_data)
        survival_curves = (
            # fmt: off
            survival_curves.loc[0:,]
            # fmt: on
            .rename_axis("time_step")
            .reset_index()
            .melt(id_vars=["time_step"])
            .merge(right=prediction_data, left_on="variable", right_index=True)
            .drop(columns=["variable"])
        )
        survival_curves = survival_curves[survival_curves["time_step"] % 5 == 0]
        survival_curves.name = f"{self.name} Survival Curves"
        return survival_curves

    def _get_prediction_data(self):
        columns = columns_from_formula(self.fit_kwargs.get("formula"))
        df = pd.DataFrame(
            list(product(*[i.unique() for _, i in self.data[columns].items()])),
            columns=columns,
        )
        return df


class ChiSquare:
    def __init__(self, pvalue=0.05):
        """
        Statistical Test: Performs chisquared contingency test

        If chi-squared test is significant, follow up with
        posthoc tests for all pairwise comparisons.
        Multiple comparisons are bonferronni corrected.
        """
        self.name = "chisquared_contingency"
        self.pvalue = pvalue
        self.contingency_df = None

    def run(self, data, group1_column, group2_column, run_posthoc=True):
        """Run chisquare test and optional posthoc tests

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with two columns to create a contingency table. Each column must have
            categorical features
        group1_column : str
            The column name for the first grouping column, must be categorical
        group2_column : str
            The column name for the second grouping column, must be categorical
        run_posthoc : bool
            Whether to run posthoc tests if the main chisquared test is significant,
            default True.
        """
        self.contingency_df = self._create_contingency_data(
            data, group1_column, group2_column
        )
        chi2, p, dof, ex = chi2_contingency(self.contingency_df)
        results = {
            "test_type": self.name,
            "statistic": chi2,
            "pvalue": p,
        }
        if run_posthoc and results["pvalue"] < self.pvalue:
            results["significant_posthoc_tests"] = self._posthoc_tests()
        return results

    def _create_contingency_data(self, df, group1_column, group2_column):
        """Create contingency table from a dataframe with two grouping columns

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with two columns to create a contingency table. Each column must have
            categorical features
        group1_column : str
            The column name for the first grouping column, must be categorical
        group2_column : str
            The column name for the second grouping column, must be categorical
        """
        contingency_df = (
            df.groupby([group1_column, group2_column])
            .size()
            .reset_index(name="counts")
            .pivot(group1_column, group2_column)
        )
        return contingency_df

    def _posthoc_tests(self):
        """Run bonferronni corrected posthoc tests on contingency table"""
        posthoc_tests = []
        all_combinations = list(combinations(self.contingency_df.index, 2))
        bonferronni_p = self.pvalue / len(all_combinations)
        for comb in all_combinations:
            # subset df into a dataframe containing only the pair "comb"
            new_df = self.contingency_df[
                (self.contingency_df.index == comb[0])
                | (self.contingency_df.index == comb[1])
            ]
            # running chi2 test
            try:
                chi2, p, dof, ex = chi2_contingency(new_df, correction=False)
            except ValueError as e:
                global_logger.error(
                    "Posthoc Chi2 test could not be run, likely due to insufficient"
                    f" outcome frequencies. Error produced below:\n {traceback.print_exc()}"
                )
            if p < bonferronni_p:
                posthoc_tests.append(
                    {
                        "test_type": self.name,
                        "comparison": comb,
                        "chi2": chi2,
                        "pvalue": p,
                        "significance_threshold": bonferronni_p,
                    }
                )
            return sorted(posthoc_tests, key=lambda x: x["pvalue"])


class OneWayAnova:
    def __init__(self, pvalue=0.05):
        self.name = "oneway_anova"
        self.pvalue = pvalue
        self.data = None

    def run(self, df, grouping_col, outcome_col, run_posthoc=True):
        """Run one-way ANOVA and optional posthoc tests

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with two columns - a grouping column and a continuous outcome column
        grouping_col : str
            The column name for the first grouping column, must be categorical
        outcome_col : str
            The column name for the outcome, must be continuous
        run_posthoc : bool
            Whether to run posthoc tests if the main chisquared test is significant,
            default True.
        """
        self._setup(df, grouping_col, outcome_col)
        overall_test = f_oneway(*self.data["groups"])
        results = {
            "test_type": "oneway_anova",
            "statistic": overall_test.statistic,
            "pvalue": overall_test.pvalue,
        }
        if run_posthoc and results["pvalue"] < self.pvalue:
            results["significant_posthoc_tests"] = self._posthoc_tests(
                df[outcome_col], df[outcome_col]
            )
        return results

    def _setup(self, df, grouping_col, outcome_col):
        groups = df.groupby(grouping_col)[outcome_col]
        group_lists = groups.apply(list)
        labels = np.array(group_lists.index)
        self.data = {"groups": group_lists, "labels": labels}

    def _posthoc_tests(self, outcome_col, grouping_col):
        """Run Tukey HSD posthoc tests on each label"""
        posthoc_tests = []

        results = pairwise_tukeyhsd(outcome_col, grouping_col)
        results_df = pd.DataFrame(
            {
                "test_type": "tukey_hsd",
                "pvalue": results.pvalues,
                "statistic": results.meandiffs,
                "reject": results.reject,
                "comparison": list(combinations(results.groupsunique, 2)),
                "significance_threshold": self.pvalue,
            }
        )
        results_df = results_df[results_df["reject"] == True]
        posthoc_tests = results_df.to_dict(orient="records")

        return sorted(posthoc_tests, key=lambda x: x["pvalue"])
