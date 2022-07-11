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

    Parameters
    ----------
    sensitive_features :  pandas.Series
        The segmentation feature which should be used to create subgroups to analyze.
    y_true : (List, pandas.Series, numpy.ndarray)
        The ground-truth labels (for classification) or target values (for regression).
    y_pred : (List, pandas.Series, numpy.ndarray)
        The predictions of the model
    p_value : int
        The significance value to evaluate statistical tests
    """

    def __init__(self,
                 sensitive_features,
                 y_pred,
                 y_true=None,
                 p_value=.05,
                 simulation_n=1000
                 ):
        self.sensitive_features = sensitive_features
        self.sf_name = self.sensitive_features.name
        self.y_true = y_true
        self.y_pred = y_pred
        self.type_of_target = type_of_target(self.y_pred)
        # create df
        df_input = {'model': self.y_pred}
        if self.y_true is not None:
            df_input['data'] = self.y_true
        self.df = pd.DataFrame(df_input)
        self.df = pd.concat([self.sensitive_features, self.df], axis=1)
        # other parameters
        self.pvalue = p_value
        self.simulation_n = simulation_n

    def run(self):
        self.results = {'descriptive': self.describe()}
        if self.type_of_target == 'binary':
            self.results['statistics'] = self.classification_stats()
        elif self.type_of_target == 'multiclass':
            self.results['statistics'] = self.classification_stats()
        else:
            self.results['statistics'] = self.continuous_stats()
        return self

    def prepare_results(self):
        if self.results:
            all_results = pd.DataFrame()
            artifacts = [
                'model', 'data'] if self.y_true is not None else ['model']
            for artifact in artifacts:
                desc = self.results['descriptive'][artifact]
                desc_metadata = {'group_means': desc['summary']['mean'],
                                 'highest_group': desc['highest_group'],
                                 'lowest_group': desc['lowest_group']}
                results = [{'metric_type': k, 'value': v, 'metadata': desc_metadata}
                           for k, v in desc.items() if 'demographic_parity' in k]
                # add statistics
                stats = self.results['statistics'][artifact]
                overall_equity = {'metric_type': 'equity_test',
                                  'subtype': 'overall_test',
                                  'value': stats['equity_test']['pvalue'],
                                  'metadata': stats['equity_test']}
                results.append(overall_equity)
                # add posthoc tests if needed
                if stats['significant_posthoc_tests']:
                    for test in stats['significant_posthoc_tests']:
                        results.append(
                            {
                                'metric_type': 'equity_test',
                                'subtype': 'posthoc_test',
                                'value': test['pvalue'],
                                'comparison_groups': list(test['comparison']),
                                'metadata': test
                            }
                        )
                results = pd.DataFrame(results)
                results['artifact'] = artifact
                all_results = pd.concat([all_results, results])
            return all_results
        else:
            raise NotRunError(
                "Results not created yet. Call 'run' with appropriate arguments before preparing results"
            )

    def describe(self):
        results = {'model': {'summary': self.df.groupby(
            self.sf_name).model.describe()}}
        if self.y_true is not None:
            results['data'] = {'summary': self.df.groupby(
                self.sf_name).data.describe()}
        for k in results.keys():
            r = results[k]['summary']
            results[k]['highest_group'] = r['mean'].idxmax()
            results[k]['lowest_group'] = r['mean'].idxmin()
            results[k]['demographic_parity_difference'] = r['mean'].max() - \
                r['mean'].min()
            results[k]['demographic_parity_ratio'] = r['mean'].max() / \
                r['mean'].min()
        return results

    def classification_stats(self):
        return self._run_stats(self._chisquare_contingency)

    def continuous_stats(self):
        # check for proportion bounding
        if self._check_range(self.y_pred, 0, 1):
            self._proportion_transformation()
            return self._run_stats(self._anova_tukey_hsd,
                                   'transformed_model',
                                   'transformed_data')
        else:
            return self._run_stats(self._anova_tukey_hsd)

    def _chisquare_contingency(self, output_col):
        """
        Statistical Test: Performs chisquared contingency test

        Parameters
        ----------
        output_col : str
            either "model" or "data"

        """
        contingency_df = self.df.groupby([self.sf_name, output_col])\
            .count().reset_index()\
            .pivot(self.sf_name, output_col)
        chi2, p, dof, ex = chi2_contingency(contingency_df)
        results = {'equity_test': {
            'test_type': 'chisquared_contingency', 'statistic': chi2, 'pvalue': p}}
        # run bonferronni corrected posthoc tests if significant
        if results['equity_test']['pvalue'] < self.pvalue:
            posthoc_tests = []
            all_combinations = list(combinations(contingency_df.index, 2))
            bonferronni_p = self.pvalue/len(all_combinations)
            for comb in all_combinations:
                # subset df into a dataframe containing only the pair "comb"
                new_df = contingency_df[(contingency_df.index == comb[0]) | (
                    contingency_df.index == comb[1])]
                # running chi2 test
                chi2, p, dof, ex = chi2_contingency(new_df, correction=False)
                if p < bonferronni_p:
                    posthoc_tests.append({'comparison': comb, 'chi2': chi2,
                                          'pvalue': p, 'significance_threshold': bonferronni_p})
            results['significant_posthoc_tests'] = sorted(
                posthoc_tests, key=lambda x: x['pvalue'])
        return results

    def _anova_tukey_hsd(self, output_col):
        """Statistical Test: Performs One way Anova and Tukey HSD Test"""
        groups = self.df.groupby(self.sf_name)[output_col]
        group_lists = groups.apply(list)
        labels = np.array(group_lists.index)
        overall_test = f_oneway(*group_lists)
        results = {'equity_test': {'test_type': 'oneway_anova',
                                   'statistic': overall_test.statistic,
                                   'pvalue': overall_test.pvalue}}
        # run posthoc test if significant
        if results['equity_test']['pvalue'] < self.pvalue:
            posthoc_tests = []
            r = tukey_hsd(*group_lists.values)
            sig_compares = r.pvalue < self.pvalue
            for indices in zip(*np.where(sig_compares)):
                specific_labels = np.take(labels, indices)
                statistic = r.statistic[indices]
                posthoc_tests.append({'comparison': specific_labels, 'statistic': statistic,
                                      'pvalue': r.pvalue[indices], 'significance_threshold': self.pvalue})
            results['significant_posthoc_tests'] = sorted(
                posthoc_tests, key=lambda x: x['pvalue'])
        return results

    # helper functions
    def _check_range(self, lst, lower_bound, upper_bound):
        return min(lst) >= lower_bound and max(lst) <= upper_bound

    def _normalize_counts(self, f_1, f_2):
        """Normalizes frequencies in f_1 to f_2"""
        f_1 = np.array(f_1)
        f_2 = np.array(f_2)
        return (f_1/f_1.sum()*sum(f_2))

    def _proportion_transformation(self):
        def logit(x):
            eps = 1E-6
            return np.log(x/(1-x+eps)+eps)
        self.df['transformed_model'] = self.df.model.apply(logit)
        self.df['transformed_data'] = self.df.data.apply(logit)

    def _run_stats(self, fun, model_col='model', data_col='data'):
        results = {}
        results['model'] = fun(model_col)
        if self.y_true is not None:
            results['data'] = fun(data_col)
        return results
