from credoai.prism import Comparator
from credoai.evaluators.utils.validation import check_instance
from credoai.evidence import MetricContainer
from credoai.utils.common import ValidationError

import pandas as pd
import numpy as np
from copy import deepcopy


# Utility for EvidenceType -> ComparatorType


class MetricComparator(Comparator):
    """
    Class for comparing metric evidence objects

    Supported comparisons for Metrics include difference, maximal values for each metric,
    and minimal values for each metric. Comparisons are evaluated on intersection of the
    metrics represented in the two provided MetricContainer objects. Comparison values for
    non-intersecting metrics will be NoneType wrapped in output container.

    Inputs:
        EvidenceContainers: dictionary of {name_of_model: MetricContainer} key-value pairs

    Output, stored in a LensComparison object, is result of comparisons.
    """

    def __init__(self, EvidenceContainers):
        # attributes all comparators will need
        self.EvidenceContainers = EvidenceContainers
        self.evaluations = set()
        self._setup()
        self._validate()

    def _setup(self):
        ##Evaluations are akin to metrics.
        # Eg. MetricContainer contains a df with results for various metrics
        # We want a list of all metrics run across all EvidenceContainers supplied to the Comparator
        for container in self.EvidenceContainers.values():
            for metric in container.df.type:
                self.evaluations.add(metric)

        self.comparisons = {}
        # internal container for tracking the results of comparisons

        # some metadata...?
        # labels?

    def _validate(self):
        """
        Check that provided containers are all MetricContainer type
        Check that len >= 2
        """
        for container in self.EvidenceContainers.values():
            check_instance(container, MetricContainer)

        if len(self.EvidenceContainers) < 2:
            raise ValidationError("Expected multiple evidence objects to compare.")

    def to_output_container(self):
        """
        Converts self.comparisons results to PrismContainer object which can then be passed (through some more abstraction)
        to the Credo AI Governance platform
        """
        pass

    # Comparison types (Not clear if these need to be defined in the base class since they won't all apply broadly)
    def scalar_difference(self, abs=False):
        """
        Outputs: N/A
            Adds an output dictionary to self.comparisons:
                Dict structure:
                    Keys: metric names
                    Values: pd.DataFrame objects, each with shape len(self.EvidenceContainers), len(self.EvidenceContainers)
                    DataFrame i contains results for metric i:
                        Pairwise difference between MetricContainers j and k
                        If abs == True, return the abolute difference between metrics results
                        If metric is not measured for Container j or k, DataFrame[j, k] is None
        """
        self.comparisons["scalar_difference"] = {}
        for metric in self.evaluations:
            comparison_dict = {}
            for model_1, metric_ev1 in self.EvidenceContainers.items():
                comparisons = []
                if metric not in metric_ev1.df.type.values:
                    # Needing to assume the underlying dataframe will have column "type"
                    # seems a little brittle
                    comparisons = [None] * len(self.EvidenceContainers)
                else:
                    for metric_ev2 in self.EvidenceContainers.values():
                        if metric in metric_ev2.df.type.values:
                            # Assuming the column with the value will be named "value" is
                            # brittle. As is needing to check the 0-index to get the metric result
                            comparisons.append(
                                metric_ev1.df[
                                    metric_ev1.df["type"] == metric
                                ].value.iloc[0]
                                - metric_ev2.df[
                                    metric_ev2.df["type"] == metric
                                ].value.iloc[0]
                            )
                            if abs:
                                comparisons[-1] = np.abs(comparisons[-1])
                        else:
                            comparisons.append(None)

                comparison_dict[model_1] = comparisons
            comparison_df = pd.DataFrame.from_dict(
                comparison_dict, orient="index", columns=self.EvidenceContainers.keys()
            )
            self.comparisons["scalar_difference"][metric] = comparison_df

    def superlative_eval(self, superlative, superlative_name):
        self.comparisons[superlative_name] = {}
        for metric in self.evaluations:
            self.comparisons[superlative_name][metric] = superlative(
                [
                    ev.df[ev.df["type"] == metric].value.iloc[0]
                    for ev in self.EvidenceContainers.values()
                ]
            )

    def highest_eval(self):
        """
        Outputs: dictionary of values signifying the maximal value for each self.evaluation
        Useful for upper-bounding metrics; E.g. want to get upper bound on parity gaps across models
        e.g. returns {'precision_score_parity_gap': .6, 'precision_score_parity_ratio': .3, ...}
        """
        self.superlative_eval(max, "highest_eval")

    def lowest_eval(self):
        """
        Outputs: dictionary of values signifying the minimal value for each self.evaluation
        Useful for lower-bounding metrics; E.g., want to get a lower bound on performance
        e.g. returns {'precision_score': .4, 'accuracy_score': .53, ...}
        """
        self.superlative_eval(min, "lowest_eval")
