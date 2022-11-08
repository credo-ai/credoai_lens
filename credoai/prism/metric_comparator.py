from credoai.prism import Comparator
from credoai.evaluators.utils.validation import check_instance
from credoai.evidence import MetricContainer
from credoai.utils.common import ValidationError

import pandas as pd
import numpy as np


# Utility for EvidenceType -> ComparatorType


class MetricComparator(Comparator):
    """
    Class for comparing metric evidence objects

    User specifies MetricContainer objects as input.

    Supported comparisons for Metrics include difference, maximal values for each metric,
    and minimal values for each metric. Comparisons are evaluated on intersection of the
    metrics represented in the two provided MetricContainer objects. Comparison values for
    non-intersecting metrics will be NoneType wrapped in output container.

    Output, stored in a LensComparison object, is result of comparisons.
    """

    def __init__(self, **EvidenceContainers):
        # attributes all comparators will need
        self.EvidenceContainers = EvidenceContainers
        self.evaluations = []
        self._setup()
        self._validate()

    def _setup(self):
        ##Evaluations are akin to metrics.
        # Eg. MetricContainer contains a df with results for various metrics
        # We want a list of all metrics run across all EvidenceContainers supplied to the Comparator
        for container in self.EvidenceContainers:
            for metric in container:
                self.evaluations.append(metric.name)

        self.comparisons = []
        # internal container for tracking the results of comparisons

        # some metadata...?
        # labels?

    def _validate(self):
        """
        Check that provided containers are all MetricContainer type
        Check that len >= 2
        """
        for container in self.EvidenceContainers:
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
        Outputs: len(self.evaluations) DataFrames each with shape = (len(self.EvidenceContainers), len(self.EvidenceContainers))
        DataFrame i contains pairwise difference between MetricContainers j and k for evaluation (metric) i
        If self.evaluations[j] or self.evaluations[k] is null, output None

        If abs == True, return the abolute difference between metrics results
        """
        for metric in self.evaluations:
            comparison_df = pd.DataFrame()
            for metric_ev1 in self.EvidenceContainers:
                comparisons = []
                if metric not in metric_ev1.df.index:
                    comparisons = [None] * len(self.EvidenceContainers)
                else:
                    for metric_ev2 in self.EvidenceContainers:
                        comparisons.append(
                            metric_ev1.df[metric] - metric_ev2.df[metric]
                        )
                        if abs:
                            comparisons[-1] = np.abs(comparisons[-1])

                comparison_df = pd.concat(
                    comparison_df,
                    pd.Series(
                        comparisons, index=self.EvidenceContainers, name=metric_ev1
                    ),
                )

            self.comparisons.append(comparison_df)

    def highest_eval(self):
        """
        Outputs: dictionary of values signifying the maximal value for each self.evaluation
        Useful for upper-bounding metrics; E.g. want to get upper bound on parity gaps across models
        e.g. returns {'precision_score_parity_gap': .6, 'precision_score_parity_ratio': .3, ...}
        """
        pass

    def lowest_eval(self):
        """
        Outputs: dictionary of values signifying the minimal value for each self.evaluation
        Useful for lower-bounding metrics; E.g., want to get a lower bound on performance
        e.g. returns {'precision_score': .4, 'accuracy_score': .53, ...}
        """
        pass
