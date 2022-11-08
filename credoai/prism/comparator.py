from abc import ABC


# Utility for EvidenceType -> ComparatorType


class Comparator(ABC):
    """
    Abstract base class for Lens/Prism Comparators

    Comparators provide functionality to assess differences in Lens Evaluator results
    across datasets, models, etc. Some differences will be numeric while others will be
    "change tracking" (a la 'diff').

    User will specify EvidenceContainer objects as input. Comparators output will be in
    a custom-crafted data type and will depend on user input (e.g. output difference vs.
    pointer to model with higher performance).

    Different Comparator will exist for each possible EvidenceContainer input.
    """

    def __init__(self, **EvidenceContainers):
        # attributes all comparators will need
        self.EvidenceContainers = EvidenceContainers
        self.evaluations = []  # to contain all evaluations run, e.g. each metric
        self.comparisons = []  # internal container for tracking results of comparisons
        self._setup()
        self._validate()

    def _setup(self):
        pass
        # some metadata...?
        # labels?

    def _validate(self):
        """
        Check that provided EvidenceContainers are all the same type
        Check that len >= 2
        """
        pass

    def to_output_container(self):
        """
        Converts self.comparisons results to PrismContainer object which can then be passed (through some more abstraction)
        to the Credo AI Governance platform
        """
        pass

    # Comparison types (Not clear if these need to be defined in the base class since they won't all apply broadly)
    def scalar_difference(self, abs=True):
        """
        Outputs: len(self.evaluations) DataFrames each with shape = (len(self.EvidenceContainers), len(self.EvidenceContainers))
        DataFrame i contains pairwise comparisons for evaluation i between each pair of EvidenceContainers j and k
        If self.evaluations[j] or self.evaluations[k] is null, output None
        """
        pass

    def tabular_difference(self):
        """
        Same idea as above except subtracting dataframe outputs (e.g. output of ModelFairness for same metric and sensitive features sets)
        Need to check (validate) that dataframes have the same size


        Need validation to confirm sensitive features and evals are identical
        """
        pass

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

    def get_changes(self):
        """
        For non-numerical Evidence. Outputs a qualitative summary of the differences between the two objects (e.g. models, data, etc.)

        Follow up with Ed/Kyle to see how to proceed (he maybe can solve it on Platform side & save me work (: )
        """
        pass
