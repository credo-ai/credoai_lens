from abc import ABC, abstractclassmethod


# Utility for EvidenceType -> ComparatorType


class Comparator(ABC):
    """
    Abstract base class for Lens/Prism Comparators

    Comparators provide functionality to assess differences in Lens Evaluator results
    across datasets, models, etc. Some differences will be numeric while others will be
    "change tracking".

    Inputs:
        EvidenceContainers: Iterable of EvidenceContainer objects


    Different Comparator will exist for each possible EvidenceContainer input.
    """

    def __init__(self, EvidenceContainers):
        # attributes all comparators will need
        self.EvidenceContainers = EvidenceContainers
        self.evaluations = set()  # to contain all evaluations run, e.g. each metric
        self.comparisons = {}  # internal container for tracking results of comparisons
        self._validate()
        self._setup()

    @abstractclassmethod
    def _setup(self):
        """"""
        ...

    @abstractclassmethod
    def _validate(self):
        """
        Check that provided EvidenceContainers are all the same type
        Check that len >= 2
        """
        ...

    @abstractclassmethod
    def compare(self):
        """The main function to run the comparison logic"""
        ...
