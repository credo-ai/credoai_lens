"""
Abstract Class for Prism tasks.
"""
from abc import ABC, abstractclassmethod
from typing import List
from credoai.lens import Lens


class Task(ABC):
    """
    Abstract class for prism tasks.

    A task is any operation that can be run on a set of Lens objects or their results.

    Tasks can be run by prism directly, or they can cross reference. For example,
    a task could:

        1. Create multiple instances of Lens objects, e.g., by sampling the assessment dataset
        2. Call another task to compare or extract statistics from the resulting Lens runs.
    """

    def __init__(self):
        self.pipelines: List[Lens] = []

    @abstractclassmethod
    def _validate(self):
        """
        Method encompassing any form of task parameters validation.
        """
        ...

    @abstractclassmethod
    def _setup(self):
        """
        This method holds any setup procedure necessary for the task to run effectively.
        """
        ...

    def __call__(self, **kwargs):
        """
        This method is used internally by Prism to pass lens pipelines to a task.
        """
        self.__dict__.update(kwargs)
        self._validate()
        self._setup()
        return self
