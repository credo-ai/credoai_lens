from abc import ABC, abstractclassmethod


class Task(ABC):
    """
    Abstract class for prism tasks.

    A taks is any operation that prism might execute on a collection
    of Lens objects.
    """

    def __init__(self):
        self.pipelines: list = []

    @abstractclassmethod
    def _validate(self):
        ...

    @abstractclassmethod
    def _setup(self):
        ...

    def __call__(self, **kwargs):
        self.__dict__.update(kwargs)
        self._validate()
        self._setup()
        return self
