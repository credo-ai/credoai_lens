from typing import List

from credoai.lens import Lens
from credoai.prism.task import Task
from credoai.utils import ValidationError, global_logger
from credoai.utils.common import NotRunError


class Prism:
    """
    **Experimental**

    Orchestrates the run of complex operations (Tasks) on
    an arbitrary amount of Lens objects.

    Parameters
    ----------
    lenses : List[Lens]
        A list of Lens objects. The only requirement is for the Lens objects to
        be instantiated with their necessary artifacts. One or multiple Lens objects
        can be provided, each Task will validate that the amount of objects provided
        is suitable for its requirement.
    task : Task
        A task instance, instantiated with all the required parameters.
    """

    def __init__(self, lenses: List[Lens], task: Task):
        self.lenses = lenses
        self.task = task
        self._validate()
        self.run_flag = False
        self.compare_results: List = []
        self.results: List = []

    def _validate(self):
        """
        Validate Prism parameters.

        Raises
        ------
        ValidationError

        """
        for step in self.lenses:
            if not isinstance(step, Lens):
                raise ValidationError("Step must be a Lens instance")
        if not isinstance(self.task, Task):
            raise ValidationError(
                "The parameter task should be an instance of credoai.prism.Task"
            )

    def _run(self):
        """
        Runs the Lens pipelines if they were not already.
        """
        for step in self.lenses:
            try:
                step.get_results()
                global_logger.info(f"{step.model.name} pipeline already run")
            except NotRunError:
                step.run()
                global_logger.info("Running step")

        self.run_flag = True

    def execute(self):
        """
        Executes the task run.
        """
        # Check if already executed
        if not self.run_flag:
            self._run()

        self.results.append(self.task(pipelines=self.lenses).run().get_results())

    def get_results(self):
        """
        Returns prism results.
        """
        return self.results

    def get_pipelines_results(self):
        """
        Returns individual results of all the Lens objects runs.
        """
        return [x.get_results() for x in self.lenses]
