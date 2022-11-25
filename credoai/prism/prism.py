from typing import List
from credoai.lens import Lens
from credoai.utils import ValidationError, global_logger
from credoai.utils.common import NotRunError
from credoai.prism.compare import Compare
from credoai.prism.task import Task


class Prism:
    def __init__(self, pipelines: List[Lens], task: Task):
        self.pipelines = pipelines
        self.task = task
        self._validate_pipeline()
        self.run_flag = False
        self.compare_results: List = []
        self.results: List = []

    def _validate_pipeline(self):
        for step in self.pipelines:
            if not isinstance(step, Lens):
                raise ValidationError("Step must be a Lens instance")

    def run(self):
        for step in self.pipelines:
            try:
                step.get_results()
                global_logger.info(f"{step.model.name} pipeline already run")
            except NotRunError:
                step.run()
                global_logger.info("Running step")

        self.run_flag = True

    def execute(self):
        # Check if already executed
        if not self.run_flag:
            self.run()

        self.results.append(self.task(pipelines=self.pipelines).run().get_results())

    def get_results(self):
        return self.results
