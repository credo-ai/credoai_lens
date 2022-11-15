from typing import List
from credoai.lens import Lens
from credoai.utils import ValidationError, global_logger
from credoai.utils.common import NotRunError
from credoai.prism.compare import Compare


class Prism:
    SUPPORTED_TASKS = {"compare": Compare}

    def __init__(self, network: List[Lens], tasks: dict):
        self.network = network
        self.tasks = tasks
        self.supported_tasks = {
            k: v for k, v in tasks.items() if k in self.SUPPORTED_TASKS
        }
        self._validate_pipeline()
        self.run_flag = False
        self.compare_results: List = []
        self.results = []

    def _validate_pipeline(self):
        if not self.supported_tasks:
            raise ValidationError("No supported tasks were found")
        for step in self.network:
            if not isinstance(step, Lens):
                raise ValidationError("Step must be a Lens instance")

    def run(self):
        for step in self.network:
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
        for task_name, task_params in self.supported_tasks.items():
            task = self.SUPPORTED_TASKS[task_name](self.network, **task_params)
            self.results.append(task.run().get_results())

    def get_results(self):
        return self.results
