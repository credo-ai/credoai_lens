from typing import List, Optional, Union
from credoai.lens import Lens
from credoai.utils import ValidationError, global_logger
from credoai.utils.common import NotRunError


class Prism:
    def __init__(
        self,
        network: Optional[List[Lens]] = None,
        comparisons: Union[list, dict] = None,
    ):
        self.network = network
        self.comparisons = comparisons
        self._validate_pipeline()
        self.run_flag = False
        self.compare_results: List = []

    def _validate_pipeline(self):
        for step in self.network:
            if not isinstance(step, Lens):
                raise ValidationError("Step must be a Lens instance")
            if self.model is None:
                raise ValidationError(f"No model found for pipeline: {step.model.name}")

    def run(self):
        for step in self.network:
            try:
                step.get_results()
                global_logger.info(f"{step.model.name} pipeline already run")
            except NotRunError:
                step.run()
                global_logger.info("Running step")

        self.run_flag = True

    def compare(self):
        if not self.run_flag:
            self.run()

    def get_results(self):
        return self.compare_results
