from typing import List, Optional, Union
from credoai.lens import Lens
from credoai.utils import ValidationError, global_logger
from credoai.utils.common import NotRunError


# from credoconnect import governance
# TODO: extend governance in credoconnect to include extra functionality:
# - Coordinating multiple Lens run (dependance on multiple datasets/models)
# - New evidences that are only valid for Prism, e.g., comparisons...

# TODO:Add a way of linking datasets to models so we can build a pipeline of
# Lens steps fulfilling governance requirements.


class Prism:
    def __init__(
        self,
        network: Optional[List[Lens]] = None,
        comparisons: Union[list, dict] = None,
        model_inputs: List[model_like] = None,
        datasets_inputs: List[datasets] = None,
        governance=None,
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

    def _create_step_id(self):
        """
        Creates a step id

        Function of artifacts, maybe some hash
        """
        pass

    def run(self):
        for step in self.network:
            try:
                step.get_results()
                global_logger.info("Step already run")
            except NotRunError:
                step.run()
                global_logger.info("Running step")

        self.run_flag = True

    def compare(
        self,
    ):
        """
        Compare method, takes:
        - Steps results
        - Comparators objects
        - Indication of what comparisons to run
        - which steps to compare (1 vs all, a couple, n -> time comparisons)
        """
        if not self.run_flag:
            self.run()
        self.compare_results.append(do_something())

    def _pipeline_builder(self):
        """Depends on governance"""
        ...

    def get_results(self):
        return self.compare_results
