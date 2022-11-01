from typing import List, Optional
from credoai.lens import Lens
from credoai.utils import ValidationError, global_logger
from credoai.utils.common import NotRunError


# from credoconnect import governance
# TODO: extend governance in credoconnect to include extra functionality:
# - Coordinating multiple Lens run (dependance on multiple datasets/models)
# - New evidences that are only valid for Prism, e.g., comparisons...


class Prism:
    def __init__(self, network: Optional[List[Lens]] = None):
        self.network = network
        self._validate_pipeline()

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
