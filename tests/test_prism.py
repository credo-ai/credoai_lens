from copy import deepcopy

from credoai.evaluators import Performance
from credoai.prism import Prism
from credoai.prism.compare import Compare


def test_prism_comparator(init_lens_credit):
    lens, _, _ = init_lens_credit

    # Duplicate instance
    lens2 = deepcopy(lens)
    lens2.model.name = "credit_classifier_duplicate"

    # Add evaluators
    lens.add(Performance(["accuracy_score"]))
    lens2.add(Performance(["accuracy_score"]))

    # Init lens and run
    prism_test = Prism([lens2, lens], task=Compare(ref="credit_default_classifier"))
    prism_test.execute()
    assert prism_test.get_results()
