import pytest


def test_integration(init_lens_integration):
    lens, temp_file, gov = init_lens_integration

    lens.run()

    pytest.assume(lens.get_results())
    pytest.assume(lens.send_to_governance())
    pytest.assume(gov.export(temp_file))
    # Send results to Credo API Platform
    pytest.assume(gov.export())
