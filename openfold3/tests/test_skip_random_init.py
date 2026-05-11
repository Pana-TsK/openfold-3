import openfold3.core.model.primitives.initialization as initialization
from openfold3.entry_points.experiment_runner import skip_random_init


def test_skip_random_init_context_manager():
    original_func = initialization.trunc_normal_init_

    with skip_random_init():
        # function should be nop
        assert initialization.trunc_normal_init_ is not original_func
        assert initialization.trunc_normal_init_.__name__ == "noop_init"

    # function should be restored
    assert initialization.trunc_normal_init_ is original_func


