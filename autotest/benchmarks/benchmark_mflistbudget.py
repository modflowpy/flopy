import pytest

from flopy.utils.mflistfile import MfListBudget


@pytest.fixture
def mf_lbf(example_data_path) -> MfListBudget:
    return MfListBudget(
        example_data_path / "freyberg_multilayer_transient" / "freyberg.list"
    )


@pytest.mark.benchmark
def test_mflistbudget_load(benchmark, mf_lbf):
    benchmark(lambda: MfListBudget(mf_lbf.file_name))
