import pytest

from flopy.utils.mflistfile import MfusgListBudget


@pytest.fixture
def mfusg_lbf(example_data_path) -> MfusgListBudget:
    return MfusgListBudget(
        example_data_path
        / "mfusg_test"
        / "03A_conduit_unconfined"
        / "output"
        / "ex3A.lst"
    )


@pytest.mark.benchmark
def test_mfusglistbudget_load(benchmark, mfusg_lbf):
    benchmark(lambda: MfusgListBudget(mfusg_lbf.file_name))
