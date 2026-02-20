import pytest

from flopy.utils.mflistfile import Mf6ListBudget


@pytest.fixture
def mf6_lbf(example_data_path) -> Mf6ListBudget:
    return Mf6ListBudget(
        example_data_path / "mf6" / "test001a_Tharmonic" / "flow15.lst"
    )


@pytest.mark.benchmark
def test_mf6listbudget_load(benchmark, mf6_lbf):
    benchmark(lambda: Mf6ListBudget(mf6_lbf.file_name))


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "incremental", [True, False], ids=["incremental", "no_incremental"]
)
def test_mf6listbudget_get_data(benchmark, mf6_lbf, incremental):
    benchmark(lambda: mf6_lbf.get_data(incremental=incremental))


@pytest.mark.benchmark
def test_mf6listbudget_get_budget(benchmark, mf6_lbf):
    benchmark(mf6_lbf.get_budget)


@pytest.mark.benchmark
def test_mf6listbudget_to_dataframe(benchmark, mf6_lbf):
    benchmark(mf6_lbf.get_dataframes)
