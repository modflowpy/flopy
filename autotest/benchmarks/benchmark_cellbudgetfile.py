from pathlib import Path

import pytest

from flopy.utils import CellBudgetFile


@pytest.fixture
def cbcf(example_data_path) -> CellBudgetFile:
    return CellBudgetFile(
        example_data_path
        / "mf6"
        / "create_tests"
        / "test021_twri"
        / "expected_output"
        / "twri.cbc"
    )


@pytest.mark.benchmark
def test_cellbudgetfile_load(benchmark, cbcf):
    benchmark(lambda: CellBudgetFile(cbcf.filename))


@pytest.mark.benchmark
def test_cellbudgetfile_get_data_all(benchmark, cbcf):
    unique_records = cbcf.headers[["text", "imeth"]].drop_duplicates()
    term = unique_records.iloc[0]["text"]
    benchmark(lambda: cbcf.get_data(text=term))


@pytest.mark.benchmark
def test_cellbudgetfile_get_data_one(benchmark, cbcf):
    benchmark(lambda: cbcf.get_data(kstpkper=(0, 1)))


@pytest.mark.benchmark
def test_cellbudgetfile_list_records(benchmark, cbcf):
    benchmark(cbcf.list_records)


@pytest.mark.benchmark
def test_cellbudgetfile_list_unique_records(benchmark, cbcf):
    benchmark(cbcf.list_unique_records)


@pytest.mark.benchmark
def test_cellbudgetfile_get_times(benchmark, cbcf):
    benchmark(cbcf.get_times)


@pytest.mark.benchmark
def test_cellbudgetfile_get_kstpkper(benchmark, cbcf):
    benchmark(cbcf.get_kstpkper)
