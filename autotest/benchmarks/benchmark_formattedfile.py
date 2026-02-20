"""
Benchmarks for flopy.utils.formattedfile.FormattedHeadFile operations including:
- Formatted (ASCII) head file loading
- Data extraction for single stress period
- Time series extraction
- Full data retrieval
"""

import numpy as np
import pytest

from flopy.utils.formattedfile import FormattedHeadFile


@pytest.mark.benchmark
@pytest.mark.slow
def test_formattedfile_load(benchmark, example_data_path):
    pth = example_data_path / "mf2005_test" / "test1tr.githds"
    benchmark(lambda: FormattedHeadFile(pth))


@pytest.fixture
def fhd(example_data_path) -> FormattedHeadFile:
    return FormattedHeadFile(str(example_data_path / "mf2005_test" / "test1tr.githds"))


@pytest.mark.benchmark
def test_formattedfile_get_data_totim(benchmark, fhd):
    times = fhd.get_times()
    mid_time = times[len(times) // 2]
    benchmark(lambda: fhd.get_data(totim=mid_time))


@pytest.mark.benchmark
def test_formattedfile_get_data_first(benchmark, fhd):
    benchmark(lambda: fhd.get_data(idx=0))


@pytest.mark.benchmark
def test_formattedfile_get_data_last(benchmark, fhd):
    times = fhd.get_times()
    benchmark(lambda: fhd.get_data(totim=times[-1]))


@pytest.mark.benchmark
@pytest.mark.slow
def test_formattedfile_get_alldata(benchmark, fhd):
    benchmark(fhd.get_alldata)


@pytest.mark.benchmark
def test_formattedfile_get_ts(benchmark, fhd):
    # Use a valid cell index based on test file dimensions (1, 15, 10)
    benchmark(lambda: fhd.get_ts((0, 7, 5)))


@pytest.mark.benchmark
def test_formattedfile_get_times(benchmark, fhd):
    benchmark(fhd.get_times)


@pytest.mark.benchmark
def test_formattedfile_get_kstpkper(benchmark, fhd):
    benchmark(fhd.get_kstpkper)


@pytest.mark.benchmark
@pytest.mark.slow
def test_formattedfile_many_stress_periods(benchmark, fhd):
    benchmark(fhd.get_alldata)
