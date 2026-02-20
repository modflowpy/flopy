"""
Benchmarks for flopy.utils.HeadFile operations including:
- HeadFile initialization
- get_data() for single time steps
- get_alldata() for full file reading
- get_ts() for time series extraction
"""

from pathlib import Path

import pytest

from flopy.utils import HeadFile


@pytest.mark.benchmark
def test_headfile_load(benchmark, example_data_path):
    pth = (
        example_data_path
        / "mf6"
        / "create_tests"
        / "test021_twri"
        / "expected_output"
        / "twri.hds"
    )
    benchmark(lambda: HeadFile(pth))


@pytest.fixture
def hdsf(example_data_path) -> HeadFile:
    return HeadFile(
        example_data_path
        / "mf6"
        / "create_tests"
        / "test021_twri"
        / "expected_output"
        / "twri.hds"
    )


@pytest.mark.benchmark
def test_headfile_get_data_single(benchmark, hdsf):
    times = hdsf.get_times()
    mid_time = times[len(times) // 2] if len(times) > 1 else times[0]
    benchmark(lambda: hdsf.get_data(totim=mid_time))


@pytest.mark.benchmark
def test_headfile_get_alldata(benchmark, hdsf):
    benchmark(hdsf.get_alldata)


@pytest.mark.benchmark
def test_headfile_get_ts(benchmark, hdsf):
    benchmark(lambda: hdsf.get_ts((0, 10, 10)))


@pytest.mark.benchmark
def test_headfile_get_kstpkper(benchmark, hdsf):
    # Use the first available kstpkper from the file
    kstpkpers = hdsf.get_kstpkper()
    kstpkper = kstpkpers[0] if kstpkpers else (0, 0)
    benchmark(lambda: hdsf.get_data(kstpkper=kstpkper))


@pytest.mark.benchmark
def test_headfile_list_records(benchmark, hdsf):
    benchmark(hdsf.list_records)


@pytest.mark.benchmark
def test_headfile_get_times(benchmark, hdsf):
    benchmark(hdsf.get_times)


@pytest.mark.benchmark
def test_headfile_get_kstpkper_list(benchmark, hdsf):
    benchmark(hdsf.get_kstpkper)


@pytest.mark.benchmark
def test_headfile_get_alldata_mf6(benchmark, hdsf):
    benchmark(hdsf.get_alldata)
