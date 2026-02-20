import pytest

from flopy.utils import HeadUFile


@pytest.mark.benchmark
def test_headufile_load(benchmark, example_data_path):
    hds_file = example_data_path / "unstructured" / "headu.githds"
    benchmark(lambda: HeadUFile(str(hds_file)))


@pytest.fixture
def huf(example_data_path) -> HeadUFile:
    return HeadUFile(str(example_data_path / "unstructured" / "headu.githds"))


@pytest.mark.benchmark
def test_headufile_get_data(benchmark, huf):
    times = huf.get_times()
    mid_time = times[len(times) // 2] if len(times) > 0 else times[0]
    benchmark(lambda: huf.get_data(totim=mid_time))


@pytest.mark.benchmark
@pytest.mark.slow
def test_headufile_get_alldata(benchmark, huf):
    benchmark(huf.get_alldata)


@pytest.mark.benchmark
def test_headufile_get_ts(benchmark, huf):
    benchmark(lambda: huf.get_ts(0))
