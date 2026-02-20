import pytest

from flopy.utils.sfroutputfile import SfrFile


@pytest.mark.benchmark
def test_sfrfile_load(benchmark, example_data_path):
    sfr_file = example_data_path / "sfr_examples" / "test1tr.flw"
    benchmark(lambda: SfrFile(str(sfr_file)))


@pytest.fixture
def sfrf(example_data_path) -> SfrFile:
    return SfrFile(str(example_data_path / "sfr_examples" / "test1tr.flw"))


@pytest.mark.benchmark
def test_sfrfile_get_nstrm(benchmark, sfrf):
    df = sfrf.get_dataframe()
    benchmark(lambda: SfrFile.get_nstrm(df))


@pytest.mark.benchmark
def test_sfrfile_get_results(benchmark, sfrf):
    benchmark(lambda: sfrf.get_results(segment=1, reach=1))


@pytest.mark.benchmark
def test_sfrfile_get_times(benchmark, sfrf):
    benchmark(sfrf.get_times)


@pytest.mark.benchmark
def test_sfrfile_get_dataframe(benchmark, sfrf):
    benchmark(sfrf.get_dataframe)
