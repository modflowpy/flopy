import pytest

from autotest.benchmarks.benchmark_pathlinefile import ex01_mp7_model
from autotest.test_mp7 import ex01_mf6_model
from flopy.utils.modpathfile import EndpointFile


@pytest.fixture
def epf(ex01_mp7_model) -> EndpointFile:
    mp, ws = ex01_mp7_model
    mp.write_input()
    success, buff = mp.run_model()
    assert success
    return EndpointFile(ws / f"{mp.name}.mpend")


@pytest.mark.benchmark
def test_endpointfile_load(benchmark, epf):
    benchmark(lambda: EndpointFile(epf.fname))


@pytest.mark.benchmark
def test_endpointfile_get_data(benchmark, epf):
    benchmark(epf.get_data)


@pytest.mark.benchmark
def test_endpointfile_get_alldata(benchmark, epf):
    benchmark(epf.get_alldata)


@pytest.mark.benchmark
def test_endpointfile_to_geodataframe(benchmark, ex01_mf6_model, epf):
    pytest.importorskip("geopandas")
    sim, function_tmpdir = ex01_mf6_model
    gwf = sim.get_model()
    benchmark(lambda: epf.to_geodataframe(gwf.modelgrid))
