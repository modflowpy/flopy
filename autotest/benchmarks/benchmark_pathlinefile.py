import pytest

from autotest.test_mp7 import ex01_mf6_model, ex01_mf6_model_name
from flopy.modpath.mp7 import Modpath7
from flopy.utils.modpathfile import PathlineFile


@pytest.fixture
def ex01_mp7_model(ex01_mf6_model):
    sim, function_tmpdir = ex01_mf6_model
    success, buff = sim.run_simulation()
    assert success, buff
    gwf = sim.get_model(ex01_mf6_model_name)
    mpnam = f"{ex01_mf6_model_name}_mp"
    mp_ws = function_tmpdir / "mp7"
    mp_ws.mkdir()
    return Modpath7.create_mp7(
        modelname=mpnam,
        trackdir="forward",
        flowmodel=gwf,
        exe_name="mp7",
        model_ws=mp_ws,
        rowcelldivisions=1,
        columncelldivisions=1,
        layercelldivisions=1,
    ), mp_ws


@pytest.fixture
def plf(ex01_mp7_model) -> PathlineFile:
    mp, ws = ex01_mp7_model
    mp.write_input()
    success, buff = mp.run_model()
    assert success
    return PathlineFile(ws / f"{mp.name}.mppth")


@pytest.mark.benchmark
def test_pathlinefile_load(benchmark, plf):
    benchmark(lambda: PathlineFile(plf.fname))


@pytest.mark.benchmark
def test_pathlinefile_to_geodataframe(benchmark, ex01_mf6_model, plf):
    pytest.importorskip("geopandas")
    sim, function_tmpdir = ex01_mf6_model
    gwf = sim.get_model()
    benchmark(lambda: plf.to_geodataframe(gwf.modelgrid))


@pytest.mark.benchmark
def test_pathlinefile_get_data(benchmark, plf):
    benchmark(plf.get_data)


@pytest.mark.benchmark
def test_pathlinefile_get_alldata(benchmark, plf):
    benchmark(plf.get_alldata)


@pytest.mark.benchmark
def test_pathlinefile_get_destination_data(benchmark, plf):
    dest_cells = list(range(0, 100))
    benchmark(lambda: plf.get_destination_pathline_data(dest_cells))
