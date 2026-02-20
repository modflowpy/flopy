import pytest
from modflow_devtools.misc import has_pkg

from .conftest import load_mf6_sim, load_mf2005_model


@pytest.mark.benchmark
def test_mf6_export_shapefile(benchmark, function_tmpdir):
    sim = load_mf6_sim(function_tmpdir, model_key="freyberg")
    gwf = sim.get_model()
    output_path = function_tmpdir / "export.shp"
    benchmark(lambda: gwf.export(str(output_path)))


@pytest.mark.benchmark
def test_mf2005_export_shapefile(benchmark, function_tmpdir):
    model = load_mf2005_model(function_tmpdir, model_key="freyberg")
    output_path = function_tmpdir / "export_mf2005.shp"
    benchmark(lambda: model.export(str(output_path)))


@pytest.mark.benchmark
@pytest.mark.skipif(not has_pkg("netCDF4"), reason="requires netCDF4")
def test_mf6_export_netcdf(benchmark, function_tmpdir):
    import uuid

    sim = load_mf6_sim(function_tmpdir, model_key="freyberg")
    gwf = sim.get_model()

    def export_netcdf():
        # Use unique filename for each iteration to avoid file locking issues on Windows
        output_path = function_tmpdir / f"export_{uuid.uuid4().hex[:8]}.nc"
        gwf.export(str(output_path), fmt="netcdf")

    benchmark(export_netcdf)


@pytest.mark.benchmark
@pytest.mark.skipif(not has_pkg("geopandas"), reason="requires geopandas")
def test_mf6_modelgrid_to_geodataframe(benchmark, function_tmpdir):
    sim = load_mf6_sim(function_tmpdir, model_key="freyberg")
    gwf = sim.get_model()
    benchmark(gwf.modelgrid.to_geodataframe)


@pytest.mark.benchmark
@pytest.mark.skipif(not has_pkg("vtk"), reason="requires vtk")
def test_mf6_export_vtk(benchmark, function_tmpdir):
    sim = load_mf6_sim(function_tmpdir, model_key="freyberg")
    gwf = sim.get_model()
    output_path = function_tmpdir / "export.vtk"
    benchmark(lambda: gwf.export(str(output_path), fmt="vtk"))
