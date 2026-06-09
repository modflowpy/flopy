"""
Test functions in flopy/export/shapefile_utils.py
"""

import numpy as np
import pytest
from modflow_devtools.markers import requires_pkg

import flopy
from flopy.discretization import StructuredGrid, UnstructuredGrid, VertexGrid
from flopy.discretization.grid import Grid
from flopy.export.shapefile_utils import model_attributes_to_shapefile, shp2recarray
from flopy.utils.crs import get_shapefile_crs

from .test_export import disu_sim
from .test_grid import minimal_unstructured_grid_info, minimal_vertex_grid_info


@requires_pkg("geopandas")
def test_model_attributes_to_shapefile(example_data_path, function_tmpdir):
    # freyberg mf2005 model
    name = "freyberg"
    namfile = f"{name}.nam"
    ws = example_data_path / name
    m = flopy.modflow.Modflow.load(namfile, model_ws=ws, check=False, verbose=False)
    shpfile_path = function_tmpdir / f"{name}.shp"
    pkg_names = ["DIS", "BAS6", "LPF", "WEL", "RIV", "RCH", "OC", "PCG"]
    gdf = m.to_geodataframe(package_names=pkg_names)
    gdf.to_file(shpfile_path)
    assert shpfile_path.exists()

    # freyberg mf6 model
    name = "mf6-freyberg"
    sim = flopy.mf6.MFSimulation.load(sim_name=name, sim_ws=example_data_path / name)
    m = sim.get_model()
    shpfile_path = function_tmpdir / f"{name}.shp"
    pkg_names = ["dis", "bas6", "npf", "wel", "riv", "rch", "oc", "pcg"]
    gdf = m.to_geodataframe(package_names=pkg_names)
    gdf.to_file(shpfile_path)
    assert shpfile_path.exists()

    # model with a DISU grid with no angldegx arrays
    # (https://github.com/modflowpy/flopy/issues/1775)
    name = "mf6-disu"
    sim = disu_sim(name, function_tmpdir, missing_arrays=True)
    m = sim.get_model(name)
    shpfile_path = function_tmpdir / f"{name}.shp"
    pkg_names = ["dis"]
    gdf = m.to_geodataframe(package_names=pkg_names)
    gdf.to_file(shpfile_path)
    assert shpfile_path.exists()


@requires_pkg("geopandas")
def test_model_attributes_to_shapefile_modelgrid_kwarg(function_tmpdir):
    """Repro https://github.com/modflowpy/flopy/issues/2744

    The modelgrid kwarg to model_attributes_to_shapefile should be used as
    the geometry source if provided, overriding the model's own modelgrid.
    """
    import warnings

    nrow, ncol = 3, 4
    delr = np.ones(ncol) * 10.0
    delc = np.ones(nrow) * 10.0
    crs = 26916

    # Model without a DIS package: modelgrid is a bare Grid with no geometry.
    # Without the fix, this reproduces the reported error:
    #   TypeError: Grid.to_geodataframe() missing 1 required positional argument: 'features'
    sim = flopy.mf6.MFSimulation(sim_name="test", sim_ws=str(function_tmpdir))
    gwf = flopy.mf6.ModflowGwf(sim, modelname="test")
    mg = StructuredGrid(delr=delr, delc=delc, nlay=1, crs=crs)
    shpfile = function_tmpdir / "test_no_dis.shp"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        model_attributes_to_shapefile(shpfile, gwf, modelgrid=mg)
    assert shpfile.exists()

    # The error in the issue could also have been avoided by retrieving the
    # modelgrid after attaching the DIS to the model. Before DIS is added,
    # gwf.modelgrid is the bare base Grid class. After DIS is attached, it
    # becomes a StructuredGrid whose to_geodataframe() needs no 'features'.
    sim1 = flopy.mf6.MFSimulation(sim_name="test_b", sim_ws=str(function_tmpdir))
    gwf1 = flopy.mf6.ModflowGwf(sim1, modelname="test_b")
    assert isinstance(gwf1.modelgrid, Grid)
    flopy.mf6.ModflowGwfdis(gwf1, nlay=1, nrow=nrow, ncol=ncol)
    mg1 = gwf1.modelgrid
    assert isinstance(mg1, StructuredGrid)
    shpfile1 = function_tmpdir / "test_dis_first.shp"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        model_attributes_to_shapefile(
            shpfile1, gwf1, package_names=["dis"], modelgrid=mg1
        )
    assert shpfile1.exists()

    # Without a modelgrid kwarg and no DIS, to_geodataframe raises a clear
    # AttributeError rather than the cryptic TypeError about 'features'.
    sim_err = flopy.mf6.MFSimulation(sim_name="test_err", sim_ws=str(function_tmpdir))
    gwf_err = flopy.mf6.ModflowGwf(sim_err, modelname="test_err")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(AttributeError, match="discretization package"):
            model_attributes_to_shapefile(function_tmpdir / "test_err.shp", gwf_err)

    # Model with a DIS package but a different modelgrid passed via kwarg:
    # the kwarg modelgrid's CRS should take precedence.
    sim2 = flopy.mf6.MFSimulation(sim_name="test2", sim_ws=str(function_tmpdir))
    gwf2 = flopy.mf6.ModflowGwf(sim2, modelname="test2")
    flopy.mf6.ModflowGwfdis(gwf2, nlay=1, nrow=nrow, ncol=ncol)

    mg2 = StructuredGrid(delr=delr, delc=delc, nlay=1, crs=crs)
    shpfile2 = function_tmpdir / "test_with_dis.shp"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        model_attributes_to_shapefile(
            shpfile2, gwf2, package_names=["dis"], modelgrid=mg2
        )
    assert shpfile2.exists()


@requires_pkg("geopandas")
def test_create_geodataframe(
    minimal_unstructured_grid_info, minimal_vertex_grid_info, function_tmpdir
):
    d = minimal_unstructured_grid_info
    delr = np.ones(10)
    delc = np.ones(10)
    crs = 26916
    shapefilename = function_tmpdir / "grid.shp"
    sg = StructuredGrid(delr=delr, delc=delc, nlay=1, crs=crs)
    gdf = sg.to_geodataframe()
    gdf.to_file(shapefilename)

    data = gdf.to_records()
    # check that row and column appear as integers in recarray
    assert np.issubdtype(data.dtype["row"], np.integer)
    assert np.issubdtype(data.dtype["col"], np.integer)
    assert len(data) == sg.nnodes
    written_crs = get_shapefile_crs(shapefilename)
    assert written_crs.to_epsg() == crs

    usg = UnstructuredGrid(**d, crs=crs)
    gdf = usg.to_geodataframe()
    gdf.to_file(shapefilename)

    data = gdf.to_records()
    assert len(data) == usg.nnodes
    written_crs = get_shapefile_crs(shapefilename)
    assert written_crs.to_epsg() == crs

    d = minimal_vertex_grid_info
    vg = VertexGrid(**d, crs=crs)
    gdf = vg.to_geodataframe()
    gdf.to_file(shapefilename)

    data = gdf.to_records()
    assert len(data) == vg.nnodes
    written_crs = get_shapefile_crs(shapefilename)
    assert written_crs.to_epsg() == crs
