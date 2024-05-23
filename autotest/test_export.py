import math
import os
import shutil
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pytest
from flaky import flaky
from modflow_devtools.markers import (
    excludes_platform,
    requires_exe,
    requires_pkg,
)
from modflow_devtools.misc import has_pkg

import flopy
from autotest.conftest import get_example_data_path
from flopy.discretization import StructuredGrid, UnstructuredGrid
from flopy.export import NetCdf
from flopy.export.shapefile_utils import recarray2shp, shp2recarray
from flopy.export.utils import (
    export_array,
    export_array_contours,
    export_contourf,
    export_contours,
)
from flopy.export.vtk import Vtk
from flopy.mf6 import (
    MFSimulation,
    ModflowGwf,
    ModflowGwfchd,
    ModflowGwfdis,
    ModflowGwfdisu,
    ModflowGwfdisv,
    ModflowGwfic,
    ModflowGwfnpf,
    ModflowIms,
    ModflowTdis,
)
from flopy.modflow import Modflow, ModflowDis
from flopy.modpath import Modpath6, Modpath6Bas
from flopy.utils import (
    CellBudgetFile,
    HeadFile,
    PathlineFile,
    import_optional_dependency,
)
from flopy.utils import postprocessing as pp
from flopy.utils.crs import get_authority_crs
from flopy.utils.geometry import Polygon

HAS_PYPROJ = has_pkg("pyproj", strict=True)
if HAS_PYPROJ:
    import pyproj


def namfiles() -> List[Path]:
    mf2005_path = get_example_data_path() / "mf2005_test"
    return list(mf2005_path.rglob("*.nam"))


def disu_sim(name, tmpdir, missing_arrays=False):
    """
    Get a simulation with a GWF model on a DISU grid,
    optionally removing angldegx arrays. In this case
    a warning is currently shown but export proceeds.
    """

    from flopy.utils.gridgen import Gridgen

    Lx = 10000.0
    Ly = 10500.0
    nlay = 3
    nrow = 21
    ncol = 20
    delr = Lx / ncol
    delc = Ly / nrow
    top = 400
    botm = [220, 200, 0]

    ml5 = Modflow()
    dis5 = ModflowDis(
        ml5,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
    )

    g = Gridgen(ml5.modelgrid, model_ws=str(tmpdir))

    xmin = 7 * delr
    xmax = 12 * delr
    ymin = 8 * delc
    ymax = 13 * delc
    rfpoly = [
        [
            [
                (xmin, ymin),
                (xmax, ymin),
                (xmax, ymax),
                (xmin, ymax),
                (xmin, ymin),
            ]
        ]
    ]
    g.add_refinement_features(
        rfpoly,
        "polygon",
        2,
        [
            0,
        ],
    )
    g.build(verbose=False)

    gridprops = g.get_gridprops_disu6()
    if missing_arrays:
        del gridprops["angldegx"]

    sim = MFSimulation(sim_name=name, sim_ws=tmpdir, exe_name="mf6")
    tdis = ModflowTdis(sim)
    ims = ModflowIms(sim)
    gwf = ModflowGwf(sim, modelname=name, save_flows=True)
    dis = ModflowGwfdisu(gwf, **gridprops)

    ic = ModflowGwfic(
        gwf, strt=np.random.random_sample(gwf.modelgrid.nnodes) * 350
    )
    npf = ModflowGwfnpf(
        gwf, k=np.random.random_sample(gwf.modelgrid.nnodes) * 10
    )

    return sim


@pytest.fixture
def unstructured_grid(example_data_path):
    ws = example_data_path / "unstructured"

    # load vertices
    verts = load_verts(ws / "ugrid_verts.dat")

    # load the index list into iverts, xc, and yc
    iverts, xc, yc = load_iverts(ws / "ugrid_iverts.dat", closed=True)

    # create a 3 layer model grid
    ncpl = np.array(3 * [len(iverts)])
    nnodes = np.sum(ncpl)

    top = np.ones(nnodes)
    botm = np.ones(nnodes)

    # set top and botm elevations
    i0 = 0
    i1 = ncpl[0]
    elevs = [100, 0, -100, -200]
    for ix, cpl in enumerate(ncpl):
        top[i0:i1] *= elevs[ix]
        botm[i0:i1] *= elevs[ix + 1]
        i0 += cpl
        i1 += cpl

    return UnstructuredGrid(
        vertices=verts,
        iverts=iverts,
        xcenters=xc,
        ycenters=yc,
        top=top,
        botm=botm,
        ncpl=ncpl,
    )


@requires_pkg("shapefile")
@pytest.mark.parametrize("pathlike", (True, False))
def test_output_helper_shapefile_export(
    pathlike, function_tmpdir, example_data_path
):
    ml = Modflow.load(
        "freyberg.nam",
        model_ws=str(example_data_path / "freyberg_multilayer_transient"),
    )
    head = HeadFile(os.path.join(ml.model_ws, "freyberg.hds"))
    cbc = CellBudgetFile(os.path.join(ml.model_ws, "freyberg.cbc"))

    if pathlike:
        outpath = function_tmpdir / "test-pathlike.shp"
    else:
        outpath = os.path.join(function_tmpdir, "test.shp")
    flopy.export.utils.output_helper(
        outpath,
        ml,
        {"HDS": head, "cbc": cbc},
        mflay=1,
        kper=10,
    )


@requires_pkg("shapefile")
@pytest.mark.slow
def test_freyberg_export(function_tmpdir, example_data_path):
    # steady state
    name = "freyberg"
    namfile = f"{name}.nam"
    ws = example_data_path / name
    m = flopy.modflow.Modflow.load(
        namfile, model_ws=ws, check=False, verbose=False
    )

    # test export at model, package and object levels
    shpfile_path = function_tmpdir / "model.shp"
    m.export(shpfile_path)
    assert shpfile_path.exists()

    shpfile_path = function_tmpdir / "wel.shp"
    m.wel.export(shpfile_path)
    assert shpfile_path.exists()

    shpfile_path = function_tmpdir / "hk.shp"
    m.lpf.hk.export(shpfile_path)
    assert shpfile_path.exists()

    shpfile_path = function_tmpdir / "riv_spd.shp"
    m.riv.stress_period_data.export(shpfile_path)
    assert shpfile_path.exists()

    # transient
    # (doesn't work at model level because the total size of
    #  the attribute fields exceeds the shapefile limit)
    ws = example_data_path / "freyberg_multilayer_transient"
    m = flopy.modflow.Modflow.load(
        namfile,
        model_ws=ws,
        verbose=False,
        load_only=["DIS", "BAS6", "NWT", "OC", "RCH", "WEL", "DRN", "UPW"],
    )
    # test export without instantiating a modelgrid
    m.modelgrid.crs = None
    shape = function_tmpdir / f"{name}_drn_sparse.shp"
    m.drn.stress_period_data.export(shape, sparse=True)
    for suffix in [".dbf", ".shp", ".shx"]:
        part = shape.with_suffix(suffix)
        assert part.exists()
        part.unlink()
    assert not shape.with_suffix(".prj").exists()

    m.modelgrid = StructuredGrid(
        delc=m.dis.delc.array, delr=m.dis.delr.array, crs=3070
    )
    # test export with a modelgrid, regardless of whether or not wkt was found
    m.drn.stress_period_data.export(shape, sparse=True)
    for suffix in [".dbf", ".prj", ".shp", ".shx"]:
        part = shape.with_suffix(suffix)
        assert part.exists()
        part.unlink()

    m.modelgrid = StructuredGrid(
        delc=m.dis.delc.array, delr=m.dis.delr.array, crs=3070
    )
    # verify that attributes have same modelgrid as parent
    assert m.drn.stress_period_data.mg.crs == m.modelgrid.crs
    assert m.drn.stress_period_data.mg.xoffset == m.modelgrid.xoffset
    assert m.drn.stress_period_data.mg.yoffset == m.modelgrid.yoffset
    assert m.drn.stress_period_data.mg.angrot == m.modelgrid.angrot

    # get wkt text from pyproj
    wkt = m.modelgrid.crs.to_wkt()

    # if wkt text was fetched from pyproj
    if wkt is not None:
        # test default package export
        shape = function_tmpdir / f"{name}_dis.shp"
        m.dis.export(shape)
        for suffix in [".dbf", ".prj", ".shp", ".shx"]:
            part = shape.with_suffix(suffix)
            assert part.exists()
            if suffix == ".prj":
                assert part.read_text() == wkt
            part.unlink()

        # test default package export to higher level dir ?

        # test sparse package export
        shape = function_tmpdir / f"{name}_drn_sparse.shp"
        m.drn.stress_period_data.export(shape, sparse=True)
        for suffix in [".dbf", ".prj", ".shp", ".shx"]:
            part = shape.with_suffix(suffix)
            assert part.exists()
            if suffix == ".prj":
                assert part.read_text() == wkt


@requires_pkg("shapefile")
@pytest.mark.parametrize("missing_arrays", [True, False])
@pytest.mark.slow
def test_disu_export(function_tmpdir, missing_arrays):
    name = "export_disu"
    # check that missing angldegx array is tolerated
    # https://github.com/modflowpy/flopy/issues/1775
    sim = disu_sim(name, function_tmpdir, missing_arrays)
    m = sim.get_model(name)

    # test export at model level
    shpfile_path = function_tmpdir / "model.shp"
    m.export(shpfile_path)
    assert shpfile_path.exists()

    # test export at package level
    shpfile_path = function_tmpdir / "disu.shp"
    m.disu.export(shpfile_path)
    assert shpfile_path.exists()


# for now, test with and without a coordinate reference system
@pytest.mark.parametrize("crs", (None, 26916))
@requires_pkg("netCDF4", "pyproj")
def test_export_output(crs, function_tmpdir, example_data_path):
    ml = Modflow.load(
        "freyberg.nam", model_ws=str(example_data_path / "freyberg")
    )
    ml.modelgrid.crs = crs
    hds_pth = os.path.join(ml.model_ws, "freyberg.githds")
    hds = flopy.utils.HeadFile(hds_pth)

    out_pth = function_tmpdir / f"freyberg_{crs}.out.nc"
    nc = flopy.export.utils.output_helper(
        out_pth, ml, {"freyberg.githds": hds}
    )
    var = nc.nc.variables.get("head")
    arr = var[:]
    ibound_mask = ml.bas6.ibound.array == 0
    arr_mask = arr.mask[0]
    assert np.array_equal(ibound_mask, arr_mask)

    # close the netcdf file
    nc.nc.close()

    # verify that the CRS was written correctly
    import netCDF4
    import pyproj

    ds = netCDF4.Dataset(out_pth)
    read_crs = pyproj.CRS.from_cf(ds["latitude_longitude"].__dict__)
    # currently, NetCDF files are only written
    # in the 4326 coordinate reference system
    # (lat/lon WGS 84)
    assert read_crs == get_authority_crs(4326)


@requires_pkg("shapefile")
def test_write_gridlines_shapefile(function_tmpdir):
    import shapefile

    from flopy.discretization import StructuredGrid
    from flopy.export.shapefile_utils import write_gridlines_shapefile

    sg = StructuredGrid(
        delr=np.ones(10) * 1.1,
        # cell spacing along model rows
        delc=np.ones(10) * 1.1,
        # cell spacing along model columns
        crs=26715,
    )
    outshp = function_tmpdir / "gridlines.shp"
    write_gridlines_shapefile(outshp, sg)

    for suffix in [".dbf", ".shp", ".shx"]:
        assert outshp.with_suffix(suffix).exists()
    assert outshp.with_suffix(".prj").exists() == HAS_PYPROJ

    with shapefile.Reader(str(outshp)) as sf:
        assert sf.shapeType == shapefile.POLYLINE
        assert len(sf) == 22


@requires_pkg("shapefile")
def test_export_shapefile_polygon_closed(function_tmpdir):
    from shapefile import Reader

    xll, yll = 468970, 3478635
    xur, yur = 681010, 3716462

    spacing = 2000

    ncol = int((xur - xll) / spacing)
    nrow = int((yur - yll) / spacing)
    print(nrow, ncol)

    m = flopy.modflow.Modflow("test.nam", crs="EPSG:32614", xll=xll, yll=yll)

    flopy.modflow.ModflowDis(
        m, delr=spacing, delc=spacing, nrow=nrow, ncol=ncol
    )

    shp_file = os.path.join(function_tmpdir, "test_polygon.shp")
    m.dis.export(shp_file)

    shp = Reader(shp_file)
    for shape in shp.iterShapes():
        if len(shape.points) != 5:
            raise AssertionError("Shapefile polygon is not closed!")

    shp.close()


@excludes_platform("Windows")
@requires_pkg("rasterio", "shapefile", "scipy")
def test_export_array(function_tmpdir, example_data_path):
    import rasterio
    from scipy.ndimage import rotate

    namfile = "freyberg.nam"
    model_ws = example_data_path / "freyberg"
    m = flopy.modflow.Modflow.load(
        namfile,
        model_ws=model_ws,
        verbose=False,
        load_only=["DIS", "BAS6"],
    )
    m.modelgrid.set_coord_info(angrot=45)
    nodata = -9999
    export_array(
        m.modelgrid,
        os.path.join(function_tmpdir, "fb.asc"),
        m.dis.top.array,
        nodata=nodata,
    )
    arr = np.loadtxt(function_tmpdir / "fb.asc", skiprows=6)

    m.modelgrid.write_shapefile(function_tmpdir / "grid.shp")

    # check bounds
    with open(function_tmpdir / "fb.asc") as src:
        for line in src:
            if "xllcorner" in line.lower():
                val = float(line.strip().split()[-1])
                assert np.abs(val - m.modelgrid.extent[0]) < 1e-6
                # ascii grid origin will differ if it was unrotated
                # without scipy.rotate
                # assert np.abs(val - m.modelgrid.xoffset) < 1e-6
            if "yllcorner" in line.lower():
                val = float(line.strip().split()[-1])
                assert np.abs(val - m.modelgrid.extent[2]) < 1e-6
                # without scipy.rotate
                # assert np.abs(val - m.modelgrid.yoffset) < 1e-6
            if "cellsize" in line.lower():
                val = float(line.strip().split()[-1])
                rot_cellsize = (
                    np.cos(np.radians(m.modelgrid.angrot))
                    * m.modelgrid.delr[0]
                )
                break

    rotated = rotate(m.dis.top.array, m.modelgrid.angrot, cval=nodata)
    assert rotated.shape == arr.shape

    export_array(
        m.modelgrid,
        function_tmpdir / "fb.tif",
        m.dis.top.array,
        nodata=nodata,
    )
    with rasterio.open(function_tmpdir / "fb.tif") as src:
        arr = src.read(1)
        assert src.shape == (m.nrow, m.ncol)
        # TODO: these tests currently fail -- fix is in progress
        # assert np.abs(src.bounds[0] - m.modelgrid.extent[0]) < 1e-6
        # assert np.abs(src.bounds[1] - m.modelgrid.extent[1]) < 1e-6
        pass


@requires_pkg("netCDF4", "pyproj")
def test_netcdf_classmethods(function_tmpdir, example_data_path):
    namfile = "freyberg.nam"
    name = namfile.replace(".nam", "")
    model_ws = example_data_path / "freyberg_multilayer_transient"
    ml = flopy.modflow.Modflow.load(
        namfile,
        model_ws=model_ws,
        check=False,
        verbose=True,
        load_only=[],
    )

    f = ml.export(function_tmpdir / "freyberg.nc")
    v1_set = set(f.nc.variables.keys())
    fnc = function_tmpdir / "freyberg.new.nc"
    new_f = flopy.export.NetCdf.zeros_like(f, output_filename=fnc)
    v2_set = set(new_f.nc.variables.keys())
    diff = v1_set.symmetric_difference(v2_set)
    assert len(diff) == 0, str(diff)

    # close the netcdf file
    f.nc.close()
    new_f.nc.close()


@requires_pkg("shapefile")
def test_shapefile_ibound(function_tmpdir, example_data_path):
    from shapefile import Reader

    shape_name = os.path.join(function_tmpdir, "test.shp")
    namfile = "freyberg.nam"
    model_ws = example_data_path / "freyberg_multilayer_transient"
    ml = flopy.modflow.Modflow.load(
        namfile,
        model_ws=model_ws,
        check=False,
        verbose=True,
        load_only=["bas6"],
    )
    ml.export(shape_name)
    shape = Reader(shape_name)
    field_names = [item[0] for item in shape.fields][1:]
    ib_idx = field_names.index("ibound_1")
    txt = f"should be int instead of {type(shape.record(0)[ib_idx])}"
    assert isinstance(shape.record(0)[ib_idx], int), txt
    shape.close()


@requires_pkg("shapefile")
@pytest.mark.slow
@pytest.mark.parametrize("namfile", namfiles())
def test_shapefile(function_tmpdir, namfile):
    from shapefile import Reader

    model = flopy.modflow.Modflow.load(
        namfile.name, model_ws=namfile.parent, verbose=False
    )
    assert model, f"Could not load namefile {namfile}"

    msg = f"Could not load {namfile} model"
    assert isinstance(model, flopy.modflow.Modflow), msg

    fnc_name = function_tmpdir / f"{model.name}.shp"
    fnc = model.export(fnc_name)
    # fnc2 = m.export(fnc_name, package_names=None)
    # fnc3 = m.export(fnc_name, package_names=['DIS'])

    s = Reader(fnc_name)
    assert (
        s.numRecords == model.nrow * model.ncol
    ), f"wrong number of records in shapefile {fnc_name}"


@requires_pkg("shapefile")
@pytest.mark.slow
@pytest.mark.parametrize("namfile", namfiles())
def test_shapefile_export_modelgrid_override(function_tmpdir, namfile):
    from shapefile import Reader

    model = flopy.modflow.Modflow.load(
        namfile.name, model_ws=str(namfile.parent), verbose=False
    )
    grid = model.modelgrid
    modelgrid = StructuredGrid(
        grid.delc * 0.3048,
        grid.delr * 0.3048,
        grid.top,
        grid.botm,
        grid.idomain,
        grid.lenuni,
        grid.crs,
        xoff=grid.xoffset,
        yoff=grid.yoffset,
        angrot=grid.angrot,
    )

    assert model, f"Could not load namefile {namfile}"
    assert isinstance(model, flopy.modflow.Modflow)

    fnc_name = function_tmpdir / f"{model.name}.shp"
    model.export(fnc_name, modelgrid=modelgrid)

    # TODO: do we want to test exports with package_names options too?
    #   (both currently fail)
    # fnc2 = model.export(fnc_name, package_names=None)
    # fnc3 = model.export(fnc_name, package_names=['DIS'])

    s = Reader(fnc_name)
    s.close()


@requires_pkg("netCDF4", "pyproj")
@pytest.mark.slow
@pytest.mark.parametrize("namfile", namfiles())
def test_export_netcdf(function_tmpdir, namfile):
    from netCDF4 import Dataset

    model = flopy.modflow.Modflow.load(
        namfile.name, model_ws=namfile.parent, verbose=False
    )
    if model.dis.lenuni == 0:
        model.dis.lenuni = 1

    if model.dis.botm.shape[0] != model.nlay:
        print("skipping...botm.shape[0] != nlay")
        return

    assert model, f"Could not load namefile {namfile}"
    assert isinstance(model, flopy.modflow.Modflow)

    fnc = model.export(function_tmpdir / f"{model.name}.nc")
    fnc.write()
    fnc_name = function_tmpdir / f"{model.name}.nc"
    fnc = model.export(fnc_name)
    fnc.write()

    nc = Dataset(fnc_name, "r")
    nc.close()


@requires_pkg("shapefile")
def test_export_array2(function_tmpdir):
    nrow = 7
    ncol = 11
    crs = 4431

    # no epsg code
    modelgrid = StructuredGrid(
        delr=np.ones(ncol) * 1.1, delc=np.ones(nrow) * 1.1
    )
    filename = os.path.join(function_tmpdir, "myarray1.shp")
    a = np.arange(nrow * ncol).reshape((nrow, ncol))
    export_array(modelgrid, filename, a)
    assert os.path.isfile(filename), "did not create array shapefile"

    # with modelgrid epsg code
    modelgrid = StructuredGrid(
        delr=np.ones(ncol) * 1.1, delc=np.ones(nrow) * 1.1, crs=crs
    )
    filename = os.path.join(function_tmpdir, "myarray2.shp")
    a = np.arange(nrow * ncol).reshape((nrow, ncol))
    export_array(modelgrid, filename, a)
    assert os.path.isfile(filename), "did not create array shapefile"

    # with passing in epsg code
    modelgrid = StructuredGrid(
        delr=np.ones(ncol) * 1.1, delc=np.ones(nrow) * 1.1
    )
    filename = os.path.join(function_tmpdir, "myarray3.shp")
    a = np.arange(nrow * ncol).reshape((nrow, ncol))
    export_array(modelgrid, filename, a, crs=crs)
    assert os.path.isfile(filename), "did not create array shapefile"


@requires_pkg("shapefile", "shapely")
def test_export_array_contours_structured(function_tmpdir):
    nrow = 7
    ncol = 11
    crs = 4431

    # no epsg code
    modelgrid = StructuredGrid(
        delr=np.ones(ncol) * 1.1, delc=np.ones(nrow) * 1.1
    )
    filename = function_tmpdir / "myarraycontours1.shp"
    a = np.arange(nrow * ncol).reshape((nrow, ncol))
    export_array_contours(modelgrid, filename, a)
    assert os.path.isfile(filename), "did not create contour shapefile"

    # with modelgrid coordinate reference
    modelgrid = StructuredGrid(
        delr=np.ones(ncol) * 1.1,
        delc=np.ones(nrow) * 1.1,
        crs=crs,
    )
    filename = function_tmpdir / "myarraycontours2.shp"
    a = np.arange(nrow * ncol).reshape((nrow, ncol))
    export_array_contours(modelgrid, filename, a)
    assert os.path.isfile(filename), "did not create contour shapefile"

    # with passing in coordinate reference
    modelgrid = StructuredGrid(
        delr=np.ones(ncol) * 1.1, delc=np.ones(nrow) * 1.1
    )
    filename = function_tmpdir / "myarraycontours3.shp"
    a = np.arange(nrow * ncol).reshape((nrow, ncol))
    export_array_contours(modelgrid, filename, a, crs=crs)
    assert os.path.isfile(filename), "did not create contour shapefile"


@requires_pkg("shapefile", "shapely")
def test_export_array_contours_unstructured(
    function_tmpdir, unstructured_grid
):
    from shapefile import Reader

    grid = unstructured_grid
    fname = function_tmpdir / "myarraycontours1.shp"
    export_array_contours(grid, fname, np.arange(grid.nnodes))
    assert fname.is_file(), "did not create contour shapefile"

    # visual debugging
    grid.plot(alpha=0.2)
    with Reader(fname) as r:
        shapes = r.shapes()
        for s in shapes:
            x = [i[0] for i in s.points[:]]
            y = [i[1] for i in s.points[:]]
            plt.plot(x, y)

    # plt.show()


from autotest.test_gridgen import sim_disu_diff_layers


@requires_pkg("shapefile", "shapely")
def test_export_array_contours_unstructured_diff_layers(
    function_tmpdir, sim_disu_diff_layers
):
    from shapefile import Reader

    gwf = sim_disu_diff_layers.get_model()
    grid = gwf.modelgrid
    a = np.arange(grid.nnodes)
    for layer in range(3):
        fname = function_tmpdir / f"contours.{layer}.shp"
        export_array_contours(grid, fname, a, layer=layer)
        assert fname.is_file(), "did not create contour shapefile"

    # visual debugging
    fig, axes = plt.subplots(1, 3, subplot_kw={"aspect": "equal"})
    for layer, ax in enumerate(axes):
        fname = function_tmpdir / f"contours.{layer}.shp"
        with Reader(fname) as r:
            shapes = r.shapes()
            for s in shapes:
                x = [i[0] for i in s.points[:]]
                y = [i[1] for i in s.points[:]]
                ax.plot(x, y)
            grid.plot(ax=ax, alpha=0.2, layer=layer)

    # plt.show()


@requires_pkg("shapefile", "shapely")
def test_export_contourf(function_tmpdir, example_data_path):
    from shapefile import Reader

    filename = function_tmpdir / "myfilledcontours.shp"
    mpath = example_data_path / "freyberg"
    ml = Modflow.load("freyberg.nam", model_ws=mpath)
    hds_pth = Path(ml.model_ws) / "freyberg.githds"
    hds = flopy.utils.HeadFile(hds_pth)
    head = hds.get_data()
    levels = np.arange(10, 30, 0.5)

    mapview = flopy.plot.PlotMapView(model=ml)
    contour_set = mapview.contour_array(
        head, masked_values=[999.0], levels=levels, filled=True
    )

    # with pathlib.Path
    export_contourf(filename, contour_set)
    plt.close()
    assert filename.is_file(), "did not create contourf shapefile"

    # with str path
    export_contourf(str(filename), contour_set)
    plt.close()
    assert filename.is_file(), "did not create contourf shapefile"

    with Reader(filename) as r:
        shapes = r.shapes()
        # expect 65 with standard mpl contours (structured grids), 86 with tricontours
        assert (
            len(shapes) >= 65
        ), "multipolygons were skipped in contourf routine"

        # debugging
        # for s in shapes:
        #     x = [i[0] for i in s.points[:]]
        #     y = [i[1] for i in s.points[:]]
        #     plt.plot(x, y)
        # plt.show()


@pytest.mark.mf6
@requires_pkg("shapefile", "shapely")
def test_export_contours(function_tmpdir, example_data_path):
    from shapefile import Reader

    filename = function_tmpdir / "mycontours.shp"
    mpath = example_data_path / "freyberg"
    ml = Modflow.load("freyberg.nam", model_ws=mpath)
    hds_pth = Path(ml.model_ws) / "freyberg.githds"
    hds = flopy.utils.HeadFile(hds_pth)
    head = hds.get_data()
    levels = np.arange(10, 30, 0.5)

    mapview = flopy.plot.PlotMapView(model=ml)
    contour_set = mapview.contour_array(
        head, masked_values=[999.0], levels=levels
    )

    export_contours(filename, contour_set)
    plt.close()
    if not os.path.isfile(filename):
        raise AssertionError("did not create contour shapefile")

    with Reader(filename) as r:
        shapes = r.shapes()
        # expect 65 with standard mpl contours (structured grids), 86 with tricontours
        assert len(shapes) >= 65

        # debugging
        # for s in shapes:
        #     x = [i[0] for i in s.points[:]]
        #     y = [i[1] for i in s.points[:]]
        #     plt.plot(x, y)
        # plt.show()


@pytest.mark.mf6
@requires_pkg("shapely")
def test_mf6_grid_shp_export(function_tmpdir):
    nlay = 2
    nrow = 10
    ncol = 10
    top = 1
    nper = 2
    perlen = 1
    nstp = 1
    tsmult = 1
    perioddata = [[perlen, nstp, tsmult]] * 2
    botm = np.zeros((2, 10, 10))

    m = flopy.modflow.Modflow(
        "junk",
        version="mfnwt",
        model_ws=function_tmpdir,
    )
    dis = flopy.modflow.ModflowDis(
        m,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        nper=nper,
        perlen=perlen,
        nstp=nstp,
        tsmult=tsmult,
        top=top,
        botm=botm,
    )

    smg = StructuredGrid(
        delc=np.ones(nrow),
        delr=np.ones(ncol),
        top=dis.top.array,
        botm=botm,
        idomain=1,
        xoff=10,
        yoff=10,
    )

    # River package (MFlist)
    spd = flopy.modflow.ModflowRiv.get_empty(10)
    spd["i"] = np.arange(10)
    spd["j"] = [5, 5, 6, 6, 7, 7, 7, 8, 9, 9]
    spd["stage"] = np.linspace(1, 0.7, 10)
    spd["rbot"] = spd["stage"] - 0.1
    spd["cond"] = 50.0
    riv = flopy.modflow.ModflowRiv(m, stress_period_data={0: spd})

    # Recharge package (transient 2d)
    rech = {0: 0.001, 1: 0.002}
    rch = flopy.modflow.ModflowRch(m, rech=rech)

    # mf6 version of same model
    mf6name = "junk6"
    sim = flopy.mf6.MFSimulation(
        sim_name=mf6name,
        version="mf6",
        exe_name="mf6",
        sim_ws=function_tmpdir,
    )
    tdis = flopy.mf6.modflow.mftdis.ModflowTdis(
        sim, pname="tdis", time_units="DAYS", nper=nper, perioddata=perioddata
    )
    gwf = flopy.mf6.ModflowGwf(
        sim, modelname=mf6name, model_nam_file=f"{mf6name}.nam"
    )
    dis6 = flopy.mf6.ModflowGwfdis(
        gwf, pname="dis", nlay=nlay, nrow=nrow, ncol=ncol, top=top, botm=botm
    )

    def cellid(k, i, j, nrow, ncol):
        return k * nrow * ncol + i * ncol + j

    # Riv6
    spd6 = flopy.mf6.ModflowGwfriv.stress_period_data.empty(
        gwf, maxbound=len(spd)
    )
    # spd6[0]['cellid'] = cellid(spd.k, spd.i, spd.j, m.nrow, m.ncol)
    spd6[0]["cellid"] = list(zip(spd.k, spd.i, spd.j))
    for c in spd.dtype.names:
        if c in spd6[0].dtype.names:
            spd6[0][c] = spd[c]
    # MFTransient list apparently requires entries for additional stress periods,
    # even if they are the same
    spd6[1] = spd6[0]
    # irch = np.zeros((nrow, ncol))
    riv6 = flopy.mf6.ModflowGwfriv(gwf, stress_period_data=spd6)
    rch6 = flopy.mf6.ModflowGwfrcha(gwf, recharge=rech)

    riv6spdarrays = dict(riv6.stress_period_data.masked_4D_arrays_itr())
    rivspdarrays = dict(riv.stress_period_data.masked_4D_arrays_itr())
    for k, v in rivspdarrays.items():
        assert (
            np.abs(np.nansum(v) - np.nansum(riv6spdarrays[k])) < 1e-6
        ), f"variable {k} is not equal"
        pass

    if not has_pkg("shapefile"):
        return

    # rch6.export('{}/mf6.shp'.format(baseDir))
    m.export(function_tmpdir / "mfnwt.shp")
    gwf.export(function_tmpdir / "mf6.shp")

    # check that the two shapefiles are the same
    ra = shp2recarray(function_tmpdir / "mfnwt.shp")
    ra6 = shp2recarray(function_tmpdir / "mf6.shp")

    # check first and last exported cells
    assert ra.geometry[0] == ra6.geometry[0]
    assert ra.geometry[-1] == ra6.geometry[-1]
    # fields
    different_fields = list(set(ra.dtype.names).difference(ra6.dtype.names))
    different_fields = [
        f for f in different_fields if "thick" not in f and "rech" not in f
    ]
    assert len(different_fields) == 0
    for lay in np.arange(m.nlay) + 1:
        assert np.sum(np.abs(ra[f"rech_{lay}"] - ra6[f"rechar{lay}"])) < 1e-6
    common_fields = set(ra.dtype.names).intersection(ra6.dtype.names)
    common_fields.remove("geometry")
    # array values
    for c in common_fields:
        for it, it6 in zip(ra[c], ra6[c]):
            if math.isnan(it):
                assert math.isnan(it6)
            else:
                assert np.abs(it - it6) < 1e-6


@requires_pkg("shapefile")
@pytest.mark.slow
def test_export_huge_shapefile(function_tmpdir):
    nlay = 2
    nrow = 200
    ncol = 200
    top = 1
    nper = 2
    perlen = 1
    nstp = 1
    tsmult = 1
    # perioddata = [[perlen, nstp, tsmult]] * 2
    botm = np.zeros((nlay, nrow, ncol))

    m = flopy.modflow.Modflow(
        "junk", version="mfnwt", model_ws=function_tmpdir
    )
    flopy.modflow.ModflowDis(
        m,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        nper=nper,
        perlen=perlen,
        nstp=nstp,
        tsmult=tsmult,
        top=top,
        botm=botm,
    )

    m.export(function_tmpdir / "huge.shp")


@requires_pkg("netCDF4", "pyproj")
def test_polygon_from_ij(function_tmpdir):
    """test creation of a polygon from an i, j location using get_vertices()."""
    m = Modflow("toy_model", model_ws=function_tmpdir)

    botm = np.zeros((2, 10, 10))
    botm[0, :, :] = 1.5
    botm[1, 5, 5] = 4  # negative layer thickness!
    botm[1, 6, 6] = 4
    dis = ModflowDis(
        nrow=10, ncol=10, nlay=2, delr=100, delc=100, top=3, botm=botm, model=m
    )

    fname = function_tmpdir / "toy.model.nc"
    ncdf = NetCdf(fname, m)
    ncdf.write()

    fname = function_tmpdir / "toy_model_two.nc"
    m.export(fname)

    fname = function_tmpdir / "toy_model_dis.nc"
    dis.export(fname)

    mg = m.modelgrid
    mg.set_coord_info(
        xoff=mg._xul_to_xll(600000.0, -45.0),
        yoff=mg._yul_to_yll(5170000, -45.0),
        angrot=-45.0,
        crs="EPSG:26715",
    )

    recarray = np.array(
        [
            (0, 5, 5, 0.1, True, "s0"),
            (1, 4, 5, 0.2, False, "s1"),
            (0, 7, 8, 0.3, True, "s2"),
        ],
        dtype=[
            ("k", "<i8"),
            ("i", "<i8"),
            ("j", "<i8"),
            ("stuff", "<f4"),
            ("stuf", "|b1"),
            ("stf", object),
        ],
    ).view(np.recarray)

    # vertices for a model cell
    geoms = [
        Polygon(m.modelgrid.get_cell_vertices(i, j))
        for i, j in zip(recarray.i, recarray.j)
    ]

    assert geoms[0].type == "Polygon"
    assert np.abs(geoms[0].bounds[-1] - 5169292.893203464) < 1e-4


@flaky
@requires_pkg("netCDF4", "pyproj", "shapely")
def test_polygon_from_ij_with_epsg(function_tmpdir):
    ws = function_tmpdir
    m = Modflow("toy_model", model_ws=ws)

    botm = np.zeros((2, 10, 10))
    botm[0, :, :] = 1.5
    botm[1, 5, 5] = 4  # negative layer thickness!
    botm[1, 6, 6] = 4
    dis = ModflowDis(
        nrow=10, ncol=10, nlay=2, delr=100, delc=100, top=3, botm=botm, model=m
    )

    fname = ws / "toy.model.nc"
    ncdf = NetCdf(fname, m)
    ncdf.write()

    fname = ws / "toy_model_two.nc"
    m.export(fname)

    fname = ws / "toy_model_dis.nc"
    dis.export(fname)

    mg = m.modelgrid
    mg.set_coord_info(
        xoff=mg._xul_to_xll(600000.0, -45.0),
        yoff=mg._yul_to_yll(5170000, -45.0),
        angrot=-45.0,
        crs="EPSG:26715",
    )

    recarray = np.array(
        [
            (0, 5, 5, 0.1, True, "s0"),
            (1, 4, 5, 0.2, False, "s1"),
            (0, 7, 8, 0.3, True, "s2"),
        ],
        dtype=[
            ("k", "<i8"),
            ("i", "<i8"),
            ("j", "<i8"),
            ("stuff", "<f4"),
            ("stuf", "|b1"),
            ("stf", object),
        ],
    ).view(np.recarray)

    # vertices for a model cell
    geoms = [
        Polygon(m.modelgrid.get_cell_vertices(i, j))
        for i, j in zip(recarray.i, recarray.j)
    ]

    fpth = os.path.join(ws, "test.shp")
    recarray2shp(recarray, geoms, fpth, crs=26715)

    fpth = os.path.join(ws, "test.prj")
    fpth2 = os.path.join(ws, "26715.prj")
    shutil.copy(fpth, fpth2)
    fpth = os.path.join(ws, "test.shp")
    recarray2shp(recarray, geoms, fpth, prjfile=fpth2)

    # test_dtypes
    fpth = os.path.join(ws, "test.shp")
    ra = shp2recarray(fpth)
    assert "int" in ra.dtype["k"].name
    assert "float" in ra.dtype["stuff"].name
    assert "bool" in ra.dtype["stuf"].name
    assert "object" in ra.dtype["stf"].name


def count_lines_in_file(filepath):
    with open(filepath) as f:
        n = sum(1 for _ in f)
    return n


def is_binary_file(filepath):
    is_binary = False
    with open(filepath) as f:
        try:
            for _ in f:
                pass
        except UnicodeDecodeError:
            is_binary = True
    return is_binary


@requires_pkg("vtk")
def test_vtk_export_array2d(function_tmpdir, example_data_path):
    # test mf 2005 freyberg
    mpath = example_data_path / "freyberg_multilayer_transient"
    namfile = "freyberg.nam"
    m = Modflow.load(
        namfile, model_ws=mpath, verbose=False, load_only=["dis", "bas6"]
    )

    # export and check
    m.dis.top.export(function_tmpdir, name="top", fmt="vtk", binary=False)
    assert count_lines_in_file(function_tmpdir / "top.vtk") == 17615

    # with smoothing
    m.dis.top.export(
        function_tmpdir,
        fmt="vtk",
        name="top_smooth",
        binary=False,
        smooth=True,
    )
    assert count_lines_in_file(function_tmpdir / "top_smooth.vtk") == 17615


@requires_pkg("vtk")
def test_vtk_export_array3d(function_tmpdir, example_data_path):
    # test mf 2005 freyberg
    mpath = example_data_path / "freyberg_multilayer_transient"
    namfile = "freyberg.nam"
    m = Modflow.load(
        namfile,
        model_ws=mpath,
        verbose=False,
        load_only=["dis", "bas6", "upw"],
    )

    # export and check
    m.upw.hk.export(function_tmpdir, fmt="vtk", name="hk", binary=False)
    assert count_lines_in_file(function_tmpdir / "hk.vtk") == 17615

    # with point scalars
    m.upw.hk.export(
        function_tmpdir,
        fmt="vtk",
        name="hk_points",
        point_scalars=True,
        binary=False,
    )
    assert count_lines_in_file(function_tmpdir / "hk_points.vtk") == 19482

    # with point scalars and binary
    m.upw.hk.export(
        function_tmpdir,
        fmt="vtk",
        name="hk_points_bin",
        point_scalars=True,
    )
    assert is_binary_file(function_tmpdir / "hk_points_bin.vtk")


@requires_pkg("vtk")
def test_vtk_transient_array_2d(function_tmpdir, example_data_path):
    # test mf 2005 freyberg
    ws = function_tmpdir
    mpath = example_data_path / "freyberg_multilayer_transient"
    namfile = "freyberg.nam"
    m = Modflow.load(
        namfile,
        model_ws=mpath,
        verbose=False,
        load_only=["dis", "bas6", "rch"],
    )

    kpers = [0, 1, 1096]

    # export and check
    m.rch.rech.export(ws, fmt="vtk", kpers=kpers, binary=False, xml=True)
    assert count_lines_in_file(function_tmpdir / "rech_000001.vtk") == 26837
    assert count_lines_in_file(function_tmpdir / "rech_001096.vtk") == 26837

    # with binary

    m.rch.rech.export(ws, fmt="vtk", binary=True, kpers=kpers)
    assert is_binary_file(function_tmpdir / "rech_000001.vtk")
    assert is_binary_file(function_tmpdir / "rech_001096.vtk")


@requires_pkg("vtk")
@pytest.mark.slow
def test_vtk_add_packages(function_tmpdir, example_data_path):
    # test mf 2005 freyberg
    ws = function_tmpdir
    mpath = example_data_path / "freyberg_multilayer_transient"
    namfile = "freyberg.nam"
    m = Modflow.load(
        namfile,
        model_ws=mpath,
        verbose=False,
        load_only=["dis", "bas6", "upw", "DRN"],
    )

    # dis export and check
    # todo: pakbase.export() for vtk!!!!
    m.dis.export(ws, fmt="vtk", xml=True, binary=False)
    filetocheck = function_tmpdir / "DIS.vtk"
    # totalbytes = os.path.getsize(filetocheck)
    # assert(totalbytes==1019857)
    assert count_lines_in_file(filetocheck) == 27239

    # upw with point scalar output
    m.upw.export(ws, fmt="vtk", xml=True, binary=False, point_scalars=True)
    assert count_lines_in_file(function_tmpdir / "UPW.vtk") == 42445

    # bas with smoothing on
    m.bas6.export(ws, fmt="vtk", binary=False, smooth=True)
    assert count_lines_in_file(function_tmpdir / "BAS6.vtk") == 17883

    # transient package drain
    kpers = [0, 1, 1096]
    m.drn.export(ws, fmt="vtk", binary=False, xml=True, kpers=kpers, pvd=True)
    assert count_lines_in_file(function_tmpdir / "DRN_000001.vtu") == 27239
    assert count_lines_in_file(function_tmpdir / "DRN_001096.vtu") == 27239

    # dis with binary
    m.dis.export(ws, fmt="vtk", binary=True)
    assert is_binary_file(function_tmpdir / "DIS.vtk")

    # upw with point scalars and binary
    m.upw.export(ws, fmt="vtk", point_scalars=True, binary=True)
    assert is_binary_file(function_tmpdir / "UPW.vtk")


@pytest.mark.mf6
@requires_pkg("vtk")
def test_vtk_mf6(function_tmpdir, example_data_path):
    # test mf6
    mf6expth = example_data_path / "mf6"
    mf6sims = [
        "test045_lake1ss_table",
        "test036_twrihfb",
        "test045_lake2tr",
        "test006_2models_mvr",
    ]

    for simnm in mf6sims:
        print(simnm)
        simpth = mf6expth / simnm
        loaded_sim = MFSimulation.load(simnm, "mf6", "mf6", simpth)
        sim_models = loaded_sim.model_names
        print(sim_models)
        for mname in sim_models:
            print(mname)
            m = loaded_sim.get_model(mname)
            m.export(function_tmpdir, fmt="vtk", binary=False)

    # check one
    filetocheck = function_tmpdir / "twrihfb2015_000000.vtk"
    # totalbytes = os.path.getsize(filetocheck)
    # assert(totalbytes==21609)
    assert count_lines_in_file(filetocheck) == 9537


@requires_pkg("vtk")
@pytest.mark.slow
def test_vtk_binary_head_export(function_tmpdir, example_data_path):
    # test mf 2005 freyberg
    mpth = example_data_path / "freyberg_multilayer_transient"
    namfile = "freyberg.nam"
    hdsfile = mpth / "freyberg.hds"
    heads = HeadFile(hdsfile)
    m = Modflow.load(
        namfile, model_ws=mpth, verbose=False, load_only=["dis", "bas6"]
    )
    filetocheck = function_tmpdir / "freyberg_head_000003.vtu"

    # export and check

    vtkobj = Vtk(m, pvd=True, xml=True)
    vtkobj.add_heads(
        heads, kstpkper=[(0, 0), (0, 199), (0, 354), (0, 454), (0, 1089)]
    )
    vtkobj.write(function_tmpdir / "freyberg_head")

    assert count_lines_in_file(filetocheck) == 34
    filetocheck.unlink()

    # with point scalars

    vtkobj = Vtk(m, pvd=True, xml=True, point_scalars=True)
    vtkobj.add_heads(
        heads, kstpkper=[(0, 0), (0, 199), (0, 354), (0, 454), (0, 1089)]
    )
    vtkobj.write(function_tmpdir / "freyberg_head")

    assert count_lines_in_file(filetocheck) == 34
    filetocheck.unlink()

    # with smoothing

    vtkobj = Vtk(m, pvd=True, xml=True, smooth=True)
    vtkobj.add_heads(
        heads, kstpkper=[(0, 0), (0, 199), (0, 354), (0, 454), (0, 1089)]
    )
    vtkobj.write(function_tmpdir / "freyberg_head")

    assert count_lines_in_file(filetocheck) == 34


@requires_pkg("vtk")
@pytest.mark.slow
def test_vtk_cbc(function_tmpdir, example_data_path):
    # test mf 2005 freyberg

    mpth = example_data_path / "freyberg_multilayer_transient"
    namfile = "freyberg.nam"
    cbcfile = os.path.join(mpth, "freyberg.cbc")
    cbc = CellBudgetFile(cbcfile)
    m = Modflow.load(
        namfile, model_ws=mpth, verbose=False, load_only=["dis", "bas6"]
    )

    # export and check with point scalar
    vtkobj = Vtk(m, binary=False, xml=True, pvd=True, point_scalars=True)
    vtkobj.add_cell_budget(cbc, kstpkper=[(0, 0), (0, 1), (0, 2)])
    vtkobj.write(function_tmpdir / "freyberg_CBC")

    assert (
        count_lines_in_file(function_tmpdir / "freyberg_CBC_000000.vtu")
        == 39243
    )

    # with point scalars and binary
    vtkobj = Vtk(m, xml=True, pvd=True, point_scalars=True)
    vtkobj.add_cell_budget(cbc, kstpkper=[(0, 0), (0, 1), (0, 2)])
    vtkobj.write(function_tmpdir / "freyberg_CBC")
    assert (
        count_lines_in_file(function_tmpdir / "freyberg_CBC_000000.vtu") == 28
    )


@requires_pkg("vtk")
@pytest.mark.slow
def test_vtk_vector(function_tmpdir, example_data_path):
    # test mf 2005 freyberg
    mpth = example_data_path / "freyberg_multilayer_transient"
    namfile = "freyberg.nam"
    cbcfile = os.path.join(mpth, "freyberg.cbc")
    hdsfile = os.path.join(mpth, "freyberg.hds")
    cbc = CellBudgetFile(cbcfile)
    keys = ["FLOW RIGHT FACE", "FLOW FRONT FACE", "FLOW LOWER FACE"]
    vectors = [cbc.get_data(text=t)[0] for t in keys]
    hds = HeadFile(hdsfile)
    head = hds.get_data()
    m = Modflow.load(
        namfile, model_ws=mpth, verbose=False, load_only=["dis", "bas6", "upw"]
    )
    q = pp.get_specific_discharge(vectors, m, head)

    filenametocheck = function_tmpdir / "discharge.vtu"

    # export and check with point scalar
    vtkobj = Vtk(m, xml=True, binary=False, point_scalars=True)
    vtkobj.add_vector(q, "discharge")
    vtkobj.write(filenametocheck)

    assert count_lines_in_file(filenametocheck) == 36045

    # with point scalars and binary
    vtkobj = Vtk(m, point_scalars=True)
    vtkobj.add_vector(q, "discharge")
    vtkobj.write(filenametocheck)

    assert is_binary_file(filenametocheck)

    # test at cell centers
    q = pp.get_specific_discharge(vectors, m, head)

    filenametocheck = function_tmpdir / "discharge_verts.vtu"
    vtkobj = Vtk(m, xml=True, binary=False)
    vtkobj.add_vector(q, "discharge")
    vtkobj.write(filenametocheck)

    assert count_lines_in_file(filenametocheck) == 27645

    # with values directly given at vertices and binary
    vtkobj = Vtk(m, xml=True, binary=True)
    vtkobj.add_vector(q, "discharge")
    vtkobj.write(filenametocheck)

    assert count_lines_in_file(filenametocheck) == 25


@requires_pkg("vtk")
def test_vtk_unstructured(function_tmpdir, unstructured_grid):
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkIOLegacy import vtkUnstructuredGridReader

    grid = unstructured_grid

    outfile = function_tmpdir / "disu_grid.vtu"
    vtkobj = Vtk(
        modelgrid=grid, vertical_exageration=2, binary=True, smooth=False
    )
    vtkobj.add_array(grid.top, "top")
    vtkobj.add_array(grid.botm, "botm")
    vtkobj.write(outfile)

    assert is_binary_file(outfile)

    reader = vtkUnstructuredGridReader()
    reader.SetFileName(str(outfile))
    reader.ReadAllFieldsOn()
    reader.Update()

    data = reader.GetOutput()

    top2 = vtk_to_numpy(data.GetCellData().GetArray("top"))

    assert np.allclose(
        np.ravel(grid.top), top2
    ), "Field data not properly written"


@requires_pkg("vtk", "pyvista")
def test_vtk_to_pyvista(function_tmpdir):
    from pprint import pformat

    from autotest.test_mp7_cases import Mp7Cases

    case_mf6 = Mp7Cases.mp7_mf6(function_tmpdir)
    case_mf6.write_input()
    success, buff = case_mf6.run_model()
    assert success, f"MP7 model ({case_mf6.name}) failed: {pformat(buff)}"

    gwf = case_mf6.flowmodel
    plf = PathlineFile(Path(case_mf6.model_ws) / f"{case_mf6.name}.mppth")
    pls = plf.get_alldata()

    vtk = Vtk(model=gwf, binary=True, smooth=False)
    assert not any(vtk.to_pyvista())

    vtk.add_model(gwf)
    grid = vtk.to_pyvista()
    assert grid.n_cells == gwf.modelgrid.nnodes

    vtk.add_pathline_points(pls)
    grid, pathlines = vtk.to_pyvista()
    n_pts = sum([pl.shape[0] for pl in pls])
    assert pathlines.n_points == n_pts
    assert pathlines.n_cells == n_pts + len(pls)
    assert "particleid" in pathlines.point_data
    assert "time" in pathlines.point_data
    assert "k" in pathlines.point_data

    # uncomment to debug
    # grid.plot()
    # pathlines.plot()


@pytest.mark.mf6
@requires_pkg("vtk")
def test_vtk_vertex(function_tmpdir, example_data_path):
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkIOLegacy import vtkUnstructuredGridReader

    # disv test
    workspace = example_data_path / "mf6" / "test003_gwfs_disv"
    # outfile = os.path.join("vtk_transient_test", "vtk_pacakages")
    sim = MFSimulation.load(sim_ws=workspace)
    gwf = sim.get_model("gwf_1")

    outfile = function_tmpdir / "disv.vtk"
    vtkobj = Vtk(model=gwf, binary=True, smooth=False)
    vtkobj.add_model(gwf)
    vtkobj.write(outfile)

    outfile = outfile.parent / f"{outfile.stem}_000000.vtk"
    assert outfile.exists(), "Vertex VTK File was not written"

    reader = vtkUnstructuredGridReader()
    reader.SetFileName(str(outfile))
    reader.ReadAllFieldsOn()
    reader.Update()

    data = reader.GetOutput()

    hk2 = vtk_to_numpy(data.GetCellData().GetArray("k"))
    hk = gwf.npf.k.array
    hk[gwf.modelgrid.idomain == 0] = np.nan

    assert np.allclose(
        np.ravel(hk), hk2, equal_nan=True
    ), "Field data not properly written"


@requires_exe("mf2005")
@requires_pkg("vtk")
def test_vtk_pathline(function_tmpdir, example_data_path):
    from vtkmodules.vtkIOLegacy import vtkUnstructuredGridReader

    # pathline test for vtk
    ml = Modflow.load(
        "freyberg.nam",
        model_ws=example_data_path / "freyberg",
        exe_name="mf2005",
    )

    ml.change_model_ws(new_pth=function_tmpdir)
    ml.write_input()
    ml.run_model()

    mpp = Modpath6(
        "freybergmpp",
        modflowmodel=ml,
        model_ws=function_tmpdir,
        exe_name="mp6",
    )
    mpbas = Modpath6Bas(
        mpp,
        hnoflo=ml.bas6.hnoflo,
        hdry=ml.lpf.hdry,
        ibound=ml.bas6.ibound.array,
        prsity=0.2,
        prsityCB=0.2,
    )
    sim = mpp.create_mpsim(
        trackdir="backward", simtype="pathline", packages="WEL"
    )
    mpp.write_input()
    mpp.run_model()

    pthfile = os.path.join(function_tmpdir, mpp.sim.pathline_file)
    pthobj = PathlineFile(pthfile)
    travel_time_max = 200.0 * 365.25 * 24.0 * 60.0 * 60.0
    plines = pthobj.get_alldata(totim=travel_time_max, ge=False)

    outfile = function_tmpdir / "pathline.vtk"

    vtkobj = Vtk(model=ml, binary=True, vertical_exageration=50, smooth=False)
    vtkobj.add_model(ml)
    vtkobj.add_pathline_points(plines)
    vtkobj.write(outfile)

    outfile = outfile.parent / f"{outfile.stem}_pathline.vtk"
    assert outfile.exists(), "Pathline VTK file not properly written"

    reader = vtkUnstructuredGridReader()
    reader.SetFileName(str(outfile))
    reader.ReadAllFieldsOn()
    reader.Update()

    data = reader.GetOutput()

    from vtkmodules.util import numpy_support

    totim = numpy_support.vtk_to_numpy(data.GetPointData().GetArray("time"))
    pid = numpy_support.vtk_to_numpy(
        data.GetPointData().GetArray("particleid")
    )

    maxtime = 0
    for p in plines:
        if np.max(p["time"]) > maxtime:
            maxtime = np.max(p["time"])

    assert len(totim) == 12054, "Array size is incorrect"
    assert np.abs(np.max(totim) - maxtime) < 100, "time values are incorrect"
    assert len(np.unique(pid)) == len(
        plines
    ), "number of particles are incorrect for modpath VTK"


def grid2disvgrid(nrow, ncol):
    """Simple function to create disv verts and iverts for a regular grid of size nrow, ncol"""

    def lower_left_point(i, j, ncol):
        return i * (ncol + 1) + j

    mg = np.meshgrid(
        np.linspace(0, ncol, ncol + 1), np.linspace(0, nrow, nrow + 1)
    )
    verts = np.vstack((mg[0].flatten(), mg[1].flatten())).transpose()

    # in the creation of iverts here, we intentionally do not close the cell polygon
    iverts = []
    for i in range(nrow):
        for j in range(ncol):
            iv_cell = []
            iv_cell.append(lower_left_point(i, j, ncol))
            iv_cell.append(lower_left_point(i, j + 1, ncol))
            iv_cell.append(lower_left_point(i + 1, j + 1, ncol))
            iv_cell.append(lower_left_point(i + 1, j, ncol))
            iverts.append(iv_cell)
    return verts, iverts


def load_verts(fname):
    verts = np.genfromtxt(
        fname, dtype=[int, float, float], names=["iv", "x", "y"]
    )
    verts["iv"] -= 1  # zero based
    return verts


def load_iverts(fname, closed=False):
    iverts = []
    xc = []
    yc = []
    with open(fname) as f:
        for line in f:
            ll = line.strip().split()
            if not closed:
                iverts.append([int(i) - 1 for i in ll[4:-1]])
            else:
                iverts.append([int(i) - 1 for i in ll[4:]])
            xc.append(float(ll[1]))
            yc.append(float(ll[2]))
    return iverts, np.array(xc), np.array(yc)


@pytest.mark.mf6
@requires_pkg("vtk")
def test_vtk_add_model_without_packages_names(function_tmpdir):
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkIOLegacy import vtkUnstructuredGridReader

    from flopy.export.vtk import Vtk

    name = "mymodel"
    sim = MFSimulation(sim_name=name, sim_ws=function_tmpdir, exe_name="mf6")
    tdis = ModflowTdis(sim)
    ims = ModflowIms(sim)
    gwf = ModflowGwf(sim, modelname=name, save_flows=True)
    dis = ModflowGwfdis(gwf, nrow=3, ncol=3)
    ic = ModflowGwfic(gwf)
    npf = ModflowGwfnpf(gwf, save_specific_discharge=True)
    chd = ModflowGwfchd(
        gwf, stress_period_data=[[(0, 0, 0), 1.0], [(0, 2, 2), 0.0]]
    )

    # Export model without specifying packages_names parameter

    # create the vtk output
    gwf = sim.get_model()
    vtkobj = Vtk(gwf, binary=False)
    vtkobj.add_model(gwf)
    vtkobj.write(function_tmpdir / "gwf.vtk")

    # load the output using the vtk standard library
    gridreader = vtkUnstructuredGridReader()
    gridreader.SetFileName(str(function_tmpdir / "gwf_000000.vtk"))
    gridreader.Update()
    grid = gridreader.GetOutput()

    # get the points
    vtk_points = grid.GetPoints()
    vtk_points = vtk_points.GetData()
    vtk_points = vtk_to_numpy(vtk_points)

    # get cell locations (ia format of point to cell relationship)
    cell_locations = vtk_to_numpy(grid.GetCellLocationsArray())
    cell_locations_answer = np.array([0, 8, 16, 24, 32, 40, 48, 56, 64])
    print(f"Found cell locations {cell_locations} in vtk file.")
    print(f"Expecting cell locations {cell_locations_answer}")
    errmsg = "vtk cell locations do not match expected result."
    assert np.allclose(cell_locations, cell_locations_answer), errmsg

    cell_types = vtk_to_numpy(grid.GetCellTypesArray())
    cell_types_answer = np.array(9 * [42])
    print(f"Found cell types {cell_types} in vtk file.")
    print(f"Expecting cell types {cell_types_answer}")
    errmsg = "vtk cell types do not match expected result."
    assert np.allclose(cell_types, cell_types_answer), errmsg


@pytest.mark.mf6
@requires_pkg("vtk", "shapely")
def test_vtk_export_disv1_model(function_tmpdir):
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkIOLegacy import vtkUnstructuredGridReader

    from flopy.export.vtk import Vtk

    name = "mymodel"
    sim = MFSimulation(sim_name=name, sim_ws=function_tmpdir, exe_name="mf6")
    tdis = ModflowTdis(sim)
    ims = ModflowIms(sim)
    gwf = ModflowGwf(sim, modelname=name, save_flows=True)

    nlay, nrow, ncol = 1, 3, 3
    from flopy.discretization import StructuredGrid

    mg = StructuredGrid(
        delc=np.array(nrow * [1]),
        delr=np.array(ncol * [1]),
        top=np.zeros((nrow, ncol)),
        botm=np.zeros((nlay, nrow, ncol)) - 1,
        idomain=np.ones((nlay, nrow, ncol)),
    )

    from flopy.utils.cvfdutil import gridlist_to_disv_gridprops

    gridprops = gridlist_to_disv_gridprops([mg])
    gridprops["top"] = 0
    gridprops["botm"] = np.zeros((nlay, nrow * ncol), dtype=float) - 1
    gridprops["nlay"] = nlay

    disv = ModflowGwfdisv(gwf, **gridprops)
    ic = ModflowGwfic(gwf, strt=10)
    npf = ModflowGwfnpf(gwf)

    # Export model without specifying packages_names parameter
    # create the vtk output
    gwf = sim.get_model()
    vtkobj = Vtk(gwf, binary=False)
    vtkobj.add_model(gwf)
    f = function_tmpdir / "gwf.vtk"
    vtkobj.write(f)

    # load the output using the vtk standard library
    gridreader = vtkUnstructuredGridReader()
    gridreader.SetFileName(str(f))
    gridreader.Update()
    grid = gridreader.GetOutput()

    # get the points
    vtk_points = grid.GetPoints()
    vtk_points = vtk_points.GetData()
    vtk_points = vtk_to_numpy(vtk_points)
    # print(vtk_points)

    # get cell locations (ia format of point to cell relationship)
    cell_locations = vtk_to_numpy(grid.GetCellLocationsArray())
    cell_locations_answer = np.array([0, 8, 16, 24, 32, 40, 48, 56, 64])
    print(f"Found cell locations {cell_locations} in vtk file.")
    print(f"Expecting cell locations {cell_locations_answer}")
    errmsg = "vtk cell locations do not match expected result."
    assert np.allclose(cell_locations, cell_locations_answer), errmsg

    cell_types = vtk_to_numpy(grid.GetCellTypesArray())
    cell_types_answer = np.array(9 * [42])
    print(f"Found cell types {cell_types} in vtk file.")
    print(f"Expecting cell types {cell_types_answer}")
    errmsg = "vtk cell types do not match expected result."
    assert np.allclose(cell_types, cell_types_answer), errmsg


@pytest.mark.mf6
@requires_pkg("vtk", "shapely")
def test_vtk_export_disv2_model(function_tmpdir):
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkIOLegacy import vtkUnstructuredGridReader

    from flopy.export.vtk import Vtk

    # in this case, test for iverts that do not explicitly close the cell polygons
    name = "mymodel"
    sim = MFSimulation(sim_name=name, sim_ws=function_tmpdir, exe_name="mf6")
    tdis = ModflowTdis(sim)
    ims = ModflowIms(sim)
    gwf = ModflowGwf(sim, modelname=name, save_flows=True)

    nlay, nrow, ncol = 1, 3, 3
    verts, iverts = grid2disvgrid(3, 3)
    from flopy.utils.cvfdutil import get_disv_gridprops

    gridprops = get_disv_gridprops(verts, iverts)

    gridprops["top"] = 0
    gridprops["botm"] = np.zeros((nlay, nrow * ncol), dtype=float) - 1
    gridprops["nlay"] = nlay

    disv = ModflowGwfdisv(gwf, **gridprops)
    ic = ModflowGwfic(gwf, strt=10)
    npf = ModflowGwfnpf(gwf)

    # Export model without specifying packages_names parameter
    # create the vtk output
    gwf = sim.get_model()
    vtkobj = Vtk(gwf, binary=False)
    vtkobj.add_model(gwf)
    f = function_tmpdir / "gwf.vtk"
    vtkobj.write(f)

    # load the output using the vtk standard library
    gridreader = vtkUnstructuredGridReader()
    gridreader.SetFileName(str(f))
    gridreader.Update()
    grid = gridreader.GetOutput()

    # get the points
    vtk_points = grid.GetPoints()
    vtk_points = vtk_points.GetData()
    vtk_points = vtk_to_numpy(vtk_points)
    # print(vtk_points)

    # get cell locations (ia format of point to cell relationship)
    cell_locations = vtk_to_numpy(grid.GetCellLocationsArray())
    cell_locations_answer = np.array([0, 8, 16, 24, 32, 40, 48, 56, 64])
    print(f"Found cell locations {cell_locations} in vtk file.")
    print(f"Expecting cell locations {cell_locations_answer}")
    errmsg = "vtk cell locations do not match expected result."
    assert np.allclose(cell_locations, cell_locations_answer), errmsg

    cell_types = vtk_to_numpy(grid.GetCellTypesArray())
    cell_types_answer = np.array(9 * [42])
    print(f"Found cell types {cell_types} in vtk file.")
    print(f"Expecting cell types {cell_types_answer}")
    errmsg = "vtk cell types do not match expected result."
    assert np.allclose(cell_types, cell_types_answer), errmsg


@requires_pkg("vtk")
def test_vtk_export_disu1_grid(function_tmpdir, example_data_path):
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkIOLegacy import vtkUnstructuredGridReader

    from flopy.export.vtk import Vtk

    u_data_ws = example_data_path / "unstructured"

    # test exporting open cell vertices
    # load vertices
    verts = load_verts(u_data_ws / "ugrid_verts.dat")

    # load the index list into iverts, xc, and yc
    iverts, xc, yc = load_iverts(u_data_ws / "ugrid_iverts.dat")

    # create a 3 layer model grid
    ncpl = np.array(3 * [len(iverts)])
    nnodes = np.sum(ncpl)

    top = np.ones(nnodes)
    botm = np.ones(nnodes)

    # set top and botm elevations
    i0 = 0
    i1 = ncpl[0]
    elevs = [100, 0, -100, -200]
    for ix, cpl in enumerate(ncpl):
        top[i0:i1] *= elevs[ix]
        botm[i0:i1] *= elevs[ix + 1]
        i0 += cpl
        i1 += cpl

    # create the modelgrid
    modelgrid = UnstructuredGrid(
        vertices=verts,
        iverts=iverts,
        xcenters=xc,
        ycenters=yc,
        top=top,
        botm=botm,
        ncpl=ncpl,
    )

    outfile = function_tmpdir / "disu_grid.vtu"
    vtkobj = Vtk(
        modelgrid=modelgrid,
        vertical_exageration=2,
        binary=True,
        smooth=False,
    )
    vtkobj.add_array(modelgrid.top, "top")
    vtkobj.add_array(modelgrid.botm, "botm")
    vtkobj.write(outfile)

    gridreader = vtkUnstructuredGridReader()
    gridreader.SetFileName(str(outfile))
    gridreader.Update()
    grid = gridreader.GetOutput()

    # get the points
    vtk_points = grid.GetPoints()
    vtk_points = vtk_points.GetData()
    vtk_points = vtk_to_numpy(vtk_points)

    # get cell locations (ia format of point to cell relationship)
    cell_locations = vtk_to_numpy(grid.GetCellLocationsArray())[0:9]
    cell_locations_answer = np.array([0, 8, 16, 24, 32, 40, 48, 56, 64])
    print(f"Found cell locations {cell_locations} in vtk file.")
    print(f"Expecting cell locations {cell_locations_answer}")
    errmsg = "vtk cell locations do not match expected result."
    assert np.allclose(cell_locations, cell_locations_answer), errmsg

    cell_types = vtk_to_numpy(grid.GetCellTypesArray())
    cell_types_answer = np.array(654 * [42])
    print(f"Found cell types {cell_types[0:9]} in vtk file.")
    print(f"Expecting cell types {cell_types_answer[0:9]}")
    errmsg = "vtk cell types do not match expected result."
    assert np.allclose(cell_types, cell_types_answer), errmsg


@requires_pkg("vtk")
def test_vtk_export_disu2_grid(function_tmpdir, example_data_path):
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkIOLegacy import vtkUnstructuredGridReader

    from flopy.export.vtk import Vtk

    u_data_ws = example_data_path / "unstructured"

    # test exporting closed cell vertices
    # load vertices
    verts = load_verts(u_data_ws / "ugrid_verts.dat")

    # load the index list into iverts, xc, and yc
    iverts, xc, yc = load_iverts(u_data_ws / "ugrid_iverts.dat", closed=True)

    # create a 3 layer model grid
    ncpl = np.array(3 * [len(iverts)])
    nnodes = np.sum(ncpl)

    top = np.ones(nnodes)
    botm = np.ones(nnodes)

    # set top and botm elevations
    i0 = 0
    i1 = ncpl[0]
    elevs = [100, 0, -100, -200]
    for ix, cpl in enumerate(ncpl):
        top[i0:i1] *= elevs[ix]
        botm[i0:i1] *= elevs[ix + 1]
        i0 += cpl
        i1 += cpl

    # create the modelgrid
    modelgrid = UnstructuredGrid(
        vertices=verts,
        iverts=iverts,
        xcenters=xc,
        ycenters=yc,
        top=top,
        botm=botm,
        ncpl=ncpl,
    )

    outfile = function_tmpdir / "disu_grid.vtu"
    vtkobj = Vtk(
        modelgrid=modelgrid,
        vertical_exageration=2,
        binary=True,
        smooth=False,
    )
    vtkobj.add_array(modelgrid.top, "top")
    vtkobj.add_array(modelgrid.botm, "botm")
    vtkobj.write(outfile)

    gridreader = vtkUnstructuredGridReader()
    gridreader.SetFileName(str(outfile))
    gridreader.Update()
    grid = gridreader.GetOutput()

    # get the points
    vtk_points = grid.GetPoints()
    vtk_points = vtk_points.GetData()
    vtk_points = vtk_to_numpy(vtk_points)

    # get cell locations (ia format of point to cell relationship)
    cell_locations = vtk_to_numpy(grid.GetCellLocationsArray())[0:9]
    cell_locations_answer = np.array([0, 8, 16, 24, 32, 40, 48, 56, 64])
    print(f"Found cell locations {cell_locations} in vtk file.")
    print(f"Expecting cell locations {cell_locations_answer}")
    errmsg = "vtk cell locations do not match expected result."
    assert np.allclose(cell_locations, cell_locations_answer), errmsg

    cell_types = vtk_to_numpy(grid.GetCellTypesArray())
    cell_types_answer = np.array(654 * [42])
    print(f"Found cell types {cell_types[0:9]} in vtk file.")
    print(f"Expecting cell types {cell_types_answer[0:9]}")
    errmsg = "vtk cell types do not match expected result."
    assert np.allclose(cell_types, cell_types_answer), errmsg


@pytest.mark.mf6
@requires_exe("mf6", "gridgen")
@requires_pkg("vtk", "shapefile", "shapely")
def test_vtk_export_disu_model(function_tmpdir):
    from vtkmodules.util.numpy_support import vtk_to_numpy

    from flopy.export.vtk import Vtk

    name = "vtk_export_disu"
    sim = disu_sim(name, function_tmpdir)
    gwf = sim.get_model(name)

    # export grid
    vtk = import_optional_dependency("vtk")

    vtkobj = Vtk(gwf, binary=False)
    vtkobj.add_model(gwf)
    f = function_tmpdir / "gwf.vtk"
    vtkobj.write(f)

    # load the output using the vtk standard library
    gridreader = vtk.vtkUnstructuredGridReader()
    gridreader.SetFileName(str(f))
    gridreader.Update()
    grid = gridreader.GetOutput()

    # get the points
    vtk_points = grid.GetPoints()
    vtk_points = vtk_points.GetData()
    vtk_points = vtk_to_numpy(vtk_points)
    # print(vtk_points)

    # get cell locations (ia format of point to cell relationship)
    cell_locations = vtk_to_numpy(grid.GetCellLocationsArray())[0:9]
    cell_locations_answer = np.array([0, 8, 16, 24, 32, 40, 48, 56, 64])
    print(f"First nine cell locations {cell_locations} in vtk file.")
    print(f"Expecting first nine cell locations {cell_locations_answer}")
    errmsg = "vtk cell locations do not match expected result."
    assert np.allclose(cell_locations, cell_locations_answer), errmsg

    cell_types = vtk_to_numpy(grid.GetCellTypesArray())
    cell_types_answer = np.array(1770 * [42])
    print(f"First nine cell types {cell_types[0:9]} in vtk file.")
    print(f"Expecting fist nine cell types {cell_types_answer[0:9]}")
    errmsg = "vtk cell types do not match expected result."
    assert np.allclose(cell_types, cell_types_answer), errmsg

    # now check that the data is consistent with that in npf and ic

    k_vtk = vtk_to_numpy(grid.GetCellData().GetArray("k"))
    if not np.allclose(gwf.npf.k.array, k_vtk):
        raise AssertionError("'k' array not written in proper node order")

    strt_vtk = vtk_to_numpy(grid.GetCellData().GetArray("strt"))
    if not np.allclose(gwf.ic.strt.array, strt_vtk):
        raise AssertionError("'strt' array not written in proper node order")
