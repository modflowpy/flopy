import math
import os
import shutil
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pytest
from autotest.conftest import (
    excludes_platform,
    get_example_data_path,
    has_pkg,
    requires_exe,
    requires_pkg,
    requires_spatial_reference,
)
from flaky import flaky

import flopy
from flopy.discretization import StructuredGrid, UnstructuredGrid
from flopy.export import NetCdf
from flopy.export.shapefile_utils import (
    EpsgReference,
    recarray2shp,
    shp2recarray,
)
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
from flopy.utils.geometry import Polygon


def namfiles() -> List[Path]:
    mf2005_path = get_example_data_path() / "mf2005_test"
    return list(mf2005_path.rglob("*.nam"))


@requires_pkg("shapefile")
def test_output_helper_shapefile_export(tmpdir, example_data_path):

    ml = Modflow.load(
        "freyberg.nam",
        model_ws=str(example_data_path / "freyberg_multilayer_transient"),
    )
    head = HeadFile(os.path.join(ml.model_ws, "freyberg.hds"))
    cbc = CellBudgetFile(os.path.join(ml.model_ws, "freyberg.cbc"))

    flopy.export.utils.output_helper(
        os.path.join(tmpdir, "test.shp"),
        ml,
        {"HDS": head, "cbc": cbc},
        mflay=1,
        kper=10,
    )


@requires_pkg("pandas", "shapefile")
@pytest.mark.slow
def test_freyberg_export(tmpdir, example_data_path):

    # steady state
    name = "freyberg"
    namfile = f"{name}.nam"
    ws = str(example_data_path / name)
    m = flopy.modflow.Modflow.load(
        namfile, model_ws=ws, check=False, verbose=False
    )

    # test export at model, package and object levels
    m.export(f"{tmpdir}/model.shp")
    m.wel.export(f"{tmpdir}/wel.shp")
    m.lpf.hk.export(f"{tmpdir}/hk.shp")
    m.riv.stress_period_data.export(f"{tmpdir}/riv_spd.shp")

    # transient
    # (doesn't work at model level because the total size of
    #  the attribute fields exceeds the shapefile limit)
    ws = str(example_data_path / "freyberg_multilayer_transient")
    m = flopy.modflow.Modflow.load(
        namfile,
        model_ws=ws,
        verbose=False,
        load_only=["DIS", "BAS6", "NWT", "OC", "RCH", "WEL", "DRN", "UPW"],
    )
    # test export without instantiating an sr
    shape = tmpdir / f"{name}_drn_sparse.shp"
    m.drn.stress_period_data.export(str(shape), sparse=True)
    for suffix in [".dbf", ".shp", ".shx"]:
        part = shape.with_suffix(suffix)
        assert part.exists()
        part.unlink()
    assert not shape.with_suffix(".prj").exists()

    m.modelgrid = StructuredGrid(
        delc=m.dis.delc.array, delr=m.dis.delr.array, epsg=3070
    )
    # test export with an sr, regardless of whether or not wkt was found
    m.drn.stress_period_data.export(str(shape), sparse=True)
    for suffix in [".dbf", ".prj", ".shp", ".shx"]:
        part = shape.with_suffix(suffix)
        assert part.exists()
        part.unlink()

    m.modelgrid = StructuredGrid(
        delc=m.dis.delc.array, delr=m.dis.delr.array, epsg=3070
    )
    # verify that attributes have same sr as parent
    assert m.drn.stress_period_data.mg.epsg == m.modelgrid.epsg
    assert m.drn.stress_period_data.mg.proj4 == m.modelgrid.proj4
    assert m.drn.stress_period_data.mg.xoffset == m.modelgrid.xoffset
    assert m.drn.stress_period_data.mg.yoffset == m.modelgrid.yoffset
    assert m.drn.stress_period_data.mg.angrot == m.modelgrid.angrot

    # get wkt text was fetched from spatialreference.org
    wkt = flopy.export.shapefile_utils.CRS.get_spatialreference(
        m.modelgrid.epsg
    )

    # if wkt text was fetched from spatialreference.org
    if wkt is not None:
        # test default package export
        shape = tmpdir / f"{name}_dis.shp"
        m.dis.export(str(shape))
        for suffix in [".dbf", ".prj", ".shp", ".shx"]:
            part = shape.with_suffix(suffix)
            assert part.exists()
            if suffix == ".prj":
                assert part.read_text() == wkt
            part.unlink()

        # test default package export to higher level dir ?

        # test sparse package export
        shape = tmpdir / f"{name}_drn_sparse.shp"
        m.drn.stress_period_data.export(str(shape), sparse=True)
        for suffix in [".dbf", ".prj", ".shp", ".shx"]:
            part = shape.with_suffix(suffix)
            assert part.exists()
            if suffix == ".prj":
                assert part.read_text() == wkt


@requires_pkg("netCDF4", "pyproj")
def test_export_output(tmpdir, example_data_path):

    ml = Modflow.load(
        "freyberg.nam", model_ws=str(example_data_path / "freyberg")
    )
    hds_pth = os.path.join(ml.model_ws, "freyberg.githds")
    hds = flopy.utils.HeadFile(hds_pth)

    out_pth = os.path.join(tmpdir, "freyberg.out.nc")
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


@requires_pkg("shapefile")
def test_write_gridlines_shapefile(tmpdir):
    import shapefile

    from flopy.discretization import StructuredGrid
    from flopy.export.shapefile_utils import write_gridlines_shapefile

    sg = StructuredGrid(
        delr=np.ones(10) * 1.1,
        # cell spacing along model rows
        delc=np.ones(10) * 1.1,
        # cell spacing along model columns
        epsg=26715,
    )
    outshp = tmpdir / "gridlines.shp"
    write_gridlines_shapefile(outshp, sg)

    for suffix in [".dbf", ".prj", ".shp", ".shx"]:
        assert outshp.with_suffix(suffix).exists()

    with shapefile.Reader(str(outshp)) as sf:
        assert sf.shapeType == shapefile.POLYLINE
        assert len(sf) == 22


@flaky
@requires_pkg("shapefile", "shapely")
def test_write_grid_shapefile(tmpdir):
    from shapefile import Reader

    from flopy.discretization import StructuredGrid
    from flopy.export.shapefile_utils import write_grid_shapefile

    sg = StructuredGrid(
        delr=np.ones(10) * 1.1,
        # cell spacing along model rows
        delc=np.ones(10) * 1.1,
        # cell spacing along model columns
        epsg=26715,
    )
    outshp = tmpdir / "junk.shp"
    write_grid_shapefile(outshp, sg, array_dict={})

    for suffix in [".dbf", ".prj", ".shp", ".shx"]:
        assert outshp.with_suffix(suffix).exists()

    # test that vertices aren't getting altered by writing shapefile
    # check that pyshp reads integers
    # this only check that row/column were recorded as "N"
    # not how they will be cast by python or numpy
    sfobj = Reader(str(outshp))
    for f in sfobj.fields:
        if f[0] == "row" or f[0] == "column":
            assert f[1] == "N"
    recs = list(sfobj.records())
    for r in recs[0]:
        assert isinstance(r, int)
    sfobj.close()

    # check that row and column appear as integers in recarray
    ra = shp2recarray(outshp)
    assert np.issubdtype(ra.dtype["row"], np.integer)
    assert np.issubdtype(ra.dtype["column"], np.integer)

    try:  # check that fiona reads integers
        import fiona

        with fiona.open(outshp) as src:
            meta = src.meta
            assert "int" in meta["schema"]["properties"]["row"]
            assert "int" in meta["schema"]["properties"]["column"]
    except ImportError:
        pass


@requires_pkg("shapefile")
def test_export_shapefile_polygon_closed(tmpdir):
    from shapefile import Reader

    xll, yll = 468970, 3478635
    xur, yur = 681010, 3716462

    spacing = 2000

    ncol = int((xur - xll) / spacing)
    nrow = int((yur - yll) / spacing)
    print(nrow, ncol)

    m = flopy.modflow.Modflow(
        "test.nam", proj4_str="EPSG:32614", xll=xll, yll=yll
    )

    flopy.modflow.ModflowDis(
        m, delr=spacing, delc=spacing, nrow=nrow, ncol=ncol
    )

    shp_file = os.path.join(tmpdir, "test_polygon.shp")
    m.dis.export(shp_file)

    shp = Reader(shp_file)
    for shape in shp.iterShapes():
        if len(shape.points) != 5:
            raise AssertionError("Shapefile polygon is not closed!")

    shp.close()


@excludes_platform("Windows")
@requires_pkg("rasterio", "shapefile", "scipy")
def test_export_array(tmpdir, example_data_path):
    import rasterio
    from scipy.ndimage import rotate

    namfile = "freyberg.nam"
    model_ws = example_data_path / "freyberg"
    m = flopy.modflow.Modflow.load(
        namfile,
        model_ws=str(model_ws),
        verbose=False,
        load_only=["DIS", "BAS6"],
    )
    m.modelgrid.set_coord_info(angrot=45)
    nodata = -9999
    export_array(
        m.modelgrid,
        os.path.join(tmpdir, "fb.asc"),
        m.dis.top.array,
        nodata=nodata,
    )
    arr = np.loadtxt(os.path.join(tmpdir, "fb.asc"), skiprows=6)

    m.modelgrid.write_shapefile(os.path.join(tmpdir, "grid.shp"))

    # check bounds
    with open(os.path.join(tmpdir, "fb.asc")) as src:
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
        os.path.join(tmpdir, "fb.tif"),
        m.dis.top.array,
        nodata=nodata,
    )
    with rasterio.open(os.path.join(tmpdir, "fb.tif")) as src:
        arr = src.read(1)
        assert src.shape == (m.nrow, m.ncol)
        # TODO: these tests currently fail -- fix is in progress
        # assert np.abs(src.bounds[0] - m.modelgrid.extent[0]) < 1e-6
        # assert np.abs(src.bounds[1] - m.modelgrid.extent[1]) < 1e-6
        pass


@requires_pkg("netCDF4", "pyproj")
def test_netcdf_classmethods(tmpdir, example_data_path):
    namfile = "freyberg.nam"
    name = namfile.replace(".nam", "")
    model_ws = example_data_path / "freyberg_multilayer_transient"
    ml = flopy.modflow.Modflow.load(
        namfile,
        model_ws=str(model_ws),
        check=False,
        verbose=True,
        load_only=[],
    )

    f = ml.export(os.path.join(tmpdir, "freyberg.nc"))
    v1_set = set(f.nc.variables.keys())
    fnc = os.path.join(tmpdir, "freyberg.new.nc")
    new_f = flopy.export.NetCdf.zeros_like(f, output_filename=fnc)
    v2_set = set(new_f.nc.variables.keys())
    diff = v1_set.symmetric_difference(v2_set)
    assert len(diff) == 0, str(diff)

    # close the netcdf file
    f.nc.close()
    new_f.nc.close()


def test_wkt_parse(example_shapefiles):
    """Test parsing of Coordinate Reference System parameters
    from well-known-text in .prj files."""

    from flopy.export.shapefile_utils import CRS

    geocs_params = [
        "wktstr",
        "geogcs",
        "datum",
        "spheroid_name",
        "semi_major_axis",
        "inverse_flattening",
        "primem",
        "gcs_unit",
    ]

    for prj in example_shapefiles:
        with open(prj) as src:
            wkttxt = src.read()
            wkttxt = wkttxt.replace("'", '"')
        if len(wkttxt) > 0 and "projcs" in wkttxt.lower():
            crsobj = CRS(esri_wkt=wkttxt)
            assert isinstance(crsobj.crs, dict)
            for k in geocs_params:
                assert crsobj.__dict__[k] is not None
            projcs_params = [
                k for k in crsobj.__dict__ if k not in geocs_params
            ]
            if crsobj.projcs is not None:
                for k in projcs_params:
                    if k in wkttxt.lower():
                        assert crsobj.__dict__[k] is not None


@requires_pkg("shapefile")
def test_shapefile_ibound(tmpdir, example_data_path):
    from shapefile import Reader

    shape_name = os.path.join(tmpdir, "test.shp")
    namfile = "freyberg.nam"
    model_ws = example_data_path / "freyberg_multilayer_transient"
    ml = flopy.modflow.Modflow.load(
        namfile,
        model_ws=str(model_ws),
        check=False,
        verbose=True,
        load_only=["bas6"],
    )
    ml.export(shape_name)
    shape = Reader(shape_name)
    field_names = [item[0] for item in shape.fields][1:]
    ib_idx = field_names.index("ibound_1")
    txt = f"should be int instead of {type(shape.record(0)[ib_idx])}"
    assert type(shape.record(0)[ib_idx]) == int, txt
    shape.close()


@requires_pkg("pandas", "shapefile")
@pytest.mark.slow
@pytest.mark.parametrize("namfile", namfiles())
def test_shapefile(tmpdir, namfile):
    from shapefile import Reader

    model = flopy.modflow.Modflow.load(
        namfile.name, model_ws=str(namfile.parent), verbose=False
    )
    assert model, f"Could not load namefile {namfile}"

    msg = f"Could not load {namfile} model"
    assert isinstance(model, flopy.modflow.Modflow), msg

    fnc_name = os.path.join(tmpdir, f"{model.name}.shp")
    fnc = model.export(fnc_name)
    # fnc2 = m.export(fnc_name, package_names=None)
    # fnc3 = m.export(fnc_name, package_names=['DIS'])

    s = Reader(fnc_name)
    assert (
        s.numRecords == model.nrow * model.ncol
    ), f"wrong number of records in shapefile {fnc_name}"


@requires_pkg("pandas", "shapefile")
@pytest.mark.slow
@pytest.mark.parametrize("namfile", namfiles())
def test_shapefile_export_modelgrid_override(tmpdir, namfile):
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
        grid.epsg,
        grid.proj4,
        xoff=grid.xoffset,
        yoff=grid.yoffset,
        angrot=grid.angrot,
    )

    assert model, f"Could not load namefile {namfile}"
    assert isinstance(model, flopy.modflow.Modflow)

    fnc_name = os.path.join(tmpdir, f"{model.name}.shp")
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
def test_export_netcdf(tmpdir, namfile):
    from netCDF4 import Dataset

    model = flopy.modflow.Modflow.load(
        namfile.name, model_ws=str(namfile.parent), verbose=False
    )
    if model.dis.lenuni == 0:
        model.dis.lenuni = 1

    if model.dis.botm.shape[0] != model.nlay:
        print("skipping...botm.shape[0] != nlay")
        return

    assert model, f"Could not load namefile {namfile}"
    assert isinstance(model, flopy.modflow.Modflow)

    fnc = model.export(os.path.join(tmpdir, f"{model.name}.nc"))
    fnc.write()
    fnc_name = os.path.join(tmpdir, f"{model.name}.nc")
    fnc = model.export(fnc_name)
    fnc.write()

    nc = Dataset(fnc_name, "r")
    nc.close()


@requires_pkg("shapefile")
def test_export_array2(tmpdir):
    nrow = 7
    ncol = 11
    epsg = 4111

    # no epsg code
    modelgrid = StructuredGrid(
        delr=np.ones(ncol) * 1.1, delc=np.ones(nrow) * 1.1
    )
    filename = os.path.join(tmpdir, "myarray1.shp")
    a = np.arange(nrow * ncol).reshape((nrow, ncol))
    export_array(modelgrid, filename, a)
    assert os.path.isfile(filename), "did not create array shapefile"

    # with modelgrid epsg code
    modelgrid = StructuredGrid(
        delr=np.ones(ncol) * 1.1, delc=np.ones(nrow) * 1.1, epsg=epsg
    )
    filename = os.path.join(tmpdir, "myarray2.shp")
    a = np.arange(nrow * ncol).reshape((nrow, ncol))
    export_array(modelgrid, filename, a)
    assert os.path.isfile(filename), "did not create array shapefile"

    # with passing in epsg code
    modelgrid = StructuredGrid(
        delr=np.ones(ncol) * 1.1, delc=np.ones(nrow) * 1.1
    )
    filename = os.path.join(tmpdir, "myarray3.shp")
    a = np.arange(nrow * ncol).reshape((nrow, ncol))
    export_array(modelgrid, filename, a, epsg=epsg)
    assert os.path.isfile(filename), "did not create array shapefile"


@requires_pkg("shapefile", "shapely")
def test_export_array_contours(tmpdir):
    nrow = 7
    ncol = 11
    epsg = 4111

    # no epsg code
    modelgrid = StructuredGrid(
        delr=np.ones(ncol) * 1.1, delc=np.ones(nrow) * 1.1
    )
    filename = os.path.join(tmpdir, "myarraycontours1.shp")
    a = np.arange(nrow * ncol).reshape((nrow, ncol))
    export_array_contours(modelgrid, filename, a)
    assert os.path.isfile(filename), "did not create contour shapefile"

    # with modelgrid epsg code
    modelgrid = StructuredGrid(
        delr=np.ones(ncol) * 1.1, delc=np.ones(nrow) * 1.1, epsg=epsg
    )
    filename = os.path.join(tmpdir, "myarraycontours2.shp")
    a = np.arange(nrow * ncol).reshape((nrow, ncol))
    export_array_contours(modelgrid, filename, a)
    assert os.path.isfile(filename), "did not create contour shapefile"

    # with passing in epsg code
    modelgrid = StructuredGrid(
        delr=np.ones(ncol) * 1.1, delc=np.ones(nrow) * 1.1
    )
    filename = os.path.join(tmpdir, "myarraycontours3.shp")
    a = np.arange(nrow * ncol).reshape((nrow, ncol))
    export_array_contours(modelgrid, filename, a, epsg=epsg)
    assert os.path.isfile(filename), "did not create contour shapefile"


@requires_pkg("shapefile", "shapely")
def test_export_contourf(tmpdir, example_data_path):
    from shapefile import Reader

    filename = os.path.join(tmpdir, "myfilledcontours.shp")
    mpath = example_data_path / "freyberg"
    ml = Modflow.load("freyberg.nam", model_ws=mpath)
    hds_pth = os.path.join(ml.model_ws, "freyberg.githds")
    hds = flopy.utils.HeadFile(hds_pth)
    head = hds.get_data()
    levels = np.arange(10, 30, 0.5)

    mapview = flopy.plot.PlotMapView(model=ml)
    contour_set = mapview.contour_array(
        head, masked_values=[999.0], levels=levels, filled=True
    )

    export_contourf(filename, contour_set)
    plt.close()
    if not os.path.isfile(filename):
        raise AssertionError("did not create contourf shapefile")

    with Reader(filename) as r:
        shapes = r.shapes()
        if len(shapes) != 65:
            raise AssertionError(
                "multipolygons were skipped in contourf routine"
            )


@pytest.mark.mf6
@requires_pkg("shapefile", "shapely")
def test_export_contours(tmpdir, example_data_path):
    from shapefile import Reader

    filename = os.path.join(tmpdir, "mycontours.shp")
    mpath = example_data_path / "freyberg"
    ml = Modflow.load("freyberg.nam", model_ws=mpath)
    hds_pth = os.path.join(ml.model_ws, "freyberg.githds")
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
        assert len(shapes) == 65


@pytest.mark.mf6
@requires_pkg("shapely")
def test_mf6_grid_shp_export(tmpdir):
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
        model_ws=str(tmpdir),
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
        sim_ws=str(tmpdir),
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
    m.export(str(tmpdir / "mfnwt.shp"))
    gwf.export(str(tmpdir / "mf6.shp"))

    # check that the two shapefiles are the same
    ra = shp2recarray(str(tmpdir / "mfnwt.shp"))
    ra6 = shp2recarray(str(tmpdir / "mf6.shp"))

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
def test_export_huge_shapefile(tmpdir):
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

    m = flopy.modflow.Modflow("junk", version="mfnwt", model_ws=str(tmpdir))
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

    m.export(str(tmpdir / "huge.shp"))


@requires_pkg("netCDF4", "pyproj")
def test_polygon_from_ij(tmpdir):
    """test creation of a polygon from an i, j location using get_vertices()."""
    ws = str(tmpdir)
    m = Modflow("toy_model", model_ws=ws)

    botm = np.zeros((2, 10, 10))
    botm[0, :, :] = 1.5
    botm[1, 5, 5] = 4  # negative layer thickness!
    botm[1, 6, 6] = 4
    dis = ModflowDis(
        nrow=10, ncol=10, nlay=2, delr=100, delc=100, top=3, botm=botm, model=m
    )

    fname = os.path.join(ws, "toy.model.nc")
    ncdf = NetCdf(fname, m)
    ncdf.write()

    fname = os.path.join(ws, "toy_model_two.nc")
    m.export(fname)

    fname = os.path.join(ws, "toy_model_dis.nc")
    dis.export(fname)

    mg = m.modelgrid
    mg.set_coord_info(
        xoff=mg._xul_to_xll(600000.0, -45.0),
        yoff=mg._yul_to_yll(5170000, -45.0),
        angrot=-45.0,
        proj4="EPSG:26715",
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


@requires_pkg("netCDF4", "pyproj")
@requires_spatial_reference
def test_polygon_from_ij_with_epsg(tmpdir):
    ws = str(tmpdir)
    m = Modflow("toy_model", model_ws=ws)

    botm = np.zeros((2, 10, 10))
    botm[0, :, :] = 1.5
    botm[1, 5, 5] = 4  # negative layer thickness!
    botm[1, 6, 6] = 4
    dis = ModflowDis(
        nrow=10, ncol=10, nlay=2, delr=100, delc=100, top=3, botm=botm, model=m
    )

    fname = os.path.join(ws, "toy.model.nc")
    ncdf = NetCdf(fname, m)
    ncdf.write()

    fname = os.path.join(ws, "toy_model_two.nc")
    m.export(fname)

    fname = os.path.join(ws, "toy_model_dis.nc")
    dis.export(fname)

    mg = m.modelgrid
    mg.set_coord_info(
        xoff=mg._xul_to_xll(600000.0, -45.0),
        yoff=mg._yul_to_yll(5170000, -45.0),
        angrot=-45.0,
        proj4="EPSG:26715",
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
    recarray2shp(recarray, geoms, fpth, epsg=26715)

    # tries to connect to https://spatialreference.org,
    # might fail with CERTIFICATE_VERIFY_FAILED (on Mac,
    # run Python Install Certificates) but intermittent
    # 502s are also possible and possibly unavoidable)
    ep = EpsgReference()
    prj = ep.to_dict()

    assert 26715 in prj

    fpth = os.path.join(ws, "test.prj")
    fpth2 = os.path.join(ws, "26715.prj")
    shutil.copy(fpth, fpth2)
    fpth = os.path.join(ws, "test.shp")
    recarray2shp(recarray, geoms, fpth, prj=fpth2)

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
def test_vtk_export_array2d(tmpdir, example_data_path):
    # test mf 2005 freyberg
    mpath = str(example_data_path / "freyberg_multilayer_transient")
    namfile = "freyberg.nam"
    m = Modflow.load(
        namfile, model_ws=mpath, verbose=False, load_only=["dis", "bas6"]
    )

    # export and check
    m.dis.top.export(str(tmpdir), name="top", fmt="vtk", binary=False)
    assert count_lines_in_file(tmpdir / "top.vtk") == 17615

    # with smoothing
    m.dis.top.export(
        str(tmpdir), fmt="vtk", name="top_smooth", binary=False, smooth=True
    )
    assert count_lines_in_file(tmpdir / "top_smooth.vtk") == 17615


@requires_pkg("vtk")
def test_vtk_export_array3d(tmpdir, example_data_path):
    # test mf 2005 freyberg
    mpath = str(example_data_path / "freyberg_multilayer_transient")
    namfile = "freyberg.nam"
    m = Modflow.load(
        namfile,
        model_ws=mpath,
        verbose=False,
        load_only=["dis", "bas6", "upw"],
    )

    # export and check
    m.upw.hk.export(str(tmpdir), fmt="vtk", name="hk", binary=False)
    assert count_lines_in_file(tmpdir / "hk.vtk") == 17615

    # with point scalars
    m.upw.hk.export(
        str(tmpdir),
        fmt="vtk",
        name="hk_points",
        point_scalars=True,
        binary=False,
    )
    assert count_lines_in_file(tmpdir / "hk_points.vtk") == 19482

    # with point scalars and binary
    m.upw.hk.export(
        str(tmpdir),
        fmt="vtk",
        name="hk_points_bin",
        point_scalars=True,
    )
    assert is_binary_file(tmpdir / "hk_points_bin.vtk")


@requires_pkg("vtk")
def test_vtk_transient_array_2d(tmpdir, example_data_path):
    # test mf 2005 freyberg
    ws = str(tmpdir)
    mpath = str(example_data_path / "freyberg_multilayer_transient")
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
    assert count_lines_in_file(tmpdir / "rech_000001.vtk") == 26837
    assert count_lines_in_file(tmpdir / "rech_001096.vtk") == 26837

    # with binary

    m.rch.rech.export(ws, fmt="vtk", binary=True, kpers=kpers)
    assert is_binary_file(tmpdir / "rech_000001.vtk")
    assert is_binary_file(tmpdir / "rech_001096.vtk")


@requires_pkg("vtk")
@pytest.mark.slow
def test_vtk_export_packages(tmpdir, example_data_path):
    # test mf 2005 freyberg
    ws = str(tmpdir)
    mpath = str(example_data_path / "freyberg_multilayer_transient")
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
    filetocheck = tmpdir / "DIS.vtk"
    # totalbytes = os.path.getsize(filetocheck)
    # assert(totalbytes==1019857)
    assert count_lines_in_file(filetocheck) == 27239

    # upw with point scalar output
    m.upw.export(ws, fmt="vtk", xml=True, binary=False, point_scalars=True)
    assert count_lines_in_file(tmpdir / "UPW.vtk") == 42445

    # bas with smoothing on
    m.bas6.export(ws, fmt="vtk", binary=False, smooth=True)
    assert count_lines_in_file(tmpdir / "BAS6.vtk") == 17883

    # transient package drain
    kpers = [0, 1, 1096]
    m.drn.export(ws, fmt="vtk", binary=False, xml=True, kpers=kpers, pvd=True)
    assert count_lines_in_file(tmpdir / "DRN_000001.vtu") == 27239
    assert count_lines_in_file(tmpdir / "DRN_001096.vtu") == 27239

    # dis with binary
    m.dis.export(ws, fmt="vtk", binary=True)
    assert is_binary_file(tmpdir / "DIS.vtk")

    # upw with point scalars and binary
    m.upw.export(ws, fmt="vtk", point_scalars=True, binary=True)
    assert is_binary_file(tmpdir / "UPW.vtk")


@pytest.mark.mf6
@requires_pkg("vtk")
def test_vtk_mf6(tmpdir, example_data_path):
    # test mf6
    mf6expth = str(example_data_path / "mf6")
    mf6sims = [
        "test045_lake1ss_table",
        "test036_twrihfb",
        "test045_lake2tr",
        "test006_2models_mvr",
    ]

    for simnm in mf6sims:
        print(simnm)
        simpth = os.path.join(mf6expth, simnm)
        loaded_sim = MFSimulation.load(simnm, "mf6", "mf6", simpth)
        sim_models = loaded_sim.model_names
        print(sim_models)
        for mname in sim_models:
            print(mname)
            m = loaded_sim.get_model(mname)
            m.export(str(tmpdir), fmt="vtk", binary=False)

    # check one
    filetocheck = tmpdir / "twrihfb2015_000000.vtk"
    # totalbytes = os.path.getsize(filetocheck)
    # assert(totalbytes==21609)
    assert count_lines_in_file(filetocheck) == 9537


@requires_pkg("vtk")
@pytest.mark.slow
def test_vtk_binary_head_export(tmpdir, example_data_path):
    # test mf 2005 freyberg
    mpth = str(example_data_path / "freyberg_multilayer_transient")
    namfile = "freyberg.nam"
    hdsfile = os.path.join(mpth, "freyberg.hds")
    heads = HeadFile(hdsfile)
    m = Modflow.load(
        namfile, model_ws=mpth, verbose=False, load_only=["dis", "bas6"]
    )
    filetocheck = tmpdir / "freyberg_head_000003.vtu"

    # export and check

    vtkobj = Vtk(m, pvd=True, xml=True)
    vtkobj.add_heads(
        heads, kstpkper=[(0, 0), (0, 199), (0, 354), (0, 454), (0, 1089)]
    )
    vtkobj.write(tmpdir / "freyberg_head")

    assert count_lines_in_file(filetocheck) == 34
    filetocheck.unlink()

    # with point scalars

    vtkobj = Vtk(m, pvd=True, xml=True, point_scalars=True)
    vtkobj.add_heads(
        heads, kstpkper=[(0, 0), (0, 199), (0, 354), (0, 454), (0, 1089)]
    )
    vtkobj.write(tmpdir / "freyberg_head")

    assert count_lines_in_file(filetocheck) == 34
    filetocheck.unlink()

    # with smoothing

    vtkobj = Vtk(m, pvd=True, xml=True, smooth=True)
    vtkobj.add_heads(
        heads, kstpkper=[(0, 0), (0, 199), (0, 354), (0, 454), (0, 1089)]
    )
    vtkobj.write(tmpdir / "freyberg_head")

    assert count_lines_in_file(filetocheck) == 34


@requires_pkg("vtk")
@pytest.mark.slow
def test_vtk_cbc(tmpdir, example_data_path):
    # test mf 2005 freyberg

    ws = str(tmpdir)
    mpth = str(example_data_path / "freyberg_multilayer_transient")
    namfile = "freyberg.nam"
    cbcfile = os.path.join(mpth, "freyberg.cbc")
    cbc = CellBudgetFile(cbcfile)
    m = Modflow.load(
        namfile, model_ws=mpth, verbose=False, load_only=["dis", "bas6"]
    )

    # export and check with point scalar
    vtkobj = Vtk(m, binary=False, xml=True, pvd=True, point_scalars=True)
    vtkobj.add_cell_budget(cbc, kstpkper=[(0, 0), (0, 1), (0, 2)])
    vtkobj.write(tmpdir / "freyberg_CBC")

    assert count_lines_in_file(tmpdir / "freyberg_CBC_000000.vtu") == 39243

    # with point scalars and binary
    vtkobj = Vtk(m, xml=True, pvd=True, point_scalars=True)
    vtkobj.add_cell_budget(cbc, kstpkper=[(0, 0), (0, 1), (0, 2)])
    vtkobj.write(tmpdir / "freyberg_CBC")
    assert count_lines_in_file(tmpdir / "freyberg_CBC_000000.vtu") == 28


@requires_pkg("vtk")
@pytest.mark.slow
def test_vtk_vector(tmpdir, example_data_path):
    # test mf 2005 freyberg
    mpth = str(example_data_path / "freyberg_multilayer_transient")
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

    filenametocheck = tmpdir / "discharge.vtu"

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

    filenametocheck = tmpdir / "discharge_verts.vtu"
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
def test_vtk_unstructured(tmpdir, example_data_path):
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkIOLegacy import vtkUnstructuredGridReader

    u_data_ws = example_data_path / "unstructured"

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

    outfile = tmpdir / "disu_grid.vtu"
    vtkobj = Vtk(
        modelgrid=modelgrid, vertical_exageration=2, binary=True, smooth=False
    )
    vtkobj.add_array(modelgrid.top, "top")
    vtkobj.add_array(modelgrid.botm, "botm")
    vtkobj.write(outfile)

    assert is_binary_file(outfile)

    reader = vtkUnstructuredGridReader()
    reader.SetFileName(str(outfile))
    reader.ReadAllFieldsOn()
    reader.Update()

    data = reader.GetOutput()

    top2 = vtk_to_numpy(data.GetCellData().GetArray("top"))

    assert np.allclose(np.ravel(top), top2), "Field data not properly written"


@pytest.mark.mf6
@requires_pkg("vtk")
def test_vtk_vertex(tmpdir, example_data_path):
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkIOLegacy import vtkUnstructuredGridReader

    # disv test
    workspace = str(example_data_path / "mf6" / "test003_gwfs_disv")
    # outfile = os.path.join("vtk_transient_test", "vtk_pacakages")
    sim = MFSimulation.load(sim_ws=workspace)
    gwf = sim.get_model("gwf_1")

    outfile = tmpdir / "disv.vtk"
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
@requires_pkg("pandas", "vtk")
def test_vtk_pathline(tmpdir, example_data_path):
    from vtkmodules.vtkIOLegacy import vtkUnstructuredGridReader

    # pathline test for vtk
    ws = str(example_data_path / "freyberg")
    ml = Modflow.load("freyberg.nam", model_ws=ws, exe_name="mf2005")

    ml.change_model_ws(new_pth=str(tmpdir))
    ml.write_input()
    ml.run_model()

    mpp = Modpath6(
        "freybergmpp", modflowmodel=ml, model_ws=str(tmpdir), exe_name="mp6"
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

    pthfile = os.path.join(tmpdir, mpp.sim.pathline_file)
    pthobj = PathlineFile(pthfile)
    travel_time_max = 200.0 * 365.25 * 24.0 * 60.0 * 60.0
    plines = pthobj.get_alldata(totim=travel_time_max, ge=False)

    outfile = tmpdir / "pathline.vtk"

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

    totim = numpy_support.vtk_to_numpy(data.GetCellData().GetArray("time"))
    pid = numpy_support.vtk_to_numpy(data.GetCellData().GetArray("particleid"))

    maxtime = 0
    for p in plines:
        if np.max(p["time"]) > maxtime:
            maxtime = np.max(p["time"])

    if not len(totim) == 12054:
        raise AssertionError("Array size is incorrect for modpath VTK")

    if not np.abs(np.max(totim) - maxtime) < 100:
        raise AssertionError("time values are incorrect for modpath VTK")

    if not len(np.unique(pid)) == len(plines):
        raise AssertionError(
            "number of particles are incorrect for modpath VTK"
        )


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
def test_vtk_export_model_without_packages_names(tmpdir):
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkIOLegacy import vtkUnstructuredGridReader

    from flopy.export.vtk import Vtk

    name = "mymodel"
    sim = MFSimulation(sim_name=name, sim_ws=str(tmpdir), exe_name="mf6")
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
    vtkobj.write(tmpdir / "gwf.vtk")

    # load the output using the vtk standard library
    gridreader = vtkUnstructuredGridReader()
    gridreader.SetFileName(str(tmpdir / "gwf_000000.vtk"))
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
@requires_pkg("vtk")
def test_vtk_export_disv1_model(tmpdir):
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkIOLegacy import vtkUnstructuredGridReader

    from flopy.export.vtk import Vtk

    name = "mymodel"
    sim = MFSimulation(sim_name=name, sim_ws=str(tmpdir), exe_name="mf6")
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
    f = tmpdir / "gwf.vtk"
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
@requires_pkg("vtk")
def test_vtk_export_disv2_model(tmpdir):
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkIOLegacy import vtkUnstructuredGridReader

    from flopy.export.vtk import Vtk

    # in this case, test for iverts that do not explicitly close the cell polygons
    name = "mymodel"
    sim = MFSimulation(sim_name=name, sim_ws=str(tmpdir), exe_name="mf6")
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
    f = tmpdir / "gwf.vtk"
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
def test_vtk_export_disu1_grid(tmpdir, example_data_path):
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

    outfile = tmpdir / "disu_grid.vtu"
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
def test_vtk_export_disu2_grid(tmpdir, example_data_path):
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

    outfile = tmpdir / "disu_grid.vtu"
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
@requires_pkg("vtk", "shapefile")
def test_vtk_export_disu_model(tmpdir):
    from vtkmodules.util.numpy_support import vtk_to_numpy

    from flopy.export.vtk import Vtk
    from flopy.utils.gridgen import Gridgen

    name = "mymodel"

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

    sim = MFSimulation(sim_name=name, sim_ws=str(tmpdir), exe_name="mf6")
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

    # export grid
    vtk = import_optional_dependency("vtk")

    vtkobj = Vtk(gwf, binary=False)
    vtkobj.add_model(gwf)
    f = tmpdir / "gwf.vtk"
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
