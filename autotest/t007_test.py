# Test export module
import sys

sys.path.append("..")
import copy
import glob
import os
import shutil
import numpy as np
import flopy

pth = os.path.join("..", "examples", "data", "mf2005_test")
namfiles = [namfile for namfile in os.listdir(pth) if namfile.endswith(".nam")]
# skip = ["MNW2-Fig28.nam", "testsfr2.nam", "testsfr2_tab.nam"]
skip = []

tpth = os.path.join("temp", "t007")
# make the directory if it does not exist
if not os.path.isdir(tpth):
    os.makedirs(tpth)

npth = os.path.join("temp", "t007", "netcdf")
# delete the directory if it exists
if os.path.isdir(npth):
    shutil.rmtree(npth)
# make the directory
os.makedirs(npth)

spth = os.path.join("temp", "t007", "shapefile")
# make the directory if it does not exist
if not os.path.isdir(spth):
    os.makedirs(spth)


def import_shapefile():
    try:
        import shapefile
    except ImportError:
        return None
    if int(shapefile.__version__.split(".")[0]) < 2:
        return None
    return shapefile


def remove_shp(shpname):
    os.remove(shpname)
    for ext in ["prj", "shx", "dbf"]:
        fname = shpname.replace("shp", ext)
        if os.path.exists(fname):
            os.remove(fname)


def export_mf6_netcdf(path):
    print(f"in export_mf6_netcdf: {path}")
    sim = flopy.mf6.modflow.mfsimulation.MFSimulation.load(sim_ws=path)
    for name, model in sim.get_model_itr():
        export_netcdf(model)


def export_mf2005_netcdf(namfile):
    print(f"in export_mf2005_netcdf: {namfile}")
    if namfile in skip:
        return
    m = flopy.modflow.Modflow.load(namfile, model_ws=pth, verbose=False)
    if m.dis.lenuni == 0:
        m.dis.lenuni = 1
        # print('skipping...lenuni==0 (undefined)')
        # return
    # if sum(m.dis.laycbd) != 0:
    if m.dis.botm.shape[0] != m.nlay:
        print("skipping...botm.shape[0] != nlay")
        return
    assert m, f"Could not load namefile {namfile}"
    msg = f"Could not load {namfile} model"
    assert isinstance(m, flopy.modflow.Modflow), msg
    export_netcdf(m)


def export_netcdf(m):
    # Do not fail if netCDF4 not installed
    try:
        import netCDF4
        import pyproj
    except:
        return
    fnc = m.export(os.path.join(npth, f"{m.name}.nc"))
    fnc.write()
    fnc_name = os.path.join(npth, f"{m.name}.nc")
    try:
        fnc = m.export(fnc_name)
        fnc.write()
    except Exception as e:
        msg = f"ncdf export fail for namfile {m.name}:\n{e!s}  "
        raise Exception(msg)

    try:
        nc = netCDF4.Dataset(fnc_name, "r")
    except Exception as e:
        msg = f"ncdf import fail for nc file {fnc_name}:\n{e!s}"
        raise Exception()
    return


def export_shapefile(namfile):
    print(f"in export_shapefile: {namfile}")
    shp = import_shapefile()
    if shp is None:
        return

    m = flopy.modflow.Modflow.load(namfile, model_ws=pth, verbose=False)

    assert m, f"Could not load namefile {namfile}"
    msg = f"Could not load {namfile} model"
    assert isinstance(m, flopy.modflow.Modflow), msg
    fnc_name = os.path.join(spth, f"{m.name}.shp")

    try:
        fnc = m.export(fnc_name)
        # fnc2 = m.export(fnc_name, package_names=None)
        # fnc3 = m.export(fnc_name, package_names=['DIS'])
    except Exception as e:
        msg = f"shapefile export fail for namfile {namfile}:\n{e!s}"
        raise Exception(msg)

    try:
        s = shp.Reader(fnc_name)
    except Exception as e:
        msg = f"shapefile import fail for {fnc_name}:\n{e!s}"
        raise Exception(msg)
    msg = f"wrong number of records in shapefile {fnc_name}:{s.numRecords}"
    assert s.numRecords == m.nrow * m.ncol, msg
    return


def export_shapefile_modelgrid_override(namfile):
    print(f"in export_modelgrid_override: {namfile}")
    shp = import_shapefile()
    if shp is None:
        return

    from flopy.discretization import StructuredGrid

    m = flopy.modflow.Modflow.load(namfile, model_ws=pth, verbose=False)
    mg0 = m.modelgrid
    modelgrid = StructuredGrid(
        mg0.delc * 0.3048,
        mg0.delr * 0.3048,
        mg0.top,
        mg0.botm,
        mg0.idomain,
        mg0.lenuni,
        mg0.epsg,
        mg0.proj4,
        xoff=mg0.xoffset,
        yoff=mg0.yoffset,
        angrot=mg0.angrot,
    )

    assert m, f"Could not load namefile {namfile}"
    assert isinstance(m, flopy.modflow.Modflow)
    fnc_name = os.path.join(spth, f"{m.name}.shp")

    try:
        fnc = m.export(fnc_name, modelgrid=modelgrid)
        # fnc2 = m.export(fnc_name, package_names=None)
        # fnc3 = m.export(fnc_name, package_names=['DIS'])

    except Exception as e:
        msg = f"shapefile export fail for namfile {namfile}:\n{e!s}"
        raise Exception(msg)
    try:
        s = shp.Reader(fnc_name)
    except Exception as e:
        msg = f"shapefile import fail for {fnc_name}:{e!s}"
        raise Exception(msg)


def test_output_helper_shapefile_export():
    if import_shapefile() is None:
        return
    ws = os.path.join(
        "..", "examples", "data", "freyberg_multilayer_transient"
    )
    name = "freyberg.nam"

    ml = flopy.modflow.Modflow.load(name, model_ws=ws)

    head = flopy.utils.HeadFile(os.path.join(ws, "freyberg.hds"))
    cbc = flopy.utils.CellBudgetFile(os.path.join(ws, "freyberg.cbc"))
    flopy.export.utils.output_helper(
        os.path.join("temp", "test.shp"),
        ml,
        {"HDS": head, "cbc": cbc},
        mflay=1,
        kper=10,
    )


def test_freyberg_export():
    if import_shapefile() is None:
        return

    from flopy.discretization import StructuredGrid

    namfile = "freyberg.nam"

    # steady state
    model_ws = "../examples/data/freyberg"
    m = flopy.modflow.Modflow.load(
        namfile, model_ws=model_ws, check=False, verbose=False
    )
    # test export at model, package and object levels
    m.export(f"{spth}/model.shp")
    m.wel.export(f"{spth}/wel.shp")
    m.lpf.hk.export(f"{spth}/hk.shp")
    m.riv.stress_period_data.export(f"{spth}/riv_spd.shp")

    # transient
    # (doesn't work at model level because the total size of
    #  the attribute fields exceeds the shapefile limit)
    model_ws = "../examples/data/freyberg_multilayer_transient/"
    m = flopy.modflow.Modflow.load(
        namfile,
        model_ws=model_ws,
        verbose=False,
        load_only=["DIS", "BAS6", "NWT", "OC", "RCH", "WEL", "DRN", "UPW"],
    )
    # test export without instantiating an sr
    outshp = os.path.join(spth, f"{namfile[:-4]}_drn_sparse.shp")
    m.drn.stress_period_data.export(outshp, sparse=True)
    assert os.path.exists(outshp)
    remove_shp(outshp)
    m.modelgrid = StructuredGrid(
        delc=m.dis.delc.array, delr=m.dis.delr.array, epsg=3070
    )
    # test export with an sr, regardless of whether or not wkt was found
    m.drn.stress_period_data.export(outshp, sparse=True)
    assert os.path.exists(outshp)
    remove_shp(outshp)
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
        outshp = os.path.join(spth, f"{namfile[:-4]}_dis.shp")
        m.dis.export(outshp)
        prjfile = outshp.replace(".shp", ".prj")
        with open(prjfile) as src:
            prjtxt = src.read()
        assert prjtxt == wkt
        remove_shp(outshp)

        # test default package export to higher level dir
        outshp = os.path.join(spth, f"{namfile[:-4]}_dis.shp")
        m.dis.export(outshp)
        prjfile = outshp.replace(".shp", ".prj")
        with open(prjfile) as src:
            prjtxt = src.read()
        assert prjtxt == wkt
        remove_shp(outshp)

        # test sparse package export
        outshp = os.path.join(spth, f"{namfile[:-4]}_drn_sparse.shp")
        m.drn.stress_period_data.export(outshp, sparse=True)
        prjfile = outshp.replace(".shp", ".prj")
        assert os.path.exists(prjfile)
        with open(prjfile) as src:
            prjtxt = src.read()
        assert prjtxt == wkt
        remove_shp(outshp)


def test_export_output():

    # Do not fail if netCDF4 not installed
    try:
        import netCDF4
        import pyproj
    except:
        return

    model_ws = os.path.join("..", "examples", "data", "freyberg")
    ml = flopy.modflow.Modflow.load("freyberg.nam", model_ws=model_ws)
    hds_pth = os.path.join(model_ws, "freyberg.githds")
    hds = flopy.utils.HeadFile(hds_pth)

    out_pth = os.path.join(npth, "freyberg.out.nc")
    nc = flopy.export.utils.output_helper(
        out_pth, ml, {"freyberg.githds": hds}
    )
    var = nc.nc.variables.get("head")
    arr = var[:]
    ibound_mask = ml.bas6.ibound.array == 0
    arr_mask = arr.mask[0]
    assert np.array_equal(ibound_mask, arr_mask)


def test_write_shapefile():
    sf = import_shapefile()
    if not sf:
        return

    from flopy.discretization import StructuredGrid
    from flopy.export.shapefile_utils import shp2recarray
    from flopy.export.shapefile_utils import write_grid_shapefile

    sg = StructuredGrid(
        delr=np.ones(10) * 1.1,
        # cell spacing along model rows
        delc=np.ones(10) * 1.1,
        # cell spacing along model columns
        epsg=26715,
    )
    outshp = os.path.join(tpth, "junk.shp")
    write_grid_shapefile(outshp, sg, array_dict={})

    # test that vertices aren't getting altered by writing shapefile
    # check that pyshp reads integers
    # this only check that row/column were recorded as "N"
    # not how they will be cast by python or numpy
    sfobj = sf.Reader(outshp)
    for f in sfobj.fields:
        if f[0] == "row" or f[0] == "column":
            assert f[1] == "N"
    recs = list(sfobj.records())
    for r in recs[0]:
        assert isinstance(r, int)

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
    except:
        pass


def test_shapefile_polygon_closed():
    shapefile = import_shapefile()
    if shapefile is None:
        return

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

    shp_file = os.path.join(spth, "test_polygon.shp")
    m.dis.export(shp_file)

    shp = shapefile.Reader(shp_file)
    for shape in shp.iterShapes():
        if len(shape.points) != 5:
            raise AssertionError("Shapefile polygon is not closed!")


def test_export_array():
    from flopy.export import utils

    try:
        from scipy.ndimage import rotate
    except:
        rotate = False
        pass

    namfile = "freyberg.nam"
    model_ws = "../examples/data/freyberg_multilayer_transient/"
    m = flopy.modflow.Modflow.load(
        namfile, model_ws=model_ws, verbose=False, load_only=["DIS", "BAS6"]
    )
    m.modelgrid.set_coord_info(angrot=45)
    nodata = -9999
    utils.export_array(
        m.modelgrid,
        os.path.join(tpth, "fb.asc"),
        m.dis.top.array,
        nodata=nodata,
    )
    arr = np.loadtxt(os.path.join(tpth, "fb.asc"), skiprows=6)

    if import_shapefile() is not None:
        m.modelgrid.write_shapefile(os.path.join(tpth, "grid.shp"))

    # check bounds
    with open(os.path.join(tpth, "fb.asc")) as src:
        for line in src:
            if "xllcorner" in line.lower():
                val = float(line.strip().split()[-1])
                # ascii grid origin will differ if it was unrotated
                if rotate:
                    assert np.abs(val - m.modelgrid.extent[0]) < 1e-6
                else:
                    assert np.abs(val - m.modelgrid.xoffset) < 1e-6
            if "yllcorner" in line.lower():
                val = float(line.strip().split()[-1])
                if rotate:
                    assert np.abs(val - m.modelgrid.extent[2]) < 1e-6
                else:
                    assert np.abs(val - m.modelgrid.yoffset) < 1e-6
            if "cellsize" in line.lower():
                val = float(line.strip().split()[-1])
                rot_cellsize = (
                    np.cos(np.radians(m.modelgrid.angrot))
                    * m.modelgrid.delr[0]
                )
                break
    rotate = False
    rasterio = None
    if rotate:
        rotated = rotate(m.dis.top.array, m.modelgrid.angrot, cval=nodata)

    if rotate:
        assert rotated.shape == arr.shape

    try:
        # test GeoTIFF export
        import rasterio
    except:
        pass
    if rasterio is not None:
        utils.export_array(
            m.modelgrid,
            os.path.join(tpth, "fb.tif"),
            m.dis.top.array,
            nodata=nodata,
        )
        with rasterio.open(os.path.join(tpth, "fb.tif")) as src:
            arr = src.read(1)
            assert src.shape == (m.nrow, m.ncol)
            # TODO: these tests currently fail -- fix is in progress
            # assert np.abs(src.bounds[0] - m.modelgrid.extent[0]) < 1e-6
            # assert np.abs(src.bounds[1] - m.modelgrid.extent[1]) < 1e-6


def test_mbase_modelgrid():

    ml = flopy.modflow.Modflow(
        modelname="test", xll=500.0, rotation=12.5, start_datetime="1/1/2016"
    )
    try:
        print(ml.modelgrid.xcentergrid)
    except:
        pass
    else:
        raise Exception("should have failed")

    dis = flopy.modflow.ModflowDis(ml, nrow=10, ncol=5, delr=np.arange(5))

    assert ml.modelgrid.xoffset == 500
    assert ml.modelgrid.yoffset == 0.0
    assert ml.modelgrid.proj4 is None
    ml.model_ws = tpth

    ml.write_input()
    ml1 = flopy.modflow.Modflow.load("test.nam", model_ws=ml.model_ws)
    assert str(ml1.modelgrid) == str(ml.modelgrid)
    assert ml1.start_datetime == ml.start_datetime
    assert ml1.modelgrid.proj4 is None


def test_mt_modelgrid():

    ml = flopy.modflow.Modflow(
        modelname="test",
        xll=500.0,
        proj4_str="epsg:2193",
        rotation=12.5,
        start_datetime="1/1/2016",
    )
    dis = flopy.modflow.ModflowDis(ml, nrow=10, ncol=5, delr=np.arange(5))

    assert ml.modelgrid.xoffset == 500
    assert ml.modelgrid.yoffset == 0.0
    assert ml.modelgrid.epsg == 2193
    assert ml.modelgrid.idomain is None
    ml.model_ws = tpth

    mt = flopy.mt3d.Mt3dms(
        modelname="test_mt",
        modflowmodel=ml,
        model_ws=ml.model_ws,
        verbose=True,
    )

    assert mt.modelgrid.xoffset == ml.modelgrid.xoffset
    assert mt.modelgrid.yoffset == ml.modelgrid.yoffset
    assert mt.modelgrid.epsg == ml.modelgrid.epsg
    assert mt.modelgrid.angrot == ml.modelgrid.angrot
    assert np.array_equal(mt.modelgrid.idomain, ml.modelgrid.idomain)

    # no modflowmodel
    swt = flopy.seawat.Seawat(
        modelname="test_swt",
        modflowmodel=None,
        mt3dmodel=None,
        model_ws=ml.model_ws,
        verbose=True,
    )
    assert swt.modelgrid is swt.dis is swt.bas6 is None

    # passing modflowmodel
    swt = flopy.seawat.Seawat(
        modelname="test_swt",
        modflowmodel=ml,
        mt3dmodel=mt,
        model_ws=ml.model_ws,
        verbose=True,
    )

    assert (
        swt.modelgrid.xoffset == mt.modelgrid.xoffset == ml.modelgrid.xoffset
    )
    assert (
        swt.modelgrid.yoffset == mt.modelgrid.yoffset == ml.modelgrid.yoffset
    )
    assert mt.modelgrid.epsg == ml.modelgrid.epsg == swt.modelgrid.epsg
    assert mt.modelgrid.angrot == ml.modelgrid.angrot == swt.modelgrid.angrot
    assert np.array_equal(mt.modelgrid.idomain, ml.modelgrid.idomain)
    assert np.array_equal(swt.modelgrid.idomain, ml.modelgrid.idomain)

    # bas and btn present
    ibound = np.ones(ml.dis.botm.shape)
    ibound[0][0:5] = 0
    bas = flopy.modflow.ModflowBas(ml, ibound=ibound)
    assert ml.modelgrid.idomain is not None

    mt = flopy.mt3d.Mt3dms(
        modelname="test_mt",
        modflowmodel=ml,
        model_ws=ml.model_ws,
        verbose=True,
    )
    btn = flopy.mt3d.Mt3dBtn(mt, icbund=ml.bas6.ibound.array)

    # reload swt
    swt = flopy.seawat.Seawat(
        modelname="test_swt",
        modflowmodel=ml,
        mt3dmodel=mt,
        model_ws=ml.model_ws,
        verbose=True,
    )

    assert (
        ml.modelgrid.xoffset == mt.modelgrid.xoffset == swt.modelgrid.xoffset
    )
    assert (
        mt.modelgrid.yoffset == ml.modelgrid.yoffset == swt.modelgrid.yoffset
    )
    assert mt.modelgrid.epsg == ml.modelgrid.epsg == swt.modelgrid.epsg
    assert mt.modelgrid.angrot == ml.modelgrid.angrot == swt.modelgrid.angrot
    assert np.array_equal(mt.modelgrid.idomain, ml.modelgrid.idomain)
    assert np.array_equal(swt.modelgrid.idomain, ml.modelgrid.idomain)


def test_free_format_flag():

    Lx = 100.0
    Ly = 100.0
    nlay = 1
    nrow = 51
    ncol = 51
    delr = Lx / ncol
    delc = Ly / nrow
    top = 0
    botm = [-1]
    ms = flopy.modflow.Modflow(rotation=20.0)
    dis = flopy.modflow.ModflowDis(
        ms,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
    )
    bas = flopy.modflow.ModflowBas(ms, ifrefm=True)
    assert ms.free_format_input == bas.ifrefm
    ms.free_format_input = False
    assert ms.free_format_input == bas.ifrefm
    ms.free_format_input = True
    bas.ifrefm = False
    assert ms.free_format_input == bas.ifrefm
    bas.ifrefm = True
    assert ms.free_format_input == bas.ifrefm

    ms.model_ws = tpth
    ms.write_input()
    ms1 = flopy.modflow.Modflow.load(ms.namefile, model_ws=ms.model_ws)
    assert ms1.free_format_input == ms.free_format_input
    assert ms1.free_format_input == ms1.bas6.ifrefm
    ms1.free_format_input = False
    assert ms1.free_format_input == ms1.bas6.ifrefm
    bas.ifrefm = False
    assert ms1.free_format_input == ms1.bas6.ifrefm
    bas.ifrefm = True
    assert ms1.free_format_input == ms1.bas6.ifrefm


def test_sr():

    m = flopy.modflow.Modflow(
        "test",
        model_ws="./temp",
        xll=12345,
        yll=12345,
        proj4_str="test test test",
    )
    flopy.modflow.ModflowDis(m, 10, 10, 10)
    m.write_input()
    mm = flopy.modflow.Modflow.load("test.nam", model_ws="./temp")
    extents = mm.modelgrid.extent
    if extents[2] != 12345:
        raise AssertionError()
    if extents[3] != 12355:
        raise AssertionError()
    if mm.modelgrid.proj4 != "test test test":
        raise AssertionError()

    mm.dis.top = 5000

    if not np.allclose(mm.dis.top.array, mm.modelgrid.top):
        raise AssertionError("modelgrid failed dynamic update test")


def test_dis_sr():

    delr = 640
    delc = 640
    nrow = np.ceil(59040.0 / delc).astype(int)
    ncol = np.ceil(33128.0 / delr).astype(int)
    nlay = 3

    xul = 2746975.089
    yul = 1171446.45
    rotation = -39
    bg = flopy.modflow.Modflow(modelname="base")
    dis = flopy.modflow.ModflowDis(
        bg,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        lenuni=1,
        rotation=rotation,
        xul=xul,
        yul=yul,
        proj4_str="epsg:2243",
    )

    # Use StructuredGrid instead
    x, y = bg.modelgrid.get_coords(0, delc * nrow)
    np.testing.assert_almost_equal(x, xul)
    np.testing.assert_almost_equal(y, yul)


def test_mf6_modelgrid_update():
    base_dir = os.path.join("..", "examples", "data", "mf6")
    # dis
    model_ws = os.path.join(base_dir, "test001a_Tharmonic")
    sim = flopy.mf6.MFSimulation.load(sim_ws=model_ws)
    gwf = sim.get_model("flow15")

    mg = gwf.modelgrid
    gwf.dis.top = 12

    if not np.allclose(gwf.dis.top.array, gwf.modelgrid.top):
        raise AssertionError("StructuredGrid failed dynamic update test")

    # disv
    model_ws = os.path.join(base_dir, "test003_gwfs_disv")
    sim = flopy.mf6.MFSimulation.load(sim_ws=model_ws)
    gwf = sim.get_model("gwf_1")

    mg = gwf.modelgrid
    gwf.disv.top = 6.12

    if not np.allclose(gwf.disv.top.array, gwf.modelgrid.top):
        raise AssertionError("VertexGrid failed dynamic update test")

    # disu
    model_ws = os.path.join(base_dir, "test006_gwf3")
    sim = flopy.mf6.MFSimulation.load(sim_ws=model_ws)
    gwf = sim.get_model("gwf_1")

    mg = gwf.modelgrid
    gwf.disu.top = 101

    if not np.allclose(gwf.disu.top.array, gwf.modelgrid.top):
        raise AssertionError("UnstructuredGrid failed dynamic update test")


def test_twri_mg():
    name = "twri.nam"
    ml = flopy.modflow.Modflow.load(name, model_ws=pth, check=False)
    mg = ml.modelgrid
    assert isinstance(
        mg, flopy.discretization.StructuredGrid
    ), "modelgrid is not an StructuredGrid instance"
    shape = (3, 15, 15)
    assert (
        mg.shape == shape
    ), f"modelgrid shape {mg.shape} not equal to {shape}"
    thick = mg.thick
    shape = (5, 15, 15)
    assert (
        thick.shape == shape
    ), f"thickness shape {thick.shape} not equal to {shape}"
    return


def test_mg():
    from flopy.utils import geometry

    Lx = 100.0
    Ly = 100.0
    nlay = 1
    nrow = 51
    ncol = 51
    delr = Lx / ncol
    delc = Ly / nrow
    top = 0
    botm = [-1]
    ms = flopy.modflow.Modflow()
    dis = flopy.modflow.ModflowDis(
        ms,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
    )
    bas = flopy.modflow.ModflowBas(ms, ifrefm=True)
    t = ms.modelgrid.thick

    # test instantiation of an empty basic Structured Grid
    mg = flopy.discretization.StructuredGrid(dis.delc.array, dis.delr.array)

    # test instantiation of Structured grid with offsets
    mg = flopy.discretization.StructuredGrid(
        dis.delc.array, dis.delr.array, xoff=1, yoff=1
    )

    assert abs(ms.modelgrid.extent[-1] - Ly) < 1e-3  # , txt
    ms.modelgrid.set_coord_info(xoff=111, yoff=0)
    assert ms.modelgrid.xoffset == 111
    ms.modelgrid.set_coord_info()

    xll, yll = 321.0, 123.0
    angrot = 20.0
    ms.modelgrid = flopy.discretization.StructuredGrid(
        delc=ms.dis.delc.array,
        delr=ms.dis.delr.array,
        xoff=xll,
        yoff=xll,
        angrot=angrot,
        lenuni=2,
    )

    # test that transform for arbitrary coordinates
    # is working in same as transform for model grid
    mg2 = flopy.discretization.StructuredGrid(
        delc=ms.dis.delc.array, delr=ms.dis.delr.array, lenuni=2
    )
    x = mg2.xcellcenters[0]
    y = mg2.ycellcenters[0]
    mg2.set_coord_info(xoff=xll, yoff=yll, angrot=angrot)
    xt, yt = geometry.transform(x, y, xll, yll, mg2.angrot_radians)

    assert np.sum(xt - ms.modelgrid.xcellcenters[0]) < 1e-3
    assert np.sum(yt - ms.modelgrid.ycellcenters[0]) < 1e-3

    # test inverse transform
    x0, y0 = 9.99, 2.49
    x1, y1 = geometry.transform(x0, y0, xll, yll, angrot)
    x2, y2 = geometry.transform(x1, y1, xll, yll, angrot, inverse=True)
    assert np.abs(x2 - x0) < 1e-6
    assert np.abs(y2 - y0) < 1e6

    ms.start_datetime = "1-1-2016"
    assert ms.start_datetime == "1-1-2016"
    assert ms.dis.start_datetime == "1-1-2016"

    ms.model_ws = tpth

    ms.write_input()
    ms1 = flopy.modflow.Modflow.load(ms.namefile, model_ws=ms.model_ws)

    assert str(ms1.modelgrid) == str(ms.modelgrid)
    assert ms1.start_datetime == ms.start_datetime
    assert ms1.modelgrid.lenuni == ms.modelgrid.lenuni


def test_epsgs():
    import flopy.export.shapefile_utils as shp

    # test setting a geographic (lat/lon) coordinate reference
    # (also tests shapefile_utils.CRS parsing of geographic crs info)
    delr = np.ones(10)
    delc = np.ones(10)
    mg = flopy.discretization.StructuredGrid(delr=delr, delc=delc)

    mg.epsg = 102733
    assert mg.epsg == 102733, f"mg.epsg is not 102733 ({mg.epsg})"

    t_value = mg.__repr__()
    if not "proj4_str:epsg:102733" in t_value:
        raise AssertionError(
            f"proj4_str:epsg:102733 not in mg.__repr__(): ({t_value})"
        )

    mg.epsg = 4326  # WGS 84
    crs = shp.CRS(epsg=4326)
    if crs.grid_mapping_attribs is not None:
        assert crs.crs["proj"] == "longlat"
        t_value = crs.grid_mapping_attribs["grid_mapping_name"]
        assert (
            t_value == "latitude_longitude"
        ), f"grid_mapping_name is not latitude_longitude: {t_value}"

    t_value = mg.__repr__()
    if not "proj4_str:epsg:4326" in t_value:
        raise AssertionError(
            f"proj4_str:epsg:4326 not in sr.__repr__(): ({t_value})"
        )


def test_dynamic_xll_yll():
    nlay, nrow, ncol = 1, 10, 5
    delr, delc = 250, 500
    xll, yll = 286.80, 29.03
    # test scaling of length units
    ms2 = flopy.modflow.Modflow()
    dis = flopy.modflow.ModflowDis(
        ms2, nlay=nlay, nrow=nrow, ncol=ncol, delr=delr, delc=delc
    )

    ms2.modelgrid.set_coord_info(xoff=xll, yoff=yll, angrot=30.0)
    xll1, yll1 = ms2.modelgrid.xoffset, ms2.modelgrid.yoffset

    assert xll1 == xll, f"modelgrid.xoffset ({xll1}) is not equal to {xll}"
    assert yll1 == yll, f"modelgrid.yoffset ({yll1}) is not equal to {yll}"

    # check that xll, yll are being recomputed
    xll += 10.0
    yll += 21.0
    ms2.modelgrid.set_coord_info(xoff=xll, yoff=yll, angrot=30.0)
    xll1, yll1 = ms2.modelgrid.xoffset, ms2.modelgrid.yoffset

    assert xll1 == xll, f"modelgrid.xoffset ({xll1}) is not equal to {xll}"
    assert yll1 == yll, f"modelgrid.yoffset ({yll1}) is not equal to {yll}"


def test_namfile_readwrite():
    nlay, nrow, ncol = 1, 30, 5
    delr, delc = 250, 500
    xll, yll = 272300, 5086000
    fm = flopy.modflow
    m = fm.Modflow(modelname="junk", model_ws=os.path.join("temp", "t007"))
    dis = fm.ModflowDis(
        m, nlay=nlay, nrow=nrow, ncol=ncol, delr=delr, delc=delc
    )
    m.modelgrid = flopy.discretization.StructuredGrid(
        delc=m.dis.delc.array,
        delr=m.dis.delr.array,
        top=m.dis.top.array,
        botm=m.dis.botm.array,
        # lenuni=3,
        # length_multiplier=.3048,
        xoff=xll,
        yoff=yll,
        angrot=30,
    )

    # test reading and writing of SR information to namfile
    m.write_input()
    m2 = fm.Modflow.load("junk.nam", model_ws=os.path.join("temp", "t007"))

    t_value = abs(m2.modelgrid.xoffset - xll)
    msg = f"m2.modelgrid.xoffset ({m2.modelgrid.xoffset}) does not equal {xll}"
    assert t_value < 1e-2, msg

    t_value = abs(m2.modelgrid.yoffset - yll)
    msg = f"m2.modelgrid.yoffset ({m2.modelgrid.yoffset}) does not equal {yll}"
    assert t_value < 1e-2

    msg = f"m2.modelgrid.angrot ({m2.modelgrid.angrot}) does not equal 30"
    assert m2.modelgrid.angrot == 30, msg

    model_ws = os.path.join(
        "..", "examples", "data", "freyberg_multilayer_transient"
    )
    ml = flopy.modflow.Modflow.load(
        "freyberg.nam",
        model_ws=model_ws,
        verbose=False,
        check=False,
        exe_name="mfnwt",
    )

    assert ml.modelgrid.xoffset == ml.modelgrid._xul_to_xll(619653)
    assert ml.modelgrid.yoffset == ml.modelgrid._yul_to_yll(3353277)
    assert ml.modelgrid.angrot == 15.0


def test_read_usgs_model_reference():
    nlay, nrow, ncol = 1, 30, 5
    delr, delc = 250, 500
    # xll, yll = 272300, 5086000
    model_ws = os.path.join("temp", "t007")
    mrf = os.path.join(model_ws, "usgs.model.reference")
    shutil.copy("../examples/data/usgs.model.reference", mrf)

    xul, yul = 0, 0
    with open(mrf) as foo:
        for line in foo:
            if "xul" in line.lower():
                xul = float(line.strip().split()[1])
            elif "yul" in line.lower():
                yul = float(line.strip().split()[1])
            else:
                continue

    fm = flopy.modflow
    m = fm.Modflow(modelname="junk", model_ws=model_ws)
    # feet and days
    dis = fm.ModflowDis(
        m,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        lenuni=1,
        itmuni=4,
    )
    m.write_input()

    # test reading of SR information from usgs.model.reference
    m2 = fm.Modflow.load("junk.nam", model_ws=os.path.join("temp", "t007"))
    from flopy.discretization import StructuredGrid

    mg = StructuredGrid(delr=dis.delr.array, delc=dis.delc.array)
    mg.read_usgs_model_reference_file(mrf)
    m2.modelgrid = mg

    if abs(mg.xvertices[0, 0] - xul) > 0.01:
        raise AssertionError()
    if abs(mg.yvertices[0, 0] - yul) > 0.01:
        raise AssertionError

    assert m2.modelgrid.xoffset == mg.xoffset
    assert m2.modelgrid.yoffset == mg.yoffset
    assert m2.modelgrid.angrot == mg.angrot
    assert m2.modelgrid.epsg == mg.epsg

    # test reading non-default units from usgs.model.reference
    shutil.copy(mrf, f"{mrf}_copy")
    with open(f"{mrf}_copy") as src:
        with open(mrf, "w") as dst:
            for line in src:
                if "epsg" in line:
                    line = line.replace("102733", "4326")
                dst.write(line)

    m2 = fm.Modflow.load("junk.nam", model_ws=os.path.join("temp", "t007"))
    m2.modelgrid.read_usgs_model_reference_file(mrf)

    assert m2.modelgrid.epsg == 4326
    # have to delete this, otherwise it will mess up other tests
    to_del = glob.glob(f"{mrf}*")
    for f in to_del:
        if os.path.exists(f):
            os.remove(os.path.join(f))
    assert True


def test_rotation():
    m = flopy.modflow.Modflow(rotation=20.0)
    dis = flopy.modflow.ModflowDis(
        m, nlay=1, nrow=40, ncol=20, delr=250.0, delc=250.0, top=10, botm=0
    )
    xul, yul = 500000, 2934000
    mg = flopy.discretization.StructuredGrid(
        delc=m.dis.delc.array, delr=m.dis.delr.array
    )
    mg._angrot = 45.0
    mg.set_coord_info(mg._xul_to_xll(xul), mg._yul_to_yll(yul), angrot=45.0)

    xll, yll = mg.xoffset, mg.yoffset
    assert np.abs(mg.xvertices[0, 0] - xul) < 1e-4
    assert np.abs(mg.yvertices[0, 0] - yul) < 1e-4

    mg2 = flopy.discretization.StructuredGrid(
        delc=m.dis.delc.array, delr=m.dis.delr.array
    )
    mg2._angrot = -45.0
    mg2.set_coord_info(
        mg2._xul_to_xll(xul), mg2._yul_to_yll(yul), angrot=-45.0
    )

    xll2, yll2 = mg2.xoffset, mg2.yoffset
    assert np.abs(mg2.xvertices[0, 0] - xul) < 1e-4
    assert np.abs(mg2.yvertices[0, 0] - yul) < 1e-4

    mg3 = flopy.discretization.StructuredGrid(
        delc=m.dis.delc.array,
        delr=m.dis.delr.array,
        xoff=xll2,
        yoff=yll2,
        angrot=-45.0,
    )

    assert np.abs(mg3.xvertices[0, 0] - xul) < 1e-4
    assert np.abs(mg3.yvertices[0, 0] - yul) < 1e-4

    mg4 = flopy.discretization.StructuredGrid(
        delc=m.dis.delc.array,
        delr=m.dis.delr.array,
        xoff=xll,
        yoff=yll,
        angrot=45.0,
    )

    assert np.abs(mg4.xvertices[0, 0] - xul) < 1e-4
    assert np.abs(mg4.yvertices[0, 0] - yul) < 1e-4


def test_modelgrid_with_PlotMapView():
    import matplotlib.pyplot as plt

    m = flopy.modflow.Modflow(rotation=20.0)
    dis = flopy.modflow.ModflowDis(
        m, nlay=1, nrow=40, ncol=20, delr=250.0, delc=250.0, top=10, botm=0
    )
    # transformation assigned by arguments
    xll, yll, rotation = 500000.0, 2934000.0, 45.0

    def check_vertices():
        xllp, yllp = pc._paths[0].vertices[0]
        assert np.abs(xllp - xll) < 1e-6
        assert np.abs(yllp - yll) < 1e-6

    m.modelgrid.set_coord_info(xoff=xll, yoff=yll, angrot=rotation)
    modelmap = flopy.plot.PlotMapView(model=m)
    pc = modelmap.plot_grid()
    check_vertices()
    plt.close()

    modelmap = flopy.plot.PlotMapView(modelgrid=m.modelgrid)
    pc = modelmap.plot_grid()
    check_vertices()
    plt.close()

    mf = flopy.modflow.Modflow()

    # Model domain and grid definition
    dis = flopy.modflow.ModflowDis(
        mf,
        nlay=1,
        nrow=10,
        ncol=20,
        delr=1.0,
        delc=1.0,
    )
    xul, yul = 100.0, 210.0
    mg = mf.modelgrid
    mf.modelgrid.set_coord_info(
        xoff=mg._xul_to_xll(xul, 0.0), yoff=mg._yul_to_yll(yul, 0.0)
    )
    verts = [[101.0, 201.0], [119.0, 209.0]]
    modelxsect = flopy.plot.PlotCrossSection(model=mf, line={"line": verts})
    patchcollection = modelxsect.plot_grid()
    plt.close()


def test_mapview_plot_bc():
    from matplotlib.collections import QuadMesh, PathCollection
    import matplotlib.pyplot as plt

    sim_name = "mfsim.nam"
    sim_path = os.path.join(
        "..", "examples", "data", "mf6", "test003_gwfs_disv"
    )
    sim = flopy.mf6.MFSimulation.load(sim_name=sim_name, sim_ws=sim_path)
    ml6 = sim.get_model("gwf_1")
    ml6.modelgrid.set_coord_info(angrot=-14)
    mapview = flopy.plot.PlotMapView(model=ml6)
    mapview.plot_bc("CHD")
    ax = mapview.ax

    if len(ax.collections) == 0:
        raise AssertionError("Boundary condition was not drawn")

    for col in ax.collections:
        if not isinstance(col, (QuadMesh, PathCollection)):
            raise AssertionError("Unexpected collection type")
    plt.close()

    sim_name = "mfsim.nam"
    sim_path = os.path.join("..", "examples", "data", "mf6", "test045_lake2tr")
    sim = flopy.mf6.MFSimulation.load(sim_name=sim_name, sim_ws=sim_path)

    ml6 = sim.get_model("lakeex2a")
    mapview = flopy.plot.PlotMapView(model=ml6)
    mapview.plot_bc("LAK")
    mapview.plot_bc("SFR")

    ax = mapview.ax
    if len(ax.collections) == 0:
        raise AssertionError("Boundary condition was not drawn")

    for col in ax.collections:
        if not isinstance(col, (QuadMesh, PathCollection)):
            raise AssertionError("Unexpected collection type")
    plt.close()

    sim_name = "mfsim.nam"
    sim_path = os.path.join(
        "..", "examples", "data", "mf6", "test006_2models_mvr"
    )
    sim = flopy.mf6.MFSimulation.load(sim_name=sim_name, sim_ws=sim_path)

    ml6 = sim.get_model("parent")
    ml6c = sim.get_model("child")
    ml6c.modelgrid.set_coord_info(xoff=700, yoff=0, angrot=0)

    mapview = flopy.plot.PlotMapView(model=ml6)
    mapview.plot_bc("MAW")

    mapview2 = flopy.plot.PlotMapView(model=ml6c, ax=mapview.ax)
    mapview2.plot_bc("MAW")
    ax = mapview2.ax

    if len(ax.collections) == 0:
        raise AssertionError("Boundary condition was not drawn")

    for col in ax.collections:
        if not isinstance(col, (QuadMesh, PathCollection)):
            raise AssertionError("Unexpected collection type")
    plt.close()

    sim_name = "mfsim.nam"
    sim_path = os.path.join(
        "..", "examples", "data", "mf6", "test001e_UZF_3lay"
    )
    sim = flopy.mf6.MFSimulation.load(sim_name=sim_name, sim_ws=sim_path)
    ml6 = sim.get_model("gwf_1")

    mapview = flopy.plot.PlotMapView(model=ml6)
    mapview.plot_bc("UZF")

    if len(ax.collections) == 0:
        raise AssertionError("Boundary condition was not drawn")

    for col in ax.collections:
        if not isinstance(col, (QuadMesh, PathCollection)):
            raise AssertionError("Unexpected collection type")
    plt.close()


def test_crosssection_plot_bc():
    from matplotlib.collections import PatchCollection
    import matplotlib.pyplot as plt

    sim_name = "mfsim.nam"
    sim_path = os.path.join(
        "..", "examples", "data", "mf6", "test003_gwfs_disv"
    )
    sim = flopy.mf6.MFSimulation.load(sim_name=sim_name, sim_ws=sim_path)
    ml6 = sim.get_model("gwf_1")
    xc = flopy.plot.PlotCrossSection(ml6, line={"line": ([0, 5.5], [10, 5.5])})
    xc.plot_bc("CHD")
    ax = xc.ax

    if len(ax.collections) == 0:
        raise AssertionError("Boundary condition was not drawn")

    for col in ax.collections:
        if not isinstance(col, PatchCollection):
            raise AssertionError("Unexpected collection type")
    plt.close()

    sim_name = "mfsim.nam"
    sim_path = os.path.join("..", "examples", "data", "mf6", "test045_lake2tr")
    sim = flopy.mf6.MFSimulation.load(sim_name=sim_name, sim_ws=sim_path)

    ml6 = sim.get_model("lakeex2a")
    xc = flopy.plot.PlotCrossSection(ml6, line={"row": 10})
    xc.plot_bc("LAK")
    xc.plot_bc("SFR")

    ax = xc.ax
    if len(ax.collections) == 0:
        raise AssertionError("Boundary condition was not drawn")

    for col in ax.collections:
        if not isinstance(col, PatchCollection):
            raise AssertionError("Unexpected collection type")
    plt.close()

    sim_name = "mfsim.nam"
    sim_path = os.path.join(
        "..", "examples", "data", "mf6", "test006_2models_mvr"
    )
    sim = flopy.mf6.MFSimulation.load(sim_name=sim_name, sim_ws=sim_path)

    ml6 = sim.get_model("parent")
    xc = flopy.plot.PlotCrossSection(ml6, line={"column": 1})
    xc.plot_bc("MAW")

    ax = xc.ax
    if len(ax.collections) == 0:
        raise AssertionError("Boundary condition was not drawn")

    for col in ax.collections:
        if not isinstance(col, PatchCollection):
            raise AssertionError("Unexpected collection type")
    plt.close()

    sim_name = "mfsim.nam"
    sim_path = os.path.join(
        "..", "examples", "data", "mf6", "test001e_UZF_3lay"
    )
    sim = flopy.mf6.MFSimulation.load(sim_name=sim_name, sim_ws=sim_path)
    ml6 = sim.get_model("gwf_1")

    xc = flopy.plot.PlotCrossSection(ml6, line={"row": 0})
    xc.plot_bc("UZF")

    ax = xc.ax
    if len(ax.collections) == 0:
        raise AssertionError("Boundary condition was not drawn")

    for col in ax.collections:
        if not isinstance(col, PatchCollection):
            raise AssertionError("Unexpected collection type")
    plt.close()


def test_tricontour_NaN():
    from flopy.plot import PlotMapView
    import numpy as np
    from flopy.discretization import StructuredGrid
    import matplotlib.pyplot as plt

    arr = np.random.rand(10, 10) * 100
    arr[-1, :] = np.nan
    delc = np.array([10] * 10, dtype=float)
    delr = np.array([8] * 10, dtype=float)
    top = np.ones((10, 10), dtype=float)
    botm = np.ones((3, 10, 10), dtype=float)
    botm[0] = 0.75
    botm[1] = 0.5
    botm[2] = 0.25
    idomain = np.ones((3, 10, 10))
    idomain[0, 0, :] = 0
    vmin = np.nanmin(arr)
    vmax = np.nanmax(arr)
    levels = np.linspace(vmin, vmax, 7)

    grid = StructuredGrid(
        delc=delc,
        delr=delr,
        top=top,
        botm=botm,
        idomain=idomain,
        lenuni=1,
        nlay=3,
        nrow=10,
        ncol=10,
    )

    pmv = PlotMapView(modelgrid=grid, layer=0)
    contours = pmv.contour_array(a=arr)

    for ix, lev in enumerate(contours.levels):
        if not np.allclose(lev, levels[ix]):
            raise AssertionError("TriContour NaN catch Failed")

    plt.close()


def test_get_vertices():
    from flopy.discretization import StructuredGrid

    m = flopy.modflow.Modflow(rotation=20.0)
    nrow, ncol = 40, 20
    dis = flopy.modflow.ModflowDis(
        m, nlay=1, nrow=nrow, ncol=ncol, delr=250.0, delc=250.0, top=10, botm=0
    )
    mmg = m.modelgrid
    xul, yul = 500000, 2934000
    mg = StructuredGrid(
        delc=m.dis.delc.array,
        delr=m.dis.delr.array,
        xoff=mmg._xul_to_xll(xul, 45.0),
        yoff=mmg._yul_to_yll(xul, 45.0),
        angrot=45.0,
    )

    xgrid = mg.xvertices
    ygrid = mg.yvertices
    # a1 = np.array(mg.xyvertices)
    a1 = np.array(
        [
            [xgrid[0, 0], ygrid[0, 0]],
            [xgrid[0, 1], ygrid[0, 1]],
            [xgrid[1, 1], ygrid[1, 1]],
            [xgrid[1, 0], ygrid[1, 0]],
        ]
    )

    a2 = np.array(mg.get_cell_vertices(0, 0))
    assert np.array_equal(a1, a2)


def test_get_lrc_get_node():
    nlay, nrow, ncol = 3, 4, 5
    ml = flopy.modflow.Modflow()
    dis = flopy.modflow.ModflowDis(
        ml, nlay=nlay, nrow=nrow, ncol=ncol, top=50, botm=[0, -1, -2]
    )
    nodes = list(range(nlay * nrow * ncol))
    indices = np.indices((nlay, nrow, ncol))
    layers = indices[0].flatten()
    rows = indices[1].flatten()
    cols = indices[2].flatten()
    for node, (l, r, c) in enumerate(zip(layers, rows, cols)):
        # ensure get_lrc returns zero-based layer row col
        lrc = dis.get_lrc(node)[0]
        assert lrc == (
            l,
            r,
            c,
        ), f"get_lrc() returned {lrc}, expecting {l, r, c}"
        # ensure get_node returns zero-based node number
        n = dis.get_node((l, r, c))[0]
        assert node == n, f"get_node() returned {n}, expecting {node}"
    return


def test_vertex_model_dot_plot():
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams["figure.max_open_warning"] = 36
    # load up the vertex example problem
    sim_name = "mfsim.nam"
    sim_path = "../examples/data/mf6/test003_gwftri_disv"
    disv_sim = flopy.mf6.MFSimulation.load(
        sim_name=sim_name, version="mf6", exe_name="mf6", sim_ws=sim_path
    )
    disv_ml = disv_sim.get_model("gwf_1")
    ax = disv_ml.plot()
    assert isinstance(ax, list)
    assert len(ax) == 36
    plt.close("all")


def test_model_dot_plot():
    import matplotlib.pyplot as plt

    loadpth = os.path.join("..", "examples", "data", "mf2005_test")
    ml = flopy.modflow.Modflow.load(
        "ibs2k.nam", "mf2k", model_ws=loadpth, check=False
    )
    ax = ml.plot()
    assert isinstance(ax, list), "ml.plot() ax is is not a list"
    assert len(ax) == 18, f"number of axes ({len(ax)}) is not equal to 18"
    plt.close("all")

    # plot specific dataset
    ax = ml.bcf6.hy.plot()
    assert isinstance(ax, list), "ml.bcf6.hy.plot() ax is is not a list"
    assert len(ax) == 2, f"number of hy axes ({len(ax)}) is not equal to 2"

    # special case where nlay != plottable
    ax = ml.bcf6.vcont.plot()
    assert isinstance(
        ax, plt.Axes
    ), "ml.bcf6.vcont.plot() ax is is not of type plt.Axes"
    plt.close("all")


def test_get_rc_from_node_coordinates():
    m = flopy.modflow.Modflow(rotation=20.0)
    nrow, ncol = 10, 10
    dis = flopy.modflow.ModflowDis(
        m, nlay=1, nrow=nrow, ncol=ncol, delr=100.0, delc=100.0, top=10, botm=0
    )
    r, c = m.dis.get_rc_from_node_coordinates([50.0, 110.0], [50.0, 220.0])
    assert np.array_equal(r, np.array([9, 7]))
    assert np.array_equal(c, np.array([0, 1]))

    # test variable delr and delc spacing
    mf = flopy.modflow.Modflow()
    delc = [0.5] * 5 + [2.0] * 5
    delr = [0.5] * 5 + [2.0] * 5
    nrow = 10
    ncol = 10
    mfdis = flopy.modflow.ModflowDis(
        mf, nrow=nrow, ncol=ncol, delr=delr, delc=delc
    )  # , xul=50, yul=1000)
    ygrid, xgrid, zgrid = mfdis.get_node_coordinates()
    for i in range(nrow):
        for j in range(ncol):
            x = xgrid[j]
            y = ygrid[i]
            r, c = mfdis.get_rc_from_node_coordinates(x, y)
            assert r == i, f"row {r} not equal {i} for xy ({x}, {y})"
            assert c == j, f"col {c} not equal {j} for xy ({x}, {y})"


def test_netcdf_classmethods():

    # Do not fail if netCDF4 not installed
    try:
        import netCDF4
        import pyproj
    except:
        return

    nam_file = "freyberg.nam"
    model_ws = os.path.join(
        "..", "examples", "data", "freyberg_multilayer_transient"
    )
    ml = flopy.modflow.Modflow.load(
        nam_file, model_ws=model_ws, check=False, verbose=True, load_only=[]
    )

    f = ml.export(os.path.join(npth, "freyberg.nc"))
    v1_set = set(f.nc.variables.keys())
    fnc = os.path.join(npth, "freyberg.new.nc")
    new_f = flopy.export.NetCdf.zeros_like(f, output_filename=fnc)
    v2_set = set(new_f.nc.variables.keys())
    diff = v1_set.symmetric_difference(v2_set)
    assert len(diff) == 0, str(diff)


def test_wkt_parse():
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

    prjs = glob.glob("../examples/data/prj_test/*")
    for prj in prjs:
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


def test_shapefile_ibound():
    shapefile = import_shapefile()
    if not shapefile:
        return

    shape_name = os.path.join(spth, "test.shp")
    nam_file = "freyberg.nam"
    model_ws = os.path.join(
        "..", "examples", "data", "freyberg_multilayer_transient"
    )
    ml = flopy.modflow.Modflow.load(
        nam_file,
        model_ws=model_ws,
        check=False,
        verbose=True,
        load_only=["bas6"],
    )
    ml.export(shape_name)
    shp = shapefile.Reader(shape_name)
    field_names = [item[0] for item in shp.fields][1:]
    ib_idx = field_names.index("ibound_1")
    txt = f"should be int instead of {type(shp.record(0)[ib_idx])}"
    assert type(shp.record(0)[ib_idx]) == int, txt


def test_shapefile():
    for namfile in namfiles:
        yield export_shapefile, namfile
    return


def test_shapefile_export_modelgrid_override():
    for namfile in namfiles[0:2]:
        yield export_shapefile_modelgrid_override, namfile
    return


def test_netcdf():
    for namfile in namfiles:
        yield export_mf2005_netcdf, namfile
    return


def build_netcdf():
    for namfile in namfiles:
        export_mf2005_netcdf(namfile)
    return


def build_sfr_netcdf():
    namfile = "testsfr2.nam"
    export_mf2005_netcdf(namfile)
    return


def test_export_array2():
    if import_shapefile() is None:
        return
    from flopy.discretization import StructuredGrid
    from flopy.export.utils import export_array

    nrow = 7
    ncol = 11
    epsg = 4111

    # no epsg code
    modelgrid = StructuredGrid(
        delr=np.ones(ncol) * 1.1, delc=np.ones(nrow) * 1.1
    )
    filename = os.path.join(spth, "myarray1.shp")
    a = np.arange(nrow * ncol).reshape((nrow, ncol))
    export_array(modelgrid, filename, a)
    assert os.path.isfile(filename), "did not create array shapefile"

    # with modelgrid epsg code
    modelgrid = StructuredGrid(
        delr=np.ones(ncol) * 1.1, delc=np.ones(nrow) * 1.1, epsg=epsg
    )
    filename = os.path.join(spth, "myarray2.shp")
    a = np.arange(nrow * ncol).reshape((nrow, ncol))
    export_array(modelgrid, filename, a)
    assert os.path.isfile(filename), "did not create array shapefile"

    # with passing in epsg code
    modelgrid = StructuredGrid(
        delr=np.ones(ncol) * 1.1, delc=np.ones(nrow) * 1.1
    )
    filename = os.path.join(spth, "myarray3.shp")
    a = np.arange(nrow * ncol).reshape((nrow, ncol))
    export_array(modelgrid, filename, a, epsg=epsg)
    assert os.path.isfile(filename), "did not create array shapefile"
    return


def test_export_array_contours():
    if import_shapefile() is None:
        return
    from flopy.discretization import StructuredGrid
    from flopy.export.utils import export_array_contours

    nrow = 7
    ncol = 11
    epsg = 4111

    # no epsg code
    modelgrid = StructuredGrid(
        delr=np.ones(ncol) * 1.1, delc=np.ones(nrow) * 1.1
    )
    filename = os.path.join(spth, "myarraycontours1.shp")
    a = np.arange(nrow * ncol).reshape((nrow, ncol))
    export_array_contours(modelgrid, filename, a)
    assert os.path.isfile(filename), "did not create contour shapefile"

    # with modelgrid epsg code
    modelgrid = StructuredGrid(
        delr=np.ones(ncol) * 1.1, delc=np.ones(nrow) * 1.1, epsg=epsg
    )
    filename = os.path.join(spth, "myarraycontours2.shp")
    a = np.arange(nrow * ncol).reshape((nrow, ncol))
    export_array_contours(modelgrid, filename, a)
    assert os.path.isfile(filename), "did not create contour shapefile"

    # with passing in epsg code
    modelgrid = StructuredGrid(
        delr=np.ones(ncol) * 1.1, delc=np.ones(nrow) * 1.1
    )
    filename = os.path.join(spth, "myarraycontours3.shp")
    a = np.arange(nrow * ncol).reshape((nrow, ncol))
    export_array_contours(modelgrid, filename, a, epsg=epsg)
    assert os.path.isfile(filename), "did not create contour shapefile"
    return


def test_export_contourf():
    try:
        import shapely
    except:
        return
    if import_shapefile() is None:
        return
    import matplotlib.pyplot as plt
    from flopy.export.utils import export_contourf

    filename = os.path.join(spth, "myfilledcontours.shp")
    a = np.random.random((10, 10))
    cs = plt.contourf(a)
    export_contourf(filename, cs)
    assert os.path.isfile(filename), "did not create contourf shapefile"
    plt.close()
    return


def main():
    # test_shapefile()
    # test_shapefile_ibound()
    # test_netcdf_classmethods()

    # for namfile in namfiles:
    #    export_mf2005_netcdf(namfile)
    #    export_shapefile(namfile)

    # for namfile in namfiles[0:2]:
    #     export_shapefile_modelgrid_override(namfile)

    # test_netcdf_overloads()
    # test_netcdf_classmethods()
    # build_netcdf()
    # build_sfr_netcdf()
    # test_twri_mg()
    # test_mg()
    # test_mbase_modelgrid()
    # test_mt_modelgrid()
    # test_rotation()
    # test_model_dot_plot()
    # test_get_lrc_get_node()
    # test_vertex_model_dot_plot()
    # test_sr_with_Map()
    # test_modelgrid_with_PlotMapView()
    test_epsgs()
    # test_sr_scaling()
    # test_read_usgs_model_reference()
    # test_dynamic_xll_yll()
    # test_namfile_readwrite()
    # test_free_format_flag()
    # test_get_vertices()
    # test_export_output()
    # for namfile in namfiles:
    test_freyberg_export()
    # test_export_array()
    # test_write_shapefile()
    # test_wkt_parse()
    # test_get_rc_from_node_coordinates()
    # test_export_array()
    # test_export_array_contours()
    # test_tricontour_NaN()
    # test_export_contourf()
    test_sr()
    # test_mf6_modelgrid_update()
    # test_shapefile_polygon_closed()
    # test_mapview_plot_bc()
    # test_crosssection_plot_bc()
    # test_output_helper_shapefile_export()


if __name__ == "__main__":

    main()
