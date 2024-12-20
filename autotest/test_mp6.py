import os
import shutil
from pprint import pformat
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
from modflow_devtools.markers import requires_exe, requires_pkg
from numpy.lib.recfunctions import repack_fields

import flopy
from autotest.conftest import get_example_data_path
from flopy.discretization import StructuredGrid
from flopy.export.shapefile_utils import shp2recarray
from flopy.modflow import Modflow, ModflowMnw2
from flopy.modpath import Modpath6, Modpath6Bas
from flopy.modpath.mp6sim import Modpath6Sim, StartingLocationsFile
from flopy.plot import PlotMapView
from flopy.utils import EndpointFile, PathlineFile, TimeseriesFile
from flopy.utils.flopy_io import loadtxt

pytestmark = pytest.mark.mf6


@pytest.fixture
def mp6_test_path(example_data_path):
    return example_data_path / "mp6"


def copy_modpath_files(source, model_ws, baseName):
    files = [
        file
        for file in os.listdir(source)
        if file.startswith(baseName) and os.path.isfile(source / file)
    ]
    for file in files:
        src = get_example_data_path() / "mp6" / file
        dst = model_ws / file
        print(f"copying {src} -> {dst}")
        shutil.copy(src, dst)


def test_mpsim(function_tmpdir, mp6_test_path):
    copy_modpath_files(mp6_test_path, function_tmpdir, "EXAMPLE.")

    m = Modflow.load("EXAMPLE.nam", model_ws=function_tmpdir)
    m.get_package_list()

    mp = Modpath6(
        modelname="ex6",
        exe_name="mp6",
        modflowmodel=m,
        model_ws=function_tmpdir,
        dis_file=f"{m.name}.dis",
        head_file=f"{m.name}.hed",
        budget_file=f"{m.name}.bud",
    )

    mpb = Modpath6Bas(mp, hdry=m.lpf.hdry, laytyp=m.lpf.laytyp, ibound=1, prsity=0.1)

    sim = mp.create_mpsim(trackdir="forward", simtype="endpoint", packages="RCH")
    mp.write_input()

    # replace the well with an mnw
    node_data = np.array(
        [
            (3, 12, 12, "well1", "skin", -1, 0, 0, 0, 1.0, 2.0, 5.0, 6.2),
            (4, 12, 12, "well1", "skin", -1, 0, 0, 0, 0.5, 2.0, 5.0, 6.2),
        ],
        dtype=[
            ("k", int),
            ("i", int),
            ("j", int),
            ("wellid", object),
            ("losstype", object),
            ("pumploc", int),
            ("qlimit", int),
            ("ppflag", int),
            ("pumpcap", int),
            ("rw", float),
            ("rskin", float),
            ("kskin", float),
            ("zpump", float),
        ],
    ).view(np.recarray)

    stress_period_data = {
        0: np.array(
            [(0, "well1", -150000.0)],
            dtype=[("per", int), ("wellid", object), ("qdes", float)],
        )
    }
    m.remove_package("WEL")
    mnw2 = ModflowMnw2(
        model=m,
        mnwmax=1,
        node_data=node_data,
        stress_period_data=stress_period_data,
        itmp=[1, -1, -1],
    )
    # test creation of modpath simulation file for MNW2
    # (not a very robust test)
    sim = mp.create_mpsim(trackdir="backward", simtype="pathline", packages="MNW2")
    mp.write_input()

    # test StartingLocationsFile._write_wo_pandas
    for use_pandas in [True, False]:
        sim = Modpath6Sim(model=mp)
        # starting locations file
        stl = StartingLocationsFile(model=mp, use_pandas=use_pandas)
        stldata = StartingLocationsFile.get_empty_starting_locations_data(npt=2)
        stldata["label"] = ["p1", "p2"]
        stldata[1]["i0"] = 5
        stldata[1]["j0"] = 6
        stldata[1]["xloc0"] = 0.1
        stldata[1]["yloc0"] = 0.2
        stl.data = stldata
        mp.write_input()
        stllines = open(function_tmpdir / "ex6.loc").readlines()
        assert stllines[3].strip() == "group1"
        assert int(stllines[4].strip()) == 2
        assert stllines[6].strip().split()[-1] == "p2"


@requires_pkg("pyshp", "shapely", name_map={"pyshp": "shapefile"})
def test_get_destination_data(function_tmpdir, mp6_test_path):
    copy_modpath_files(mp6_test_path, function_tmpdir, "EXAMPLE.")
    copy_modpath_files(mp6_test_path, function_tmpdir, "EXAMPLE-3.")

    m = Modflow.load("EXAMPLE.nam", model_ws=function_tmpdir)

    mg1 = m.modelgrid
    mg1.set_coord_info(
        xoff=mg1._xul_to_xll(0.0, 30.0),
        yoff=mg1._yul_to_yll(0.0, 30.0),
        angrot=30.0,
    )

    mg = StructuredGrid(delc=m.dis.delc.array, delr=m.dis.delr.array)
    mg.set_coord_info(
        xoff=mg._xul_to_xll(1000.0, 30.0),
        yoff=mg._yul_to_yll(1000.0, 30.0),
        angrot=30.0,
    )

    # test deprecation
    m.dis.export(function_tmpdir / "dis.shp")

    pthld = PathlineFile(function_tmpdir / "EXAMPLE-3.pathline")
    epd = EndpointFile(function_tmpdir / "EXAMPLE-3.endpoint")

    well_epd = epd.get_destination_endpoint_data(dest_cells=[(4, 12, 12)])
    well_pthld = pthld.get_destination_pathline_data(
        dest_cells=[(4, 12, 12)], to_recarray=True
    )

    # same particle IDs should be in both endpoint data and pathline data
    tval = len(set(well_epd.particleid).difference(set(well_pthld.particleid)))
    msg = "same particle IDs should be in both endpoint data and pathline data"
    assert tval == 0, msg

    # check that all starting locations are included in the pathline data
    # (pathline data slice not just endpoints)
    starting_locs = repack_fields(well_epd[["k0", "i0", "j0"]])
    pathline_locs = np.array(
        np.array(well_pthld)[["k", "i", "j"]].tolist(),
        dtype=starting_locs.dtype,
    )
    assert np.all(np.isin(starting_locs, pathline_locs))

    # test writing a shapefile of endpoints
    epd.write_shapefile(
        well_epd,
        direction="starting",
        shpname=function_tmpdir / "starting_locs.shp",
        mg=m.modelgrid,
    )

    # test writing shapefile of pathlines
    fpth = function_tmpdir / "pathlines_1per.shp"
    pthld.write_shapefile(
        well_pthld,
        one_per_particle=True,
        direction="starting",
        mg=m.modelgrid,
        shpname=fpth,
    )
    fpth = function_tmpdir / "pathlines_1per_end.shp"
    pthld.write_shapefile(
        well_pthld,
        one_per_particle=True,
        direction="ending",
        mg=m.modelgrid,
        shpname=fpth,
    )
    # test writing shapefile of pathlines
    fpth = function_tmpdir / "pathlines_1per2.shp"
    pthld.write_shapefile(
        well_pthld,
        one_per_particle=True,
        direction="starting",
        mg=mg,
        shpname=fpth,
    )
    # test writing shapefile of pathlines
    fpth = function_tmpdir / "pathlines_1per2_ll.shp"
    pthld.write_shapefile(
        well_pthld,
        one_per_particle=True,
        direction="starting",
        mg=mg,
        shpname=fpth,
    )
    fpth = function_tmpdir / "pathlines.shp"
    pthld.write_shapefile(
        well_pthld, one_per_particle=False, mg=m.modelgrid, shpname=fpth
    )

    # test that endpoints were rotated and written correctly
    ra = shp2recarray(function_tmpdir / "starting_locs.shp")
    p3 = ra.geometry[ra.particleid == 4][0]
    xorig, yorig = m.modelgrid.get_coords(well_epd.x0[0], well_epd.y0[0])
    assert p3.x - xorig + p3.y - yorig < 1e-4
    xorig, yorig = mg1.xcellcenters[3, 4], mg1.ycellcenters[3, 4]
    assert np.abs(p3.x - xorig + p3.y - yorig) < 1e-4  # this also checks for 1-based

    # test that particle attribute information is consistent with pathline file
    ra = shp2recarray(function_tmpdir / "pathlines.shp")
    inds = (ra.particleid == 8) & (ra.i == 12) & (ra.j == 12)
    assert ra.time[inds][0] - 20181.7 < 0.1
    assert ra.xloc[inds][0] - 0.933 < 0.01

    # test that k, i, j are correct for single geometry pathlines, forwards
    # and backwards
    ra = shp2recarray(function_tmpdir / "pathlines_1per.shp")
    assert ra.i[0] == 4, ra.j[0] == 5
    ra = shp2recarray(function_tmpdir / "pathlines_1per_end.shp")
    assert ra.i[0] == 13, ra.j[0] == 13

    # test use of arbitrary spatial reference and offset
    mg1.set_coord_info(
        xoff=mg.xoffset,
        yoff=mg.yoffset,
        angrot=mg.angrot,
        crs=mg.epsg,
    )
    ra = shp2recarray(function_tmpdir / "pathlines_1per2.shp")
    p3_2 = ra.geometry[ra.particleid == 4][0]
    test1 = mg1.xcellcenters[3, 4]
    test2 = mg1.ycellcenters[3, 4]
    assert (
        np.abs(p3_2.x[0] - mg1.xcellcenters[3, 4] + p3_2.y[0] - mg1.ycellcenters[3, 4])
        < 1e-4
    )

    # arbitrary spatial reference with ll specified instead of ul
    ra = shp2recarray(function_tmpdir / "pathlines_1per2_ll.shp")
    p3_2 = ra.geometry[ra.particleid == 4][0]
    mg.set_coord_info(xoff=mg.xoffset, yoff=mg.yoffset, angrot=30.0)
    assert (
        np.abs(p3_2.x[0] - mg.xcellcenters[3, 4] + p3_2.y[0] - mg.ycellcenters[3, 4])
        < 1e-4
    )

    xul = 3628793
    yul = 21940389

    m = Modflow.load("EXAMPLE.nam", model_ws=function_tmpdir)

    mg4 = m.modelgrid
    mg4.set_coord_info(
        xoff=mg4._xul_to_xll(xul, 0.0),
        yoff=mg4._yul_to_yll(yul, 0.0),
        angrot=0.0,
        crs=mg4.epsg,
    )

    fpth = function_tmpdir / "dis2.shp"
    m.dis.export(fpth)
    pthobj = PathlineFile(function_tmpdir / "EXAMPLE-3.pathline")
    fpth = function_tmpdir / "pathlines_1per3.shp"
    pthobj.write_shapefile(shpname=fpth, direction="ending", mg=mg4)


def test_loadtxt(function_tmpdir, mp6_test_path):
    copy_modpath_files(mp6_test_path, function_tmpdir, "EXAMPLE-3.")

    pthfile = function_tmpdir / "EXAMPLE-3.pathline"
    pthld = PathlineFile(pthfile)
    ra = loadtxt(pthfile, delimiter=" ", skiprows=3, dtype=pthld.dtype)
    ra2 = loadtxt(
        pthfile, delimiter=" ", skiprows=3, dtype=pthld.dtype, use_pandas=False
    )
    assert np.array_equal(ra, ra2)


@requires_exe("mf2005")
def test_modpath(function_tmpdir, example_data_path):
    pth = example_data_path / "freyberg"
    mfnam = "freyberg.nam"

    m = Modflow.load(
        mfnam,
        model_ws=pth,
        verbose=True,
        exe_name="mf2005",
        check=False,
    )
    assert m.load_fail is False

    m.change_model_ws(function_tmpdir)
    m.write_input()

    success, buff = m.run_model(silent=False)
    assert success, "modflow model run did not terminate successfully"

    # create the forward modpath file
    mpnam = "freybergmp"
    mp = Modpath6(
        mpnam,
        exe_name="mp6",
        modflowmodel=m,
        model_ws=function_tmpdir,
    )
    mpbas = Modpath6Bas(
        mp,
        hnoflo=m.bas6.hnoflo,
        hdry=m.lpf.hdry,
        ibound=m.bas6.ibound.array,
        prsity=0.2,
        prsityCB=0.2,
    )
    sim = mp.create_mpsim(trackdir="forward", simtype="endpoint", packages="RCH")

    # write forward particle track files
    mp.write_input()

    if success:
        success, buff = mp.run_model(silent=False)
        assert success, "forward modpath model run did not terminate successfully"

    mpnam = "freybergmpp"
    mpp = Modpath6(
        mpnam,
        exe_name="mp6",
        modflowmodel=m,
        model_ws=function_tmpdir,
    )
    mpbas = Modpath6Bas(
        mpp,
        hnoflo=m.bas6.hnoflo,
        hdry=m.lpf.hdry,
        ibound=m.bas6.ibound.array,
        prsity=0.2,
        prsityCB=0.2,
    )
    sim = mpp.create_mpsim(trackdir="backward", simtype="pathline", packages="WEL")

    # write backward particle track files
    mpp.write_input()

    if success:
        success, buff = mpp.run_model(silent=False)
        assert success, "backward modpath model run did not terminate successfully"

    # load modpath output files
    if success:
        endfile = function_tmpdir / mp.sim.endpoint_file
        pthfile = function_tmpdir / mpp.sim.pathline_file

        # load the endpoint data
        try:
            endobj = EndpointFile(endfile)
        except:
            assert False, "could not load endpoint file"
        ept = endobj.get_alldata()
        assert ept.shape == (695,), "shape of endpoint file is not (695,)"

        # load the pathline data
        try:
            pthobj = PathlineFile(pthfile)
        except:
            assert False, "could not load pathline file"
        plines = pthobj.get_alldata()
        assert len(plines) == 576, "there are not 576 particle pathlines in file"

        # load the modflow files for model map
        mfnam = "freyberg.nam"
        m = Modflow.load(
            mfnam,
            model_ws=function_tmpdir,
            verbose=True,
            forgive=False,
            exe_name="mf2005",
        )

        # load modpath output files
        pthfile = function_tmpdir / "freybergmpp.mppth"

        # load the pathline data
        pthobj = PathlineFile(pthfile)

        # determine version
        ver = pthobj.version
        assert ver == 6, f"{pthfile} is not a MODPATH version 6 pathline file"

        # get all pathline data
        plines = pthobj.get_alldata()

        mm = PlotMapView(model=m)
        mm.plot_pathline(plines, colors="blue", layer="all")

        # plot the grid and ibound array
        mm.plot_grid()
        mm.plot_ibound()
        fpth = function_tmpdir / "pathline.png"
        plt.savefig(fpth)
        plt.close()

        mm = PlotMapView(model=m)
        mm.plot_pathline(plines, colors="green", layer=0)

        # plot the grid and ibound array
        mm.plot_grid()
        mm.plot_ibound()

        fpth = function_tmpdir / "pathline2.png"
        plt.savefig(fpth)
        plt.close()

        mm = PlotMapView(model=m)
        mm.plot_pathline(plines, colors="red")

        # plot the grid and ibound array
        mm.plot_grid()
        mm.plot_ibound()

        fpth = function_tmpdir / "pathline3.png"
        plt.savefig(fpth)
        plt.close()


def test_mp6_timeseries_load(example_data_path):
    pth = example_data_path / "mp5"
    files = [pth / name for name in sorted(os.listdir(pth)) if ".timeseries" in name]
    for file in files:
        print(file)
        eval_timeseries(file)


def eval_timeseries(file):
    ts = TimeseriesFile(file)
    assert isinstance(
        ts, TimeseriesFile
    ), f"{os.path.basename(file)} is not an instance of TimeseriesFile"

    # get the all of the data
    tsd = ts.get_alldata()
    assert (
        len(tsd) > 0
    ), f"could not load data using get_alldata() from {os.path.basename(file)}."

    # get the data for the last particleid
    partid = ts.get_maxid()
    assert partid is not None, (
        "could not get maximum particleid using get_maxid() from "
        f"{os.path.basename(file)}."
    )

    tsd = ts.get_data(partid=partid)
    assert tsd.shape[0] > 0, (
        f"could not load data for particleid {partid} using get_data() from "
        f"{os.path.basename(file)}. Maximum partid = {ts.get_maxid()}."
    )

    timemax = ts.get_maxtime() / 2.0
    assert timemax is not None, (
        "could not get maximum time using get_maxtime() from "
        f"{os.path.basename(file)}."
    )

    tsd = ts.get_alldata(totim=timemax)
    assert len(tsd) > 0, (
        f"could not load data for totim>={timemax} using get_alldata() from "
        f"{os.path.basename(file)}. Maximum totim = {ts.get_maxtime()}."
    )

    timemax = ts.get_maxtime()
    assert timemax is not None, (
        "could not get maximum time using get_maxtime() from "
        f"{os.path.basename(file)}."
    )

    tsd = ts.get_alldata(totim=timemax, ge=False)
    assert len(tsd) > 0, (
        f"could not load data for totim<={timemax} using get_alldata() from "
        f"{os.path.basename(file)}. Maximum totim = {ts.get_maxtime()}."
    )


def get_mf2005_model(name, ws, alt=False):
    nrow = 3
    ncol = 4
    nlay = 2
    nper = 1
    l1_ibound = np.array([[[-1, -1, -1, -1], [-1, 1, 1, -1], [-1, -1, -1, -1]]])
    l2_ibound = np.ones((1, nrow, ncol))
    l2_ibound_alt = np.ones((1, nrow, ncol))
    l2_ibound_alt[0, 0, 0] = 0
    bt1 = np.ones((1, nrow, ncol)) + 5
    bt2 = np.ones((1, nrow, ncol)) + 3
    ctx = SimpleNamespace(
        nrow=nrow,
        ncol=ncol,
        nlay=nlay,
        nper=nper,
        l1_ibound=l1_ibound,
        l2_ibound=l2_ibound,
        l2_ibound_alt=l2_ibound_alt,
        ibound=np.concatenate((l1_ibound, l2_ibound_alt if alt else l2_ibound), axis=0),
        laytype=[0, 0 if alt else 1],
        hnoflow=-888,
        hdry=-777,
        top=np.zeros((1, nrow, ncol)) + 10,
        bt1=bt1,
        bt2=bt2,
        botm=np.concatenate((bt1, bt2), axis=0),
        ipakcb=740,
    )

    # create modflow model
    m = flopy.modflow.Modflow(
        modelname=name + ("alt" if alt else ""),
        namefile_ext="nam",
        version="mf2005",
        exe_name="mf2005",
        model_ws=ws,
    )

    # dis
    dis = flopy.modflow.ModflowDis(
        model=m,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        nper=nper,
        delr=1.0,
        delc=1.0,
        laycbd=0,
        top=ctx.top,
        botm=ctx.botm,
        perlen=1,
        nstp=1,
        tsmult=1,
        steady=True,
    )

    # bas
    bas = flopy.modflow.ModflowBas(
        model=m,
        ibound=ctx.ibound,
        strt=10,
        ifrefm=True,
        ixsec=False,
        ichflg=False,
        stoper=None,
        hnoflo=ctx.hnoflow,
        extension="bas",
        unitnumber=None,
        filenames=None,
    )
    # lpf
    lpf = flopy.modflow.ModflowLpf(
        model=m,
        ipakcb=ctx.ipakcb,
        laytyp=ctx.laytype,
        hk=10,
        vka=10,
        hdry=ctx.hdry,
    )

    # well
    wel = flopy.modflow.ModflowWel(
        model=m,
        ipakcb=ctx.ipakcb,
        stress_period_data={0: [[1, 1, 1, -5.0]]},
    )

    flopy.modflow.ModflowPcg(m, hclose=0.001, rclose=0.001, mxiter=150, iter1=30)

    ocspd = {}
    for p in range(nper):
        ocspd[(p, 0)] = ["save head", "save budget"]
    # pretty sure it just uses the last for everything
    ocspd[(0, 0)] = ["save head", "save budget"]
    flopy.modflow.ModflowOc(m, stress_period_data=ocspd)

    return m, ctx


@requires_exe("mf2005", "mp6")
@pytest.mark.parametrize("alt", [True, False])
def test_data_pass_no_modflow(function_tmpdir, alt):
    """
    test that user can pass and create a mp model without an accompanying modflow model
    """

    ml, ctx = get_mf2005_model("data_pass", function_tmpdir, alt)
    ml.write_input()
    success, buff = ml.run_model()
    assert success, pformat(buff)

    dis_file = f"{ml.name}.dis"
    bud_file = f"{ml.name}.cbc"
    hd_file = f"{ml.name}.hds"

    mp = flopy.modpath.Modpath6(
        modelname=ml.name,
        simfile_ext="mpsim",
        namefile_ext="mpnam",
        version="modpath",
        exe_name="mp6",
        modflowmodel=None,  # do not pass modflow model
        dis_file=dis_file,
        head_file=hd_file,
        budget_file=bud_file,
        model_ws=ml.model_ws,
        external_path=None,
        verbose=False,
        load=True,
        listunit=7,
    )

    assert mp.head_file == hd_file
    assert mp.budget_file == bud_file
    assert mp.dis_file == dis_file
    assert mp.nrow_ncol_nlay_nper == (ctx.nrow, ctx.ncol, ctx.nlay, ctx.nper)

    mpbas = flopy.modpath.Modpath6Bas(
        mp,
        hnoflo=ctx.hnoflow,
        hdry=ctx.hdry,
        def_face_ct=0,
        bud_label=None,
        def_iface=None,
        laytyp=ctx.laytype,
        ibound=ctx.ibound,
        prsity=0.30,
        prsityCB=0.30,
        extension="mpbas",
        unitnumber=86,
    )
    # test layertype is created correctly
    assert np.isclose(mpbas.laytyp.array, ctx.laytype).all()
    # test ibound is pulled from modflow model
    assert np.isclose(mpbas.ibound.array, ctx.ibound).all()

    sim = flopy.modpath.Modpath6Sim(model=mp)
    stl = flopy.modpath.mp6sim.StartingLocationsFile(model=mp)
    stldata = stl.get_empty_starting_locations_data(npt=2)
    stldata["label"] = ["p1", "p2"]
    stldata[1]["k0"] = 0
    stldata[1]["i0"] = 0
    stldata[1]["j0"] = 0
    stldata[1]["xloc0"] = 0.1
    stldata[1]["yloc0"] = 0.2
    stl.data = stldata
    mp.write_input()
    success, buff = mp.run_model()
    assert success


@requires_exe("mf2005", "mp6")
@pytest.mark.parametrize("alt", [True, False])
def test_data_pass_with_modflow(function_tmpdir, alt):
    """
    test that user specified head files etc. are preferred
    over files from the modflow model
    """

    ml, ctx = get_mf2005_model("data_pass", function_tmpdir, alt)
    ml.write_input()
    success, buff = ml.run_model()
    assert success, pformat(buff)

    dis_file = f"{ml.name}.dis"
    bud_file = f"{ml.name}.cbc"
    hd_file = f"{ml.name}.hds"

    mp = flopy.modpath.Modpath6(
        modelname=ml.name,
        simfile_ext="mpsim",
        namefile_ext="mpnam",
        version="modpath",
        exe_name="mp6",
        modflowmodel=ml,
        dis_file=dis_file,
        head_file=hd_file,
        budget_file=bud_file,
        model_ws=ml.model_ws,
        external_path=None,
        verbose=False,
        load=False,
        listunit=7,
    )

    assert mp.head_file == hd_file
    assert mp.budget_file == bud_file
    assert mp.dis_file == dis_file
    assert mp.nrow_ncol_nlay_nper == (ctx.nrow, ctx.ncol, ctx.nlay, ctx.nper)

    mpbas = flopy.modpath.Modpath6Bas(
        mp,
        hnoflo=ctx.hnoflow,
        hdry=ctx.hdry,
        def_face_ct=0,
        bud_label=None,
        def_iface=None,
        laytyp=ctx.laytype,
        ibound=ctx.ibound,
        prsity=0.30,
        prsityCB=0.30,
        extension="mpbas",
        unitnumber=86,
    )

    # test layertype is created correctly!
    assert np.isclose(mpbas.laytyp.array, ctx.laytype).all()
    # test ibound is pulled from modflow model
    assert np.isclose(mpbas.ibound.array, ctx.ibound).all()

    sim = flopy.modpath.Modpath6Sim(model=mp)
    stl = flopy.modpath.mp6sim.StartingLocationsFile(model=mp)
    stldata = stl.get_empty_starting_locations_data(npt=2)
    stldata["label"] = ["p1", "p2"]
    stldata[1]["k0"] = 0
    stldata[1]["i0"] = 0
    stldata[1]["j0"] = 0
    stldata[1]["xloc0"] = 0.1
    stldata[1]["yloc0"] = 0.2
    stl.data = stldata
    mp.write_input()
    success, buff = mp.run_model()
    assert success


@requires_exe("mf2005", "mp6")
@pytest.mark.parametrize("alt", [True, False])
def test_just_from_model(function_tmpdir, alt):
    """
    test that user specified head files etc. are preferred
    over files from the modflow model
    """

    ml, ctx = get_mf2005_model("data_pass", function_tmpdir, alt)
    ml.write_input()
    success, buff = ml.run_model()
    assert success, pformat(buff)

    dis_file = f"{ml.name}.dis"
    bud_file = f"{ml.name}.cbc"
    hd_file = f"{ml.name}.hds"

    mp = flopy.modpath.Modpath6(
        modelname="modpathtest",
        simfile_ext="mpsim",
        namefile_ext="mpnam",
        version="modpath",
        exe_name="mp6",
        modflowmodel=ml,
        dis_file=None,
        head_file=None,
        budget_file=None,
        model_ws=ml.model_ws,
        external_path=None,
        verbose=False,
        load=False,
        listunit=7,
    )

    assert mp.head_file == hd_file
    assert mp.budget_file == bud_file
    assert mp.dis_file == dis_file
    assert mp.nrow_ncol_nlay_nper == (ctx.nrow, ctx.ncol, ctx.nlay, ctx.nper)

    mpbas = flopy.modpath.Modpath6Bas(
        mp,
        hnoflo=ctx.hnoflow,
        hdry=ctx.hdry,
        def_face_ct=0,
        bud_label=None,
        def_iface=None,
        laytyp=None,
        ibound=None,
        prsity=0.30,
        prsityCB=0.30,
        extension="mpbas",
        unitnumber=86,
    )
    # test layertype is created correctly!
    assert np.isclose(mpbas.laytyp.array, ctx.laytype).all()

    # test ibound is pulled from modflow model
    assert np.isclose(mpbas.ibound.array, ctx.ibound).all()

    sim = flopy.modpath.Modpath6Sim(model=mp)
    stl = flopy.modpath.mp6sim.StartingLocationsFile(model=mp)
    stldata = stl.get_empty_starting_locations_data(npt=2)
    stldata["label"] = ["p1", "p2"]
    stldata[1]["k0"] = 0
    stldata[1]["i0"] = 0
    stldata[1]["j0"] = 0
    stldata[1]["xloc0"] = 0.1
    stldata[1]["yloc0"] = 0.2
    stl.data = stldata
    mp.write_input()
    success, buff = mp.run_model()
    assert success


def get_mp6_model(m, ctx, name, ws, use_pandas):
    mp = flopy.modpath.Modpath6(
        modelname=name,
        simfile_ext="mpsim",
        namefile_ext="mpnam",
        version="modpath",
        exe_name="mp6",
        modflowmodel=m,
        dis_file=None,
        head_file=None,
        budget_file=None,
        model_ws=ws,
        external_path=None,
        verbose=False,
        load=True,
        listunit=7,
    )

    mpbas = flopy.modpath.Modpath6Bas(
        mp,
        hnoflo=ctx.hnoflow,
        hdry=ctx.hdry,
        def_face_ct=0,
        bud_label=None,
        def_iface=None,
        laytyp=ctx.laytype,
        ibound=ctx.ibound,
        prsity=0.30,
        prsityCB=0.30,
        extension="mpbas",
        unitnumber=86,
    )

    sim = flopy.modpath.Modpath6Sim(model=mp)
    stl = flopy.modpath.mp6sim.StartingLocationsFile(model=mp, use_pandas=use_pandas)
    stldata = stl.get_empty_starting_locations_data(npt=2)
    stldata["label"] = ["p1", "p2"]
    stldata[1]["k0"] = 0
    stldata[1]["i0"] = 0
    stldata[1]["j0"] = 0
    stldata[1]["xloc0"] = 0.1
    stldata[1]["yloc0"] = 0.2
    stl.data = stldata

    return mp


@requires_exe("mf2005")
def test_mp_pandas(function_tmpdir):
    name = "mp_pandas"
    ml, ctx = get_mf2005_model(name, function_tmpdir)
    ml.write_input()
    success, _ = ml.run_model()
    assert success

    mp_pandas = get_mp6_model(ml, ctx, name, function_tmpdir, use_pandas=True)
    mp_no_pandas = get_mp6_model(ml, ctx, name, function_tmpdir, use_pandas=False)

    mp_no_pandas.write_input()
    success, buff = mp_no_pandas.run_model()
    assert success, pformat(buff)

    mp_pandas.write_input()
    success, buff = mp_pandas.run_model()
    assert success, pformat(buff)

    # read the two files and ensure they are identical
    with open(mp_pandas.get_package("loc").fn_path) as f:
        particles_pandas = f.readlines()
    with open(mp_no_pandas.get_package("loc").fn_path) as f:
        particles_no_pandas = f.readlines()
    assert particles_pandas == particles_no_pandas
