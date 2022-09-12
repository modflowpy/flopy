import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pytest
from autotest.conftest import get_example_data_path, requires_exe, requires_pkg

from flopy.discretization import StructuredGrid
from flopy.export.shapefile_utils import shp2recarray
from flopy.mf6 import (
    MFSimulation,
    ModflowGwf,
    ModflowGwfdis,
    ModflowGwfic,
    ModflowGwfnpf,
    ModflowGwfoc,
    ModflowGwfrcha,
    ModflowGwfriv,
    ModflowGwfwel,
    ModflowIms,
    ModflowTdis,
)
from flopy.modflow import (
    Modflow,
    ModflowBas,
    ModflowDis,
    ModflowLpf,
    ModflowMnw2,
    ModflowOc,
    ModflowPcg,
    ModflowRch,
    ModflowRiv,
    ModflowWel,
)
from flopy.modpath import (
    Modpath6,
    Modpath6Bas,
    Modpath7,
    Modpath7Bas,
    Modpath7Sim,
    ParticleData,
    ParticleGroup,
)
from flopy.modpath.mp6sim import Modpath6Sim, StartingLocationsFile
from flopy.plot import PlotMapView
from flopy.utils import EndpointFile, PathlineFile
from flopy.utils.flopy_io import loadtxt
from flopy.utils.recarray_utils import ra_slice

pytestmark = pytest.mark.mf6

MP6_TEST_PATH = get_example_data_path(__file__) / "mp6"


@pytest.fixture
def mp6_test_path(example_data_path):
    return example_data_path / "mp6"


def copy_modpath_files(source, model_ws, baseName):
    files = [
        file
        for file in os.listdir(source)
        if file.startswith(baseName)
        and os.path.isfile(os.path.join(source, file))
    ]
    for file in files:
        src = os.path.join(MP6_TEST_PATH, file)
        dst = os.path.join(model_ws, file)
        print(f"copying {src} -> {dst}")
        shutil.copy(src, dst)


def test_mpsim(tmpdir, mp6_test_path):
    copy_modpath_files(str(mp6_test_path), str(tmpdir), "EXAMPLE.")

    m = Modflow.load("EXAMPLE.nam", model_ws=str(tmpdir))
    m.get_package_list()

    mp = Modpath6(
        modelname="ex6",
        exe_name="mp6",
        modflowmodel=m,
        model_ws=str(tmpdir),
        dis_file=f"{m.name}.dis",
        head_file=f"{m.name}.hed",
        budget_file=f"{m.name}.bud",
    )

    mpb = Modpath6Bas(
        mp, hdry=m.lpf.hdry, laytyp=m.lpf.laytyp, ibound=1, prsity=0.1
    )

    sim = mp.create_mpsim(
        trackdir="forward", simtype="endpoint", packages="RCH"
    )
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
    sim = mp.create_mpsim(
        trackdir="backward", simtype="pathline", packages="MNW2"
    )
    mp.write_input()

    sim = Modpath6Sim(model=mp)
    # starting locations file
    stl = StartingLocationsFile(model=mp)
    stldata = StartingLocationsFile.get_empty_starting_locations_data(npt=2)
    stldata["label"] = ["p1", "p2"]
    stldata[1]["i0"] = 5
    stldata[1]["j0"] = 6
    stldata[1]["xloc0"] = 0.1
    stldata[1]["yloc0"] = 0.2
    stl.data = stldata
    mp.write_input()
    stllines = open(os.path.join(str(tmpdir), "ex6.loc")).readlines()
    assert stllines[3].strip() == "group1"
    assert int(stllines[4].strip()) == 2
    assert stllines[6].strip().split()[-1] == "p2"


@requires_pkg("pandas", "shapefile")
def test_get_destination_data(tmpdir, mp6_test_path):
    copy_modpath_files(str(mp6_test_path), str(tmpdir), "EXAMPLE.")
    copy_modpath_files(str(mp6_test_path), str(tmpdir), "EXAMPLE-3.")

    m = Modflow.load("EXAMPLE.nam", model_ws=str(tmpdir))

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
    m.dis.export(str(tmpdir / "dis.shp"))

    pthld = PathlineFile(str(tmpdir / "EXAMPLE-3.pathline"))
    epd = EndpointFile(str(tmpdir / "EXAMPLE-3.endpoint"))

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
    starting_locs = ra_slice(well_epd, ["k0", "i0", "j0"])
    pathline_locs = np.array(
        np.array(well_pthld)[["k", "i", "j"]].tolist(),
        dtype=starting_locs.dtype,
    )
    assert np.all(np.in1d(starting_locs, pathline_locs))

    # test writing a shapefile of endpoints
    epd.write_shapefile(
        well_epd,
        direction="starting",
        shpname=str(tmpdir / "starting_locs.shp"),
        mg=m.modelgrid,
    )

    # test writing shapefile of pathlines
    fpth = str(tmpdir / "pathlines_1per.shp")
    pthld.write_shapefile(
        well_pthld,
        one_per_particle=True,
        direction="starting",
        mg=m.modelgrid,
        shpname=fpth,
    )
    fpth = str(tmpdir / "pathlines_1per_end.shp")
    pthld.write_shapefile(
        well_pthld,
        one_per_particle=True,
        direction="ending",
        mg=m.modelgrid,
        shpname=fpth,
    )
    # test writing shapefile of pathlines
    fpth = str(tmpdir / "pathlines_1per2.shp")
    pthld.write_shapefile(
        well_pthld,
        one_per_particle=True,
        direction="starting",
        mg=mg,
        shpname=fpth,
    )
    # test writing shapefile of pathlines
    fpth = str(tmpdir / "pathlines_1per2_ll.shp")
    pthld.write_shapefile(
        well_pthld,
        one_per_particle=True,
        direction="starting",
        mg=mg,
        shpname=fpth,
    )
    fpth = str(tmpdir / "pathlines.shp")
    pthld.write_shapefile(
        well_pthld, one_per_particle=False, mg=m.modelgrid, shpname=fpth
    )

    # test that endpoints were rotated and written correctly
    ra = shp2recarray(str(tmpdir / "starting_locs.shp"))
    p3 = ra.geometry[ra.particleid == 4][0]
    xorig, yorig = m.modelgrid.get_coords(well_epd.x0[0], well_epd.y0[0])
    assert p3.x - xorig + p3.y - yorig < 1e-4
    xorig, yorig = mg1.xcellcenters[3, 4], mg1.ycellcenters[3, 4]
    assert (
        np.abs(p3.x - xorig + p3.y - yorig) < 1e-4
    )  # this also checks for 1-based

    # test that particle attribute information is consistent with pathline file
    ra = shp2recarray(str(tmpdir / "pathlines.shp"))
    inds = (ra.particleid == 8) & (ra.i == 12) & (ra.j == 12)
    assert ra.time[inds][0] - 20181.7 < 0.1
    assert ra.xloc[inds][0] - 0.933 < 0.01

    # test that k, i, j are correct for single geometry pathlines, forwards
    # and backwards
    ra = shp2recarray(str(tmpdir / "pathlines_1per.shp"))
    assert ra.i[0] == 4, ra.j[0] == 5
    ra = shp2recarray(str(tmpdir / "pathlines_1per_end.shp"))
    assert ra.i[0] == 13, ra.j[0] == 13

    # test use of arbitrary spatial reference and offset
    mg1.set_coord_info(
        xoff=mg.xoffset,
        yoff=mg.yoffset,
        angrot=mg.angrot,
        epsg=mg.epsg,
        proj4=mg.proj4,
    )
    ra = shp2recarray(str(tmpdir / "pathlines_1per2.shp"))
    p3_2 = ra.geometry[ra.particleid == 4][0]
    test1 = mg1.xcellcenters[3, 4]
    test2 = mg1.ycellcenters[3, 4]
    assert (
        np.abs(
            p3_2.x[0]
            - mg1.xcellcenters[3, 4]
            + p3_2.y[0]
            - mg1.ycellcenters[3, 4]
        )
        < 1e-4
    )

    # arbitrary spatial reference with ll specified instead of ul
    ra = shp2recarray(str(tmpdir / "pathlines_1per2_ll.shp"))
    p3_2 = ra.geometry[ra.particleid == 4][0]
    mg.set_coord_info(xoff=mg.xoffset, yoff=mg.yoffset, angrot=30.0)
    assert (
        np.abs(
            p3_2.x[0]
            - mg.xcellcenters[3, 4]
            + p3_2.y[0]
            - mg.ycellcenters[3, 4]
        )
        < 1e-4
    )

    xul = 3628793
    yul = 21940389

    m = Modflow.load("EXAMPLE.nam", model_ws=str(tmpdir))

    mg4 = m.modelgrid
    mg4.set_coord_info(
        xoff=mg4._xul_to_xll(xul, 0.0),
        yoff=mg4._yul_to_yll(yul, 0.0),
        angrot=0.0,
        epsg=mg4.epsg,
        proj4=mg4.proj4,
    )

    fpth = str(tmpdir / "dis2.shp")
    m.dis.export(fpth)
    pthobj = PathlineFile(str(tmpdir / "EXAMPLE-3.pathline"))
    fpth = str(tmpdir / "pathlines_1per3.shp")
    pthobj.write_shapefile(shpname=fpth, direction="ending", mg=mg4)


@requires_pkg("pandas")
def test_loadtxt(tmpdir, mp6_test_path):
    copy_modpath_files(str(mp6_test_path), str(tmpdir), "EXAMPLE-3.")

    pthfile = str(tmpdir / "EXAMPLE-3.pathline")
    pthld = PathlineFile(pthfile)
    ra = loadtxt(pthfile, delimiter=" ", skiprows=3, dtype=pthld.dtype)
    ra2 = loadtxt(
        pthfile, delimiter=" ", skiprows=3, dtype=pthld.dtype, use_pandas=False
    )
    assert np.array_equal(ra, ra2)

    # epfilewithnans = os.path.join('../examples/data/mp6/', 'freybergmp.mpend')
    # epd = EndpointFile(epfilewithnans)


@requires_exe("mf2005")
@requires_pkg("pandas")
def test_modpath(tmpdir, example_data_path):
    pth = example_data_path / "freyberg"
    mfnam = "freyberg.nam"

    m = Modflow.load(
        mfnam,
        model_ws=str(pth),
        verbose=True,
        exe_name="mf2005",
        check=False,
    )
    assert m.load_fail is False

    m.change_model_ws(str(tmpdir))
    m.write_input()

    success, buff = m.run_model(silent=False)
    assert success, "modflow model run did not terminate successfully"

    # create the forward modpath file
    mpnam = "freybergmp"
    mp = Modpath6(
        mpnam,
        exe_name="mp6",
        modflowmodel=m,
        model_ws=str(tmpdir),
    )
    mpbas = Modpath6Bas(
        mp,
        hnoflo=m.bas6.hnoflo,
        hdry=m.lpf.hdry,
        ibound=m.bas6.ibound.array,
        prsity=0.2,
        prsityCB=0.2,
    )
    sim = mp.create_mpsim(
        trackdir="forward", simtype="endpoint", packages="RCH"
    )

    # write forward particle track files
    mp.write_input()

    if success:
        success, buff = mp.run_model(silent=False)
        assert (
            success
        ), "forward modpath model run did not terminate successfully"

    mpnam = "freybergmpp"
    mpp = Modpath6(
        mpnam,
        exe_name="mp6",
        modflowmodel=m,
        model_ws=str(tmpdir),
    )
    mpbas = Modpath6Bas(
        mpp,
        hnoflo=m.bas6.hnoflo,
        hdry=m.lpf.hdry,
        ibound=m.bas6.ibound.array,
        prsity=0.2,
        prsityCB=0.2,
    )
    sim = mpp.create_mpsim(
        trackdir="backward", simtype="pathline", packages="WEL"
    )

    # write backward particle track files
    mpp.write_input()

    if success:
        success, buff = mpp.run_model(silent=False)
        assert (
            success
        ), "backward modpath model run did not terminate successfully"

    # load modpath output files
    if success:
        endfile = os.path.join(str(tmpdir), mp.sim.endpoint_file)
        pthfile = os.path.join(str(tmpdir), mpp.sim.pathline_file)

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
        assert (
            len(plines) == 576
        ), "there are not 576 particle pathlines in file"

        # load the modflow files for model map
        mfnam = "freyberg.nam"
        m = Modflow.load(
            mfnam,
            model_ws=str(tmpdir),
            verbose=True,
            forgive=False,
            exe_name="mf2005",
        )

        # load modpath output files
        pthfile = os.path.join(str(tmpdir), "freybergmpp.mppth")

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
        fpth = os.path.join(str(tmpdir), "pathline.png")
        plt.savefig(fpth)
        plt.close()

        mm = PlotMapView(model=m)
        mm.plot_pathline(plines, colors="green", layer=0)

        # plot the grid and ibound array
        mm.plot_grid()
        mm.plot_ibound()

        fpth = os.path.join(str(tmpdir), "pathline2.png")
        plt.savefig(fpth)
        plt.close()

        mm = PlotMapView(model=m)
        mm.plot_pathline(plines, colors="red")

        # plot the grid and ibound array
        mm.plot_grid()
        mm.plot_ibound()

        fpth = os.path.join(str(tmpdir), "pathline3.png")
        plt.savefig(fpth)
        plt.close()


exe_names = {"mf2005": "mf2005", "mf6": "mf6", "mp7": "mp7"}

nper, nstp, perlen, tsmult = 1, 1, 1.0, 1.0
nlay, nrow, ncol = 3, 21, 20
delr = delc = 500.0
top = 400.0
botm = [220.0, 200.0, 0.0]
laytyp = [1, 0, 0]
kh = [50.0, 0.01, 200.0]
kv = [10.0, 0.01, 20.0]
wel_loc = (2, 10, 9)
wel_q = -150000.0
rch = 0.005
riv_h = 320.0
riv_z = 317.0
riv_c = 1.0e5

zone3 = np.ones((nrow, ncol), dtype=np.int32)
zone3[wel_loc[1:]] = 2
zones = [1, 1, zone3]

# create particles
partlocs = []
partids = []
for i in range(nrow):
    partlocs.append((0, i, 2))
    partids.append(i)
part0 = ParticleData(partlocs, structured=True, particleids=partids)
pg0 = ParticleGroup(
    particlegroupname="PG1", particledata=part0, filename="ex01a.sloc"
)

v = [(0,), (400,)]
pids = [1, 2]  # [1000, 1001]
part1 = ParticleData(v, structured=False, drape=1, particleids=pids)
pg1 = ParticleGroup(
    particlegroupname="PG2", particledata=part1, filename="ex01a.pg2.sloc"
)

particlegroups = [pg0, pg1]


@pytest.fixture
def case_mf6(tmpdir):
    """
    MODPATH 7 example 1 for MODFLOW 6
    """

    ws = os.path.join(str(tmpdir), "mf6")
    nm = "ex01_mf6"
    exe_name = exe_names["mf6"]

    # Create the Flopy simulation object
    sim = MFSimulation(sim_name=nm, exe_name="mf6", version="mf6", sim_ws=ws)

    # Create the Flopy temporal discretization object
    pd = (perlen, nstp, tsmult)
    tdis = ModflowTdis(
        sim, pname="tdis", time_units="DAYS", nper=nper, perioddata=[pd]
    )

    # Create the Flopy groundwater flow (gwf) model object
    model_nam_file = f"{nm}.nam"
    gwf = ModflowGwf(
        sim, modelname=nm, model_nam_file=model_nam_file, save_flows=True
    )

    # Create the Flopy iterative model solver (ims) Package object
    ims = ModflowIms(
        sim,
        pname="ims",
        complexity="SIMPLE",
        inner_hclose=1e-6,
        rcloserecord=1e-3,
        outer_hclose=1e-6,
        outer_maximum=50,
        inner_maximum=100,
    )

    # create gwf file
    dis = ModflowGwfdis(
        gwf,
        pname="dis",
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        length_units="FEET",
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
    )
    # Create the initial conditions package
    ic = ModflowGwfic(gwf, pname="ic", strt=top)

    # Create the node property flow package
    npf = ModflowGwfnpf(gwf, pname="npf", icelltype=laytyp, k=kh, k33=kv)

    # recharge
    ModflowGwfrcha(gwf, recharge=rch)
    # wel
    wd = [(wel_loc, wel_q)]
    ModflowGwfwel(gwf, maxbound=1, stress_period_data={0: wd})
    # river
    rd = []
    for i in range(nrow):
        rd.append([(0, i, ncol - 1), riv_h, riv_c, riv_z])
    ModflowGwfriv(gwf, stress_period_data={0: rd})
    # Create the output control package
    headfile = f"{nm}.hds"
    head_record = [headfile]
    budgetfile = f"{nm}.cbb"
    budget_record = [budgetfile]
    saverecord = [("HEAD", "ALL"), ("BUDGET", "ALL")]
    oc = ModflowGwfoc(
        gwf,
        pname="oc",
        saverecord=saverecord,
        head_filerecord=head_record,
        budget_filerecord=budget_record,
    )

    # Write the datasets
    sim.write_simulation()

    # Run the simulation
    success, buff = sim.run_simulation()
    assert success, "mf6 model did not run"

    # create modpath files
    exe_name = exe_names["mp7"]
    mp = Modpath7(
        modelname=f"{nm}_mp", flowmodel=gwf, exe_name=exe_name, model_ws=ws
    )
    defaultiface6 = {"RCH": 6, "EVT": 6}
    mpbas = Modpath7Bas(mp, porosity=0.1, defaultiface=defaultiface6)
    mpsim = Modpath7Sim(
        mp,
        simulationtype="combined",
        trackingdirection="forward",
        weaksinkoption="pass_through",
        weaksourceoption="pass_through",
        budgetoutputoption="summary",
        budgetcellnumbers=[1049, 1259],
        traceparticledata=[1, 1000],
        referencetime=[0, 0, 0.0],
        stoptimeoption="extend",
        timepointdata=[500, 1000.0],
        zonedataoption="on",
        zones=zones,
        particlegroups=particlegroups,
    )

    # write modpath datasets
    mp.write_input()

    # run modpath
    success, buff = mp.run_model()
    assert success, f"mp7 model ({mp.name}) did not run"

    return mp


@pytest.fixture
def case_mf2005(tmpdir):
    """
    MODPATH 7 example 1 for MODFLOW-2005
    """

    ws = os.path.join(str(tmpdir), "mf2005")
    nm = "ex01_mf2005"
    exe_name = exe_names["mf2005"]
    iu_cbc = 130
    m = Modflow(nm, model_ws=ws, exe_name=exe_name)
    ModflowDis(
        m,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        nper=nper,
        itmuni=4,
        lenuni=1,
        perlen=perlen,
        nstp=nstp,
        tsmult=tsmult,
        steady=True,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
    )
    ModflowLpf(m, ipakcb=iu_cbc, laytyp=laytyp, hk=kh, vka=kv)
    ModflowBas(m, ibound=1, strt=top)
    # recharge
    ModflowRch(m, ipakcb=iu_cbc, rech=rch, nrchop=1)
    # wel
    wd = [i for i in wel_loc] + [wel_q]
    ModflowWel(m, ipakcb=iu_cbc, stress_period_data={0: wd})
    # river
    rd = []
    for i in range(nrow):
        rd.append([0, i, ncol - 1, riv_h, riv_c, riv_z])
    ModflowRiv(m, ipakcb=iu_cbc, stress_period_data={0: rd})
    # output control
    ModflowOc(
        m,
        stress_period_data={
            (0, 0): ["save head", "save budget", "print head"]
        },
    )
    ModflowPcg(m, hclose=1e-6, rclose=1e-3, iter1=100, mxiter=50)

    m.write_input()

    success, buff = m.run_model()
    assert success, "mf2005 model did not run"

    # create modpath files
    exe_name = exe_names["mp7"]
    mp = Modpath7(
        modelname=f"{nm}_mp", flowmodel=m, exe_name=exe_name, model_ws=ws
    )
    defaultiface = {"RECHARGE": 6, "ET": 6}
    mpbas = Modpath7Bas(mp, porosity=0.1, defaultiface=defaultiface)
    mpsim = Modpath7Sim(
        mp,
        simulationtype="combined",
        trackingdirection="forward",
        weaksinkoption="pass_through",
        weaksourceoption="pass_through",
        budgetoutputoption="summary",
        budgetcellnumbers=[1049, 1259],
        traceparticledata=[1, 1000],
        referencetime=[0, 0, 0.0],
        stoptimeoption="extend",
        timepointdata=[500, 1000.0],
        zonedataoption="on",
        zones=zones,
        particlegroups=particlegroups,
    )

    # write modpath datasets
    mp.write_input()
    return mp


@requires_exe(*exe_names)
def test_pathline_output(case_mf2005, case_mf6):
    success, buff = case_mf2005.run_model()
    assert success, f"modpath model ({case_mf2005.name}) did not run"

    success, buff = case_mf6.run_model()
    assert success, f"modpath model ({case_mf6.name}) did not run"

    fpth0 = os.path.join(case_mf2005.model_ws, "ex01_mf2005_mp.mppth")
    p = PathlineFile(fpth0)
    maxtime0 = p.get_maxtime()
    maxid0 = p.get_maxid()
    p0 = p.get_alldata()
    fpth1 = os.path.join(case_mf6.model_ws, "ex01_mf6_mp.mppth")
    p = PathlineFile(fpth1)
    maxtime1 = p.get_maxtime()
    maxid1 = p.get_maxid()
    p1 = p.get_alldata()

    # check maxid
    msg = (
        f"pathline maxid ({maxid0}) in {os.path.basename(fpth0)} are not "
        f"equal to the pathline maxid ({maxid1}) in {os.path.basename(fpth1)}"
    )
    assert maxid0 == maxid1, msg


@requires_pkg("pandas")
def test_endpoint_output(case_mf2005, case_mf6):
    success, buff = case_mf2005.run_model()
    assert success, f"modpath model ({case_mf2005.name}) did not run"

    success, buff = case_mf6.run_model()
    assert success, f"modpath model ({case_mf6.name}) did not run"

    # if models not run then there will be no output
    fpth0 = os.path.join(case_mf2005.model_ws, "ex01_mf2005_mp.mpend")
    e = EndpointFile(fpth0)
    maxtime0 = e.get_maxtime()
    maxid0 = e.get_maxid()
    maxtravel0 = e.get_maxtraveltime()
    e0 = e.get_alldata()
    fpth1 = os.path.join(case_mf6.model_ws, "ex01_mf6_mp.mpend")
    e = EndpointFile(fpth1)
    maxtime1 = e.get_maxtime()
    maxid1 = e.get_maxid()
    maxtravel1 = e.get_maxtraveltime()
    e1 = e.get_alldata()

    # check maxid
    msg = (
        f"endpoint maxid ({maxid0}) in {os.path.basename(fpth0)} are not "
        f"equal to the endpoint maxid ({maxid1}) in {os.path.basename(fpth1)}"
    )
    assert maxid0 == maxid1, msg

    # check that endpoint data are approximately the same
    names = ["x", "y", "z", "x0", "y0", "z0"]
    dtype = np.dtype(
        [
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("x0", np.float32),
            ("y0", np.float32),
            ("z0", np.float32),
        ]
    )
    d = np.rec.fromarrays((e0[name] - e1[name] for name in names), dtype=dtype)
    msg = (
        f"endpoints in {os.path.basename(fpth0)} are not equal (within 1e-5) "
        f"to the endpoints  in {os.path.basename(fpth1)}"
    )
    # assert not np.allclose(t0, t1), msg


def test_pathline_plotting(case_mf6):
    success, buff = case_mf6.run_model()
    assert success, f"modpath model ({case_mf6.name}) did not run"

    modelgrid = case_mf6.flowmodel.modelgrid
    nodes = list(range(modelgrid.nnodes))

    fpth1 = os.path.join(case_mf6.model_ws, "ex01_mf6_mp.mppth")
    p = PathlineFile(fpth1)
    p1 = p.get_alldata()
    pls = p.get_destination_data(nodes)

    pmv = PlotMapView(modelgrid=modelgrid, layer=0)
    pmv.plot_grid()
    linecol = pmv.plot_pathline(pls, layer="all")
    linecol2 = pmv.plot_pathline(p1, layer="all")
    if not len(linecol._paths) == len(linecol2._paths):
        raise AssertionError(
            "plot_pathline not properly splitting particles from recarray"
        )
    plt.close()
