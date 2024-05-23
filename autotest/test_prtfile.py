import pytest

import flopy
from autotest.conftest import get_project_root_path
from flopy.utils.prtfile import PathlineFile

pytestmark = pytest.mark.mf6
proj_root = get_project_root_path()

nlay = 1
nrow = 10
ncol = 10
top = 1.0
botm = [0.0]
nper = 1
perlen = 1.0
nstp = 1
tsmult = 1.0
porosity = 0.1


def get_partdata(grid, rpts):
    """
    Make a flopy.modpath.ParticleData from the given grid and release points.
    """

    if grid.grid_type == "structured":
        return flopy.modpath.ParticleData(
            partlocs=[grid.get_lrc(p[0])[0] for p in rpts],
            structured=True,
            localx=[p[1] for p in rpts],
            localy=[p[2] for p in rpts],
            localz=[p[3] for p in rpts],
            timeoffset=0,
            drape=0,
        )
    else:
        return flopy.modpath.ParticleData(
            partlocs=[p[0] for p in rpts],
            structured=False,
            localx=[p[1] for p in rpts],
            localy=[p[2] for p in rpts],
            localz=[p[3] for p in rpts],
            timeoffset=0,
            drape=0,
        )


@pytest.fixture
def gwf_sim(function_tmpdir):
    gwf_ws = function_tmpdir / "gwf"
    gwf_name = "plotutil_gwf"

    # create simulation
    sim = flopy.mf6.MFSimulation(
        sim_name=gwf_name,
        exe_name="mf6",
        version="mf6",
        sim_ws=gwf_ws,
    )

    # create tdis package
    flopy.mf6.modflow.mftdis.ModflowTdis(
        sim,
        pname="tdis",
        time_units="DAYS",
        nper=nper,
        perioddata=[(perlen, nstp, tsmult)],
    )

    # create gwf model
    gwf = flopy.mf6.ModflowGwf(sim, modelname=gwf_name, save_flows=True)

    # create gwf discretization
    flopy.mf6.modflow.mfgwfdis.ModflowGwfdis(
        gwf,
        pname="dis",
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
    )

    # create gwf initial conditions package
    flopy.mf6.modflow.mfgwfic.ModflowGwfic(gwf, pname="ic")

    # create gwf node property flow package
    flopy.mf6.modflow.mfgwfnpf.ModflowGwfnpf(
        gwf,
        pname="npf",
        save_saturation=True,
        save_specific_discharge=True,
    )

    # create gwf chd package
    spd = {
        0: [[(0, 0, 0), 1.0, 1.0], [(0, 9, 9), 0.0, 0.0]],
        1: [[(0, 0, 0), 0.0, 0.0], [(0, 9, 9), 1.0, 2.0]],
    }
    chd = flopy.mf6.ModflowGwfchd(
        gwf,
        pname="CHD-1",
        stress_period_data=spd,
        auxiliary=["concentration"],
    )

    # create gwf output control package
    # output file names
    gwf_budget_file = f"{gwf_name}.bud"
    gwf_head_file = f"{gwf_name}.hds"
    oc = flopy.mf6.ModflowGwfoc(
        gwf,
        budget_filerecord=gwf_budget_file,
        head_filerecord=gwf_head_file,
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )

    # create iterative model solution for gwf model
    ims = flopy.mf6.ModflowIms(sim)

    return sim


@pytest.fixture
def mp7_sim(gwf_sim):
    gwf = gwf_sim.get_model()
    ws = gwf_sim.sim_path.parent
    mp7_ws = ws / "mp7"
    releasepts_mp7 = [
        # node number, localx, localy, localz
        (0, float(f"0.{i + 1}"), float(f"0.{i + 1}"), 0.5)
        for i in range(9)
    ]

    partdata = get_partdata(gwf.modelgrid, releasepts_mp7)
    mp7_name = "plotutil_mp7"
    pg = flopy.modpath.ParticleGroup(
        particlegroupname="G1",
        particledata=partdata,
        filename=f"{mp7_name}.sloc",
    )
    mp = flopy.modpath.Modpath7(
        modelname=mp7_name,
        flowmodel=gwf,
        exe_name="mp7",
        model_ws=mp7_ws,
        headfilename=f"{gwf.name}.hds",
        budgetfilename=f"{gwf.name}.bud",
    )
    mpbas = flopy.modpath.Modpath7Bas(
        mp,
        porosity=porosity,
    )
    mpsim = flopy.modpath.Modpath7Sim(
        mp,
        simulationtype="pathline",
        trackingdirection="forward",
        budgetoutputoption="summary",
        stoptimeoption="extend",
        particlegroups=[pg],
    )

    return mp


@pytest.fixture
def prt_sim(gwf_sim):
    ws = gwf_sim.sim_path.parent
    gwf_ws = ws / "gwf"
    prt_ws = ws / "prt"
    prt_name = "plotutil_prt"
    gwf_name = "plotutil_gwf"
    releasepts_prt = [
        # particle index, k, i, j, x, y, z
        [i, 0, 0, 0, float(f"0.{i + 1}"), float(f"9.{i + 1}"), 0.5]
        for i in range(9)
    ]

    # create simulation
    sim = flopy.mf6.MFSimulation(
        sim_name=prt_name,
        exe_name="mf6",
        version="mf6",
        sim_ws=prt_ws,
    )

    # create tdis package
    flopy.mf6.modflow.mftdis.ModflowTdis(
        sim,
        pname="tdis",
        time_units="DAYS",
        nper=nper,
        perioddata=[
            (
                perlen,
                nstp,
                tsmult,
            )
        ],
    )

    # create prt model
    prt = flopy.mf6.ModflowPrt(sim, modelname=prt_name, save_flows=True)

    # create prt discretization
    flopy.mf6.modflow.mfgwfdis.ModflowGwfdis(
        prt,
        pname="dis",
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
    )

    # create mip package
    flopy.mf6.ModflowPrtmip(prt, pname="mip", porosity=porosity)

    # create prp package
    prp_track_file = f"{prt_name}.prp.trk"
    prp_track_csv_file = f"{prt_name}.prp.trk.csv"
    flopy.mf6.ModflowPrtprp(
        prt,
        pname="prp1",
        filename=f"{prt_name}_1.prp",
        nreleasepts=len(releasepts_prt),
        packagedata=releasepts_prt,
        perioddata={0: ["FIRST"]},
        track_filerecord=[prp_track_file],
        trackcsv_filerecord=[prp_track_csv_file],
        stop_at_weak_sink="saws" in prt_name,
        boundnames=True,
        exit_solve_tolerance=1e-5,
    )

    # create output control package
    prt_budget_file = f"{prt_name}.bud"
    prt_track_file = f"{prt_name}.trk"
    prt_track_csv_file = f"{prt_name}.trk.csv"
    flopy.mf6.ModflowPrtoc(
        prt,
        pname="oc",
        budget_filerecord=[prt_budget_file],
        track_filerecord=[prt_track_file],
        trackcsv_filerecord=[prt_track_csv_file],
        saverecord=[("BUDGET", "ALL")],
    )

    # create the flow model interface
    gwf_budget_file = gwf_ws / f"{gwf_name}.bud"
    gwf_head_file = gwf_ws / f"{gwf_name}.hds"
    flopy.mf6.ModflowPrtfmi(
        prt,
        packagedata=[
            ("GWFHEAD", gwf_head_file),
            ("GWFBUDGET", gwf_budget_file),
        ],
    )

    # add explicit model solution
    ems = flopy.mf6.ModflowEms(
        sim,
        pname="ems",
        filename=f"{prt_name}.ems",
    )
    sim.register_solution_package(ems, [prt.name])

    return sim


# todo: finish tests

# @pytest.mark.parametrize("csv", [False, True])
# def test_init(gwf_sim, prt_sim, csv):
#
#
#     file = PathlineFile(path, header_filename=header_path)
#     assert file.fname == path
#     if path.suffix == ".csv":
#         assert len(file._data) == len(open(path).readlines()) - 1
#
#
# @pytest.mark.parametrize("csv", [False, True])
# def test_intersect(gwf_sim, prt_sim, csv):
#
#
#     file = PathlineFile(path, header_filename=header_path)
#     nodes = [1, 11, 21]
#     intersection = file.intersect(nodes)
#     assert any(intersection)
#     assert all(d.icell in nodes for d in intersection.itertuples())
#
#
# @pytest.mark.parametrize("csv", [False, True])
# def test_validate(gwf_sim, prt_sim, csv):
#
#
#     file = PathlineFile(path, header_filename=header_path)
#     file.validate()
#
