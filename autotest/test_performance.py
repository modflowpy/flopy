import inspect
from os.path import join
from pathlib import Path
from shutil import copytree

import numpy as np
import pytest
from flopy.utils import PathlineFile, EndpointFile

from flopy.modpath import Modpath7

from flopy.mf6 import MFSimulation, ModflowTdis, ModflowGwf, ModflowIms, ModflowGwfdis, ModflowGwfic, ModflowGwfnpf, ModflowGwfrcha, ModflowGwfwel, \
    ModflowGwfriv, ModflowGwfoc

from autotest.conftest import requires_exes

from flopy.modflow import Modflow, ModflowDis, ModflowRch, ModflowWel, ModflowSfr2


def build_basic_modflow_model(ws, name):
    m = Modflow(name, model_ws=ws)

    size = 100
    nlay = 10
    nper = 10
    nsfr = int((size ** 2) / 5)

    dis = ModflowDis(
        m,
        nper=nper,
        nlay=nlay,
        nrow=size,
        ncol=size,
        top=nlay,
        botm=list(range(nlay)),
    )

    rch = ModflowRch(
        m, rech={k: 0.001 - np.cos(k) * 0.001 for k in range(nper)}
    )

    ra = ModflowWel.get_empty(size ** 2)
    well_spd = {}
    for kper in range(nper):
        ra_per = ra.copy()
        ra_per["k"] = 1
        ra_per["i"] = (
            (np.ones((size, size)) * np.arange(size))
                .transpose()
                .ravel()
                .astype(int)
        )
        ra_per["j"] = list(range(size)) * size
        well_spd[kper] = ra
    wel = ModflowWel(m, stress_period_data=well_spd)

    # SFR package
    rd = ModflowSfr2.get_empty_reach_data(nsfr)
    rd["iseg"] = range(len(rd))
    rd["ireach"] = 1
    sd = ModflowSfr2.get_empty_segment_data(nsfr)
    sd["nseg"] = range(len(sd))
    sfr = ModflowSfr2(reach_data=rd, segment_data=sd, model=m)

    return m


@pytest.mark.slow
def test_model_init_time(tmpdir, benchmark):
    name = inspect.getframeinfo(inspect.currentframe()).function
    benchmark(lambda: build_basic_modflow_model(ws=str(tmpdir), name=name))


@pytest.mark.slow
def test_model_write_time(tmpdir, benchmark):
    name = inspect.getframeinfo(inspect.currentframe()).function
    model = build_basic_modflow_model(ws=str(tmpdir), name=name)
    benchmark(lambda: model.write_input())


@pytest.mark.slow
def test_model_load_time(tmpdir, benchmark):
    name = inspect.getframeinfo(inspect.currentframe()).function
    model = build_basic_modflow_model(ws=str(tmpdir), name=name)
    model.write_input()
    benchmark(lambda: Modflow.load(f"{name}.nam", model_ws=str(tmpdir), check=False))


@pytest.fixture(scope="session")
def mp7_simulation(session_tmpdir):
    ws = str(session_tmpdir / "mp7_model")

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

    def get_nodes(locs):
        nodes = []
        for k, i, j in locs:
            nodes.append(k * nrow * ncol + i * ncol + j)
        return nodes

    name = "mp7_perf_test"

    # Create the Flopy simulation object
    sim = MFSimulation(
        sim_name=name, exe_name="mf6", version="mf6", sim_ws=ws
    )

    # Create the Flopy temporal discretization object
    pd = (perlen, nstp, tsmult)
    tdis = ModflowTdis(
        sim, pname="tdis", time_units="DAYS", nper=nper, perioddata=[pd]
    )

    # Create the Flopy groundwater flow (gwf) model object
    model_nam_file = "{}.nam".format(name)
    gwf = ModflowGwf(
        sim, modelname=name, model_nam_file=model_nam_file, save_flows=True
    )

    # Create the Flopy iterative model solver (ims) Package object
    ims = ModflowIms(
        sim,
        pname="ims",
        complexity="SIMPLE",
        outer_hclose=1e-6,
        inner_hclose=1e-6,
        rcloserecord=1e-6,
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
    npf = ModflowGwfnpf(
        gwf, pname="npf", icelltype=laytyp, k=kh, k33=kv
    )

    # recharge
    ModflowGwfrcha(gwf, recharge=rch)
    # wel
    wd = [(wel_loc, wel_q)]
    ModflowGwfwel(
        gwf, maxbound=1, stress_period_data={0: wd}
    )

    # river
    rd = []
    for i in range(nrow):
        rd.append([(0, i, ncol - 1), riv_h, riv_c, riv_z])
    ModflowGwfriv(gwf, stress_period_data={0: rd})

    # Create the output control package
    headfile = "{}.hds".format(name)
    head_record = [headfile]
    budgetfile = "{}.cbb".format(name)
    budget_record = [budgetfile]
    saverecord = [("HEAD", "ALL"), ("BUDGET", "ALL")]
    oc = ModflowGwfoc(
        gwf,
        pname="oc",
        saverecord=saverecord,
        head_filerecord=head_record,
        budget_filerecord=budget_record,
    )

    sim.write_simulation()
    success, buff = sim.run_simulation(silent=True)
    assert success, "mf6 model did not run"

    # get locations to track data
    nodew = get_nodes([wel_loc])
    cellids = gwf.riv.stress_period_data.get_data()[0]["cellid"]
    nodesr = get_nodes(cellids)

    nodew = get_nodes([wel_loc])
    cellids = gwf.riv.stress_period_data.get_data()[0]["cellid"]
    nodesr = get_nodes(cellids)

    forward_model_name = name + "_forward"

    # create basic forward tracking modpath simulation
    mp = Modpath7.create_mp7(
        modelname=forward_model_name,
        trackdir="forward",
        flowmodel=gwf,
        model_ws=ws,
        rowcelldivisions=1,
        columncelldivisions=1,
        layercelldivisions=1,
        exe_name="mp7",
    )

    # write modpath datasets and run forward model
    mp.write_input()
    mp.run_model(silent=True)

    backward_model_name = name + "_backward"

    # create basic backward tracking modpath simulation
    mp = Modpath7.create_mp7(
        modelname=backward_model_name,
        trackdir="backward",
        flowmodel=gwf,
        model_ws=ws,
        rowcelldivisions=5,
        columncelldivisions=5,
        layercelldivisions=5,
        nodes=nodew + nodesr,
        exe_name="mp7",
    )

    # write modpath datasets and run backward model
    mp.write_input()
    mp.run_model(silent=True)

    return sim, forward_model_name, backward_model_name, nodew, nodesr


@requires_exes(["mf6", "mp7"])
@pytest.mark.skip(reason="skip until performance improves (https://github.com/modflowpy/flopy/issues/1479)")
@pytest.mark.slow
@pytest.mark.parametrize("direction", ["forward", "backward"])
@pytest.mark.parametrize("locations", ["well", "river"])
def test_get_destination_pathline_data(tmpdir, benchmark, mp7_simulation, direction, locations):
    sim, forward_model_name, backward_model_name, nodew, nodesr = mp7_simulation
    ws = tmpdir / "ws"

    # copy simulation data from fixture setup to temp workspace
    copytree(sim.simulation_data.mfpath.get_sim_path(), ws)

    # make sure we have pathline files
    forward_path = ws / f"{forward_model_name}.mppth"
    backward_path = ws / f"{backward_model_name}.mppth"
    assert forward_path.is_file()
    assert backward_path.is_file()

    # get pathline file corresponding to parametrized direction
    pathline_file = PathlineFile(str(backward_path) if direction == "backward" else str(forward_path))

    # run benchmark (only 1 round with 1 iteration, since this is slow)
    pathline_data = benchmark(lambda: pathline_file.get_destination_pathline_data(dest_cells=nodew if locations == "well" else nodesr))


@requires_exes(["mf6", "mp7"])
@pytest.mark.slow
@pytest.mark.parametrize("direction", ["forward", "backward"])
@pytest.mark.parametrize("locations", ["well", "river"])
def test_get_destination_endpoint_data(tmpdir, benchmark, mp7_simulation, direction, locations):
    sim, forward_model_name, backward_model_name, nodew, nodesr = mp7_simulation
    ws = tmpdir / "ws"

    # copy simulation data from fixture setup to temp workspace
    copytree(sim.simulation_data.mfpath.get_sim_path(), ws)

    # make sure we have endpoint files
    forward_end = ws / f"{forward_model_name}.mpend"
    backward_end = ws / f"{backward_model_name}.mpend"
    assert forward_end.is_file()
    assert backward_end.is_file()

    # get endpoint file corresponding to parametrized direction
    endpoint_file = EndpointFile(str(backward_end) if direction == "backward" else str(forward_end))

    # run benchmark (only 1 round with 1 iteration, since this is slow)
    endpoint_data = benchmark(lambda: endpoint_file.get_destination_endpoint_data(dest_cells=nodew if locations == "well" else nodesr))
