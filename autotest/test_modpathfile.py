import inspect
import io
import pstats
from shutil import copytree

import pytest

from flopy.mf6 import MFSimulation, ModflowTdis, ModflowGwf, ModflowIms, ModflowGwfic, ModflowGwfdis, ModflowGwfnpf, ModflowGwfrcha, ModflowGwfwel, \
    ModflowGwfriv, ModflowGwfoc
from flopy.modpath import Modpath7
from flopy.utils import PathlineFile, EndpointFile

from autotest.conftest import requires_exe


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


@requires_exe("mf6", "mp7")
@pytest.mark.skip(reason="pending https://github.com/modflowpy/flopy/issues/1479")
@pytest.mark.slow
@pytest.mark.parametrize("direction", ["forward", "backward"])
@pytest.mark.parametrize("locations", ["well", "river"])
def test_get_destination_pathline_data(tmpdir, mp7_simulation, direction, locations, benchmark):
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

    # run benchmark
    benchmark(lambda: pathline_file.get_destination_pathline_data(dest_cells=nodew if locations == "well" else nodesr))


@requires_exe("mf6", "mp7")
@pytest.mark.parametrize("direction", ["forward", "backward"])
@pytest.mark.parametrize("locations", ["well", "river"])
def test_get_destination_endpoint_data(tmpdir, mp7_simulation, direction, locations, benchmark):
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

    # run benchmark
    benchmark(lambda: endpoint_file.get_destination_endpoint_data(dest_cells=nodew if locations == "well" else nodesr))
