import inspect
import io
import pstats
from itertools import repeat
from shutil import copytree

import numpy as np
import numpy.lib.recfunctions as rfn
import pytest
from modflow_devtools.markers import requires_exe, requires_pkg

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
from flopy.modpath import Modpath7
from flopy.utils import EndpointFile, PathlineFile

pytestmark = pytest.mark.mf6


def __create_and_run_simulation(
    ws,
    name,
    nrow,
    ncol,
    perlen,
    nstp,
    tsmult,
    nper,
    nlay,
    delr,
    delc,
    top,
    botm,
    laytyp,
    kh,
    kv,
    rch,
    wel_loc,
    wel_q,
    riv_h,
    riv_c,
    riv_z,
):
    def get_nodes(locs):
        nodes = []
        for k, i, j in locs:
            nodes.append(k * nrow * ncol + i * ncol + j)
        return nodes

    # Create the Flopy simulation object
    sim = MFSimulation(sim_name=name, exe_name="mf6", version="mf6", sim_ws=ws)

    # Create the Flopy temporal discretization object
    pd = (perlen, nstp, tsmult)
    tdis = ModflowTdis(
        sim, pname="tdis", time_units="DAYS", nper=nper, perioddata=[pd]
    )

    # Create the Flopy groundwater flow (gwf) model object
    model_nam_file = f"{name}.nam"
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
    headfile = f"{name}.hds"
    head_record = [headfile]
    budgetfile = f"{name}.cbb"
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


@pytest.fixture(scope="module")
def mp7_small(module_tmpdir):
    return __create_and_run_simulation(
        ws=module_tmpdir / "mp7_small",
        name="mp7_small",
        nper=1,
        nstp=1,
        perlen=1.0,
        tsmult=1.0,
        nlay=3,
        nrow=11,
        ncol=10,
        delr=500.0,
        delc=500.0,
        top=400.0,
        botm=[220.0, 200.0, 0.0],
        laytyp=[1, 0, 0],
        kh=[50.0, 0.01, 200.0],
        kv=[10.0, 0.01, 20.0],
        wel_loc=(2, 10, 9),
        wel_q=-150000.0,
        rch=0.005,
        riv_h=320.0,
        riv_z=317.0,
        riv_c=1.0e5,
    )


@pytest.fixture(scope="module")
def mp7_large(module_tmpdir):
    return __create_and_run_simulation(
        ws=module_tmpdir / "mp7_large",
        name="mp7_large",
        nper=1,
        nstp=1,
        perlen=1.0,
        tsmult=1.0,
        nlay=3,
        nrow=21,
        ncol=20,
        delr=500.0,
        delc=500.0,
        top=400.0,
        botm=[220.0, 200.0, 0.0],
        laytyp=[1, 0, 0],
        kh=[50.0, 0.01, 200.0],
        kv=[10.0, 0.01, 20.0],
        wel_loc=(2, 10, 9),
        wel_q=-150000.0,
        rch=0.005,
        riv_h=320.0,
        riv_z=317.0,
        riv_c=1.0e5,
    )


@requires_exe("mf6")
def test_pathline_file_sorts_in_ctor(
    function_tmpdir, module_tmpdir, mp7_small
):
    sim, forward_model_name, backward_model_name, nodew, nodesr = mp7_small
    ws = function_tmpdir / "ws"

    # copytree(sim.simulation_data.mfpath.get_sim_path(), ws)
    copytree(module_tmpdir / "mp7_small", ws)

    forward_path = ws / f"{forward_model_name}.mppth"
    assert forward_path.is_file()

    pathline_file = PathlineFile(forward_path)
    assert np.all(
        pathline_file._data[:-1]["particleid"]
        <= pathline_file._data[1:]["particleid"]
    )


@requires_exe("mf6", "mp7")
@pytest.mark.slow
@pytest.mark.parametrize("direction", ["forward", "backward"])
@pytest.mark.parametrize("locations", ["well", "river"])
def test_get_destination_pathline_data(
    function_tmpdir, mp7_large, direction, locations, benchmark
):
    sim, forward_model_name, backward_model_name, nodew, nodesr = mp7_large
    ws = function_tmpdir / "ws"

    copytree(sim.simulation_data.mfpath.get_sim_path(), ws)

    forward_path = ws / f"{forward_model_name}.mppth"
    backward_path = ws / f"{backward_model_name}.mppth"
    assert forward_path.is_file()
    assert backward_path.is_file()

    pathline_file = PathlineFile(
        backward_path if direction == "backward" else forward_path
    )
    benchmark(
        lambda: pathline_file.get_destination_pathline_data(
            dest_cells=nodew if locations == "well" else nodesr
        )
    )


@requires_exe("mf6", "mp7")
@pytest.mark.slow
@pytest.mark.parametrize("direction", ["forward", "backward"])
@pytest.mark.parametrize("locations", ["well", "river"])
def test_get_destination_endpoint_data(
    function_tmpdir, mp7_large, direction, locations, benchmark
):
    sim, forward_model_name, backward_model_name, nodew, nodesr = mp7_large
    ws = function_tmpdir / "ws"

    copytree(sim.simulation_data.mfpath.get_sim_path(), ws)

    forward_end = ws / f"{forward_model_name}.mpend"
    backward_end = ws / f"{backward_model_name}.mpend"
    assert forward_end.is_file()
    assert backward_end.is_file()

    endpoint_file = EndpointFile(
        backward_end if direction == "backward" else forward_end
    )
    benchmark(
        lambda: endpoint_file.get_destination_endpoint_data(
            dest_cells=nodew if locations == "well" else nodesr
        )
    )


@pytest.mark.parametrize("longfieldname", [True, False])
@requires_exe("mf6", "mp7")
@requires_pkg("shapefile", "shapely")
def test_write_shapefile(function_tmpdir, mp7_small, longfieldname):
    from shapefile import Reader

    # setup and run model, then copy outputs to function_tmpdir
    sim, forward_model_name, _, _, _ = mp7_small
    gwf = sim.get_model()
    grid = gwf.modelgrid
    ws = function_tmpdir / "ws"
    copytree(sim.simulation_data.mfpath.get_sim_path(), ws)

    # make sure forward model output exists
    forward_path = ws / f"{forward_model_name}.mppth"
    assert forward_path.is_file()

    # load pathlines from file
    pathline_file = PathlineFile(forward_path)
    pathlines = pathline_file.get_alldata()

    # define shapefile path
    shp_file = ws / "pathlines.shp"

    # add a column to the pathline recarray
    fieldname = "newfield" + ("longname" if longfieldname else "")
    fieldval = "x"
    pathlines = [
        rfn.append_fields(
            pl, fieldname, list(repeat(fieldval, len(pl))), dtypes="|S1"
        )
        for pl in pathlines
    ]

    # write the pathline recarray to shapefile
    pathline_file.write_shapefile(
        pathline_data=pathlines,
        shpname=shp_file,
        one_per_particle=False,
        mg=grid,
    )

    # make sure shapefile exists
    assert shp_file.is_file()

    # load shapefile
    with Reader(shp_file) as reader:
        fieldnames = [f[0] for f in reader.fields[1:]]
        fieldname = "newfiname_" if longfieldname else fieldname
        assert fieldname in fieldnames
        assert all(r[fieldname] == fieldval for r in reader.iterRecords())
