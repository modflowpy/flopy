import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from modflow_devtools.markers import requires_exe

from flopy.mf6 import (
    MFSimulation,
    ModflowGwf,
    ModflowGwfchd,
    ModflowGwfdis,
    ModflowGwfic,
    ModflowGwfnpf,
    ModflowGwfoc,
    ModflowIms,
    ModflowTdis,
)
from flopy.mf6.utils import get_residuals, get_structured_faceflows
from flopy.modflow import Modflow, ModflowDis, ModflowLpf, ModflowUpw
from flopy.plot import PlotMapView
from flopy.utils import get_transmissivities
from flopy.utils.postprocessing import (
    get_gradients,
    get_specific_discharge,
    get_water_table,
)


@pytest.fixture(scope="module")
def mf6_freyberg_path(example_data_path):
    return example_data_path / "mf6-freyberg"


@pytest.mark.mf6
@requires_exe("mf6")
def test_faceflows(function_tmpdir, mf6_freyberg_path):
    sim = MFSimulation.load(
        sim_name="freyberg",
        exe_name="mf6",
        sim_ws=mf6_freyberg_path,
    )

    # change the simulation workspace
    sim.set_sim_path(function_tmpdir)

    # write the model simulation files
    sim.write_simulation()

    # run the simulation
    sim.run_simulation()

    # get output
    gwf = sim.get_model("freyberg")
    head = gwf.output.head().get_data()
    cbc = gwf.output.budget()

    spdis = cbc.get_data(text="DATA-SPDIS")[0]
    flowja = cbc.get_data(text="FLOW-JA-FACE")[0]

    frf, fff, flf = get_structured_faceflows(
        flowja,
        grb_file=function_tmpdir / "freyberg.dis.grb",
    )
    Qx, Qy, Qz = get_specific_discharge(
        (frf, fff, flf),
        gwf,
    )
    sqx, sqy, sqz = get_specific_discharge(
        (frf, fff, flf),
        gwf,
        head=head,
    )
    qx, qy, qz = get_specific_discharge(spdis, gwf)

    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    ax = fig.add_subplot(1, 3, 1, aspect="equal")
    mm = PlotMapView(model=gwf, ax=ax)
    Q0 = mm.plot_vector(Qx, Qy)
    assert isinstance(Q0, matplotlib.quiver.Quiver)

    ax = fig.add_subplot(1, 3, 2, aspect="equal")
    mm = PlotMapView(model=gwf, ax=ax)
    q0 = mm.plot_vector(sqx, sqy)
    assert isinstance(q0, matplotlib.quiver.Quiver)

    ax = fig.add_subplot(1, 3, 3, aspect="equal")
    mm = PlotMapView(model=gwf, ax=ax)
    q1 = mm.plot_vector(qx, qy)
    assert isinstance(q1, matplotlib.quiver.Quiver)

    plt.close("all")

    # uv0 = np.column_stack((q0.U, q0.V))
    # uv1 = np.column_stack((q1.U, q1.V))
    # diff = uv1 - uv0
    # assert (
    #     np.allclose(uv0, uv1)
    # ), "get_faceflows quivers are not equal to specific discharge vectors"
    return


@pytest.mark.mf6
@requires_exe("mf6")
def test_flowja_residuals(function_tmpdir, mf6_freyberg_path):
    sim = MFSimulation.load(
        sim_name="freyberg",
        exe_name="mf6",
        sim_ws=mf6_freyberg_path,
    )

    # change the simulation workspace
    sim.set_sim_path(function_tmpdir)

    # write the model simulation files
    sim.write_simulation()

    # run the simulation
    sim.run_simulation()

    # get output
    gwf = sim.get_model("freyberg")
    grb_file = function_tmpdir / "freyberg.dis.grb"
    cbc = gwf.output.budget()

    spdis = cbc.get_data(text="DATA-SPDIS")[0]
    flowja = cbc.get_data(text="FLOW-JA-FACE")[0]

    residual = get_residuals(flowja, grb_file=grb_file)
    qx, qy, qz = get_specific_discharge(spdis, gwf)

    fig = plt.figure(figsize=(6, 9), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1, aspect="equal")
    mm = PlotMapView(model=gwf, ax=ax)
    r0 = mm.plot_array(residual)
    assert isinstance(
        r0, matplotlib.collections.QuadMesh
    ), "r0 not type matplotlib.collections.QuadMesh"
    q0 = mm.plot_vector(qx, qy)
    assert isinstance(
        q0, matplotlib.quiver.Quiver
    ), "q0 not type matplotlib.quiver.Quiver"
    mm.plot_grid(lw=0.5, color="black")
    mm.plot_ibound()
    plt.colorbar(r0, shrink=0.5)
    plt.title("Cell Residual, cubic meters per second")

    plt.close("all")


@pytest.mark.mf6
@requires_exe("mf6")
def test_structured_faceflows_3d(function_tmpdir):
    name = "mymodel"
    sim = MFSimulation(sim_name=name, sim_ws=function_tmpdir, exe_name="mf6")
    tdis = ModflowTdis(sim)
    ims = ModflowIms(sim)
    gwf = ModflowGwf(sim, modelname=name, save_flows=True)
    dis = ModflowGwfdis(
        gwf, nlay=3, nrow=10, ncol=10, top=0, botm=[-1, -2, -3]
    )
    ic = ModflowGwfic(gwf)
    npf = ModflowGwfnpf(gwf, save_specific_discharge=True)
    chd = ModflowGwfchd(
        gwf, stress_period_data=[[(0, 0, 0), 1.0], [(0, 9, 9), 0.0]]
    )
    budget_file = name + ".bud"
    head_file = name + ".hds"
    oc = ModflowGwfoc(
        gwf,
        budget_filerecord=budget_file,
        head_filerecord=head_file,
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )
    sim.write_simulation()
    sim.run_simulation()

    head = gwf.output.head().get_data()
    bud = gwf.output.budget()
    flowja = bud.get_data(text="FLOW-JA-FACE")[0]
    frf, fff, flf = get_structured_faceflows(
        flowja,
        grb_file=function_tmpdir / "mymodel.dis.grb",
    )
    assert (
        frf.shape == head.shape
    ), f"frf.shape {frf.shape} != head.shape {head.shape}"
    assert (
        fff.shape == head.shape
    ), f"frf.shape {frf.shape} != head.shape {head.shape}"
    assert (
        flf.shape == head.shape
    ), f"frf.shape {frf.shape} != head.shape {head.shape}"


def test_get_transmissivities(function_tmpdir):
    sctop = [-0.25, 0.5, 1.7, 1.5, 3.0, 2.5, 3.0, -10.0]
    scbot = [-1.0, -0.5, 1.2, 0.5, 1.5, -0.2, 2.5, -11.0]
    heads = np.array(
        [
            [1.0, 2.0, 2.05, 3.0, 4.0, 2.5, 2.5, 2.5],
            [1.1, 2.1, 2.2, 2.0, 3.5, 3.0, 3.0, 3.0],
            [1.2, 2.3, 2.4, 0.6, 3.4, 3.2, 3.2, 3.2],
        ]
    )
    nl, nr = heads.shape
    nc = nr
    botm = np.ones((nl, nr, nc), dtype=float)
    top = np.ones((nr, nc), dtype=float) * 2.1
    hk = np.ones((nl, nr, nc), dtype=float) * 2.0
    for i in range(nl):
        botm[nl - i - 1, :, :] = i

    m = Modflow("junk", version="mfnwt", model_ws=function_tmpdir)
    dis = ModflowDis(m, nlay=nl, nrow=nr, ncol=nc, botm=botm, top=top)
    upw = ModflowUpw(m, hk=hk)

    # test with open intervals
    r, c = np.arange(nr), np.arange(nc)
    T = get_transmissivities(heads, m, r=r, c=c, sctop=sctop, scbot=scbot)
    assert (
        T
        - np.array(
            [
                [0.0, 0, 0.0, 0.0, 0.2, 0.2, 2.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 0.0, 0.0],
                [2.0, 1.0, 0.0, 0.2, 0.0, 2.0, 0.0, 2.0],
            ]
        )
    ).sum() < 1e-3

    # test without specifying open intervals
    T = get_transmissivities(heads, m, r=r, c=c)
    assert (
        T
        - np.array(
            [
                [0.0, 0.0, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2],
                [0.2, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0, 1.2, 2.0, 2.0, 2.0, 2.0],
            ]
        )
    ).sum() < 1e-3


def test_get_water_table():
    hdry = -1e30
    hds = np.ones((3, 3, 3), dtype=float) * hdry
    hds[-1, :, :] = 2.0
    hds[1, 1, 1] = 1.0
    hds[0, -1, -1] = 1e30
    wt = get_water_table(hds)
    assert wt.shape == (3, 3)
    assert wt[1, 1] == 1.0
    assert np.sum(wt) == 17.0

    hdry = -9999
    hds = np.ones((3, 3, 3), dtype=float) * hdry
    hds[-1, :, :] = 2.0
    hds[1, 1, 1] = 1.0
    hds[0, -1, -1] = 9999
    wt = get_water_table(hds, hdry=-9999, hnoflo=9999)
    assert wt.shape == (3, 3)
    assert wt[1, 1] == 1.0
    assert np.sum(wt) == 17.0

    hds2 = np.array([hds, hds])
    wt = get_water_table(hds2, hdry=-9999, hnoflo=9999)
    assert wt.shape == (2, 3, 3)
    assert np.sum(wt[:, 1, 1]) == 2.0
    assert np.sum(wt) == 34.0


def test_get_sat_thickness_gradients(function_tmpdir):
    nodata = -9999.0
    hds = np.ones((3, 3, 3), dtype=float) * nodata
    hds[1, :, :] = 2.4
    hds[0, 1, 1] = 3.2
    hds[2, :, :] = 2.5
    hds[1, 1, 1] = 3.0
    hds[2, 1, 1] = 2.6

    nl, nr, nc = hds.shape
    botm = np.ones((nl, nr, nc), dtype=float)
    top = np.ones((nr, nc), dtype=float) * 4.0
    botm[0, :, :] = 3.0
    botm[1, :, :] = 2.0

    m = Modflow("junk", version="mfnwt", model_ws=function_tmpdir)
    dis = ModflowDis(m, nlay=nl, nrow=nr, ncol=nc, botm=botm, top=top)
    lpf = ModflowLpf(m, laytyp=np.ones(nl))

    grad = get_gradients(hds, m, nodata=nodata)
    dh = np.diff(hds[:, 1, 1])
    dz = np.array([-0.7, -1.0])
    assert np.abs(dh / dz - grad[:, 1, 1]).sum() < 1e-6
    dh = np.diff(hds[:, 1, 0])
    dz = np.array([np.nan, -0.9])
    assert np.nansum(np.abs(dh / dz - grad[:, 1, 0])) < 1e-6

    sat_thick = m.modelgrid.saturated_thick(hds, mask=nodata)
    assert (
        np.abs(np.sum(sat_thick[:, 1, 1] - np.array([0.2, 1.0, 1.0]))) < 1e-6
    ), "failed saturated thickness comparison (grid.thick())"
