import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from modflow_devtools.markers import requires_exe

import flopy
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


@pytest.fixture
def mf2005_freyberg_path(example_data_path):
    return example_data_path / "freyberg"


@pytest.fixture
def mf6_freyberg_path(example_data_path):
    return example_data_path / "mf6-freyberg"


@pytest.mark.parametrize(
    "nlay, nrow, ncol",
    [
        # extended in 1 dimension
        [3, 1, 1],
        [1, 3, 1],
        [1, 1, 3],
        # 2D
        [3, 3, 1],
        [1, 3, 3],
        [3, 1, 3],
        # 3D
        [3, 3, 3],
    ],
)
@pytest.mark.mf6
@requires_exe("mf6")
def test_get_structured_faceflows(function_tmpdir, nlay, nrow, ncol):
    name = "gsff"
    sim = flopy.mf6.MFSimulation(
        sim_name=name, exe_name="mf6", version="mf6", sim_ws=function_tmpdir
    )

    # tdis
    tdis = flopy.mf6.ModflowTdis(
        sim,
        nper=1,
        perioddata=[(1.0, 1, 1.0)],
    )

    # gwf
    gwf = flopy.mf6.ModflowGwf(
        sim,
        modelname=name,
        model_nam_file="{}.nam".format(name),
        save_flows=True,
    )

    # dis
    botm = (
        np.ones((nlay, nrow, ncol))
        * np.arange(nlay - 1, -1, -1)[:, np.newaxis, np.newaxis]
    )
    dis = flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        top=nlay,
        botm=botm,
    )

    # initial conditions
    h0 = nlay * 2
    start = h0 * np.ones((nlay, nrow, ncol))
    ic = flopy.mf6.ModflowGwfic(gwf, pname="ic", strt=start)

    # constant head
    chd_rec = []
    max_dim = max(nlay, nrow, ncol)
    h = np.linspace(11, 13, max_dim)
    iface = 6  # top
    for i in range(0, max_dim):
        # ((layer,row,col),head,iface)
        cell_id = (0, 0, i) if ncol > 1 else (0, i, 0) if nrow > 1 else (i, 0, 0)
        chd_rec.append((cell_id, h[i], iface))
    chd = flopy.mf6.ModflowGwfchd(
        gwf,
        auxiliary=[("iface",)],
        stress_period_data=chd_rec,
        print_input=True,
        print_flows=True,
        save_flows=True,
    )

    # node property flow
    npf = flopy.mf6.ModflowGwfnpf(gwf, save_specific_discharge=True)

    # output control
    budgetfile = "{}.cbb".format(name)
    budget_filerecord = [budgetfile]
    saverecord = [("BUDGET", "ALL")]
    oc = flopy.mf6.ModflowGwfoc(
        gwf,
        saverecord=saverecord,
        budget_filerecord=budget_filerecord,
    )

    # solver
    ims = flopy.mf6.ModflowIms(sim)

    # write and run the model
    sim.write_simulation()
    sim.check()
    success, buff = sim.run_simulation()
    assert success

    # load budget output
    budget = gwf.output.budget()
    flow_ja_face = budget.get_data(text="FLOW-JA-FACE")[0]
    frf, fff, flf = get_structured_faceflows(
        flow_ja_face,
        grb_file=function_tmpdir / f"{gwf.name}.dis.grb",
        verbose=True,
    )

    # expect nonzero flows only in extended (>1 cell) dimensions
    assert np.any(frf) == (ncol > 1)
    assert np.any(fff) == (nrow > 1)
    assert np.any(flf) == (nlay > 1)


@pytest.mark.mf6
@requires_exe("mf6")
def test_get_structured_faceflows_freyberg(
    function_tmpdir, mf2005_freyberg_path, mf6_freyberg_path
):
    # create workspaces
    mf6_ws = function_tmpdir / "mf6"
    mf2005_ws = function_tmpdir / "mf2005"

    # run freyberg mf6
    sim = MFSimulation.load(
        sim_name="freyberg",
        exe_name="mf6",
        sim_ws=mf6_freyberg_path,
    )
    sim.set_sim_path(mf6_ws)
    sim.write_simulation()
    sim.run_simulation()

    # get freyberg mf6 output and compute structured faceflows
    gwf = sim.get_model("freyberg")
    mf6_head = gwf.output.head().get_data()
    mf6_cbc = gwf.output.budget()
    mf6_spdis = mf6_cbc.get_data(text="DATA-SPDIS")[0]
    mf6_flowja = mf6_cbc.get_data(text="FLOW-JA-FACE")[0]
    mf6_frf, mf6_fff, mf6_flf = get_structured_faceflows(
        mf6_flowja,
        grb_file=mf6_ws / "freyberg.dis.grb",
    )
    assert mf6_frf.shape == mf6_fff.shape == mf6_flf.shape == mf6_head.shape
    assert not np.any(mf6_flf)  # only 1 layer

    # run freyberg mf2005
    model = Modflow.load("freyberg", model_ws=mf2005_freyberg_path)
    model.change_model_ws(mf2005_ws)
    model.write_input()
    model.run_model()

    # get freyberg mf2005 output
    mf2005_cbc = flopy.utils.CellBudgetFile(mf2005_ws / "freyberg.cbc")
    mf2005_frf, mf2005_fff = (
        mf2005_cbc.get_data(text="FLOW RIGHT FACE", full3D=True)[0],
        mf2005_cbc.get_data(text="FLOW FRONT FACE", full3D=True)[0],
    )

    # compare mf2005 faceflows with converted mf6 faceflows
    assert mf2005_frf.shape == mf2005_fff.shape == mf6_head.shape
    assert np.allclose(mf6_frf, np.flip(mf2005_frf, 0), atol=1e-3)
    assert np.allclose(mf6_fff, np.flip(mf2005_fff, 0), atol=1e-3)

    Qx, Qy, Qz = get_specific_discharge(
        (mf6_frf, mf6_fff, mf6_flf),
        gwf,
    )
    sqx, sqy, sqz = get_specific_discharge(
        (mf6_frf, mf6_fff, mf6_flf),
        gwf,
        head=mf6_head,
    )
    qx, qy, qz = get_specific_discharge(mf6_spdis, gwf)

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

    # plt.show()
    plt.close("all")


@pytest.mark.mf6
@requires_exe("mf6")
def test_get_structured_faceflows_idomain(
    function_tmpdir,
):
    name = "gsffi"

    Lx = 1000
    Ly = 1000

    ncol = 100
    nrow = 100
    nlay = 3
    top = 60
    botm = [40, 20, 0]

    Qwell = -1000

    # Simulation
    sim = flopy.mf6.MFSimulation(
        sim_name=name,
        version="mf6",
        exe_name="mf6",
        sim_ws=function_tmpdir,
    )

    tdis = flopy.mf6.ModflowTdis(
        simulation=sim,
        time_units="DAYS",
        nper=1,
        perioddata=[(1, 1, 1)],
    )

    ims = flopy.mf6.ModflowIms(
        simulation=sim,
        inner_dvclose=1e-6,
    )

    # Groundwater flow model
    gwf = flopy.mf6.ModflowGwf(
        simulation=sim,
        modelname=name,
        save_flows=True,
    )

    idomain = np.ones((nlay, nrow, ncol))
    for r in range(40, 60):
        for c in range(40, 60):
            idomain[1, r, c] = -1

    dis = flopy.mf6.ModflowGwfdis(
        model=gwf,
        length_units="METERS",
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=Lx / ncol,
        delc=Ly / nrow,
        top=top,
        botm=botm,
        idomain=idomain,
    )

    npf = flopy.mf6.ModflowGwfnpf(
        model=gwf,
        icelltype=[0, 0, 0],
        k=[10, 0.01, 10],
        k33=[1, 0.001, 1],
    )

    well_list = [
        [
            (nlay - 1, nrow // 2, ncol // 2),
            Qwell,
        ]
    ]
    well_spd = {0: well_list}

    wel = flopy.mf6.ModflowGwfwel(
        model=gwf,
        stress_period_data=well_spd,
    )

    chd_list = []
    for r in range(nrow):
        for c in range(ncol):
            chd_list.append(
                [
                    (0, r, c),
                    top,
                ]
            )
    chd_spd = {0: chd_list}

    chd = flopy.mf6.ModflowGwfchd(
        model=gwf,
        stress_period_data=chd_spd,
    )

    ic = flopy.mf6.ModflowGwfic(
        model=gwf,
        strt=top,
    )

    oc = flopy.mf6.ModflowGwfoc(
        model=gwf,
        budget_filerecord=f"{name}.cbc",
        head_filerecord=f"{name}.hds",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )

    sim.write_simulation(silent=True)
    success, _ = sim.run_simulation(silent=True)
    assert success

    cbb = gwf.output.budget()  # get handle to binary budget file
    Qja = cbb.get_data(text="FLOW-JA-FACE")[0]
    cbc = flopy.mf6.utils.get_structured_faceflows(
        Qja, f"{function_tmpdir}/{name}.dis.grb"
    )
    cbf = cbc[2]

    cbf0 = cbf[0, :, :]
    # Sum vertical cell-face flows for all cells in the top aquifer
    Qv_sum = cbf0.sum()
    idx = idomain[1, :, :] == -1
    Qv_wind = cbf0[idx].sum()  # Flow through aquitard window
    Qv_aqui = cbf0[~idx].sum()  # Flow across aquitard

    print(f"Total flow across bottom of upper aquifer {Qv_sum:0.2f} m^3/d")
    print(f"Flow across bottom of upper aquifer to aquitard {Qv_aqui:0.2f} m^3/d")
    print(f"Flow across bottom of upper aquifer to lower aquifer {Qv_wind:0.2f} m^3/d")

    print(np.isclose(-Qwell, Qv_sum, atol=1e-3))
    assert np.isclose(-Qwell, Qv_sum, atol=1e-3)
    assert Qv_wind > Qv_aqui


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
def test_structured_faceflows_3d_shape(function_tmpdir):
    name = "mymodel"
    sim = MFSimulation(sim_name=name, sim_ws=function_tmpdir, exe_name="mf6")
    tdis = ModflowTdis(sim)
    ims = ModflowIms(sim)
    gwf = ModflowGwf(sim, modelname=name, save_flows=True)
    dis = ModflowGwfdis(gwf, nlay=3, nrow=10, ncol=10, top=0, botm=[-1, -2, -3])
    ic = ModflowGwfic(gwf)
    npf = ModflowGwfnpf(gwf, save_specific_discharge=True)
    chd = ModflowGwfchd(gwf, stress_period_data=[[(0, 0, 0), 1.0], [(0, 9, 9), 0.0]])
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
    assert frf.shape == head.shape, f"frf.shape {frf.shape} != head.shape {head.shape}"
    assert fff.shape == head.shape, f"frf.shape {frf.shape} != head.shape {head.shape}"
    assert flf.shape == head.shape, f"frf.shape {frf.shape} != head.shape {head.shape}"


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

    sat_thick = m.modelgrid.saturated_thickness(hds, mask=nodata)
    assert (
        np.abs(np.sum(sat_thick[:, 1, 1] - np.array([0.2, 1.0, 1.0]))) < 1e-6
    ), "failed saturated thickness comparison (grid.thick())"
