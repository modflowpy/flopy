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


@pytest.mark.mf6
@requires_exe("mf6")
def test_get_structured_faceflows_2d_right_lower(function_tmpdir):
    """
    Reproduce https://github.com/modflowpy/flopy/issues/1911
    """
    name = "gsff_2drl"
    sim = flopy.mf6.MFSimulation(
        sim_name=name, exe_name="mf6", version="mf6", sim_ws=function_tmpdir
    )

    # Simulation time:
    tdis = flopy.mf6.ModflowTdis(
        sim,
        pname="tdis",
        time_units="DAYS",
        nper=1,
        perioddata=[(1.0, 1, 1.0)],
    )

    # Nam file
    model_nam_file = "{}.nam".format(name)

    # Groundwater flow object:
    gwf = flopy.mf6.ModflowGwf(
        sim,
        modelname=name,
        model_nam_file=model_nam_file,
        save_flows=True,
    )

    # Grid properties:
    # Lx = 20 #problem length [m]
    Lx = 2000  # problem length [m]
    Ly = 1  # problem width [m]
    H = 10  # aquifer height [m]
    delx = 1  # block size x direction
    dely = 1  # block size y direction
    delz = 1  # block size z direction

    # nlay = 10
    nlay = 10
    ncol = int(Lx / delx)  # number of columns
    nrow = int(Ly / dely)  # number of layers

    # Flopy Discretizetion Objects (DIS)
    bottom_array = np.ones((nlay, nrow, ncol))
    bottom_range = np.arange(nlay - 1, -1, -1)
    bottom_range = bottom_range[:, np.newaxis, np.newaxis]
    bottom_array = bottom_array * bottom_range

    dis = flopy.mf6.ModflowGwfdis(
        gwf,
        xorigin=0.0,
        yorigin=0.0,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=dely,
        delc=delx,
        top=nlay * delz,
        botm=bottom_array,
    )

    # initial conditions
    h0 = nlay * 2
    start = h0 * np.ones((nlay, nrow, ncol))
    ic = flopy.mf6.ModflowGwfic(gwf, pname="ic", strt=start)

    # node property flow
    k = 1e-4 * np.ones((nlay, nrow, ncol))
    k[1:3, :, 300:1701] = 1e-8
    npf = flopy.mf6.ModflowGwfnpf(
        gwf,
        icelltype=0,  # This we define the model as convertible (water table aquifer)
        k=k,
    )

    # constant head
    chd_rec = []
    h = np.linspace(11, 13, ncol)
    i = 0
    for col in range(0, ncol):
        # ((layer,row,col),head,iface)
        chd_rec.append(((0, 0, col), h[i], 6))
        i += 1
    chd = flopy.mf6.ModflowGwfchd(
        gwf,
        auxiliary=[("iface",)],
        stress_period_data=chd_rec,
        print_input=True,
        print_flows=True,
        save_flows=True,
    )

    # output control
    headfile = "{}.hds".format(name)
    head_filerecord = [headfile]
    budgetfile = "{}.cbb".format(name)
    budget_filerecord = [budgetfile]
    saverecord = [("HEAD", "ALL"), ("BUDGET", "ALL")]
    printrecord = [("HEAD", "LAST")]
    oc = flopy.mf6.ModflowGwfoc(
        gwf,
        saverecord=saverecord,
        head_filerecord=head_filerecord,
        budget_filerecord=budget_filerecord,
        printrecord=printrecord,
    )

    # solver
    ims = flopy.mf6.ModflowIms(
        sim,
        pname="ims",
        complexity="SIMPLE",
        outer_maximum=10,
        inner_maximum=1500,
        inner_dvclose=1e-3,
        rcloserecord=[0.01, "STRICT"],
    )

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

    # expect only nonzero right and lower face flows
    assert np.any(frf)
    assert not np.any(fff)
    assert np.any(flf)

    # load head output
    # head = gwf.output.head()
    # head_array = head.get_data()

    # plot map view
    # fig, ax = plt.subplots(1, 1, figsize=(24, 6), constrained_layout=True)
    # ax.set_title("Head Results")
    # mm = flopy.plot.PlotMapView(model=gwf, ax=ax)
    # mm.plot_array(np.log(k))
    # mm.plot_grid(lw=0.05, color="0.5")
    # mm.plot_vector(frf, fff)
    # plt.show()

    # plot cross section
    # fig, ax = plt.subplots(1, 1, figsize=(24, 6), constrained_layout=True)
    # contour_intervals = np.arange(11, 13, 0.5)
    # xc = flopy.plot.PlotCrossSection(model=gwf, ax=ax, line = {"row": 0})
    # pa = xc.plot_array(np.log(k))
    # linecollection = xc.plot_grid(lw=0.05, color="0.5")
    # contours = xc.contour_array(
    #     head_array,
    #     levels=contour_intervals,
    #     colors="black",
    # )
    # xc.plot_vector(frf, fff, flf)
    # ax.clabel(contours, fmt="%2.1f")
    # cb = plt.colorbar(pa, shrink=0.5, ax=ax)
    # plt.show()


@pytest.mark.parametrize(
    "nlay, nrow, ncol",
    [
        [5, 1, 1],
        [1, 5, 1],
        [1, 1, 5],
    ],
)
@pytest.mark.mf6
@requires_exe("mf6")
def test_get_structured_faceflows_1d(function_tmpdir, nlay, nrow, ncol):
    name = "gsff_1d"
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
        cell_id = (
            (0, 0, i) if ncol > 1 else (0, i, 0) if nrow > 1 else (i, 0, 0)
        )
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

    assert np.any(frf) == bool(ncol > 1)
    assert np.any(fff) == bool(nrow > 1)
    assert np.any(flf) == bool(nlay > 1)


@pytest.mark.skip(reason="todo")
@pytest.mark.parametrize(
    "nlay, nrow, ncol",
    [
        [5, 5, 1],
        [1, 5, 5],
        [5, 1, 5],
    ],
)
@pytest.mark.mf6
@requires_exe("mf6")
def test_get_structured_faceflows_2d(function_tmpdir, nlay, nrow, ncol):
    pass


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
    assert not np.any(mf6_flf)

    # run freyberg mf2005
    model = Modflow.load("freyberg", model_ws=mf2005_freyberg_path)
    model.change_model_ws(mf2005_ws)
    model.write_input()
    model.run_model()

    # get freyberg mf2005 output
    mf2005_cbc = flopy.utils.CellBudgetFile(mf2005_ws / "freyberg.cbc")
    # mf2005_spdis = mf2005_cbc.get_data(text="DATA-SPDIS")[0]
    mf2005_frf, mf2005_fff = (
        mf2005_cbc.get_data(text="FLOW RIGHT FACE", full3D=True)[0],
        mf2005_cbc.get_data(text="FLOW FRONT FACE", full3D=True)[0],
    )

    assert mf2005_frf.shape == mf2005_fff.shape == mf6_head.shape
    assert np.allclose(mf6_frf, mf2005_frf)
    assert np.allclose(mf6_fff, mf2005_fff)

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

    # uv0 = np.column_stack((q0.U, q0.V))
    # uv1 = np.column_stack((q1.U, q1.V))
    # diff = uv1 - uv0
    # assert (
    #     np.allclose(uv0, uv1)
    # ), "get_faceflows quivers are not equal to specific discharge vectors"


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

    sat_thick = m.modelgrid.saturated_thickness(hds, mask=nodata)
    assert (
        np.abs(np.sum(sat_thick[:, 1, 1] - np.array([0.2, 1.0, 1.0]))) < 1e-6
    ), "failed saturated thickness comparison (grid.thick())"
