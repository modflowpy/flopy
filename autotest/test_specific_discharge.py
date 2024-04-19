# Test postprocessing and plotting functions related to specific discharge:
# - get_extended_budget()
# - get_specific_discharge()
# - PlotMapView.plot_vector()
# - PlotCrossSection.plot_vector()

# More precisely:
# - two models are created: one for mf2005 and one for mf6
# - the two models are virtually identical; in fact, the options are such that
#   the calculated heads are indeed exactly the same (which is, by the way,
#   quite remarkable!)
# - the model is a very small synthetic test case that just contains enough
#   things to allow for the functions to be thoroughly tested

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.quiver import Quiver

import flopy.utils.binaryfile as bf
from flopy.mf6 import (
    MFSimulation,
    ModflowGwf,
    ModflowGwfdis,
    ModflowGwfdrn,
    ModflowGwfghb,
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
    ModflowDrn,
    ModflowGhb,
    ModflowLpf,
    ModflowOc,
    ModflowPcg,
    ModflowRch,
    ModflowRiv,
    ModflowWel,
)
from flopy.plot import PlotCrossSection, PlotMapView
from flopy.utils.postprocessing import (
    get_extended_budget,
    get_specific_discharge,
)

# model domain, grid definition and properties
Lx = 100.0
Ly = 100.0
ztop = 0.0
zbot = -100.0
nlay = 4
nrow = 4
ncol = 4
delr = Lx / ncol
delc = Ly / nrow
delv = (ztop - zbot) / nlay
botm = np.linspace(ztop, zbot, nlay + 1)
hk = 1.0
rchrate = 0.1
lay_to_plot = 1

# variables for the BAS (mf2005) or DIS (mf6) package
ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)
ibound[1, 0, 1] = 0  # set a no-flow cell
strt = np.ones((nlay, nrow, ncol), dtype=np.float32)

# add inflow through west boundary using WEL package
Q = 100.0
wel_list = []
wel_list_iface = []
for k in range(nlay):
    for i in range(nrow):
        wel_list.append([k, i, 0, Q])
        wel_list_iface.append(wel_list[-1] + [1])

# allow flow through north, south, and bottom boundaries using GHB package
ghb_head = -30.0  # low enough to have dry cells in first layer
ghb_cond = hk * delr * delv / (0.5 * delc)
ghb_list = []
ghb_list_iface = []
for k in range(1, nlay):
    for j in range(ncol):
        if not (k == 1 and j == 1):  # skip no-flow cell
            ghb_list.append([k, 0, j, ghb_head, ghb_cond])
            ghb_list_iface.append(ghb_list[-1] + [4])
        ghb_list.append([k, nrow - 1, j, ghb_head, ghb_cond])
        ghb_list_iface.append(ghb_list[-1] + [3])
for i in range(nrow):
    for j in range(ncol):
        ghb_list.append([nlay - 1, i, j, ghb_head, ghb_cond])
        ghb_list_iface.append(ghb_list[-1] + [5])

# river in the eastern part
riv_stage = -30.0
riv_cond = hk * delr * delc / (0.5 * delv)
riv_rbot = riv_stage - 5.0
riv_list = []
for i in range(nrow):
    riv_list.append([1, i, ncol - 1, riv_stage, riv_cond, riv_rbot])

# drain in the south part
drn_stage = -30.0
drn_cond = hk * delc * delv / (0.5 * delr)
drn_list = []
for j in range(ncol):
    drn_list.append([1, i, nrow - 1, drn_stage, drn_cond])

boundary_ifaces = {
    "WELLS": wel_list_iface,
    "HEAD DEP BOUNDS": ghb_list_iface,
    "RIVER LEAKAGE": 2,
    "DRAIN": 3,
    "RECHARGE": 6,
}


@pytest.fixture
def mf2005_model(function_tmpdir):
    # create modflow model
    mf = Modflow("mf2005", model_ws=function_tmpdir, exe_name="mf2005")

    # cell by cell flow file unit number
    cbc_unit_nb = 53

    # create DIS package
    dis = ModflowDis(
        mf, nlay, nrow, ncol, delr=delr, delc=delc, top=ztop, botm=botm[1:]
    )

    # create BAS package
    bas = ModflowBas(mf, ibound=ibound, strt=strt)

    # create LPF package
    laytyp = np.zeros(nlay)
    laytyp[0] = 1
    laywet = np.zeros(nlay)
    laywet[0] = 1
    lpf = ModflowLpf(
        mf,
        hk=hk,
        ipakcb=cbc_unit_nb,
        laytyp=laytyp,
        laywet=laywet,
        wetdry=-0.01,
    )

    # create WEL package
    wel_dict = {0: wel_list}
    wel = ModflowWel(mf, stress_period_data=wel_dict, ipakcb=cbc_unit_nb)

    # create GHB package
    ghb_dict = {0: ghb_list}
    ghb = ModflowGhb(mf, stress_period_data=ghb_dict, ipakcb=cbc_unit_nb)

    # create RIV package
    riv_dict = {0: riv_list}
    riv = ModflowRiv(mf, stress_period_data=riv_dict, ipakcb=cbc_unit_nb)

    # create DRN package
    drn_dict = {0: drn_list}
    drn = ModflowDrn(mf, stress_period_data=drn_dict, ipakcb=cbc_unit_nb)

    # create RCH package
    rch = ModflowRch(mf, rech=rchrate, ipakcb=cbc_unit_nb)

    # create OC package
    spd = {(0, 0): ["print head", "print budget", "save head", "save budget"]}
    oc = ModflowOc(mf, stress_period_data=spd, compact=True)

    # create PCG package
    pcg = ModflowPcg(mf)

    # write the MODFLOW model input files
    mf.write_input()

    return mf, function_tmpdir


@pytest.fixture
def mf6_model(function_tmpdir):
    # create simulation
    sim = MFSimulation(
        sim_name="mf6",
        version="mf6",
        exe_name="mf6",
        sim_ws=function_tmpdir,
    )

    # create tdis package
    tdis_rc = [(1.0, 1, 1.0)]
    tdis = ModflowTdis(
        sim, pname="tdis", time_units="DAYS", perioddata=tdis_rc
    )

    # create gwf model
    gwf = ModflowGwf(
        sim,
        modelname="mf6",
        model_nam_file="mf6.nam",
    )
    gwf.name_file.save_flows = True

    # create iterative model solution and register the gwf model with it
    rcloserecord = [1e-5, "STRICT"]
    ims = ModflowIms(
        sim,
        pname="ims",
        print_option="SUMMARY",
        complexity="SIMPLE",
        outer_hclose=1.0e-5,
        outer_maximum=50,
        under_relaxation="NONE",
        inner_maximum=30,
        inner_hclose=1.0e-5,
        rcloserecord=rcloserecord,
        linear_acceleration="CG",
        scaling_method="NONE",
        reordering_method="NONE",
        relaxation_factor=0.99,
    )
    sim.register_ims_package(ims, [gwf.name])

    # create dis package
    dis = ModflowGwfdis(
        gwf,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=ztop,
        botm=botm[1:],
        idomain=ibound,
    )

    # initial conditions
    ic = ModflowGwfic(gwf, pname="ic", strt=strt)

    # create node property flow package
    rewet_record = [("WETFCT", 0.1, "IWETIT", 1, "IHDWET", 0)]
    icelltype = np.zeros(ibound.shape)
    icelltype[0, :, :] = 1
    wetdry = np.zeros(ibound.shape)
    wetdry[0, :, :] = -0.01
    npf = ModflowGwfnpf(
        gwf,
        icelltype=icelltype,
        k=hk,
        rewet_record=rewet_record,
        wetdry=wetdry,
        cvoptions=True,
        save_specific_discharge=True,
    )

    # create wel package
    welspd = [[(wel_i[0], wel_i[1], wel_i[2]), wel_i[3]] for wel_i in wel_list]
    wel = ModflowGwfwel(gwf, print_input=True, stress_period_data=welspd)

    # create ghb package
    ghbspd = [
        [(ghb_i[0], ghb_i[1], ghb_i[2]), ghb_i[3], ghb_i[4]]
        for ghb_i in ghb_list
    ]
    ghb = ModflowGwfghb(gwf, print_input=True, stress_period_data=ghbspd)

    # create riv package
    rivspd = [
        [(riv_i[0], riv_i[1], riv_i[2]), riv_i[3], riv_i[4], riv_i[5]]
        for riv_i in riv_list
    ]
    riv = ModflowGwfriv(gwf, stress_period_data=rivspd)

    # create drn package
    drnspd = [
        [(drn_i[0], drn_i[1], drn_i[2]), drn_i[3], drn_i[4]]
        for drn_i in drn_list
    ]
    drn = ModflowGwfdrn(gwf, print_input=True, stress_period_data=drnspd)

    # create rch package
    rch = ModflowGwfrcha(gwf, recharge=rchrate)

    # output control
    oc = ModflowGwfoc(
        gwf,
        pname="oc",
        budget_filerecord="mf6.cbc",
        head_filerecord="mf6.hds",
        headprintrecord=[("COLUMNS", 10, "WIDTH", 15, "DIGITS", 6, "GENERAL")],
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        printrecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )

    # write input files
    sim.write_simulation()

    return sim, function_tmpdir


def basic_check(Qx_ext, Qy_ext, Qz_ext):
    # check shape
    assert Qx_ext.shape == (nlay, nrow, ncol + 1)
    assert Qy_ext.shape == (nlay, nrow + 1, ncol)
    assert Qz_ext.shape == (nlay + 1, nrow, ncol)

    # check sign
    assert Qx_ext[2, 1, 1] > 0
    assert Qy_ext[2, 1, 1] > 0
    assert Qz_ext[2, 1, 1] < 0


def local_balance_check(Qx_ext, Qy_ext, Qz_ext, hdsfile=None, model=None):
    # calculate water balance at every cell
    local_balance = (
        Qx_ext[:, :, :-1]
        - Qx_ext[:, :, 1:]
        + Qy_ext[:, 1:, :]
        - Qy_ext[:, :-1, :]
        + Qz_ext[1:, :, :]
        - Qz_ext[:-1, :, :]
    )

    # calculate total flow through every cell
    local_total = (
        np.abs(Qx_ext[:, :, :-1])
        + np.abs(Qx_ext[:, :, 1:])
        + np.abs(Qy_ext[:, 1:, :])
        + np.abs(Qy_ext[:, :-1, :])
        + np.abs(Qz_ext[1:, :, :])
        + np.abs(Qz_ext[:-1, :, :])
    )

    # we should disregard no-flow and dry cells
    if hdsfile is not None and model is not None:
        hds = bf.HeadFile(hdsfile, precision="single")
        head = hds.get_data()
        noflo_or_dry = np.logical_or(head == model.hnoflo, head == model.hdry)
        local_balance[noflo_or_dry] = np.nan

    # check water balance = 0 at every cell
    rel_err = local_balance / local_total
    max_rel_err = np.nanmax(rel_err)
    assert np.allclose(max_rel_err + 1.0, 1.0)


@pytest.mark.xfail(reason="occasional Unexpected collection type")
def test_extended_budget_default(mf2005_model):
    # build and run MODFLOW 2005 model
    mf, function_tmpdir = mf2005_model
    mf.run_model()

    # load and postprocess
    Qx_ext, Qy_ext, Qz_ext = get_extended_budget(
        function_tmpdir / "mf2005.cbc"
    )

    # basic check
    basic_check(Qx_ext, Qy_ext, Qz_ext)

    # overall check
    overall = np.sum(Qx_ext) + np.sum(Qy_ext) + np.sum(Qz_ext)
    assert np.allclose(overall, -1122.4931640625)

    # call other evaluations
    specific_discharge_default(function_tmpdir)
    specific_discharge_comprehensive(function_tmpdir)


def extended_budget_comprehensive(function_tmpdir):
    # load and postprocess
    mf = Modflow.load(function_tmpdir / "mf2005.nam", check=False)
    Qx_ext, Qy_ext, Qz_ext = get_extended_budget(
        function_tmpdir / "mf2005.cbc",
        boundary_ifaces=boundary_ifaces,
        hdsfile=function_tmpdir / "mf2005.hds",
        model=mf,
    )

    # basic check
    basic_check(Qx_ext, Qy_ext, Qz_ext)

    # local balance check
    local_balance_check(
        Qx_ext, Qy_ext, Qz_ext, function_tmpdir / "mf2005.hds", mf
    )

    # overall check
    overall = np.sum(Qx_ext) + np.sum(Qy_ext) + np.sum(Qz_ext)
    assert np.allclose(overall, -1110.646240234375)


def specific_discharge_default(function_tmpdir):
    # load and postprocess
    mf = Modflow.load(function_tmpdir / "mf2005.nam", check=False)
    cbc = bf.CellBudgetFile(function_tmpdir / "mf2005.cbc")
    keys = ["FLOW RIGHT FACE", "FLOW FRONT FACE", "FLOW LOWER FACE"]
    vectors = [cbc.get_data(text=t)[0] for t in keys]
    qx, qy, qz = get_specific_discharge(vectors, mf)

    # overall check
    overall = np.sum(qx) + np.sum(qy) + np.sum(qz)
    assert np.allclose(overall, -1.7959892749786377)


def specific_discharge_comprehensive(function_tmpdir):
    hds = bf.HeadFile(function_tmpdir / "mf2005.hds")
    head = hds.get_data()
    # load and postprocess
    mf = Modflow.load(function_tmpdir / "mf2005.nam", check=False)
    Qx_ext, Qy_ext, Qz_ext = get_extended_budget(
        function_tmpdir / "mf2005.cbc",
        boundary_ifaces=boundary_ifaces,
        hdsfile=function_tmpdir / "mf2005.hds",
        model=mf,
    )

    qx, qy, qz = get_specific_discharge((Qx_ext, Qy_ext, Qz_ext), mf, head)

    # check nan values
    assert np.isnan(qx[0, 0, 2])
    assert np.isnan(qx[1, 0, 1])

    # overall check
    overall = np.nansum(qz)  # np.nansum(qx) + np.nansum(qy) + np.nansum(qz)
    assert np.allclose(overall, -4.43224582939148)

    # plot discharge in map view
    lay = 1
    modelmap = PlotMapView(model=mf, layer=lay)
    quiver = modelmap.plot_vector(
        qx, qy, normalize=True, masked_values=[qx[lay, 0, 0]], color="orange"
    )
    # check plot
    ax = modelmap.ax
    if len(ax.collections) == 0:
        raise AssertionError("Discharge vector was not drawn")
    for col in ax.collections:
        if not isinstance(col, Quiver):
            raise AssertionError(f"Unexpected collection type: {type(col)}")
    assert np.sum(quiver.Umask) == 4
    pos = np.sum(quiver.X) + np.sum(quiver.Y)
    assert np.allclose(pos, 1600.0)
    val = np.sum(quiver.U) + np.sum(quiver.V)
    assert np.allclose(val, 12.0225525)

    # close figure
    plt.close()

    # plot discharge in cross-section view
    hds = bf.HeadFile(function_tmpdir / "mf2005.hds", precision="single")
    head = hds.get_data()
    row = 0
    xsect = PlotCrossSection(model=mf, line={"row": row})
    quiver = xsect.plot_vector(
        qx,
        qy,
        qz,
        head=head,
        normalize=True,
        masked_values=qx[0, row, :2],
        color="orange",
    )

    # check plot
    ax = xsect.ax
    if len(ax.collections) == 0:
        raise AssertionError("Discharge vector was not drawn")
    for col in ax.collections:
        if not isinstance(col, Quiver):
            raise AssertionError(f"Unexpected collection type: {type(col)}")
    assert np.sum(quiver.Umask) == 7
    X = np.ma.masked_where(quiver.Umask, quiver.X)
    Y = np.ma.masked_where(quiver.Umask, quiver.Y)
    pos = X.sum() + Y.sum()
    assert np.allclose(pos, -152.0747652053833)
    U = np.ma.masked_where(quiver.Umask, quiver.U)
    V = np.ma.masked_where(quiver.Umask, quiver.V)
    val = U.sum() + V.sum()
    assert np.allclose(val, -3.3428158026088326)

    # close figure
    plt.close()


@pytest.mark.mf6
@pytest.mark.xfail(
    reason="occasional Unexpected collection type: <class 'matplotlib.collections.LineCollection'>"
)
def test_specific_discharge_mf6(mf6_model):
    # build and run MODFLOW 6 model
    sim, function_tmpdir = mf6_model
    sim.run_simulation()

    # load and postprocess
    sim = MFSimulation.load(
        sim_name="mf6", sim_ws=function_tmpdir, verbosity_level=0
    )
    gwf = sim.get_model("mf6")
    hds = bf.HeadFile(function_tmpdir / "mf6.hds")
    head = hds.get_data()
    cbc = bf.CellBudgetFile(function_tmpdir / "mf6.cbc")
    spdis = cbc.get_data(text="SPDIS")[0]
    qx, qy, qz = get_specific_discharge(spdis, gwf, head)

    # check nan values
    assert np.isnan(qx[0, 0, 2])
    assert np.isnan(qx[1, 0, 1])

    # overall check
    overall = np.nansum(qx) + np.nansum(qy) + np.nansum(qz)
    assert np.allclose(overall, -2.5768726154495947)

    # plot discharge in map view
    lay = 1
    modelmap = PlotMapView(model=gwf, layer=lay)
    quiver = modelmap.plot_vector(qx, qy, normalize=True)

    # check plot
    ax = modelmap.ax
    assert len(ax.collections) != 0, "Discharge vector was not drawn"
    for col in ax.collections:
        assert isinstance(
            col, Quiver
        ), f"Unexpected collection type: {type(col)}"
    assert np.sum(quiver.Umask) == 1
    pos = np.sum(quiver.X) + np.sum(quiver.Y)
    assert np.allclose(pos, 1600.0)
    val = np.sum(quiver.U) + np.sum(quiver.V)
    assert np.allclose(val, 11.10085455942011)

    # close figure
    plt.close()

    # plot discharge in cross-section view
    hds = bf.HeadFile(function_tmpdir / "mf6.hds", precision="double")
    head = hds.get_data()
    row = 0
    xsect = PlotCrossSection(model=gwf, line={"row": row})
    quiver = xsect.plot_vector(qx, qy, qz, head=head, normalize=True)

    # check plot
    ax = xsect.ax
    if len(ax.collections) == 0:
        raise AssertionError("Discharge vector was not drawn")
    for col in ax.collections:
        if not isinstance(col, Quiver):
            raise AssertionError(f"Unexpected collection type: {type(col)}")
    assert np.sum(quiver.Umask) == 3
    X = np.ma.masked_where(quiver.Umask, quiver.X)
    Y = np.ma.masked_where(quiver.Umask, quiver.Y)
    pos = X.sum() + Y.sum()
    assert np.allclose(pos, -145.94665962387785)
    U = np.ma.masked_where(quiver.Umask, quiver.U)
    V = np.ma.masked_where(quiver.Umask, quiver.V)
    val = U.sum() + V.sum()
    assert np.allclose(val, -4.49596527119806)

    # close figure
    plt.close()
