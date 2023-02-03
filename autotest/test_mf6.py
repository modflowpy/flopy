import os

import numpy as np
import pytest
from modflow_devtools.markers import requires_exe

import flopy
from flopy.mf6 import (
    MFModel,
    MFSimulation,
    ModflowGwf,
    ModflowGwfchd,
    ModflowGwfdis,
    ModflowGwfdisu,
    ModflowGwfdisv,
    ModflowGwfdrn,
    ModflowGwfevt,
    ModflowGwfevta,
    ModflowGwfghb,
    ModflowGwfgnc,
    ModflowGwfgwf,
    ModflowGwfgwt,
    ModflowGwfhfb,
    ModflowGwfic,
    ModflowGwflak,
    ModflowGwfmaw,
    ModflowGwfmvr,
    ModflowGwfnam,
    ModflowGwfnpf,
    ModflowGwfoc,
    ModflowGwfrch,
    ModflowGwfrcha,
    ModflowGwfriv,
    ModflowGwfsfr,
    ModflowGwfsto,
    ModflowGwfuzf,
    ModflowGwfwel,
    ModflowGwtadv,
    ModflowGwtdis,
    ModflowGwtic,
    ModflowGwtmst,
    ModflowGwtoc,
    ModflowGwtssm,
    ModflowIms,
    ModflowNam,
    ModflowTdis,
    ModflowUtllaktab,
)
from flopy.mf6.coordinates.modeldimensions import (
    DataDimensions,
    ModelDimensions,
    PackageDimensions,
)
from flopy.mf6.data.mffileaccess import MFFileAccessArray
from flopy.mf6.data.mfstructure import MFDataItemStructure, MFDataStructure
from flopy.mf6.mfbase import MFFileMgmt
from flopy.mf6.modflow import (
    mfgwf,
    mfgwfdis,
    mfgwfdrn,
    mfgwfic,
    mfgwfnpf,
    mfgwfoc,
    mfgwfriv,
    mfgwfsto,
    mfgwfwel,
    mfims,
    mftdis,
)
from flopy.mf6.modflow.mfsimulation import MFSimulationData
from flopy.utils import (
    CellBudgetFile,
    HeadFile,
    Mf6ListBudget,
    Mf6Obs,
    ZoneBudget6,
)
from flopy.utils.observationfile import CsvFile

pytestmark = pytest.mark.mf6


def write_head(
    fbin,
    data,
    kstp=1,
    kper=1,
    pertim=1.0,
    totim=1.0,
    text="            HEAD",
    ilay=1,
):
    dt = np.dtype(
        [
            ("kstp", "i4"),
            ("kper", "i4"),
            ("pertim", "f8"),
            ("totim", "f8"),
            ("text", "a16"),
            ("ncol", "i4"),
            ("nrow", "i4"),
            ("ilay", "i4"),
        ]
    )
    nrow = data.shape[0]
    ncol = data.shape[1]
    h = np.array((kstp, kper, pertim, totim, text, ncol, nrow, ilay), dtype=dt)
    h.tofile(fbin)
    data.tofile(fbin)
    return


def get_gwf_model(sim, gwfname, gwfpath, modelshape, chdspd=None, welspd=None):
    nlay, nrow, ncol = modelshape
    delr = 1.0
    delc = 1.0
    top = 1.0
    botm = [0.0]
    strt = 1.0
    hk = 1.0
    laytyp = 0

    gwf = ModflowGwf(
        sim,
        modelname=gwfname,
        save_flows=True,
    )
    gwf.set_model_relative_path(gwfpath)

    dis = ModflowGwfdis(
        gwf,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
    )

    # initial conditions
    ic = ModflowGwfic(gwf, strt=strt)

    # node property flow
    npf = ModflowGwfnpf(
        gwf,
        icelltype=laytyp,
        k=hk,
        save_specific_discharge=True,
    )

    # chd files
    if chdspd is not None:
        chd = ModflowGwfchd(
            gwf,
            stress_period_data=chdspd,
            save_flows=False,
            pname="CHD-1",
        )

    # wel files
    if welspd is not None:
        wel = ModflowGwfwel(
            gwf,
            print_input=True,
            print_flows=True,
            stress_period_data=welspd,
            save_flows=False,
            auxiliary="CONCENTRATION",
            pname="WEL-1",
        )

    # output control
    oc = ModflowGwfoc(
        gwf,
        budget_filerecord="{}.cbc".format(gwfname),
        head_filerecord="{}.hds".format(gwfname),
        headprintrecord=[("COLUMNS", 10, "WIDTH", 15, "DIGITS", 6, "GENERAL")],
        saverecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
        printrecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
    )
    return gwf


def get_gwt_model(sim, gwtname, gwtpath, modelshape, sourcerecarray=None):
    nlay, nrow, ncol = modelshape
    delr = 1.0
    delc = 1.0
    top = 1.0
    botm = [0.0]
    strt = 1.0
    hk = 1.0
    laytyp = 0

    gwt = MFModel(
        sim,
        model_type="gwt6",
        modelname=gwtname,
        model_rel_path=gwtpath,
    )
    gwt.name_file.save_flows = True

    dis = ModflowGwtdis(
        gwt,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
    )

    # initial conditions
    ic = ModflowGwtic(gwt, strt=0.0)

    # advection
    adv = ModflowGwtadv(gwt, scheme="upstream")

    # mass storage and transfer
    mst = ModflowGwtmst(gwt, porosity=0.1)

    # sources
    ssm = ModflowGwtssm(gwt, sources=sourcerecarray)

    # output control
    oc = ModflowGwtoc(
        gwt,
        budget_filerecord="{}.cbc".format(gwtname),
        concentration_filerecord="{}.ucn".format(gwtname),
        concentrationprintrecord=[
            ("COLUMNS", 10, "WIDTH", 15, "DIGITS", 6, "GENERAL")
        ],
        saverecord=[("CONCENTRATION", "LAST"), ("BUDGET", "LAST")],
        printrecord=[("CONCENTRATION", "LAST"), ("BUDGET", "LAST")],
    )
    return gwt


def test_string_to_file_path():
    import platform

    if platform.system().lower() == "windows":
        unc_path = r"\\server\path\path"
        new_path = MFFileMgmt.string_to_file_path(unc_path)
        assert unc_path == new_path, "UNC path error"

        abs_path = r"C:\Users\some_user\path"
        new_path = MFFileMgmt.string_to_file_path(abs_path)
        assert abs_path == new_path, "Absolute path error"

        rel_path = r"..\path\some_path"
        new_path = MFFileMgmt.string_to_file_path(rel_path)
        assert rel_path == new_path, "Relative path error"

    else:
        abs_path = "/mnt/c/some_user/path"
        new_path = MFFileMgmt.string_to_file_path(abs_path)
        assert abs_path == new_path, "Absolute path error"

        rel_path = "../path/some_path"
        new_path = MFFileMgmt.string_to_file_path(rel_path)
        assert rel_path == new_path, "Relative path error"


def test_subdir(function_tmpdir):
    sim = MFSimulation(sim_ws=str(function_tmpdir))
    tdis = ModflowTdis(sim)
    gwf = ModflowGwf(sim, model_rel_path="level2")
    ims = ModflowIms(sim)
    sim.register_ims_package(ims, [])
    dis = ModflowGwfdis(gwf)
    sim.set_all_data_external(external_data_folder="dat")
    sim.write_simulation()

    sim_r = MFSimulation.load(
        "mfsim.nam",
        sim_ws=sim.simulation_data.mfpath.get_sim_path(),
    )
    gwf_r = sim_r.get_model()
    assert (
        gwf.dis.delc.get_file_entry() == gwf_r.dis.delc.get_file_entry()
    ), "Something wrong with model external paths"

    sim_r.set_all_data_internal()
    sim_r.set_all_data_external(
        external_data_folder=os.path.join("dat", "dat_l2")
    )
    sim_r.write_simulation()

    sim_r2 = MFSimulation.load(
        "mfsim.nam",
        sim_ws=sim_r.simulation_data.mfpath.get_sim_path(),
    )
    gwf_r2 = sim_r.get_model()
    assert (
        gwf_r.dis.delc.get_file_entry() == gwf_r2.dis.delc.get_file_entry()
    ), "Something wrong with model external paths"


def test_binary_read(function_tmpdir):
    test_ex_name = "binary_read"
    nlay = 3
    nrow = 10
    ncol = 10

    modelgrid = flopy.discretization.StructuredGrid(
        nlay=nlay, nrow=nrow, ncol=ncol
    )

    arr = np.arange(nlay * nrow * ncol).astype(np.float64)
    data_shape = (nlay, nrow, ncol)
    data_size = nlay * nrow * ncol
    arr.shape = data_shape

    sim_data = MFSimulationData("integration", None)
    dstruct = MFDataItemStructure()
    dstruct.is_cellid = False
    dstruct.name = "fake"
    dstruct.data_items = [
        None,
    ]
    mfstruct = MFDataStructure(dstruct, False, "ic", None)
    mfstruct.data_item_structures = [
        dstruct,
    ]
    mfstruct.path = [
        "fake",
    ]

    md = ModelDimensions("test", None)
    pd = PackageDimensions([md], None, "integration")
    dd = DataDimensions(pd, mfstruct)

    binfile = str(function_tmpdir / "structured_layered.hds")
    with open(binfile, "wb") as foo:
        for ix, a in enumerate(arr):
            write_head(foo, a, ilay=ix)

    fa = MFFileAccessArray(mfstruct, dd, sim_data, None, None)
    arr2 = fa.read_binary_data_from_file(
        binfile, data_shape, data_size, np.float64, modelgrid
    )[0]

    assert np.allclose(arr, arr2), "Binary read for layered Structured failed"

    binfile = str(function_tmpdir / "structured_flat.hds")
    with open(binfile, "wb") as foo:
        a = np.expand_dims(np.ravel(arr), axis=0)
        write_head(foo, a, ilay=1)

    arr2 = fa.read_binary_data_from_file(
        binfile, data_shape, data_size, np.float64, modelgrid
    )[0]

    assert np.allclose(arr, arr2), "Binary read for flat Structured failed"

    ncpl = nrow * ncol
    data_shape = (nlay, ncpl)
    arr.shape = data_shape
    modelgrid = flopy.discretization.VertexGrid(nlay=nlay, ncpl=ncpl)

    fa = MFFileAccessArray(mfstruct, dd, sim_data, None, None)

    binfile = str(function_tmpdir / "vertex_layered.hds")
    with open(binfile, "wb") as foo:
        tarr = arr.reshape((nlay, 1, ncpl))
        for ix, a in enumerate(tarr):
            write_head(foo, a, ilay=ix)

    arr2 = fa.read_binary_data_from_file(
        binfile, data_shape, data_size, np.float64, modelgrid
    )[0]

    assert np.allclose(arr, arr2), "Binary read for layered Vertex failed"

    binfile = str(function_tmpdir / "vertex_flat.hds")
    with open(binfile, "wb") as foo:
        a = np.expand_dims(np.ravel(arr), axis=0)
        write_head(foo, a, ilay=1)

    arr2 = fa.read_binary_data_from_file(
        binfile, data_shape, data_size, np.float64, modelgrid
    )[0]

    assert np.allclose(arr, arr2), "Binary read for flat Vertex failed"

    nlay = 3
    ncpl = [50, 100, 150]
    data_shape = (np.sum(ncpl),)
    arr.shape = data_shape
    modelgrid = flopy.discretization.UnstructuredGrid(ncpl=ncpl)

    fa = MFFileAccessArray(mfstruct, dd, sim_data, None, None)

    binfile = str(function_tmpdir / "unstructured.hds")
    with open(binfile, "wb") as foo:
        a = np.expand_dims(arr, axis=0)
        write_head(foo, a, ilay=1)

    arr2 = fa.read_binary_data_from_file(
        binfile, data_shape, data_size, np.float64, modelgrid
    )[0]

    assert np.allclose(arr, arr2), "Binary read for Unstructured failed"


@requires_exe("mf6")
def test_write_simulation(function_tmpdir):
    sim = MFSimulation(sim_ws=str(function_tmpdir))
    assert isinstance(sim, MFSimulation)

    tdis = ModflowTdis(sim)
    assert isinstance(tdis, ModflowTdis)

    gwfgwf = ModflowGwfgwf(
        sim, exgtype="gwf6-gwf6", exgmnamea="gwf1", exgmnameb="gwf2"
    )
    assert isinstance(gwfgwf, ModflowGwfgwf)

    gwf = ModflowGwf(sim)
    assert isinstance(gwf, ModflowGwf)

    ims = ModflowIms(sim)
    assert isinstance(ims, ModflowIms)
    sim.register_ims_package(ims, [])

    dis = ModflowGwfdis(gwf)
    assert isinstance(dis, ModflowGwfdis)

    disu = ModflowGwfdisu(gwf)
    assert isinstance(disu, ModflowGwfdisu)

    disv = ModflowGwfdisv(gwf)
    assert isinstance(disv, ModflowGwfdisv)

    npf = ModflowGwfnpf(gwf)
    assert isinstance(npf, ModflowGwfnpf)

    ic = ModflowGwfic(gwf)
    assert isinstance(ic, ModflowGwfic)

    sto = ModflowGwfsto(gwf)
    assert isinstance(sto, ModflowGwfsto)

    hfb = ModflowGwfhfb(gwf)
    assert isinstance(hfb, ModflowGwfhfb)

    gnc = ModflowGwfgnc(gwf)
    assert isinstance(gnc, ModflowGwfgnc)

    chd = ModflowGwfchd(gwf)
    assert isinstance(chd, ModflowGwfchd)

    wel = ModflowGwfwel(gwf)
    assert isinstance(wel, ModflowGwfwel)

    drn = ModflowGwfdrn(gwf)
    assert isinstance(drn, ModflowGwfdrn)

    riv = ModflowGwfriv(gwf)
    assert isinstance(riv, ModflowGwfriv)

    ghb = ModflowGwfghb(gwf)
    assert isinstance(ghb, ModflowGwfghb)

    rch = ModflowGwfrch(gwf)
    assert isinstance(rch, ModflowGwfrch)

    rcha = ModflowGwfrcha(gwf)
    assert isinstance(rcha, ModflowGwfrcha)

    evt = ModflowGwfevt(gwf)
    assert isinstance(evt, ModflowGwfevt)

    evta = ModflowGwfevta(gwf)
    assert isinstance(evta, ModflowGwfevta)

    maw = ModflowGwfmaw(gwf)
    assert isinstance(maw, ModflowGwfmaw)

    sfr = ModflowGwfsfr(gwf)
    assert isinstance(sfr, ModflowGwfsfr)

    lak = ModflowGwflak(gwf)
    assert isinstance(lak, ModflowGwflak)

    uzf = ModflowGwfuzf(gwf)
    assert isinstance(uzf, ModflowGwfuzf)

    mvr = ModflowGwfmvr(gwf)
    assert isinstance(mvr, ModflowGwfmvr)

    # Write files
    sim.write_simulation()

    # Verify files were written
    assert os.path.isfile(os.path.join(str(function_tmpdir), "mfsim.nam"))
    exts_model = [
        "nam",
        "dis",
        "disu",
        "disv",
        "npf",
        "ic",
        "sto",
        "hfb",
        "gnc",
        "chd",
        "wel",
        "drn",
        "riv",
        "ghb",
        "rch",
        "rcha",
        "evt",
        "evta",
        "maw",
        "sfr",
        "lak",
        "mvr",
    ]
    exts_sim = ["gwfgwf", "ims", "tdis"]
    for ext in exts_model:
        fname = os.path.join(str(function_tmpdir), f"model.{ext}")
        assert os.path.isfile(fname), f"{fname} not found"
    for ext in exts_sim:
        fname = os.path.join(str(function_tmpdir), f"sim.{ext}")
        assert os.path.isfile(fname), f"{fname} not found"


@requires_exe("mf6")
def test_create_and_run_model(function_tmpdir):
    # names
    sim_name = "testsim"
    model_name = "testmodel"
    exe_name = "mf6"

    # set up simulation
    tdis_name = f"{sim_name}.tdis"
    sim = MFSimulation(
        sim_name=sim_name,
        version="mf6",
        exe_name=exe_name,
        sim_ws=str(function_tmpdir),
    )
    tdis_rc = [(6.0, 2, 1.0), (6.0, 3, 1.0)]
    tdis = mftdis.ModflowTdis(
        sim, time_units="DAYS", nper=2, perioddata=tdis_rc
    )

    # create model instance
    model = mfgwf.ModflowGwf(
        sim, modelname=model_name, model_nam_file=f"{model_name}.nam"
    )

    # create solution and add the model
    ims_package = mfims.ModflowIms(
        sim,
        print_option="ALL",
        complexity="SIMPLE",
        outer_dvclose=0.00001,
        outer_maximum=50,
        under_relaxation="NONE",
        inner_maximum=30,
        inner_dvclose=0.00001,
        linear_acceleration="CG",
        preconditioner_levels=7,
        preconditioner_drop_tolerance=0.01,
        number_orthogonalizations=2,
    )
    sim.register_ims_package(ims_package, [model_name])

    # add packages to model
    dis_package = mfgwfdis.ModflowGwfdis(
        model,
        length_units="FEET",
        nlay=1,
        nrow=1,
        ncol=10,
        delr=500.0,
        delc=500.0,
        top=100.0,
        botm=50.0,
        filename=f"{model_name}.dis",
    )
    ic_package = mfgwfic.ModflowGwfic(
        model,
        strt=[
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,
        ],
        filename=f"{model_name}.ic",
    )
    npf_package = mfgwfnpf.ModflowGwfnpf(
        model, save_flows=True, icelltype=1, k=100.0
    )

    sto_package = mfgwfsto.ModflowGwfsto(
        model, save_flows=True, iconvert=1, ss=0.000001, sy=0.15
    )

    wel_package = mfgwfwel.ModflowGwfwel(
        model,
        print_input=True,
        print_flows=True,
        save_flows=True,
        maxbound=2,
        stress_period_data=[((0, 0, 4), -2000.0), ((0, 0, 7), -2.0)],
    )
    wel_package.stress_period_data.add_transient_key(1)
    wel_package.stress_period_data.set_data([((0, 0, 4), -200.0)], 1)

    drn_package = mfgwfdrn.ModflowGwfdrn(
        model,
        print_input=True,
        print_flows=True,
        save_flows=True,
        maxbound=1,
        stress_period_data=[((0, 0, 0), 80, 60.0)],
    )

    riv_package = mfgwfriv.ModflowGwfriv(
        model,
        print_input=True,
        print_flows=True,
        save_flows=True,
        maxbound=1,
        stress_period_data=[((0, 0, 9), 110, 90.0, 100.0)],
    )
    oc_package = mfgwfoc.ModflowGwfoc(
        model,
        budget_filerecord=[f"{model_name}.cbc"],
        head_filerecord=[f"{model_name}.hds"],
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        printrecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )
    oc_package.saverecord.add_transient_key(1)
    oc_package.saverecord.set_data([("HEAD", "ALL"), ("BUDGET", "ALL")], 1)
    oc_package.printrecord.add_transient_key(1)
    oc_package.printrecord.set_data([("HEAD", "ALL"), ("BUDGET", "ALL")], 1)

    # write the simulation input files
    sim.write_simulation()

    # run the simulation and look for output
    success, buff = sim.run_simulation()
    assert success


@requires_exe("mf6")
def test_get_set_data_record(function_tmpdir):
    # names
    sim_name = "testrecordsim"
    model_name = "testrecordmodel"
    exe_name = "mf6"

    # set up simulation
    tdis_name = f"{sim_name}.tdis"
    sim = MFSimulation(
        sim_name=sim_name,
        version="mf6",
        exe_name=exe_name,
        sim_ws=str(function_tmpdir),
    )
    tdis_rc = [(10.0, 4, 1.0), (6.0, 3, 1.0)]
    tdis = mftdis.ModflowTdis(
        sim, time_units="DAYS", nper=2, perioddata=tdis_rc
    )

    # create model instance
    model = mfgwf.ModflowGwf(
        sim, modelname=model_name, model_nam_file=f"{model_name}.nam"
    )

    # create solution and add the model
    ims_package = mfims.ModflowIms(
        sim,
        print_option="ALL",
        complexity="SIMPLE",
        outer_dvclose=0.00001,
        outer_maximum=50,
        under_relaxation="NONE",
        inner_maximum=30,
        inner_dvclose=0.00001,
        linear_acceleration="CG",
        preconditioner_levels=7,
        preconditioner_drop_tolerance=0.01,
        number_orthogonalizations=2,
    )
    sim.register_ims_package(ims_package, [model_name])

    # add packages to model
    dis_package = mfgwfdis.ModflowGwfdis(
        model,
        length_units="FEET",
        nlay=3,
        nrow=10,
        ncol=10,
        delr=500.0,
        delc=500.0,
        top=100.0,
        botm=[50.0, 10.0, -50.0],
        filename=f"{model_name}.dis",
    )
    ic_package = mfgwfic.ModflowGwfic(
        model,
        strt=[100.0, 90.0, 80.0],
        filename=f"{model_name}.ic",
    )
    npf_package = mfgwfnpf.ModflowGwfnpf(
        model, save_flows=True, icelltype=1, k=50.0, k33=1.0
    )

    sto_package = mfgwfsto.ModflowGwfsto(
        model, save_flows=True, iconvert=1, ss=0.000001, sy=0.15
    )
    # wel packages
    period_one = ModflowGwfwel.stress_period_data.empty(
        model,
        maxbound=3,
        aux_vars=["var1", "var2", "var3"],
        boundnames=True,
        timeseries=True,
    )
    period_one[0][0] = ((0, 9, 2), -50.0, -1, -2, -3, None)
    period_one[0][1] = ((1, 4, 7), -100.0, 1, 2, 3, "well_1")
    period_one[0][2] = ((1, 3, 2), -20.0, 4, 5, 6, "well_2")
    period_two = ModflowGwfwel.stress_period_data.empty(
        model,
        maxbound=2,
        aux_vars=["var1", "var2", "var3"],
        boundnames=True,
        timeseries=True,
    )
    period_two[0][0] = ((2, 3, 2), -80.0, 1, 2, 3, "well_2")
    period_two[0][1] = ((2, 4, 7), -10.0, 4, 5, 6, "well_1")
    stress_period_data = {}
    stress_period_data[0] = period_one[0]
    stress_period_data[1] = period_two[0]
    wel_package = ModflowGwfwel(
        model,
        print_input=True,
        print_flows=True,
        auxiliary=[("var1", "var2", "var3")],
        maxbound=5,
        stress_period_data=stress_period_data,
        boundnames=True,
        save_flows=True,
    )
    # rch package
    rch_period_list = []
    for row in range(0, 10):
        for col in range(0, 10):
            rch_amt = (1 + row / 10) * (1 + col / 10)
            rch_period_list.append(((0, row, col), rch_amt, 0.5))
    rch_period = {}
    rch_period[0] = rch_period_list
    rch_package = ModflowGwfrch(
        model,
        fixed_cell=True,
        auxiliary="MULTIPLIER",
        auxmultname="MULTIPLIER",
        print_input=True,
        print_flows=True,
        save_flows=True,
        maxbound=54,
        stress_period_data=rch_period,
    )

    # write simulation to new location
    sim.set_all_data_external()
    sim.write_simulation()

    # test get_record, set_record for list data
    wel = model.get_package("wel")
    spd_record = wel.stress_period_data.get_record()
    well_sp_1 = spd_record[0]
    assert (
        well_sp_1["filename"] == "testrecordmodel.wel_stress_period_data_1.txt"
    )
    assert well_sp_1["binary"] is False
    assert well_sp_1["data"][0][0] == (0, 9, 2)
    assert well_sp_1["data"][0][1] == -50.0
    # modify
    del well_sp_1["filename"]
    well_sp_1["data"][0][0] = (1, 9, 2)
    well_sp_2 = spd_record[1]
    del well_sp_2["filename"]
    well_sp_2["data"][0][0] = (1, 1, 1)
    # save
    spd_record[0] = well_sp_1
    spd_record[1] = well_sp_2
    wel.stress_period_data.set_record(spd_record)
    # verify changes
    spd_record = wel.stress_period_data.get_record()
    well_sp_1 = spd_record[0]
    assert "filename" not in well_sp_1
    assert well_sp_1["data"][0][0] == (1, 9, 2)
    assert well_sp_1["data"][0][1] == -50.0
    well_sp_2 = spd_record[1]
    assert "filename" not in well_sp_2
    assert well_sp_2["data"][0][0] == (1, 1, 1)
    spd = wel.stress_period_data.get_data()
    assert spd[0][0][0] == (1, 9, 2)
    # change well_sp_2 back to external
    well_sp_2["filename"] = "wel_spd_data_2.txt"
    spd_record[1] = well_sp_2
    wel.stress_period_data.set_record(spd_record)
    # change well_sp_2 data
    spd[1][0][0] = (1, 2, 2)
    wel.stress_period_data.set_data(spd)
    # verify changes
    spd_record = wel.stress_period_data.get_record()
    well_sp_2 = spd_record[1]
    assert well_sp_2["filename"] == "wel_spd_data_2.txt"
    assert well_sp_2["data"][0][0] == (1, 2, 2)

    # test get_data/set_data vs get_record/set_record
    dis = model.get_package("dis")
    botm = dis.botm.get_record()
    assert len(botm) == 3
    layer_2 = botm[1]
    layer_3 = botm[2]
    # verify layer 2
    assert layer_2["filename"] == "testrecordmodel.dis_botm_layer2.txt"
    assert layer_2["binary"] is False
    assert layer_2["factor"] == 1.0
    assert layer_2["iprn"] is None
    assert layer_2["data"][0][0] == 10.0
    # change and set layer 2
    layer_2["filename"] = "botm_layer2.txt"
    layer_2["binary"] = True
    layer_2["iprn"] = 3
    layer_2["factor"] = 2.0
    layer_2["data"] = layer_2["data"] * 0.5
    botm[1] = layer_2
    # change and set layer 3
    del layer_3["filename"]
    layer_3["factor"] = 0.5
    layer_3["data"] = layer_3["data"] * 2.0
    botm[2] = layer_3
    dis.botm.set_record(botm)

    # get botm in two different ways, verifying changes made
    botm_record = dis.botm.get_record()
    layer_1 = botm_record[0]
    assert layer_1["filename"] == "testrecordmodel.dis_botm_layer1.txt"
    assert layer_1["binary"] is False
    assert layer_1["iprn"] is None
    assert layer_1["data"][0][0] == 50.0
    layer_2 = botm_record[1]
    assert layer_2["filename"] == "botm_layer2.txt"
    assert layer_2["binary"] is True
    assert layer_2["factor"] == 2.0
    assert layer_2["iprn"] == 3
    assert layer_2["data"][0][0] == 5.0
    layer_3 = botm_record[2]
    assert "filename" not in layer_3
    assert layer_3["factor"] == 0.5
    assert layer_3["data"][0][0] == -100.0
    botm_data = dis.botm.get_data(apply_mult=True)
    assert botm_data[0][0][0] == 50.0
    assert botm_data[1][0][0] == 10.0
    assert botm_data[2][0][0] == -50.0
    botm_data = dis.botm.get_data()
    assert botm_data[0][0][0] == 50.0
    assert botm_data[1][0][0] == 5.0
    assert botm_data[2][0][0] == -100.0
    # modify and set botm data with set_data
    botm_data[0][0][0] = 6.0
    botm_data[1][0][0] = -8.0
    botm_data[2][0][0] = -205.0
    dis.botm.set_data(botm_data)
    # verify that data changed and metadata did not change
    botm_record = dis.botm.get_record()
    layer_1 = botm_record[0]
    assert layer_1["filename"] == "testrecordmodel.dis_botm_layer1.txt"
    assert layer_1["binary"] is False
    assert layer_1["iprn"] is None
    assert layer_1["data"][0][0] == 6.0
    assert layer_1["data"][0][1] == 50.0
    layer_2 = botm_record[1]
    assert layer_2["filename"] == "botm_layer2.txt"
    assert layer_2["binary"] is True
    assert layer_2["factor"] == 2.0
    assert layer_2["iprn"] == 3
    assert layer_2["data"][0][0] == -8.0
    assert layer_2["data"][0][1] == 5.0
    layer_3 = botm_record[2]
    assert "filename" not in layer_3
    assert layer_3["factor"] == 0.5
    assert layer_3["data"][0][0] == -205.0
    botm_data = dis.botm.get_data()
    assert botm_data[0][0][0] == 6.0
    assert botm_data[1][0][0] == -8.0
    assert botm_data[2][0][0] == -205.0

    spd_record = rch_package.stress_period_data.get_record()
    assert 0 in spd_record
    assert isinstance(spd_record[0], dict)
    assert "filename" in spd_record[0]
    assert (
        spd_record[0]["filename"]
        == "testrecordmodel.rch_stress_period_data_1.txt"
    )
    assert "binary" in spd_record[0]
    assert spd_record[0]["binary"] is False
    assert "data" in spd_record[0]
    assert spd_record[0]["data"][0][0] == (0, 0, 0)
    spd_record[0]["data"][0][0] = (0, 0, 8)
    rch_package.stress_period_data.set_record(spd_record)

    spd_data = rch_package.stress_period_data.get_data()
    assert spd_data[0][0][0] == (0, 0, 8)
    spd_data[0][0][0] = (0, 0, 7)
    rch_package.stress_period_data.set_data(spd_data)

    spd_record = rch_package.stress_period_data.get_record()
    assert isinstance(spd_record[0], dict)
    assert "filename" in spd_record[0]
    assert (
        spd_record[0]["filename"]
        == "testrecordmodel.rch_stress_period_data_1.txt"
    )
    assert "binary" in spd_record[0]
    assert spd_record[0]["binary"] is False
    assert "data" in spd_record[0]
    assert spd_record[0]["data"][0][0] == (0, 0, 7)

    sim.write_simulation()


@requires_exe("mf6")
def test_output(function_tmpdir, example_data_path):
    ex_name = "test001e_UZF_3lay"
    sim_ws = str(example_data_path / "mf6" / ex_name)
    sim = MFSimulation.load(sim_ws=sim_ws, exe_name="mf6")
    sim.set_sim_path(str(function_tmpdir))
    sim.write_simulation()
    success, buff = sim.run_simulation()
    assert success, f"simulation {sim.name} did not run"

    ml = sim.get_model("gwf_1")

    bud = ml.oc.output.budget()
    budcsv = ml.oc.output.budgetcsv()
    assert budcsv.file.closed
    hds = ml.oc.output.head()
    lst = ml.oc.output.list()

    idomain = np.ones(ml.modelgrid.shape, dtype=int)
    zonbud = ml.oc.output.zonebudget(idomain)

    assert isinstance(bud, CellBudgetFile)
    assert isinstance(budcsv, CsvFile)
    assert isinstance(hds, HeadFile)
    assert isinstance(zonbud, ZoneBudget6)
    assert isinstance(lst, Mf6ListBudget)

    bud = ml.output.budget()
    budcsv = ml.output.budgetcsv()
    hds = ml.output.head()
    zonbud = ml.output.zonebudget(idomain)
    lst = ml.output.list()

    assert isinstance(bud, CellBudgetFile)
    assert isinstance(budcsv, CsvFile)
    assert isinstance(hds, HeadFile)
    assert isinstance(zonbud, ZoneBudget6)
    assert isinstance(lst, Mf6ListBudget)

    uzf = ml.uzf
    uzf_bud = uzf.output.budget()
    uzf_budcsv = uzf.output.budgetcsv()
    conv = uzf.output.package_convergence()
    uzf_obs = uzf.output.obs()
    uzf_zonbud = uzf.output.zonebudget(idomain)

    assert isinstance(uzf_bud, CellBudgetFile)
    assert isinstance(uzf_budcsv, CsvFile)
    if conv is not None:
        assert isinstance(conv, CsvFile)
    assert isinstance(uzf_obs, Mf6Obs)
    assert isinstance(uzf_zonbud, ZoneBudget6)
    assert ml.dis.output.methods() is None


@requires_exe("mf6")
@pytest.mark.slow
def test_output_add_observation(function_tmpdir, example_data_path):
    model_name = "lakeex2a"
    sim_ws = str(example_data_path / "mf6" / "test045_lake2tr")
    sim = MFSimulation.load(sim_ws=sim_ws, exe_name="mf6")
    gwf = sim.get_model(model_name)

    # remove sfr_obs and add a new sfr obs
    sfr = gwf.sfr
    obs_file = f"{model_name}.sfr.obs"
    csv_file = f"{obs_file}.csv"
    obs_dict = {
        csv_file: [
            ("l08_stage", "stage", (8,)),
            ("l09_stage", "stage", (9,)),
            ("l14_stage", "stage", (14,)),
            ("l15_stage", "stage", (15,)),
        ]
    }
    gwf.sfr.obs.initialize(
        filename=obs_file, digits=10, print_input=True, continuous=obs_dict
    )

    sim.set_sim_path(str(function_tmpdir))
    sim.write_simulation()

    success, buff = sim.run_simulation()
    assert success, f"simulation {sim.name} did not run"

    # check that .output finds the newly added OBS package
    sfr_obs = gwf.sfr.output.obs()

    assert isinstance(
        sfr_obs, Mf6Obs
    ), "remove and add observation test (Mf6Output) failed"


@requires_exe("mf6")
def test_array(function_tmpdir):
    # get_data
    # empty data in period block vs data repeating
    # array
    # aux values, test that they work the same as other arrays (is a value
    # of zero always used even if aux is defined in a previous stress
    # period?)

    sim_name = "test_array"
    model_name = "test_array"
    out_dir = str(function_tmpdir)
    tdis_name = "{}.tdis".format(sim_name)
    sim = MFSimulation(
        sim_name=sim_name, version="mf6", exe_name="mf6", sim_ws=out_dir
    )
    tdis_rc = [(6.0, 2, 1.0), (6.0, 3, 1.0), (6.0, 3, 1.0), (6.0, 3, 1.0)]
    tdis = ModflowTdis(sim, time_units="DAYS", nper=4, perioddata=tdis_rc)
    ims_package = ModflowIms(
        sim,
        pname="my_ims_file",
        filename=f"{sim_name}.ims",
        print_option="ALL",
        complexity="SIMPLE",
        outer_dvclose=0.0001,
        outer_maximum=50,
        under_relaxation="NONE",
        inner_maximum=30,
        inner_dvclose=0.0001,
        linear_acceleration="CG",
        preconditioner_levels=7,
        preconditioner_drop_tolerance=0.01,
        number_orthogonalizations=2,
    )
    model = ModflowGwf(
        sim, modelname=model_name, model_nam_file="{}.nam".format(model_name)
    )

    dis = ModflowGwfdis(
        model,
        length_units="FEET",
        nlay=4,
        nrow=2,
        ncol=2,
        delr=5000.0,
        delc=5000.0,
        top=100.0,
        botm=[50.0, 0.0, -50.0, -100.0],
        filename=f"{model_name} 1.dis",
    )
    ic_package = ModflowGwfic(model, strt=90.0, filename=f"{model_name}.ic")
    npf_package = ModflowGwfnpf(
        model,
        pname="npf_1",
        save_flows=True,
        alternative_cell_averaging="logarithmic",
        icelltype=1,
        k=50.0,
    )

    oc_package = ModflowGwfoc(
        model,
        budget_filerecord=[("test_array.cbc",)],
        head_filerecord=[("test_array.hds",)],
        saverecord={
            0: [("HEAD", "ALL"), ("BUDGET", "ALL")],
            1: [],
        },
        printrecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )

    aux = {1: [[50.0], [1.3]], 3: [[200.0], [1.5]]}
    irch = {1: [[0, 2], [2, 1]], 2: [[0, 1], [2, 3]]}
    rcha = ModflowGwfrcha(
        model,
        print_input=True,
        print_flows=True,
        auxiliary=[("var1", "var2")],
        irch=irch,
        recharge={1: 0.0001, 2: 0.00001},
        aux=aux,
    )
    val_irch = rcha.irch.array.sum(axis=(1, 2, 3))
    assert val_irch[0] == 4
    assert val_irch[1] == 5
    assert val_irch[2] == 6
    assert val_irch[3] == 6
    val_irch_2 = rcha.irch.get_data()
    assert val_irch_2[0] is None
    assert val_irch_2[1][1, 1] == 1
    assert val_irch_2[2][1, 1] == 3
    assert val_irch_2[3] is None
    val_irch_2_3 = rcha.irch.get_data(3)
    assert val_irch_2_3 is None
    val_rch = rcha.recharge.array.sum(axis=(1, 2, 3))
    assert val_rch[0] == 0.0
    assert val_rch[1] == 0.0004
    assert val_rch[2] == 0.00004
    assert val_rch[3] == 0.00004
    val_rch_2 = rcha.recharge.get_data()
    assert val_rch_2[0] is None
    assert val_rch_2[1][0, 0] == 0.0001
    assert val_rch_2[2][0, 0] == 0.00001
    assert val_rch_2[3] is None
    aux_data_0 = rcha.aux.get_data(0)
    assert aux_data_0 is None
    aux_data_1 = rcha.aux.get_data(1)
    assert aux_data_1[0][0][0] == 50.0
    aux_data_2 = rcha.aux.get_data(2)
    assert aux_data_2 is None
    aux_data_3 = rcha.aux.get_data(3)
    assert aux_data_3[0][0][0] == 200.0

    welspdict = {1: [[(0, 0, 0), 0.25, 0.0]], 2: [[(0, 0, 0), 0.1, 0.0]]}
    wel = ModflowGwfwel(
        model,
        print_input=True,
        print_flows=True,
        mover=True,
        stress_period_data=welspdict,
        save_flows=False,
        auxiliary="CONCENTRATION",
        pname="WEL-1",
    )
    wel_array = wel.stress_period_data.array
    assert wel_array[0] is None
    assert wel_array[1][0][1] == 0.25
    assert wel_array[2][0][1] == 0.1
    assert wel_array[3][0][1] == 0.1

    drnspdict = {
        0: [[(0, 0, 0), 60.0, 10.0]],
        2: [],
        3: [[(0, 0, 0), 55.0, 5.0]],
    }
    drn = ModflowGwfdrn(
        model,
        print_input=True,
        print_flows=True,
        stress_period_data=drnspdict,
        save_flows=False,
        pname="DRN-1",
    )
    drn_array = drn.stress_period_data.array
    assert drn_array[0][0][1] == 60.0
    assert drn_array[1][0][1] == 60.0
    assert drn_array[2] is None
    assert drn_array[3][0][1] == 55.0
    drn_gd_0 = drn.stress_period_data.get_data(0)
    assert drn_gd_0[0][1] == 60.0
    drn_gd_1 = drn.stress_period_data.get_data(1)
    assert drn_gd_1 is None
    drn_gd_2 = drn.stress_period_data.get_data(2)
    assert drn_gd_2 == []
    drn_gd_3 = drn.stress_period_data.get_data(3)
    assert drn_gd_3[0][1] == 55.0

    ghbspdict = {
        0: [[(0, 1, 1), 60.0, 10.0]],
    }
    ghb = ModflowGwfghb(
        model,
        print_input=True,
        print_flows=True,
        stress_period_data=ghbspdict,
        save_flows=False,
        pname="GHB-1",
    )

    lakpd = [(0, 70.0, 1), (1, 65.0, 1)]
    lakecn = [
        (0, 0, (0, 0, 0), "HORIZONTAL", 1.0, 60.0, 90.0, 10.0, 1.0),
        (1, 0, (0, 1, 1), "HORIZONTAL", 1.0, 60.0, 90.0, 10.0, 1.0),
    ]
    lak_tables = [(0, "lak01.tab"), (1, "lak02.tab")]
    lak = ModflowGwflak(
        model,
        pname="lak",
        print_input=True,
        mover=True,
        nlakes=2,
        noutlets=0,
        ntables=1,
        packagedata=lakpd,
        connectiondata=lakecn,
        tables=lak_tables,
    )

    table_01 = [
        (30.0, 100000.0, 10000.0),
        (40.0, 200500.0, 10100.0),
        (50.0, 301200.0, 10130.0),
        (60.0, 402000.0, 10180.0),
        (70.0, 503000.0, 10200.0),
        (80.0, 700000.0, 20000.0),
    ]
    lak_tab = ModflowUtllaktab(
        model,
        filename="lak01.tab",
        nrow=6,
        ncol=3,
        table=table_01,
    )

    table_02 = [
        (40.0, 100000.0, 10000.0),
        (50.0, 200500.0, 10100.0),
        (60.0, 301200.0, 10130.0),
        (70.0, 402000.0, 10180.0),
        (80.0, 503000.0, 10200.0),
        (90.0, 700000.0, 20000.0),
    ]
    lak_tab_2 = ModflowUtllaktab(
        model,
        filename="lak02.tab",
        nrow=6,
        ncol=3,
        table=table_02,
    )
    wel_name_1 = wel.name[0]
    lak_name_2 = lak.name[0]
    package_data = [(wel_name_1,), (lak_name_2,)]
    period_data = [(wel_name_1, 0, lak_name_2, 0, "FACTOR", 1.0)]
    fname = f"{model.name}.input.mvr"
    mvr = ModflowGwfmvr(
        parent_model_or_package=model,
        filename=fname,
        print_input=True,
        print_flows=True,
        maxpackages=2,
        maxmvr=1,
        packages=package_data,
        perioddata=period_data,
    )

    # test writing and loading model
    sim.write_simulation()
    sim.run_simulation()

    test_sim = MFSimulation.load(
        sim_name,
        "mf6",
        "mf6",
        out_dir,
        write_headers=False,
    )
    model = test_sim.get_model()
    dis = model.get_package("dis")
    rcha = model.get_package("rcha")
    wel = model.get_package("wel")
    drn = model.get_package("drn")
    lak = model.get_package("lak")
    lak_tab = model.get_package("laktab")
    assert os.path.split(dis.filename)[1] == f"{model_name} 1.dis"
    # do same tests as above
    val_irch = rcha.irch.array.sum(axis=(1, 2, 3))
    assert val_irch[0] == 4
    assert val_irch[1] == 5
    assert val_irch[2] == 6
    assert val_irch[3] == 6
    val_irch_2 = rcha.irch.get_data()
    assert val_irch_2[0] is None
    assert val_irch_2[1][1, 1] == 1
    assert val_irch_2[2][1, 1] == 3
    assert val_irch_2[3] is None
    val_rch = rcha.recharge.array.sum(axis=(1, 2, 3))
    assert val_rch[0] == 0.0
    assert val_rch[1] == 0.0004
    assert val_rch[2] == 0.00004
    assert val_rch[3] == 0.00004
    val_rch_2 = rcha.recharge.get_data()
    assert val_rch_2[0] is None
    assert val_rch_2[1][0, 0] == 0.0001
    assert val_rch_2[2][0, 0] == 0.00001
    assert val_rch_2[3] is None
    aux_data_0 = rcha.aux.get_data(0)
    assert aux_data_0 is None
    aux_data_1 = rcha.aux.get_data(1)
    assert aux_data_1[0][0][0] == 50.0
    aux_data_2 = rcha.aux.get_data(2)
    assert aux_data_2 is None
    aux_data_3 = rcha.aux.get_data(3)
    assert aux_data_3[0][0][0] == 200.0

    wel_array = wel.stress_period_data.array
    assert wel_array[0] is None
    assert wel_array[1][0][1] == 0.25
    assert wel_array[2][0][1] == 0.1
    assert wel_array[3][0][1] == 0.1

    drn_array = drn.stress_period_data.array
    assert drn_array[0][0][1] == 60.0
    assert drn_array[1][0][1] == 60.0
    assert drn_array[2] is None
    assert drn_array[3][0][1] == 55.0
    drn_gd_0 = drn.stress_period_data.get_data(0)
    assert drn_gd_0[0][1] == 60.0
    drn_gd_1 = drn.stress_period_data.get_data(1)
    assert drn_gd_1 is None
    drn_gd_2 = drn.stress_period_data.get_data(2)
    assert drn_gd_2 == []
    drn_gd_3 = drn.stress_period_data.get_data(3)
    assert drn_gd_3[0][1] == 55.0

    lak_tab_array = lak.tables.get_data()
    assert lak_tab_array[0][1] == "lak01.tab"
    assert lak_tab_array[1][1] == "lak02.tab"

    assert len(lak_tab) == 2
    lak_tab_1 = lak_tab[0].table.get_data()
    assert lak_tab_1[0][0] == 30.0
    assert lak_tab_1[5][2] == 20000.0
    lak_tab_2 = lak_tab[1].table.get_data()
    assert lak_tab_2[0][0] == 40.0
    assert lak_tab_2[4][1] == 503000.0


@requires_exe("mf6")
def test_multi_model(function_tmpdir):
    # init paths
    test_ex_name = "test_multi_model"
    model_names = ["gwf_model_1", "gwf_model_2", "gwt_model_1", "gwt_model_2"]

    # temporal discretization
    nper = 1
    perlen = [5.0]
    nstp = [200]
    tsmult = [1.0]
    tdis_rc = []
    for i in range(nper):
        tdis_rc.append((perlen[i], nstp[i], tsmult[i]))

    # build MODFLOW 6 files
    sim = MFSimulation(
        sim_name=test_ex_name,
        version="mf6",
        exe_name="mf6",
        sim_ws=str(function_tmpdir),
    )
    # create tdis package
    tdis = ModflowTdis(
        sim, time_units="DAYS", nper=nper, perioddata=tdis_rc, pname="sim.tdis"
    )

    # grid information
    nlay, nrow, ncol = 1, 1, 50

    # Create gwf1 model
    welspd = {0: [[(0, 0, 0), 1.0, 1.0]]}
    chdspd = None
    gwf1 = get_gwf_model(
        sim,
        model_names[0],
        model_names[0],
        (nlay, nrow, ncol),
        chdspd=chdspd,
        welspd=welspd,
    )

    # Create gwf2 model
    welspd = {0: [[(0, 0, 1), 0.5, 0.5]]}
    chdspd = {0: [[(0, 0, ncol - 1), 0.0000000]]}
    gwf2 = get_gwf_model(
        sim,
        model_names[1],
        model_names[1],
        (nlay, nrow, ncol),
        chdspd=chdspd,
        welspd=welspd,
    )
    lakpd = [(0, -100.0, 1)]
    lakecn = [(0, 0, (0, 0, 0), "HORIZONTAL", 1.0, 0.1, 1.0, 10.0, 1.0)]
    lak_2 = ModflowGwflak(
        gwf2,
        pname="lak2",
        print_input=True,
        mover=True,
        nlakes=1,
        noutlets=0,
        ntables=0,
        packagedata=lakpd,
        connectiondata=lakecn,
    )

    # gwf-gwf
    gwfgwf_data = []
    for col in range(0, ncol):
        gwfgwf_data.append(
            [(0, 0, col), (0, 0, col), 1, 0.5, 0.5, 1.0, 0.0, 1.0]
        )
    gwfgwf = ModflowGwfgwf(
        sim,
        exgtype="GWF6-GWF6",
        nexg=len(gwfgwf_data),
        exgmnamea=gwf1.name,
        exgmnameb=gwf2.name,
        exchangedata=gwfgwf_data,
        auxiliary=["ANGLDEGX", "CDIST"],
        filename="flow1_flow2.gwfgwf",
    )
    # set up mvr package
    wel_1 = gwf1.get_package("wel")
    wel_1.mover.set_data(True)
    wel_name_1 = wel_1.name[0]
    lak_name_2 = lak_2.name[0]
    package_data = [(gwf1.name, wel_name_1), (gwf2.name, lak_name_2)]
    period_data = [
        (gwf1.name, wel_name_1, 0, gwf2.name, lak_name_2, 0, "FACTOR", 1.0)
    ]
    fname = "gwfgwf.input.mvr"
    gwfgwf.mvr.initialize(
        filename=fname,
        modelnames=True,
        print_input=True,
        print_flows=True,
        maxpackages=2,
        maxmvr=1,
        packages=package_data,
        perioddata=period_data,
    )

    gnc_data = []
    for col in range(0, ncol):
        if col < ncol / 2.0:
            gnc_data.append(((0, 0, col), (0, 0, col), (0, 0, col + 1), 0.25))
        else:
            gnc_data.append(((0, 0, col), (0, 0, col), (0, 0, col - 1), 0.25))

    # set up gnc package
    fname = "gwfgwf.input.gnc"
    gwfgwf.gnc.initialize(
        filename=fname,
        print_input=True,
        print_flows=True,
        numgnc=ncol,
        numalphaj=1,
        gncdata=gnc_data,
    )

    # Observe flow for exchange
    gwfgwfobs = {}
    obs_list = []
    for col in range(0, ncol):
        obs_list.append([f"exchange_flow_{col}", "FLOW-JA-FACE", (col,)])
    gwfgwfobs["gwfgwf.output.obs.csv"] = obs_list
    fname = "gwfgwf.input.obs"
    gwfgwf.obs.initialize(
        filename=fname, digits=25, print_input=True, continuous=gwfgwfobs
    )

    # Create gwt model
    sourcerecarray = [("WEL-1", "AUX", "CONCENTRATION")]
    gwt = get_gwt_model(
        sim,
        model_names[2],
        model_names[2],
        (nlay, nrow, ncol),
        sourcerecarray=sourcerecarray,
    )

    # GWF GWT exchange
    gwfgwt = ModflowGwfgwt(
        sim,
        exgtype="GWF6-GWT6",
        exgmnamea=model_names[0],
        exgmnameb=model_names[2],
        filename="flow1_transport1.gwfgwt",
    )

    # solver settings
    nouter, ninner = 100, 300
    hclose, rclose, relax = 1e-6, 1e-6, 1.0

    # create iterative model solution and register the gwf model with it
    imsgwf = ModflowIms(
        sim,
        print_option="SUMMARY",
        outer_dvclose=hclose,
        outer_maximum=nouter,
        under_relaxation="NONE",
        inner_maximum=ninner,
        inner_dvclose=hclose,
        rcloserecord=rclose,
        linear_acceleration="BICGSTAB",
        scaling_method="NONE",
        reordering_method="NONE",
        relaxation_factor=relax,
        filename="flow.ims",
    )

    # create iterative model solution and register the gwt model with it
    imsgwt = ModflowIms(
        sim,
        print_option="SUMMARY",
        outer_dvclose=hclose,
        outer_maximum=nouter,
        under_relaxation="NONE",
        inner_maximum=ninner,
        inner_dvclose=hclose,
        rcloserecord=rclose,
        linear_acceleration="BICGSTAB",
        scaling_method="NONE",
        reordering_method="NONE",
        relaxation_factor=relax,
        filename="transport.ims",
    )
    sim.register_ims_package(imsgwt, [gwt.name])

    sim.write_simulation()
    sim.run_simulation()

    # reload simulation
    sim2 = MFSimulation.load(sim_ws=str(function_tmpdir))

    # check ims registration
    solution_recarray = sim2.name_file.solutiongroup
    for solution_group_num in solution_recarray.get_active_key_list():
        rec_array = solution_recarray.get_data(solution_group_num[0])
        assert rec_array[0][1] == "flow.ims"
        assert rec_array[0][2] == model_names[0]
        assert rec_array[0][3] == model_names[1]
        assert rec_array[1][1] == "transport.ims"
        assert rec_array[1][2] == model_names[2]

    # create a new gwt model
    sourcerecarray = [("WEL-1", "AUX", "CONCENTRATION")]
    gwt_2 = get_gwt_model(
        sim,
        model_names[3],
        model_names[3],
        (nlay, nrow, ncol),
        sourcerecarray=sourcerecarray,
    )
    # register gwt model with transport.ims
    sim.register_ims_package(imsgwt, gwt_2.name)
    # flow and transport exchange
    gwfgwt = ModflowGwfgwt(
        sim,
        exgtype="GWF6-GWT6",
        exgmnamea=model_names[1],
        exgmnameb=model_names[3],
        filename="flow2_transport2.gwfgwt",
    )
    # save and run updated model
    sim.write_simulation()
    sim.run_simulation()

    with pytest.raises(
        flopy.mf6.mfbase.FlopyException,
        match='Extraneous kwargs "param_does_not_exist" '
        "provided to MFPackage.",
    ):
        # test kwargs error checking
        wel = ModflowGwfwel(
            gwf2,
            print_input=True,
            print_flows=True,
            stress_period_data=welspd,
            save_flows=False,
            auxiliary="CONCENTRATION",
            pname="WEL-1",
            param_does_not_exist=True,
        )


@requires_exe("mf6")
def test_namefile_creation(tmpdir):
    test_ex_name = "test_namefile"
    # build MODFLOW 6 files
    sim = MFSimulation(
        sim_name=test_ex_name,
        version="mf6",
        exe_name="mf6",
        sim_ws=str(tmpdir),
    )

    tdis_rc = [(6.0, 2, 1.0), (6.0, 3, 1.0), (6.0, 3, 1.0), (6.0, 3, 1.0)]
    tdis = ModflowTdis(sim, time_units="DAYS", nper=4, perioddata=tdis_rc)
    ims_package = ModflowIms(
        sim,
        pname="my_ims_file",
        filename=f"{test_ex_name}.ims",
        print_option="ALL",
        complexity="SIMPLE",
        outer_dvclose=0.0001,
        outer_maximum=50,
        under_relaxation="NONE",
        inner_maximum=30,
        inner_dvclose=0.0001,
        linear_acceleration="CG",
        preconditioner_levels=7,
        preconditioner_drop_tolerance=0.01,
        number_orthogonalizations=2,
    )
    model = ModflowGwf(
        sim,
        modelname=test_ex_name,
        model_nam_file="{}.nam".format(test_ex_name),
    )

    # try to create simulation name file
    ex_happened = False
    try:
        nam = ModflowNam(sim)
    except flopy.mf6.mfbase.FlopyException:
        ex_happened = True
    assert ex_happened

    # try to create model name file
    ex_happened = False
    try:
        nam = ModflowGwfnam(model)
    except flopy.mf6.mfbase.FlopyException:
        ex_happened = True
    assert ex_happened

@requires_exe("mf6")
def test_solver_ems(tmpdir):
    # path to MODFLOW 6 executable
    mf6_exe_name = "mf6"

    # base model name and workspace
    model_name_base = "mp7ex01"
    ws_base = os.path.join(tmpdir, f"t507_{model_name_base}")

    # nper, nstp, perlen, tsmult = 1, 1, 1.0, 1.0
    # issue(mf6): particle location output is currently only at the end of each time step
    nper, nstp, perlen, tsmult = 1, 54, 54000.0, 1.0
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
    rch_iface = 6
    rch_iflowface = -1
    riv_h = 320.0
    riv_z = 317.0
    riv_c = 1.0e5
    riv_iface = 6
    riv_iflowface = -1

    # porosity
    porosity = 0.1

    # starting location template size for example 1B;
    # in the original example, there is initially a
    # 3x3 array of particles in each cell in layer 1;
    # in this flopy notebook, there is initially one
    # particle in each cell in layer 1; the original
    # 3x3 particle arrays can be restored simply by
    # setting sloc_tmpl_size below to 3 instead of 1.
    sloc_tmpl_size = 1

    # create particle release point data for example 1A
    releasepts = {}
    releasepts["1A"] = []
    zrpt = top
    k = 0
    j = 2
    for i in range(nrow):
        nrpt = i
        xrpt = (j + 0.5) * delr
        yrpt = (nrow - i - 0.5) * delc
        rpt = [nrpt, k, i, j, xrpt, yrpt, zrpt]
        releasepts["1A"].append(rpt)

    # create particle release point data for example 1B
    releasepts["1B"] = []
    ndivc = sloc_tmpl_size
    ndivr = sloc_tmpl_size
    deldivc = delc / ndivc
    deldivr = delr / ndivr
    k = 0
    zrpt = top
    nrpt = -1
    for i in range(nrow):
        y0 = (nrow - i - 1) * delc
        for j in range(ncol):
            x0 = j * delr
            for idiv in range(ndivc):
                dy = (idiv + 0.5) * deldivc
                yrpt = y0 + dy
                for jdiv in range(ndivr):
                    dx = (jdiv + 0.5) * deldivr
                    xrpt = x0 + dx
                    nrpt += 1
                    rpt = [nrpt, k, i, j, xrpt, yrpt, zrpt]
                    releasepts["1B"].append(rpt)

    ws_mf6 = os.path.join(ws_base, "mf6")
    nm_mf6 = "mf6_" + model_name_base

    # create the simulation object
    sim = flopy.mf6.MFSimulation(
        sim_name=nm_mf6, exe_name=mf6_exe_name, version="mf6", sim_ws=ws_mf6
    )

    # create the temporal discretization package
    pd = (perlen, nstp, tsmult)
    tdis = flopy.mf6.modflow.mftdis.ModflowTdis(
        sim, pname="tdis", time_units="DAYS", nper=nper, perioddata=[pd]
    )

    # create the groundwater-flow (gwf) model
    model_nam_file = "{}.nam".format(nm_mf6)
    gwf = flopy.mf6.ModflowGwf(
        sim, modelname=nm_mf6, model_nam_file=model_nam_file, save_flows=True
    )

    # create the discretization package
    dis = flopy.mf6.modflow.mfgwfdis.ModflowGwfdis(
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

    # create the initial conditions package
    ic = flopy.mf6.modflow.mfgwfic.ModflowGwfic(gwf, pname="ic", strt=top)

    # create the node property flow package
    npf = flopy.mf6.modflow.mfgwfnpf.ModflowGwfnpf(
        gwf,
        pname="npf",
        icelltype=laytyp,
        k=kh,
        k33=kv,
        save_saturation=True,
        save_specific_discharge=True,
    )

    # create the recharge package
    flopy.mf6.modflow.mfgwfrcha.ModflowGwfrcha(
        gwf,
        recharge=rch,
        auxiliary=["iface", "iflowface"],
        aux=[rch_iface, rch_iflowface],
    )

    # create the well package
    wd = [(wel_loc, wel_q)]
    flopy.mf6.modflow.mfgwfwel.ModflowGwfwel(
        gwf, maxbound=1, stress_period_data={0: wd}
    )

    # create the river package
    rd = []
    for i in range(nrow):
        rd.append(
            [(0, i, ncol - 1), riv_h, riv_c, riv_z, riv_iface, riv_iflowface]
        )
    flopy.mf6.modflow.mfgwfriv.ModflowGwfriv(
        gwf, auxiliary=["iface", "iflowface"], stress_period_data={0: rd}
    )

    # create the output control package
    headfile = "{}.hds".format(nm_mf6)
    head_record = [headfile]
    budgetfile = "{}.cbb".format(nm_mf6)
    budget_record = [budgetfile]
    saverecord = [("HEAD", "ALL"), ("BUDGET", "ALL")]
    oc = flopy.mf6.modflow.mfgwfoc.ModflowGwfoc(
        gwf,
        pname="oc",
        saverecord=saverecord,
        head_filerecord=head_record,
        budget_filerecord=budget_record,
    )

    # create the prt model
    nm_prt = nm_mf6 + "_prt"
    prt = flopy.mf6.ModflowPrt(
        sim, modelname=nm_prt, model_nam_file="{}.nam".format(nm_prt)
    )

    # create the prt discretization package
    dis = flopy.mf6.modflow.mfgwfdis.ModflowGwfdis(
        prt,
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

    # create the prt model input package
    mip = flopy.mf6.ModflowPrtmip(prt, pname="mip", porosity=porosity)

    # create the prt particle release point (prp) package
    # for example 1A
    nreleasepts1a = len(releasepts["1A"])
    pd = {
        0: ["FIRST"],
    }
    prp1a = flopy.mf6.ModflowPrtprp(
        prt,
        pname="prp1a",  # filename="{}_1a.prp".format(nm_prt),
        nreleasepts=nreleasepts1a,
        packagedata=releasepts["1A"],
        perioddata=pd,
    )

    # create the prt particle release point (prp) package
    # for example 1B
    nreleasepts1b = len(releasepts["1B"])
    pd = {
        0: ["FIRST"],
    }
    prp1b = flopy.mf6.ModflowPrtprp(
        prt,
        pname="prp1b",  # filename="{}_1b.prp".format(nm_prt),
        nreleasepts=nreleasepts1b,
        packagedata=releasepts["1B"],
        perioddata=pd,
    )

    # create the prt output control package
    budgetfile_prt = "{}.cbb".format(nm_prt)
    budget_record = [budgetfile_prt]
    oc = flopy.mf6.ModflowPrtoc(
        prt,
        pname="oc",
        budget_filerecord=budget_record,
        saverecord=[("BUDGET", "ALL")],
    )

    # create the prt flow model interface
    fmi = flopy.mf6.ModflowPrtfmi(prt)

    # create the gwf-prt model exchange
    gwfprt = flopy.mf6.ModflowGwfprt(
        sim,
        exgtype="GWF6-PRT6",
        exgmnamea=nm_mf6,
        exgmnameb=nm_prt,
        filename="{}.gwfprt".format(nm_mf6),
    )

    # set up an iterative model solution (IMS) for the flow model
    ims = flopy.mf6.ModflowIms(
        sim,
        pname="ims",
        complexity="SIMPLE",
        outer_dvclose=1e-6,
        inner_dvclose=1e-6,
        rcloserecord=1e-6,
    )

    # set up an explicit model solution (EMS) for the particle-tracking model
    # issue(flopy): using IMS until mfsimulation.py is updated for EMS by Scott
    # issue(mf6): code is kluged to create EMS if first model in IMS is a PRT
    ems = flopy.mf6.ModflowEms(
        sim,
        pname="ems",
        filename="{}.ems".format(nm_prt),
    )
    sim.register_solution_package(ems, [prt.name])

    # write the datasets
    sim.write_simulation()

"""
def test_sfr_connections(function_tmpdir, example_data_path):
    '''MODFLOW just warns if any reaches are unconnected
    flopy fails to load model if reach 1 is unconnected, fine with other unconnected'''
    data_path = str(example_data_path / "mf6" / "test666_sfrconnections")
    sim_ws = str(function_tmpdir)
    for test in ['sfr0', 'sfr1']:
        sim_name = "test_sfr"
        model_name = "test_sfr"
        tdis_name = "{}.tdis".format(sim_name)
        sim = MFSimulation(
            sim_name=sim_name, version="mf6", exe_name="mf6", sim_ws=sim_ws
        )
        tdis_rc = [(1.0, 1, 1.0)]
        tdis = ModflowTdis(sim, time_units="DAYS", nper=1, perioddata=tdis_rc)
        ims_package = ModflowIms(
            sim,
            pname="my_ims_file",
            filename=f"{sim_name}.ims",
            print_option="ALL",
            complexity="SIMPLE",
        )
        model = ModflowGwf(
            sim, modelname=model_name, model_nam_file="{}.nam".format(model_name)
        )

        dis = ModflowGwfdis(
            model,
            length_units="FEET",
            nlay=1,
            nrow=5,
            ncol=5,
            delr=5000.0,
            delc=5000.0,
            top=100.0,
            botm=-100.0,
            filename=f"{model_name}.dis",
        )
        ic_package = ModflowGwfic(model, filename=f"{model_name}.ic")
        npf_package = ModflowGwfnpf(
            model,
            pname="npf",
            save_flows=True,
            alternative_cell_averaging="logarithmic",
            icelltype=1,
            k=50.0,
        )

        cnfile = f'mf6_{test}_connection.txt'
        pkfile = f'mf6_{test}_package.txt'

        with open(os.path.join(data_path, pkfile), 'r') as f:
            nreaches = len(f.readlines())
        sfr = ModflowGwfsfr(model,
                            packagedata={'filename': os.path.join(data_path, pkfile)},
                            connectiondata={'filename': os.path.join(data_path, cnfile)},
                            nreaches=nreaches,
                            pname='sfr',
                            unit_conversion=86400
                            )
        sim.set_all_data_external()
        sim.write_simulation()
        success, buff = sim.run_simulation()
        assert success, f"simulation {sim.name} did not run"

        #reload simulation
        sim2 = MFSimulation.load(sim_ws=sim_ws)
        sim.set_all_data_external()
        sim.write_simulation()
        success, buff = sim.run_simulation()
        assert success, f"simulation {sim.name} did not run after being reloaded"
"""
