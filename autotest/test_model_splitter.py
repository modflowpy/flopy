import numpy as np
import pytest
import yaml
from modflow_devtools.markers import requires_exe, requires_pkg
from modflow_devtools.misc import set_dir

import flopy
from autotest.conftest import get_example_data_path
from flopy.mf6 import MFSimulation
from flopy.mf6.utils import Mf6Splitter


@requires_exe("mf6")
def test_structured_model_splitter(function_tmpdir):
    sim_path = get_example_data_path() / "mf6-freyberg"

    sim = MFSimulation.load(sim_ws=sim_path)
    sim.set_sim_path(function_tmpdir)
    sim.write_simulation()
    sim.run_simulation()

    gwf = sim.get_model()
    modelgrid = gwf.modelgrid

    array = np.ones((modelgrid.nrow, modelgrid.ncol), dtype=int)
    ncol = 1
    for row in range(modelgrid.nrow):
        if row != 0 and row % 2 == 0:
            ncol += 1
        array[row, ncol:] = 100

    mfsplit = Mf6Splitter(sim)
    new_sim = mfsplit.split_model(array)

    new_sim.set_sim_path(function_tmpdir / "split_model")
    new_sim.write_simulation()
    new_sim.run_simulation()

    original_heads = gwf.output.head().get_alldata()[-1]

    ml0 = new_sim.get_model("freyberg_001")
    ml1 = new_sim.get_model("freyberg_100")

    heads0 = ml0.output.head().get_alldata()[-1]
    heads1 = ml1.output.head().get_alldata()[-1]

    new_heads = mfsplit.reconstruct_array({1: heads0, 100: heads1})

    err_msg = "Heads from original and split models do not match"
    np.testing.assert_allclose(new_heads, original_heads, err_msg=err_msg)


@requires_exe("mf6")
def test_vertex_model_splitter(function_tmpdir):
    sim_path = get_example_data_path() / "mf6" / "test003_gwftri_disv"

    sim = MFSimulation.load(sim_ws=sim_path)
    sim.set_sim_path(function_tmpdir)
    sim.write_simulation()
    sim.run_simulation()

    gwf = sim.get_model()
    modelgrid = gwf.modelgrid

    array = np.zeros((modelgrid.ncpl,), dtype=int)
    array[0:85] = 1

    mfsplit = Mf6Splitter(sim)
    new_sim = mfsplit.split_model(array)

    new_sim.set_sim_path(function_tmpdir / "split_model")
    new_sim.write_simulation()
    new_sim.run_simulation()

    original_heads = np.squeeze(gwf.output.head().get_alldata()[-1])

    ml0 = new_sim.get_model("gwf_1_0")
    ml1 = new_sim.get_model("gwf_1_1")
    heads0 = ml0.output.head().get_alldata()[-1]
    heads1 = ml1.output.head().get_alldata()[-1]

    new_heads = mfsplit.reconstruct_array({0: heads0, 1: heads1})

    err_msg = "Heads from original and split models do not match"
    np.testing.assert_allclose(
        new_heads, original_heads, rtol=0.002, atol=0.01, err_msg=err_msg
    )


@requires_exe("mf6")
def test_unstructured_model_splitter(function_tmpdir):
    sim_path = get_example_data_path() / "mf6" / "test006_gwf3"

    sim = MFSimulation.load(sim_ws=sim_path)
    sim.set_sim_path(function_tmpdir)
    sim.write_simulation()
    sim.run_simulation()

    gwf = sim.get_model()
    modelgrid = gwf.modelgrid

    array = np.zeros((modelgrid.nnodes,), dtype=int)
    array[65:] = 1

    mfsplit = Mf6Splitter(sim)
    new_sim = mfsplit.split_model(array)

    new_sim.set_sim_path(function_tmpdir / "split_model")
    new_sim.write_simulation()
    new_sim.run_simulation()

    original_heads = np.squeeze(gwf.output.head().get_alldata()[-1])

    ml0 = new_sim.get_model("gwf_1_0")
    ml1 = new_sim.get_model("gwf_1_1")
    heads0 = ml0.output.head().get_alldata()[-1]
    heads1 = ml1.output.head().get_alldata()[-1]

    new_heads = mfsplit.reconstruct_array({0: heads0, 1: heads1})

    err_msg = "Heads from original and split models do not match"
    np.testing.assert_allclose(new_heads, original_heads, err_msg=err_msg)


@requires_exe("mf6")
@pytest.mark.slow
def test_model_with_lak_sfr_mvr(function_tmpdir):
    sim_path = get_example_data_path() / "mf6" / "test045_lake2tr"

    sim = MFSimulation.load(sim_ws=sim_path)
    sim.set_sim_path(function_tmpdir)
    sim.write_simulation()
    sim.run_simulation()

    gwf = sim.get_model()
    modelgrid = gwf.modelgrid

    array = np.zeros((modelgrid.nrow, modelgrid.ncol), dtype=int)
    array[0:14, :] = 1

    mfsplit = Mf6Splitter(sim)
    new_sim = mfsplit.split_model(array)

    new_sim.set_sim_path(function_tmpdir / "split_model")
    new_sim.write_simulation()
    new_sim.run_simulation()

    original_heads = gwf.output.head().get_alldata()[-1]

    ml0 = new_sim.get_model("lakeex2a_0")
    ml1 = new_sim.get_model("lakeex2a_1")
    heads0 = ml0.output.head().get_alldata()[-1]
    heads1 = ml1.output.head().get_alldata()[-1]

    new_heads = mfsplit.reconstruct_array({0: heads0, 1: heads1})

    err_msg = "Heads from original and split models do not match"
    np.testing.assert_allclose(new_heads, original_heads, err_msg=err_msg)


@requires_pkg("pymetis")
@requires_exe("mf6")
@pytest.mark.slow
def test_metis_splitting_with_lak_sfr(function_tmpdir):
    sim_path = get_example_data_path() / "mf6" / "test045_lake2tr"

    sim = MFSimulation.load(sim_ws=sim_path)
    sim.set_sim_path(function_tmpdir)
    sim.write_simulation()
    sim.run_simulation()

    gwf = sim.get_model()

    mfsplit = Mf6Splitter(sim)
    array = mfsplit.optimize_splitting_mask(nparts=4)

    cellids = sim.get_model().lak.connectiondata.array.cellid
    cellids = [(i[1], i[2]) for i in cellids]
    cellids = tuple(zip(*cellids))
    lak_test = np.unique(array[cellids])
    if len(lak_test) > 2:
        raise AssertionError(
            "optimize_splitting_mask is not correcting for lakes properly"
        )

    new_sim = mfsplit.split_model(array)
    new_sim.set_sim_path(function_tmpdir / "split_model")
    new_sim.write_simulation()
    new_sim.run_simulation()

    original_heads = gwf.output.head().get_alldata()[-1]

    array_dict = {}
    for model in range(4):
        ml = new_sim.get_model(f"lakeex2a_{model}")
        heads0 = ml.output.head().get_alldata()[-1]
        array_dict[model] = heads0

    new_heads = mfsplit.reconstruct_array(array_dict)

    err_msg = "Heads from original and split models do not match"
    np.testing.assert_allclose(new_heads, original_heads, err_msg=err_msg)


@requires_exe("mf6")
@requires_pkg("pymetis")
def test_save_load_node_mapping(function_tmpdir):
    sim_path = get_example_data_path() / "mf6-freyberg"
    new_sim_path = function_tmpdir / "mf6-freyberg/split_model"
    json_file = new_sim_path / "node_map.json"
    nparts = 5

    sim = MFSimulation.load(sim_ws=sim_path)
    sim.set_sim_path(function_tmpdir)
    sim.write_simulation()
    sim.run_simulation()

    original_heads = sim.get_model().output.head().get_alldata()[-1]

    mfsplit = Mf6Splitter(sim)
    array = mfsplit.optimize_splitting_mask(nparts=nparts)
    new_sim = mfsplit.split_model(array)
    new_sim.set_sim_path(new_sim_path)
    new_sim.write_simulation()
    new_sim.run_simulation()
    original_node_map = mfsplit._node_map

    mfsplit.save_node_mapping(json_file)

    new_sim2 = MFSimulation.load(sim_ws=new_sim_path)

    mfsplit2 = Mf6Splitter(new_sim2)
    mfsplit2.load_node_mapping(new_sim2, json_file)
    saved_node_map = mfsplit2._node_map

    for k, v1 in original_node_map.items():
        v2 = saved_node_map[k]
        if not v1 == v2:
            raise AssertionError("Node map read/write not returning proper values")

    array_dict = {}
    for model in range(nparts):
        ml = new_sim2.get_model(f"freyberg_{model}")
        heads0 = ml.output.head().get_alldata()[-1]
        array_dict[model] = heads0

    new_heads = mfsplit2.reconstruct_array(array_dict)
    err_msg = "Heads from original and split models do not match"
    np.testing.assert_allclose(new_heads, original_heads, err_msg=err_msg)


def test_control_records(function_tmpdir):
    nrow = 10
    ncol = 10
    nper = 3

    # create base simulation
    full_ws = function_tmpdir / "full"
    full_ws.mkdir()
    with set_dir(full_ws):
        sim = flopy.mf6.MFSimulation(full_ws)
        ims = flopy.mf6.ModflowIms(sim, complexity="SIMPLE")

        tdis = flopy.mf6.ModflowTdis(
            sim,
            nper=nper,
            perioddata=((1.0, 1, 1.0), (1.0, 1, 1.0), (1.0, 1, 1.0)),
        )

        gwf = flopy.mf6.ModflowGwf(sim, save_flows=True)

        botm2 = np.ones((nrow, ncol)) * 20
        dis = flopy.mf6.ModflowGwfdis(
            gwf,
            nlay=2,
            nrow=nrow,
            ncol=ncol,
            delr=1,
            delc=1,
            top=35,
            botm=[30, botm2],
            idomain=1,
        )

        ic = flopy.mf6.ModflowGwfic(gwf, strt=32)
        npf = flopy.mf6.ModflowGwfnpf(
            gwf,
            k=[
                1.0,
                {
                    "data": np.ones((10, 10)) * 0.75,
                    "filename": "k.l2.txt",
                    "iprn": 1,
                    "factor": 1,
                },
            ],
            k33=[
                np.ones((nrow, ncol)),
                {
                    "data": np.ones((nrow, ncol)) * 0.5,
                    "filename": "k33.l2.bin",
                    "iprn": 1,
                    "factor": 1,
                    "binary": True,
                },
            ],
        )

        wel_rec = [
            ((0, 4, 5), -10),
        ]

        spd = {
            0: wel_rec,
            1: {"data": wel_rec, "filename": "wel.1.txt"},
            2: {"data": wel_rec, "filename": "wel.2.bin", "binary": True},
        }

        wel = flopy.mf6.ModflowGwfwel(gwf, stress_period_data=spd)

        chd_rec = []
        for cond, j in ((30, 0), (22, 9)):
            for i in range(10):
                chd_rec.append(((0, i, j), cond))

        chd = flopy.mf6.ModflowGwfchd(gwf, stress_period_data={0: chd_rec})

    # define splitting array
    arr = np.zeros((10, 10), dtype=int)
    arr[0:5, :] = 1

    # split
    split_ws = function_tmpdir / "split"
    split_ws.mkdir()
    with set_dir(split_ws):
        mfsplit = flopy.mf6.utils.Mf6Splitter(sim)
        new_sim = mfsplit.split_model(arr)

    ml1 = new_sim.get_model("model_1")

    kls = ml1.npf.k._data_storage.layer_storage.multi_dim_list
    if kls[0].data_storage_type.value != 2:
        raise AssertionError("Constants not being preserved for MFArray")

    if kls[1].data_storage_type.value != 3 or kls[1].binary:
        raise AssertionError("External ascii files not being preserved for MFArray")

    k33ls = ml1.npf.k33._data_storage.layer_storage.multi_dim_list
    if k33ls[1].data_storage_type.value != 3 or not k33ls[1].binary:
        raise AssertionError("Binary file input not being preserved for MFArray")

    spd_ls1 = ml1.wel.stress_period_data.get_record(1)
    spd_ls2 = ml1.wel.stress_period_data.get_record(2)

    if spd_ls1["filename"] is None or spd_ls1["binary"]:
        raise AssertionError("External ascii files not being preserved for MFList")

    if spd_ls2["filename"] is None or not spd_ls2["binary"]:
        raise AssertionError(
            "External binary file input not being preserved for MFList"
        )


@requires_exe("mf6")
def test_empty_packages(function_tmpdir):
    new_sim_path = function_tmpdir / "split_model"

    sim = flopy.mf6.MFSimulation(sim_ws=new_sim_path)
    ims = flopy.mf6.ModflowIms(sim, print_option="all", complexity="simple")
    tdis = flopy.mf6.ModflowTdis(sim)

    nrow, ncol = 1, 14
    base_name = "sfr01gwfgwf"
    gwf = flopy.mf6.ModflowGwf(sim, modelname=base_name, save_flows=True)
    dis = flopy.mf6.ModflowGwfdis(gwf, nrow=1, ncol=14, top=0.0, botm=-1.0)
    npf = flopy.mf6.ModflowGwfnpf(
        gwf,
        save_flows=True,
        save_specific_discharge=True,
        icelltype=0,
        k=20.0,
        k33=20.0,
    )
    ic = flopy.mf6.ModflowGwfic(gwf, strt=0.0)
    chd = flopy.mf6.ModflowGwfchd(
        gwf,
        stress_period_data={
            0: [
                ((0, 0, 13), 0.0),
            ]
        },
    )
    wel = flopy.mf6.ModflowGwfwel(
        gwf,
        stress_period_data={
            0: [
                ((0, 0, 0), 1.0),
            ]
        },
    )

    # Build SFR records
    packagedata = [
        (0, (0, 0, 0), 1.0, 1.0, 1.0, 0.0, 0.1, 0.0, 1.0, 1, 1.0, 0),
        (1, (0, 0, 1), 1.0, 1.0, 1.0, 0.0, 0.1, 0.0, 1.0, 2, 1.0, 0),
        (2, (0, 0, 2), 1.0, 1.0, 1.0, 0.0, 0.1, 0.0, 1.0, 2, 1.0, 0),
        (3, (0, 0, 3), 1.0, 1.0, 1.0, 0.0, 0.1, 0.0, 1.0, 2, 1.0, 0),
        (4, (0, 0, 4), 1.0, 1.0, 1.0, 0.0, 0.1, 0.0, 1.0, 2, 1.0, 0),
        (5, (0, 0, 5), 1.0, 1.0, 1.0, 0.0, 0.1, 0.0, 1.0, 2, 1.0, 0),
        (6, (0, 0, 6), 1.0, 1.0, 1.0, 0.0, 0.1, 0.0, 1.0, 2, 1.0, 0),
        (7, (0, 0, 7), 1.0, 1.0, 1.0, 0.0, 0.1, 0.0, 1.0, 2, 1.0, 0),
        (8, (0, 0, 8), 1.0, 1.0, 1.0, 0.0, 0.1, 0.0, 1.0, 2, 1.0, 0),
        (9, (0, 0, 9), 1.0, 1.0, 1.0, 0.0, 0.1, 0.0, 1.0, 2, 1.0, 0),
        (10, (0, 0, 10), 1.0, 1.0, 1.0, 0.0, 0.1, 0.0, 1.0, 2, 1.0, 0),
        (11, (0, 0, 11), 1.0, 1.0, 1.0, 0.0, 0.1, 0.0, 1.0, 2, 1.0, 0),
        (12, (0, 0, 12), 1.0, 1.0, 1.0, 0.0, 0.1, 0.0, 1.0, 2, 1.0, 0),
        (13, (0, 0, 13), 1.0, 1.0, 1.0, 0.0, 0.1, 0.0, 1.0, 1, 1.0, 0),
    ]

    connectiondata = [
        (0, -1),
        (1, 0, -2),
        (2, 1, -3),
        (3, 2, -4),
        (4, 3, -5),
        (5, 4, -6),
        (6, 5, -7),
        (7, 6, -8),
        (8, 7, -9),
        (9, 8, -10),
        (10, 9, -11),
        (11, 10, -12),
        (12, 11, -13),
        (13, 12),
    ]

    sfr = flopy.mf6.ModflowGwfsfr(
        gwf,
        print_input=True,
        print_stage=True,
        print_flows=True,
        save_flows=True,
        stage_filerecord=f"{base_name}.sfr.stg",
        budget_filerecord=f"{base_name}.sfr.bud",
        nreaches=14,
        packagedata=packagedata,
        connectiondata=connectiondata,
        perioddata={
            0: [
                (0, "INFLOW", 1.0),
            ]
        },
    )

    array = np.zeros((nrow, ncol), dtype=int)
    array[0, 7:] = 1
    mfsplit = Mf6Splitter(sim)
    new_sim = mfsplit.split_model(array)

    m0 = new_sim.get_model(f"{base_name}_0")
    m1 = new_sim.get_model(f"{base_name}_1")

    assert not m0.get_package(
        name="chd_0"
    ), f"Empty CHD file written to {base_name}_0 model"
    assert not m1.get_package(
        name="wel_0"
    ), f"Empty WEL file written to {base_name}_1 model"

    mvr_status0 = m0.sfr.mover.array
    mvr_status1 = m0.sfr.mover.array

    assert (
        mvr_status0 and mvr_status1
    ), "Mover status being overwritten in options splitting"


@requires_exe("mf6")
def test_transient_array(function_tmpdir):
    name = "tarr"
    new_sim_path = function_tmpdir / f"{name}_split_model"
    nper = 3
    tdis_data = [
        (300000.0, 1, 1.0),
        (36500.0, 10, 1.5),
        (300000, 1, 1.0),
    ]
    nlay, nrow, ncol = 3, 21, 20
    xlen, ylen = 10000.0, 10500.0
    delc = ylen / nrow
    delr = xlen / ncol
    steady = {0: True, 2: True}
    transient = {1: True}

    sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=new_sim_path)
    ims = flopy.mf6.ModflowIms(sim, complexity="simple")
    tdis = flopy.mf6.ModflowTdis(sim, nper=nper, perioddata=tdis_data)

    gwf = flopy.mf6.ModflowGwf(
        sim,
        modelname=name,
        save_flows=True,
    )
    dis = flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=400,
        botm=[220.0, 200.0, 0],
        length_units="meters",
    )
    npf = flopy.mf6.ModflowGwfnpf(
        gwf,
        icelltype=1,
        k=[50.0, 0.01, 200.0],
        k33=[10.0, 0.01, 20.0],
    )
    sto = flopy.mf6.ModflowGwfsto(
        gwf,
        iconvert=1,
        ss=0.0001,
        sy=0.1,
        steady_state=steady,
        transient=transient,
    )
    ic = flopy.mf6.ModflowGwfic(gwf, strt=400.0)
    oc = flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord=f"{name}.hds",
        budget_filerecord=f"{name}.cbc",
        saverecord={0: [("head", "all"), ("budget", "all")]},
    )

    rch = flopy.mf6.ModflowGwfrcha(gwf, recharge=0.005)

    chd_spd = [(0, i, ncol - 1, 320) for i in range(nrow)]
    chd = flopy.mf6.ModflowGwfchd(
        gwf,
        stress_period_data=chd_spd,
    )

    well_spd = {
        0: [(0, 10, 9, -75000.0)],
        2: [(0, 10, 9, -75000.0), (2, 12, 4, -100000.0)],
    }
    wel = flopy.mf6.ModflowGwfwel(
        gwf,
        stress_period_data=well_spd,
    )

    sarr = np.ones((nrow, ncol), dtype=int)
    sarr[:, int(ncol / 2) :] = 2
    mfsplit = Mf6Splitter(sim)
    new_sim = mfsplit.split_model(sarr)

    for name in new_sim.model_names:
        g = new_sim.get_model(name)
        d = {}
        for key in (
            0,
            2,
        ):
            d[key] = g.sto.steady_state.get_data(key)
        assert d == steady, (
            "storage steady_state dictionary " + f"does not match for model '{name}'"
        )
        d = {}
        for key in (1,):
            d[key] = g.sto.transient.get_data(key)
        assert d == transient, (
            "storage package transient dictionary "
            + f"does not match for model '{name}'"
        )


@requires_exe("mf6")
@requires_pkg("pymetis")
def test_idomain_none(function_tmpdir):
    name = "id_test"
    sim_path = function_tmpdir
    new_sim_path = function_tmpdir / f"{name}_split_model"

    tdis_data = [
        (0.0, 1, 1.0),
        (300000.0, 1, 1.0),
        (36500.0, 10, 1.5),
        (300000, 1, 1.0),
    ]

    nper = len(tdis_data)
    nlay, nrow, ncol = 3, 21, 20
    xlen, ylen = 10000.0, 10500.0
    delc = ylen / nrow
    delr = xlen / ncol

    top = 400.0
    botm = [220.0, 200.0, 0]
    K11 = [50.0, 0.01, 200.0]
    K33 = [10.0, 0.01, 20.0]
    Ss, Sy = 0.0001, 0.1
    H_east = 320.0
    recharge = 0.005
    idomain = None

    sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=sim_path)
    tdis = flopy.mf6.ModflowTdis(
        sim, nper=nper, perioddata=tdis_data, time_units="days"
    )
    ims = flopy.mf6.ModflowIms(
        sim,
        complexity="simple",
        print_option="all",
        outer_dvclose=1e-6,
        inner_dvclose=1e-6,
    )

    gwf = flopy.mf6.ModflowGwf(sim, modelname=name, save_flows=True)
    dis = flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
        length_units="meters",
        idomain=idomain,
    )
    npf = flopy.mf6.ModflowGwfnpf(
        gwf, icelltype=1, save_specific_discharge=True, k=K11, k33=K33
    )
    sto = flopy.mf6.ModflowGwfsto(
        gwf,
        iconvert=1,
        ss=Ss,
        sy=Sy,
        steady_state={0: True, 3: True},
        transient={2: True},
    )
    ic = flopy.mf6.ModflowGwfic(gwf, strt=top)
    oc = flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord=f"{name}.hds",
        budget_filerecord=f"{name}.cbc",
        saverecord={0: [("head", "all"), ("budget", "all")]},
    )
    rch = flopy.mf6.ModflowGwfrcha(gwf, recharge=recharge)

    chd_spd = [(0, i, ncol - 1, H_east) for i in range(nrow)]
    chd = flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chd_spd)

    well_spd = {
        1: [(0, 10, 9, -75000.0)],
        2: [(0, 10, 9, -75000.0), (2, 12, 4, -100000.0)],
    }
    wel = flopy.mf6.ModflowGwfwel(gwf, stress_period_data=well_spd)
    sim.write_simulation()
    sim.run_simulation()

    ms = Mf6Splitter(sim)
    sarr = ms.optimize_splitting_mask(3)
    new_sim = ms.split_model(sarr)
    new_sim.set_sim_path(new_sim_path)
    new_sim.write_simulation()
    new_sim.run_simulation()

    kstpkper = (9, 2)
    head = gwf.output.head().get_data(kstpkper=kstpkper)
    head_dict = {}
    for idx, modelname in enumerate(new_sim.model_names):
        mnum = int(modelname.split("_")[-1])
        h = new_sim.get_model(modelname).output.head().get_data(kstpkper=kstpkper)
        head_dict[mnum] = h
    new_head = ms.reconstruct_array(head_dict)

    err_msg = "Heads from original and split models do not match"
    np.testing.assert_allclose(new_head, head, atol=1e-07, err_msg=err_msg)


@requires_exe("mf6")
def test_unstructured_complex_disu(function_tmpdir):
    sim_path = function_tmpdir
    split_sim_path = sim_path / "model_split"

    # build the simulation structure
    sim = flopy.mf6.MFSimulation(sim_ws=sim_path)
    ims = flopy.mf6.ModflowIms(sim, complexity="SIMPLE")
    tdis = flopy.mf6.ModflowTdis(sim)

    mname = "disu_model"
    gwf = flopy.mf6.ModflowGwf(sim, modelname=mname)

    # start structured and then create a USG from it
    nlay = 1
    nrow = 10
    ncol = 10
    delc = np.ones((nrow,))
    delr = np.ones((ncol,))
    top = np.ones((nrow, ncol))
    botm = np.zeros((nlay, nrow, ncol))
    idomain = np.ones(botm.shape, dtype=int)
    idomain[0, 1, 4] = 0
    idomain[0, 8, 5] = 0

    grid = flopy.discretization.StructuredGrid(
        delc=delc, delr=delr, top=top, botm=botm, idomain=idomain
    )

    # build the USG connection information
    neighbors = grid.neighbors(method="rook", reset=True)
    iac, ja, ihc, cl12, hwva, angldegx = [], [], [], [], [], []
    for cell, neigh in neighbors.items():
        iac.append(len(neigh) + 1)
        ihc.extend(
            [
                1,
            ]
            * (len(neigh) + 1)
        )
        ja.extend(
            [
                cell,
            ]
            + neigh
        )
        cl12.extend(
            [
                0,
            ]
            + [
                1,
            ]
            * len(neigh)
        )
        hwva.extend(
            [
                0,
            ]
            + [
                1,
            ]
            * len(neigh)
        )
        adx = [
            0,
        ]
        for n in neigh:
            ev = cell - n
            if ev == -1 * ncol:
                adx.append(270)
            elif ev == ncol:
                adx.append(90)
            elif ev == -1:
                adx.append(0)
            else:
                adx.append(180)
        angldegx.extend(adx)

    # build iverts and verts. Do not use shared iverts and mess with verts a
    #   tiny bit
    verts, cell2d = [], []
    xverts, yverts = grid.cross_section_vertices
    xcenters = grid.xcellcenters.ravel()
    ycenters = grid.ycellcenters.ravel()
    ivert = 0
    for cell_num, xvs in enumerate(xverts):
        if (cell_num - 3) % 10 == 0:
            xvs[2] -= 0.001
            xvs[3] -= 0.001
        yvs = yverts[cell_num]

        c2drec = [cell_num, xcenters[cell_num], ycenters[cell_num], len(xvs)]
        for ix, vert in enumerate(xvs[:-1]):
            c2drec.append(ivert)
            verts.append([ivert, vert, yvs[ix]])
            ivert += 1

        c2drec.append(c2drec[4])
        cell2d.append(c2drec)

    nodes = len(cell2d)
    nja = len(ja)
    nvert = len(verts)

    dis = flopy.mf6.ModflowGwfdisu(
        gwf,
        nodes=nodes,
        nja=nja,
        nvert=nvert,
        top=np.ravel(grid.top),
        bot=np.ravel(grid.botm),
        area=np.ones((nodes,)),
        idomain=grid.idomain.ravel(),
        iac=iac,
        ja=ja,
        ihc=ihc,
        cl12=cl12,
        hwva=hwva,
        angldegx=angldegx,
        vertices=verts,
        cell2d=cell2d,
    )

    # build npf, ic, CHD, OC package
    npf = flopy.mf6.ModflowGwfnpf(gwf)
    ic = flopy.mf6.ModflowGwfic(gwf)

    spd = []
    for i in range(nrow):
        spd.append((0 + (i * 10), 0.9))
        spd.append((9 + (i * 10), 0.5))

    chd = flopy.mf6.ModflowGwfchd(gwf, stress_period_data=spd)

    spd = {0: [("HEAD", "LAST")]}
    oc = flopy.mf6.ModflowGwfoc(gwf, head_filerecord=f"{mname}.hds", saverecord=spd)

    sim.write_simulation()
    sim.run_simulation()

    heads = gwf.output.head().get_alldata()[-1]

    array = np.zeros((nrow, ncol), dtype=int)
    array[:, 5:] = 1

    mfsplit = Mf6Splitter(sim)
    new_sim = mfsplit.split_model(array)

    new_sim.set_sim_path(split_sim_path)
    new_sim.write_simulation()
    new_sim.run_simulation()

    gwf0 = new_sim.get_model(f"{mname}_0")
    gwf1 = new_sim.get_model(f"{mname}_1")

    heads0 = gwf0.output.head().get_alldata()[-1]
    heads1 = gwf1.output.head().get_alldata()[-1]

    new_heads = mfsplit.reconstruct_array({0: heads0, 1: heads1})

    diff = np.abs(heads - new_heads)
    if np.max(diff) > 1e-07:
        raise AssertionError("Reconstructed head results outside of tolerance")


@requires_exe("mf6")
@requires_pkg("pymetis")
@requires_pkg("scipy")
def test_multi_model(function_tmpdir):
    from scipy.spatial import KDTree

    def string2geom(geostring, conversion=None):
        if conversion is None:
            multiplier = 1.0
        else:
            multiplier = float(conversion)
        res = []
        for line in geostring.split("\n"):
            if not any(line):
                continue
            line = line.strip()
            line = line.split(" ")
            x = float(line[0]) * multiplier
            y = float(line[1]) * multiplier
            res.append((x, y))
        return res

    sim_path = function_tmpdir
    split_sim_path = sim_path / "model_split"
    data_path = get_example_data_path()

    ascii_file = data_path / "geospatial/fine_topo.asc"
    fine_topo = flopy.utils.Raster.load(ascii_file)

    with open(data_path / "groundwater2023/geometries.yml") as foo:
        geometry = yaml.safe_load(foo)

    Lx = 180000
    Ly = 100000
    dx = 2500.0
    dy = 2500.0
    nrow = int(Ly / dy) + 1
    ncol = int(Lx / dx) + 1
    boundary = string2geom(geometry["boundary"])
    bp = np.array(boundary)

    stream_segs = (
        geometry["streamseg1"],
        geometry["streamseg2"],
        geometry["streamseg3"],
        geometry["streamseg4"],
    )
    sgs = [string2geom(sg) for sg in stream_segs]

    modelgrid = flopy.discretization.StructuredGrid(
        nlay=1,
        delr=np.full(ncol, dx),
        delc=np.full(nrow, dy),
        xoff=0.0,
        yoff=0.0,
        top=np.full((nrow, ncol), 1000.0),
        botm=np.full((1, nrow, ncol), -100.0),
    )

    ixs = flopy.utils.GridIntersect(modelgrid, method="vertex", rtree=True)
    result = ixs.intersect(
        [
            boundary,
        ],
        shapetype="Polygon",
    )
    r, c = list(zip(*list(result.cellids)))
    idomain = np.zeros(modelgrid.shape, dtype=int)
    idomain[:, r, c] = 1
    modelgrid._idomain = idomain

    top = fine_topo.resample_to_grid(
        modelgrid,
        band=fine_topo.bands[0],
        method="linear",
        extrapolate_edges=True,
    )
    modelgrid._top = top

    # intersect stream segments
    cellids = []
    lengths = []
    for sg in stream_segs:
        sg = string2geom(sg)
        v = ixs.intersect(sg, shapetype="LineString", sort_by_cellid=True)
        cellids += v["cellids"].tolist()
        lengths += v["lengths"].tolist()

    r, c = list(zip(*cellids))
    idomain[:, r, c] = 2
    modelgrid._idomain = idomain

    nlay = 5
    dv0 = 5.0
    hyd_cond = 10.0
    hk = np.full((nlay, nrow, ncol), hyd_cond)
    hk[1, :, 25:] = hyd_cond * 0.001
    hk[3, :, 10:] = hyd_cond * 0.00005

    # drain leakage
    leakance = hyd_cond / (0.5 * dv0)

    drn_data = []
    for cellid, length in zip(cellids, lengths):
        x = modelgrid.xcellcenters[cellid]
        width = 5.0 + (14.0 / Lx) * (Lx - x)
        conductance = leakance * length * width
        if not isinstance(cellid, tuple):
            cellid = (cellid,)
        drn_data.append((0, *cellid, top[cellid], conductance))

    discharge_data = []
    area = dx * dy
    for r in range(nrow):
        for c in range(ncol):
            if idomain[0, r, c] == 1:
                conductance = leakance * area
                discharge_data.append((0, r, c, top[r, c] - 0.5, conductance, 1.0))

    topc = np.zeros((nlay, nrow, ncol), dtype=float)
    botm = np.zeros((nlay, nrow, ncol), dtype=float)
    topc[0] = modelgrid.top.copy()
    botm[0] = topc[0] - dv0
    for idx in range(1, nlay):
        dv0 *= 1.5
        topc[idx] = botm[idx - 1]
        botm[idx] = topc[idx] - dv0

    strt = np.tile([modelgrid.top], (nlay, 1, 1))
    idomain = np.tile(
        [
            modelgrid.idomain[0],
        ],
        (5, 1, 1),
    )

    # setup recharge
    dist_from_riv = 10000.0

    grid_xx = modelgrid.xcellcenters
    grid_yy = modelgrid.ycellcenters
    riv_idxs = np.array(cellids)
    riv_xx = grid_xx[riv_idxs[:, 0], riv_idxs[:, 1]]
    riv_yy = grid_yy[riv_idxs[:, 0], riv_idxs[:, 1]]

    river_xy = np.column_stack((riv_xx.ravel(), riv_yy.ravel()))
    grid_xy = np.column_stack((grid_xx.ravel(), grid_yy.ravel()))
    tree = KDTree(river_xy)
    distance, index = tree.query(grid_xy)

    index2d = index.reshape(nrow, ncol)
    distance2d = distance.reshape(nrow, ncol)

    mountain_array = np.asarray(distance2d > dist_from_riv).nonzero()
    mountain_idxs = np.array(list(zip(mountain_array[0], mountain_array[1])))

    valley_array = np.asarray(distance2d <= dist_from_riv).nonzero()
    valley_idxs = np.array(list(zip(valley_array[0], valley_array[1])))

    max_recharge = 0.0001

    rch_orig = max_recharge * np.ones((nrow, ncol))

    rch_mnt = np.zeros((nrow, ncol))
    for idx in mountain_idxs:
        rch_mnt[idx[0], idx[1]] = max_recharge

    rch_val = np.zeros((nrow, ncol))
    for idx in valley_idxs:
        rch_val[idx[0], idx[1]] = max_recharge

    sim = flopy.mf6.MFSimulation(
        sim_ws=sim_path,
        exe_name="mf6",
        memory_print_option="summary",
    )

    nper = 10
    nsteps = 1
    year = 365.25
    dt = 1000 * year
    tdis = flopy.mf6.ModflowTdis(
        sim, nper=nper, perioddata=nper * [(nsteps * dt, nsteps, 1.0)]
    )

    gwfname = "gwf"

    imsgwf = flopy.mf6.ModflowIms(
        sim,
        complexity="simple",
        print_option="SUMMARY",
        linear_acceleration="bicgstab",
        outer_maximum=1000,
        inner_maximum=100,
        outer_dvclose=1e-4,
        inner_dvclose=1e-5,
        preconditioner_levels=2,
        relaxation_factor=0.0,
        filename=f"{gwfname}.ims",
    )

    gwf = flopy.mf6.ModflowGwf(
        sim,
        modelname=gwfname,
        print_input=False,
        save_flows=True,
        newtonoptions="NEWTON UNDER_RELAXATION",
    )

    dis = flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=dx,
        delc=dy,
        idomain=idomain,
        top=modelgrid.top,
        botm=botm,
        xorigin=0.0,
        yorigin=0.0,
    )

    ic = flopy.mf6.ModflowGwfic(gwf, strt=strt)

    npf = flopy.mf6.ModflowGwfnpf(
        gwf,
        save_specific_discharge=True,
        icelltype=1,
        k=hk,
    )

    sto = flopy.mf6.ModflowGwfsto(
        gwf,
        save_flows=True,
        iconvert=1,
        ss=0.00001,
        sy=0.35,
        steady_state={0: True, 1: False},
        transient={0: False, 1: True},
    )

    rch0 = flopy.mf6.ModflowGwfrcha(
        gwf,
        pname="rch_original",
        recharge={0: rch_orig, 1: 0.0},
        filename="gwf_original.rch",
    )

    rch1 = flopy.mf6.ModflowGwfrcha(
        gwf,
        pname="rch_mountain",
        recharge={1: rch_mnt},
        auxiliary="CONCENTRATION",
        aux={1: 1.0},
        filename="gwf_mountain.rch",
    )

    rch2 = flopy.mf6.ModflowGwfrcha(
        gwf,
        pname="rch_valley",
        recharge={1: rch_val},
        auxiliary="CONCENTRATION",
        aux={1: 1.0},
        filename="gwf_valley.rch",
    )

    drn = flopy.mf6.ModflowGwfdrn(
        gwf,
        stress_period_data=drn_data,
        pname="river",
        filename=f"{gwfname}_riv.drn",
    )

    drn_gwd = flopy.mf6.ModflowGwfdrn(
        gwf,
        auxiliary=["depth"],
        auxdepthname="depth",
        stress_period_data=discharge_data,
        pname="gwd",
        filename=f"{gwfname}_gwd.drn",
    )

    wel_spd = {0: [[4, 20, 30, 0.0], [2, 20, 60, 0.0], [2, 30, 50, 0.0]]}

    wel = flopy.mf6.ModflowGwfwel(
        gwf,
        print_input=False,
        print_flows=False,
        stress_period_data=wel_spd,
    )

    oc = flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord=f"{gwf.name}.hds",
        budget_filerecord=f"{gwf.name}.cbc",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        printrecord=[("BUDGET", "ALL")],
    )

    sim.register_ims_package(imsgwf, [gwf.name])

    def build_gwt_model(sim, gwtname, rch_package):
        conc_start = 0.0
        diffc = 0.0
        alphal = 0.1
        porosity = 0.35
        gwf = sim.get_model("gwf")
        modelgrid = gwf.modelgrid

        gwt = flopy.mf6.ModflowGwt(
            sim,
            modelname=gwtname,
            print_input=False,
            save_flows=True,
        )

        nlay, nrow, ncol = modelgrid.shape

        dis = flopy.mf6.ModflowGwtdis(
            gwt,
            nlay=nlay,
            nrow=nrow,
            ncol=ncol,
            delr=dx,
            delc=dy,
            idomain=modelgrid.idomain,
            top=modelgrid.top,
            botm=botm,
            xorigin=0.0,
            yorigin=0.0,
        )

        # initial conditions
        ic = flopy.mf6.ModflowGwtic(gwt, strt=conc_start, filename=f"{gwtname}.ic")

        # advection
        adv = flopy.mf6.ModflowGwtadv(gwt, scheme="tvd", filename=f"{gwtname}.adv")

        # dispersion
        dsp = flopy.mf6.ModflowGwtdsp(
            gwt,
            diffc=diffc,
            alh=alphal,
            alv=alphal,
            ath1=0.0,
            atv=0.0,
            filename=f"{gwtname}.dsp",
        )

        # mass storage and transfer
        mst = flopy.mf6.ModflowGwtmst(gwt, porosity=porosity, filename=f"{gwtname}.mst")

        # sources
        sourcerecarray = [
            (rch_package, "AUX", "CONCENTRATION"),
        ]
        ssm = flopy.mf6.ModflowGwtssm(
            gwt, sources=sourcerecarray, filename=f"{gwtname}.ssm"
        )

        # output control
        oc = flopy.mf6.ModflowGwtoc(
            gwt,
            budget_filerecord=f"{gwtname}.cbc",
            concentration_filerecord=f"{gwtname}.ucn",
            saverecord=[("CONCENTRATION", "ALL"), ("BUDGET", "ALL")],
        )

        return gwt

    imsgwt = flopy.mf6.ModflowIms(
        sim,
        complexity="complex",
        print_option="SUMMARY",
        linear_acceleration="bicgstab",
        outer_maximum=1000,
        inner_maximum=100,
        outer_dvclose=1e-4,
        inner_dvclose=1e-5,
        filename="gwt.ims",
    )

    gwt_mnt = build_gwt_model(sim, "gwt_mnt", "rch_mountain")
    sim.register_ims_package(imsgwt, [gwt_mnt.name])

    gwt_val = build_gwt_model(sim, "gwt_val", "rch_valley")
    sim.register_ims_package(imsgwt, [gwt_val.name])

    gwfgwt = flopy.mf6.ModflowGwfgwt(
        sim,
        exgtype="GWF6-GWT6",
        exgmnamea=gwfname,
        exgmnameb=gwt_mnt.name,
        filename="gwfgwt_mnt.exg",
    )

    gwfgwt = flopy.mf6.ModflowGwfgwt(
        sim,
        exgtype="GWF6-GWT6",
        exgmnamea=gwfname,
        exgmnameb=gwt_val.name,
        filename="gwfgwt_val.exg",
    )

    sim.write_simulation()
    sim.run_simulation()

    nparts = 2
    mfs = Mf6Splitter(sim)
    array = mfs.optimize_splitting_mask(nparts)
    new_sim = mfs.split_multi_model(array)
    new_sim.set_sim_path(split_sim_path)
    new_sim.write_simulation()
    new_sim.run_simulation()

    # compare results for each of the models
    splits = list(range(nparts))
    for name in sim.model_names:
        gwm = sim.get_model(name)
        if "concentration()" in gwm.output.methods():
            X = gwm.output.concentration().get_alldata()[-1]
        else:
            X = gwm.output.head().get_alldata()[-1]

        array_dict = {}
        for split in splits:
            mname = f"{name}_{split}"
            sp_gwm = new_sim.get_model(mname)
            if "concentration()" in sp_gwm.output.methods():
                X0 = sp_gwm.output.concentration().get_alldata()[-1]
            else:
                X0 = sp_gwm.output.head().get_alldata()[-1]

            array_dict[split] = X0

        X_split = mfs.reconstruct_array(array_dict)

        err_msg = f"Outputs from {name} and split model are not within tolerance"
        X_split[idomain == 0] = np.nan
        X[idomain == 0] = np.nan
        if name == "gwf":
            np.testing.assert_allclose(X, X_split, equal_nan=True, err_msg=err_msg)
        else:
            diff = np.abs(X_split - X)
            diff = np.nansum(diff)
            if diff > 10.25:
                raise AssertionError(
                    "Difference between output arrays: "
                    f"{diff :.2f} greater than tolerance"
                )
