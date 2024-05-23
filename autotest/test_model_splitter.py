import numpy as np
import pytest
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
        array[row, ncol:] = 2

    mfsplit = Mf6Splitter(sim)
    new_sim = mfsplit.split_model(array)

    new_sim.set_sim_path(function_tmpdir / "split_model")
    new_sim.write_simulation()
    new_sim.run_simulation()

    original_heads = gwf.output.head().get_alldata()[-1]

    ml0 = new_sim.get_model("freyberg_1")
    ml1 = new_sim.get_model("freyberg_2")

    heads0 = ml0.output.head().get_alldata()[-1]
    heads1 = ml1.output.head().get_alldata()[-1]

    new_heads = mfsplit.reconstruct_array({1: heads0, 2: heads1})

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
            raise AssertionError(
                "Node map read/write not returning proper values"
            )

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
        raise AssertionError(
            "External ascii files not being preserved for MFArray"
        )

    k33ls = ml1.npf.k33._data_storage.layer_storage.multi_dim_list
    if k33ls[1].data_storage_type.value != 3 or not k33ls[1].binary:
        raise AssertionError(
            "Binary file input not being preserved for MFArray"
        )

    spd_ls1 = ml1.wel.stress_period_data.get_record(1)
    spd_ls2 = ml1.wel.stress_period_data.get_record(2)

    if spd_ls1["filename"] is None or spd_ls1["binary"]:
        raise AssertionError(
            "External ascii files not being preserved for MFList"
        )

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

    if "chd_0" in m0.package_dict:
        raise AssertionError(f"Empty CHD file written to {base_name}_0 model")

    if "wel_0" in m1.package_dict:
        raise AssertionError(f"Empty WEL file written to {base_name}_1 model")

    mvr_status0 = m0.sfr.mover.array
    mvr_status1 = m0.sfr.mover.array

    if not mvr_status0 or not mvr_status1:
        raise AssertionError(
            "Mover status being overwritten in options splitting"
        )


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
            d[key] = g.sto.steady_state.get_data(key).get_data()
        assert d == steady, (
            "storage steady_state dictionary "
            + f"does not match for model '{name}'"
        )
        d = {}
        for key in (1,):
            d[key] = g.sto.transient.get_data(key).get_data()
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
        h = (
            new_sim.get_model(modelname)
            .output.head()
            .get_data(kstpkper=kstpkper)
        )
        head_dict[mnum] = h
    new_head = ms.reconstruct_array(head_dict)

    err_msg = "Heads from original and split models do not match"
    np.testing.assert_allclose(new_head, head, atol=1e-07, err_msg=err_msg)
