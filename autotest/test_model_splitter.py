import numpy as np
import flopy
import pytest
from autotest.conftest import get_example_data_path
from modflow_devtools.markers import requires_exe, requires_pkg

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

    sim = flopy.mf6.MFSimulation()
    ims = flopy.mf6.ModflowIms(sim, complexity="SIMPLE")

    tdis = flopy.mf6.ModflowTdis(
        sim,
        nper=nper,
        perioddata=(
            (1., 1, 1.),
            (1., 1, 1.),
            (1., 1, 1.)
        )
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
        idomain=1
    )

    ic = flopy.mf6.ModflowGwfic(gwf, strt=32)
    npf = flopy.mf6.ModflowGwfnpf(
        gwf,
        k=[
            1.,
            {
                "data": np.ones((10, 10)) * 0.75,
                "filename": "k.l2.txt",
                "iprn": 1,
                "factor": 1
            }
        ],
        k33=[
            np.ones((nrow, ncol)),
            {
                "data": np.ones((nrow, ncol)) * 0.5,
                "filename": "k33.l2.bin",
                "iprn": 1,
                "factor": 1,
                "binary": True
            }
        ]

    )

    wel_rec = [((0, 4, 5), -10), ]

    spd = {
        0: wel_rec,
        1: {
            "data": wel_rec,
            "filename": "wel.1.txt"
        },
        2: {
            "data": wel_rec,
            "filename": "wel.2.bin",
            "binary": True
        }
    }

    wel = flopy.mf6.ModflowGwfwel(
        gwf,
        stress_period_data=spd
    )

    chd_rec = []
    for cond, j in ((30, 0), (22, 9)):
        for i in range(10):
            chd_rec.append(((0, i, j), cond))

    chd = flopy.mf6.ModflowGwfchd(
        gwf,
        stress_period_data={0: chd_rec}
    )

    arr = np.zeros((10, 10), dtype=int)
    arr[0:5, :] = 1

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

    k33ls =  ml1.npf.k33._data_storage.layer_storage.multi_dim_list
    if k33ls[1].data_storage_type.value != 3 or not k33ls[1].binary:
        raise AssertionError(
            "Binary file input not being preserved for MFArray"
        )

    spd_ls1 = ml1.wel.stress_period_data._data_storage[1].layer_storage.multi_dim_list[0]
    spd_ls2 = ml1.wel.stress_period_data._data_storage[2].layer_storage.multi_dim_list[0]

    if spd_ls1.data_storage_type.value != 3 or spd_ls1.binary:
        raise AssertionError(
            "External ascii files not being preserved for MFList"
        )

    if spd_ls2.data_storage_type.value != 3 or not spd_ls2.binary:
        raise AssertionError(
            "External binary file input not being preseved for MFList"
        )


