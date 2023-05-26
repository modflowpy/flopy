import numpy as np

import pytest
from autotest.conftest import get_example_data_path
from modflow_devtools.markers import requires_exe

from flopy.mf6.utils import Mf6Splitter
from flopy.mf6 import MFSimulation


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