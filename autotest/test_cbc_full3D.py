import os
from pathlib import Path

import numpy as np
import pytest
from autotest.conftest import get_example_data_path
from modflow_devtools.markers import requires_exe

from flopy.mf6 import MFSimulation, ModflowGwfoc
from flopy.modflow import Modflow
from flopy.utils import CellBudgetFile

example_data_path = get_example_data_path()
mf2005_paths = [
    str(example_data_path / "freyberg"),
]
mf6_paths = [
    str(example_data_path / "mf6-freyberg"),
    str(example_data_path / "mf6" / "test003_gwfs_disv"),
    str(example_data_path / "mf6" / "test003_gwftri_disv"),
]


def load_mf2005(path, ws_out):
    name_file = f"{Path(path).name}.nam"
    ml = Modflow.load(
        name_file,
        model_ws=str(path),
        exe_name="mf2005",
        check=False,
    )

    # change work space
    # ws_out = os.path.join(baseDir, name)
    ml.change_model_ws(ws_out)

    # save all budget data to a cell-by cell file
    oc = ml.get_package("OC")
    oc.reset_budgetunit()
    oc.stress_period_data = {(0, 0): ["save budget"]}

    return ml


@pytest.mark.mf6
def load_mf6(path, ws_out):
    sim = MFSimulation.load(
        sim_name=Path(path).name,
        exe_name="mf6",
        sim_ws=str(path),
    )

    # change work space
    sim.set_sim_path(ws_out)

    # get the groundwater flow model(s) and redefine the output control
    # file to save cell by cell output
    models = sim.model_names
    for model in models:
        gwf = sim.get_model(model)
        gwf.name_file.save_flows = True

        gwf.remove_package("oc")
        budget_filerecord = f"{model}.cbc"
        oc = ModflowGwfoc(
            gwf,
            budget_filerecord=budget_filerecord,
            saverecord=[("BUDGET", "ALL")],
        )

    return sim


def cbc_eval_size(cbcobj, nnodes, shape3d):
    cbc_pth = cbcobj.filename

    assert cbcobj.nnodes == nnodes, (
        f"{cbc_pth} nnodes ({cbcobj.nnodes}) " f"does not equal {nnodes}"
    )
    a = np.squeeze(np.ones(cbcobj.shape, dtype=float))
    b = np.squeeze(np.ones(shape3d, dtype=float))
    assert a.shape == b.shape, (
        f"{cbc_pth} shape {cbcobj.shape} " f"does not conform to {shape3d}"
    )


def cbc_eval_data(cbcobj, shape3d):
    cbc_pth = cbcobj.filename
    print(f"{cbc_pth}:\n")
    cbcobj.list_unique_records()

    names = cbcobj.get_unique_record_names(decode=True)
    times = cbcobj.get_times()
    for name in names:
        text = name.strip()
        arr = np.squeeze(
            cbcobj.get_data(text=text, totim=times[0], full3D=True)[0]
        )
        if text != "FLOW-JA-FACE":
            b = np.squeeze(np.ones(shape3d, dtype=float))
            assert arr.shape == b.shape, (
                f"{cbc_pth} shape {arr.shape} for '{text}' budget item "
                f"does not conform to {shape3d}"
            )


def cbc_eval(cbcobj, nnodes, shape3d, modelgrid):
    cbc_pth = cbcobj.filename
    cbc_eval_size(cbcobj, nnodes, shape3d)
    cbc_eval_data(cbcobj, shape3d)
    cbcobj.close()

    cobj_mg = CellBudgetFile(
        cbc_pth,
        modelgrid=modelgrid,
        verbose=True,
    )
    cbc_eval_size(cobj_mg, nnodes, shape3d)
    cbc_eval_data(cobj_mg, shape3d)
    cobj_mg.close()


@requires_exe("mf6")
@pytest.mark.mf6
@pytest.mark.parametrize("path", mf6_paths)
def test_cbc_full3D_mf6(function_tmpdir, path):
    sim = load_mf6(path, str(function_tmpdir))

    # write the simulation
    sim.write_simulation()

    # run the simulation
    sim.run_simulation()

    # get the groundwater model and determine the size of the model grid
    gwf_name = list(sim.model_names)[0]
    gwf = sim.get_model(gwf_name)
    nnodes, shape3d = gwf.modelgrid.nnodes, gwf.modelgrid.shape

    # get the cell by cell object
    cbc = gwf.output.budget()

    # evaluate the full3D option
    cbc_eval(cbc, nnodes, shape3d, gwf.modelgrid)


@requires_exe("mf2005")
@pytest.mark.parametrize("path", mf2005_paths)
def test_cbc_full3D_mf2005(function_tmpdir, path):
    ml = load_mf2005(path, str(function_tmpdir))

    # write the model
    ml.write_input()

    # run the model
    ml.run_model()

    # determine the size of the model grid
    nnodes, shape3d = ml.modelgrid.nnodes, ml.modelgrid.shape

    # get the cell by cell object
    fpth = os.path.join(str(function_tmpdir), f"{Path(path).name}.cbc")
    cbc = CellBudgetFile(fpth)

    # evaluate the full3D option
    cbc_eval(cbc, nnodes, shape3d, ml.modelgrid)
