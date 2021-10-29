import os
import sys
import pytest

import numpy as np
import flopy

from ci_framework import baseTestDir, flopyTest

ex_pths = (
    os.path.join("..", "examples", "data", "freyberg"),
    os.path.join("..", "examples", "data", "mf6-freyberg"),
    os.path.join("..", "examples", "data", "mf6", "test003_gwfs_disv"),
    os.path.join("..", "examples", "data", "mf6", "test003_gwftri_disv"),
)
ismf6_lst = ["mf6" in pth for pth in ex_pths]
names = [os.path.basename(pth) for pth in ex_pths]

mf6_exe = "mf6"
mf2005_exe = "mf2005"
if sys.platform == "win32":
    mf6_exe += ".exe"
    mf2005_exe += ".exe"

baseDir = baseTestDir(__file__, relPath="temp", verbose=True)


def load_mf2005(name, ws_in, ws_out):
    name_file = f"{name}.nam"
    ml = flopy.modflow.Modflow.load(
        name_file,
        model_ws=ws_in,
        exe_name=mf2005_exe,
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


def load_mf6(name, ws_in, ws_out):
    sim = flopy.mf6.MFSimulation.load(
        sim_name=name,
        exe_name=mf6_exe,
        sim_ws=ws_in,
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
        oc = flopy.mf6.ModflowGwfoc(
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

    cobj_mg = flopy.utils.CellBudgetFile(
        cbc_pth,
        modelgrid=modelgrid,
        verbose=True,
    )
    cbc_eval_size(cobj_mg, nnodes, shape3d)
    cbc_eval_data(cobj_mg, shape3d)
    cobj_mg.close()

    return


def mf6_eval(name, ws_in):
    ws_out = f"{baseDir}_{name}"
    testFramework = flopyTest(verbose=True, testDirs=ws_out, create=True)

    sim = load_mf6(name, ws_in, ws_out)

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

    # teardown test
    testFramework.teardown()

    return


def mf2005_eval(name, ws_in):
    ws_out = f"{baseDir}_{name}"
    testFramework = flopyTest(verbose=True, testDirs=ws_out, create=True)

    ml = load_mf2005(name, ws_in, ws_out)

    # write the model
    ml.write_input()

    # run the model
    ml.run_model()

    # determine the size of the model grid
    nnodes, shape3d = ml.modelgrid.nnodes, ml.modelgrid.shape

    # get the cell by cell object
    fpth = os.path.join(ws_out, f"{name}.cbc")
    cbc = flopy.utils.CellBudgetFile(fpth)

    # evaluate the full3D option
    cbc_eval(cbc, nnodes, shape3d, ml.modelgrid)

    # teardown test
    testFramework.teardown()

    return


@pytest.mark.parametrize(
    "name, ismf6, ws_in",
    zip(names, ismf6_lst, ex_pths),
)
def test_cbc_full3D(name, ismf6, ws_in):
    if ismf6:
        mf6_eval(name, ws_in)
    else:
        mf2005_eval(name, ws_in)


def main():
    for (name, ismf6, ws_in) in zip(names, ismf6_lst, ex_pths):
        if ismf6:
            mf6_eval(name, ws_in)
        else:
            mf2005_eval(name, ws_in)


if __name__ == "__main__":
    main()
