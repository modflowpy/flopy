import shutil
from os.path import dirname, join
from pathlib import Path

from flaky import flaky
import pytest
from autotest.conftest import requires_exe, requires_pkg

import flopy


@flaky
@requires_exe("mflgr")
@requires_pkg("pymake")
@pytest.mark.regression
def test_simplelgr(tmpdir, example_data_path):
    """Test load and write of distributed MODFLOW-LGR example problem."""
    import pymake

    mflgr_v2_ex3_path = example_data_path / "mflgr_v2" / "ex3"

    ws = tmpdir / mflgr_v2_ex3_path.stem
    shutil.copytree(mflgr_v2_ex3_path, ws)

    # load the lgr model
    lgr = flopy.modflowlgr.ModflowLgr.load(
        "ex3.lgr", verbose=True, model_ws=ws, exe_name="mflgr"
    )

    # get the namefiles of the parent and child
    namefiles = lgr.get_namefiles()
    msg = f"get_namefiles returned {len(namefiles)} items instead of 2"
    assert len(namefiles) == 2, msg

    tpth = dirname(namefiles[0])
    assert Path(tpth) == ws, f"dir path is {tpth} not {ws}"

    # run the lgr model
    success, buff = lgr.run_model()
    assert success, "could not run original modflow-lgr model"

    # check that a parent and child were read
    msg = "modflow-lgr ex3 does not have 2 grids"
    assert lgr.ngrids == 2, msg

    model_ws2 = join(ws, "new")
    lgr.change_model_ws(new_pth=model_ws2, reset_external=True)

    # get the namefiles of the parent and child
    namefiles = lgr.get_namefiles()
    assert (
        len(namefiles) == 2
    ), f"get_namefiles returned {len(namefiles)} items instead of 2"

    tpth = dirname(namefiles[0])
    assert tpth == model_ws2, f"dir path is {tpth} not {model_ws2}"

    # write the lgr model in to the new path
    lgr.write_input()

    # run the lgr model
    success, buff = lgr.run_model()
    assert success, "could not run new modflow-lgr model"

    # compare parent results
    print("compare parent results")
    pth0 = join(ws, "ex3_parent.nam")
    pth1 = join(model_ws2, "ex3_parent.nam")
    success = pymake.compare_heads(pth0, pth1)
    assert success, "parent heads do not match"

    # compare child results
    print("compare child results")
    pth0 = join(ws, "ex3_child.nam")
    pth1 = join(model_ws2, "ex3_child.nam")
    success = pymake.compare_heads(pth0, pth1)
    assert success, "child heads do not match"
