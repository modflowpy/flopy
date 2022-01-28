"""
Some basic tests for LAKE load.
"""

import os

import pymake
import pytest
from ci_framework import FlopyTestSetup, base_test_dir

import flopy

base_dir = base_test_dir(__file__, rel_path="temp", verbose=True)

load_path = os.path.join("..", "examples", "data", "mf2005_test")
mf_items = [
    "l1b2k_bath.nam",
    "l2a_2k.nam",
    "lakeex3.nam",
    "l1b2k.nam",
    "l1a2k.nam",
]

exe_name = "mf2005"
v = flopy.which(exe_name)

run = True
if v is None:
    run = False


def load_lak(mfnam, model_ws, run):

    compth = model_ws
    pymake.setup(os.path.join(load_path, mfnam), model_ws)

    m = flopy.modflow.Modflow.load(
        mfnam,
        model_ws=model_ws,
        verbose=True,
        forgive=False,
        exe_name=exe_name,
    )
    assert m.load_fail is False

    if run:
        try:
            success, buff = m.run_model(silent=True)
        except:
            print(f"could not run base model {os.path.splitext(mfnam)[0]}")
            pass
        msg = (
            f"base model {os.path.splitext(mfnam)[0]} "
            "run did not terminate successfully"
        )
        assert success, msg
        msg = (
            f"base model {os.path.splitext(mfnam)[0]} "
            "run terminated successfully"
        )
        print(msg)
        fn0 = os.path.join(model_ws, mfnam)

    # write free format files - wont run without resetting to free format - evt external file issue
    m.free_format_input = True

    # rewrite files
    model_ws2 = os.path.join(model_ws, "external")
    m.change_model_ws(
        model_ws2, reset_external=True
    )  # l1b2k_bath wont run without this
    m.write_input()

    if run:
        try:
            success, buff = m.run_model(silent=False)
        except:
            print(f"could not run new model {os.path.splitext(mfnam)[0]}")
            pass
        msg = (
            f"new model {os.path.splitext(mfnam)[0]} "
            "run did not terminate successfully"
        )
        assert success, msg
        msg = (
            f"new model {os.path.splitext(mfnam)[0]} "
            "run terminated successfully"
        )
        print(msg)
        fn1 = os.path.join(model_ws2, mfnam)

        fsum = os.path.join(compth, f"{os.path.splitext(mfnam)[0]}.budget.out")

    if run:
        try:
            success = pymake.compare_budget(
                fn0, fn1, max_incpd=0.1, max_cumpd=0.1, outfile=fsum
            )
        except:
            print("could not perform budget comparison")

        assert success, "budget comparison failure"

    return


@pytest.mark.parametrize(
    "namfile",
    mf_items,
)
def test_mf2005load(namfile):
    dirPath = namfile.replace(".nam", "")
    model_ws = f"{base_dir}_{dirPath}"
    test_setup = FlopyTestSetup(verbose=True, test_dirs=model_ws)
    test_setup.add_test_dir(os.path.join(model_ws, "external"))

    load_lak(namfile, model_ws, run)

    return


if __name__ == "__main__":
    for namfile in mf_items:
        test_mf2005load(namfile)
