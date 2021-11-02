"""
Some basic tests for SWR2 load.
"""

import pytest
import os
import flopy
import pymake
from ci_framework import base_test_dir, FlopyTestSetup

base_dir = base_test_dir(__file__, rel_path="temp", verbose=True)

swi_path = os.path.join("..", "examples", "data", "mf2005_test")
cpth = os.path.join("temp", "t037")


mf_items = ["swiex1.nam", "swiex2_strat.nam", "swiex3.nam"]

exe_name = "mf2005"
v = flopy.which(exe_name)

run = True
if v is None:
    run = False


def load_swi(mfnam):
    name = mfnam.replace(".nam", "")
    model_ws = f"{base_dir}_{name}"
    test_setup = FlopyTestSetup(verbose=True, test_dirs=model_ws)

    pymake.setup(os.path.join(swi_path, mfnam), model_ws)

    m = flopy.modflow.Modflow.load(
        mfnam, model_ws=model_ws, verbose=True, exe_name=exe_name
    )
    assert m.load_fail is False

    if run:
        try:
            success, buff = m.run_model(silent=False)
        except:
            success = False
        assert success, "base model run did not terminate successfully"
        fn0 = os.path.join(model_ws, mfnam)

    # write free format files -
    # won't run without resetting to free format - evt external file issue
    m.free_format_input = True

    # rewrite files
    model_ws2 = os.path.join(model_ws, "flopy")
    m.change_model_ws(
        model_ws2, reset_external=True
    )  # l1b2k_bath wont run without this
    m.write_input()
    if run:
        try:
            success, buff = m.run_model(silent=False)
        except:
            success = False
        assert success, "base model run did not terminate successfully"
        fn1 = os.path.join(model_ws2, mfnam)

    if run:
        fsum = os.path.join(
            model_ws, f"{os.path.splitext(mfnam)[0]}.budget.out"
        )
        try:
            success = pymake.compare_budget(
                fn0, fn1, max_incpd=0.1, max_cumpd=0.1, outfile=fsum
            )
        except:
            success = False
            print("could not perform budget comparison")

        assert success, "budget comparison failure"

    return


@pytest.mark.parametrize(
    "namfile",
    mf_items,
)
def test_mf2005swi2load(namfile):
    load_swi(namfile)
    return


if __name__ == "__main__":
    for namfile in mf_items:
        load_swi(namfile)
