"""
Some basic tests for STR load.
"""

import pytest

import os
import flopy
import numpy as np

path = os.path.join("..", "examples", "data", "mf2005_test")
pthgw = os.path.join("..", "examples", "groundwater_paper", "uspb", "flopy")
cpth = os.path.join("temp", "t014")
# make the directory if it does not exist
if not os.path.isdir(cpth):
    os.makedirs(cpth)

mf_items = ["str.nam", "DG.nam"]
pths = [path, pthgw]

exe_name = "mf2005dbl"
v = flopy.which(exe_name)

run = True
if v is None:
    run = False


def load_str(mfnam, pth):
    m = flopy.modflow.Modflow.load(
        mfnam,
        exe_name=exe_name,
        forgive=False,
        model_ws=pth,
        verbose=True,
        check=False,
    )
    assert m.load_fail is False

    # rewrite files
    ws = os.path.join(cpth, mfnam.replace(".nam", ""))
    m.model_ws = ws
    m.write_input()

    # attempt to run the model
    if run:
        try:
            success, buff = m.run_model(silent=False)
        except:
            pass
        assert success, "base model run did not terminate successfully"

    # load files
    pth = os.path.join(ws, f"{m.name}.str")
    str2 = flopy.modflow.ModflowStr.load(pth, m)
    for name in str2.dtype.names:
        assert (
            np.array_equal(
                str2.stress_period_data[0][name],
                m.str.stress_period_data[0][name],
            )
            is True
        )
    for name in str2.dtype2.names:
        assert (
            np.array_equal(
                str2.segment_data[0][name], m.str.segment_data[0][name]
            )
            is True
        )

    return


@pytest.mark.parametrize(
    "namfile, pth",
    zip(mf_items, pths),
)
def test_mf2005load(namfile, pth):
    load_str(namfile, pth)
    return


if __name__ == "__main__":
    for namfile, pth in zip(mf_items, pths):
        load_str(namfile, pth)
