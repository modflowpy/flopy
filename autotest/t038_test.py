"""
Try to load all of the MODFLOW-USG examples in ../examples/data/mfusg_test.
These are the examples that are distributed with MODFLOW-USG.
"""

import pytest
import os
import flopy
from ci_framework import base_test_dir, FlopyTestSetup

base_dir = base_test_dir(__file__, rel_path="temp", verbose=True)

# build list of name files to try and load
usgpth = os.path.join("..", "examples", "data", "mfusg_test")
usg_files = []
for path, subdirs, files in os.walk(usgpth):
    for name in files:
        if name.endswith(".nam"):
            usg_files.append(os.path.join(path, name))


@pytest.mark.parametrize(
    "fpth",
    usg_files,
)
def test_load_usg(fpth):
    exdir, namfile = os.path.split(fpth)
    name = namfile.replace(".nam", "")
    model_ws = f"{base_dir}_test_load_usg_{name}"
    test_setup = FlopyTestSetup(verbose=True, test_dirs=model_ws)

    load_model(namfile, exdir, model_ws)


# function to load a MODFLOW-USG model and then write it back out
def load_model(namfile, load_ws, model_ws):
    m = flopy.mfusg.MfUsg.load(
        namfile,
        model_ws=load_ws,
        verbose=True,
        check=False,
    )
    assert m, f"Could not load namefile {namfile}"
    assert m.load_fail is False

    m.change_model_ws(model_ws)
    m.write_input()

    return


if __name__ == "__main__":
    for fusg in usg_files:
        test_load_usg(fusg)
