"""
Try to load all of the MODFLOW-USG examples in ../examples/data/mfusg_test.
These are the examples that are distributed with MODFLOW-USG.
"""

import pytest
import os
import flopy

# make the working directory
tpth = os.path.join("temp", "t038")
if not os.path.isdir(tpth):
    os.makedirs(tpth, exist_ok=True)

# build list of name files to try and load
usgpth = os.path.join("..", "examples", "data", "mfusg_test")
usg_files = []
for path, subdirs, files in os.walk(usgpth):
    for name in files:
        if name.endswith(".nam"):
            usg_files.append(os.path.join(path, name))

#
@pytest.mark.parametrize(
    "fpth",
    usg_files,
)
def test_load_usg(fpth):
    exdir, namfile = os.path.split(fpth)
    load_model(namfile, exdir)


# function to load a MODFLOW-USG model and then write it back out
def load_model(namfile, model_ws):
    m = flopy.mfusg.MfUsg.load(
        namfile, model_ws=model_ws, verbose=True, check=False
    )
    assert m, f"Could not load namefile {namfile}"
    assert m.load_fail is False
    m.change_model_ws(tpth)
    m.write_input()
    return


if __name__ == "__main__":
    for fusg in usg_files:
        d, f = os.path.split(fusg)
        load_model(f, d)
