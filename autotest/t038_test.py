"""
Try to load all of the MODFLOW-USG examples in ../examples/data/mfusg_test.
These are the examples that are distributed with MODFLOW-USG.
"""

import os
import flopy

# make the working directory
tpth = os.path.join("temp", "t038")
if not os.path.isdir(tpth):
    os.makedirs(tpth)

# build list of name files to try and load
usgpth = os.path.join("..", "examples", "data", "mfusg_test")
usg_files = []
for path, subdirs, files in os.walk(usgpth):
    for name in files:
        if name.endswith(".nam"):
            usg_files.append(os.path.join(path, name))

#
def test_load_usg():
    for fusg in usg_files:
        d, f = os.path.split(fusg)
        yield load_model, f, d


# function to load a MODFLOW-USG model and then write it back out
def load_model(namfile, model_ws):
    m = flopy.modflow.Modflow.load(
        namfile, model_ws=model_ws, version="mfusg", verbose=True, check=False
    )
    assert m, "Could not load namefile {}".format(namfile)
    assert m.load_fail is False
    m.change_model_ws(tpth)
    m.write_input()
    return


if __name__ == "__main__":
    for fusg in usg_files:
        d, f = os.path.split(fusg)
        load_model(f, d)
