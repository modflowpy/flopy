# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   metadata:
#     section: flopy
#     authors:
#       - name: Jeremy White
# ---

# + [markdown] pycharm={"name": "#%% md\n"}
# # Formatting ASCII output arrays
#
# ### Configuring numeric arrays written by FloPy

# + [markdown] pycharm={"name": "#%% md\n"}
# load and run the Freyberg model


# + pycharm={"name": "#%%\n"}
import os
import sys
from pathlib import Path
from pprint import pformat
from tempfile import TemporaryDirectory

import git
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pooch

import flopy

# Set name of MODFLOW exe
#  assumes executable is in users path statement
version = "mf2005"
exe_name = "mf2005"
mfexe = exe_name

# Check if we are in the repository and define the data path.

try:
    root = Path(git.Repo(".", search_parent_directories=True).working_dir)
except:
    root = None

data_path = root / "examples" / "data" if root else Path.cwd()

sim_name = "freyberg"

file_names = {
    "freyberg.bas": "63266024019fef07306b8b639c6c67d5e4b22f73e42dcaa9db18b5e0f692c097",
    "freyberg.dis": "62d0163bf36c7ee9f7ee3683263e08a0abcdedf267beedce6dd181600380b0a2",
    "freyberg.githds": "abe92497b55e6f6c73306e81399209e1cada34cf794a7867d776cfd18303673b",
    "freyberg.gitlist": "aef02c664344a288264d5f21e08a748150e43bb721a16b0e3f423e6e3e293056",
    "freyberg.lpf": "06500bff979424f58e5e4fbd07a7bdeb0c78f31bd08640196044b6ccefa7a1fe",
    "freyberg.nam": "e66321007bb603ef55ed2ba41f4035ba6891da704a4cbd3967f0c66ef1532c8f",
    "freyberg.oc": "532905839ccbfce01184980c230b6305812610b537520bf5a4abbcd3bd703ef4",
    "freyberg.pcg": "0d1686fac4680219fffdb56909296c5031029974171e25d4304e70fa96ebfc38",
    "freyberg.rch": "37a1e113a7ec16b61417d1fa9710dd111a595de738a367bd34fd4a359c480906",
    "freyberg.riv": "7492a1d5eb23d6812ec7c8227d0ad4d1e1b35631a765c71182b71e3bd6a6d31d",
    "freyberg.wel": "00aa55f59797c02f0be5318a523b36b168fc6651f238f34e8b0938c04292d3e7",
}
for fname, fhash in file_names.items():
    pooch.retrieve(
        url=f"https://github.com/modflowpy/flopy/raw/develop/examples/data/{sim_name}/{fname}",
        fname=fname,
        path=data_path / sim_name,
        known_hash=fhash,
    )

# Set the paths
loadpth = data_path / sim_name
temp_dir = TemporaryDirectory()
modelpth = temp_dir.name

# make sure modelpth directory exists
if not os.path.isdir(modelpth):
    os.makedirs(modelpth)

print(sys.version)
print(f"numpy version: {np.__version__}")
print(f"matplotlib version: {mpl.__version__}")
print(f"flopy version: {flopy.__version__}")

# +
ml = flopy.modflow.Modflow.load(
    "freyberg.nam", model_ws=loadpth, exe_name=exe_name, version=version
)
ml.model_ws = modelpth
ml.write_input()
success, buff = ml.run_model(silent=True, report=True)
assert success, pformat(buff)

files = ["freyberg.hds", "freyberg.cbc"]
for f in files:
    if os.path.isfile(os.path.join(modelpth, f)):
        msg = f"Output file located: {f}"
        print(msg)
    else:
        errmsg = f"Error. Output file cannot be found: {f}"
        print(errmsg)

# + [markdown] pycharm={"name": "#%% md\n"}
# Each ``Util2d`` instance now has a ```.format``` attribute, which is an ```ArrayFormat``` instance:

# + pycharm={"name": "#%%\n"}
print(ml.lpf.hk[0].format)

# + [markdown] pycharm={"name": "#%% md\n"}
# The ```ArrayFormat``` class exposes each of the attributes seen in the ```ArrayFormat.___str___()``` call. ```ArrayFormat``` also exposes ``.fortran``, ``.py`` and ``.numpy`` atrributes, which are the respective format descriptors:

# + pycharm={"name": "#%%\n"}
print(ml.dis.botm[0].format.fortran)
print(ml.dis.botm[0].format.py)
print(ml.dis.botm[0].format.numpy)

# + [markdown] pycharm={"name": "#%% md\n"}
# #### (re)-setting ```.format```
#
# We can reset the format using a standard fortran type format descriptor

# + pycharm={"name": "#%%\n"}
ml.dis.botm[0].format.free = False
ml.dis.botm[0].format.fortran = "(20f10.4)"
print(ml.dis.botm[0].format.fortran)
print(ml.dis.botm[0].format.py)
print(ml.dis.botm[0].format.numpy)
print(ml.dis.botm[0].format)
# -

ml.write_input()
success, buff = ml.run_model(silent=True, report=True)
if success:
    for line in buff:
        print(line)
else:
    raise ValueError("Failed to run.")

# + [markdown] pycharm={"name": "#%% md\n"}
# Let's load the model we just wrote and check that the desired ```botm[0].format``` was used:

# + pycharm={"name": "#%%\n"}
ml1 = flopy.modflow.Modflow.load("freyberg.nam", model_ws=modelpth)
print(ml1.dis.botm[0].format)

# + [markdown] pycharm={"name": "#%% md\n"}
# We can also reset individual format components (we can also generate some warnings):

# + pycharm={"name": "#%%\n"}
ml.dis.botm[0].format.width = 9
ml.dis.botm[0].format.decimal = 1
print(ml1.dis.botm[0].format)

# + [markdown] pycharm={"name": "#%% md\n"}
# We can also select ``free`` format.  Note that setting to free format resets the format attributes to the default, max precision:

# + pycharm={"name": "#%%\n"}
ml.dis.botm[0].format.free = True
print(ml1.dis.botm[0].format)
# -

ml.write_input()
success, buff = ml.run_model(silent=True, report=True)
if success:
    for line in buff:
        print(line)
else:
    raise ValueError("Failed to run.")

# + pycharm={"name": "#%%\n"}
ml1 = flopy.modflow.Modflow.load("freyberg.nam", model_ws=modelpth)
print(ml1.dis.botm[0].format)
