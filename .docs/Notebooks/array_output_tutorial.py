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
from tempfile import TemporaryDirectory

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import flopy

# Set name of MODFLOW exe
#  assumes executable is in users path statement
version = "mf2005"
exe_name = "mf2005"
mfexe = exe_name

# Set the paths
loadpth = os.path.join("..", "..", "examples", "data", "freyberg")
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
if success:
    for line in buff:
        print(line)
else:
    raise ValueError("Failed to run.")

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
