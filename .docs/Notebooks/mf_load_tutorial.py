# ---
# jupyter:
#   jupytext:
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
#     section: mf2005
# ---

# # Loading MODFLOW-2005 models
#
# This tutorial demonstrates how to load models from disk.
#
# First import `flopy`.

# +
import os
import sys
from pathlib import Path

import git
import pooch

import flopy

print(sys.version)
print(f"flopy version: {flopy.__version__}")
# -

try:
    root = Path(git.Repo(".", search_parent_directories=True).working_dir)
except:
    root = None

data_path = root / "examples" / "data" if root else Path.cwd()


# ## The `load()` method
#
# To load a MODFLOW 2005 model, use the `Modflow.load()` method. The method's first argument is the path or name of the model namefile. Other parameters include:
#
# - `model_ws`: the model workspace
# - `verbose`: whether to write diagnostic information useful for troubleshooting
# - `check`: whether to check for model configuration errors

file_names = [
    "bcf2ss.ba6",
    "bcf2ss.bc6",
    "bcf2ss.dis",
    "bcf2ss.nam",
    "bcf2ss.oc",
    "bcf2ss.pcg",
    "bcf2ss.rch",
    "bcf2ss.riv",
    "bcf2ss.wel",
]
for fname in file_names:
    pooch.retrieve(
        url=f"https://github.com/modflowpy/flopy/raw/develop/examples/data/mf2005_test/{fname}",
        fname=fname,
        path=data_path / "mf2005_test",
        known_hash=None,
    )
model_ws = data_path / "mf2005_test"
ml = flopy.modflow.Modflow.load(
    "bcf2ss.nam",
    model_ws=model_ws,
    verbose=True,
    version="mf2005",
    check=False,
)

# ## Auxiliary variables
#
# Below we load a model containig auxiliary variables, then access them.

file_names = [
    "EXAMPLE.BA6",
    "EXAMPLE.BUD",
    "EXAMPLE.DIS",
    "EXAMPLE.DIS.metadata",
    "EXAMPLE.HED",
    "EXAMPLE.LPF",
    "EXAMPLE.LST",
    "EXAMPLE.MPBAS",
    "EXAMPLE.OC",
    "EXAMPLE.PCG",
    "EXAMPLE.RCH",
    "EXAMPLE.RIV",
    "EXAMPLE.WEL",
    "EXAMPLE.mpnam",
    "EXAMPLE.nam",
]
for fname in file_names:
    pooch.retrieve(
        url=f"https://github.com/modflowpy/flopy/raw/develop/examples/data/mp6/{fname}",
        fname=fname,
        path=data_path / "mp6",
        known_hash=None,
    )
model_ws = data_path / "mp6"
ml = flopy.modflow.Modflow.load(
    "EXAMPLE.nam",
    model_ws=model_ws,
    verbose=False,
    version="mf2005",
    check=False,
)

# Auxiliary IFACE data are in the river package. Retrieve it from the model object.

riv = ml.riv.stress_period_data[0]

# Confirm that the `iface` auxiliary data have been read by looking at the `dtype`.

riv.dtype

# The `iface` data is accessible from the recarray.

riv["iface"]
