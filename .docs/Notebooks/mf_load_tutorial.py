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

import flopy

print(sys.version)
print(f"flopy version: {flopy.__version__}")
# -

# ## The `load()` method
#
# To load a MODFLOW 2005 model, use the `Modflow.load()` method. The method's first argument is the path or name of the model namefile. Other parameters include:
#
# - `model_ws`: the model workspace
# - `verbose`: whether to write diagnostic information useful for troubleshooting
# - `check`: whether to check for model configuration errors

model_ws = os.path.join("..", "..", "examples", "data", "mf2005_test")
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

model_ws = os.path.join("..", "..", "examples", "data", "mp6")
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
