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

# # Error checking for MODFLOW-2005 models

# +
import os
import sys
from tempfile import TemporaryDirectory

import flopy

print(sys.version)
print(f"flopy version: {flopy.__version__}")
# -

# #### Set the working directory

path = os.path.join("..", "..", "examples", "data", "mf2005_test")

# #### Load example dataset and change the model work space

m = flopy.modflow.Modflow.load("test1ss.nam", model_ws=path)
temp_dir = TemporaryDirectory()
workspace = temp_dir.name
m.change_model_ws(workspace)

# By default, the checker performs a model-level check when a set of model files are loaded, unless load is called with `check=False`. The load check only produces screen output if load is called with `verbose=True`. Checks are also performed at the package level when an individual package is loaded
#
# #### The `check()` method
# Each model and each package has a `check()` method. The check method has three arguments:

help(m.check)

# #### The check class
#
# By default, check is called at the model level without a summary output file, but with `verbose=True` and `level=1`. The check methods return an instance of the **check** class, which is housed with the flopy utilities.

chk = m.check()

# #### Summary array
# Most of the attributes and methods in **check** are intended to be used by the ``check()`` methods. The central attribute of **check** is the summary array:

chk.summary_array

# This is a numpy record array that summarizes errors and warnings found by the checker. The package, layer-row-column location of the error, the offending value, and a description of the error are provided. In the checker, errors and warnings are loosely defined as follows:
# #### Errors:
#
# either input that would cause MODFLOW to crash, or inputs that almost certainly mis-represent the intended conceptual model.
#
# #### Warnings:
#
# inputs that are potentially problematic, but may be intentional.
#
# each package-level check produces a **check** instance with a summary array. The model level checks combine the summary arrays from the packages into a master summary array. At the model and the package levels, the summary array is used to generate the screen output shown above. At either level, the summary array can be written to a csv file by supply a filename to the `f` argument. Specifying `level=2` prints the summary array to the screen.

m.check(level=2)

# #### example of package level check and summary file

m.rch.check()

# #### example of summary output file

m.check(f=os.path.join(workspace, "checksummary.csv"))

try:
    import pandas as pd

    summary_pth = os.path.join(workspace, "checksummary.csv")
    df = pd.read_csv(summary_pth)
except:
    df = open(summary_pth).readlines()
df

# #### checking on `write_input()`
# checking is also performed by default when `write_input()` is called at the package or model level. Checking on write is performed with the same `verbose` setting as specified for the model. However, if errors or warnings are encountered and `level=1` (default) or higher, a screen message notifies the user of the errors.
#
# By default, the checks performed on `load()` and `write_input()` save results to a summary file, which is named after the packge or the model.

m.write_input()
