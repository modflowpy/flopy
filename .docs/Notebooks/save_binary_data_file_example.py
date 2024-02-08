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
#     display_name: Python 3
#     language: python
#     name: python3
#   metadata:
#     section: flopy
#     authors:
#       - name: Joseph Hughes
# ---

# # Saving array data to MODFLOW-style binary files
#

# +
import os
import shutil
import sys
from tempfile import TemporaryDirectory

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import flopy

print(sys.version)
print(f"numpy version: {np.__version__}")
print(f"matplotlib version: {mpl.__version__}")
print(f"flopy version: {flopy.__version__}")

# +
nlay, nrow, ncol = 1, 20, 10

# temporary directory
temp_dir = TemporaryDirectory()
model_ws = os.path.join(temp_dir.name, "binary_data")

if os.path.exists(model_ws):
    shutil.rmtree(model_ws)

precision = "single"  # or 'double'
dtype = np.float32  # or np.float64

mf = flopy.modflow.Modflow(model_ws=model_ws)
dis = flopy.modflow.ModflowDis(
    mf, nlay=nlay, nrow=nrow, ncol=ncol, delr=20, delc=10
)
# -

# Create a linear data array

# +
# create the first row of data
b = np.linspace(10, 1, num=ncol, dtype=dtype).reshape(1, ncol)

# extend data to every row
b = np.repeat(b, nrow, axis=0)

# print the shape and type of the data
print(b.shape)
# -

# Plot the data array

pmv = flopy.plot.PlotMapView(model=mf)
v = pmv.plot_array(b)
pmv.plot_grid()
plt.colorbar(v, shrink=0.5)

# Write the linear data array to a binary file

# +
text = "head"

# write a binary data file
pertim = dtype(1.0)
header = flopy.utils.BinaryHeader.create(
    bintype=text,
    precision=precision,
    text=text,
    nrow=nrow,
    ncol=ncol,
    ilay=1,
    pertim=pertim,
    totim=pertim,
    kstp=1,
    kper=1,
)
pth = os.path.join(model_ws, "bottom.bin")
flopy.utils.Util2d.write_bin(b.shape, pth, b, header_data=header)
# -

# Read the binary data file

bo = flopy.utils.HeadFile(pth, precision=precision)
br = bo.get_data(idx=0)

# Plot the data that was read from the binary file

pmv = flopy.plot.PlotMapView(model=mf)
v = pmv.plot_array(br)
pmv.plot_grid()
plt.colorbar(v, shrink=0.5)

# Plot the difference in the two values

pmv = flopy.plot.PlotMapView(model=mf)
v = pmv.plot_array(b - br)
pmv.plot_grid()
plt.colorbar(v, shrink=0.5)

try:
    # ignore PermissionError on Windows
    temp_dir.cleanup()
except:
    pass
