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
#     section: mf2005
#     authors:
#       - name: Andy Leaf
# ---

# # SFR package Prudic and others (2004) example
# Demonstrates functionality of Flopy SFR module using the example documented by [Prudic and others (2004)](https://doi.org/10.3133/ofr20041042):
#
# #### Problem description:
#
# * Grid dimensions: 1 Layer, 15 Rows, 10 Columns
# * Stress periods: 1 steady
# * Flow package: LPF
# * Stress packages: SFR, GHB, EVT, RCH
# * Solver: SIP
#
# <img src="./img/Prudic2004_fig6.png" width="400" height="500"/>

import glob
import os
import shutil

# +
import sys
from tempfile import TemporaryDirectory

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import flopy
import flopy.utils.binaryfile as bf
from flopy.utils.sfroutputfile import SfrFile

mpl.rcParams["figure.figsize"] = (11, 8.5)

print(sys.version)
print(f"numpy version: {np.__version__}")
print(f"pandas version: {pd.__version__}")
print(f"matplotlib version: {mpl.__version__}")
print(f"flopy version: {flopy.__version__}")
# -

# Set name of MODFLOW exe
#  assumes executable is in users path statement
exe_name = "mf2005"

# # #### copy over the example files to the working directory

# +
# temporary directory
temp_dir = TemporaryDirectory()
path = temp_dir.name

gpth = os.path.join("..", "..", "examples", "data", "mf2005_test", "test1ss.*")
for f in glob.glob(gpth):
    shutil.copy(f, path)
gpth = os.path.join("..", "..", "examples", "data", "mf2005_test", "test1tr.*")
for f in glob.glob(gpth):
    shutil.copy(f, path)
# -

# ### Load example dataset, skipping the SFR package

m = flopy.modflow.Modflow.load(
    "test1ss.nam",
    version="mf2005",
    exe_name=exe_name,
    model_ws=path,
    load_only=["ghb", "evt", "rch", "dis", "bas6", "oc", "sip", "lpf"],
)

oc = m.oc
oc.stress_period_data

# ### Read pre-prepared reach and segment data into numpy recarrays using numpy.genfromtxt()
# Reach data (Item 2 in the SFR input instructions), are input and stored in a numpy record array
# https://numpy.org/doc/stable/reference/generated/numpy.recarray.html
# This allows for reach data to be indexed by their variable names, as described in the SFR input instructions.
#
# For more information on Item 2, see the Online Guide to MODFLOW:
# <https://water.usgs.gov/nrp/gwsoftware/modflow2000/MFDOC/sfr.html>

rpth = os.path.join(
    "..", "..", "examples", "data", "sfr_examples", "test1ss_reach_data.csv"
)
reach_data = np.genfromtxt(rpth, delimiter=",", names=True)
reach_data

# ### Segment Data structure
# Segment data are input and stored in a dictionary of record arrays, which

spth = os.path.join(
    "..", "..", "examples", "data", "sfr_examples", "test1ss_segment_data.csv"
)
ss_segment_data = np.genfromtxt(spth, delimiter=",", names=True)
segment_data = {0: ss_segment_data}
segment_data[0][0:1]["width1"]

# ### define dataset 6e (channel flow data) for segment 1
# dataset 6e is stored in a nested dictionary keyed by stress period and segment,
# with a list of the following lists defined for each segment with icalc == 4
# FLOWTAB(1) FLOWTAB(2) ... FLOWTAB(NSTRPTS)
# DPTHTAB(1) DPTHTAB(2) ... DPTHTAB(NSTRPTS)
# WDTHTAB(1) WDTHTAB(2) ... WDTHTAB(NSTRPTS)

channel_flow_data = {
    0: {
        1: [
            [0.5, 1.0, 2.0, 4.0, 7.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0],
            [0.25, 0.4, 0.55, 0.7, 0.8, 0.9, 1.1, 1.25, 1.4, 1.7, 2.6],
            [3.0, 3.5, 4.2, 5.3, 7.0, 8.5, 12.0, 14.0, 17.0, 20.0, 22.0],
        ]
    }
}

# ### define dataset 6d (channel geometry data) for segments 7 and 8
# dataset 6d is stored in a nested dictionary keyed by stress period and segment,
# with a list of the following lists defined for each segment with icalc == 4
# FLOWTAB(1) FLOWTAB(2) ... FLOWTAB(NSTRPTS)
# DPTHTAB(1) DPTHTAB(2) ... DPTHTAB(NSTRPTS)
# WDTHTAB(1) WDTHTAB(2) ... WDTHTAB(NSTRPTS)

channel_geometry_data = {
    0: {
        7: [
            [0.0, 10.0, 80.0, 100.0, 150.0, 170.0, 240.0, 250.0],
            [20.0, 13.0, 10.0, 2.0, 0.0, 10.0, 13.0, 20.0],
        ],
        8: [
            [0.0, 10.0, 80.0, 100.0, 150.0, 170.0, 240.0, 250.0],
            [25.0, 17.0, 13.0, 4.0, 0.0, 10.0, 16.0, 20.0],
        ],
    }
}

# ### Define SFR package variables

nstrm = len(reach_data)  # number of reaches
nss = len(segment_data[0])  # number of segments
nsfrpar = 0  # number of parameters (not supported)
nparseg = 0
const = 1.486  # constant for manning's equation, units of cfs
dleak = 0.0001  # closure tolerance for stream stage computation
ipakcb = 53  # flag for writing SFR output to cell-by-cell budget (on unit 53)
istcb2 = 81  # flag for writing SFR output to text file
dataset_5 = {0: [nss, 0, 0]}  # dataset 5 (see online guide)

# ### Instantiate SFR package
# Input arguments generally follow the variable names defined in the Online Guide to MODFLOW

sfr = flopy.modflow.ModflowSfr2(
    m,
    nstrm=nstrm,
    nss=nss,
    const=const,
    dleak=dleak,
    ipakcb=ipakcb,
    istcb2=istcb2,
    reach_data=reach_data,
    segment_data=segment_data,
    channel_geometry_data=channel_geometry_data,
    channel_flow_data=channel_flow_data,
    dataset_5=dataset_5,
    unit_number=15,
)

sfr.reach_data[0:1]

# ### Plot the SFR segments
# any column in the reach_data array can be plotted using the ```key``` argument

sfr.plot(key="iseg")

# ### Check the SFR dataset for errors

chk = sfr.check()

m.external_fnames = [os.path.split(f)[1] for f in m.external_fnames]
m.external_fnames

m.write_input()

success, buff = m.run_model(silent=True, report=True)
if success:
    for line in buff:
        print(line)
else:
    raise ValueError("Failed to run.")

# ### Load SFR formated water balance output into pandas dataframe using the `SfrFile` class

sfr_outfile = os.path.join(
    "..", "..", "examples", "data", "sfr_examples", "test1ss.flw"
)
sfrout = SfrFile(sfr_outfile)
df = sfrout.get_dataframe()
df.head()

# #### Plot streamflow and stream/aquifer interactions for a segment

inds = df.segment == 3
print(df.reach[inds].astype(str))
# ax = df.ix[inds, ['Qin', 'Qaquifer', 'Qout']].plot(x=df.reach[inds])
ax = df.loc[inds, ["reach", "Qin", "Qaquifer", "Qout"]].plot(x="reach")
ax.set_ylabel("Flow, in cubic feet per second")
ax.set_xlabel("SFR reach")

# ### Look at stage, model top, and streambed top

streambed_top = m.sfr.segment_data[0][m.sfr.segment_data[0].nseg == 3][
    ["elevup", "elevdn"]
][0]
streambed_top

df["model_top"] = m.dis.top.array[df.row.values - 1, df.column.values - 1]
fig, ax = plt.subplots()
plt.plot([1, 6], list(streambed_top), label="streambed top")
# ax = df.loc[inds, ['stage', 'model_top']].plot(ax=ax, x=df.reach[inds])
ax = df.loc[inds, ["reach", "stage", "model_top"]].plot(ax=ax, x="reach")
ax.set_ylabel("Elevation, in feet")
plt.legend()

# ### Get SFR leakage results from cell budget file

bpth = os.path.join(path, "test1ss.cbc")
cbbobj = bf.CellBudgetFile(bpth)
cbbobj.list_records()

sfrleak = cbbobj.get_data(text="  STREAM LEAKAGE")[0]
sfrleak[sfrleak == 0] = np.nan  # remove zero values

# ### Plot leakage in plan view

im = plt.imshow(
    sfrleak[0], interpolation="none", cmap="coolwarm", vmin=-3, vmax=3
)
cb = plt.colorbar(im, label="SFR Leakage, in cubic feet per second")

# ### Plot total streamflow

sfrQ = sfrleak[0].copy()
sfrQ[sfrQ == 0] = np.nan
sfrQ[df.row.values - 1, df.column.values - 1] = (
    df[["Qin", "Qout"]].mean(axis=1).values
)
im = plt.imshow(sfrQ, interpolation="none")
plt.colorbar(im, label="Streamflow, in cubic feet per second")

# ## Reading transient SFR formatted output
#
# The `SfrFile` class handles this the same way
#
# Files for the transient version of the above example were already copied to the `data` folder in the third cell above.
# First run the transient model to get the output:
# ```
# >mf2005 test1tr.nam
# ```

flopy.run_model(exe_name, "test1tr.nam", model_ws=path, silent=True)

sfrout_tr = SfrFile(os.path.join(path, "test1tr.flw"))
dftr = sfrout_tr.get_dataframe()
dftr.head()

# ### plot a hydrograph
# plot `Qout` (simulated streamflow) and `Qaquifer` (simulated stream leakage) through time

fig, axes = plt.subplots(2, 1, sharex=True)
dftr8 = dftr.loc[(dftr.segment == 8) & (dftr.reach == 5)]
dftr8.Qout.plot(ax=axes[0])
axes[0].set_ylabel("Simulated streamflow, cfs")
dftr8.Qaquifer.plot(ax=axes[1])
axes[1].set_ylabel("Leakage to aquifer, cfs")

try:
    # ignore PermissionError on Windows
    temp_dir.cleanup()
except:
    pass
