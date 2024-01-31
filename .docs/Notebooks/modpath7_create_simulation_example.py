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
#     section: modpath
#     authors:
#       - name: Joseph Hughes
# ---

# # Creating a MODPATH 7 simulation
#
# This notebook demonstrates how to create a simple forward and backward MODPATH 7 simulation using the `.create_mp7()` method. The notebooks also shows how to create subsets of endpoint output and plot MODPATH results on ModelMap objects.

import os

# +
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

# temporary directory
temp_dir = TemporaryDirectory()
model_ws = temp_dir.name
# -

# define executable names
mpexe = "mp7"
mfexe = "mf6"

# ### Flow model data

nper, nstp, perlen, tsmult = 1, 1, 1.0, 1.0
nlay, nrow, ncol = 3, 21, 20
delr = delc = 500.0
top = 400.0
botm = [220.0, 200.0, 0.0]
laytyp = [1, 0, 0]
kh = [50.0, 0.01, 200.0]
kv = [10.0, 0.01, 20.0]
wel_loc = (2, 10, 9)
wel_q = -150000.0
rch = 0.005
riv_h = 320.0
riv_z = 317.0
riv_c = 1.0e5


def get_nodes(locs):
    nodes = []
    for k, i, j in locs:
        nodes.append(k * nrow * ncol + i * ncol + j)
    return nodes


# ### MODPATH 7 using MODFLOW 6
#
# #### Create and run MODFLOW 6

# +
ws = os.path.join(model_ws, "mp7_ex1_cs")
nm = "ex01_mf6"

# Create the Flopy simulation object
sim = flopy.mf6.MFSimulation(
    sim_name=nm, exe_name=mfexe, version="mf6", sim_ws=ws
)

# Create the Flopy temporal discretization object
pd = (perlen, nstp, tsmult)
tdis = flopy.mf6.modflow.mftdis.ModflowTdis(
    sim, pname="tdis", time_units="DAYS", nper=nper, perioddata=[pd]
)

# Create the Flopy groundwater flow (gwf) model object
model_nam_file = f"{nm}.nam"
gwf = flopy.mf6.ModflowGwf(
    sim, modelname=nm, model_nam_file=model_nam_file, save_flows=True
)

# Create the Flopy iterative model solver (ims) Package object
ims = flopy.mf6.modflow.mfims.ModflowIms(
    sim,
    pname="ims",
    complexity="SIMPLE",
    outer_hclose=1e-6,
    inner_hclose=1e-6,
    rcloserecord=1e-6,
)

# create gwf file
dis = flopy.mf6.modflow.mfgwfdis.ModflowGwfdis(
    gwf,
    pname="dis",
    nlay=nlay,
    nrow=nrow,
    ncol=ncol,
    length_units="FEET",
    delr=delr,
    delc=delc,
    top=top,
    botm=botm,
)
# Create the initial conditions package
ic = flopy.mf6.modflow.mfgwfic.ModflowGwfic(gwf, pname="ic", strt=top)

# Create the node property flow package
npf = flopy.mf6.modflow.mfgwfnpf.ModflowGwfnpf(
    gwf, pname="npf", icelltype=laytyp, k=kh, k33=kv
)


# recharge
flopy.mf6.modflow.mfgwfrcha.ModflowGwfrcha(gwf, recharge=rch)
# wel
wd = [(wel_loc, wel_q)]
flopy.mf6.modflow.mfgwfwel.ModflowGwfwel(
    gwf, maxbound=1, stress_period_data={0: wd}
)
# river
rd = []
for i in range(nrow):
    rd.append([(0, i, ncol - 1), riv_h, riv_c, riv_z])
flopy.mf6.modflow.mfgwfriv.ModflowGwfriv(gwf, stress_period_data={0: rd})
# Create the output control package
headfile = f"{nm}.hds"
head_record = [headfile]
budgetfile = f"{nm}.cbb"
budget_record = [budgetfile]
saverecord = [("HEAD", "ALL"), ("BUDGET", "ALL")]
oc = flopy.mf6.modflow.mfgwfoc.ModflowGwfoc(
    gwf,
    pname="oc",
    saverecord=saverecord,
    head_filerecord=head_record,
    budget_filerecord=budget_record,
)

# Write the datasets
sim.write_simulation()
# Run the simulation
success, buff = sim.run_simulation(silent=True, report=True)
assert success, "mf6 model did not run"
for line in buff:
    print(line)
# -

# Get locations to extract data

nodew = get_nodes([wel_loc])
cellids = gwf.riv.stress_period_data.get_data()[0]["cellid"]
nodesr = get_nodes(cellids)

# #### Create and run MODPATH 7
#
# Forward tracking

# +
# create modpath files
mpnamf = f"{nm}_mp_forward"

# create basic forward tracking modpath simulation
mp = flopy.modpath.Modpath7.create_mp7(
    modelname=mpnamf,
    trackdir="forward",
    flowmodel=gwf,
    model_ws=ws,
    rowcelldivisions=1,
    columncelldivisions=1,
    layercelldivisions=1,
    exe_name=mpexe,
)

# write modpath datasets
mp.write_input()

# run modpath
succes, buff = mp.run_model(silent=True, report=True)
assert success, "mp7 forward tracking failed to run"
for line in buff:
    print(line)
# -

# Backward tracking from well and river locations

# +
# create modpath files
mpnamb = f"{nm}_mp_backward"

# create basic forward tracking modpath simulation
mp = flopy.modpath.Modpath7.create_mp7(
    modelname=mpnamb,
    trackdir="backward",
    flowmodel=gwf,
    model_ws=ws,
    rowcelldivisions=5,
    columncelldivisions=5,
    layercelldivisions=5,
    nodes=nodew + nodesr,
    exe_name=mpexe,
)

# write modpath datasets
mp.write_input()

# run modpath
success, buff = mp.run_model(silent=True, report=True)
assert success, "mp7 backward tracking failed to run"
for line in buff:
    print(line)
# -

# #### Load and Plot MODPATH 7 output
#
# ##### Forward Tracking

# Load forward tracking pathline data

fpth = os.path.join(ws, f"{mpnamf}.mppth")
p = flopy.utils.PathlineFile(fpth)
pw = p.get_destination_pathline_data(dest_cells=nodew)
pr = p.get_destination_pathline_data(dest_cells=nodesr)

# Load forward tracking endpoint data

fpth = os.path.join(ws, f"{mpnamf}.mpend")
e = flopy.utils.EndpointFile(fpth)

# Get forward particles that terminate in the well

well_epd = e.get_destination_endpoint_data(dest_cells=nodew)

# Get particles that terminate in the river boundaries

riv_epd = e.get_destination_endpoint_data(dest_cells=nodesr)

# Well and river forward tracking pathlines

colors = ["green", "orange", "red"]

# +
f, axes = plt.subplots(
    ncols=3, nrows=2, sharey=True, sharex=True, figsize=(15, 10)
)
axes = axes.flatten()

idax = 0
for k in range(nlay):
    ax = axes[idax]
    ax.set_aspect("equal")
    ax.set_title(f"Well pathlines - Layer {k + 1}")
    mm = flopy.plot.PlotMapView(model=gwf, ax=ax)
    mm.plot_grid(lw=0.5)
    mm.plot_pathline(pw, layer=k, colors=colors[k], lw=0.75)
    idax += 1

for k in range(nlay):
    ax = axes[idax]
    ax.set_aspect("equal")
    ax.set_title(f"River pathlines - Layer {k + 1}")
    mm = flopy.plot.PlotMapView(model=gwf, ax=ax)
    mm.plot_grid(lw=0.5)
    mm.plot_pathline(pr, layer=k, colors=colors[k], lw=0.75)
    idax += 1

plt.tight_layout()
# -

# Forward tracking endpoints captured by the well and river

# +
f, axes = plt.subplots(ncols=2, nrows=1, sharey=True, figsize=(10, 5))
axes = axes.flatten()

ax = axes[0]
ax.set_aspect("equal")
ax.set_title("Well recharge area")
mm = flopy.plot.PlotMapView(model=gwf, ax=ax)
mm.plot_grid(lw=0.5)
mm.plot_endpoint(well_epd, direction="starting", colorbar=True, shrink=0.5)

ax = axes[1]
ax.set_aspect("equal")
ax.set_title("River recharge area")
mm = flopy.plot.PlotMapView(model=gwf, ax=ax)
mm.plot_grid(lw=0.5)
mm.plot_endpoint(riv_epd, direction="starting", colorbar=True, shrink=0.5)
# -

# ##### Backward tracking
#
# Load backward tracking pathlines

fpth = os.path.join(ws, f"{mpnamb}.mppth")
p = flopy.utils.PathlineFile(fpth)
pwb = p.get_destination_pathline_data(dest_cells=nodew)
prb = p.get_destination_pathline_data(dest_cells=nodesr)

# Load backward tracking endpoints

fpth = os.path.join(ws, f"{mpnamb}.mpend")
e = flopy.utils.EndpointFile(fpth)
ewb = e.get_destination_endpoint_data(dest_cells=nodew, source=True)
erb = e.get_destination_endpoint_data(dest_cells=nodesr, source=True)

# Well backward tracking pathlines

# +
f, axes = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))

ax = axes[0]
ax.set_aspect("equal")
ax.set_title("Well recharge area")
mm = flopy.plot.PlotMapView(model=gwf, ax=ax)
mm.plot_grid(lw=0.5)
mm.plot_pathline(
    pwb,
    layer="all",
    colors="blue",
    lw=0.5,
    linestyle=":",
    label="captured by wells",
)
mm.plot_endpoint(ewb, direction="ending")  # , colorbar=True, shrink=0.5);

ax = axes[1]
ax.set_aspect("equal")
ax.set_title("River recharge area")
mm = flopy.plot.PlotMapView(model=gwf, ax=ax)
mm.plot_grid(lw=0.5)
mm.plot_pathline(
    prb,
    layer="all",
    colors="green",
    lw=0.5,
    linestyle=":",
    label="captured by rivers",
)

plt.tight_layout()
