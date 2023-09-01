# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: metadata
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   metadata:
#     section: mf6
# ---

# # MODFLOW 6 PRT (particle-tracking) tutorial

# This tutorial runs a GWF model then a PRT model
# in separate simulations via flow model interface.
#
# The grid is a 10x10 square with a single layer,
# the same flow system shown in the FloPy readme.
#
# Particles are released from the top left cell.
#
# ## Initial setup
#
# First, import dependencies, set up a temporary workspace, and define some filenames and model parameters.


import os
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import flopy
from flopy.utils import PathlineFile
from flopy.utils.binaryfile import HeadFile

# workspace
temp_dir = TemporaryDirectory()
ws = Path(temp_dir.name)

# model names
name = "prtfmi01"
gwfname = f"{name}_gwf"
prtname = f"{name}_prt"
mp7name = f"{name}_mp7"

# output file names
gwf_budget_file = f"{gwfname}.bud"
gwf_head_file = f"{gwfname}.hds"
prt_track_file = f"{prtname}.trk"
prt_track_csv_file = f"{prtname}.trk.csv"
mp7_pathline_file = f"{mp7name}.mppth"

# model info
nlay = 1
nrow = 10
ncol = 10
top = 1.0
botm = [0.0]
nper = 1
perlen = 1.0
nstp = 1
tsmult = 1.0
porosity = 0.1

# Define release points. We will release three particles from the top left cell of the model.

# release points
releasepts = [
    # particle index, k, i, j, x, y, z
    # (0-based indexing converted to 1-based for mf6 by flopy)
    (0, 0, 0, 0, 0.25, 9.25, 0.5),
    (1, 0, 0, 0, 0.5, 9.5, 0.5),
    (2, 0, 0, 0, 0.75, 9.75, 0.5),
]

# Particle-tracking models require a flow solution to solve for particle motion &mdash; PRT models may either run side-by-side with a GWF model with an exchange, or may consume the output of a previously run GWF model via flow model interface.
#
# In this tutorial we do the latter. First we define a GWF model and simulation, then a separate PRT model/simulation.
#
# Define a function to build a MODFLOW 6 GWF model.


def build_gwf_sim(ws, mf6):
    # create simulation
    sim = flopy.mf6.MFSimulation(
        sim_name=name,
        exe_name=mf6,
        version="mf6",
        sim_ws=ws,
    )

    # create tdis package
    flopy.mf6.modflow.mftdis.ModflowTdis(
        sim,
        pname="tdis",
        time_units="DAYS",
        nper=nper,
        perioddata=[(perlen, nstp, tsmult)],
    )

    # create gwf model
    gwf = flopy.mf6.ModflowGwf(sim, modelname=gwfname, save_flows=True)

    # create gwf discretization
    flopy.mf6.modflow.mfgwfdis.ModflowGwfdis(
        gwf,
        pname="dis",
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
    )

    # create gwf initial conditions package
    flopy.mf6.modflow.mfgwfic.ModflowGwfic(gwf, pname="ic")

    # create gwf node property flow package
    flopy.mf6.modflow.mfgwfnpf.ModflowGwfnpf(
        gwf,
        pname="npf",
        save_saturation=True,
        save_specific_discharge=True,
    )

    # create gwf chd package
    spd = {
        0: [[(0, 0, 0), 1.0, 1.0], [(0, 9, 9), 0.0, 0.0]],
        1: [[(0, 0, 0), 0.0, 0.0], [(0, 9, 9), 1.0, 2.0]],
    }
    chd = flopy.mf6.ModflowGwfchd(
        gwf,
        pname="CHD-1",
        stress_period_data=spd,
        auxiliary=["concentration"],
    )

    # create gwf output control package
    oc = flopy.mf6.ModflowGwfoc(
        gwf,
        budget_filerecord=gwf_budget_file,
        head_filerecord=gwf_head_file,
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )

    # create iterative model solution for gwf model
    ims = flopy.mf6.ModflowIms(sim)

    return sim


# Define the PRT simulation.


def build_prt_sim(ws, mf6):
    # create simulation
    sim = flopy.mf6.MFSimulation(
        sim_name=name,
        exe_name=mf6,
        version="mf6",
        sim_ws=ws,
    )

    # create tdis package
    flopy.mf6.modflow.mftdis.ModflowTdis(
        sim,
        pname="tdis",
        time_units="DAYS",
        nper=nper,
        perioddata=[(perlen, nstp, tsmult)],
    )

    # create prt model
    prt = flopy.mf6.ModflowPrt(sim, modelname=prtname)

    # create prt discretization
    flopy.mf6.modflow.mfgwfdis.ModflowGwfdis(
        prt,
        pname="dis",
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
    )

    # create mip package
    flopy.mf6.ModflowPrtmip(prt, pname="mip", porosity=porosity)

    # create prp package
    flopy.mf6.ModflowPrtprp(
        prt,
        pname="prp1",
        filename=f"{prtname}_1.prp",
        nreleasepts=len(releasepts),
        packagedata=releasepts,
        perioddata={0: ["FIRST"]},
    )

    # create output control package
    flopy.mf6.ModflowPrtoc(
        prt,
        pname="oc",
        track_filerecord=[prt_track_file],
        trackcsv_filerecord=[prt_track_csv_file],
    )

    # create the flow model interface
    flopy.mf6.ModflowPrtfmi(
        prt,
        packagedata=[
            ("GWFHEAD", gwf_head_file),
            ("GWFBUDGET", gwf_budget_file),
        ],
    )

    # add explicit model solution
    ems = flopy.mf6.ModflowEms(
        sim,
        pname="ems",
        filename=f"{prtname}.ems",
    )
    sim.register_solution_package(ems, [prt.name])

    return sim


# ## Running models
#
# Next, build and run the models. The flow model must run before the tracking model so the latter can consume the former's cell budget output file (configured above via flow model interface).

# build mf6 models
gwfsim = build_gwf_sim(ws, "mf6")
prtsim = build_prt_sim(ws, "mf6")

# run mf6 models
for sim in [gwfsim, prtsim]:
    sim.write_simulation()
    success, _ = sim.run_simulation()
    assert success

# Extract model and grid objects from the simulations for use with plots.

# extract model objects
gwf = gwfsim.get_model(gwfname)
prt = prtsim.get_model(prtname)

# extract model grid
mg = gwf.modelgrid

# ## Inspecting results
#
# Finally we can load and plot results. First make sure the expected output files were created, then load pathlines, heads, budget, and specific discharge data.

# check mf6 output files exist
assert (ws / gwf_budget_file).is_file()
assert (ws / gwf_head_file).is_file()
assert (ws / prt_track_file).is_file()
assert (ws / prt_track_csv_file).is_file()

# load pathlines
pls = pd.read_csv(ws / prt_track_csv_file)

# extract head, budget, and specific discharge results from GWF model
hds = HeadFile(ws / gwf_head_file).get_data()
bud = gwf.output.budget()
spdis = bud.get_data(text="DATA-SPDIS")[0]
qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(spdis, gwf)

# Path points/lines can be plotted straightforwardly.

# plot mf6 pathlines in map view
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(13, 13))
ax.set_aspect("equal")
pmv = flopy.plot.PlotMapView(modelgrid=mg, ax=ax)
pmv.plot_grid()
pmv.plot_array(hds[0], alpha=0.1)
pmv.plot_vector(qx, qy, normalize=True, color="white")
pathlines = pls.groupby(["imdl", "iprp", "irpt", "trelease"])
for ipl, ((imdl, iprp, irpt, trelease), pl) in enumerate(pathlines):
    pl.plot(
        title="Pathlines",
        kind="line",
        x="x",
        y="y",
        marker="o",
        ax=ax,
        legend=False,
        color=cm.plasma(ipl / len(pathlines)),
    )

# Alternatively, with FloPy's built-in pathline plotting utilities:

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(13, 13))
ax.set_aspect("equal")
pmv = flopy.plot.PlotMapView(modelgrid=mg, ax=ax)
pmv.plot_grid()
pmv.plot_array(hds[0], alpha=0.1)
pmv.plot_vector(qx, qy, normalize=True, color="white")
colors = cm.plasma(
    np.linspace(
        0, 1, pls.groupby(["imdl", "iprp", "irpt", "trelease"]).ngroups
    )
)
pmv.plot_pathline(pls, layer="all", colors=colors, linewidth=2)
