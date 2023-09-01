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

# # Migrating from MODPATH 7 to MODFLOW 6 PRT

# This example runs a GWF model, then a PRT model
# in separate simulations via flow model interface,
# then an identical MODPATH 7 model for comparison.
#
# The grid is a 10x10 square with a single layer,
# the same flow system shown in the FloPy readme.
#
# Particles are released from the top left cell.

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
simname = "prtfmi01"
gwfname = f"{simname}_gwf"
prtname = f"{simname}_prt"
mp7name = f"{simname}_mp7"

# output file names
gwf_budget_file = f"{gwfname}.bud"
gwf_head_file = f"{gwfname}.hds"
prt_track_file = f"{prtname}.trk"
prt_track_csv_file = f"{prtname}.trk.csv"
mp7_pathline_file = f"{mp7name}.mppth"

# ## Structured example

# ### Model setup

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

# Both the PRT and MP7 models depend on the same GWF model.

def build_gwf_sim(ws, mf6):
    # create simulation
    sim = flopy.mf6.MFSimulation(
        sim_name=simname,
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


# FloPy provides utilities to convert MODPATH 7 particle data
# to the appropriate format for MODFLOW 6 PRT PRP package input.

# We specify release point locations in MODPATH 7 format, then
# convert from local to global coordinates for MODFLOW 6 PRT.

# release points in mp7 format (using local coordinates)
releasepts_mp7 = [
    # node number, localx, localy, localz
    # (0-based indexing converted to 1-based for mp7 by flopy)
    (0, float(f"0.{i + 1}"), float(f"0.{i + 1}"), 0.5)
    for i in range(9)
]


def get_partdata(grid):
    return flopy.modpath.ParticleData(
        partlocs=[grid.get_lrc(p[0])[0] for p in releasepts_mp7],
        structured=True,
        localx=[p[1] for p in releasepts_mp7],
        localy=[p[2] for p in releasepts_mp7],
        localz=[p[3] for p in releasepts_mp7],
        timeoffset=0,
        drape=0,
    )


# We can now build the PRT simulation, using FloPy's `ParticleData.to_coords()` method to ease
# the conversion from MODPATH 7 to MODFLOW 6 PRT release points.


def build_prt_sim(ws, mf6):
    # create simulation
    sim = flopy.mf6.MFSimulation(
        sim_name=simname,
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

    # convert mp7 particledata to prt release points
    partdata = get_partdata(prt.modelgrid)
    coords = partdata.to_coords(prt.modelgrid)
    releasepts = [(i, 0, 0, 0, c[0], c[1], c[2]) for i, c in enumerate(coords)]

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


# ### MODPATH 7 setup

# Finally we can build the MODPATH 7 model and simulation.


def build_mp7_sim(ws, mp7, gwf):
    # convert mp7 particledata to prt release points
    partdata = get_partdata(gwf.modelgrid)

    # create modpath 7 simulation
    pg = flopy.modpath.ParticleGroup(
        particlegroupname="G1",
        particledata=partdata,
        filename=f"{mp7name}.sloc",
    )
    mp = flopy.modpath.Modpath7(
        modelname=mp7name,
        flowmodel=gwf,
        exe_name=mp7,
        model_ws=ws,
    )
    mpbas = flopy.modpath.Modpath7Bas(
        mp,
        porosity=porosity,
    )
    mpsim = flopy.modpath.Modpath7Sim(
        mp,
        simulationtype="pathline",
        trackingdirection="forward",
        budgetoutputoption="summary",
        stoptimeoption="extend",
        particlegroups=[pg],
    )

    return mp


# ### Running the models

# Construct and run the models.

# build mf6 simulations
gwfsim = build_gwf_sim(ws, "mf6")
prtsim = build_prt_sim(ws, "mf6")

# extract models and grid (useful for plotting etc)
gwf = gwfsim.get_model(gwfname)
prt = prtsim.get_model(prtname)
mg = gwf.modelgrid

# build mp7 model
mp7sim = build_mp7_sim(ws, "mp7", gwf)

# run mf6 models
for sim in [gwfsim, prtsim]:
    sim.write_simulation()
    success, _ = sim.run_simulation()
    assert success

# run mp7 model
mp7sim.write_input()
success, _ = mp7sim.run_model()
assert success

# ### Inspecting results

# Make sure the expected output files exist.

# check mf6 output files exist
assert (ws / gwf_budget_file).is_file()
assert (ws / gwf_head_file).is_file()
assert (ws / prt_track_file).is_file()
assert (ws / prt_track_csv_file).is_file()

# check mp7 output files exist
assert (ws / mp7_pathline_file).is_file()

# Load results from MODPATH 7 and MODFLOW 6 output files.

# load mf6 pathlines
pls = pd.read_csv(ws / prt_track_csv_file)

# load mp7 pathlines
plf = PathlineFile(ws / mp7_pathline_file)
mp7_pldata = pd.DataFrame(
    plf.get_destination_pathline_data(range(mg.nnodes), to_recarray=True)
)

# convert mp7 pathline fields from 0- to 1-based indexing
mp7_pldata["particleid"] = mp7_pldata["particleid"] + 1
mp7_pldata["particlegroup"] = mp7_pldata["particlegroup"] + 1
mp7_pldata["node"] = mp7_pldata["node"] + 1
mp7_pldata["k"] = mp7_pldata["k"] + 1

# load head, budget, and specific discharge from gwf model
hds = HeadFile(ws / gwf_head_file).get_data()
bud = gwf.output.budget()
spdis = bud.get_data(text="DATA-SPDIS")[0]
qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(spdis, gwf)

# Finally, plot the MODPATH 7 and MODFLOW 6 pathlines side-by-side in map view.

# +
# plot mf6 and mp7 pathlines in map view
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(13, 13))
for a in ax:
    a.set_aspect("equal")

pmv = flopy.plot.PlotMapView(modelgrid=mg, ax=ax[0])
pmv.plot_grid()
pmv.plot_array(hds[0], alpha=0.1)
pmv.plot_vector(qx, qy, normalize=True, color="white")
pathlines = pls.groupby(["imdl", "iprp", "irpt", "trelease"])
for ipl, ((imdl, iprp, irpt, trelease), pl) in enumerate(pathlines):
    pl.plot(
        title="MF6 pathlines",
        kind="line",
        x="x",
        y="y",
        marker="o",
        ax=ax[0],
        legend=False,
        color=cm.plasma(ipl / len(pathlines)),
    )

pmv = flopy.plot.PlotMapView(modelgrid=mg, ax=ax[1])
pmv.plot_grid()
pmv.plot_array(hds[0], alpha=0.1)
pmv.plot_vector(qx, qy, normalize=True, color="white")
mp7_plines = mp7_pldata.groupby(["particleid"])
for ipl, (pid, pl) in enumerate(mp7_plines):
    pl.plot(
        title="MP7 pathlines",
        kind="line",
        x="x",
        y="y",
        marker="o",
        ax=ax[1],
        legend=False,
        color=cm.plasma(ipl / len(mp7_plines)),
    )
# -

# Alternatively, FloPy built-in plotting functions can be used to plot pathlines.

# +
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(13, 13))
for a in ax:
    a.set_aspect("equal")

colors = cm.plasma(np.linspace(0, 1, len(mp7_pldata.particleid.unique())))

pmv = flopy.plot.PlotMapView(modelgrid=mg, ax=ax[0])
pmv.plot_grid()
pmv.plot_array(hds[0], alpha=0.1)
pmv.plot_vector(qx, qy, normalize=True, color="white")
pmv.plot_pathline(pls, layer="all", colors=colors)
ax[0].set_title("MP7 pathlines")

pmv = flopy.plot.PlotMapView(modelgrid=mg, ax=ax[1])
pmv.plot_grid()
pmv.plot_array(hds[0], alpha=0.1)
pmv.plot_vector(qx, qy, normalize=True, color="white")
pmv.plot_pathline(mp7_pldata, layer="all", colors=colors)
ax[1].set_title("MP7 pathlines")

# ## DISV example
#
# We now demonstrate a similar workflow for a DISV model.

# ### Model setup

# model info
nlay = 1
nper = 1
perlen = 10
nstp = 5
tsmult = 1.0
tdis_rc = [(perlen, nstp, tsmult)]
botm = [20.0]
strt = 20
nouter, ninner = 100, 300
hclose, rclose, relax = 1e-9, 1e-3, 0.97
porosity = 0.1

# Define a function to generate a vertex grid.

def create_disv_mesh():
    # Create a grid of verts
    nx, ny = (11, 11)
    x = np.linspace(0, 10, nx)
    y = np.linspace(0, 10, ny)
    xv, yv = np.meshgrid(x, y)
    yv = np.flipud(yv)

    verts = []
    vid = 0
    vert_lkup = {}
    for i in yv[:, 0]:
        for j in xv[0, :]:
            vert_lkup.update({(float(j), float(i)): vid})
            verts.append([int(vid), float(j), float(i)])
            vid += 1

    ivert = []
    ivid = 0
    xyverts = []
    xc, yc = [], []  # for storing the cell center location
    for i in yv[:-1, 0]:
        for j in xv[0, :-1]:
            xlst, ylst = [], []
            vid_lst = []
            # Start with upper-left corner and go clockwise
            for ct in [0, 1, 2, 3]:
                if ct == 0:
                    iadj = 0.0
                    jadj = 0.0
                elif ct == 1:
                    iadj = 0.0
                    jadj = 1.0
                elif ct == 2:
                    iadj = -1.0
                    jadj = 1.0
                elif ct == 3:
                    iadj = -1.0
                    jadj = 0.0

                vid = vert_lkup[(float(j + jadj), float(i + iadj))]
                vid_lst.append(vid)

                xlst.append(float(j + jadj))
                ylst.append(float(i + iadj))

            xc.append(np.mean(xlst))
            yc.append(np.mean(ylst))
            xyverts.append(list(zip(xlst, ylst)))

            rec = [ivid] + vid_lst
            ivert.append(rec)

            ivid += 1

    # finally, create a cell2d record
    cell2d = []
    for ix, iv in enumerate(ivert):
        xvt, yvt = np.array(xyverts[ix]).T
        if flopy.utils.geometry.is_clockwise(xvt, yvt):
            rec = [iv[0], xc[ix], yc[ix], len(iv[1:])] + iv[1:]
        else:
            iiv = iv[1:][::-1]
            rec = [iv[0], xc[ix], yc[ix], len(iiv)] + iiv

        cell2d.append(rec)

    return verts, cell2d

# Generate vertex grid properties.

verts, cell2d = create_disv_mesh()

# Specify release points in mp7 format (using local coordinates).

releasepts_mp7 = [
    # node number, localx, localy, localz
    # (0-based indexing converted to 1-based for mp7 by flopy)
    (i * 10, 0.5, 0.5, 0.5)
    for i in range(10)
]

# Define model names.

gwfname = f"{simname}-gwf"
prtname = f"{simname}-prt"
mp7name = f"{simname}-mp7"

# Define a function to build the GWF model.

def build_gwf_sim(ws, mf6):
    # build MODFLOW 6 files
    sim = flopy.mf6.MFSimulation(
        sim_name=simname, version="mf6", exe_name=mf6, sim_ws=ws
    )

    # create tdis package
    tdis = flopy.mf6.ModflowTdis(
        sim, time_units="DAYS", nper=nper, perioddata=tdis_rc
    )

    # create gwf model
    gwf = flopy.mf6.ModflowGwf(
        sim, modelname=gwfname, newtonoptions="NEWTON", save_flows=True
    )

    # create iterative model solution and register the gwf model with it
    ims = flopy.mf6.ModflowIms(
        sim,
        print_option="SUMMARY",
        complexity="MODERATE",
        outer_dvclose=hclose,
        outer_maximum=nouter,
        under_relaxation="DBD",
        inner_maximum=ninner,
        inner_dvclose=hclose,
        rcloserecord=rclose,
        linear_acceleration="BICGSTAB",
        scaling_method="NONE",
        reordering_method="NONE",
        relaxation_factor=relax,
    )
    sim.register_ims_package(ims, [gwf.name])

    ncpl = len(cell2d)
    nvert = len(verts)
    disv = flopy.mf6.ModflowGwfdisv(
        gwf,
        nlay=nlay,
        ncpl=ncpl,
        nvert=nvert,
        top=25.0,
        botm=botm,
        vertices=verts,
        cell2d=cell2d,
    )

    # initial conditions
    ic = flopy.mf6.ModflowGwfic(gwf, strt=strt)

    # node property flow
    npf = flopy.mf6.ModflowGwfnpf(
        gwf,
        save_flows=True,
        save_specific_discharge=True,
        save_saturation=True,
    )

    # constant head boundary
    spd = {
        0: [[(0, 0), 1.0, 1.0], [(0, 99), 0.0, 0.0]],
        # 1: [[(0, 0, 0), 0.0, 0.0], [(0, 9, 9), 1.0, 2.0]],
    }
    chd = flopy.mf6.ModflowGwfchd(
        gwf,
        pname="CHD-1",
        stress_period_data=spd,
        auxiliary=["concentration"],
    )

    # output control
    oc = flopy.mf6.ModflowGwfoc(
        gwf,
        budget_filerecord="{}.cbc".format(gwfname),
        head_filerecord="{}.hds".format(gwfname),
        headprintrecord=[("COLUMNS", 10, "WIDTH", 15, "DIGITS", 6, "GENERAL")],
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        printrecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        filename="{}.oc".format(gwfname),
    )

    # Print human-readable heads
    obs_lst = []
    for k in np.arange(0, 1, 1):
        for i in np.arange(40, 50, 1):
            obs_lst.append(["obs_" + str(i + 1), "head", (k, i)])

    obs_dict = {f"{gwfname}.obs.csv": obs_lst}
    obs = flopy.mf6.ModflowUtlobs(
        gwf, pname="head_obs", digits=20, continuous=obs_dict
    )

    return sim


def build_prt_sim(ws, mf6):
    # create simulation
    sim = flopy.mf6.MFSimulation(
        sim_name=simname,
        exe_name=mf6,
        version="mf6",
        sim_ws=ws,
    )

    # create tdis package
    tdis = flopy.mf6.ModflowTdis(
        sim, time_units="DAYS", nper=nper, perioddata=tdis_rc
    )

    # create prt model
    prt = flopy.mf6.ModflowPrt(sim, modelname=prtname)

    # create prt discretization
    ncpl = len(cell2d)
    nvert = len(verts)
    disv = flopy.mf6.ModflowGwfdisv(
        prt,
        nlay=nlay,
        ncpl=ncpl,
        nvert=nvert,
        top=25.0,
        botm=botm,
        vertices=verts,
        cell2d=cell2d,
    )

    # create mip package
    flopy.mf6.ModflowPrtmip(prt, pname="mip", porosity=porosity)

    # convert mp7 particledata to prt release points
    partdata = get_partdata(prt.modelgrid, releasepts_mp7)
    coords = partdata.to_coords(prt.modelgrid)
    releasepts = [
        (i, (0, r[0]), c[0], c[1], c[2])
        for i, (r, c) in enumerate(zip(releasepts_mp7, coords))
    ]

    # create prp package
    prp_track_file = f"{prtname}.prp.trk"
    prp_track_csv_file = f"{prtname}.prp.trk.csv"
    flopy.mf6.ModflowPrtprp(
        prt,
        pname="prp1",
        filename=f"{prtname}_1.prp",
        nreleasepts=len(releasepts),
        packagedata=releasepts,
        perioddata={0: ["FIRST"]},
        track_filerecord=[prp_track_file],
        trackcsv_filerecord=[prp_track_csv_file],
        stop_at_weak_sink="saws" in prtname,
        boundnames=True,
    )

    # create output control package
    prt_track_file = f"{prtname}.trk"
    prt_track_csv_file = f"{prtname}.trk.csv"
    flopy.mf6.ModflowPrtoc(
        prt,
        pname="oc",
        track_filerecord=[prt_track_file],
        trackcsv_filerecord=[prt_track_csv_file],
    )

    # create the flow model interface
    gwf_budget_file = f"{gwfname}.cbc"
    gwf_head_file = f"{gwfname}.hds"
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

# Define a function build the MODPATH 7 model.

def build_mp7_sim(ws, mp7, gwf):
    # convert mp7 particledata to prt release points
    partdata = get_partdata(gwf.modelgrid, releasepts_mp7)

    # create modpath 7 simulation
    pg = flopy.modpath.ParticleGroup(
        particlegroupname="G1",
        particledata=partdata,
        filename=f"{mp7name}.sloc",
    )
    mp = flopy.modpath.Modpath7(
        modelname=mp7name,
        flowmodel=gwf,
        exe_name=mp7,
        model_ws=ws,
    )
    mpbas = flopy.modpath.Modpath7Bas(
        mp,
        porosity=porosity,
    )
    mpsim = flopy.modpath.Modpath7Sim(
        mp,
        simulationtype="pathline",
        trackingdirection="forward",
        budgetoutputoption="summary",
        stoptimeoption="total",
        particlegroups=[pg],
    )

    return mp

# ### Running the models

# Construct and run the models.

# build mf6 simulations
gwfsim = build_gwf_sim(ws, "mf6")
prtsim = build_prt_sim(ws, "mf6")

# extract models and grid (useful for plotting etc)
gwf = gwfsim.get_model(gwfname)
prt = prtsim.get_model(prtname)
mg = gwf.modelgrid

# build mp7 model
mp7sim = build_mp7_sim(ws, "mp7", gwf)

# run mf6 models
for sim in [gwfsim, prtsim]:
    sim.write_simulation()
    success, _ = sim.run_simulation()
    assert success

# run mp7 model
mp7sim.write_input()
success, _ = mp7sim.run_model()
assert success