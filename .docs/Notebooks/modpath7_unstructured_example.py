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

# # Using MODPATH 7 with a DISV unstructured model
#
# This is a replication of the MODPATH Problem 2 example that is described on page 12 of the modpath_7_examples.pdf file.  The results shown here should be the same as the results in the MODPATH example, however, the vertex and node numbering used here may be different from the numbering used in MODPATH, so head values may not be compared directly without some additional mapping.

# ## Part I. Setup Notebook

# +
import sys
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

proj_root = Path.cwd().parent.parent

# run installed version of flopy or add local path
try:
    import flopy
except:
    sys.path.append(proj_root)
    import flopy

print(sys.version)
print("numpy version: {}".format(np.__version__))
print("matplotlib version: {}".format(mpl.__version__))
print("flopy version: {}".format(flopy.__version__))

# temporary directory
temp_dir = TemporaryDirectory()
workspace = Path(temp_dir.name)
# -

# ## Part II. Gridgen Creation of Model Grid
#
# Create the base model grid.

Lx = 10000.0
Ly = 10500.0
nlay = 3
nrow = 21
ncol = 20
delr = Lx / ncol
delc = Ly / nrow
top = 400
botm = [220, 200, 0]

ms = flopy.modflow.Modflow()
dis5 = flopy.modflow.ModflowDis(
    ms,
    nlay=nlay,
    nrow=nrow,
    ncol=ncol,
    delr=delr,
    delc=delc,
    top=top,
    botm=botm,
)

# Create the `Gridgen` object.

# +
from flopy.utils.gridgen import Gridgen

model_name = "mp7p2_u"
model_ws = workspace / "mp7_ex2" / "mf6"
gridgen_ws = model_ws / "gridgen"
g = Gridgen(ms.modelgrid, model_ws=gridgen_ws)
# -

# Refine the grid.

# +
rf0shp = gridgen_ws / "rf0"
xmin = 7 * delr
xmax = 12 * delr
ymin = 8 * delc
ymax = 13 * delc
rfpoly = [
    [
        list(
            reversed(
                [
                    (xmin, ymin),
                    (xmax, ymin),
                    (xmax, ymax),
                    (xmin, ymax),
                    (xmin, ymin),
                ]
            )
        )
    ]
]
g.add_refinement_features(rfpoly, "polygon", 1, range(nlay))

rf1shp = gridgen_ws / "rf1"
xmin = 8 * delr
xmax = 11 * delr
ymin = 9 * delc
ymax = 12 * delc
rfpoly = [
    [
        list(
            reversed(
                [
                    (xmin, ymin),
                    (xmax, ymin),
                    (xmax, ymax),
                    (xmin, ymax),
                    (xmin, ymin),
                ]
            )
        )
    ]
]
g.add_refinement_features(rfpoly, "polygon", 2, range(nlay))

rf2shp = gridgen_ws / "rf2"
xmin = 9 * delr
xmax = 10 * delr
ymin = 10 * delc
ymax = 11 * delc
rfpoly = [
    [
        list(
            reversed(
                [
                    (xmin, ymin),
                    (xmax, ymin),
                    (xmax, ymax),
                    (xmin, ymax),
                    (xmin, ymin),
                ]
            )
        )
    ]
]
g.add_refinement_features(rfpoly, "polygon", 3, range(nlay))
# -

# Show the model grid with refinement levels superimposed.

fig = plt.figure(figsize=(5, 5), constrained_layout=True)
ax = fig.add_subplot(1, 1, 1)
mm = flopy.plot.PlotMapView(model=ms)
mm.plot_grid()
flopy.plot.plot_shapefile(rf0shp, ax=ax, facecolor="yellow", edgecolor="none")
flopy.plot.plot_shapefile(rf1shp, ax=ax, facecolor="pink", edgecolor="none")
flopy.plot.plot_shapefile(rf2shp, ax=ax, facecolor="red", edgecolor="none");

# Build the refined grid.

g.build(verbose=False)

# Show the refined grid.

fig = plt.figure(figsize=(5, 5), constrained_layout=True)
ax = fig.add_subplot(1, 1, 1, aspect="equal")
g.plot(ax, linewidth=0.5);

# Extract the refined grid's properties.

gridprops = g.get_gridprops_disv()
ncpl = gridprops["ncpl"]
top = gridprops["top"]
botm = gridprops["botm"]
nvert = gridprops["nvert"]
vertices = gridprops["vertices"]
cell2d = gridprops["cell2d"]

# ## Part III. Create the Flopy Model

# +
# create simulation
sim = flopy.mf6.MFSimulation(
    sim_name=model_name, version="mf6", exe_name="mf6", sim_ws=model_ws
)

# create tdis package
tdis_rc = [(1000.0, 1, 1.0)]
tdis = flopy.mf6.ModflowTdis(
    sim, pname="tdis", time_units="DAYS", perioddata=tdis_rc
)

# create gwf model
gwf = flopy.mf6.ModflowGwf(
    sim, modelname=model_name, model_nam_file="{}.nam".format(model_name)
)
gwf.name_file.save_flows = True

# create iterative model solution and register the gwf model with it
ims = flopy.mf6.ModflowIms(
    sim,
    pname="ims",
    print_option="SUMMARY",
    complexity="SIMPLE",
    outer_dvclose=1.0e-5,
    outer_maximum=100,
    under_relaxation="NONE",
    inner_maximum=100,
    inner_dvclose=1.0e-6,
    rcloserecord=0.1,
    linear_acceleration="BICGSTAB",
    scaling_method="NONE",
    reordering_method="NONE",
    relaxation_factor=0.99,
)
sim.register_ims_package(ims, [gwf.name])

# disv
disv = flopy.mf6.ModflowGwfdisv(
    gwf,
    nlay=nlay,
    ncpl=ncpl,
    top=top,
    botm=botm,
    nvert=nvert,
    vertices=vertices,
    cell2d=cell2d,
)

# initial conditions
ic = flopy.mf6.ModflowGwfic(gwf, pname="ic", strt=320.0)

# node property flow
npf = flopy.mf6.ModflowGwfnpf(
    gwf,
    xt3doptions=[("xt3d")],
    icelltype=[1, 0, 0],
    k=[50.0, 0.01, 200.0],
    k33=[10.0, 0.01, 20.0],
)

# wel
wellpoints = [(4750.0, 5250.0)]
welcells = g.intersect(wellpoints, "point", 0)
# welspd = flopy.mf6.ModflowGwfwel.stress_period_data.empty(gwf, maxbound=1, aux_vars=['iface'])
welspd = [[(2, icpl), -150000, 0] for icpl in welcells["nodenumber"]]
wel = flopy.mf6.ModflowGwfwel(
    gwf, print_input=True, auxiliary=[("iface",)], stress_period_data=welspd
)

# rch
aux = [np.ones(ncpl, dtype=int) * 6]
rch = flopy.mf6.ModflowGwfrcha(
    gwf, recharge=0.005, auxiliary=[("iface",)], aux={0: [6]}
)
# riv
riverline = [[(Lx - 1.0, Ly), (Lx - 1.0, 0.0)]]
rivcells = g.intersect(riverline, "line", 0)
rivspd = [[(0, icpl), 320.0, 100000.0, 318] for icpl in rivcells["nodenumber"]]
riv = flopy.mf6.ModflowGwfriv(gwf, stress_period_data=rivspd)

# output control
oc = flopy.mf6.ModflowGwfoc(
    gwf,
    pname="oc",
    budget_filerecord="{}.cbb".format(model_name),
    head_filerecord="{}.hds".format(model_name),
    headprintrecord=[("COLUMNS", 10, "WIDTH", 15, "DIGITS", 6, "GENERAL")],
    saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    printrecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
)
# -

# Now write the simulation input files.

sim.write_simulation()

# ## Part IV. Run the MODFLOW 6 Model

success, buff = sim.run_simulation(silent=True, report=True)
assert success, "mf6 failed to run"
for line in buff:
    print(line)

# ## Part V. Import and Plot the Results

# Plot the boundary conditions on the grid.

fname = os.path.join(model_ws, model_name + ".disv.grb")
grd = flopy.mf6.utils.MfGrdFile(fname, verbose=False)
mg = grd.modelgrid
ibd = np.zeros((ncpl), dtype=int)
ibd[welcells["nodenumber"]] = 1
ibd[rivcells["nodenumber"]] = 2
ibd = np.ma.masked_equal(ibd, 0)
fig = plt.figure(figsize=(8, 8), constrained_layout=True)
ax = fig.add_subplot(1, 1, 1, aspect="equal")
pmv = flopy.plot.PlotMapView(modelgrid=mg, ax=ax)
ax.set_xlim(0, Lx)
ax.set_ylim(0, Ly)
cmap = mpl.colors.ListedColormap(
    [
        "r",
        "g",
    ]
)
pc = pmv.plot_array(ibd, cmap=cmap, edgecolor="gray")
t = ax.set_title("Boundary Conditions\n");

fname = os.path.join(model_ws, model_name + ".hds")
hdobj = flopy.utils.HeadFile(fname)
head = hdobj.get_data()
head.shape

ilay = 2
cint = 0.25
fig = plt.figure(figsize=(8, 8), constrained_layout=True)
ax = fig.add_subplot(1, 1, 1, aspect="equal")
mm = flopy.plot.PlotMapView(modelgrid=mg, ax=ax, layer=ilay)
ax.set_xlim(0, Lx)
ax.set_ylim(0, Ly)
pc = mm.plot_array(head[:, 0, :], cmap="jet", edgecolor="black")
hmin = head[ilay, 0, :].min()
hmax = head[ilay, 0, :].max()
levels = np.arange(np.floor(hmin), np.ceil(hmax) + cint, cint)
cs = mm.contour_array(head[:, 0, :], colors="white", levels=levels)
plt.clabel(cs, fmt="%.1f", colors="white", fontsize=11)
cb = plt.colorbar(pc, shrink=0.5)
t = ax.set_title(
    "Model Layer {}; hmin={:6.2f}, hmax={:6.2f}".format(ilay + 1, hmin, hmax)
);

# Inspect model cells and vertices.

# +
# zoom area
xmin, xmax = 2000, 4500
ymin, ymax = 5400, 7500

mg.get_cell_vertices
fig = plt.figure(figsize=(8, 8), constrained_layout=True)
ax = fig.add_subplot(1, 1, 1, aspect="equal")
mm = flopy.plot.PlotMapView(modelgrid=mg, ax=ax)
v = mm.plot_grid(edgecolor="black")
t = ax.set_title("Model Cells and Vertices (one-based)\n")
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

verts = mg.verts
ax.plot(verts[:, 0], verts[:, 1], "bo")
for i in range(ncpl):
    x, y = verts[i, 0], verts[i, 1]
    if xmin <= x <= xmax and ymin <= y <= ymax:
        ax.annotate(str(i + 1), verts[i, :], color="b")

xc, yc = mg.get_xcellcenters_for_layer(0), mg.get_ycellcenters_for_layer(0)
for i in range(ncpl):
    x, y = xc[i], yc[i]
    ax.plot(x, y, "ro")
    if xmin <= x <= xmax and ymin <= y <= ymax:
        ax.annotate(str(i + 1), (x, y), color="r")
# -

# ## Part VI. Create the Flopy MODPATH7 Models
#
# Define names for the MODPATH 7 simulations.

mp_namea = model_name + "a_mp"
mp_nameb = model_name + "b_mp"

# Create particles for the pathline and timeseries analysis.

# +
pcoord = np.array(
    [
        [0.000, 0.125, 0.500],
        [0.000, 0.375, 0.500],
        [0.000, 0.625, 0.500],
        [0.000, 0.875, 0.500],
        [1.000, 0.125, 0.500],
        [1.000, 0.375, 0.500],
        [1.000, 0.625, 0.500],
        [1.000, 0.875, 0.500],
        [0.125, 0.000, 0.500],
        [0.375, 0.000, 0.500],
        [0.625, 0.000, 0.500],
        [0.875, 0.000, 0.500],
        [0.125, 1.000, 0.500],
        [0.375, 1.000, 0.500],
        [0.625, 1.000, 0.500],
        [0.875, 1.000, 0.500],
    ]
)
nodew = gwf.disv.ncpl.array * 2 + welcells["nodenumber"][0]
plocs = [nodew for i in range(pcoord.shape[0])]

# create particle data
pa = flopy.modpath.ParticleData(
    plocs,
    structured=False,
    localx=pcoord[:, 0],
    localy=pcoord[:, 1],
    localz=pcoord[:, 2],
    drape=0,
)

# create backward particle group
fpth = mp_namea + ".sloc"
pga = flopy.modpath.ParticleGroup(
    particlegroupname="BACKWARD1", particledata=pa, filename=fpth
)
# -

# Create particles for endpoint analysis.

facedata = flopy.modpath.FaceDataType(
    drape=0,
    verticaldivisions1=10,
    horizontaldivisions1=10,
    verticaldivisions2=10,
    horizontaldivisions2=10,
    verticaldivisions3=10,
    horizontaldivisions3=10,
    verticaldivisions4=10,
    horizontaldivisions4=10,
    rowdivisions5=0,
    columndivisions5=0,
    rowdivisions6=4,
    columndivisions6=4,
)
pb = flopy.modpath.NodeParticleData(subdivisiondata=facedata, nodes=nodew)
# create forward particle group
fpth = mp_nameb + ".sloc"
pgb = flopy.modpath.ParticleGroupNodeTemplate(
    particlegroupname="BACKWARD2", particledata=pb, filename=fpth
)

# Create and run the pathline and timeseries analysis model.

# +
# create modpath files
mp = flopy.modpath.Modpath7(
    modelname=mp_namea, flowmodel=gwf, exe_name="mp7", model_ws=model_ws
)
flopy.modpath.Modpath7Bas(mp, porosity=0.1)
flopy.modpath.Modpath7Sim(
    mp,
    simulationtype="combined",
    trackingdirection="backward",
    weaksinkoption="pass_through",
    weaksourceoption="pass_through",
    referencetime=0.0,
    stoptimeoption="extend",
    timepointdata=[500, 1000.0],
    particlegroups=pga,
)

# write modpath datasets
mp.write_input()

# run modpath
success, buff = mp.run_model(silent=True, report=True)
assert success, "mp7 failed to run"
for line in buff:
    print(line)
# -

# Load the pathline and timeseries data.

fpth = model_ws / f"{mp_namea}.mppth"
p = flopy.utils.PathlineFile(fpth)
p0 = p.get_alldata()

fpth = model_ws / f"{mp_namea}.timeseries"
ts = flopy.utils.TimeseriesFile(fpth)
ts0 = ts.get_alldata()

# Plot the pathline and timeseries data.

fig = plt.figure(figsize=(8, 8), constrained_layout=True)
ax = fig.add_subplot(1, 1, 1, aspect="equal")
mm = flopy.plot.PlotMapView(modelgrid=mg, ax=ax)
ax.set_xlim(0, Lx)
ax.set_ylim(0, Ly)
cmap = mpl.colors.ListedColormap(
    [
        "r",
        "g",
    ]
)
v = mm.plot_array(ibd, cmap=cmap, edgecolor="gray")
mm.plot_pathline(p0, layer="all", colors="blue", lw=0.75)
colors = ["green", "orange", "red"]
for k in range(nlay):
    mm.plot_timeseries(ts0, layer=k, marker="o", lw=0, color=colors[k]);

# Create and run the endpoint analysis model.

# +
# create modpath files
mp = flopy.modpath.Modpath7(
    modelname=mp_nameb, flowmodel=gwf, exe_name="mp7", model_ws=model_ws
)
flopy.modpath.Modpath7Bas(mp, porosity=0.1)
flopy.modpath.Modpath7Sim(
    mp,
    simulationtype="endpoint",
    trackingdirection="backward",
    weaksinkoption="pass_through",
    weaksourceoption="pass_through",
    referencetime=0.0,
    stoptimeoption="extend",
    particlegroups=pgb,
)

# write modpath datasets
mp.write_input()

# run modpath
success, buff = mp.run_model(silent=True, report=True)
assert success, "mp7 failed to run"
for line in buff:
    print(line)
# -

# Load the endpoint data.

fpth = model_ws / f"{mp_nameb}.mpend"
e = flopy.utils.EndpointFile(fpth)
e0 = e.get_alldata()

# Plot the endpoint data.

fig = plt.figure(figsize=(8, 8), constrained_layout=True)
ax = fig.add_subplot(1, 1, 1, aspect="equal")
mm = flopy.plot.PlotMapView(modelgrid=mg, ax=ax)
ax.set_xlim(0, Lx)
ax.set_ylim(0, Ly)
cmap = mpl.colors.ListedColormap(
    [
        "r",
        "g",
    ]
)
v = mm.plot_array(ibd, cmap=cmap, edgecolor="gray")
mm.plot_endpoint(e0, direction="ending", colorbar=True, shrink=0.5);

# Clean up the temporary workspace.

try:
    # ignore PermissionError on Windows
    temp_dir.cleanup()
except:
    pass
