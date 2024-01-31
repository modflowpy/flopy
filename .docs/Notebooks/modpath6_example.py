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
#       - name: Andrew Leaf
# ---

# # Working with MODPATH 6
#
# This notebook demonstrates forward and backward tracking with MODPATH. The notebook also shows how to create subsets of pathline and endpoint information, plot MODPATH results on ModelMap objects, and export endpoints and pathlines as shapefiles.

import os
import shutil

# +
import sys
from pprint import pformat
from tempfile import TemporaryDirectory

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import flopy

print(sys.version)
print(f"numpy version: {np.__version__}")
print(f"matplotlib version: {mpl.__version__}")
print(f"pandas version: {pd.__version__}")
print(f"flopy version: {flopy.__version__}")
# -

# Load the MODFLOW model, then switch to a temporary working directory.

# +
from pathlib import Path

# temporary directory
temp_dir = TemporaryDirectory()
model_ws = temp_dir.name

model_path = Path.cwd().parent.parent / "examples" / "data" / "mp6"
mffiles = list(model_path.glob("EXAMPLE.*"))

m = flopy.modflow.Modflow.load("EXAMPLE.nam", model_ws=model_path)

hdsfile = flopy.utils.HeadFile(os.path.join(model_path, "EXAMPLE.HED"))
hdsfile.get_kstpkper()

hds = hdsfile.get_data(kstpkper=(0, 2))
# -

# Plot RIV bc and head results.

plt.imshow(hds[4, :, :])
plt.colorbar()

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, aspect="equal")
mapview = flopy.plot.PlotMapView(model=m, layer=4)
quadmesh = mapview.plot_ibound()
linecollection = mapview.plot_grid()
riv = mapview.plot_bc("RIV", color="g", plotAll=True)
quadmesh = mapview.plot_bc("WEL", kper=1, plotAll=True)
contour_set = mapview.contour_array(
    hds, levels=np.arange(np.min(hds), np.max(hds), 0.5), colors="b"
)
plt.clabel(contour_set, inline=1, fontsize=14)

# Now create forward particle tracking simulation where particles are released at the top of each cell in layer 1:
# * specifying the recharge package in ```create_mpsim``` releases a single particle on iface=6 of each top cell
# * start the particles at begining of per 3, step 1, as in example 3 in MODPATH6 manual
#
# **Note:** in FloPy version 3.3.5 and previous, the `Modpath6` constructor `dis_file`, `head_file` and `budget_file` arguments expected filenames relative to the model workspace. In 3.3.6 and later, full paths must be provided &mdash; if they are not, discretization, head and budget data are read directly from the model, as before.

# +
from os.path import join

mp = flopy.modpath.Modpath6(
    modelname="ex6",
    exe_name="mp6",
    modflowmodel=m,
    model_ws=str(model_path),
)

mpb = flopy.modpath.Modpath6Bas(
    mp, hdry=m.lpf.hdry, laytyp=m.lpf.laytyp, ibound=1, prsity=0.1
)

# start the particles at begining of per 3, step 1, as in example 3 in MODPATH6 manual
# (otherwise particles will all go to river)
sim = mp.create_mpsim(
    trackdir="forward",
    simtype="pathline",
    packages="RCH",
    start_time=(2, 0, 1.0),
)

shutil.copy(model_path / "EXAMPLE.DIS", join(model_ws, "EXAMPLE.DIS"))
shutil.copy(model_path / "EXAMPLE.HED", join(model_ws, "EXAMPLE.HED"))
shutil.copy(model_path / "EXAMPLE.BUD", join(model_ws, "EXAMPLE.BUD"))

mp.change_model_ws(model_ws)
mp.write_name_file()
mp.write_input()
success, buff = mp.run_model(silent=True, report=True)
assert success, pformat(buff)
# -

# Read in the endpoint file and plot particles that terminated in the well.

fpth = os.path.join(model_ws, "ex6.mpend")
epobj = flopy.utils.EndpointFile(fpth)
well_epd = epobj.get_destination_endpoint_data(dest_cells=[(4, 12, 12)])
# returns record array of same form as epobj.get_all_data()

well_epd[0:2]

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, aspect="equal")
mapview = flopy.plot.PlotMapView(model=m, layer=2)
quadmesh = mapview.plot_ibound()
linecollection = mapview.plot_grid()
riv = mapview.plot_bc("RIV", color="g", plotAll=True)
quadmesh = mapview.plot_bc("WEL", kper=1, plotAll=True)
contour_set = mapview.contour_array(
    hds, levels=np.arange(np.min(hds), np.max(hds), 0.5), colors="b"
)
plt.clabel(contour_set, inline=1, fontsize=14)
mapview.plot_endpoint(well_epd, direction="starting", colorbar=True)

# Write starting locations to a shapefile.

fpth = os.path.join(model_ws, "starting_locs.shp")
print(type(fpth))
epobj.write_shapefile(
    well_epd, direction="starting", shpname=fpth, mg=m.modelgrid
)

# Read in the pathline file and subset to pathlines that terminated in the well  .

# make a selection of cells that terminate in the well cell = (4, 12, 12)
pthobj = flopy.utils.PathlineFile(os.path.join(model_ws, "ex6.mppth"))
well_pathlines = pthobj.get_destination_pathline_data(dest_cells=[(4, 12, 12)])

# Plot the pathlines that terminate in the well and the starting locations of the particles.

# +
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, aspect="equal")
mapview = flopy.plot.PlotMapView(model=m, layer=2)
quadmesh = mapview.plot_ibound()
linecollection = mapview.plot_grid()
riv = mapview.plot_bc("RIV", color="g", plotAll=True)
quadmesh = mapview.plot_bc("WEL", kper=1, plotAll=True)
contour_set = mapview.contour_array(
    hds, levels=np.arange(np.min(hds), np.max(hds), 0.5), colors="b"
)
plt.clabel(contour_set, inline=1, fontsize=14)

mapview.plot_endpoint(well_epd, direction="starting", colorbar=True)
# for now, each particle must be plotted individually
# (plot_pathline() will plot a single line for recarray with multiple particles)
# for pid in np.unique(well_pathlines.particleid):
#   modelmap.plot_pathline(pthobj.get_data(pid), layer='all', colors='red');
mapview.plot_pathline(well_pathlines, layer="all", colors="red")
# -

# Write pathlines to a shapefile.

# +
# one line feature per particle
pthobj.write_shapefile(
    well_pathlines,
    direction="starting",
    shpname=os.path.join(model_ws, "pathlines.shp"),
    mg=m.modelgrid,
    verbose=False,
)

# one line feature for each row in pathline file
# (can be used to color lines by time or layer in a GIS)
pthobj.write_shapefile(
    well_pathlines,
    one_per_particle=False,
    shpname=os.path.join(model_ws, "pathlines_1per.shp"),
    mg=m.modelgrid,
    verbose=False,
)
# -

# Replace WEL package with MNW2, and create backward tracking simulation using particles released at MNW well.

m2 = flopy.modflow.Modflow.load(
    "EXAMPLE.nam", model_ws=str(model_path), exe_name="mf2005"
)
m2.get_package_list()

m2.nrow_ncol_nlay_nper

m2.wel.stress_period_data.data

# +
node_data = np.array(
    [
        (3, 12, 12, "well1", "skin", -1, 0, 0, 0, 1.0, 2.0, 5.0, 6.2),
        (4, 12, 12, "well1", "skin", -1, 0, 0, 0, 0.5, 2.0, 5.0, 6.2),
    ],
    dtype=[
        ("k", int),
        ("i", int),
        ("j", int),
        ("wellid", object),
        ("losstype", object),
        ("pumploc", int),
        ("qlimit", int),
        ("ppflag", int),
        ("pumpcap", int),
        ("rw", float),
        ("rskin", float),
        ("kskin", float),
        ("zpump", float),
    ],
).view(np.recarray)

stress_period_data = {
    0: np.array(
        [(0, "well1", -150000.0)],
        dtype=[("per", int), ("wellid", object), ("qdes", float)],
    )
}
# -

m2.name = "Example_mnw"
m2.remove_package("WEL")
mnw2 = flopy.modflow.ModflowMnw2(
    model=m2,
    mnwmax=1,
    node_data=node_data,
    stress_period_data=stress_period_data,
    itmp=[1, -1, -1],
)
m2.get_package_list()

# Write and run MODFLOW.

m2.change_model_ws(model_ws)
m2.write_name_file()
m2.write_input()
success, buff = m2.run_model(silent=True, report=True)
assert success, pformat(buff)

# Create a new `Modpath6` object.

# +
mp = flopy.modpath.Modpath6(
    modelname="ex6mnw",
    exe_name="mp6",
    modflowmodel=m2,
    model_ws=model_ws,
)

mpb = flopy.modpath.Modpath6Bas(
    mp, hdry=m2.lpf.hdry, laytyp=m2.lpf.laytyp, ibound=1, prsity=0.1
)
sim = mp.create_mpsim(trackdir="backward", simtype="pathline", packages="MNW2")

mp.change_model_ws(model_ws)
mp.write_name_file()
mp.write_input()
success, buff = mp.run_model(silent=True, report=True)
if success:
    for line in buff:
        print(line)
else:
    raise ValueError("Failed to run.")
# -

# Read in results and plot.

pthobj = flopy.utils.PathlineFile(os.path.join(model_ws, "ex6mnw.mppth"))
epdobj = flopy.utils.EndpointFile(os.path.join(model_ws, "ex6mnw.mpend"))
well_epd = epdobj.get_alldata()
well_pathlines = (
    pthobj.get_alldata()
)  # returns a list of recarrays; one per pathline

# +
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, aspect="equal")
mapview = flopy.plot.PlotMapView(model=m2, layer=2)
quadmesh = mapview.plot_ibound()
linecollection = mapview.plot_grid()
riv = mapview.plot_bc("RIV", color="g", plotAll=True)
quadmesh = mapview.plot_bc("MNW2", kper=1, plotAll=True)
contour_set = mapview.contour_array(
    hds, levels=np.arange(np.min(hds), np.max(hds), 0.5), colors="b"
)
plt.clabel(contour_set, inline=1, fontsize=14)

mapview.plot_pathline(
    well_pathlines, travel_time="<10000", layer="all", colors="red"
)
# -

try:
    # ignore PermissionError on Windows
    temp_dir.cleanup()
except:
    pass
