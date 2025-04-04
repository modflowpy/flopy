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
#     section: viz
#     authors:
#       - name: Christian Langevin
# ---

# # Making Maps of Your Model
# This notebook demonstrates the mapping capabilities of FloPy. It demonstrates these capabilities by loading and running existing models and then showing how the PlotMapView object and its methods can be used to make nice plots of the model grid, boundary conditions, model results, shape files, etc.
#
# ### Mapping is demonstrated for MODFLOW-2005, MODFLOW-USG, and MODFLOW-6 models in this notebook
#


# +
import os
import sys
from pathlib import Path
from pprint import pformat
from shutil import copytree
from tempfile import TemporaryDirectory

import git
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pooch
import shapefile

import flopy

print(sys.version)
print(f"numpy version: {np.__version__}")
print(f"matplotlib version: {mpl.__version__}")
print(f"flopy version: {flopy.__version__}")

# Set name of MODFLOW exe
# assumes executable is in users path statement
v2005 = "mf2005"
exe_name_2005 = "mf2005"
vmf6 = "mf6"
exe_name_mf6 = "mf6"
exe_mp = "mp6"
sim_name = "freyberg"

# Set the paths
tempdir = TemporaryDirectory(delete=False)
modelpth = Path(tempdir.name)

# Check if we are in the repository and define the data path.

try:
    root = Path(git.Repo(".", search_parent_directories=True).working_dir)
except:
    root = None

data_path = root / "examples" / "data" if root else Path.cwd()

# ### Load and Run an Existing MODFLOW-2005 Model
# A model called the "Freyberg Model" is located in the modelpth folder.  In the following code block, we load that model, then change into a new workspace (modelpth) where we recreate and run the model.  For this to work properly, the MODFLOW-2005 executable (mf2005) must be in the path.  We verify that it worked correctly by checking for the presence of freyberg.hds and freyberg.cbc.

file_names = {
    "freyberg.bas": "63266024019fef07306b8b639c6c67d5e4b22f73e42dcaa9db18b5e0f692c097",
    "freyberg.dis": "62d0163bf36c7ee9f7ee3683263e08a0abcdedf267beedce6dd181600380b0a2",
    "freyberg.githds": "abe92497b55e6f6c73306e81399209e1cada34cf794a7867d776cfd18303673b",
    "freyberg.gitlist": "aef02c664344a288264d5f21e08a748150e43bb721a16b0e3f423e6e3e293056",
    "freyberg.lpf": "06500bff979424f58e5e4fbd07a7bdeb0c78f31bd08640196044b6ccefa7a1fe",
    "freyberg.nam": "e66321007bb603ef55ed2ba41f4035ba6891da704a4cbd3967f0c66ef1532c8f",
    "freyberg.oc": "532905839ccbfce01184980c230b6305812610b537520bf5a4abbcd3bd703ef4",
    "freyberg.pcg": "0d1686fac4680219fffdb56909296c5031029974171e25d4304e70fa96ebfc38",
    "freyberg.rch": "37a1e113a7ec16b61417d1fa9710dd111a595de738a367bd34fd4a359c480906",
    "freyberg.riv": "7492a1d5eb23d6812ec7c8227d0ad4d1e1b35631a765c71182b71e3bd6a6d31d",
    "freyberg.wel": "00aa55f59797c02f0be5318a523b36b168fc6651f238f34e8b0938c04292d3e7",
}
for fname, fhash in file_names.items():
    pooch.retrieve(
        url=f"https://github.com/modflowpy/flopy/raw/develop/examples/data/{sim_name}/{fname}",
        fname=fname,
        path=data_path / sim_name,
        known_hash=fhash,
    )

# +
ml = flopy.modflow.Modflow.load(
    "freyberg.nam", model_ws=data_path / sim_name, exe_name=exe_name_2005, version=v2005
)
ml.change_model_ws(modelpth)
ml.write_input()
success, buff = ml.run_model(silent=True, report=True)
assert success, pformat(buff)

files = ["freyberg.hds", "freyberg.cbc"]
for f in files:
    if os.path.isfile(os.path.join(modelpth, f)):
        msg = f"Output file located: {f}"
        print(msg)
    else:
        errmsg = f"Error. Output file cannot be found: {f}"
        print(errmsg)

# ### Create and Run MODPATH 6 model
#
# The MODFLOW-2005 model created in the previous code block will be used to create a endpoint capture zone and pathline analysis for the pumping wells in the model.

# +
mp = flopy.modpath.Modpath6(
    "freybergmp", exe_name=exe_mp, modflowmodel=ml, model_ws=modelpth
)
mpbas = flopy.modpath.Modpath6Bas(
    mp,
    hnoflo=ml.bas6.hnoflo,
    hdry=ml.lpf.hdry,
    ibound=ml.bas6.ibound.array,
    prsity=0.2,
    prsityCB=0.2,
)
sim = mp.create_mpsim(trackdir="forward", simtype="endpoint", packages="RCH")
mp.write_input()
success, buff = mp.run_model(silent=True, report=True)
assert success, pformat(buff)

mpp = flopy.modpath.Modpath6(
    "freybergmpp", exe_name=exe_mp, modflowmodel=ml, model_ws=modelpth
)
mpbas = flopy.modpath.Modpath6Bas(
    mpp,
    hnoflo=ml.bas6.hnoflo,
    hdry=ml.lpf.hdry,
    ibound=ml.bas6.ibound.array,
    prsity=0.2,
    prsityCB=0.2,
)
sim = mpp.create_mpsim(trackdir="backward", simtype="pathline", packages="WEL")
mpp.write_input()
mpp.run_model()

# ### Creating a Map of the Model Grid
# Now that we have a model, we can use the flopy plotting utilities to make maps.  We will start by making a map of the model grid using the `PlotMapView` class and the `plot_grid()` method of that class.

# First step is to set up the plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, aspect="equal")

# Next we create an instance of the PlotMapView class
mapview = flopy.plot.PlotMapView(model=ml)

# Then we can use the plot_grid() method to draw the grid
# The return value for this function is a matplotlib LineCollection object,
# which could be manipulated (or used) later if necessary.
linecollection = mapview.plot_grid()

t = ax.set_title("Model Grid")

# ## Grid transformations and setting coordinate information
#
# The `PlotMapView` class can plot the position of the model grid in space. However, transformations must be done on the modelgrid  using `set_coord_info()`. This allows the user to set the coordinate information once, and then they are able to generate as many instanstances of `PlotMapView` as they wish, without providing the coordinate info again.
#
# Here we demonstrate the effects of these values.  In the first two plots, the grid origin (lower left corner) remains fixed at (0, 0). These first two plots demostrate how work with coordinate info in the `PlotMapView` class. The third example shows the grid origin set at (507000 E, 2927000 N)

fig = plt.figure(figsize=(18, 6))

ax = fig.add_subplot(1, 3, 1, aspect="equal")

# set modelgrid rotation
ml.modelgrid.set_coord_info(angrot=14)

# generate a plot
mapview = flopy.plot.PlotMapView(model=ml)
linecollection = mapview.plot_grid()
t = ax.set_title("rotation=14 degrees")

# re-set the modelgrid rotation
ml.modelgrid.set_coord_info(angrot=-20)

ax = fig.add_subplot(1, 3, 2, aspect="equal")
mapview = flopy.plot.PlotMapView(model=ml)
linecollection = mapview.plot_grid()
t = ax.set_title("rotation=-20 degrees")

# re-set the modelgrid origin and rotation
ml.modelgrid.set_coord_info(xoff=507000, yoff=2927000, angrot=45)

ax = fig.add_subplot(1, 3, 3, aspect="equal")
mapview = flopy.plot.PlotMapView(model=ml)
linecollection = mapview.plot_grid()
t = ax.set_title("xoffset, yoffset, and rotation")

# ### Ploting Ibound
#
# The `plot_ibound()` method can be used to plot the boundary conditions contained in the ibound arrray, which is part of the MODFLOW Basic Package.  The `plot_ibound()` method returns a matplotlib QuadMesh object (matplotlib.collections.QuadMesh).  If you are familiar with the matplotlib collections, then this may be important to you, but if not, then don't worry about the return objects of these plotting function.

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, aspect="equal")

# set the grid rotation and then plot
ml.modelgrid.set_coord_info(angrot=-14)
mapview = flopy.plot.PlotMapView(model=ml)
quadmesh = mapview.plot_ibound()
linecollection = mapview.plot_grid()

# We can also change the colors by calling the `color_noflow` and `color_ch` parameters in `plot_ibound()` and the `colors` parameter in `plot_grid()`

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, aspect="equal")
mapview = flopy.plot.PlotMapView(model=ml)
quadmesh = mapview.plot_ibound(color_noflow="red", color_ch="orange")
linecollection = mapview.plot_grid(colors="yellow")

# ### Plotting Boundary Conditions
# The plot_bc() method can be used to plot boundary conditions.  It is setup to use the following dictionary to assign colors, however, these colors can be changed in the method call.
#
#     bc_color_dict = {'default': 'black', 'WEL': 'red', 'DRN': 'yellow',
#                  'RIV': 'green', 'GHB': 'cyan', 'CHD': 'navy'}
#
# Here, we plot the location of river cells and the location of well cells.

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, aspect="equal")
mapview = flopy.plot.PlotMapView(model=ml)
quadmesh = mapview.plot_ibound()
quadmesh = mapview.plot_bc("RIV")
quadmesh = mapview.plot_bc("WEL")
linecollection = mapview.plot_grid()

# The colors can be changed by using the `color_noflow` and `color_ch` parameters in `plot_ibound()`, the `color` parameter in `plot_bc()`, and the `colors` parameter in `plot_grid()`

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, aspect="equal")
mapview = flopy.plot.PlotMapView(model=ml)
quadmesh = mapview.plot_ibound(color_noflow="red", color_ch="orange")
quadmesh = mapview.plot_bc("RIV", color="purple")
quadmesh = mapview.plot_bc("WEL", color="navy")
linecollection = mapview.plot_grid(colors="yellow")

# ### Plotting an Array
#
# `PlotMapView` has a `plot_array()` method.  The `plot_array()` method will accept either a 2D or 3D array.  If a 3D array is passed, then the `layer` parameter for the `PlotMapView` object will be used (note that the `PlotMapView` object can be created with a `layer=` argument).

# Create a random array and plot it
a = np.random.random((ml.dis.nrow, ml.dis.ncol))

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, aspect="equal")
ax.set_title("Random Array")
mapview = flopy.plot.PlotMapView(model=ml, layer=0)
quadmesh = mapview.plot_array(a)
linecollection = mapview.plot_grid()
cb = plt.colorbar(quadmesh, shrink=0.5)

# Plot the model bottom array
a = ml.dis.botm.array

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, aspect="equal")
ax.set_title("Model Bottom Elevations")
mapview = flopy.plot.PlotMapView(model=ml, layer=0)
quadmesh = mapview.plot_array(a)
linecollection = mapview.plot_grid()
cb = plt.colorbar(quadmesh, shrink=0.5)

# ### Contouring an Array
#
# `PlotMapView` also has a `contour_array()` method.  It also takes a 2D or 3D array and will contour the layer slice if 3D.

# Contour the model bottom array
a = ml.dis.botm.array

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, aspect="equal")
ax.set_title("Model Bottom Elevations")
mapview = flopy.plot.PlotMapView(model=ml, layer=0)
contour_set = mapview.contour_array(a)
linecollection = mapview.plot_grid()

plt.colorbar(contour_set, shrink=0.75)

# The contour_array() method will take any keywords
# that can be used by the matplotlib.pyplot.contour
# function. So we can pass in levels, for example.
a = ml.dis.botm.array
levels = np.arange(0, 20, 0.5)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, aspect="equal")
ax.set_title("Model Bottom Elevations")
mapview = flopy.plot.PlotMapView(model=ml, layer=0)
contour_set = mapview.contour_array(a, levels=levels)
linecollection = mapview.plot_grid()

# set up and plot a continuous colorbar in matplotlib for a contour plot
norm = mpl.colors.Normalize(
    vmin=contour_set.cvalues.min(), vmax=contour_set.cvalues.max()
)
sm = plt.cm.ScalarMappable(norm=norm, cmap=contour_set.cmap)
sm.set_array([])
fig.colorbar(sm, shrink=0.75, ax=ax)

# Array contours can be exported directly to a shapefile.

from flopy.export.utils import (  # use export_contourf for filled contours
    export_contours,
)

shp_path = os.path.join(modelpth, "contours.shp")
export_contours(shp_path, contour_set)

from shapefile import Reader

with Reader(shp_path) as r:
    nshapes = len(r.shapes())
    print("Contours:", nshapes)

# ### Plotting Heads
#
# So this means that we can easily plot results from the simulation by extracting heads using `flopy.utils.HeadFile`.  Here we plot the simulated heads.

fname = os.path.join(modelpth, "freyberg.hds")
hdobj = flopy.utils.HeadFile(fname)
head = hdobj.get_data()
levels = np.arange(10, 30, 0.5)

fig = plt.figure(figsize=(15, 10))

ax = fig.add_subplot(1, 2, 1, aspect="equal")
ax.set_title("plot_array()")
mapview = flopy.plot.PlotMapView(model=ml)
quadmesh = mapview.plot_ibound()
quadmesh = mapview.plot_array(head, alpha=0.5)
mapview.plot_bc("WEL")
linecollection = mapview.plot_grid()

ax = fig.add_subplot(1, 2, 2, aspect="equal")
ax.set_title("contour_array()")
mapview = flopy.plot.PlotMapView(model=ml)
quadmesh = mapview.plot_ibound()
mapview.plot_bc("WEL")
contour_set = mapview.contour_array(head, levels=levels)
linecollection = mapview.plot_grid()

# ### Plotting Discharge Vectors
#
# `PlotMapView` has a `plot_vector()` method, which takes vector components in the x- and y-directions at the cell centers. The x- and y-vector components are calculated from the `'FLOW RIGHT FACE'` and `'FLOW FRONT FACE'` arrays, which can be written by MODFLOW to the cell by cell budget file.  These array can be extracted from the cell by cell flow file using the `flopy.utils.CellBudgetFile` object as shown below.  Once they are extracted, they can be passed to the `postprocessing.get_specific_discharge()` method to get the discharge vectors and plotted using the `plot_vector()` method.
#
# **Note**: `postprocessing.get_specific_discharge()` also takes the head array as an optional argument.  The head array is used to convert the volumetric discharge in dimensions of $L^3/T$ to specific discharge in dimensions of $L/T$.

fname = os.path.join(modelpth, "freyberg.cbc")
cbb = flopy.utils.CellBudgetFile(fname)
head = hdobj.get_data()
frf = cbb.get_data(text="FLOW RIGHT FACE")[0]
fff = cbb.get_data(text="FLOW FRONT FACE")[0]
flf = None

qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(
    (frf, fff, None), ml
)  # no head array for volumetric discharge
sqx, sqy, sqz = flopy.utils.postprocessing.get_specific_discharge(
    (frf, fff, None), ml, head
)

fig = plt.figure(figsize=(15, 10))

ax = fig.add_subplot(1, 2, 1, aspect="equal")
ax.set_title("Volumetric discharge (" + r"$L^3/T$" + ")")
mapview = flopy.plot.PlotMapView(model=ml)
quadmesh = mapview.plot_ibound()
quadmesh = mapview.plot_array(head, alpha=0.5)
quiver = mapview.plot_vector(qx, qy)
linecollection = mapview.plot_grid()

ax = fig.add_subplot(1, 2, 2, aspect="equal")
ax.set_title("Specific discharge (" + r"$L/T$" + ")")
mapview = flopy.plot.PlotMapView(model=ml)
quadmesh = mapview.plot_ibound()
quadmesh = mapview.plot_array(head, alpha=0.5)
quiver = mapview.plot_vector(sqx, sqy)  # include the head array for specific discharge
linecollection = mapview.plot_grid()

# ### Plotting MODPATH endpoints and pathlines
#
# `PlotMapView` has a `plot_endpoint()` and `plot_pathline()` method, which takes MODPATH endpoint and pathline data and plots them on the map object. Here we load the endpoint and pathline data and plot them on the head and discharge data previously plotted. Pathlines are shown for all times less than or equal to 200 years. Recharge capture zone data for all of the pumping wells are plotted as circle markers colored by travel time.

# load the endpoint data
endfile = os.path.join(modelpth, mp.sim.endpoint_file)
endobj = flopy.utils.EndpointFile(endfile)
ept = endobj.get_alldata()

# load the pathline data
pthfile = os.path.join(modelpth, mpp.sim.pathline_file)
pthobj = flopy.utils.PathlineFile(pthfile)
plines = pthobj.get_alldata()

# plot the data
fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(1, 1, 1, aspect="equal")
ax.set_title("plot_array()")
mapview = flopy.plot.PlotMapView(model=ml)
quadmesh = mapview.plot_ibound()
quadmesh = mapview.plot_array(head, alpha=0.5)
quiver = mapview.plot_vector(sqx, sqy)
linecollection = mapview.plot_grid()
for d in ml.wel.stress_period_data[0]:
    mapview.plot_endpoint(
        ept,
        direction="starting",
        selection_direction="ending",
        selection=(d[0], d[1], d[2]),
        zorder=100,
    )

# construct maximum travel time to plot (200 years - MODFLOW time unit is seconds)
travel_time_max = 200.0 * 365.25 * 24.0 * 60.0 * 60.0
ctt = f"<={travel_time_max}"

# plot the pathlines
mapview.plot_pathline(plines, layer="all", colors="red", travel_time=ctt)

# ### Plotting a Shapefile
#
# `PlotMapView` has a `plot_shapefile()` method that can be used to quickly plot a shapefile on your map.  In order to use the `plot_shapefile()` method, you must be able to  "import shapefile".  The command `import shapefile` is part of the pyshp package.
#
# The `plot_shapefile()` function can plot points, lines, and polygons and will return a patch_collection of objects from the shapefile.  For a shapefile of polygons, the `plot_shapefile()` function will try to plot and fill them all using a different color.  For a shapefile of points, you may need to specify a radius, in model units, in order for the circles to show up properly.
#
# The shapefile must have intersecting geographic coordinates as the `PlotMapView` object in order for it to overlay correctly on the plot.  The `plot_shapefile()` method and function do not use any of the projection information that may be stored with the shapefile.  If you reset `xoff`, `yoff`, and `angrot` in the `ml.modelgrid.set_coord_info()` call below, you will see that the grid will no longer overlay correctly with the shapefile.

file_names = {
    "bedrock_outcrop_hole.dbf": "c48510bc0b04405e4d3433e6cd892351c8342a7c46215f48332a7e6292249da6",
    "bedrock_outcrop_hole.sbn": "48fd1496d84822c9637d7f3065edf4dfa2038406be8fa239cb451b1a3b28127c",
    "bedrock_outcrop_hole.sbx": "9a36aee5f3a4bcff0a453ab743a7523ea19acb8841e8273bbda34f27d7237ea5",
    "bedrock_outcrop_hole.shp": "25c241ac90dd47be28f761ba60ba94a511744f5219600e35a80a93f19ec99f97",
    "bedrock_outcrop_hole.shx": "88b06395fa4c58ea04d300e10e6f6ea81e17fb0baa20d8ac78470d19101430be",
    "bedrock_outcrop_hole_rotate14.dbf": "e05bbfc826fc069666a05e949acc833b54de51b14267c9c54b1c129b4a8ab82d",
    "bedrock_outcrop_hole_rotate14.sbn": "136d8f86b8a13abc8f0386108228ca398037cf8c28ba6077086fd7e1fd54abf7",
    "bedrock_outcrop_hole_rotate14.sbx": "1c2f2f2791db9c752fb1b355f13e46a8740ccd66654ae34d130172a3bdcda805",
    "bedrock_outcrop_hole_rotate14.shp": "3e722d8fa9331ab498dbf9544085b30f60d2e38cc82a0955792d11a4e6a4419d",
    "bedrock_outcrop_hole_rotate14.shp.xml": "ff6a3e80d10d9e68863ffe224e8130b862c13c2265d3a604342eb20a700d38fd",
    "bedrock_outcrop_hole_rotate14.shx": "32a75461fab39b21769c474901254e7cbd24073c53d62b494fd70080cfcd3383",
    "cross_section.cpg": "3ad3031f5503a4404af825262ee8232cc04d4ea6683d42c5dd0a2f2a27ac9824",
    "cross_section.dbf": "3b050b1d296a7efe1b4f001c78030d5c81f79d3cd101d459e4426944fbd4e8e7",
    "cross_section.sbn": "3b6a8f72f78f7b0d12e5823d6e8307040cfd5af88a8fb9427687d027aa805126",
    "cross_section.sbx": "72e33139aaa99a8d12922af3774bd6b1a73613fc1bc852d1a1d1426ef48a832a",
    "cross_section.shp": "0eb9e37dcbdbb5d932101c4c5bcb971271feb2c1d81d2a5f8dbc0fbf8d799ee5",
    "cross_section.shp.xml": "ff99002ecd63a843fe628c107dfb02926b6838132c6f503db38b792644fb368e",
    "cross_section.shx": "c6fa1307e1c32c535842796b24b2a0a07865065ace3324b0f6b1b71e9c1a8e1e",
    "cross_section_rotate14.cpg": "3ad3031f5503a4404af825262ee8232cc04d4ea6683d42c5dd0a2f2a27ac9824",
    "cross_section_rotate14.dbf": "72f8ed25c45a92822fe593862e543ae4167357cbc8fba4f24b889aa2bbf2729a",
    "cross_section_rotate14.sbn": "3f7a3b66cf58be8c979353d2c75777303035e19ff58d96a089dde5c95fa8b597",
    "cross_section_rotate14.sbx": "7d40bc92b42fde2af01a2805c9205c18c0fe89ae7cf1ba88ac6627b7c6a69b89",
    "cross_section_rotate14.shp": "5f0ea7a65b5ddc9a43c874035969e30d58ae578aec9feb6b0e8538b68d5bd0d2",
    "cross_section_rotate14.shp.xml": "79e38d9542ce764ace47883c673cf1d9aab16cd7851ae62a8e9bf27ce1091e13",
    "cross_section_rotate14.shx": "b750b9d44ef31e0c593e2f78acfc08813667bb73733e6524f1b417e605cae65d",
    "model_extent.cpg": "3ad3031f5503a4404af825262ee8232cc04d4ea6683d42c5dd0a2f2a27ac9824",
    "model_extent.dbf": "72f8ed25c45a92822fe593862e543ae4167357cbc8fba4f24b889aa2bbf2729a",
    "model_extent.sbn": "622376387ac9686e54acc6c57ace348c217d3a82e626274f32911a1d0006a164",
    "model_extent.sbx": "2957bc1b5c918e20089fb6f6998d60d4488995d174bac21afa8e3a2af90b3489",
    "model_extent.shp": "c72d5a4c703100e98c356c7645ad4b0bcc124c55e0757e55c8cd8663c7bf15c6",
    "model_extent.shx": "e8d3b5618f0c248b59284f4f795f5de8207aec5b15ed60ce8da5a021c1043e2f",
    "wells_locations.dbf": "965c846ec0b8f0d27570ef0bdaadfbcb6e718ed70ab89c8dda01d3b819e7a7de",
    "wells_locations.sbn": "63f8ad670c6ba53ddec13069e42cfd86f27b6d47c5d0b3f2c25dfd6fb6b55825",
    "wells_locations.sbx": "8420907d426c44c38315a5bdc0b24fe07a8cd2cc9a7fc60b817500b8cda79a34",
    "wells_locations.shp": "ee53a4532b513f5b8bcd37ee3468dc4b2c8f6afab6cfc5110d74362c79e52287",
    "wells_locations.shx": "6e816e96ed0726c2acc61392d2a82df5e9265ab5f5b00dd12f765b139840be79",
    "wells_locations_rotate14.dbf": "d9b3636b4312c2f76c837e698bcb0d8ef9f4bbaa1765c484787a9f9d7f8bbaae",
    "wells_locations_rotate14.sbn": "b436e34b8f145966b18d571b47ebc18e35671ec73fca1abbc737d9e1aa984bfb",
    "wells_locations_rotate14.sbx": "24911f8905155882ce76b0330c9ba5ed449ca985d46833ebc45eee11faabbdaf",
    "wells_locations_rotate14.shp": "695894af4678358320fb914e872cadb2613fae2e54c2d159f40c02fa558514cf",
    "wells_locations_rotate14.shp.xml": "288183eb273c1fc2facb49d51c34bcafb16710189242da48f7717c49412f3e29",
    "wells_locations_rotate14.shx": "da3374865cbf864f81dd69192ab616d1093d2159ac3c682fe2bfc4c295a28e42",
}
for fname, fhash in file_names.items():
    pooch.retrieve(
        url=f"https://github.com/modflowpy/flopy/raw/develop/examples/data/{sim_name}/gis/{fname}",
        fname=fname,
        path=data_path / sim_name / "gis",
        known_hash=fhash,
    )

copytree(data_path / sim_name / "gis", modelpth / "gis")
assert (modelpth / "gis").is_dir()

# Setup the figure and PlotMapView. Show a very faint map of ibound and
# model grid by specifying a transparency alpha value.
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, aspect="equal")

assert (modelpth / "gis").is_dir()

# reset the grid rotation and offsets to 0
ml.modelgrid.set_coord_info(xoff=0, yoff=0, angrot=0)

mapview = flopy.plot.PlotMapView(model=ml, ax=ax)

# Plot a shapefile of
assert (modelpth / "gis").is_dir()
shp = os.path.join(modelpth, "gis", "bedrock_outcrop_hole")
print(os.listdir(modelpth / "gis"))
patch_collection = mapview.plot_shapefile(
    shp,
    edgecolor="green",
    linewidths=2,
    alpha=0.5,  # facecolor='none',
)
# Plot a shapefile of a cross-section line
shp = os.path.join(modelpth, "gis", "cross_section")
patch_collection = mapview.plot_shapefile(
    shp, radius=0, lw=[3, 1.5], edgecolor=["red", "green"], facecolor="None"
)

# Plot a shapefile of well locations
shp = os.path.join(modelpth, "gis", "wells_locations")
patch_collection = mapview.plot_shapefile(shp, radius=100, facecolor="red")

# Plot the grid and boundary conditions over the top
quadmesh = mapview.plot_ibound(alpha=0.1)
quadmesh = mapview.plot_bc("RIV", alpha=0.1)
linecollection = mapview.plot_grid(alpha=0.1)

# Although the `PlotMapView`'s `plot_shapefile()` method does not consider projection information when plotting maps, it can be used to plot shapefiles when a `PlotMapView` instance is rotated and offset into geographic coordinates. The same shapefiles plotted above (but in geographic coordinates rather than model coordinates) are plotted on the rotated model grid. The offset from model coordinates to geographic coordinates relative to the lower left corner are `xoff=-2419.22`, `yoff=297.04` and the rotation angle is 14$^{\circ}$.

# Setup the figure and PlotMapView. Show a very faint map of ibound and
# model grid by specifying a transparency alpha value.

# set the modelgrid rotation and offset
ml.modelgrid.set_coord_info(
    xoff=-2419.2189559966773, yoff=297.0427372400354, angrot=-14
)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, aspect="equal")
mapview = flopy.plot.PlotMapView(model=ml)

# Plot a shapefile of
shp = os.path.join(modelpth, "gis", "bedrock_outcrop_hole_rotate14")
patch_collection = mapview.plot_shapefile(
    shp,
    edgecolor="green",
    linewidths=2,
    alpha=0.5,  # facecolor='none',
)
# Plot a shapefile of a cross-section line
shp = os.path.join(modelpth, "gis", "cross_section_rotate14")
patch_collection = mapview.plot_shapefile(shp, lw=3, edgecolor="red")

# Plot a shapefile of well locations
shp = os.path.join(modelpth, "gis", "wells_locations_rotate14")
patch_collection = mapview.plot_shapefile(shp, radius=100, facecolor="red")

# Plot the grid and boundary conditions over the top
quadmesh = mapview.plot_ibound(alpha=0.1)
linecollection = mapview.plot_grid(alpha=0.1)

# ### Plotting GIS Shapes
#
# `PlotMapView` has a `plot_shapes()` method that can be used to quickly plot GIS based shapes on your map. In order to use the `plot_shapes()` method, you must be able to "import shapefile". The command import shapefile is part of the pyshp package.
#
# The `plot_shapes()` function can plot points, lines, polygons, and multipolygons and will return a patch_collection. For a list or collection of polygons, the `plot_shapes()` function will try to plot and fill them all using a different color. For a list or collection of points, you may need to specify a radius, in model units, in order for the circles to show up properly.
#
# __Note:__ The supplied shapes must have intersecting geographic coordinates as the `PlotMapView` object in order for it to overlay correctly on the plot.
#
# `plot_shapes()` supports many GIS based input types and they are listed below:
#    + list of shapefile.Shape objects
#    + shapefile.Shapes object
#    + list of flopy.utils.geometry objects
#    + flopy.utils.geometry.Collection object
#    + list of geojson geometry objects
#    + list of geojson.Feature objects
#    + geojson.GeometryCollection object
#    + geojson.FeatureCollection object
#    + list of shapely geometry objects
#    + shapely.GeometryCollection object
#
# Here is a basic example of how to use the method:

# lets extract some shapes from our shapefiles
shp = os.path.join(modelpth, "gis", "bedrock_outcrop_hole_rotate14")
with shapefile.Reader(shp) as r:
    polygon_w_hole = [r.shape(0)]

shp = os.path.join(modelpth, "gis", "cross_section_rotate14")
with shapefile.Reader(shp) as r:
    cross_section = r.shapes()

# Plot a shapefile of well locations
shp = os.path.join(modelpth, "gis", "wells_locations_rotate14")
with shapefile.Reader(shp) as r:
    wells = r.shapes()

# Now that the shapes are extracted from the shapefiles, they can be plotted using `plot_shapes()`

# set the modelgrid rotation and offset
ml.modelgrid.set_coord_info(
    xoff=-2419.2189559966773, yoff=297.0427372400354, angrot=-14
)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, aspect="equal")
mapview = flopy.plot.PlotMapView(model=ml)

# Plot the grid and boundary conditions
quadmesh = mapview.plot_ibound()
linecollection = mapview.plot_grid()

# plot polygon(s)
patch_collection0 = mapview.plot_shapes(
    polygon_w_hole, edgecolor="orange", linewidths=2, alpha=0.5
)

# plot_line(s)
patch_collection1 = mapview.plot_shapes(cross_section, lw=3, edgecolor="red")

# plot_point(s)
patch_collection3 = mapview.plot_shapes(wells, radius=100, facecolor="k", edgecolor="k")

# ## Working with MODFLOW-6 models
#
# `PlotMapView` has support for MODFLOW-6 models and operates in the same fashion for Structured Grids, Vertex Grids, and Unstructured Grids. Here is a short example on how to plot with MODFLOW-6 structured grids using a version of the Freyberg model created for MODFLOW-6

# load the Freyberg model into mf6-flopy and run the simulation

sim_name = "mf6-freyberg"
sim_path = modelpth / "mf6"
file_names = {
    "bot.asc": "3107f907cb027460fd40ffc16cb797a78babb31988c7da326c9f500fba855b62",
    "description.txt": "94093335eec6a24711f86d4d217ccd5a7716dd9e01cb6b732bc7757d41675c09",
    "freyberg.cbc": "c8ad843b1da753eb58cf6c462ac782faf0ca433d6dcb067742d8bd698db271e3",
    "freyberg.chd": "d8b8ada8d3978daea1758b315be983b5ca892efc7d69bf6b367ceec31e0dd156",
    "freyberg.dis": "cac230a207cc8483693f7ba8ae29ce40c049036262eac4cebe17a4e2347a8b30",
    "freyberg.dis.grb": "c8c26fb1fa4b210208134b286d895397cf4b3131f66e1d9dda76338502c7e96a",
    "freyberg.hds": "926a06411ca658a89db6b5686f51ddeaf5b74ced81239cab1d43710411ba5f5b",
    "freyberg.ic": "6efb56ee9cdd704b9a76fb9efd6dae750facc5426b828713f2d2cf8d35194120",
    "freyberg.ims": "6dddae087d85417e3cdaa13e7b24165afb7f9575ab68586f3adb6c1b2d023781",
    "freyberg.nam": "cee9b7b000fe35d2df26e878d09d465250a39504f87516c897e3fa14dcda081e",
    "freyberg.npf": "81104d3546045fff0eddf5059465e560b83b492fa5a5acad1907ce18c2b9c15f",
    "freyberg.oc": "c0715acd75eabcc42c8c47260a6c1abd6c784350983f7e2e6009ddde518b80b8",
    "freyberg.rch": "a6ec1e0eda14fd2cdf618a5c0243a9caf82686c69242b783410d5abbcf971954",
    "freyberg.riv": "a8cafc8c317cbe2acbb43e2f0cfe1188cb2277a7a174aeb6f3e6438013de8088",
    "freyberg.sto": "74d748c2f0adfa0a32ee3f2912115c8f35b91011995b70c1ec6ae1c627242c41",
    "freyberg.tdis": "9965cbb17caf5b865ea41a4ec04bcb695fe15a38cb539425fdc00abbae385cbe",
    "freyberg.wel": "f19847de455598de52c05a4be745698c8cb589e5acfb0db6ab1f06ded5ff9310",
    "k11.asc": "b6a8aa46ef17f7f096d338758ef46e32495eb9895b25d687540d676744f02af5",
    "mfsim.nam": "6b8d6d7a56c52fb2bff884b3979e3d2201c8348b4bbfd2b6b9752863cbc9975e",
    "top.asc": "3ad2b131671b9faca7f74c1dd2b2f41875ab0c15027764021a89f9c95dccaa6a",
}
for fname, fhash in file_names.items():
    pooch.retrieve(
        url=f"https://github.com/modflowpy/flopy/raw/develop/examples/data/{sim_name}/{fname}",
        fname=fname,
        path=data_path / sim_name,
        known_hash=fhash,
    )

sim = flopy.mf6.MFSimulation.load(
    sim_name="mfsim.nam",
    version=vmf6,
    exe_name=exe_name_mf6,
    sim_ws=data_path / sim_name,
)
sim.set_sim_path(sim_path)
sim.write_simulation()
success, buff = sim.run_simulation()
if not success:
    print("Something bad happened.")
files = ["freyberg.hds", "freyberg.cbc"]
for f in files:
    if os.path.isfile(os.path.join(modelpth, f)):
        msg = f"Output file located: {f}"
        print(msg)
    else:
        errmsg = f"Error. Output file cannot be found: {f}"
        print(errmsg)

# ### Plotting boundary conditions and arrays
#
# This works the same as modflow-2005, however the simulation object can host a number of modflow-6 models so we need to grab a model before attempting to plot with `PlotMapView`

# get the modflow-6 model we want to plot
ml6 = sim.get_model("freyberg")
ml6.modelgrid.set_coord_info(angrot=-14)

fig = plt.figure(figsize=(15, 10))

# plot boundary conditions
ax = fig.add_subplot(1, 2, 1, aspect="equal")
mapview = flopy.plot.PlotMapView(model=ml6)
quadmesh = mapview.plot_ibound()
quadmesh = mapview.plot_bc("RIV")
quadmesh = mapview.plot_bc("WEL")
linecollection = mapview.plot_grid()
ax.set_title("Plot boundary conditions")

# plot model bottom elevations
a = ml6.dis.botm.array

ax = fig.add_subplot(1, 2, 2, aspect="equal")
ax.set_title("Model Bottom Elevations")
mapview = flopy.plot.PlotMapView(model=ml6, layer=0)
quadmesh = mapview.plot_array(a)
inactive = mapview.plot_inactive()
linecollection = mapview.plot_grid()
cb = plt.colorbar(quadmesh, shrink=0.5, ax=ax)

# ### Contouring Arrays
#
# Contouring arrays follows the same code signature for MODFLOW-6 as the MODFLOW-2005 example. Just use the `contour_array()` method

# The contour_array() method will take any keywords
# that can be used by the matplotlib.pyplot.contour
# function. So we can pass in levels, for example.
a = ml6.dis.botm.array
levels = np.arange(0, 20, 0.5)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, aspect="equal")
ax.set_title("Model Bottom Elevations")
mapview = flopy.plot.PlotMapView(model=ml6, layer=0)
contour_set = mapview.contour_array(a, levels=levels)
linecollection = mapview.plot_grid()

# set up and plot a continuous colorbar in matplotlib for a contour plot
norm = mpl.colors.Normalize(
    vmin=contour_set.cvalues.min(), vmax=contour_set.cvalues.max()
)
sm = plt.cm.ScalarMappable(norm=norm, cmap=contour_set.cmap)
sm.set_array([])
fig.colorbar(sm, shrink=0.75, ax=ax)

# ### Plotting specific discharge with a MODFLOW-6 model
#
# MODFLOW-6 includes a the PLOT_SPECIFIC_DISCHARGE flag in the NPF package to calculate and store discharge vectors for easy plotting. The postprocessing module will translate the specific dischage into vector array and `PlotMapView` has the `plot_vector()` method to use this data. The specific discharge array is stored in the cell budget file.

# get the specific discharge from the cell budget file
cbc_file = os.path.join(sim_path, "freyberg.cbc")
cbc = flopy.utils.CellBudgetFile(cbc_file)
spdis = cbc.get_data(text="SPDIS")[0]

qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(spdis, ml6)

# get the head from the head file
head_file = os.path.join(sim_path, "freyberg.hds")
head = flopy.utils.HeadFile(head_file)
hdata = head.get_alldata()[0]

# plot specific discharge using PlotMapView
fig = plt.figure(figsize=(8, 8))

mapview = flopy.plot.PlotMapView(model=ml6, layer=0)
linecollection = mapview.plot_grid()
quadmesh = mapview.plot_array(a=hdata, alpha=0.5)
quiver = mapview.plot_vector(qx, qy)
inactive = mapview.plot_inactive()

plt.title("Specific Discharge (" + r"$L/T$" + ")")
plt.colorbar(quadmesh, shrink=0.75)

# ## Vertex model plotting with MODFLOW-6
#
# FloPy fully supports vertex discretization (DISV) plotting through the `PlotMapView` class. The method calls are identical to the ones presented previously for Structured discretization (DIS) and the same matplotlib keyword arguments are supported. Let's run through an example using a vertex model grid.

# build and run vertex model grid demo problem


def run_vertex_grid_example(ws):
    """load and run vertex grid example"""
    if not os.path.exists(ws):
        os.mkdir(ws)

    from flopy.utils.gridgen import Gridgen

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

    model_name = "mp7p2"
    model_ws = os.path.join(ws, "mp7_ex2", "mf6")
    gridgen_ws = os.path.join(model_ws, "gridgen")
    g = Gridgen(ms.modelgrid, model_ws=gridgen_ws)

    rf0shp = os.path.join(gridgen_ws, "rf0")
    xmin = 7 * delr
    xmax = 12 * delr
    ymin = 8 * delc
    ymax = 13 * delc
    rfpoly = [[[(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)]]]
    g.add_refinement_features(rfpoly, "polygon", 1, range(nlay))

    rf1shp = os.path.join(gridgen_ws, "rf1")
    xmin = 8 * delr
    xmax = 11 * delr
    ymin = 9 * delc
    ymax = 12 * delc
    rfpoly = [[[(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)]]]
    g.add_refinement_features(rfpoly, "polygon", 2, range(nlay))

    rf2shp = os.path.join(gridgen_ws, "rf2")
    xmin = 9 * delr
    xmax = 10 * delr
    ymin = 10 * delc
    ymax = 11 * delc
    rfpoly = [[[(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)]]]
    g.add_refinement_features(rfpoly, "polygon", 3, range(nlay))

    g.build(verbose=False)

    gridprops = g.get_gridprops_disv()
    ncpl = gridprops["ncpl"]
    top = gridprops["top"]
    botm = gridprops["botm"]
    nvert = gridprops["nvert"]
    vertices = gridprops["vertices"]
    cell2d = gridprops["cell2d"]
    # cellxy = gridprops['cellxy']

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
        sim, modelname=model_name, model_nam_file=f"{model_name}.nam"
    )
    gwf.name_file.save_flows = True

    # create iterative model solution and register the gwf model with it
    ims = flopy.mf6.ModflowIms(
        sim,
        pname="ims",
        print_option="SUMMARY",
        complexity="SIMPLE",
        outer_hclose=1.0e-5,
        outer_maximum=100,
        under_relaxation="NONE",
        inner_maximum=100,
        inner_hclose=1.0e-6,
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
        save_specific_discharge=True,
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
        gwf,
        print_input=True,
        auxiliary=[("iface",)],
        stress_period_data=welspd,
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
        budget_filerecord=f"{model_name}.cbb",
        head_filerecord=f"{model_name}.hds",
        headprintrecord=[("COLUMNS", 10, "WIDTH", 15, "DIGITS", 6, "GENERAL")],
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        printrecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )

    sim.write_simulation()
    success, buff = sim.run_simulation(silent=True, report=True)
    if success:
        for line in buff:
            print(line)
    else:
        raise ValueError("Failed to run.")

    mp_namea = f"{model_name}a_mp"
    mp_nameb = f"{model_name}b_mp"

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
    fpth = f"{mp_namea}.sloc"
    pga = flopy.modpath.ParticleGroup(
        particlegroupname="BACKWARD1", particledata=pa, filename=fpth
    )

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
    fpth = f"{mp_nameb}.sloc"
    pgb = flopy.modpath.ParticleGroupNodeTemplate(
        particlegroupname="BACKWARD2", particledata=pb, filename=fpth
    )

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
    if success:
        for line in buff:
            print(line)
    else:
        raise ValueError("Failed to run.")

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
    assert success, pformat(buff)


run_vertex_grid_example(modelpth)

# check if model ran properly
mp7modelpth = os.path.join(modelpth, "mp7_ex2", "mf6")
files = ["mp7p2.hds", "mp7p2.cbb"]
for f in files:
    if os.path.isfile(os.path.join(mp7modelpth, f)):
        msg = f"Output file located: {f}"
        print(msg)
    else:
        errmsg = f"Error. Output file cannot be found: {f}"
        print(errmsg)

# load the simulation and get the model
vertex_sim_name = "mfsim.nam"
vertex_sim = flopy.mf6.MFSimulation.load(
    sim_name=vertex_sim_name,
    version=vmf6,
    exe_name=exe_name_mf6,
    sim_ws=mp7modelpth,
)
vertex_ml6 = vertex_sim.get_model("mp7p2")

# ### Setting MODFLOW-6 Vertex Model Grid offsets, rotation and plotting
#
# Setting the `Grid` offsets and rotation is consistent in FloPy, no matter which type of discretization the user is using. The `set_coord_info()` method on the `modelgrid` is used.
#
# Plotting works consistently too, the user just calls the `PlotMapView` class and it accounts for the discretization type

# set coordinate information on the modelgrid
vertex_ml6.modelgrid.set_coord_info(xoff=362100, yoff=4718900, angrot=-21)

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(1, 1, 1, aspect="equal")
ax.set_title("Vertex Model Grid (DISV)")

# use PlotMapView to plot a DISV (vertex) model
mapview = flopy.plot.PlotMapView(vertex_ml6, layer=0)
linecollection = mapview.plot_grid()

# ### Plotting boundary conditions with Vertex Model grids
#
# The `plot_bc()` method can be used to plot boundary conditions.  It is setup to use the following dictionary to assign colors, however, these colors can be changed in the method call.
#
#     bc_color_dict = {'default': 'black', 'WEL': 'red', 'DRN': 'yellow',
#                      'RIV': 'green', 'GHB': 'cyan', 'CHD': 'navy'}
#
# Here we plot river (RIV) cell locations

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(1, 1, 1, aspect="equal")
ax.set_title("Vertex Model Grid (DISV)")

# use PlotMapView to plot a DISV (vertex) model
mapview = flopy.plot.PlotMapView(vertex_ml6, layer=0)
riv = mapview.plot_bc("RIV")
linecollection = mapview.plot_grid()

# ### Plotting Arrays and Contouring with Vertex Model grids
#
# `PlotMapView` allows the user to plot arrays and contour with DISV based discretization. The `plot_array()` method is called in the same way as using a structured grid. The only difference is that `PlotMapView` builds a matplotlib patch collection for Vertex based grids.

# get the head output for stress period 1 from the modflow6 head file
head = flopy.utils.HeadFile(os.path.join(mp7modelpth, "mp7p2.hds"))
hdata = head.get_alldata()[0, :, :, :]

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(1, 1, 1, aspect="equal")
ax.set_title("plot_array()")

mapview = flopy.plot.PlotMapView(model=vertex_ml6, layer=2)
patch_collection = mapview.plot_array(hdata, cmap="Dark2")
linecollection = mapview.plot_grid(lw=0.25, color="k")
cb = plt.colorbar(patch_collection, shrink=0.75)

# The `contour_array()` method operates in the same way as the sturctured example.

# plotting head array and then contouring the array!
levels = np.arange(327, 332, 0.5)

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(1, 1, 1, aspect="equal")
ax.set_title("Model head contours, layer 3")

mapview = flopy.plot.PlotMapView(model=vertex_ml6, layer=2)
pc = mapview.plot_array(hdata, cmap="Dark2")

# contouring the head array
contour_set = mapview.contour_array(hdata, levels=levels, colors="white")
plt.clabel(contour_set, fmt="%.1f", colors="white", fontsize=11)
linecollection = mapview.plot_grid(lw=0.25, color="k")

cb = plt.colorbar(pc, shrink=0.75, ax=ax)

# ### Plotting MODPATH 7 results on a vertex model
#
# MODPATH-7 results can be plotted using the same built in methods as used previously to plot MODPATH-6 results. The `plot_pathline()` and `plot_timeseries()` methods are layered on the previous example to show modpath simulation results

# load the MODPATH-7 results
mp_namea = "mp7p2a_mp"
fpth = os.path.join(mp7modelpth, f"{mp_namea}.mppth")
p = flopy.utils.PathlineFile(fpth)
p0 = p.get_alldata()

fpth = os.path.join(mp7modelpth, f"{mp_namea}.timeseries")
ts = flopy.utils.TimeseriesFile(fpth)
ts0 = ts.get_alldata()

# setup the plot
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(1, 1, 1, aspect="equal")
ax.set_title("MODPATH 7 particle tracking results")

mapview = flopy.plot.PlotMapView(vertex_ml6, layer=2)

# plot and contour head arrays
pc = mapview.plot_array(hdata, cmap="Dark2")
contour_set = mapview.contour_array(hdata, levels=levels, colors="white")
plt.clabel(contour_set, fmt="%.1f", colors="white", fontsize=11)
linecollection = mapview.plot_grid(lw=0.25, color="k")
cb = plt.colorbar(pc, shrink=0.75, ax=ax)

# plot the modpath results
pline = mapview.plot_pathline(p0, layer="all", color="blue", lw=0.75)
colors = ["green", "orange", "red"]
for k in range(3):
    tseries = mapview.plot_timeseries(ts0, layer=k, marker="o", lw=0, color=colors[k])

# ### Plotting specific discharge vectors for DISV
# MODFLOW-6 includes a the PLOT_SPECIFIC_DISCHARGE flag in the NPF package to calculate and store discharge vectors for easy plotting. The postprocessing module will translate the specific dischage into vector array and `PlotMapView` has the `plot_vector()` method to use this data. The specific discharge array is stored in the cell budget file.

cbb = flopy.utils.CellBudgetFile(
    os.path.join(mp7modelpth, "mp7p2.cbb"), precision="double"
)
spdis = cbb.get_data(text="SPDIS")[0]
qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(spdis, vertex_ml6)

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(1, 1, 1, aspect="equal")
ax.set_title("Specific discharge for vertex model")

mapview = flopy.plot.PlotMapView(vertex_ml6, layer=2)
pc = mapview.plot_array(hdata, cmap="Dark2")
linecollection = mapview.plot_grid(lw=0.25, color="k")
cb = plt.colorbar(pc, shrink=0.75, ax=ax)

# plot specific discharge
quiver = mapview.plot_vector(qx, qy, normalize=True, alpha=0.60)

# ## Unstructured grid (DISU) plotting with MODFLOW-USG and MODFLOW-6
#
# Unstructured grid (DISU) plotting has support through the `PlotMapView` class and the `UnstructuredGrid` discretization object. The method calls are identical to those used for vertex (DISV) and structured (DIS) model grids. Let's run through a few unstructured grid examples

# set up the notebook for unstructured grid plotting
from flopy.discretization import UnstructuredGrid

datapth = modelpth / "unstructured"
file_names = {
    "TriMesh_local.exp": "0be6a1a1743972ba98c9d9e63ac2e457813c0809bfbda120e09a97b04411a65e",
    "TriMesh_usg.exp": "0b450f2b306253a7b2889796e7a4eea52159f509c7b28a1f65929008dd854e08",
    "Trimesh_circle.exp": "1efb86bb77060dcec20e752e242076e3bd23046f5e47d20d948bcf4623b3deb7",
    "headu.githds": "cbe94655d471470d931923f70c7548b161ea4c5a22333b7fab6e2255450cda89",
    "ugrid_iverts.dat": "7e33ec7f7d1fdbeb6cb7bc8dbcdf35f262c82aaa38dc79b4fb3fe7b53f7c7c1b",
    "ugrid_verts.dat": "59493b26c8969789bb5a06d999db7a2dac324bffee280925e123007c81e689c7",
}
for fname, fhash in file_names.items():
    pooch.retrieve(
        url=f"https://github.com/modflowpy/flopy/raw/develop/examples/data/unstructured/{fname}",
        fname=fname,
        path=data_path / "unstructured",
        known_hash=fhash,
    )

copytree(data_path / "unstructured", datapth, dirs_exist_ok=True)


# simple functions to load vertices and incidence lists
def load_verts(fname):
    verts = np.genfromtxt(fname, dtype=[int, float, float], names=["iv", "x", "y"])
    verts["iv"] -= 1  # zero based
    return verts


def load_iverts(fname):
    f = open(fname)
    iverts = []
    xc = []
    yc = []
    for line in f:
        ll = line.strip().split()
        iverts.append([int(i) - 1 for i in ll[4:]])
        xc.append(float(ll[1]))
        yc.append(float(ll[2]))
    return iverts, np.array(xc), np.array(yc)


# load vertices
fname = os.path.join(datapth, "ugrid_verts.dat")
verts = load_verts(fname)

# load the incidence list into iverts
fname = os.path.join(datapth, "ugrid_iverts.dat")
iverts, xc, yc = load_iverts(fname)

# In this case, verts is just a 2-dimensional list of x,y vertex pairs.  iverts is also a 2-dimensional list, where the outer list is of size ncells, and the inner list is a list of the vertex numbers that comprise the cell.

# Print the first 5 entries in verts and iverts
for ivert, v in enumerate(verts[:5]):
    print(f"Vertex coordinate pair for vertex {ivert}: {v}")
print("...\n")

for icell, vertlist in enumerate(iverts[:5]):
    print(f"List of vertices for cell {icell}: {vertlist}")

# A flopy `UnstructuredGrid` object can now be created using the vertices and incidence list.  The `UnstructuredGrid` object is a key part of the plotting capabilities in flopy.  In addition to the vertex information, the `UnstructuredGrid` object also needs to know how many cells are in each layer.  This is specified in the ncpl variable, which is a list of cells per layer.

ncpl = np.array(5 * [len(iverts)])
umg = UnstructuredGrid(verts, iverts, xc, yc, ncpl=ncpl, angrot=10)
print(ncpl)
print(umg)

# Now that we have an `UnstructuredGrid`, we can use the flopy `PlotMapView` object to create different types of plots, just like we do for structured grids.

f = plt.figure(figsize=(10, 10))
mapview = flopy.plot.PlotMapView(modelgrid=umg)
mapview.plot_grid()
plt.plot(umg.xcellcenters, umg.ycellcenters, "bo")

# Create a random array for layer 0, and then plot it with a color flood and contours
f = plt.figure(figsize=(10, 10))

a = np.random.random(ncpl[0]) * 100
levels = np.arange(0, 100, 30)

mapview = flopy.plot.PlotMapView(modelgrid=umg)
pc = mapview.plot_array(a, cmap="viridis")
contour_set = mapview.contour_array(a, levels=levels, colors="white")
plt.clabel(contour_set, fmt="%.1f", colors="white", fontsize=11)
linecollection = mapview.plot_grid(color="k", lw=0.5)
colorbar = plt.colorbar(pc, shrink=0.75)

# Here are some examples of some other types of grids.  The data files for these grids are located in the datapth folder.

from pathlib import Path

fig = plt.figure(figsize=(10, 30))
fnames = [fname for fname in os.listdir(datapth) if fname.endswith(".exp")]
nplot = len(fnames)
for i, f in enumerate(fnames):
    ax = fig.add_subplot(nplot, 1, i + 1, aspect="equal")
    fname = os.path.join(datapth, f)
    umga = UnstructuredGrid.from_argus_export(fname, nlay=1)
    mapview = flopy.plot.PlotMapView(modelgrid=umga, ax=ax)
    linecollection = mapview.plot_grid(colors="sienna")
    ax.set_title(Path(fname).name)

# ## Plotting using built in styles
#
# FloPy's plotting routines can be used with built in styles from the `styles` module. The `styles` module takes advantage of matplotlib's temporary styling routines by reading in pre-built style sheets. Two different types of styles have been built for flopy: `USGSMap()` and `USGSPlot()` styles which can be used to create report quality figures. The styles module also contains a number of methods that can be used for adding axis labels, text, annotations, headings, removing tick lines, and updating the current font.

# import flopy's styles
from flopy.plot import styles

# get the specific discharge from the cell budget file
cbc_file = os.path.join(sim_path, "freyberg.cbc")
cbc = flopy.utils.CellBudgetFile(cbc_file)
spdis = cbc.get_data(text="SPDIS")[0]

qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(spdis, ml6)

# get the head from the head file
head_file = os.path.join(sim_path, "freyberg.hds")
head = flopy.utils.HeadFile(head_file)
hdata = head.get_alldata()[0]

# use USGSMap style to create a discharge figure:
with styles.USGSMap():
    fig = plt.figure(figsize=(12, 12))

    mapview = flopy.plot.PlotMapView(model=ml6, layer=0)
    linecollection = mapview.plot_grid()
    quadmesh = mapview.plot_array(a=hdata, alpha=0.5)
    quiver = mapview.plot_vector(qx, qy)
    inactive = mapview.plot_inactive()
    plt.colorbar(quadmesh, shrink=0.75)

    # use styles to add a heading, xlabel, ylabel
    styles.heading(letter="A.", heading="Specific Discharge (" + r"$L/T$" + ")")
    styles.xlabel(label="Easting")
    styles.ylabel(label="Northing")

# Here is a second example showing how to change the font type using `styles`

# use USGSMap style, change font type, and plot without tick lines:
with styles.USGSMap():
    fig = plt.figure(figsize=(12, 12))

    mapview = flopy.plot.PlotMapView(model=ml6, layer=0)
    linecollection = mapview.plot_grid()
    quadmesh = mapview.plot_array(a=hdata, alpha=0.5)
    quiver = mapview.plot_vector(qx, qy)
    inactive = mapview.plot_inactive()
    plt.colorbar(quadmesh, shrink=0.75)

    # change the font type to comic sans
    (styles.set_font_type(family="fantasy", fontname="Comic Sans MS"),)

    # use styles to add a heading, xlabel, ylabel, and remove tick marks
    styles.heading(
        letter="A.",
        heading="Comic Sans: Specific Discharge (" + r"$L/T$" + ")",
        fontsize=16,
    )
    styles.xlabel(label="Easting", fontsize=12)
    styles.ylabel(label="Northing", fontsize=12)
    styles.remove_edge_ticks()

# ## Summary
#
# This notebook demonstrates some of the plotting functionality available with FloPy.  Although not described here, the plotting functionality tries to be general by passing keyword arguments passed to `PlotMapView` methods down into the `matplotlib.pyplot` routines that do the actual plotting.  For those looking to customize these plots, it may be necessary to search for the available keywords by understanding the types of objects that are created by the `PlotMapView` methods.  The `PlotMapView` methods return these `matplotlib.collections` objects so that they could be fine-tuned later in the script before plotting.
#
# Hope this gets you started!

try:
    # ignore PermissionError on Windows
    pass
    # tempdir.cleanup()
except:
    pass
# -
