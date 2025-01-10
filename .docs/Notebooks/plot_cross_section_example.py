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
#       - name: Scott Paulinski
# ---

# # Making Cross Sections of Your Model
# This notebook demonstrates the cross sectional mapping capabilities of FloPy. It demonstrates these capabilities by loading and running existing models and then showing how the `PlotCrossSection` object and its methods can be used to make nice plots of the model grid, boundary conditions, model results, shape files, etc.
#
# ### Mapping is demonstrated for MODFLOW-2005 and MODFLOW-6 models in this notebook


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

import flopy

print(sys.version)
print(f"numpy version: {np.__version__}")
print(f"matplotlib version: {mpl.__version__}")
print(f"flopy version: {flopy.__version__}")

# Set names of the MODFLOW exes
# assumes that the executable is in users path statement
v2005 = "mf2005"
exe_name_2005 = "mf2005"
vmf6 = "mf6"
exe_name_mf6 = "mf6"

# Check if we are in the repository and define the data path.

try:
    root = Path(git.Repo(".", search_parent_directories=True).working_dir)
except:
    root = None

data_path = root / "examples" / "data" if root else Path.cwd()
sim_name = "freyberg"

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


# Set the paths
tempdir = TemporaryDirectory()
modelpth = Path(tempdir.name)

# ### Load and Run an Existing MODFLOW-2005 Model
# A model called the "Freyberg Model" is located in the loadpth folder.  In the following code block, we load that model, then change into a new workspace (modelpth) where we recreate and run the model.  For this to work properly, the MODFLOW-2005 executable (mf2005) must be in the path.  We verify that it worked correctly by checking for the presence of freyberg.hds and freyberg.cbc.

# +
ml = flopy.modflow.Modflow.load(
    "freyberg.nam", model_ws=data_path / sim_name, exe_name=exe_name_2005, version=v2005
)
ml.change_model_ws(new_pth=str(modelpth))
ml.write_input()
success, buff = ml.run_model(silent=True, report=True)
assert success, pformat(buff)

files = ["freyberg.hds", "freyberg.cbc"]
for f in files:
    if os.path.isfile(os.path.join(str(modelpth), f)):
        msg = f"Output file located: {f}"
        print(msg)
    else:
        errmsg = f"Error. Output file cannot be found: {f}"
        print(errmsg)

# ### Creating a Cross-Section of the Model Grid
#
# Now that we have a model, we can use the FloPy plotting utilities to make cross-sections. We'll start by making a Map to show the model grid and basic boundary conditions. Then we'll begin making a cross section using the `PlotCrossSection` class and the `plot_grid()` method of that class.

# let's take a look at our grid before making a cross section
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, aspect="equal")
mapview = flopy.plot.PlotMapView(model=ml)
ibound = mapview.plot_ibound()
wel = mapview.plot_bc("WEL")
riv = mapview.plot_bc("RIV")
linecollection = mapview.plot_grid()

# Next we will make a cross-section of the model grid at column 6.

# First step is to set up the plot
fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(1, 1, 1)

# Next we create an instance of the PlotCrossSection class
xsect = flopy.plot.PlotCrossSection(model=ml, line={"Column": 5})

# Then we can use the plot_grid() method to draw the grid
# The return value for this function is a matplotlib LineCollection object,
# which could be manipulated (or used) later if necessary.
linecollection = xsect.plot_grid()
t = ax.set_title("Column 6 Cross-Section - Model Grid")

# ### Ploting Ibound
#
# The `plot_ibound()` method can be used to plot the boundary conditions contained in the ibound arrray, which is part of the MODFLOW Basic Package.  The `plot_ibound()` method returns a matplotlib PatchCollection object (matplotlib.collections.PatchCollection).  If you are familiar with the matplotlib collections, then this may be important to you, but if not, then don't worry about the return objects of these plotting function.

fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(1, 1, 1)

xsect = flopy.plot.PlotCrossSection(model=ml, line={"Column": 5})
patches = xsect.plot_ibound()
linecollection = xsect.plot_grid()
t = ax.set_title("Column 6 Cross-Section with IBOUND Boundary Conditions")

# Or we could change the colors!
fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(1, 1, 1)

xsect = flopy.plot.PlotCrossSection(model=ml, line={"Column": 5})
patches = xsect.plot_ibound(color_noflow="red", color_ch="orange")
linecollection = xsect.plot_grid(color="green")
t = ax.set_title("Column 6 Cross-Section with IBOUND Boundary Conditions")

# ### Plotting Boundary Conditions
# The `plot_bc()` method can be used to plot boundary conditions on a cross section.  It is setup to use the following dictionary to assign colors, however, these colors can be changed in the method call.
#
#     bc_color_dict = {'default': 'black', 'WEL': 'red', 'DRN': 'yellow',
#                  'RIV': 'green', 'GHB': 'cyan', 'CHD': 'navy'}
#
# Just like the `plot_bc()` method for `PlotMapView`, the default boundary condition colors can be changed in the method call.
#
# Here, we plot the location of well cells in column 6.

fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(1, 1, 1)

xsect = flopy.plot.PlotCrossSection(model=ml, line={"Column": 5})
patches = xsect.plot_bc("WEL", color="pink")
patches = xsect.plot_ibound()
linecollection = xsect.plot_grid()
t = ax.set_title("Column 6 Cross-Section with Boundary Conditions")

# ### Plotting an Array
#
# `PlotCrossSection` has a `plot_array()` method.  The `plot_array()` method will only accept 3D arrays for structured grids.

# Create a random array and plot it
a = np.random.random((ml.dis.nlay, ml.dis.nrow, ml.dis.ncol))

fig = plt.figure(figsize=(18, 5))
ax = fig.add_subplot(1, 1, 1)
xsect = flopy.plot.PlotCrossSection(model=ml, line={"Column": 5})
csa = xsect.plot_array(a)
patches = xsect.plot_ibound()
linecollection = xsect.plot_grid()
t = ax.set_title("Column 6 Cross-Section with Random Data")
cb = plt.colorbar(csa, shrink=0.75)

# plot the horizontal hydraulic conductivities
a = ml.lpf.hk.array

fig = plt.figure(figsize=(18, 5))
ax = fig.add_subplot(1, 1, 1)
xsect = flopy.plot.PlotCrossSection(model=ml, line={"Column": 5})
csa = xsect.plot_array(a)
patches = xsect.plot_ibound()
linecollection = xsect.plot_grid()
t = ax.set_title("Column 6 Cross-Section with Horizontal hydraulic conductivity")
cb = plt.colorbar(csa, shrink=0.75)

# ### Contouring an Array
#
# `PlotCrossSection` also has a `contour_array()` method.  It also accepts a 3D array for structured grids.

# plot the horizontal hydraulic conductivities
a = ml.lpf.hk.array

fig = plt.figure(figsize=(18, 5))
ax = fig.add_subplot(1, 1, 1)
xsect = flopy.plot.PlotCrossSection(model=ml, line={"Column": 5})
contour_set = xsect.contour_array(a, masked_values=[0], cmap="jet")
patches = xsect.plot_ibound()
linecollection = xsect.plot_grid(color="grey")
t = ax.set_title(
    "Column 6 Cross-Section contour_array() horizontal hydraulic conductivity"
)
cb = plt.colorbar(contour_set, shrink=0.75)

# ### Plotting Heads
#
# We can easily plot results from the simulation by extracting heads using `flopy.utils.HeadFile`.
#
# The head can be passed into the `plot_array()` and `contour_array()` using the `head=` keyword argument to fix the top of the colored patch and contour lines at the top of the water table in each cell, respectively.

fname = os.path.join(str(modelpth), "freyberg.hds")
hdobj = flopy.utils.HeadFile(fname)
head = hdobj.get_data()

fig = plt.figure(figsize=(18, 5))

ax = fig.add_subplot(1, 1, 1)
ax.set_title("plot_array() used to plotting Heads")
xsect = flopy.plot.PlotCrossSection(model=ml, line={"Column": 5})
pc = xsect.plot_array(head, head=head, alpha=0.5)
patches = xsect.plot_ibound(head=head)
linecollection = xsect.plot_grid()
cb = plt.colorbar(pc, shrink=0.75)

# contour array on top of heads
levels = np.arange(17, 26, 1)

fig = plt.figure(figsize=(18, 5))
ax = fig.add_subplot(1, 1, 1)
ax.set_title("contour_array() and plot_array() of head values")

# instantiate the PlotCrossSection object
xsect = flopy.plot.PlotCrossSection(model=ml, line={"Column": 5})

# plot the head array and model grid
pc = xsect.plot_array(head, masked_values=[999.0], head=head, alpha=0.5)
patches = xsect.plot_ibound(head=head)
linecollection = xsect.plot_grid()

# do black contour lines of the head array
contour_set = xsect.contour_array(head, head=head, levels=levels, colors="k")
plt.clabel(contour_set, fmt="%.1f", colors="k", fontsize=11)

cb = plt.colorbar(pc, shrink=0.75)

# ### Plotting a surface on the cross section
#
# The `plot_surface()` method allows the user to plot a surface along the cross section. Here is a short example using head data.

levels = np.arange(10, 30, 0.5)

fig = plt.figure(figsize=(18, 5))
xsect = flopy.plot.PlotCrossSection(model=ml, line={"Column": 5})

# contour array and plot ibound
ct = xsect.contour_array(
    head, masked_values=[999.0], head=head, levels=levels, linewidths=2.5
)
pc = xsect.plot_ibound(head=head)

# plot the surface and model grid
wt = xsect.plot_surface(head, color="blue", lw=2.5)
linecollection = xsect.plot_grid()

plt.title("contour_array() and plot_surface()")
cb = plt.colorbar(ct, shrink=0.75)

# ### Plotting discharge vectors
#
# `PlotCrossSection` has a `plot_vector()` method, which takes `qx`, `qy`, and `qz` vector arrays (ex. specific discharge or flow across a cell faces). The flow array values can be extracted from the cell by cell flow file using the `flopy.utils.CellBudgetFile` object as shown below.  Once they are extracted, they either be can be passed to the `plot_vector()` method or they can be post processed into specific discharge using `postprocessing.get_specific_discharge`.  Note that `get_specific_discharge()` also takes the head array as an argument.  The head array is used by `get_specific_discharge()` to convert the volumetric flow in dimensions of $L^3/T$ to specific discharge in dimensions of $L/T$ and to plot the specific discharge in the center of each saturated cell. For this problem, there is no 'FLOW LOWER FACE' array since the Freyberg Model is a one layer model.

fname = os.path.join(str(modelpth), "freyberg.cbc")
cbb = flopy.utils.CellBudgetFile(fname)
frf = cbb.get_data(text="FLOW RIGHT FACE")[0]
fff = cbb.get_data(text="FLOW FRONT FACE")[0]
qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(
    (frf, fff, None), ml, head=head
)

fig = plt.figure(figsize=(18, 5))
ax = fig.add_subplot(1, 1, 1)

ax.set_title("plot_array() and plot_vector()")
xsect = flopy.plot.PlotCrossSection(model=ml, ax=ax, line={"Column": 5})
csa = xsect.plot_array(head, head=head, alpha=0.5)
patches = xsect.plot_ibound(head=head)
linecollection = xsect.plot_grid()
quiver = xsect.plot_vector(
    qx,
    qy,
    qz,
    head=head,
    hstep=2,
    normalize=True,
    color="green",
    scale=30,
    headwidth=3,
    headlength=3,
    headaxislength=3,
    zorder=10,
)

cb = plt.colorbar(csa, shrink=0.75)

# ### Plotting a cross section from Shapefile data
#
# A shapefile can be used to define the vertices for a instance of the `PlotCrossSection` class. The function `flopy.plot.plotutil.shapefile_get_vertices()` will return a list of vertices for each polyline in a shapefile.
#
# Let's plot the shapefiles and the Freyberg model using `PlotMapView` for visualization purposes and then plot the cross-section.

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

# Setup the figure and PlotMapView. Show a very faint map of ibound and
# model grid by specifying a transparency alpha value.

# set the modelgrid rotation and offset
ml.modelgrid.set_coord_info(
    xoff=-2419.2189559966773, yoff=297.0427372400354, angrot=-14
)

fig = plt.figure(figsize=(12, 12))
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
patch_collection = mapview.plot_shapefile(
    shp, radius=0, lw=3, edgecolor="red", facecolor="None"
)

# Plot a shapefile of well locations
shp = os.path.join(modelpth, "gis", "wells_locations_rotate14")
patch_collection = mapview.plot_shapefile(shp, radius=100, facecolor="red")

# Plot the grid and boundary conditions over the top
quadmesh = mapview.plot_ibound(alpha=0.1)
linecollection = mapview.plot_grid(alpha=0.1)

# Now let's make a cross section based on this arbitrary cross-sectional line. We can load the cross sectional line vertices using `flopy.plot.plotutil.shapefile_get_vertices()`
#
# **Note**: in previous examples we passed `line={'column', 5}` to plot a cross section along a column. In this example we pass vertex information into `PlotCrossSection` using `line={'line', line[0]}` where `line[0]` is a list of vertices.

# get the vertices for cross-section lines in a shapefile
fpth = os.path.join(modelpth, "gis", "cross_section_rotate14")
line = flopy.plot.plotutil.shapefile_get_vertices(fpth)

# Set up the figure
fig = plt.figure(figsize=(18, 5))
ax = fig.add_subplot(1, 1, 1)
ax.set_title("plot_array() along an arbitrary cross-sectional line")

# plot head values along the cross sectional line
xsect = flopy.plot.PlotCrossSection(model=ml, line={"line": line[0]})
csa = xsect.plot_array(head, head=head, alpha=0.5)
patches = xsect.plot_ibound(head=head)
linecollection = xsect.plot_grid(lw=0.5)
cb = fig.colorbar(csa, ax=ax, shrink=0.5)

# ### Plotting geographic coordinates on the x-axis using the `PlotCrossSection` class
#
# The default cross section plotting method plots cells with regard to their intersection distance along the cross sectional line defined by the user. While this method is perfectly acceptable and in many cases may be preferred for plotting arbitrary cross sections, a flag has been added to plot based on geographic coordinates.
#
# The flag `geographic_coords` defaults to `False` which maintains FloPy's previous method of plotting cross sections.

# get the vertices for cross-section lines in a shapefile
fpth = os.path.join(modelpth, "gis", "cross_section_rotate14")
line = flopy.plot.plotutil.shapefile_get_vertices(fpth)

# Set up the figure
fig = plt.figure(figsize=(18, 5))
ax = fig.add_subplot(1, 1, 1)
ax.set_title("plot_array() along an arbitrary cross-sectional line")

# plot head values along the cross sectional line
xsect = flopy.plot.PlotCrossSection(
    model=ml, line={"line": line[0]}, geographic_coords=True
)
csa = xsect.plot_array(head, head=head, alpha=0.5)
patches = xsect.plot_ibound(head=head)
linecollection = xsect.plot_grid(lw=0.5)
cb = fig.colorbar(csa, ax=ax, shrink=0.5)

# ## Plotting Cross Sections with MODFLOW-6 models
#
# `PlotCrossSection` has support for MODFLOW-6 models and operates in the same fashion for Structured Grids, Vertex Grids, and Unstructured Grids. Here is a short example on how to plot with MODFLOW-6 structured grids using a version of the Freyberg model created for MODFLOW-6|

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

# +
# load the Freyberg model into mf6-flopy and run the simulation
sim = flopy.mf6.MFSimulation.load(
    sim_name="mfsim.nam",
    version=vmf6,
    exe_name=exe_name_mf6,
    sim_ws=data_path / sim_name,
)

sim.set_sim_path(modelpth)
sim.write_simulation()
success, buff = sim.run_simulation(silent=True, report=True)
assert success, pformat(buff)

files = ["freyberg.hds", "freyberg.cbc"]
for f in files:
    if os.path.isfile(os.path.join(str(modelpth), f)):
        msg = f"Output file located: {f}"
        print(msg)
    else:
        errmsg = f"Error. Output file cannot be found: {f}"
        print(errmsg)

# ### Plotting boundary conditions and arrays
#
# This works the same as modflow-2005, however the simulation object can host a number of modflow-6 models so we need to grab a model before attempting to plot with `PlotCrossSection`

# get the modflow-6 model we want to plot
ml6 = sim.get_model("freyberg")

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(2, 1, 1)

# plot boundary conditions
xsect = flopy.plot.PlotCrossSection(model=ml6, line={"Column": 5})
patches = xsect.plot_bc("WEL", color="pink")
patches = xsect.plot_ibound()
linecollection = xsect.plot_grid()
t = ax.set_title("Column 6 Cross-Section with Boundary Conditions")

# plot xxxx
ax = fig.add_subplot(2, 1, 2)
# plot the horizontal hydraulic conductivities
a = ml6.npf.k.array

xsect = flopy.plot.PlotCrossSection(model=ml6, line={"Column": 5})
csa = xsect.plot_array(a)
patches = xsect.plot_ibound()
linecollection = xsect.plot_grid()
t = ax.set_title("Column 6 Cross-Section with Horizontal hydraulic conductivity")
cb = plt.colorbar(csa, shrink=0.75)

# ### Plotting specific discharge with a MODFLOW-6 model
#
# MODFLOW-6 includes a the PLOT_SPECIFIC_DISCHARGE flag in the NPF package to calculate and store discharge vectors for easy plotting. The `postprocessing.get_specific_discharge()` method will preprocess the data into vectors and `PlotCrossSection` has the `plot_vector()` method to use this data. The specific discharge array is stored in the cell budget file.

# get the head from the head file
head_file = os.path.join(modelpth, "freyberg.hds")
hds = flopy.utils.HeadFile(head_file)
head = hds.get_alldata()[0]

# get the specific discharge from the cell budget file
cbc_file = os.path.join(modelpth, "freyberg.cbc")
cbc = flopy.utils.CellBudgetFile(cbc_file, precision="double")
spdis = cbc.get_data(text="SPDIS")[-1]
qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(spdis, ml6, head=head)

fig = plt.figure(figsize=(18, 5))
ax = fig.add_subplot(1, 1, 1)

ax.set_title("plot_array() and plot_vector()")
xsect = flopy.plot.PlotCrossSection(model=ml6, ax=ax, line={"Column": 5})
csa = xsect.plot_array(head, head=head, alpha=0.5)
patches = xsect.plot_ibound(head=head)
linecollection = xsect.plot_grid()
quiver = xsect.plot_vector(
    qx,
    qy,
    qz,
    head=head,
    hstep=2,
    normalize=True,
    color="green",
    scale=30,
    headwidth=3,
    headlength=3,
    headaxislength=3,
    zorder=10,
)

cb = plt.colorbar(csa, shrink=0.75)

# ## Vertex cross section plotting with MODFLOW-6 (DISV)
#
# FloPy fully supports vertex discretization (DISV) plotting through the `PlotCrossSection` class. The method calls are identical to the ones presented previously for Structured discretization (DIS) and the same matplotlib keyword arguments are supported. Let's run through an example using a vertex model grid.


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
modelpth = os.path.join(modelpth, "mp7_ex2", "mf6")
# from pprint import pprint
# pprint([str(p) for p in (Path(tempdir.name) / "mp7_ex2").glob('*')])
files = ["mp7p2.hds", "mp7p2.cbb"]
for f in files:
    if os.path.isfile(os.path.join(modelpth, f)):
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
    sim_ws=modelpth,
)
vertex_ml6 = vertex_sim.get_model("mp7p2")

# ### Plotting a line based cross section through the model grid
#
# Because a `VertexGrid` has no row or column number, the cross-section line must be defined explicitly. This is done by passing a dictionary to the `line` parameter with key `line` &mdash; the value may be an array-like of 2 or more points, e.g. `{"line": [(x0, y0), (x1, y1), ...]}`, or a `flopy.utils.geometry.LineString` or `shapely.geometry.LineString`. Below we show an example of setting up a cross-section line with a MODFLOW-6 DISV model.
#

line = np.array([(4700, 0), (4700, 5000), (7250, 10500)])

# Let's plot the model grid in map view to look at it
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(1, 1, 1, aspect="equal")
ax.set_title("Vertex Model Grid (DISV) with cross sectional line")

# use PlotMapView to plot a DISV (vertex) model
mapview = flopy.plot.PlotMapView(vertex_ml6, layer=0)
linecollection = mapview.plot_grid()

# plot the line over the model grid
lc = plt.plot(line.T[0], line.T[1], "r--", lw=2)

# Now we can plot a cross section of the model grid defined by this line

fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(1, 1, 1)

# Next we create an instance of the PlotCrossSection class
xsect = flopy.plot.PlotCrossSection(model=vertex_ml6, line={"line": line})

# Then we can use the plot_grid() method to draw the grid
# The return value for this function is a matplotlib LineCollection object,
# which could be manipulated (or used) later if necessary.
linecollection = xsect.plot_grid()
t = ax.set_title("Column 6 Cross-Section - Model Grid")

# ### Plotting Arrays and Contouring with Vertex Model grids
#
# `PlotCrossSection` allows the user to plot arrays and contour with DISV based discretization. The `plot_array()` method is called in the same way as using a structured grid. The only difference is that `PlotCrossSection` builds a matplotlib patch collection for Vertex based grids.

# get the head output for stress period 1 from the modflow6 head file
head = flopy.utils.HeadFile(os.path.join(modelpth, "mp7p2.hds"))
hdata = head.get_alldata()[0, :, :, :]

fig = plt.figure(figsize=(18, 5))
ax = fig.add_subplot(1, 1, 1)
ax.set_title("plot_array()")

xsect = flopy.plot.PlotCrossSection(model=vertex_ml6, line={"line": line})
patch_collection = xsect.plot_array(hdata, head=hdata, alpha=0.5)
line_collection = xsect.plot_grid()
cb = plt.colorbar(patch_collection, shrink=0.75)

# The `contour_array()` method operates in the same way as the sturctured example.

levels = np.arange(329, 337, 1)

fig = plt.figure(figsize=(18, 5))
ax = fig.add_subplot(1, 1, 1)
ax.set_title("contour_array() with a multi-layer vertex model")

xsect = flopy.plot.PlotCrossSection(model=vertex_ml6, line={"line": line})
patch_collection = xsect.plot_array(hdata, head=hdata, alpha=0.5)
line_collection = xsect.plot_grid()

contour_set = xsect.contour_array(hdata, levels=levels, colors="k")
plt.clabel(contour_set, fmt="%.1f", colors="k", fontsize=11)

cb = plt.colorbar(patch_collection, shrink=0.75)

# ### Plotting specific discharge vectors for DISV
# MODFLOW-6 includes a the PLOT_SPECIFIC_DISCHARGE flag in the NPF package to calculate and store discharge vectors for easy plotting.The `postprocessing.get_specific_discharge()` method will preprocess the data into vectors and `PlotCrossSection` has the `plot_vector()` method to use this data. The specific discharge array is stored in the cell budget file.
#
# **Note**: When plotting specific discharge, an arbitrary cross section cannot be used. The cross sectional line must be orthogonal to the model grid

# define and plot our orthogonal line
line = np.array([(0, 4700), (10000, 4700)])

# Let's plot the model grid in map view to look at it
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(1, 1, 1, aspect="equal")
ax.set_title("Vertex Model Grid (DISV) with cross sectional line")

# use PlotMapView to plot a DISV (vertex) model
mapview = flopy.plot.PlotMapView(vertex_ml6, layer=0)
linecollection = mapview.plot_grid()

# plot the line over the model grid
lc = plt.plot(line.T[0], line.T[1], "r--", lw=2)

# plot specific discharge on cross section
cbb = flopy.utils.CellBudgetFile(os.path.join(modelpth, "mp7p2.cbb"))
spdis = cbb.get_data(text="SPDIS")[-1]
qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(
    spdis, vertex_ml6, head=hdata
)

fig = plt.figure(figsize=(18, 5))
ax = fig.add_subplot(1, 1, 1)

xsect = flopy.plot.PlotCrossSection(model=vertex_ml6, line={"line": line})
patch_collection = xsect.plot_array(hdata, head=hdata, alpha=0.5)
line_collection = xsect.plot_grid()
quiver = xsect.plot_vector(
    qx,
    qy,
    qz,
    head=hdata,
    hstep=3,
    normalize=True,
    color="green",
    scale=30,
    headwidth=3,
    headlength=3,
    headaxislength=3,
    zorder=10,
)

cb = plt.colorbar(patch_collection, shrink=0.75)

# ## Plotting using built in styles
#
# FloPy's plotting routines can be used with built in styles from the `styles` module. The `styles` module takes advantage of matplotlib's temporary styling routines by reading in pre-built style sheets. Two different types of styles have been built for flopy: `USGSMap()` and `USGSPlot()` styles which can be used to create report quality figures. The styles module also contains a number of methods that can be used for adding axis labels, text, annotations, headings, removing tick lines, and updating the current font.
#
# This example will run the Keating groundwater transport model and plot results using `styles`

example_name = "ex-gwt-keating"

# Model units

length_units = "m"
time_units = "days"

# Table of model parameters

nlay = 80  # Number of layers
nrow = 1  # Number of rows
ncol = 400  # Number of columns
delr = 25.0  # Column width ($m$)
delc = 1.0  # Row width ($m$)
delz = 25.0  # Layer thickness ($m$)
top = 2000.0  # Top of model domain ($m$)
bottom = 0.0  # Bottom of model domain ($m$)
hka = 1.0e-12  # Permeability of aquifer ($m^2$)
hkc = 1.0e-18  # Permeability of aquitard ($m^2$)
h1 = 800.0  # Head on left side ($m$)
h2 = 100.0  # Head on right side ($m$)
recharge = 0.5  # Recharge ($kg/s$)
recharge_conc = 1.0  # Normalized recharge concentration (unitless)
alpha_l = 1.0  # Longitudinal dispersivity ($m$)
alpha_th = 1.0  # Transverse horizontal dispersivity ($m$)
alpha_tv = 1.0  # Transverse vertical dispersivity ($m$)
period1 = 730  # Length of first simulation period ($d$)
period2 = 29270.0  # Length of second simulation period ($d$)
porosity = 0.1  # Porosity of mobile domain (unitless)
obs1 = (49, 1, 119)  # Layer, row, and column for observation 1
obs2 = (77, 1, 359)  # Layer, row, and column for observation 2

obs1 = tuple([i - 1 for i in obs1])
obs2 = tuple([i - 1 for i in obs2])
seconds_to_days = 24.0 * 60.0 * 60.0
permeability_to_conductivity = 1000.0 * 9.81 / 1.0e-3 * seconds_to_days
hka = hka * permeability_to_conductivity
hkc = hkc * permeability_to_conductivity
botm = [top - (k + 1) * delz for k in range(nlay)]
x = np.arange(0, 10000.0, delr) + delr / 2.0
plotaspect = 1.0

# Fill hydraulic conductivity array
hydraulic_conductivity = np.ones((nlay, nrow, ncol), dtype=float) * hka
for k in range(nlay):
    if 1000.0 <= botm[k] < 1100.0:
        for j in range(ncol):
            if 3000.0 <= x[j] <= 6000.0:
                hydraulic_conductivity[k, 0, j] = hkc

# Calculate recharge by converting from kg/s to m/d
rcol = []
for jcol in range(ncol):
    if 4200.0 <= x[jcol] <= 4800.0:
        rcol.append(jcol)
number_recharge_cells = len(rcol)
rrate = recharge * seconds_to_days / 1000.0
cell_area = delr * delc
rrate = rrate / (float(number_recharge_cells) * cell_area)
rchspd = {}
rchspd[0] = [[(0, 0, j), rrate, recharge_conc] for j in rcol]
rchspd[1] = [[(0, 0, j), rrate, 0.0] for j in rcol]


def build_mf6gwf(sim_folder):
    ws = os.path.join(sim_folder, "mf6-gwt-keating")
    name = "flow"
    sim_ws = os.path.join(ws, "mf6gwf")
    sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=sim_ws, exe_name="mf6")
    tdis_ds = ((period1, 1, 1.0), (period2, 1, 1.0))
    flopy.mf6.ModflowTdis(
        sim, nper=len(tdis_ds), perioddata=tdis_ds, time_units=time_units
    )
    flopy.mf6.ModflowIms(
        sim,
        print_option="summary",
        complexity="complex",
        no_ptcrecord="all",
        outer_dvclose=1.0e-4,
        outer_maximum=2000,
        under_relaxation="dbd",
        linear_acceleration="BICGSTAB",
        under_relaxation_theta=0.7,
        under_relaxation_kappa=0.08,
        under_relaxation_gamma=0.05,
        under_relaxation_momentum=0.0,
        backtracking_number=20,
        backtracking_tolerance=2.0,
        backtracking_reduction_factor=0.2,
        backtracking_residual_limit=5.0e-4,
        inner_dvclose=1.0e-5,
        rcloserecord=[0.0001, "relative_rclose"],
        inner_maximum=100,
        relaxation_factor=0.0,
        number_orthogonalizations=2,
        preconditioner_levels=8,
        preconditioner_drop_tolerance=0.001,
    )
    gwf = flopy.mf6.ModflowGwf(
        sim, modelname=name, save_flows=True, newtonoptions=["newton"]
    )
    flopy.mf6.ModflowGwfdis(
        gwf,
        length_units=length_units,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
    )
    flopy.mf6.ModflowGwfnpf(
        gwf,
        save_specific_discharge=True,
        save_saturation=True,
        icelltype=1,
        k=hydraulic_conductivity,
    )
    flopy.mf6.ModflowGwfic(gwf, strt=600.0)
    chdspd = [[(k, 0, 0), h1] for k in range(nlay) if botm[k] < h1]
    chdspd += [[(k, 0, ncol - 1), h2] for k in range(nlay) if botm[k] < h2]
    flopy.mf6.ModflowGwfchd(
        gwf,
        stress_period_data=chdspd,
        print_input=True,
        print_flows=True,
        save_flows=False,
        pname="CHD-1",
    )
    flopy.mf6.ModflowGwfrch(
        gwf, stress_period_data=rchspd, auxiliary=["concentration"], pname="RCH-1"
    )

    head_filerecord = f"{name}.hds"
    budget_filerecord = f"{name}.bud"
    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord=head_filerecord,
        budget_filerecord=budget_filerecord,
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )
    return sim


def build_mf6gwt(sim_folder):
    ws = os.path.join(sim_folder, "mf6-gwt-keating")
    name = "trans"
    sim_ws = os.path.join(ws, "mf6gwt")
    sim = flopy.mf6.MFSimulation(
        sim_name=name,
        sim_ws=sim_ws,
        exe_name="mf6",
        continue_=True,
    )
    tdis_ds = ((period1, 73, 1.0), (period2, 2927, 1.0))
    flopy.mf6.ModflowTdis(
        sim, nper=len(tdis_ds), perioddata=tdis_ds, time_units=time_units
    )
    flopy.mf6.ModflowIms(
        sim,
        print_option="summary",
        outer_dvclose=1.0e-4,
        outer_maximum=100,
        under_relaxation="none",
        linear_acceleration="BICGSTAB",
        rcloserecord=[1000.0, "strict"],
        inner_maximum=20,
        inner_dvclose=1.0e-4,
        relaxation_factor=0.0,
        number_orthogonalizations=2,
        preconditioner_levels=8,
        preconditioner_drop_tolerance=0.001,
    )
    gwt = flopy.mf6.ModflowGwt(sim, modelname=name, save_flows=True)
    flopy.mf6.ModflowGwtdis(
        gwt,
        length_units=length_units,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
    )
    flopy.mf6.ModflowGwtic(gwt, strt=0)
    flopy.mf6.ModflowGwtmst(gwt, porosity=porosity)
    flopy.mf6.ModflowGwtadv(gwt, scheme="upstream")
    flopy.mf6.ModflowGwtdsp(
        gwt, xt3d_off=True, alh=alpha_l, ath1=alpha_th, atv=alpha_tv
    )
    pd = [
        ("GWFHEAD", "../mf6gwf/flow.hds"),
        ("GWFBUDGET", "../mf6gwf/flow.bud"),
    ]
    flopy.mf6.ModflowGwtfmi(gwt, flow_imbalance_correction=True, packagedata=pd)
    sourcerecarray = [
        ("RCH-1", "AUX", "CONCENTRATION"),
    ]
    flopy.mf6.ModflowGwtssm(gwt, sources=sourcerecarray)
    saverecord = {
        0: [
            ("CONCENTRATION", "STEPS", 10),
            ("CONCENTRATION", "LAST"),
            ("CONCENTRATION", "FREQUENCY", 10),
        ],
        1: [
            ("CONCENTRATION", "STEPS", 27, 227),
            ("CONCENTRATION", "LAST"),
            ("CONCENTRATION", "FREQUENCY", 10),
        ],
    }
    flopy.mf6.ModflowGwtoc(
        gwt,
        budget_filerecord=f"{name}.cbc",
        concentration_filerecord=f"{name}.ucn",
        concentrationprintrecord=[
            ("COLUMNS", ncol, "WIDTH", 15, "DIGITS", 6, "GENERAL")
        ],
        saverecord=saverecord,
        printrecord=[
            ("CONCENTRATION", "LAST"),
            ("BUDGET", "ALL"),
        ],
    )
    obs_data = {
        f"{name}.obs.csv": [
            ("obs1", "CONCENTRATION", obs1),
            ("obs2", "CONCENTRATION", obs2),
        ],
    }
    flopy.mf6.ModflowUtlobs(gwt, digits=10, print_input=True, continuous=obs_data)
    return sim


def build_model(ws):
    sim_mf6gwf = build_mf6gwf(ws)
    sim_mf6gwt = build_mf6gwt(ws)
    sim_mf2005 = None  # build_mf2005(sim_name)
    sim_mt3dms = None  # build_mt3dms(sim_name, sim_mf2005)
    sims = (sim_mf6gwf, sim_mf6gwt, sim_mf2005, sim_mt3dms)
    return sims


def write_model(sims, silent=True):
    sim_mf6gwf, sim_mf6gwt, sim_mf2005, sim_mt3dms = sims
    sim_mf6gwf.write_simulation(silent=silent)
    sim_mf6gwt.write_simulation(silent=silent)


def run_keating_model(ws=example_name, silent=True):
    sim = build_model(ws)
    write_model(sim, silent=silent)
    sim_mf6gwf, sim_mf6gwt, sim_mf2005, sim_mt3dms = sim

    print("Running mf6gwf model...")
    success, buff = sim_mf6gwf.run_simulation(silent=silent)
    if not success:
        print(buff)

    print("Running mf6gwt model...")
    success, buff = sim_mf6gwt.run_simulation(silent=silent)
    if not success:
        print(buff)

    return success


run_keating_model(modelpth)

# Load the flow and transport models

sim_path = os.path.join(modelpth, "mf6-gwt-keating", "mf6gwf")
tr_path = os.path.join(modelpth, "mf6-gwt-keating", "mf6gwt")
sim_name = "mfsim.nam"
sim = flopy.mf6.MFSimulation.load(
    sim_name=sim_name, version=vmf6, exe_name=exe_name_mf6, sim_ws=sim_path
)
gwf6 = sim.get_model("flow")

sim = flopy.mf6.MFSimulation.load(
    sim_name=sim_name, version=vmf6, exe_name=exe_name_mf6, sim_ws=tr_path
)
gwt6 = sim.get_model("trans")

# import styles
from flopy.plot import styles

# load head file and plot
head = gwf6.output.head().get_data()

with styles.USGSMap():
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=300, tight_layout=True)

    xsect = flopy.plot.PlotCrossSection(model=gwf6, ax=ax, line={"row": 0})
    pc = xsect.plot_array(head, head=head, cmap="jet")
    xsect.plot_bc(ftype="RCH", color="red")
    xsect.plot_bc(ftype="CHD")
    plt.colorbar(pc, shrink=0.25)

    # add a rectangle to show the confining layer
    confining_rect = mpl.patches.Rectangle(
        (3000, 1000), 3000, 100, color="gray", alpha=0.5
    )
    ax.add_patch(confining_rect)

    # set labels using styles
    styles.xlabel(label="x-position (m)")
    styles.ylabel(label="elevation (m)")
    styles.heading(letter="A.", heading="Simulated hydraulic head", fontsize=10)
    ax.set_aspect(1.0)

# Plotting concentration model results using the `USGSMap()` style

# load the transport output file
cobj = gwt6.output.concentration()
plot_times = [100, 1000, 3000]
obs1 = (48, 0, 118)  # Layer, row, and column for observation 1
obs2 = (76, 0, 358)  # Layer, row, and column for observation 2
xgrid, _, zgrid = gwf6.modelgrid.xyzcellcenters


with styles.USGSPlot():
    fig, axes = plt.subplots(3, 1, figsize=(15, 9), tight_layout=True)
    for ix, totim in enumerate(plot_times):
        heading = f"Time = {totim}"
        conc = cobj.get_data(totim=totim)
        ax = axes[ix]
        xsect = flopy.plot.PlotCrossSection(model=gwf6, ax=ax, line={"row": 0})
        pc = xsect.plot_array(conc, head=head, cmap="jet", vmin=0, vmax=1)
        xsect.plot_bc(ftype="RCH", color="red")
        xsect.plot_bc(ftype="CHD")

        # plot confining layer
        confining_rect = mpl.patches.Rectangle(
            (3000, 1000), 3000, 100, color="gray", alpha=0.5
        )
        ax.add_patch(confining_rect)

        # set axis labels and title using styles
        styles.ylabel(ax=ax, label="elevation (m)", fontsize=10)
        if ix == 2:
            styles.xlabel(ax=ax, label="x-position (m)", fontsize=10)
        styles.heading(ax=ax, heading=heading, idx=ix, fontsize=12)

        ax.set_aspect(1.0)

        # add observation locations based on grid cell centers
        for k, i, j in [obs1, obs2]:
            x = xgrid[i, j]
            z = zgrid[k, i, j]
            ax.plot(x, z, mfc="yellow", mec="black", marker="o", ms="8")

# ## Summary
#
# This notebook demonstrates some of the plotting functionality available with flopy.  Although not described here, the plotting functionality tries to be general by passing keyword arguments passed to the `PlotCrossSection` methods down into the `matplotlib.pyplot` routines that do the actual plotting.  For those looking to customize these plots, it may be necessary to search for the available keywords by understanding the types of objects that are created by the `PlotCrossSection` methods.  The `PlotCrossSection` methods return these matplotlib.collections objects so that they could be fine-tuned later in the script before plotting.
#
# Hope this gets you started!

try:
    # ignore PermissionError on Windows
    tempdir.cleanup()
except:
    pass
