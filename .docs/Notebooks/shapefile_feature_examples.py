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
#     section: flopy
#     authors:
#       - name: Andy Leaf
# ---

# # Working with shapefiles
#
# This notebook shows some lower-level functionality in `flopy` for working with shapefiles
# including:
# * `recarray2shp` convience function for writing a numpy record array to a shapefile
# * `shp2recarray` convience function for quickly reading a shapefile into a numpy recarray
# * `utils.geometry` classes for writing shapefiles of model input/output. For example, quickly writing a shapefile of model cells with errors identified by the checker
# * examples of how the `Point` and `LineString` classes can be used to quickly plot pathlines and endpoints from MODPATH (these are also used by the `PathlineFile` and `EndpointFile` classes to write shapefiles of this output)

# +
import os
import shutil
import sys
import warnings
from tempfile import TemporaryDirectory

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import flopy
from flopy.export.shapefile_utils import recarray2shp, shp2recarray
from flopy.utils import geometry
from flopy.utils.geometry import LineString, Point, Polygon
from flopy.utils.modpathfile import EndpointFile, PathlineFile

warnings.simplefilter("ignore", UserWarning)
print(sys.version)
print(f"numpy version: {np.__version__}")
print(f"matplotlib version: {mpl.__version__}")
print(f"flopy version: {flopy.__version__}")
# -

# ### write a numpy record array to a shapefile
# in this case, we want to visualize output from the checker
# first make a toy model

# +
temp_dir = TemporaryDirectory()
workspace = temp_dir.name

m = flopy.modflow.Modflow("toy_model", model_ws=workspace)
botm = np.zeros((2, 10, 10))
botm[0, :, :] = 1.5
botm[1, 5, 5] = 4  # negative layer thickness!
botm[1, 6, 6] = 4
dis = flopy.modflow.ModflowDis(
    nrow=10, ncol=10, nlay=2, delr=100, delc=100, top=3, botm=botm, model=m
)
# -

# ### set coordinate information

grid = m.modelgrid
grid.set_coord_info(xoff=600000, yoff=5170000, crs="EPSG:26715", angrot=45)

chk = dis.check()
chk.summary_array

# ### make geometry objects for the cells with errors
# *  geometry objects allow the shapefile writer to be simpler and agnostic about the kind of geometry

get_vertices = (
    m.modelgrid.get_cell_vertices
)  # function to get the referenced vertices for a model cell
geoms = [Polygon(get_vertices(i, j)) for i, j in chk.summary_array[["i", "j"]]]

geoms[0].type

geoms[0].exterior

geoms[0].bounds

geoms[0].plot()  # this feature requires descartes

# ### write the shapefile
# * the projection (.prj) file can be written using an epsg code
# * or copied from an existing .prj file

# +
from pathlib import Path

recarray2shp(chk.summary_array, geoms, os.path.join(workspace, "test.shp"), crs=26715)
shape_path = os.path.join(workspace, "test.prj")

# + pycharm={"name": "#%%\n"}
shutil.copy(shape_path, os.path.join(workspace, "26715.prj"))
recarray2shp(
    chk.summary_array,
    geoms,
    os.path.join(workspace, "test.shp"),
    prjfile=os.path.join(workspace, "26715.prj"),
)
# -

# ### read it back in
# * flopy geometry objects representing the shapes are stored in the 'geometry' field

# + pycharm={"name": "#%%\n"}
ra = shp2recarray(os.path.join(workspace, "test.shp"))
ra

# + pycharm={"name": "#%%\n"}
ra.geometry[0].plot()
# -

# ## Other geometry types
#
# ### Linestring
# * create geometry objects for pathlines from a MODPATH simulation
# * plot the paths using the built in plotting method

pthfile = PathlineFile("../../examples/data/mp6/EXAMPLE-3.pathline")
pthdata = pthfile._data.view(np.recarray)

# +
length_mult = 1.0  # multiplier to convert coordinates from model to real world
rot = 0  # grid rotation

particles = np.unique(pthdata.particleid)
geoms = []
for pid in particles:
    ra = pthdata[pthdata.particleid == pid]

    x, y = geometry.rotate(
        ra.x * length_mult, ra.y * length_mult, grid.xoffset, grid.yoffset, rot
    )
    z = ra.z
    geoms.append(LineString(list(zip(x, y, z))))
# -

geoms[0]

geoms[0].plot()

# + tags=["nbsphinx-thumbnail"]
fig, ax = plt.subplots()
for g in geoms:
    g.plot(ax=ax)
ax.autoscale()
ax.set_aspect(1)
# -

# ## Points

eptfile = EndpointFile("../../examples/data/mp6/EXAMPLE-3.endpoint")
eptdata = eptfile.get_alldata()

# +
x, y = geometry.rotate(
    eptdata["x0"] * length_mult,
    eptdata["y0"] * length_mult,
    grid.xoffset,
    grid.yoffset,
    rot,
)
z = eptdata["z0"]

geoms = [Point(x[i], y[i], z[i]) for i in range(len(eptdata))]
# -

fig, ax = plt.subplots()
for g in geoms:
    g.plot(ax=ax)
ax.autoscale()
ax.set_aspect(2e-6)

# + pycharm={"name": "#%%\n"}
try:
    # ignore PermissionError on Windows
    temp_dir.cleanup()
except:
    pass
