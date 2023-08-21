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
#     section: export
#     authors:
#       - name: Andy Leaf
# ---

# # Shapefile export demo
# The goal of this notebook is to demonstrate ways to export model information to shapefiles.
# This example will cover:
# * basic exporting of information for a model, individual package, or dataset
# * custom exporting of combined data from different packages
# * general exporting and importing of geographic data from other sources

import os

# +
import sys
from tempfile import TemporaryDirectory

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import flopy

print(sys.version)
print(f"numpy version: {np.__version__}")
print(f"matplotlib version: {mpl.__version__}")
print(f"flopy version: {flopy.__version__}")

# +
# temporary directory
temp_dir = TemporaryDirectory()
outdir = os.path.join(temp_dir.name, "shapefile_export")

# load an existing model
model_ws = "../../examples/data/freyberg"
m = flopy.modflow.Modflow.load(
    "freyberg.nam",
    model_ws=model_ws,
    verbose=False,
    check=False,
    exe_name="mfnwt",
)
# -

m.get_package_list()

# ### set the model coordinate information
# the coordinate information where the grid is located in a projected coordinate system (e.g. UTM)

grid = m.modelgrid
grid.set_coord_info(xoff=273170, yoff=5088657, crs=26916)

grid.extent

# ## Declarative export using attached `.export()` methods
# #### Export the whole model to a single shapefile

fname = f"{outdir}/model.shp"
m.export(fname)

ax = plt.subplot(1, 1, 1, aspect="equal")
extents = grid.extent
pc = flopy.plot.plot_shapefile(fname, ax=ax, edgecolor="k", facecolor="none")
ax.set_xlim(extents[0], extents[1])
ax.set_ylim(extents[2], extents[3])
ax.set_title(fname)

fname = f"{outdir}/wel.shp"
m.wel.export(fname)

# ### Export a package to a shapefile

# ### Export a FloPy list or array object

m.lpf.hk

fname = f"{outdir}/hk.shp"
m.lpf.hk.export(f"{outdir}/hk.shp")

ax = plt.subplot(1, 1, 1, aspect="equal")
extents = grid.extent
a = m.lpf.hk.array.ravel()
pc = flopy.plot.plot_shapefile(fname, ax=ax, a=a)
ax.set_xlim(extents[0], extents[1])
ax.set_ylim(extents[2], extents[3])
ax.set_title(fname)

m.riv.stress_period_data

m.riv.stress_period_data.export(f"{outdir}/riv_spd.shp")

# ### MfList.export() exports the whole grid by default, regardless of the locations of the boundary cells
# `sparse=True` only exports the boundary cells in the MfList

m.riv.stress_period_data.export(f"{outdir}/riv_spd.shp", sparse=True)

m.wel.stress_period_data.export(f"{outdir}/wel_spd.shp", sparse=True)

# ## Ad-hoc exporting using `recarray2shp`
# * The main idea is to create a recarray with all of the attribute information, and a list of geometry features (one feature per row in the recarray)
# * each geometry feature is an instance of the `Point`, `LineString` or `Polygon` classes in `flopy.utils.geometry`. The shapefile format requires all the features to be of the same type.
# * We will use pandas dataframes for these examples because they are easy to work with, and then convert them to recarrays prior to exporting.
#

from flopy.export.shapefile_utils import recarray2shp

# ### combining data from different packages
# write a shapefile of RIV and WEL package cells

wellspd = pd.DataFrame(m.wel.stress_period_data[0])
rivspd = pd.DataFrame(m.riv.stress_period_data[0])
spd = pd.concat([wellspd, rivspd])
spd.head()

# ##### Create a list of Polygon features from the cell vertices stored in the modelgrid object

# +
from flopy.utils.geometry import Polygon

vertices = []
for row, col in zip(spd.i, spd.j):
    vertices.append(grid.get_cell_vertices(row, col))
polygons = [Polygon(vrt) for vrt in vertices]
polygons
# -

# ##### write the shapefile

fname = f"{outdir}/bcs.shp"
recarray2shp(spd.to_records(), geoms=polygons, shpname=fname, crs=grid.epsg)

ax = plt.subplot(1, 1, 1, aspect="equal")
extents = grid.extent
pc = flopy.plot.plot_shapefile(fname, ax=ax)
ax.set_xlim(extents[0], extents[1])
ax.set_ylim(extents[2], extents[3])
ax.set_title(fname)

# ### exporting other data
# Suppose we have some well data with actual locations that we want to export to a shapefile

welldata = pd.DataFrame(
    {
        "wellID": np.arange(0, 10),
        "q": np.random.randn(10) * 100 - 1000,
        "x_utm": np.random.rand(10) * 5000 + grid.xoffset,
        "y_utm": grid.yoffset + np.random.rand(10) * 10000,
    }
)
welldata.head()

# ##### convert the x, y coorindates to point features and then export

# +
from flopy.utils.geometry import Point

geoms = [Point(x, y) for x, y in zip(welldata.x_utm, welldata.y_utm)]

fname = f"{outdir}/wel_data.shp"
recarray2shp(welldata.to_records(), geoms=geoms, shpname=fname, crs=grid.epsg)
# -

ax = plt.subplot(1, 1, 1, aspect="equal")
extents = grid.extent
pc = flopy.plot.plot_shapefile(fname, ax=ax, radius=100)
ax.set_xlim(extents[0], extents[1])
ax.set_ylim(extents[2], extents[3])
ax.set_title(fname)

# ### Adding attribute data to an existing shapefile
# Suppose we have a GIS coverage representing the river in the riv package

# +
from flopy.utils.geometry import LineString

### make up a linestring shapefile of the river reaches
i, j = m.riv.stress_period_data[0].i, m.riv.stress_period_data[0].j
x0 = grid.xyzcellcenters[0][i[0], j[0]]
x1 = grid.xyzcellcenters[0][i[-1], j[-1]]
y0 = grid.xyzcellcenters[1][i[0], j[0]]
y1 = grid.xyzcellcenters[1][i[-1], j[-1]]
x = np.linspace(x0, x1, m.nrow + 1)
y = np.linspace(y0, y1, m.nrow + 1)
l0 = zip(list(zip(x[:-1], y[:-1])), list(zip(x[1:], y[1:])))
lines = [LineString(l) for l in l0]

rivdata = pd.DataFrame(m.riv.stress_period_data[0])
rivdata["reach"] = np.arange(len(lines))
lines_shapefile = f"{outdir}/riv_reaches.shp"
recarray2shp(
    rivdata.to_records(index=False),
    geoms=lines,
    shpname=lines_shapefile,
    crs=grid.epsg,
)
# -

ax = plt.subplot(1, 1, 1, aspect="equal")
extents = grid.extent
pc = flopy.plot.plot_shapefile(lines_shapefile, ax=ax, radius=25)
ax.set_xlim(extents[0], extents[1])
ax.set_ylim(extents[2], extents[3])
ax.set_title(lines_shapefile)

# #### read in the GIS coverage using `shp2recarray`
# `shp2recarray` reads a shapefile into a numpy record array, which can easily be converted to a DataFrame

from flopy.export.shapefile_utils import shp2recarray

linesdata = shp2recarray(lines_shapefile)
linesdata = pd.DataFrame(linesdata)
linesdata.head()

# ##### Suppose we have some flow information that we read in from the cell budget file

# make up some fluxes between the river and aquifer at each reach
q = np.random.randn(len(linesdata)) + 1
q

# ##### Add reachs fluxes and cumulative flow to lines DataFrame

linesdata["qreach"] = q
linesdata["qstream"] = np.cumsum(q)

recarray2shp(
    linesdata.drop("geometry", axis=1).to_records(),
    geoms=linesdata.geometry.values,
    shpname=lines_shapefile,
    crs=grid.epsg,
)

ax = plt.subplot(1, 1, 1, aspect="equal")
extents = grid.extent
pc = flopy.plot.plot_shapefile(lines_shapefile, ax=ax, radius=25)
ax.set_xlim(extents[0], extents[1])
ax.set_ylim(extents[2], extents[3])
ax.set_title(lines_shapefile)

# ## Overriding the model's modelgrid with a user supplied modelgrid
#
# In some cases it may be necessary to override the model's modelgrid instance with a seperate modelgrid. An example of this is if the model discretization is in feet and the user would like it projected in meters. Exporting can be accomplished by supplying a modelgrid as a `kwarg` in any of the `export()` methods within flopy. Below is an example:

# +
mg0 = m.modelgrid

# build a new modelgrid instance with discretization in meters
modelgrid = flopy.discretization.StructuredGrid(
    delc=mg0.delc * 0.3048,
    delr=mg0.delr * 0.3048,
    top=mg0.top,
    botm=mg0.botm,
    idomain=mg0.idomain,
    xoff=mg0.xoffset * 0.3048,
    yoff=mg0.yoffset * 0.3048,
)

# exporting an entire model
m.export(f"{outdir}/freyberg.shp", modelgrid=modelgrid)
# -

# And for a specific parameter the method is the same

fname = f"{outdir}/hk.shp"
m.lpf.hk.export(fname, modelgrid=modelgrid)

ax = plt.subplot(1, 1, 1, aspect="equal")
extents = modelgrid.extent
a = m.lpf.hk.array.ravel()
pc = flopy.plot.plot_shapefile(fname, ax=ax, a=a)
ax.set_xlim(extents[0], extents[1])
ax.set_ylim(extents[2], extents[3])
ax.set_title(fname)

try:
    # ignore PermissionError on Windows
    temp_dir.cleanup()
except:
    pass
