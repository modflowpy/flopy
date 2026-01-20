# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.13.5
#   metadata:
#     authors:
#     - name: Andy Leaf
#     section: export
# ---

# # Shapefile export demo
# The goal of this notebook is to demonstrate ways to export model information to shapefiles.
# This example will cover:
# * basic exporting of information for a model, individual package, or dataset
# * custom exporting of combined data from different packages
# * general exporting and importing of geographic data from other sources

import os
import sys

# +
from pathlib import Path
from tempfile import TemporaryDirectory

import geopandas as gpd
import git
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pooch

import flopy

print(sys.version)
print(f"numpy version: {np.__version__}")
print(f"matplotlib version: {mpl.__version__}")
print(f"flopy version: {flopy.__version__}")

# +
# temporary directory
temp_dir = TemporaryDirectory()
outdir = Path(temp_dir.name) / "shapefile_export"
outdir.mkdir(exist_ok=True)

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

# load an existing model
model_ws = data_path / sim_name
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

# ## Declarative export using `.to_geodataframe()` method
# #### Export the whole model to a single shapefile

fname = f"{outdir}/model.shp"
gdf = m.to_geodataframe(shorten_attr=True)
gdf.to_file(fname)

ax = plt.subplot(1, 1, 1, aspect="equal")
extents = grid.extent
pc = flopy.plot.plot_shapefile(fname, ax=ax, edgecolor="k", facecolor="none")
ax.set_xlim(extents[0], extents[1])
ax.set_ylim(extents[2], extents[3])
ax.set_title(fname)

fname = f"{outdir}/wel.shp"
gdf = m.wel.to_geodataframe()
gdf.to_file(fname)

# ### Export a package to a shapefile

# ### Export a FloPy list or array object

m.lpf.hk

fname = f"{outdir}/hk.shp"
gdf = m.lpf.hk.to_geodataframe()
gdf.to_file(fname)

ax = plt.subplot(1, 1, 1, aspect="equal")
extents = grid.extent
a = m.lpf.hk.array.ravel()
pc = flopy.plot.plot_shapefile(fname, ax=ax, a=a)
ax.set_xlim(extents[0], extents[1])
ax.set_ylim(extents[2], extents[3])
ax.set_title(fname)

m.riv.stress_period_data

gdf = m.riv.stress_period_data.to_geodataframe()
gdf.to_file(f"{outdir}/riv_spd.shp")

# ### MfList.to_geodataframe() exports the whole grid by default, regardless of the locations of the boundary cells
# `full_grid=False` only exports the boundary cells in the MfList

gdf = m.riv.stress_period_data.to_geodataframe(full_grid=False)
gdf.to_file(f"{outdir}/riv_spd.shp")

gdf = m.wel.stress_period_data.to_geodataframe(full_grid=False)
gdf.to_file(f"{outdir}/wel_spd.shp")

# ## Ad-hoc exporting using `to_geodataframe()`

# ### combining data from different packages
# write a shapefile of RIV and WEL package cells

gdf = m.wel.stress_period_data.to_geodataframe(shorten_attr=True)
gdf = m.riv.stress_period_data.to_geodataframe(gdf=gdf, shorten_attr=True)
gdf.head()

# ##### write the shapefile

fname = f"{outdir}/bcs.shp"
gdf.to_file(fname)

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
from flopy.utils.geospatial_utils import GeoSpatialCollection

geoms = [Point(x, y) for x, y in zip(welldata.x_utm, welldata.y_utm)]
geoms = GeoSpatialCollection(geoms, "Point").shape

gdf = gpd.GeoDataFrame(welldata, geometry=geoms)
gdf.to_file(f"{outdir}/wel_data.shp")
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

# make up a linestring shapefile of the river reaches
i, j = m.riv.stress_period_data[0].i, m.riv.stress_period_data[0].j
x0 = grid.xyzcellcenters[0][i[0], j[0]]
x1 = grid.xyzcellcenters[0][i[-1], j[-1]]
y0 = grid.xyzcellcenters[1][i[0], j[0]]
y1 = grid.xyzcellcenters[1][i[-1], j[-1]]
x = np.linspace(x0, x1, m.nrow + 1)
y = np.linspace(y0, y1, m.nrow + 1)
l0 = zip(list(zip(x[:-1], y[:-1])), list(zip(x[1:], y[1:])))
lines = [LineString(l) for l in l0]

gdf = gpd.GeoDataFrame(data=m.riv.stress_period_data[0], geometry=lines)
gdf["reach"] = np.arange(len(lines))
gdf = gdf.set_crs(epsg=grid.epsg)

lines_shapefile = f"{outdir}/riv_reaches.shp"
gdf.to_file(lines_shapefile)

# -

ax = plt.subplot(1, 1, 1, aspect="equal")
extents = grid.extent
pc = flopy.plot.plot_shapefile(lines_shapefile, ax=ax, radius=25)
ax.set_xlim(extents[0], extents[1])
ax.set_ylim(extents[2], extents[3])
ax.set_title(lines_shapefile)

# #### read in the GIS coverage using geopandas
# `read_file()` reads a geospatial file into a GeoDataFrame

linesdata = gpd.read_file(lines_shapefile)
linesdata.head()

# ##### Suppose we have some flow information that we read in from the cell budget file

# make up some fluxes between the river and aquifer at each reach
q = np.random.randn(len(linesdata)) + 1
q

# ##### Add reachs fluxes and cumulative flow to lines DataFrame

linesdata["qreach"] = q
linesdata["qstream"] = np.cumsum(q)
linesdata.to_file(lines_shapefile)

ax = plt.subplot(1, 1, 1, aspect="equal")
extents = grid.extent
pc = flopy.plot.plot_shapefile(lines_shapefile, ax=ax, radius=25)
ax.set_xlim(extents[0], extents[1])
ax.set_ylim(extents[2], extents[3])
ax.set_title(lines_shapefile)

# ## Overriding the model's modelgrid with a user supplied modelgrid
#
# In some cases it may be necessary to override the model's modelgrid instance with a seperate modelgrid. An example of this is if the model discretization is in feet and the user would like it projected in meters. Exporting can be accomplished by supplying a GeoDataFrame in the `to_geodataframe()` methods within flopy. Below is an example:

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
gdf = modelgrid.to_geodataframe()

gdf = m.to_geodataframe(gdf=gdf)
gdf.to_file(f"{outdir}/freyberg.shp")
# -

# And for a specific parameter the method is the same

fname = f"{outdir}/hk.shp"
gdf = modelgrid.to_geodataframe()
gdfhk = m.lpf.hk.to_geodataframe(gdf=gdf)
gdfhk.to_file(fname)

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
