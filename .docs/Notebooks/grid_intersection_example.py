# %%
# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: flopy
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
#     version: 3.13.7
#   metadata:
#     section: dis
# ---

# %% [markdown]
# # <a id="top"></a>Intersecting model grids with shapes
#
# _Note: This feature requires the shapely package (which is an optional FloPy dependency)._
#
# This notebook shows the grid intersection functionality in flopy. The
# intersection methods are available through the `GridIntersect` object. A flopy
# modelgrid is passed to instantiate the object. Then the modelgrid can be
# intersected with Points, LineStrings and Polygons and their Multi variants.
#
# ### Table of Contents
# - [GridIntersect Class](#gridclass)
# - [Rectangular regular grid](#rectgrid)
#     - [Polygon with regular grid](#rectgrid.1)
#     - [MultiLineString with regular grid](#rectgrid.2)
#     - [MultiPoint with regular grid](#rectgrid.3)
# - [Vertex grid](#trigrid)
#     - [Polygon with triangular grid](#trigrid.1)
#     - [MultiLineString with triangular grid](#trigrid.2)
#     - [MultiPoint with triangular grid](#trigrid.3)

# %% [markdown]
# Import packages

# %%
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import shapely
from shapely.geometry import LineString, MultiLineString, MultiPoint, Point, Polygon

import flopy
import flopy.discretization as fgrid
import flopy.plot as fplot
from flopy.utils import GridIntersect

print(sys.version)
print(f"numpy version: {np.__version__}")
print(f"matplotlib version: {mpl.__version__}")
print(f"shapely version: {shapely.__version__}")
print(f"flopy version: {flopy.__version__}")

# %% [markdown]
# ## <a id="gridclass"></a>[GridIntersect Class](#top)
#
# The GridIntersect class is constructed by passing a flopy modelgrid object to
# the constructor. There are options users can select to change how the
# intersection is calculated.
#
# - `rtree`: either `True` (default) or `False`. When True, an STR-tree is built,
# which allows for fast spatial queries. Building the STR-tree takes some
# time however. Setting the option to False avoids building the STR-tree but requires
# the intersection calculation to loop through more candidate grid cells. It is generally
# recommended to set this option to True.
# - `local`: either `False` (default) or `True`. When True the local model coordinates
# are used. When False, real-world coordinates are used. Can be useful if shapes are
# defined in local coordinates.
#
# The important methods in the GridIntersect object are:
#
# - `intersects()`: returns cellids for gridcells that intersect a shape (accepts
# shapely geometry objects, arrays of shapely geometry objects, flopy geometry object,
# shapefile.Shape objects, and geojson objects).
# - `intersect()`: for intersecting the modelgrid with point, linestrings, and
# polygon geometries (accepts shapely geometry objects, flopy geometry object,
# shapefile.Shape objects, and geojson objects)
# - `points_to_cellids()`: for quickly locating points in the modelgrid and
# getting a 1:1 mapping of points to cellids. Especially useful when passing in array
# of points.
# - `ix.plot_intersection_result()`: for plotting intersection results
#
# In the following sections examples of intersections are shown for structured
# and vertex grids for different types of shapes (Polygon, LineString and Point).

# %% [markdown]
# ## <a id="rectgrid"></a>[Rectangular regular grid](#top)

# %%
delc = 10 * np.ones(10, dtype=float)
delr = 10 * np.ones(10, dtype=float)

# %%
xoff = 0.0
yoff = 0.0
angrot = 0.0
sgr = fgrid.StructuredGrid(
    delc,
    delr,
    top=np.ones(100).reshape((10, 10)),
    botm=np.zeros(100).reshape((1, 10, 10)),
    xoff=xoff,
    yoff=yoff,
    angrot=angrot,
)

# %%
sgr.plot()
# %% [markdown]
# ### <a id="rectgrid.1"></a>[Polygon with regular grid](#top)
# Polygon to intersect with:

# %%
p = Polygon(
    shell=[(15, 15), (20, 50), (35, 80.0), (80, 50), (80, 40), (40, 5), (15, 12)],
    holes=[[(25, 25), (25, 45), (45, 45), (45, 25)]],
)

# %%
# Create the GridIntersect class for our modelgrid.
gi = GridIntersect(sgr)

# %% [markdown]
# Do the intersect operation for a polygon

# %%
result = gi.intersect(p, geo_dataframe=False)

# %% [markdown]
# The results are returned as a geopandas.GeoDataFrame or numpy.recarray containing several fields based on the intersection performed. An explanation of the data in each of the possible columns is given below:
# - `cellid`: contains the cell ids of the intersected grid cells
# - `row`: contains the row index of the intersected grid cells (only for structured grids)
# - `col`: contains the column index of the intersected grid cells (only for structured grids)
# - `areas`: contains the area of the polygon in that grid cell (only for polygons)
# - `lengths`: contains the length of the linestring in that grid cell (only for linestrings)
# - `ixshapes`/`geometry`: contains the shapely object representing the intersected shape (useful for plotting the result)
#
# __Note__: The `cellids` column is deprecated since flopy 3.11 but still included in the result for backward compatibility. It contains (row, column) tuples for structured grids and cellids for vertex grids.
#
# Looking at the first few entries of the results of the polygon intersection. Note that you can convert the result to a GeoDataFrame (if geopandas is installed) with `geo_dataframe=True`.

# %%
gi.intersect(p, geo_dataframe=True).head()

# %% [markdown]
# The rows and columns can be easily obtained.

# %%
result.row, result.col

# %% [markdown]
# Or the areas

# %%
result.areas

# %% [markdown]
# If a user is only interested in which cells the shape intersects (and not the areas or
# the actual shape of the intersected object) with there is also the `intersects()`
# method. This method works for all types of shapely geometries including arrays of
# shapely geometries.
#
# This method returns `shp_id` and `cellid` columns. The `shp_id` column contains the
# index of the geometry in the original input shape provided by the user. This is useful
# when the input is an array of shapely geometries. In this case we have only one polygon,
# so the `shp_id` is always equal to 0. For structured grids the row and column indices
# are returned in the `row` and `col` columns.

# %%
gi.intersects(p, dataframe=True)

# %% [markdown]
# The results of an intersection can be visualized with the `GridIntersect.plot_intersection_result()` method.

# %%
# create a figure and plot the grid
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

# the intersection object contains some helpful plotting commands
gi.plot_intersection_result(result, ax=ax)

# add black x at cell centers
for irow, icol in zip(result.row, result.col):
    (h2,) = ax.plot(
        sgr.xcellcenters[0, icol],
        sgr.ycellcenters[irow, 0],
        "kx",
        label="centroids of intersected gridcells",
    )

# add legend
ax.legend([h2], [i.get_label() for i in [h2]], loc="best")
# %% [markdown]
# The `intersect()` method contains several keyword arguments that specifically deal with polygons:
#
# - `contains_centroid`: only store intersection result if cell centroid is contained within polygon
# - `min_area_fraction`: minimal intersecting cell area (expressed as a fraction of the total cell area) to include cells in intersection result
#
# Two examples showing the usage of these keyword arguments are shown below.
#
# Example with `contains_centroid` set to True, only cells in which centroid is within the intersected polygon are stored. Note the difference with the previous result.

# %%
# contains_centroid example

result2 = gi.intersect(p, contains_centroid=True, geo_dataframe=True)

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
gi.plot_intersection_result(result2, ax=ax)

# add black x at cell centers
for irow, icol in zip(result2.row, result2.col):
    (h2,) = ax.plot(
        sgr.xcellcenters[0, icol],
        sgr.ycellcenters[irow, 0],
        "kx",
        label="centroids of intersected gridcells",
    )

# add legend
ax.legend([h2], [i.get_label() for i in [h2]], loc="best")
# %% [markdown]
# Example with `min_area_threshold` set to 0.35, the intersection result in a cell should cover 35% or more of the cell area.

# %%
# min_area_threshold example

result3 = gi.intersect(p, min_area_fraction=0.35)

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
gi.plot_intersection_result(result3, ax=ax)

# add black x at cell centers
for irow, icol in zip(result3.row, result3.col):
    (h2,) = ax.plot(
        sgr.xcellcenters[0, icol],
        sgr.ycellcenters[irow, 0],
        "kx",
        label="centroids of intersected gridcells",
    )
# add legend
ax.legend([h2], [i.get_label() for i in [h2]], loc="best")
# %% [markdown]
# ### <a id="rectgrid.2"></a>[Polyline with regular grid](#top)
# MultiLineString to intersect with:

# %%
ls1 = LineString([(95, 105), (30, 50)])
ls2 = LineString([(30, 50), (90, 22)])
ls3 = LineString([(90, 22), (0, 0)])
mls = MultiLineString(lines=[ls1, ls2, ls3])

# %%
result = gi.intersect(mls, geo_dataframe=True)

# %% [markdown]
# Plot the result

# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
sgr.plot(ax=ax)
gi.plot_intersection_result(result, ax=ax, cmap="tab20")

for irow, icol in zip(result.row, result.col):
    (h2,) = ax.plot(
        sgr.xcellcenters[0, icol],
        sgr.ycellcenters[irow, 0],
        "kx",
        label="centroids of intersected gridcells",
    )

ax.legend([h2], [i.get_label() for i in [h2]], loc="best")
# %% [markdown]
# ### [MultiPoint with regular grid](#top)<a id="rectgrid.3"></a>
#
# MultiPoint to intersect with

# %%
mp = MultiPoint(
    points=[Point(50.0, 0.0), Point(45.0, 45.0), Point(10.0, 10.0), Point(150.0, 100.0)]
)

# %% [markdown]
# For points and linestrings there is a keyword argument `return_all_intersections` which will return multiple intersection results for points or (parts of) linestrings on cell boundaries. As an example, the difference is shown with the MultiPoint intersection. Note the number of red "+" symbols indicating the centroids of intersected cells, in the bottom left case, there are 4 results because the point lies exactly on the intersection between 4 grid cells.

# %%
result = gi.intersect(mp, geo_dataframe=True)
result_all = gi.intersect(mp, return_all_intersections=True, geo_dataframe=True)

# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
sgr.plot(ax=ax)
gi.plot_point(result.geometry, ax=ax, ms=10, color="C0")
gi.plot_point(result_all.geometry, ax=ax, ms=10, marker=".", color="C3")

for irow, icol in zip(result.row, result.col):
    (h2,) = ax.plot(
        sgr.xcellcenters[0, icol],
        sgr.ycellcenters[irow, 0],
        "kx",
        ms=15,
        label="centroids of intersected cells",
    )

for irow, icol in zip(result_all.row, result_all.col):
    (h3,) = ax.plot(
        sgr.xcellcenters[0, icol],
        sgr.ycellcenters[irow, 0],
        "C3+",
        ms=15,
        label="centroids with `return_all_intersections=True`",
    )

ax.legend([h2, h3], [i.get_label() for i in [h2, h3]], loc="best")
# %% [markdown]
# In addition to the `intersect()` and `intersects()` methods there is function to
# quickly get the cellids for many points. This is done with `points_to_cellids()`.
#
# First lets create an array containing shapely Points.

# %%
n_pts = 10
rng = np.random.default_rng(seed=42)
x_coords = rng.uniform(0, 100, n_pts)
y_coords = rng.uniform(0, 100, n_pts)
random_points = shapely.points(x_coords, y_coords)

# %% [markdown]
# Now find the cellid for each point. `points_to_cellids` will only return a single result
# for each point. In case a point is on the boundary between multiple cells, it will
# return the cell with the lowest cellid.


# %%
result = gi.points_to_cellids(random_points)
result

# %% [markdown]
# Plot the result

# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
sgr.plot(ax=ax)
ax.plot(x_coords, y_coords, "ro", ms=5, label="random points")
gi.plot_polygon(gi.geoms[result.cellid], ax=ax, fc="yellow", edgecolor="k", zorder=5)
# %% [markdown]
# Note that `points_to_cellids()` returns NA for points that lie outside the model grid.

# %%
gi.points_to_cellids(shapely.points([50, 110], [55, 50]))

# %% [markdown]
# ## <a id="trigrid"></a>[Vertex Grid](#top)

# %%
cell2d = [
    [0, 83.33333333333333, 66.66666666666667, 3, 4, 2, 7],
    [1, 16.666666666666668, 33.333333333333336, 3, 4, 0, 5],
    [2, 33.333333333333336, 83.33333333333333, 3, 1, 8, 4],
    [3, 16.666666666666668, 66.66666666666667, 3, 5, 1, 4],
    [4, 33.333333333333336, 16.666666666666668, 3, 6, 0, 4],
    [5, 66.66666666666667, 16.666666666666668, 3, 4, 3, 6],
    [6, 83.33333333333333, 33.333333333333336, 3, 7, 3, 4],
    [7, 66.66666666666667, 83.33333333333333, 3, 8, 2, 4],
]
vertices = [
    [0, 0.0, 0.0],
    [1, 0.0, 100.0],
    [2, 100.0, 100.0],
    [3, 100.0, 0.0],
    [4, 50.0, 50.0],
    [5, 0.0, 50.0],
    [6, 50.0, 0.0],
    [7, 100.0, 50.0],
    [8, 50.0, 100.0],
]
tgr = fgrid.VertexGrid(vertices, cell2d)

# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
pmv = fplot.PlotMapView(modelgrid=tgr)
pmv.plot_grid(ax=ax)
# %% [markdown]
# ### <a id="trigrid.1"></a>[Polygon with triangular grid](#top)

# %%
gi2 = GridIntersect(tgr)

# %%
result = gi2.intersect(p, geo_dataframe=True)

# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
gi2.plot_intersection_result(result, ax=ax)

# only cells that intersect with shape
for cellid in result.cellid:
    (h2,) = ax.plot(
        tgr.xcellcenters[cellid],
        tgr.ycellcenters[cellid],
        "kx",
        label="centroids of intersected gridcells",
    )

ax.legend([h2], [i.get_label() for i in [h2]], loc="best")
# %% [markdown]
# ### <a id="trigrid.2"></a>[LineString with triangular grid](#top)

# %%
result = gi2.intersect(mls, geo_dataframe=True)

# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
gi2.plot_intersection_result(result, ax=ax, lw=3)

for cellid in result.cellid:
    (h2,) = ax.plot(
        tgr.xcellcenters[cellid],
        tgr.ycellcenters[cellid],
        "kx",
        label="centroids of intersected gridcells",
    )

ax.legend([h2], [i.get_label() for i in [h2]], loc="best")
# %% [markdown]
# ### <a id="trigrid.3"></a>[MultiPoint with triangular grid](#top)

# %%
result = gi2.intersect(mp, geo_dataframe=True)
result_all = gi2.intersect(mp, return_all_intersections=True, geo_dataframe=True)

# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
gi2.plot_intersection_result(result, ax=ax, color="k", zorder=5, ms=10)

for cellid in result.cellid:
    (h2,) = ax.plot(
        tgr.xcellcenters[cellid],
        tgr.ycellcenters[cellid],
        "kx",
        ms=15,
        label="centroids of intersected cells",
    )
for cellid in result_all.cellid:
    (h3,) = ax.plot(
        tgr.xcellcenters[cellid],
        tgr.ycellcenters[cellid],
        "r+",
        ms=15,
        label="centroids with return_all_intersections=True",
    )

ax.legend([h2, h3], [i.get_label() for i in [h2, h3]], loc="best")

# %%
