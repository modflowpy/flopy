#!/usr/bin/env python
# coding: utf-8

# # Creating HFB input data from shapefile linework
#
# This notebook shows examples of how to create a horizontal flow barrier package (HFB) from shapefile information in FloPy. FloPy has support for creating HFB inputs from LineString like vector data for all discretization types (`DIS`, `DISV`, `DISU`).
#
# This notebook starts off by generating two simple models:
#    - a structured (DIS) model
#    - a vertex (DISV) model
#
# And then shows how to generate HFB input for each of these models.

# In[1]:


from tempfile import TemporaryDirectory

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString

import flopy
from flopy.utils import make_hfb_array

temp_dir = TemporaryDirectory()
workspace = temp_dir.name


# ### Generate a Structured (DIS) model

# In[2]:


def simple_structured_model():
    """
    Method to generate a 10x10 structured grid model
    """
    lx = 100
    ly = 100
    nlay = 1
    nrow = 10
    ncol = 10
    delc = np.full((nrow,), ly / nrow)
    delr = np.full((ncol,), ly / ncol)
    top = np.full((nrow, ncol), 10)
    botm = np.zeros((nlay, nrow, ncol))
    idomain = np.ones(botm.shape, dtype=int)

    sim = flopy.mf6.MFSimulation(sim_ws="tmp_struct")
    ims = flopy.mf6.ModflowIms(sim, complexity="SIMPLE")
    tdis = flopy.mf6.ModflowTdis(sim)

    gwf = flopy.mf6.ModflowGwf(sim, modelname="hfb_model")
    dis = flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delc=delc,
        delr=delr,
        top=top,
        botm=botm,
        idomain=idomain,
    )
    npf = flopy.mf6.ModflowGwfnpf(gwf)
    sto = flopy.mf6.ModflowGwfsto(gwf)
    ic = flopy.mf6.ModflowGwfic(gwf, strt=np.full((nlay, nrow, ncol), 9))

    return sim


# ### Generate a voronoi (DISV) model

# In[3]:


def simple_vertex_model(workspace):
    """
    Method to generate a voroni vertex grid model
    """
    from flopy.utils.triangle import Triangle
    from flopy.utils.voronoi import VoronoiGrid

    geom = [(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)]
    tri = Triangle(angle=30, model_ws=workspace)
    tri.add_polygon(geom)
    tri.add_region((5, 5), 0, maximum_area=40)
    tri.build()

    vor = VoronoiGrid(tri)
    pkg_props = vor.get_disv_gridprops()
    ncpl = pkg_props["ncpl"]
    nlay = 1
    top = np.full((ncpl,), 10)
    botm = np.zeros((nlay, ncpl))
    idomain = np.ones(botm.shape, dtype=int)

    sim = flopy.mf6.MFSimulation(sim_ws="tmp_struct")
    ims = flopy.mf6.ModflowIms(sim, complexity="SIMPLE")
    tdis = flopy.mf6.ModflowTdis(sim)

    gwf = flopy.mf6.ModflowGwf(sim, modelname="hfb_model")
    disv = flopy.mf6.ModflowGwfdisv(
        gwf, nlay=1, top=top, botm=botm, idomain=idomain, **pkg_props
    )
    npf = flopy.mf6.ModflowGwfnpf(gwf)
    sto = flopy.mf6.ModflowGwfsto(gwf)
    ic = flopy.mf6.ModflowGwfic(gwf, strt=np.full((nlay, ncpl), 9))

    return sim


# ### Create HFB inputs for a Structured (DIS) model

# The `make_hfb_array` utility is used to intersect `LineString` like data with `modelgrid` instances to produce HFB data. The `make_hfb_array` utility accepts two parameters:
#    - `modelgrid` : a flopy Grid instance
#    - `geom` : a geospatial LineString like object that can be: an iterable of points, GeoJSON, Shapely.geometry.LineString, or a shapefile.Shape object.

# First define a fault line and visualize it with the model

# In[4]:


# define a fault line
fault = [(0, 4), (50, 55), (100, 55)]

# build simulation
sim = simple_structured_model()
gwf = sim.get_model()
modelgrid = gwf.modelgrid

fig, ax = plt.subplots(figsize=(5, 5))
pmv = flopy.plot.PlotMapView(modelgrid=modelgrid, ax=ax)
pmv.plot_grid()
plt.plot(np.array(fault).T[0], np.array(fault).T[1], "r--")
# now create hfb data with the `make_hfb_array` utility

# In[5]:


hfbs = flopy.utils.make_hfb_array(modelgrid, fault)
hfbs[0:5]


# notice that the `hydchr` parameter is not filled out; this is left up to the user

# In[6]:


hfbs["hydchr"] = 1e-05


# Now we can build an HFB package and compare it to the fault line that was defined above

# In[7]:


hfb = flopy.mf6.ModflowGwfhfb(gwf, stress_period_data={0: hfbs})


# In[8]:


# plot up the results
fig, ax = plt.subplots(figsize=(5, 5))
pmv = flopy.plot.PlotMapView(modelgrid=modelgrid)
pmv.plot_grid()
pmv.plot_bc(package=hfb, color="b", lw=2)
plt.plot(np.array(fault).T[0], np.array(fault).T[1], "r--")
# ### Create HFB inputs for a Vertex (DISV) model

# The `make_hfb_array` utility is used to intersect `LineString` like data with `modelgrid` instances to produce HFB data. The `make_hfb_array` utility accepts two parameters:
#    - `modelgrid` : a flopy Grid instance
#    - `geom` : a geospatial LineString like object that can be: an iterable of points, GeoJSON, Shapely.geometry.LineString, or a shapefile.Shape object.

# First define a fault line and visualize it with the model

# In[9]:


# define a fault line
fault = [(0, 4), (50, 55), (100, 55)]

# build simulation
sim = simple_vertex_model(workspace)
gwf = sim.get_model()
modelgrid = gwf.modelgrid

fig, ax = plt.subplots(figsize=(5, 5))
pmv = flopy.plot.PlotMapView(modelgrid=modelgrid, ax=ax)
pmv.plot_grid()
plt.plot(np.array(fault).T[0], np.array(fault).T[1], "r--")
# now create hfb data with the `make_hfb_array` utility

# In[10]:


hfbs = flopy.utils.make_hfb_array(modelgrid, fault)
hfbs[0:5]


# notice that the `hydchr` parameter is not filled out; this is left up to the user

# In[11]:


hfbs["hydchr"] = 3e-06


# Now we can build an HFB package and compare it to the fault line that was defined above

# In[12]:


hfb = flopy.mf6.ModflowGwfhfb(gwf, stress_period_data={0: hfbs})


# In[13]:


# plot up the results
fig, ax = plt.subplots(figsize=(5, 5))
pmv = flopy.plot.PlotMapView(modelgrid=modelgrid)
pmv.plot_grid()
pmv.plot_bc(package=hfb, color="b", lw=2)
plt.plot(np.array(fault).T[0], np.array(fault).T[1], "r--")
# ### Working with multiple fault geometries
#
# The `make_hfb_array` only accepts a single LineString geometry at a time. This was done by design for a couple reasons.
#    1) This gives the user an opportunity to set the `hydchr` parameter to a unique value for each fault line
#    2) This also gives the user an opportunity to save the boundary condition data (cellid1, cellid2) to an external file for each fault line that can be useful during model calibration with automated software (e.g., PEST++)
#
# As a result of these design descisions recarray data will need to be concatenated when working with multiple LineString geometries.

# For this example, a geodataframe of two fault lines is created and then used to build a HFB package

# In[14]:


geom1 = LineString([(0, 4), (50, 44)])
geom2 = LineString([(50, 44), (88, 100)])
d = {"name": ["fault1", "fault2"], "geometry": [geom1, geom2]}
gdf = gpd.GeoDataFrame(d)
gdf


# Visualize the faults on the modelgrid

# In[15]:


# build simulation
sim = simple_structured_model()
gwf = sim.get_model()
modelgrid = gwf.modelgrid

fig, ax = plt.subplots(figsize=(5, 5))
pmv = flopy.plot.PlotMapView(modelgrid=modelgrid, ax=ax)
pmv.plot_grid()
gdf.geometry.plot(color="r", linestyle="--", ax=ax)
# now create hfb data with the `make_hfb_array` utility and concatenate it

# In[16]:


hydchrs = {"fault1": 1e-05, "fault2": 1e-06}

hfb_ras = []
for name, geom in zip(gdf.name, gdf.geometry):
    hfbs = make_hfb_array(modelgrid, geom)
    hfbs["hydchr"] = hydchrs[name]
    hfb_ras.append(hfbs)

hfbs = np.concatenate(hfb_ras)
hfbs = hfbs.view(np.recarray)
hfbs


# Now create the hfb package from this data and visualize it for comparison with the fault lines

# In[17]:


hfb = flopy.mf6.ModflowGwfhfb(gwf, stress_period_data={0: hfbs})


# In[18]:


# plot up the results
fig, ax = plt.subplots(figsize=(5, 5))
pmv = flopy.plot.PlotMapView(modelgrid=modelgrid)
pmv.plot_grid()
pmv.plot_bc(package=hfb, color="b", lw=2)
gdf.geometry.plot(color="r", linestyle="--", ax=ax)
# In[19]:


try:
    temp_dir.cleanup()
except PermissionError:
    # can occur on windows: https://docs.python.org/3/library/tempfile.html#tempfile.TemporaryDirectory
    pass
