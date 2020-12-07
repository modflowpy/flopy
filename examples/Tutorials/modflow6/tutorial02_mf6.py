# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # MODFLOW 6 Tutorial 2: Working with Package Variables
#
# This tutorial shows how to view access and change the underlying package
# variables for MODFLOW 6 objects in flopy.  Interaction with a FloPy
# MODFLOW 6 model is different from other models, such as MODFLOW-2005,
# MT3D, and SEAWAT, for example.

# ## Package Import

import os
import numpy as np
import matplotlib.pyplot as plt
import flopy

# ## Create Simple Demonstration Model
#
# This tutorial uses a simple demonstration simulation with one GWF Model.
# The model has 3 layers, 4 rows, and 5 columns.  The model is set up to
# use multiple model layers in order to demonstrate some of the layered
# functionality in FloPy.

name = "tutorial02_mf6"
sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=".")
flopy.mf6.ModflowTdis(
    sim, nper=10, perioddata=[[365.0, 1, 1.0] for _ in range(10)]
)
flopy.mf6.ModflowIms(sim)
gwf = flopy.mf6.ModflowGwf(sim, modelname=name, save_flows=True)
flopy.mf6.ModflowGwfdis(gwf, nlay=3, nrow=4, ncol=5)
flopy.mf6.ModflowGwfic(gwf)
flopy.mf6.ModflowGwfnpf(gwf, save_specific_discharge=True)
flopy.mf6.ModflowGwfchd(
    gwf, stress_period_data=[[(0, 0, 0), 1.0], [(2, 3, 4), 0.0]]
)
budget_file = name + ".bud"
head_file = name + ".hds"
flopy.mf6.ModflowGwfoc(
    gwf,
    budget_filerecord=budget_file,
    head_filerecord=head_file,
    saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
)
print("Done creating simulation.")

# ## Accessing Simulation-Level Packages, Models, and Model Packages
#
# At this point a simulation is available in memory.  In this particular case
# the simulation was created directly using Python code; however, the
# simulation might also have been loaded from existing model files using
# the `flopy.mf6.MFSimulation.load()` function.

# Once a MODFLOW 6 simulation is available in memory, the contents of the
# simulation object can be listed using a simple print command

print(sim)

# Simulation-level packages, models, and model packages are shown from the
# when printing the simulation object.  In this case, you should see the
# all of the contents of sim and some information about each FloPy object
# that is part of sim.

# To get the tdis package and print the contents, we can do the following

tdis = sim.tdis
print(tdis)

# To get the Iterative Model Solution (IMS) object, we use the following
# syntax

ims = sim.get_package("ims_-1")
print(ims)

# Or because there is only one IMS object for this simulation, we can
# access it as

ims = sim.get_package("ims")
print(ims)

# When printing the sim object, there is also a simulation package called
# nam.  This package contains the information that is written to the mfsim.nam
# file, which is the primary file that MODFLOW 6 reads when it first starts

nam = sim.get_package("nam")
print(nam)

# To see the models that are contained within the simulation, we can get a
# list of their names as follows

print(sim.model_names)

# sim.model_names returns the keys of an ordered dictionary, which isn't very
# useful to us, but we can convert that to a list and then go through that
# list and print information about each model in the simulation.  In this
# case there is only one model, but had there been more models, we would
# see them listed here

model_names = list(sim.model_names)
for mname in model_names:
    print(mname)

# If we want to get a model from a simulation, then we use the get_model()
# method of the sim object.  Here we go through all the models in the
# simulation and print the model name and the model type.

model_names = list(sim.model_names)
for mname in model_names:
    m = sim.get_model(mname)
    print(m.name, m.model_type)

# For this simple case here with only one GWF model, we can very easily get
# the FloPy representation of the GWF model as

gwf = sim.get_model("tutorial02_mf6")

# Now that we have the gwf object, we can print it, and see what's it
# contains.

print(gwf)

# What we see here is the information that we saw when we printed the sim
# object.

# One of the most common operations on a model is to see what packages are in
# in and then get packages of interest.  A list of packages in a model can
# obtained as

package_list = gwf.get_package_list()
print(package_list)

# As you might expect we can access each package in this list with
# gwf.get_package().  Thus, the following syntax can be used to obtain and
# print the contents of the Discretization Package

dis = gwf.get_package("dis")
print(dis)

# The Python type for this dis package is simply

print(type(dis))

# ## FloPy MODFLOW 6 Scalar Variables (MFScalar)
#
# Once we are able to get any package from a FloPy simulation object, a next
# step is to be able to access individual package variables.  We could see the
# variables in the dis object when we used the print(dis) command.  If we
# wanted to see information about ncol, for example, we print it as

print(dis.ncol)

# By doing so, we get information about ncol, but we do not get an integer
# value.  Instead we get a representation of ncol and how it is stored in
# FloPy.  The type of ncol is

print(type(dis.ncol))

# When we print the of ncol, we see that it is an MFScalar type.  This is a
# specific type of information used by FloPy to represent variables that are
# stored in in MODFLOW 6 objects.

# If we want to get a more useful representation of ncol, then we use the
# get_data() method, which is available on all of the underlying flopy
# package variables.  Thus, for ncol, we can get the integer value and the
# type that is returned as

ncol = dis.ncol.get_data()
print(ncol)
print(type(ncol))

# ## FloPy MODFLOW 6 Arrays (MFArray)
#
# The dis object also has several arrays stored with it.  For example the
# discretization has botm, which stores the bottom elevation for every model
# cell in the grid.  We can see the type of botm as

print(type(dis.botm))

# The MFArray class in flopy is designed to efficiently store array information
# for MODFLOW 6 variables.  In this case, dis.botm has a constant value of
# zero, which we can see with

print(dis.botm)

# If we want to get an array representation of dis.botm, we can use

print(dis.botm.get_data())

# or

print(dis.botm.array)

# In both of these cases, a full three-dimensional array is created on the fly
# and provided back to the user.  In this particular case, the returned arrays
# are the same; however, if a factor is assigned to this array, as will be
# seen later, dis.botm.array will have values with the factor applied whereas
# dis.botm.get_data() will be the data values stored in the array without the
# factor applied.

# We can see here that the shape of the returned array is correct and is
# (nlay, nrow, ncol)

print(dis.botm.array.shape)

# and that it is a numpy array

print(type(dis.botm.array))

# We can easily change the value of this array as a constant using the
# set_data() method

dis.botm.set_data(-10)
print(dis.botm)

# Or alternatively, we could call set_data() with an array as long as it is
# the correct shape

shp = dis.botm.array.shape
a = np.arange(shp[0] * shp[1] * shp[2]).reshape(shp)
dis.botm.set_data(a)
print(dis.botm.array)

# We even have an option to see how dis.botm will be written to the MODFLOW 6
# Discretization input file

print(dis.botm.get_file_entry())

# ### Layered Data
#
# When we look at what will be written to the discretization input file, we
# see that the entire dis.botm array is written as one long array with the
# number of values equal to nlay * nrow * ncol.  And this whole-array
# specification may be of use in some cases.  Often times, however, it is
# easier to work with each layer separately.  An MFArray object, such as
# dis.botm can be set to be a layered array as follows

dis.botm.make_layered()

# By changing dis.botm to layered, we are then able to manage each layer
# separately.  Before doing so, however, we need to pass in data that can be
# separated into three layers.  An array of the correct size is one option

a = np.arange(shp[0] * shp[1] * shp[2]).reshape(shp)
dis.botm.set_data(a)

# Now that dis.botm has been set to be layered, if we print information about
# it, we see that each layer is stored separately, however, dis.botm.array
# will still return a full three-dimensional array.

print(type(dis.botm))
print(dis.botm)

# We also see that each layer is printed separately to the Discretization
# Package input file, and that the LAYERED keyword is activated:

print(dis.botm.get_file_entry())

# Working with a layered array provides lots of flexibility.  For example,
# constants can be set for some layers, but arrays for others:

dis.botm.set_data([-1, -a[2], -200])
print(dis.botm.get_file_entry())

# To gain full control over an individual layers, layer information can be
# provided as a dictionary:

a0 = {"factor": 0.5, "iprn": 1, "data": np.ones((4, 5))}
a1 = -100
a2 = {"factor": 1.0, "iprn": 14, "data": -100 * np.ones((4, 5))}
dis.botm.set_data([a0, a1, a2])
print(dis.botm.get_file_entry())

# Here we say that the FACTOR has been set to 0.5 for the first layer and an
# alternative print flag is set for the last layer.
#
# Because we are specifying a factor for the top layer, we can also see that
# the get_data() method returns the array without the factor applied

print(dis.botm.get_data())

# whereas .array returns the array with the factor applied

print(dis.botm.array)

# ### External Files
#
# If we want to store the bottom elevations for the bottom layer to an
# external file, then the dictionary that we pass in for the third layer
# can be given a filename keyword

a0 = {"factor": 0.5, "iprn": 1, "data": np.ones((4, 5))}
a1 = -100
a2 = {
    "filename": "dis.botm.3.txt",
    "factor": 2.0,
    "iprn": 1,
    "data": -100 * np.ones((4, 5)),
}
dis.botm.set_data([a0, a1, a2])
print(dis.botm.get_file_entry())

# And we can even have our data be stored in binary format by adding
# a 'binary' key to the layer dictionary and setting it's value to True.

a0 = {"factor": 0.5, "iprn": 1, "data": np.ones((4, 5))}
a1 = -100
a2 = {
    "filename": "dis.botm.3.bin",
    "factor": 2.0,
    "iprn": 1,
    "data": -100 * np.ones((4, 5)),
    "binary": True,
}
dis.botm.set_data([a0, a1, a2])
print(dis.botm.get_file_entry())
print(dis.botm.array)

# Note that we could have also specified botm this way as part of the
# original flopy.mf6.ModflowGwfdis constructor:

a0 = {"factor": 0.5, "iprn": 1, "data": np.ones((4, 5))}
a1 = -100
a2 = {
    "filename": "dis.botm.3.bin",
    "factor": 2.0,
    "iprn": 1,
    "data": -100 * np.ones((4, 5)),
    "binary": True,
}
botm = [a0, a1, a2]
flopy.mf6.ModflowGwfdis(gwf, nlay=3, nrow=4, ncol=5, botm=botm)

# ## Stress Period Data (MFTransientList)
#
# Data that varies during a simulation is often stored in flopy in a special
# data structured call an MFTransientList.  We can see how one of these behaves
# by looking at the stress period data for the constant head package.  We
# can access the constant head package by getting it from the GWF model using
# the package name:

chd = gwf.get_package("chd_0")
print(chd)

# We can now look at the type and contents of the stress period data
print(type(chd.stress_period_data))
print(chd.stress_period_data)

# We can get a dictionary of the stress period data using the get_data()
# method:

spd = chd.stress_period_data.get_data()
print(spd)

# Here we see that they key in the dictionary is the stress period number,
# which is zero based and the value in the dictionary is a numpy recarray,
# which is a numpy array that is optimized to store columns of infomration.
# The first column contains a tuple, which is the layer, row, and column
# number.  The second column contains the head value.

# more to come...

# ## Time Series

# more to come...

# ## Observations

# more to come...

# ## Activating External Output

# more to come...

# ## Plotting

# more to come...
