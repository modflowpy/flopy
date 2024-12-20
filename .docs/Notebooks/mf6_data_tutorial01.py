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
#   metadata:
#     section: mf6
# ---

# # MODFLOW 6: Accessing Simulation Settings, Models, and Packages
#
# This tutorial shows how to view, access, and change the underlying package
# variables for MODFLOW 6 objects in FloPy.  Interaction with a FloPy
# MODFLOW 6 model is different from other models, such as MODFLOW-2005,
# MT3D, and SEAWAT, for example.
#
# The MODFLOW 6 simulation structure is arranged in the following
# generalized way:
#
# >       SIMULATION --> PACKAGE --> Data
# >
# >       SIMULATION --> MODEL --> PACKAGE (--> PACKAGE) --> Data
#
# This tutorial focuses on accessing simulation-wide FloPy settings and
# how to create and access models and packages.  Tutorial 3, 4, and 5 offer a
# more in depth look at observation, time series, and time array series
# packages, and tutorial 6, 7, 8, and 9 offer a more in depth look at the data.

# ## Create Simple Demonstration Model
#
# This tutorial uses a simple demonstration simulation with one GWF Model.
# The model has 3 layers, 4 rows, and 5 columns.  The model is set up to
# use multiple model layers in order to demonstrate some of the layered
# functionality in FloPy.

# package import
from tempfile import TemporaryDirectory

import flopy

temp_dir = TemporaryDirectory()
workspace = temp_dir.name
name = "tutorial01"

# set up simulation and basic packages
sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=workspace)
flopy.mf6.ModflowTdis(sim, nper=10, perioddata=[[365.0, 1, 1.0] for _ in range(10)])
flopy.mf6.ModflowIms(sim)
gwf = flopy.mf6.ModflowGwf(sim, modelname=name, save_flows=True)
botm = [30.0, 20.0, 10.0]
flopy.mf6.ModflowGwfdis(gwf, nlay=3, nrow=4, ncol=5, top=50.0, botm=botm)
flopy.mf6.ModflowGwfic(gwf)
flopy.mf6.ModflowGwfnpf(gwf, save_specific_discharge=True)
flopy.mf6.ModflowGwfchd(gwf, stress_period_data=[[(0, 0, 0), 1.0], [(2, 3, 4), 0.0]])
budget_file = f"{name}.bud"
head_file = f"{name}.hds"
flopy.mf6.ModflowGwfoc(
    gwf,
    budget_filerecord=budget_file,
    head_filerecord=head_file,
    saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
)
print("Done creating simulation.")

# ## Accessing Simulation-Level Settings
#
# FloPy has a number of settings that can be set for the entire simulation.
# These include how much information FloPy writes to the console, how to
# format the MODFLOW package files, and whether to verify MODFLOW data.

# The verbosity level, which determines how much FloPy writes to command line
# output.  The options are 1 for quiet, 2 for normal, and 3 for verbose.
# Below we set the verbosity level to verbose.

sim.simulation_data.verbosity_level = 3

# We can also set the number of spaces to indent data when writing package
# files by setting the indent string.

sim.simulation_data.indent_string = "    "

# Next we set the precision and number of characters written for floating
# point variables.

sim.float_precision = 8
sim.float_characters = 15

# Lastly, we disable verify_data and auto_set_sizes for faster performance.
# With these options disabled FloPy will not do any checking or autocorrecting
# of your data.

sim.verify_data = False
sim.auto_set_sizes = False

# ## Accessing Models and Packages
#
# At this point a simulation is available in memory.  In this particular case
# the simulation was created directly using Python code; however, the
# simulation might also have been loaded from existing model files using
# the `FloPy.mf6.MFSimulation.load()` function.

# Once a MODFLOW 6 simulation is available in memory, the contents of the
# simulation object can be listed using a simple print command.

print(sim)

# Simulation-level packages, models, and model packages can be shown by
# printing the simulation object.  In this case, you should see the
# all of the contents of simulation and some information about each FloPy
# object that is part of simulation.

# To get the `TDIS` package and print the contents, we can do the following

tdis = sim.tdis
print(tdis)

# To get the Iterative Model Solution (`IMS`) object, we use the following
# syntax

ims = sim.get_package("ims_-1")
print(ims)

# Or because there is only one `IMS` object for this simulation, we can
# access it as

ims = sim.get_package("ims")
print(ims)

# When printing the sim object, there is also a simulation package called
# nam.  This package contains the information that is written to the
# `mfsim.nam` file, which is the primary file that MODFLOW 6 reads when it
# first starts.  The nam package is automatically updated for you by FloPy and
# does not require modification.

nam = sim.get_package("nam")
print(nam)

# To see the models that are contained within the simulation, we can get a
# list of their names as follows

print(sim.model_names)

# `sim.model_names` returns the keys of an ordered dictionary, which isn't very
# useful to us, but we can convert that to a list and then go through that
# list and print information about each model in the simulation.  In this
# case there is only one model, but had there been more models, we would
# see them listed here

model_names = list(sim.model_names)
for mname in model_names:
    print(mname)

# If we want to get a model from a simulation, then we use the `get_model()`
# method of the sim object.  Here we go through all the models in the
# simulation and print the model name and the model type.

model_names = list(sim.model_names)
for mname in model_names:
    m = sim.get_model(mname)
    print(m.name, m.model_type)

# For this simple case here with only one `GWF` model, we can very easily get
# the FloPy representation of the `GWF` model as

gwf = sim.get_model(name)

# Now that we have the `GWF` object, we can print it, and see what's it
# contains.

print(gwf)

# What we see here is the information that we saw when we printed the sim
# object.

# One of the most common operations on a model is to see what packages are in
# it and then get packages of interest.  A list of packages in a model can
# obtained as

package_list = gwf.get_package_list()
print(package_list)

# As you might expect we can access each package in this list with
# gwf.get_package().  Thus, the following syntax can be used to obtain and
# print the contents of the `DIS` Package

dis = gwf.get_package("dis")
print(dis)

# The Python type for this dis package is simply

print(type(dis))

try:
    temp_dir.cleanup()
except PermissionError:
    # can occur on windows: https://docs.python.org/3/library/tempfile.html#tempfile.TemporaryDirectory
    pass
