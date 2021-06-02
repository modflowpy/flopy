# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: "1.5"
#       jupytext_version: 1.5.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # MODFLOW 6 Tutorial 9: Data Storage Information - How and Where to Store MODFLOW-6 Data
#
# This tutorial shows the different options for storing MODFLOW data in FloPy.
# Interaction with a FloPy MODFLOW 6 model is different from other models,
# such as MODFLOW-2005, MT3D, and SEAWAT, for example.
#
# FloPy stores model data in data objects (`MFDataArray`, `MFDataList`,
# `MFDataScalar` objects) that are accessible from packages.  Data can be
# added to a package by using the appropriate parameters when the package is
# constructed and through package attributes.
#
# The MODFLOW 6 simulation structure is arranged in the following
# generalized way:
#
# >       Simulation --> Package --> DATA
# >
# >       Simulation --> Model --> Package (--> Package) --> DATA
#
#
# This tutorial focuses on the different storage options for MODFLOW data.

# ## Introduction to Data Storage Information
# MODFLOW array and list data can either be stored internally or externally in
# text or binary files. Additionally array data can have a factor applied to
# them and can have a format flag/code to define how these data will be
# formatted. This data storage information is specified within a python
# dictionary.  The python dictionary must contain a "data" key where the data
# is stored and supports several other keys that determine how and where the
# data is stored.
#
# The following code sets up a basic simulation with a groundwater flow model.
# for the example below.

# package import
import os
import numpy as np
import flopy

# set up where simulation workspace will be stored
workspace = os.path.join("data", "mf6_working_with_data")
name = "example_1"
if not os.path.exists(workspace):
    os.makedirs(workspace)
# create the FloPy simulation and tdis objects
sim = flopy.mf6.MFSimulation(
    sim_name=name, exe_name="mf6", version="mf6", sim_ws=workspace
)
tdis = flopy.mf6.modflow.mftdis.ModflowTdis(
    sim,
    pname="tdis",
    time_units="DAYS",
    nper=2,
    perioddata=[(1.0, 1, 1.0), (1.0, 1, 1.0)],
)
# create the Flopy groundwater flow (gwf) model object
model_nam_file = "{}.nam".format(name)
gwf = flopy.mf6.ModflowGwf(sim, modelname=name, model_nam_file=model_nam_file)
# create the flopy iterative model solver (ims) package object
ims = flopy.mf6.modflow.mfims.ModflowIms(sim, pname="ims", complexity="SIMPLE")
# create the discretization package
bot = np.linspace(-50.0 / 3.0, -3.0, 3)
delrow = delcol = 4.0
dis = flopy.mf6.modflow.mfgwfdis.ModflowGwfdis(
    gwf,
    pname="dis",
    nogrb=True,
    nlay=3,
    nrow=10,
    ncol=10,
    delr=delrow,
    delc=delcol,
    top=0.0,
    botm=bot,
)

# ## Setting up a Data Storage Information Dictionary
#
# To store data externally add a `filename` key to the dictionary whose
# value is the file where you want to store the data. Add a `binary` key with
# value True to make the file a binary file. Add a `prn` key whose value is
# a format code to set the format code for the data. For array data add a
# `factor` key whose value is a positive floating point number to add a
# factor/multiplier for the array.

# Below a dictionary is created that defines how a `k33` array will be stored.
# The dictionary specifies that the `k33` array be stored in the binary file
# k33.txt with a factor of 1.0 applied to the array and a print code of 1.
# The `k33` array data is constructed as a numpy array.

k33_values = np.full((3, 10, 10), 1.1)
k33 = {
    "filename": "k33.txt",
    "factor": 1.0,
    "data": k33_values,
    "iprn": 1,
    "binary": "True",
}

# The `NPF` package is then created with the `k33` array in an external binary
# file.  This binary file is created when the simulation method
# `write_simulation` is called.

npf = flopy.mf6.ModflowGwfnpf(
    gwf,
    pname="npf",
    save_flows=True,
    icelltype=[1, 1, 1],
    k=10.0,
    k22=5.0,
    k33=k33,
    xt3doptions="xt3d rhs",
    rewet_record="REWET WETFCT 1.0 IWETIT 1 IHDWET 0",
)

# External files can be set for specific layers of data.  If we want to store
# the bottom elevations for the third model layer to an external file, then
# the dictionary that we pass in for the third layer can be given a
# "filename" key.

a0 = {"factor": 0.5, "iprn": 1, "data": np.ones((10, 10))}
a1 = -100
a2 = {
    "filename": "dis.botm.3.txt",
    "factor": 2.0,
    "iprn": 1,
    "data": -100 * np.ones((10, 10)),
}

# A list containing data for the three specified layers is then passed in to
# the `botm` object's `set_data` method.

dis.botm.set_data([a0, a1, a2])
print(dis.botm.get_file_entry())

# Note that we could have also specified `botm` this way as part of the
# original `flopy.mf6.ModflowGwfdis` constructor:

a0 = {"factor": 0.5, "iprn": 1, "data": np.ones((10, 10))}
a1 = -100
a2 = {
    "filename": "dis.botm.3.bin",
    "factor": 2.0,
    "iprn": 1,
    "data": -100 * np.ones((4, 5)),
    "binary": True,
}
botm = [a0, a1, a2]
flopy.mf6.ModflowGwfdis(gwf, nlay=3, nrow=10, ncol=10, botm=botm)
