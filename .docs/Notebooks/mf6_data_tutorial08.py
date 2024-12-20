# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: metadata
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   metadata:
#     section: mf6
# ---

# # MODFLOW 6: External Files, Binary Data, and Performance Optimization
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
# This tutorial focuses on the different storage options for MODFLOW data and
# how to optimize data storage read/write speed.

# ## Introduction to Data Storage Options
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
from tempfile import TemporaryDirectory

import numpy as np

import flopy

# set up where simulation workspace will be stored
temp_dir = TemporaryDirectory()
workspace = temp_dir.name
name = "tutorial08_mf6_data"

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
model_nam_file = f"{name}.nam"
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
# the `botm` object's `set_record` method.

dis.botm.set_record([a0, a1, a2])
print(dis.botm.get_file_entry())

# The botm data and its attributes (filename, binary, factor, iprn) can be
# retrieved as a dictionary of dictionaries using get_record.

botm_record = dis.botm.get_record()
print("botm layer 1 record:")
print(botm_record[0])
print("\nbotm layer 2 record:")
print(botm_record[1])
print("\nbotm layer 3 record:")
print(botm_record[2])

# The botm record retrieved can be modified and then saved with set_record.
# For example, the array data's "factor" can be modified and saved.

botm_record[0]["factor"] = 0.6
dis.botm.set_record(botm_record)

# The updated value can then be retrieved.

botm_record = dis.botm.get_record()
print(f"botm layer 1 factor:  {botm_record[0]['factor']}")

# The get_record and set_record methods can also be used with list data to get
# and set the data and its "filename" and "binary" attributes. This is
# demonstrated with the wel package.  First, a wel package is constructed.

welspdict = {
    0: {"filename": "well_sp1.txt", "data": [[(0, 0, 0), 0.25]]},
    1: [[(0, 0, 0), 0.1]],
}
wel = flopy.mf6.ModflowGwfwel(
    gwf,
    print_input=True,
    print_flows=True,
    stress_period_data=welspdict,
    save_flows=False,
)

# The wel stress period data and associated "filename" and "binary" attributes
# can be retrieved with get_record.

spd_record = wel.stress_period_data.get_record()
print("Stress period 1 record:")
print(spd_record[0])
print("\nStress period 2 record:")
print(spd_record[1])

# The wel data and associated attributes can be changed by modifying the
# record and then saving it with the set_record method.

spd_record[0]["filename"] = "well_package_sp1.txt"
spd_record[0]["binary"] = True
spd_record[1]["filename"] = "well_package_sp2.bin"
wel.stress_period_data.set_record(spd_record)

# The changes can be verified by calling get_record again.

spd_record = wel.stress_period_data.get_record()
print(f"New filename for stress period 1:  {spd_record[0]['filename']}")
print(f"New binary flag for stress period 1:  {spd_record[0]['binary']}")
print(f"New filename for stress period 2:  {spd_record[1]['filename']}")

# An alternative to individually setting each file to external is to call the set_all_files_external method (there is also a set_all_files_internal method to do the opposite). While this requires less code, it does not give you the ability to set the names of each individual external file. By setting the binary attribute to True, flopy will store data to binary files wherever possible.

sim.set_all_data_external(binary=True)

# ## Optimizing FloPy Performance
#
# By default FloPy will perform a number of verification checks on your data
# when FloPy loads or saves that data.  For large datasets turning these
# verification checks off can significantly improve FloPy's performance.
# Additionally, storing files externally can help minimize the amount of data
# FloPy reads/writes when loading and saving a simulation.  The following
# steps will help you optimize FloPy's performance for large datasets.
#
# 1) Turn off FloPy verification checks and FloPy's option to automatically
# update "maxbound".  This can be turned off on an existing
# simulation by either individually turning off each of these settings.

sim.simulation_data.auto_set_sizes = False
sim.simulation_data.verify_data = False
sim.write_simulation()

# or by setting lazy_io to True.

sim.simulation_data.lazy_io = True
sim.write_simulation()

# These options can also be turned off when loading an existing simulation
# or creating a new simulation by setting lazy_io to True.

sim2 = flopy.mf6.MFSimulation.load(sim_ws=workspace, lazy_io=True)

sim3 = flopy.mf6.MFSimulation(lazy_io=True)

# 2) Whenever possible save large datasets to external binary files.  Binary
# files are more compact and the data will be read and written significantly
# faster.  See MODFLOW-6 documentation for which packages and data support
# binary files.

# store all well period data in external binary files
spd_record[0]["binary"] = True
spd_record[1]["binary"] = True
wel.stress_period_data.set_record(spd_record)

# 3) For datasets that do not support binary files, save them to
# external text files.  When loading a simulation, FloPy will always parse
# MODFLOW-6 package files, but will not parse external text files when
# auto_set_sizes and verify data are both set to False.  Additionally, if
# write_simulation is later called, external files will not be re-written
# unless either the data or the file path has changed.

# store lak period data in external text files
period = {
    0: {"filename": "lak_sp1.txt", "data": [(0, "STAGE", 10.0)]},
    1: {"filename": "lak_sp2.txt", "data": [(0, "STAGE", 15.0)]},
}
lakpd = [(0, -2.0, 1)]
lakecn = [(0, 0, (0, 1, 0), "HORIZONTAL", 1.0, -5.0, 0.0, 10.0, 10.0)]
lak = flopy.mf6.ModflowGwflak(
    gwf,
    pname="lak-1",
    nlakes=1,
    noutlets=0,
    ntables=0,
    packagedata=lakpd,
    connectiondata=lakecn,
    perioddata=period,
)

try:
    temp_dir.cleanup()
except PermissionError:
    # can occur on windows: https://docs.python.org/3/library/tempfile.html#tempfile.TemporaryDirectory
    pass
