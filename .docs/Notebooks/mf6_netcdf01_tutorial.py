# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # MODFLOW 6: Generate MODFLOW 6 NetCDF input from existing FloPy sim
#
# ## NetCDF tutorial 1: MODFLOW 6 structured input file
#
# This tutorial shows how to generate a MODFLOW 6 NetCDF file from
# an existing FloPy simulation. Two methods will be demonstrated that
# generate a simulation with package data stored in a model NetCDF
# file. The first method is non-interactive- FloPy will generate the
# file with a modified `write_simulation()` call.  The second method
# is interactive, which provides an oppurtinity to modify the dataset
# before it is written to NetCDF.
#
# For more information on supported MODFLOW 6 NetCDF formats see:
# [MODFLOW NetCDF Format](https://github.com/MODFLOW-ORG/modflow6/wiki/MODFLOW-NetCDF-Format).
#
# Note that NetCDF is only supported by the Extended version of MODFLOW 6.
# A nightly windows build of Extended MODFLOW 6 is available from
# [nightly build](https://github.com/MODFLOW-ORG/modflow6-nightly-build).

# package import
import sys
from pathlib import Path
from pprint import pformat, pprint
from tempfile import TemporaryDirectory

import git
import numpy as np
import pooch
import xarray as xr

import flopy

print(sys.version)
print(f"flopy version: {flopy.__version__}")

sim_name = "uzf01"

# Check if we are in the repository and define the data path.

try:
    root = Path(git.Repo(".", search_parent_directories=True).working_dir)
except:
    root = None

data_path = root / "examples" / "data" / "mf6" / "netcdf" if root else Path.cwd()

file_names = {
    "mfsim.nam": None,
    "uzf01.dis": None,
    "uzf01.ghb.obs": None,
    "uzf01.ghbg": None,
    "uzf01.ic": None,
    "uzf01.ims": None,
    "uzf01.nam": None,
    "uzf01.npf": None,
    "uzf01.obs": None,
    "uzf01.oc": None,
    "uzf01.sto": None,
    "uzf01.tdis": None,
    "uzf01.uzf": None,
    "uzf01.uzf.obs": None,
}

for fname, fhash in file_names.items():
    pooch.retrieve(
        url=f"https://github.com/modflowpy/flopy/raw/develop/examples/data/{sim_name}/{fname}",
        fname=fname,
        path=data_path / sim_name,
        known_hash=fhash,
    )

# ## Create simulation workspace

# create temporary directories
temp_dir = TemporaryDirectory()
workspace = Path(temp_dir.name)

# ## Load and run baseline simulation
#
# For the purposes of this tutorial, the specifics of this simulation
# other than it is a candidate for NetCDF input are not a focus. It
# is a NetCDF input candidate because it defines a supported model type
# (`GWF6`) with a structured discretization and packages that support
# NetCDF input parameters.

# load and run the non-netcdf simulation
sim = flopy.mf6.MFSimulation.load(sim_ws=data_path / sim_name)
sim.set_sim_path(workspace)
sim.write_simulation()
success, buff = sim.run_simulation(silent=True, report=True)
assert success, pformat(buff)

# ## Create NetCDF based simulation method 1
#
# This is the most straightforward way to create a NetCDF simulation
# from the loaded ascii input simulation. Simply define the `netcdf`
# argument to `write_simulation()` to be either `structured` or
# `layered`, depending on the desired format of the generated NetCDF
# file.
#
# The name of the created file can be specified by first setting the
# model `name_file.nc_filerecord` attribute to the desired name. If
# this step is not taken, the default name of `{model_name}.input.nc`
# is used.

# create directory for netcdf sim
sim.set_sim_path(workspace / "netcdf1")
# set model name file nc_filerecord attribute to export name
gwf = sim.get_model("uzf01")
gwf.name_file.nc_filerecord = "uzf01.structured.nc"
# write simulation with structured NetCDF file
sim.write_simulation(netcdf="structured")

# success, buff = sim.run_simulation(silent=True, report=True)
# assert success, pformat(buff)

# ## Repeat method 1 with layered mesh NetCDF format

# create directory for netcdf sim
sim.set_sim_path(workspace / "netcdf2")
# set model name file nc_filerecord attribute to export name
gwf = sim.get_model("uzf01")
gwf.name_file.nc_filerecord = "uzf01.layered.nc"
# write simulation with with layered mesh NetCDF file
sim.write_simulation(netcdf="layered")

# success, buff = sim.run_simulation(silent=True, report=True)
# assert success, pformat(buff)

# ## Create NetCDF based simulation method 2
#
# Reset the simulation path and set the `GWF` name file `nc_filerecord`
# attribute to the name of the intended input NetCDF file. Display
# the resultant name file changes.
#
# When we write the updated simulation, all packages that support NetCDF
# input parameters will be converted. We will therefore need to create a
# NetCDF input file containing arrays for the `DIS`, `NPF`, `IC`, `STO`,
# and `GHBG` packages. Data will be copied from the package objects into
# dataset arrays.
#
# Flopy will not generate the NetCDF input file when the `netcdf` argument
# to `write_simulation()` is set to `nofile`. This step is needed, however,
# to update ascii input with the keywords required to support the model
# NetCDF file that we will generate.

# create directory for netcdf sim
sim.set_sim_path(workspace / "netcdf3")
# set model name file nc_filerecord attribute to export name
gwf = sim.get_model("uzf01")
gwf.name_file.nc_filerecord = "uzf01.structured.nc"
# write simulation with ASCII inputs tagged for NetCDF
# but do not create NetCDF file
sim.write_simulation(netcdf="nofile")

# ## Show name file with NetCDF input configured

# show name file with NetCDF input configured
with open(workspace / "netcdf3" / "uzf01.nam", "r") as fh:
    print(fh.read())

# ## Show example package file with NetCDF keywords

# show example package file with NetCDF input configured
with open(workspace / "netcdf3" / "uzf01.ic", "r") as fh:
    print(fh.read())

# ## Create dataset
#
# Create the base xarray dataset from the modelgrid object. This
# will add required dimensions and coordinate variables to the
# dataset according to the grid specification. Modeltime is needed
# for timeseries support.

# create the dataset
ds = gwf.modelgrid.dataset(modeltime=gwf.modeltime)

# ## Access model NetCDF attributes
#
# Access model scoped NetCDF details by storing the dictionary
# returned from `netcdf_info()`. In particular, we need to set dataset
# scoped attributes that are stored in the model netcdf info dict.
#
# First, retrieve and store the netcdf info dictionary and display
# its contents. Then, in the following step, update the dataset with
# the model scoped attributes defined in the dictionary.
#
# These 2 operations can also be accomplished by calling `update_dataset()`
# on the model object. Analogous functions for the package are shown
# below.

# get model netcdf info
nc_info = gwf.netcdf_info()
pprint(nc_info)

# update dataset directly with required attributes
for a in nc_info["attrs"]:
    ds.attrs[a] = nc_info["attrs"][a]

# ## Update the dataset with supported `DIS` arrays
#
# Add NetCDF supported data arrays in package to dataset. Internally, this call
# uses a `netcdf_info()` package dictionary to determine candidate variables
# and relevant information about them. Alternatively, this dictionary can
# be directly accessed, updated, and passed to the `update_dataset()` function.
# That workflow will be demonstrated in the `NPF` package update which follows.

# update dataset with `DIS` arrays
dis = gwf.get_package("dis")
ds = dis.update_dataset(ds)

# ## Update array data
#
# We have created dataset array variables for the package but they do not yet
# define the expected input data for MODFLOW 6. We will take advantage of the
# existing simulation objects and update the dataset.
#
# Default dataset variable names are defined in the package `netcdf_info()`
# dictionary. Here we will use the info dictionary to programmatically update
# the dataset- for remaining packages we will hardcode the variable names
# being updated for maximum clarity.

nc_info = dis.netcdf_info()
for v in nc_info:
    name = nc_info[v]["attrs"]["modflow_input"].rsplit("/", 1)[1].lower()
    d = getattr(dis, name)
    ds[nc_info[v]["varname"]].values = d.get_data()

# ## Access `NPF` package NetCDF attributes
#
# Access package scoped NetCDF details by storing the dictionary returned
# from `netcdf_info()`. We need to set package variable attributes that are
# stored in the package netcdf info dict, but we also need other information
# that is relevant to creating the variables themselves.
#
# The contents of the info dictionary are shown and then, in the following
# step, the dictionary and the dataset are passed to a helper routine that
# create the intended array variables.

# get npf package netcdf info
npf = gwf.get_package("npf")
nc_info = npf.netcdf_info()
pprint(nc_info)

# ## Update package `netcdf_info` dictionary and dataset
#
# Here we replace the default name for the `NPF K` input parameter and add
# the `standard_name` attribute to it's attribute dictionary.  The dictionary
# is then passed to the `update_dataset()` function. Note the updated name
# is used in the subsequent block when updating the array values.

# update dataset with `NPF` arrays
nc_info["k"]["varname"] = "npf_k_updated"
nc_info["k"]["attrs"]["standard_name"] = "soil_hydraulic_conductivity_at_saturation"
ds = npf.update_dataset(ds, netcdf_info=nc_info)

# ## Update array data

# update dataset from npf arrays
ds["npf_icelltype"].values = npf.icelltype.get_data()
ds["npf_k_updated"].values = npf.k.get_data()

# ## Show dataset `NPF K` parameter with updates

# print dataset npf k variable
print(ds["npf_k_updated"])

# ## Update the dataset with supported `IC` arrays

# ic
ic = gwf.get_package("ic")
ds = ic.update_dataset(ds)
ds["ic_strt"].values = ic.strt.get_data()

# ## Update the dataset with supported `STO` arrays

# storage
sto = gwf.get_package("sto")
ds = sto.update_dataset(ds)
ds["sto_iconvert"].values = sto.iconvert.get_data()
ds["sto_sy"].values = sto.sy.get_data()
ds["sto_ss"].values = sto.ss.get_data()

# ## Update the dataset with supported `GHBG` arrays

# update dataset with 'GHBG' arrays
ghbg = gwf.get_package("ghbg_0")
ds = ghbg.update_dataset(ds)

# ## Update array data

# update bhead netcdf array from flopy perioddata
# timeseries step index is first of stress period
for p in ghbg.bhead.get_data():
    istp = sum(gwf.modeltime.nstp[0:p])
    ds["ghbg_0_bhead"].values[istp] = ghbg.bhead.get_data()[p]

# update cond netcdf array from flopy perioddata
# timeseries step index is first of stress period
for p in ghbg.cond.get_data():
    istp = sum(gwf.modeltime.nstp[0:p])
    ds["ghbg_0_cond"].values[istp] = ghbg.cond.get_data()[p]

# ## Display generated dataset

# show the dataset
print(ds)

# ## Export generated dataset to NetCDF

# write dataset to netcdf
ds.to_netcdf(
    workspace / "netcdf3" / "uzf01.structured.nc", format="NETCDF4", engine="netcdf4"
)

# ## Run MODFLOW 6 simulation with NetCDF input
#
# The simulation generated by this tutorial should be runnable by
# Extended MODFLOW 6, available from the nightly-build repository
# (linked above).

# success, buff = sim.run_simulation(silent=True, report=True)
# assert success, pformat(buff)
