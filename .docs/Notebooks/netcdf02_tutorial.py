# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # MODFLOW 6: Generate MODFLOW 6 NetCDF input from existing FloPy sim
#
# ## NetCDF tutorial 2: MODFLOW 6 UGRID layered mesh input file
#
# This tutorial demonstrates how to generate a MODFLOW 6 NetCDF file from
# an existing FloPy simulation. In the tutorial, candidate array data is
# added to an xarray dataset and annotated so that the generated NetCDF
# file can be read by MODFLOW 6 as model input.
#
# This tutorial generates a UGRID layered mesh NetCDF variant - for more
# information on supported MODFLOW 6 NetCDF formats see:
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

sim_name = "uzf02"

# Check if we are in the repository and define the data path.

try:
    root = Path(git.Repo(".", search_parent_directories=True).working_dir)
except:
    root = None

data_path = root / "examples" / "data" / "mf6" / "netcdf" if root else Path.cwd()

file_names = {
    "mfsim.nam": None,
    "uzf02.disv": None,
    "uzf02.ghbg": None,
    "uzf02.ic": None,
    "uzf02.ims": None,
    "uzf02.nam": None,
    "uzf02.npf": None,
    "uzf02.obs": None,
    "uzf02.oc": None,
    "uzf02.sto": None,
    "uzf02.tdis": None,
    "uzf02.uzf": None,
    "uzf02.uzf.obs": None,
    "uzf02.uzfobs": None,
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
# (`GWF6`) with a vertex discretization and packages that support
# NetCDF input parameters. Vertex (`DISV`) discretizations are only
# supported by the `UGRID layered mesh` NetCDF format and as such, the
# `mesh` attribute will be set to `layered` when passed to FloPy functions
# in this tutorial.

# load and run the non-netcdf simulation
sim = flopy.mf6.MFSimulation.load(sim_ws=data_path / sim_name)
# sim = flopy.mf6.MFSimulation.load(sim_ws=Path("./netcdf02"))
sim.set_sim_path(workspace)
sim.write_simulation()
success, buff = sim.run_simulation(silent=True, report=True)
assert success, pformat(buff)

# ## Create NetCDF based simulation
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
# Flopy does not currently generate the NetCDF input file. This tutorial
# shows one way that can be accomplished.

# create directory for netcdf sim
sim.set_sim_path(workspace / "netcdf")
# set model name file nc_filerecord attribute to export name
gwf = sim.get_model("uzf02")
gwf.name_file.nc_filerecord = "uzf02.layered.nc"
# write simulation with ASCII inputs tagged for NetCDF
sim.write_simulation(netcdf=True)
# show name file with NetCDF input configured
with open(workspace / "netcdf" / "uzf02.nam", "r") as fh:
    print(fh.read())
# show example package file with NetCDF input configured
with open(workspace / "netcdf" / "uzf02.ic", "r") as fh:
    print(fh.read())

# ## Create dataset
#
# Create the base xarray dataset from the modelgrid object. This
# will add required dimensions and coordinate variables to the
# dataset according to the grid specification. Modeltime is needed
# for timeseries support.

# create the dataset
ds = gwf.modelgrid.dataset(modeltime=gwf.modeltime, mesh="layered")

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
nc_info = gwf.netcdf_info(mesh="layered")
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
disv = gwf.get_package("disv")
ds = disv.update_dataset(ds, mesh="layered")

# ## Update array data
#
# We have created dataset array variables for the package but they do not yet
# define the expected input data for MODFLOW 6. We will take advantage of the
# existing simulation objects and update the dataset.
#
# Default dataset variable names are defined in the package `netcdf_info()`
# dictionary.

# update dataset from dis arrays
ds["disv_top"].values = disv.top.get_data()
for l in range(gwf.modelgrid.nlay):
    ds[f"disv_botm_l{l + 1}"].values = disv.botm.get_data()[l]

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
nc_info = npf.netcdf_info(mesh="layered")
pprint(nc_info)

# ## Update package `netcdf_info` dictionary and dataset
#
# Here we update the `NPF K` layer 1 input parameter to add the
# `standard_name` attribute to it's attribute dictionary.  The dictionary
# is then passed to the `update_dataset()` function. Note the updated name
# is used in the subsequent block when updating the array values.

# update dataset with `NPF` arrays
nc_info["k/layer1"]["attrs"]["standard_name"] = (
    "soil_hydraulic_conductivity_at_saturation"
)
ds = npf.update_dataset(ds, netcdf_info=nc_info, mesh="layered")

# ## Update `NPF` array data

# update dataset from npf arrays
for l in range(gwf.modelgrid.nlay):
    ds[f"npf_icelltype_l{l + 1}"].values = npf.icelltype.get_data()[l]
    ds[f"npf_k_l{l + 1}"].values = npf.k.get_data()[l]
    ds[f"npf_k33_l{l + 1}"].values = npf.k33.get_data()[l]

# ## Show dataset `NPF K` parameter with updates

# print dataset npf k variable
print(ds["npf_k_l1"])

# ## Update the dataset with supported `IC` arrays

# ic
ic = gwf.get_package("ic")
ds = ic.update_dataset(ds, mesh="layered")
for l in range(gwf.modelgrid.nlay):
    ds[f"ic_strt_l{l + 1}"].values = ic.strt.get_data()[l]

# ## Update the dataset with supported `STO` arrays

# storage
sto = gwf.get_package("sto")
ds = sto.update_dataset(ds, mesh="layered")
for l in range(gwf.modelgrid.nlay):
    ds[f"sto_iconvert_l{l + 1}"].values = sto.iconvert.get_data()[l]
    ds[f"sto_sy_l{l + 1}"].values = sto.sy.get_data()[l]
    ds[f"sto_ss_l{l + 1}"].values = sto.ss.get_data()[l]

# ## Update the dataset with supported `GHBG` arrays

# update dataset with 'GHBG' arrays
ghbg = gwf.get_package("ghbg_0")
ds = ghbg.update_dataset(ds, mesh="layered")

# ## Update `GHBG` array data

# update bhead netcdf array from flopy perioddata
# timeseries step index is first of stress period
for p in ghbg.bhead.get_data():
    if ghbg.bhead.get_data()[p] is not None:
        istp = sum(gwf.modeltime.nstp[0:p])
        for l in range(gwf.modelgrid.nlay):
            ds[f"ghbg_0_bhead_l{l + 1}"].values[istp] = ghbg.bhead.get_data()[p][l]

# update cond netcdf array from flopy perioddata
# timeseries step index is first of stress period
for p in ghbg.cond.get_data():
    if ghbg.cond.get_data()[p] is not None:
        istp = sum(gwf.modeltime.nstp[0:p])
        for l in range(gwf.modelgrid.nlay):
            ds[f"ghbg_0_cond_l{l + 1}"].values[istp] = ghbg.cond.get_data()[p][l]

# ## Display generated dataset

# show the dataset
print(ds)

# ## Export generated dataset to NetCDF

# write dataset to netcdf
ds.to_netcdf(workspace / "netcdf/uzf02.layered.nc", format="NETCDF4", engine="netcdf4")

# ## Run MODFLOW 6 simulation with NetCDF input
#
# The simulation generated by this tutorial should be runnable by
# Extended MODFLOW 6, available from the nightly-build repository
# (linked above).

# success, buff = sim.run_simulation(silent=True, report=True)
# assert success, pformat(buff)
