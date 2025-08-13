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

import numpy as np
import xarray as xr

import flopy

print(sys.version)
print(f"flopy version: {flopy.__version__}")

# ## Define `DNODATA` constant
#
# `DNODATA` is an important constant for MODFLOW 6 timeseries grid input
# data. It signifies that the cell has no data defined for the time step
# in question. These cell values are discarded and have no impact on the
# simulation.

# DNODATA constant
DNODATA = 3.0e30

# ## Define ASCII input baseline simulation
#
# For the purposes of this tutorial, the specifics of this simulation
# other than it is a candidate for NetCDF input are not a focus. It
# is a NetCDF input candidate because it defines a candidate model
# type (`GWF6`) with a vertex discretization and packages that support
# NetCDF input parameters.
#
# A NetCDF dataset will be created from array data in the `IC`, and
# `GHBG` packages. Data will be copied from the package objects into
# dataset arrays.


# A FloPy ASCII base simulation that will be updated use netcdf inputs
def create_sim(ws):
    name = "uzf02"
    nlay = 5
    nrow = 10
    ncol = 10
    ncpl = nrow * ncol
    delr = 1.0
    delc = 1.0
    nper = 5
    perlen = [10] * 5
    nstp = [5] * 5
    tsmult = len(perlen) * [1.0]
    top = 25.0
    botm = [20.0, 15.0, 10.0, 5.0, 0.0]
    strt = 20
    nouter, ninner = 100, 300
    hclose, rclose, relax = 1e-9, 1e-3, 0.97

    # use flopy util to get disv arguments
    disvkwargs = flopy.utils.gridutil.get_disv_kwargs(
        nlay, nrow, ncol, delr, delc, top, botm
    )

    # Work up UZF data
    iuzno = 0
    cellid = 0
    uzf_pkdat = []
    vks = 10.0
    thtr = 0.05
    thts = 0.30
    thti = 0.15
    eps = 3.5

    for k in np.arange(nlay):
        for i in np.arange(0, ncpl, 1):
            if k == 0:
                landflg = 1
                surfdp = 0.25
            else:
                landflg = 0
                surfdp = 1e-6

            if k == nlay - 1:
                ivertcon = -1
            else:
                ivertcon = iuzno + ncpl

            bndnm = "uzf" + f"{int(i + 1):03d}"
            uzf_pkdat.append(
                # iuzno     cellid landflag ivertcn surfdp vks thtr thts thti eps [bndnm]
                [
                    iuzno,
                    (k, i),
                    landflg,
                    ivertcon,
                    surfdp,
                    vks,
                    thtr,
                    thts,
                    thti,
                    eps,
                    bndnm,
                ]
            )

            iuzno += 1

    extdp = 14.0
    extwc = 0.055
    pet = 0.001
    zero = 0.0
    uzf_spd = {}
    for t in np.arange(0, nper, 1):
        spd = []
        iuzno = 0
        for k in np.arange(nlay):
            for i in np.arange(0, ncpl, 1):
                if k == 0:
                    if t == 0:
                        finf = 0.15
                    if t == 1:
                        finf = 0.15
                    if t == 2:
                        finf = 0.15
                    if t == 3:
                        finf = 0.15
                    if t == 4:
                        finf = 0.15

                spd.append([iuzno, finf, pet, extdp, extwc, zero, zero, zero])
                iuzno += 1

        uzf_spd.update({t: spd})

    # Work up the GHBG boundary
    ghb_ids = [(ncol - 1) + i * ncol for i in range(nrow)]
    abhead = np.full((nlay, ncpl), DNODATA, dtype=float)
    acond = np.full((nlay, ncpl), DNODATA, dtype=float)
    cond = 1e4
    for k in np.arange(3, 5, 1):
        for i in ghb_ids:
            abhead[k, i] = 14.0
            acond[k, i] = cond

    # build MODFLOW 6 files
    sim = flopy.mf6.MFSimulation(
        sim_name=name, version="mf6", exe_name="mf6", sim_ws=ws
    )

    # time discretization
    tdis_rc = []
    for i in range(nper):
        tdis_rc.append((perlen[i], nstp[i], tsmult[i]))

    # create tdis package
    tdis = flopy.mf6.ModflowTdis(sim, time_units="DAYS", nper=nper, perioddata=tdis_rc)

    # create gwf model
    gwf = flopy.mf6.ModflowGwf(
        sim, modelname=name, newtonoptions="NEWTON", save_flows=True
    )

    # create iterative model solution and register the gwf model with it
    ims = flopy.mf6.ModflowIms(
        sim,
        print_option="SUMMARY",
        complexity="MODERATE",
        outer_dvclose=hclose,
        outer_maximum=nouter,
        under_relaxation="DBD",
        inner_maximum=ninner,
        inner_dvclose=hclose,
        rcloserecord=rclose,
        linear_acceleration="BICGSTAB",
        scaling_method="NONE",
        reordering_method="NONE",
        relaxation_factor=relax,
    )
    sim.register_ims_package(ims, [gwf.name])

    # disv
    disv = flopy.mf6.ModflowGwfdisv(gwf, **disvkwargs)

    # initial conditions
    ic = flopy.mf6.ModflowGwfic(gwf, strt=strt)

    # node property flow
    npf = flopy.mf6.ModflowGwfnpf(gwf, save_flows=True, icelltype=1, k=0.1, k33=1)

    # aquifer storage
    sto = flopy.mf6.ModflowGwfsto(gwf, iconvert=1, ss=1e-5, sy=0.2, transient=True)

    # general-head boundary
    ghb = flopy.mf6.ModflowGwfghbg(gwf, print_flows=True, bhead=abhead, cond=acond)

    # unsaturated-zone flow
    etobs = []
    i = 4
    # Seems as though these are 1-based and not 0-based, like the rest of flopy
    for j in list(np.arange(40, 50, 1)) + list(np.arange(140, 150, 1)):
        etobs.append(("uzet_" + str(j + 1), "uzet", (j,)))
        etobs.append(("uzf-gwet_" + str(j + 1), "uzf-gwet", (j,)))

    uzf_obs = {f"{name}.uzfobs": etobs}

    uzf = flopy.mf6.ModflowGwfuzf(
        gwf,
        print_flows=True,
        save_flows=True,
        simulate_et=True,
        simulate_gwseep=True,
        linear_gwet=True,
        observations=uzf_obs,
        boundnames=True,
        ntrailwaves=15,
        nwavesets=40,
        nuzfcells=len(uzf_pkdat),
        packagedata=uzf_pkdat,
        perioddata=uzf_spd,
        budget_filerecord=f"{name}.uzf.bud",
    )

    # output control
    oc = flopy.mf6.ModflowGwfoc(
        gwf,
        budget_filerecord=f"{name}.cbc",
        head_filerecord=f"{name}.hds",
        headprintrecord=[("COLUMNS", 10, "WIDTH", 15, "DIGITS", 6, "GENERAL")],
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        printrecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        filename=f"{name}.oc",
    )

    # Print human-readable heads
    obs_lst = []
    for k in np.arange(0, 1, 1):
        for i in np.arange(40, 50, 1):
            obs_lst.append(["obs_" + str(i + 1), "head", (k, i)])

    obs_dict = {f"{name}.obs.csv": obs_lst}
    obs = flopy.mf6.ModflowUtlobs(gwf, pname="head_obs", digits=20, continuous=obs_dict)

    return sim


# ## Create helper function to update dataset
#
# This function updates an xarray dataset to add variables described
# in a FloPy provided dictionary.
#
# The dimmap variable relates NetCDF dimension names to a value.


# A subroutine that can update an xarray dataset with package
# netcdf information stored in a dict
def add_netcdf_vars(dataset, nc_info, dimmap):
    def _data_shape(shape):
        dims_l = []
        for d in shape:
            dims_l.append(dimmap[d])

        return dims_l

    for v in nc_info:
        varname = nc_info[v]["varname"]
        layered = varname.split("/")
        if len(layered) > 1:
            l = layered[1][6]
            varname = f"{layered[0]}_l{l}"
        data = np.full(
            _data_shape(nc_info[v]["netcdf_shape"]),
            nc_info[v]["attrs"]["_FillValue"],
            dtype=nc_info[v]["xarray_type"],
        )
        var_d = {varname: (nc_info[v]["netcdf_shape"], data)}
        dataset = dataset.assign(var_d)
        for a in nc_info[v]["attrs"]:
            dataset[varname].attrs[a] = nc_info[v]["attrs"][a]

    return dataset


# ## Create simulation workspace

# create temporary directories
temp_dir = TemporaryDirectory()
workspace = Path(temp_dir.name)

# ## Write and run baseline simulation

# run the non-netcdf simulation
sim = create_sim(ws=workspace)
sim.write_simulation()
success, buff = sim.run_simulation(silent=True, report=True)
assert success, pformat(buff)

# ## Create NetCDF based simulation
#
# Reset the simulation path and set the `GWF` name file `nc_filerecord`
# attribute to the name of the intended input NetCDF file. Display
# the resultant name file changes.

# create directory for netcdf sim
# set model name file nc_filerecord attribute to export name
sim.set_sim_path(workspace / "netcdf")
gwf = sim.get_model("uzf02")
gwf.name_file.nc_filerecord = "uzf02.layered.nc"
sim.write_simulation()
with open(workspace / "netcdf" / "uzf02.nam", "r") as fh:
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

# get model netcdf info
nc_info = gwf.netcdf_info(mesh="layered")
pprint(nc_info)

# update dataset with required attributes
for a in nc_info["attrs"]:
    ds.attrs[a] = nc_info["attrs"][a]

# ## Map dataset dimension names to values

# define dimensional info
dimmap = {
    "time": sum(gwf.modeltime.nstp),
    "z": gwf.modelgrid.nlay,
    "nmesh_face": gwf.modelgrid.ncpl,
}

# ## Access package NetCDF attributes
#
# Access package scoped NetCDF details by storing the dictionary returned
# from `netcdf_info()`. We need to set package variable attributes that are
# stored in the package netcdf info dict, but we also need other information
# that is relevant to creating the variables themselves.
#
# The contents of the info dictionary are shown and then, in the following
# step, the dictionary and the dataset are passed to a helper routine that
# create the intended array variables.

# get ic package netcdf info
ic = gwf.get_package("ic")
nc_info = ic.netcdf_info(mesh="layered")
pprint(nc_info)

# create ic dataset variables
ds = add_netcdf_vars(ds, nc_info, dimmap)

# ## Update array data
#
# We have created dataset array variables for the package but they do not yet
# define the expected input data for MODFLOW 6. We will take advantage of the
# existing simulation objects and update the dataset.

# update dataset from ic strt array
for l in range(gwf.modelgrid.nlay):
    ds[f"ic_strt_l{l + 1}"].values = ic.strt.get_data()[l].flatten()

# ## Update MODFLOW 6 package input file
#
# MODFLOW 6 input data for the package is now in the dataset. Once the NetCDF
# file is generated, we need to configure MODFLOW 6 so that it looks to that
# file for the package array input. The ASCII file will no longer defined the
# arrays- instead the array names will be followed by the NETCDF keyword.
#
# We will simply overwrite the entire MODFLOW 6 `IC` package input file with the
# following code block.

# rewrite mf6 ic input to read from netcdf
with open(workspace / "netcdf" / "uzf02.ic", "w") as f:
    f.write("BEGIN options\n")
    f.write("END options\n\n")
    f.write("BEGIN griddata\n")
    f.write("  strt NETCDF\n")
    f.write("END griddata\n")
with open(workspace / "netcdf" / "uzf02.ic", "r") as fh:
    print(fh.read())

# ## Update MODFLOW 6 package input file
#
# Follow the same process as above for the `GHBG` package. The difference is
# that this is PERIOD input and therefore stored as timeseries data in the
# NetCDF file. As NETCDF timeseries are defined in terms of total number of
# simulation steps, care must be taken in the translation of FloPy period
# data to the timeseries.

# get ghbg package netcdf info
ghbg = gwf.get_package("ghbg_0")
nc_info = ghbg.netcdf_info(mesh="layered")
pprint(nc_info)

# create ghbg dataset variables
ds = add_netcdf_vars(ds, nc_info, dimmap)

# update bhead netcdf array from flopy perioddata
# timeseries step index is first of stress period
for p in ghbg.bhead.get_data():
    if ghbg.bhead.get_data()[p] is not None:
        istp = sum(gwf.modeltime.nstp[0:p])
        for l in range(gwf.modelgrid.nlay):
            ds[f"ghbg_0_bhead_l{l + 1}"].values[istp] = ghbg.bhead.get_data()[p][
                l
            ].flatten()

# update cond netcdf array from flopy perioddata
# timeseries step index is first of stress period
for p in ghbg.cond.get_data():
    if ghbg.cond.get_data()[p] is not None:
        istp = sum(gwf.modeltime.nstp[0:p])
        for l in range(gwf.modelgrid.nlay):
            ds[f"ghbg_0_cond_l{l + 1}"].values[istp] = ghbg.cond.get_data()[p][
                l
            ].flatten()

# rewrite mf6 ghbg input to read from netcdf
with open(workspace / "netcdf/uzf02.ghbg", "w") as f:
    f.write("BEGIN options\n")
    f.write("  READARRAYGRID\n")
    f.write("  PRINT_FLOWS\n")
    f.write("END options\n\n")
    f.write("BEGIN period 1\n")
    f.write("  bhead NETCDF\n")
    f.write("  cond NETCDF\n")
    f.write("END period 1\n")
with open(workspace / "netcdf" / "uzf02.ghbg", "r") as fh:
    print(fh.read())

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
