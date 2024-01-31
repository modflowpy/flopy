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
#   metadata:
#     section: mf6
# ---

# # MODFLOW 6: Working with MODFLOW Scalar Data
#
# This tutorial shows how to view, access, and change the underlying data
# variables for MODFLOW 6 objects in FloPy.  Interaction with a FloPy
# MODFLOW 6 model is different from other models, such as MODFLOW-2005,
# MT3D, and SEAWAT, for example.
#
# FloPy stores model data in data objects (`MFDataArray`, `MFDataList`,
# `MFDataScalar` objects) that are accessible from packages Data can be added
# to a package by using the appropriate parameters when the package is
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
# This tutorial focuses on MODFLOW Data that is a single integer or string,
# or consists of boolean flag(s).  These data are stored by FloPy in a
# MFScalar object and are referred to as MODFLOW scalar data.

# ## Introduction to MODFLOW Scalar Data
#
# MODFLOW single integer, strings, or boolean flag(s) are stored by FloPy as
# scalar data in `MFScalar` objects.  The different types of scalar data are
# described below.

# 1. Single integer values. Examples include `nrow`, `ncol`, `nlay`, and
#    `nper`.
# 2. Single string values.  Examples include `time_units` and `length_units`.
# 3. Boolean flags.  These can be found in the options section of most
#    packages.  These include perched, `nogrb`, `print_input`, and
#    `save_flows`.
# 4. Boolean flags with an additional optional flag.  These include
#    `newton under_relaxation` and `xt3d rhs`.
#
# In the following all four types of scalar data will be added to a model.
# Before adding data to your model first create a simulation (`MFSimulation`)
# and a model (`MFModel`) object in FloPy.

# package import
from tempfile import TemporaryDirectory

import numpy as np

import flopy

# set up where simulation workspace will be stored
temp_dir = TemporaryDirectory()
workspace = temp_dir.name
name = "tutorial05_mf6_data"

# create the flopy simulation object
sim = flopy.mf6.MFSimulation(
    sim_name=name, exe_name="mf6", version="mf6", sim_ws=workspace
)

# create the flopy groundwater flow (gwf) model object
model_nam_file = f"{name}.nam"
gwf = flopy.mf6.ModflowGwf(sim, modelname=name, model_nam_file=model_nam_file)
# create the flopy iterative model solver (ims) package object
# (both pname and complexity are scalar data)
ims = flopy.mf6.modflow.mfims.ModflowIms(sim, pname="ims", complexity="SIMPLE")

# ## Adding MODFLOW Single Integer and String Values
#
# Single integer and string values can be assigned on construction of the
# `MFScalar` data object, and can be assigned or changed after construction.
#
# Below, a `TDIS` package is constructed with the `time_units` and `nper`
# parameters being assigned "DAYS" and "2", respectively.

# create the FloPy temporal discretization object
tdis = flopy.mf6.modflow.mftdis.ModflowTdis(
    sim,
    pname="tdis",
    time_units="DAYS",
    nper=2,
    perioddata=[(1.0, 1, 1.0), (1.0, 1, 1.0)],
)

# Next, `time_units` is reassigned a value after construction by using `TDIS`'s
# `time_units` attribute.

tdis.time_units = "MONTHS"

# ## Setting MODFLOW Boolean Flags
# Boolean flags can be assigned a True or False value. In the example below
# `nogrb` is assigned a value of True and then changed to false.
#
# For this example, first some values are first defined for the discretization
# package

nlay = 3
h = 50.0
length = 400.0
n = 10
bot = np.linspace(-h / nlay, -h, nlay)
delrow = delcol = length / (n - 1)

# Below the discretization package is created.  The MODFLOW `nogrb` option
# assigned a value of True, switching this option on.

dis = flopy.mf6.modflow.mfgwfdis.ModflowGwfdis(
    gwf,
    pname="dis",
    nogrb=True,
    nlay=nlay,
    nrow=n,
    ncol=n,
    delr=delrow,
    delc=delcol,
    top=0.0,
    botm=bot,
)

# The `nogrb` option is then switched off by setting the `DIS` package's
# `nogrb` attribute to False.

dis.nogrb = False

# Boolean flags with an additional optional flag can either be specified by:
#
# 1. Specifying the entire line as it would be displayed in the package file
#    as a string (`xt3doptions="xt3d rhs"`)
# 2. Specifying each flag name in a list (`xt3doptions=["xt3d", "rhs"]`)
#
# To turn off both flags use an empty string (`xt3doptions=""`) or an empty
# list (`xt3doptions=[]`).

# First, an `NPF` package is created.  `xt3doptions` can either be turned on or
# off, and if it is on `rhs` can optionally be turned on.  `xt3doptions` is set
# to the string "xt3d rhs", turning both options on.

# create the node property flow package with xt3doptions as single
npf = flopy.mf6.modflow.mfgwfnpf.ModflowGwfnpf(
    gwf,
    rewet_record="REWET WETFCT 1.0 IWETIT 1 IHDWET 0",
    pname="npf",
    icelltype=1,
    k=1.0,
    save_flows=True,
    xt3doptions="xt3d rhs",
)

# Next, the `rhs` option is turned off by setting `xt3doptions` to the string
# "xt3d".

npf.xt3doptions = "xt3d"

# Finally, both `xt3d` and `rhs` are turned off by setting `xt3doptions` to an
# empty string.

npf.xt3doptions = ""

# ## Retrieving MODFLOW Scalar Data
#
# MODFLOW scalar data can be retrieved with `get_data`, `repr`/`str`,
# or `get_file_entry`.
#
# | Retrieval Method    | Description           |
# | :---                |    :----              |
# | get_data    | Returns scalar value              |
# | repr/str    | Returns string with a header describing how data is stored (internal, external) with a string representation of the data on the next line          |
# | get_file_entry   | Returns string with the scalar keyword (if any) followed by a space and a string representation of the scalar value (if any).  This is the format used by the MODFLOW-6 package file.  This is the format used by the MODFLOW-6 package file.        |

# The `IMS` package's `complexity` option and the `NPF` package's
# `xt3doptions` are printed below using the different data retrieval methods
# highlighted above.

# First the complexity data is printed using the `get_data` method.

print(ims.complexity.get_data())

# The xt3doptions data can also be printed with `get_data`.

print(npf.xt3doptions.get_data())

# The complexity data is then printed with repr

print(repr(ims.complexity))

# The xt3doptions data is printed with repr

print(str(npf.xt3doptions))

# The complexity data is printed as it would appear in a MODFLOW 6 file using
# the `get_file_entry` method.

print(ims.complexity.get_file_entry())

# The xt3doptions data is printed as it would appear in a MODFLOW 6 file using
# the `get_file_entry` method.

print(npf.xt3doptions.get_file_entry())

try:
    temp_dir.cleanup()
except PermissionError:
    # can occur on windows: https://docs.python.org/3/library/tempfile.html#tempfile.TemporaryDirectory
    pass
