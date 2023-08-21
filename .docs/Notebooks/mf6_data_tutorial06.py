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

# # MODFLOW 6: Working with MODFLOW List Data.
#
# This tutorial shows how to view, access, and change the underlying data
# variables for MODFLOW 6 objects in FloPy.  Interaction with a FloPy
# MODFLOW 6 model is different from other models, such as MODFLOW-2005,
# MT3D, and SEAWAT, for example.
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
# This tutorial focuses on MODFLOW Data from the `PackageData`,
# `ConnectionData`, `StressPeriodData`, and other similar blocks.  These
# blocks contain data with columns, data that fits into a numpy recarray,
# pandas data frame, or a spreadsheet with column headers. These data are
# stored by FloPy in a `MFList` or `MFTransientList` object and a referred to
# as MODFLOW list data.

# ## Introduction to MODFLOW List Data
#
# MODFLOW contains list data that can be conveniently stored in a numpy
# recarray or a pandas dataframe.  These data are either a single or multiple
# row of data, with each column containing the same data type.
#
# Some MODFLOW list data only contains a single row, like the `OC` package's
# `head print_format` option and the `NPF` package's `rewet_record`.  Other
# MODFLOW list data can contain multiple rows, like the `MAW` package's
# `packagedata` and `connectiondata`.  FloPy stores both single row and
# multiple row list data in `MFList` objects.
#
# MODFLOW stress period data can contain lists of data for one or more stress
# periods.  FloPy stores stress period list data in `MFTransientList` objects.
# Note that not all MODFLOW stress period data is "list" data that fits neatly
# in a recarray or a panda's dataframe. Some packages including `RCH` and
# `EVT` have a `READASARRAYS` option that allows stress period data to be
# inputted as an array.  When `READASARRAYS` is selected FloPy stores stress
# period array data in an `MFTransientArray` object (see tutorial 8).
#
# Examples of using FloPy to store, update, and retrieve different types of
# MODFLOW list data are given below.  The examples start by first creating a
# simulation (`MFSimulation`) and a model (`MFModel`) object in FloPy.

# package import
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import flopy

# set up where simulation workspace will be stored
temp_dir = TemporaryDirectory()
workspace = temp_dir.name
name = "tutorial06_mf6_data"

# create the Flopy simulation and tdis objects
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

# ## Adding MODFLOW Package Data, Connection Data, and Option Lists
#
# MODFLOW Package data, connection data, and option lists are stored by FloPy
# as numpy recarrays.  FloPy does accept numpy recarrays as input, but does
# has other supported formats discussed below.
#
# MODFLOW option lists that only contain a single row or data can be either
# specified by:
#
# 1. Specifying a string containing the entire line as it would be displayed
#    in the package file (`rewet_record="REWET WETFCT 1.0 IWETIT 1 IHDWET 0"`)
# 2. Specifying the data in a tuple within a list
#    (`rewet_record=[("WETFCT", 1.0, "IWETIT", 1, "IHDWET", 0)]`)
#
# In the example below the npf package is created setting the `rewet_record`
# option to a string of text as would be typed into the package file.

npf = flopy.mf6.modflow.mfgwfnpf.ModflowGwfnpf(
    gwf,
    rewet_record="REWET WETFCT 1.0 IWETIT 1 IHDWET 0",
    pname="npf",
    icelltype=1,
    k=1.0,
    save_flows=True,
    xt3doptions="xt3d rhs",
)

# `rewet_record` is then set using the npf package's `rewet_record` property.
# This time 'rewet_record' is defined using a tuple within a list.

npf.rewet_record = [("WETFCT", 1.1, "IWETIT", 0, "IHDWET", 1)]

# MODFLOW multirow lists, like package data and connection data, can be
# specified:
#
# 1. As a list of tuples where each tuple represents a row in the list
#   (stress_period_data = [((1, 2, 3), 20.0), ((1, 7, 3), 25.0)])
# 2. As a numpy recarray.  Building a numpy recarray is more complicated and
#    is beyond the scope of this guide.
#
# In the example below the chd package is created, setting `stress_period_data`
# as a list of tuples.

# We build the chd package using an array of tuples for stress_period_data
# stress_period_data = [(first_chd_cell, head), (second_chd_cell, head), ...]
# Note that the cellid information (layer, row, column) is encapsulated in
# a tuple.

stress_period_data = [((1, 8, 8), 100.0), ((1, 9, 9), 105.0)]
# build chd package
chd = flopy.mf6.modflow.mfgwfchd.ModflowGwfchd(
    gwf,
    pname="chd",
    maxbound=len(stress_period_data),
    stress_period_data=stress_period_data,
    save_flows=True,
)

# ## Adding Stress Period List Data
#
# MODFLOW stress period data is stored by FloPy as a dictionary of numpy
# recarrays, where each dictionary key is a zero-based stress period and each
# dictionary value is a recarray containing the stress period data for that
# stress period.  FloPy keeps this stress period data in a `MFTransientList`
# object and this data type is referred to as a transient list.
#
# FloPy accepts stress period data as a dictionary of numpy recarrays, but also
# supports replacing the recarrays with lists of tuples discussed above.
# Stress period data spanning multiple stress periods must be specified as a
# dictionary of lists where the dictionary key is the stress period expressed
# as a zero-based integer.
#
# The example below creates `stress_period_data` for the wel package with the
# first stress period containing a single well and the second stress period
# empty.  When empty stress period data is entered FloPy writes an empty
# stress period block to the package file.

# First we create wel package with stress_period_data dictionary
# keys as zero-based integers so key "0" is stress period 1

stress_period_data = {
    0: [((2, 3, 1), -25.0)],  # stress period 1 well data
    1: [],
}  # stress period 2 well data is empty

# Then, using the dictionary created above, we build the wel package.

wel = flopy.mf6.ModflowGwfwel(
    gwf,
    print_input=True,
    print_flows=True,
    stress_period_data=stress_period_data,
    save_flows=False,
    pname="WEL-1",
)

# ## Retrieving MODFLOW Package Data, Connection Data, and Option Lists
#
# MODFLOW package data, connection data, and option lists can be retrieved
# with `get_data`, `array`, `repr`/`str`,
# or get_file_entry.
#
# | Retrieval Method    | Description           |
# | :---                |    :----              |
# | get_data    | Returns recarray              |
# | array       | Return recarray               |
# | repr/str    | Returns string with storage information followed by recarray's repr/str           |
# | get_file_entry   | Returns string containing data formatted for the MODFLOW-6 package file. Certain zero-based numbers, like layer, row, column, are converted to one-based numbers.           |

# The `NPF` package's `rewet_record` is printed below using the different data
# retrieval methods highlighted above.

# First we use the `get_data` method to get the rewet_record as a recarray.

print(npf.rewet_record.get_data())

# Next we use the `array` method, which also returns a recarray.

print(npf.rewet_record.array)

# Then we use repr to print a string representation of rewet_record.

print(repr(npf.rewet_record))

# Using str prints a similar string representation of rewet_record.

print(str(npf.rewet_record))

# Last, using the `get_file_entry` method the data is printed as it would
# appear in a MODFLOW 6 file.

print(npf.rewet_record.get_file_entry())

# ## Retrieving MODFLOW Stress Period List Data
# Stress period data can be retrieved with `get_data`, `array`, `repr`/`str`,
# or `get_file_entry`.
#
# | Retrieval Method    | Description           |
# | :---                |    :----              |
# | get_data    | Returns dictionary of recarrays              |
# | array       | Return single recarray for all stress periods               |
# | repr/str    | Returns string with storage information followed by recarray repr/str for each recarray          |
# | get_file_entry(key)   | Returns string containing data formatted for the MODFLOW-6 package file for the stress period specified by key         |

# The `WEL` package's `stress_period_data` is printed below using the
# different data retrieval methods highlighted above.

# First we use the `get_data` method to get the stress period data as a
# dictionary of recarrays.

print(wel.stress_period_data.get_data())

# Next we use the `array` attribute to get the stress period data as a single
# recarray.

print(wel.stress_period_data.array)

# repr can be used to generate a string representation of stress period data.

print(repr(wel.stress_period_data))

# str produces a similar string representation of stress period data.

print(str(wel.stress_period_data))

# The `get_file_entry` method prints the stress period data as it would
# appear in a MODFLOW 6 file.

print(wel.stress_period_data.get_file_entry(0))

try:
    temp_dir.cleanup()
except PermissionError:
    # can occur on windows: https://docs.python.org/3/library/tempfile.html#tempfile.TemporaryDirectory
    pass
