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

# # MODFLOW 6: Working with MODFLOW Grid Array Data
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
# This tutorial focuses on MODFLOW grid array data from the `GridData` and
# other similar blocks.  These blocks contain data in a one or more dimensional
# array format organized by dimensions, which can include layer, row, column,
# and stress period. These data are stored by FloPy in a `MFArray` or
# `MFTransientArray` object and a referred to as array data.

# ## Introduction to MODFLOW Array Data
# MODFLOW array data use the `MFArray` or `MFTransientArray` FloPy classes and
# are stored in numpy ndarrays.  Most MODFLOW array data are two (row, column)
# or three (layer, row, column) dimensional and represent data on the model
# grid.  Other MODFLOW array data contain data by stress period.  The
# following list summarizes the different types of MODFLOW array data.

# * Time-invariant multi-dimensional array data.  This includes:
#   1. One and two dimensional arrays that do not have a layer dimension.
#      Examples include `top`, `delc`, and `delr`.
#   2. Three dimensional arrays that can contain a layer dimension.
#      Examples include `botm`, `idomain`, and `k`.
# * Transient arrays that can change with time and therefore contain arrays of
#    data for one or more stress periods.  Examples include `irch` and
#    `recharge` in the `RCHA` package.
#
# In the example below a three dimensional ndarray is constructed for the
# `DIS` package's `botm` array.  First, the a simulation and groundwater-flow
# model are set up.

# package import
from tempfile import TemporaryDirectory

import numpy as np

import flopy

# set up where simulation workspace will be stored
temp_dir = TemporaryDirectory()
workspace = temp_dir.name
name = "tutorial07_mf6_data"

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

# Then a three-dimensional ndarray of floating point values is created using
# numpy's `linspace` method.

bot = np.linspace(-50.0 / 3.0, -3.0, 3)
delrow = delcol = 4.0

# The `DIS` package is then created passing the three-dimensional array to the
# `botm` parameter.  The `botm` array defines the model's cell bottom
# elevations.

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

# ## Adding MODFLOW Grid Array Data
# MODFLOW grid array data, like the data found in the `NPF` package's
# `GridData` block, can be specified as:
#
# 1. A constant value
# 2. A n-dimensional list
# 3. A numpy ndarray
#
# Additionally, layered grid data (generally arrays with a layer dimension) can
# be specified by layer.
#
# In the example below `icelltype` is specified as constants by layer, `k` is
# specified as a numpy ndarray, `k22` is specified as an array by layer, and
# `k33` is specified as a constant.

# First `k` is set up as a 3 layer, by 10 row, by 10 column array with all
# values set to 10.0 using numpy's full method.

k = np.full((3, 10, 10), 10.0)

# Next `k22` is set up as a three dimensional list of nested lists. This
# option can be useful for those that are familiar with python lists but are
# not familiar with the numpy library.

k22_row = []
for row in range(0, 10):
    k22_row.append(8.0)
k22_layer = []
for col in range(0, 10):
    k22_layer.append(k22_row)
k22 = [k22_layer, k22_layer, k22_layer]

# `K33` is set up as a single constant value.  Whenever an array has all the
# same values the easiest and most efficient way to set it up is as a constant
# value.  Constant values also take less space to store.

k33 = 1.0

# The `k`, `k22`, and `k33` values defined above are then passed in on
# construction of the npf package.

npf = flopy.mf6.ModflowGwfnpf(
    gwf,
    pname="npf",
    save_flows=True,
    icelltype=[1, 1, 1],
    k=k,
    k22=k22,
    k33=k33,
    xt3doptions="xt3d rhs",
    rewet_record="REWET WETFCT 1.0 IWETIT 1 IHDWET 0",
)

# ### Layered Data
#
# When we look at what will be written to the npf input file, we
# see that the entire `npf.k22` array is written as one long array with the
# number of values equal to `nlay` * `nrow` * `ncol`.  And this whole-array
# specification may be of use in some cases.  Often times, however, it is
# easier to work with each layer separately.  An `MFArray` object, such as
# `npf.k22` can be converted to a layered array as follows.

npf.k22.make_layered()

# By changing `npf.k22` to layered, we are then able to manage each layer
# separately.  Before doing so, however, we need to pass in data that can be
# separated into three layers.  An array of the correct size is one option.

shp = npf.k22.array.shape
a = np.arange(shp[0] * shp[1] * shp[2]).reshape(shp)
npf.k22.set_data(a)

# Now that `npf.k22` has been set to be layered, if we print information about
# it, we see that each layer is stored separately, however, `npf.k22.array`
# will still return a full three-dimensional array.

print(type(npf.k22))
print(npf.k22)

# We also see that each layer is printed separately to the npf
# Package input file, and that the LAYERED keyword is activated:

print(npf.k22.get_file_entry())

# Working with a layered array provides lots of flexibility.  For example,
# constants can be set for some layers, but arrays for others:

npf.k22.set_data([1, a[2], 200])
print(npf.k22.get_file_entry())

# To gain full control over an individual layers, layer information can be
# provided as a dictionary:

a0 = {"factor": 0.5, "iprn": 1, "data": 100 * np.ones((10, 10))}
a1 = 50
a2 = {"factor": 1.0, "iprn": 14, "data": 30 * np.ones((10, 10))}
npf.k22.set_data([a0, a1, a2])
print(npf.k22.get_file_entry())

# Here we say that the FACTOR has been set to 0.5 for the first layer and an
# alternative print flag is set for the last layer.
#
# Because we are specifying a factor for the top layer, we can also see that
# the `get_data()` method returns the array without the factor applied

print(npf.k22.get_data())

# whereas the `array` property returns the array with the factor applied

print(npf.k22.array)

# ## Adding MODFLOW Stress Period Array Data
# Transient array data spanning multiple stress periods must be specified as a
# dictionary of arrays, where the dictionary key is the stress period,
# expressed as a zero-based integer, and the dictionary value is the grid
# data for that stress period.

# In the following example a `RCHA` package is created.  First a dictionary
# is created that contains recharge for the model's two stress periods.
# Recharge is specified as a constant value in this example, though it could
# also be specified as a 3-dimensional ndarray or list of lists.

rch_sp1 = 0.01
rch_sp2 = 0.03
rch_spd = {0: rch_sp1, 1: rch_sp2}

# The `RCHA` package is created and the dictionary constructed above is passed
# in as the `recharge` parameter.

rch = flopy.mf6.ModflowGwfrcha(
    gwf, readasarrays=True, pname="rch", print_input=True, recharge=rch_spd
)

# ## Retrieving Grid Array Data
# Grid data can be retrieved with `get_data`, `array`, `[]`, `repr`/`str`, or
# `get_file_entry`.
#
# | Retrieval Method    | Description           |
# | :---                |    :----              |
# | get_data    | Returns ndarray of data without multiplier applied, unless the apply_mult parameter is set to True             |
# | array       |  Returns ndarray of data with multiplier applied               |
# | [<layer>]       |  Returns a particular layer of data if data is layered, otherwise <layer> is an array index              |
# | repr/str    | Returns string with storage information followed by ndarray repr/str         |
# | get_file_entry(layer)   | Returns string containing data formatted for the MODFLOW-6 package file.  If layer is not specified returns all layers, otherwise returns just the layer specified.         |

# Below the `NPF` `k` array is retrieved using the various methods highlighted
# above.

# First, we use the `get_data` method to get k as an ndarray.

print(npf.k.get_data())

# Next, we use the `array` attribute which also gets k as an ndarray.

print(npf.k.array)

# We can also use the `[]` to get a single layer of data as an ndarray.

print(npf.k[0])

# repr gives a string representation of the data.

print(repr(npf.k))

# str gives a similar string representation of the data.

print(str(npf.k))

# The method `get_file_entry` prints the data as it would appear in a
# MODFLOW 6 file.

print(npf.k.get_file_entry())

# ## Retrieving MODFLOW Stress Period Array Data
# Transient array data can be retrieved with `get_data`, `array`, `repr`/`str`,
# or `get_file_entry`.
#
# | Retrieval Method    | Description           |
# | :---                |    :----              |
# | get_data    | Returns dictionary of ndarrays without multiplier applied.  The dictionary key is the stress period as a zero-based integer.            |
# | array       |  Returns ndarray of data for all stress periods (stress period is an added dimension)               |
# | repr/str    | Returns string with storage information followed by ndarray repr/str for all stress periods        |
# | get_file_entry(layer)   | Returns string containing data formatted for the MODFLOW-6 package file.  Use <key> to specify a stress period (zero-based integer).        |

# Below the `RCHA` `recharge` array is retrieved using the various methods
# highlighted above.

# First, we use the `get_data` method to get the recharge data as a dictionary
# of ndarrays.  The dictionary key is a the stress period (zero based).

print(rch.recharge.get_data())

# Next, we use the `array` attribute to get the data as an 4-dimensional
# ndarray.

print(rch.recharge.array)

# repr gives a string representation of the data.

print(repr(rch.recharge))

# str gives a similar representation of the data.

print(str(rch.recharge))

# We can use the `get_file_entry` method to get the data as it would appear in
# a MODFLOW 6 file for a specific stress period.  In this case we are getting
# the data for stress period 2 (stress periods are treated as 0-based in
# FloPy).

print(rch.recharge.get_file_entry(1))

try:
    temp_dir.cleanup()
except PermissionError:
    # can occur on windows: https://docs.python.org/3/library/tempfile.html#tempfile.TemporaryDirectory
    pass
