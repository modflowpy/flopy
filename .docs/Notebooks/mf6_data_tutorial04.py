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

# # MODFLOW 6: Time Array Series Packages
#

# ## Introduction to Time Array Series
#
# Time array series can be set for any package through the `package.tas`
# object, and each `package.tas` object has several attributes that can be set:
#
# | Attribute | Type | Description |
# | :---      | :---- | :----      |
# | package.tas.filename | str | Name of time series file to create. The default is packagename + ".tas". |
# | package.tas.tas_array | {double:[double]} | Array containing the time array series information for specific times. |
# | package.tas.time_series_namerecord | str | Name by which a package references a particular time-array series. The name must be unique among all time-array series used in a package. |
# | package.tas.interpolation_methodrecord | list (of strings) | List of interpolation methods to use for time array series. Method must be either "stepwise" or "linear". |
# | package.tas.sfacrecord_single | float | Scale factor to multiply the time array series data column. Can only be used if there is one time series data column. |
#
# The following code sets up a simulation used in the time array series
# examples.

# package import
from tempfile import TemporaryDirectory

import numpy as np

import flopy

# set up where simulation workspace will be stored
temp_dir = TemporaryDirectory()
workspace = temp_dir.name
name = "tutorial04_mf6_data"

# create the Flopy simulation and tdis objects
sim = flopy.mf6.MFSimulation(
    sim_name=name, exe_name="mf6", version="mf6", sim_ws=workspace
)
tdis_rc = [(1.0, 1, 1.0), (10.0, 5, 1.0), (10.0, 5, 1.0), (10.0, 1, 1.0)]
tdis_package = flopy.mf6.modflow.mftdis.ModflowTdis(
    sim, time_units="DAYS", nper=4, perioddata=tdis_rc
)
# create the Flopy groundwater flow (gwf) model object
model_nam_file = f"{name}.nam"
gwf = flopy.mf6.ModflowGwf(sim, modelname=name, model_nam_file=model_nam_file)
# create the flopy iterative model solver (ims) package object
ims = flopy.mf6.modflow.mfims.ModflowIms(sim, pname="ims", complexity="SIMPLE")
# create the discretization package
bot = np.linspace(-3.0, -50.0 / 3.0, 3)
delrow = delcol = 4.0
dis = flopy.mf6.modflow.mfgwfdis.ModflowGwfdis(
    gwf,
    pname="dis",
    nogrb=True,
    nlay=3,
    nrow=101,
    ncol=101,
    delr=delrow,
    delc=delcol,
    top=0.0,
    botm=bot,
)
# create the initial condition (ic) and node property flow (npf) packages
ic_package = flopy.mf6.modflow.mfgwfic.ModflowGwfic(gwf, strt=50.0)
npf_package = flopy.mf6.modflow.mfgwfnpf.ModflowGwfnpf(
    gwf,
    save_flows=True,
    icelltype=[1, 0, 0],
    k=[5.0, 0.1, 4.0],
    k33=[0.5, 0.005, 0.1],
)

# ## Time Array Series Example 1
#
# Time array series data can be passed into the `timearrayseries` parameter on
# construction of any package that supports time array series.  This example
# uses the `timearrayseries` parameter to create a time array series and then
# uses the package's `tas` property to finish the time array series setup.
#
# This example uses time array series data in a `RCHA` package.  The time array
# series is built as a dictionary with the times as the keys and the recharge
# values as the values.

tas = {0.0: 0.000002, 200.0: 0.0000001}

# The time array series data is then passed into the `timearrayseries`
# parameter when the `RCHA` package is constructed.

rcha = flopy.mf6.modflow.mfgwfrcha.ModflowGwfrcha(
    gwf, timearrayseries=tas, recharge="TIMEARRAYSERIES rcharray_1"
)

# Time array series attributes can be set by access the time array series
# package object through the `rcha.tas` attribute.

# finish defining the time array series properties
rcha.tas.time_series_namerecord = "rcharray_1"
rcha.tas.interpolation_methodrecord = "LINEAR"

# The simulation is then written to files and run.

sim.write_simulation()
sim.run_simulation()

# clean up for next example
gwf.remove_package("rcha")

# ## Time Array Series Example 2
#
# A time array series can be added after a package is created by calling the
# package's `tas` attribute's `initialize` method.  Initialize allows you to
# define all time array series attributes including file name, the time array
# series data, name record, and method record.

# First a recharge package is built.

# create recharge package with recharge pointing to a time array series
# not yet defined. FloPy will generate a warning that there is not yet a
# time series name record for recharray_1
rcha = flopy.mf6.modflow.mfgwfrcha.ModflowGwfrcha(
    gwf, recharge="TIMEARRAYSERIES rcharray_1"
)

# Then a time array series dictionary is created as done in example 1.

tas = {0.0: 0.000002, 200.0: 0.0000001}

# The time array series data are added by calling the `initialize` method from
# the 'RCHA' package's tas attribute.  The time array series file name, name
# record, and method record, along with the time series data are set in the
# `initialize` method.

# initialize the time array series
rcha.tas.initialize(
    filename="method2.tas",
    tas_array=tas,
    time_series_namerecord="rcharray_1",
    interpolation_methodrecord="LINEAR",
)

try:
    temp_dir.cleanup()
except PermissionError:
    # can occur on windows: https://docs.python.org/3/library/tempfile.html#tempfile.TemporaryDirectory
    pass
