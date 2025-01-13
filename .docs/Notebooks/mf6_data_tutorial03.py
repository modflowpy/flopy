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

# # MODFLOW 6: Time Series Packages
#

# ## Introduction to Time Series
#
# Time series can be set for any package through the `package.ts` object, and
# each `package.ts` object has several attributes that can be set:
#
# | Attribute | Type | Description |
# | :---      | :---- | :----      |
# | package.ts.filename | str | Name of time series file to create. The default is packagename + ".ts".|
# | package.ts.timeseries | recarray | Array containing the time series information |
# | package.ts.time_series_namerecord | str or list | List of names of the time series data columns. Default is to use names from timeseries.dtype.names[1:]. |
# | package.ts.interpolation_methodrecord_single | str | Interpolation method. Must be only one time series record. If there are multiple time series records, then the methods attribute must be used. |
# | package.ts.interpolation_methodrecord | float | Scale factor to multiply the time series data column. Can only be used if there is one time series data column. |
# | package.ts.sfacrecord_single | float | Scale factor to multiply the time series data column. Can only be used if there is one time series data column. |
# | package.ts.sfacrecord | list (of floats) | Scale factors to multiply the time series data columns. |
#
# The following code sets up a simulation used in the time series examples.

# package import
from tempfile import TemporaryDirectory

import numpy as np

import flopy

# set up where simulation workspace will be stored
temp_dir = TemporaryDirectory()
workspace = temp_dir.name
name = "tutorial03_mf6_data"

# create the flopy simulation and tdis objects
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

# ## Time Series Example 1
#
# One way to construct a time series is to pass the time series data to
# the parent package constructor.
#
# This example uses time series data in a `GHB` package.  First the `GHB`
# `stress_period_data` is built.

# build ghb stress period data
ghb_spd_ts = {}
ghb_period = []
for layer, cond in zip(range(1, 3), [15.0, 1500.0]):
    for row in range(0, 15):
        ghb_period.append(((layer, row, 9), "tides", cond, "Estuary-L2"))
ghb_spd_ts[0] = ghb_period

# Next the time series data is built.  The time series data is constructed as
# a list of tuples, with each tuple containing a time and the value (or values)
# at that time.  The time series data is put in a dictionary along with
# additional time series information including filename,
# time_series_namerecord, interpolation_methodrecord, and sfacrecord.

# build ts data
ts_data = []
for n in range(0, 365):
    time = float(n / 11.73)
    val = float(n / 60.0)
    ts_data.append((time, val))
ts_dict = {
    "filename": "tides.ts",
    "time_series_namerecord": "tide",
    "timeseries": ts_data,
    "interpolation_methodrecord": "linearend",
    "sfacrecord": 1.1,
}

# The `GHB` package is then constructed, passing the time series data into the
# `timeseries` parameter.

# build ghb package
ghb = flopy.mf6.modflow.mfgwfghb.ModflowGwfghb(
    gwf,
    print_input=True,
    print_flows=True,
    save_flows=True,
    boundnames=True,
    timeseries=ts_dict,
    pname="ghb",
    maxbound=30,
    stress_period_data=ghb_spd_ts,
)

# Time series attributes, like `time_series_namerecord`, can be modified
# using the `ghb.ts` object.

# set required time series attributes
ghb.ts.time_series_namerecord = "tides"

# clean up for next example
gwf.remove_package("ghb")

# ## Time Series Example 2
#
# Another way to construct a time series is to initialize the time series
# through the `ghb.ts.initialize` method.  Additional time series can then be
# appended using the `append_package` method.
#
# First the `GHB` stress period data is built.

# build ghb stress period data
ghb_spd_ts = {}
ghb_period = []
for layer, cond in zip(range(1, 3), [15.0, 1500.0]):
    for row in range(0, 15):
        if row < 10:
            ghb_period.append(((layer, row, 9), "tides", cond, "Estuary-L2"))
        else:
            ghb_period.append(((layer, row, 9), "wl", cond, "Estuary-L2"))
ghb_spd_ts[0] = ghb_period

# Next the time series data is built.  The time series data is constructed as
# a list of tuples, with each tuple containing a time and the value (or values)
# at that time.

# build ts data
ts_data = []
for n in range(0, 365):
    time = float(n / 11.73)
    val = float(n / 60.0)
    ts_data.append((time, val))
ts_data2 = []
for n in range(0, 365):
    time = float(1.0 + (n / 12.01))
    val = float(n / 60.0)
    ts_data2.append((time, val))
ts_data3 = []
for n in range(0, 365):
    time = float(10.0 + (n / 12.01))
    val = float(n / 60.0)
    ts_data3.append((time, val))

# A ghb package is constructed without the time series data

# build ghb package
ghb = flopy.mf6.modflow.mfgwfghb.ModflowGwfghb(
    gwf,
    print_input=True,
    print_flows=True,
    save_flows=True,
    boundnames=True,
    pname="ghb",
    maxbound=30,
    stress_period_data=ghb_spd_ts,
)

# The first time series data are added by calling the initialize method from
# the `ghb.ts` object.  The times series package's file name,
# name record, method record, and sfac record, along with the time series data
# are set in the initialize method.

# initialize first time series
ghb.ts.initialize(
    filename="tides.ts",
    timeseries=ts_data,
    time_series_namerecord="tides",
    interpolation_methodrecord="linearend",
    sfacrecord=1.1,
)

# The remaining time series data are added using the `append_package` method.
# The `append_package` method takes the same parameters as the initialize
# method.

# append additional time series
ghb.ts.append_package(
    filename="wls.ts",
    timeseries=ts_data2,
    time_series_namerecord="wl",
    interpolation_methodrecord="stepwise",
    sfacrecord=1.2,
)
# append additional time series
ghb.ts.append_package(
    filename="wls2.ts",
    timeseries=ts_data3,
    time_series_namerecord="wl2",
    interpolation_methodrecord="stepwise",
    sfacrecord=1.3,
)

# Information can be retrieved from time series packages using the `ts`
# attribute of its parent package.  Below the interpolation method record
# for each of the three time series are retrieved.

print(
    "{} is using {} interpolation".format(
        ghb.ts[0].filename, ghb.ts[0].interpolation_methodrecord.get_data()[0][0]
    )
)
print(
    "{} is using {} interpolation".format(
        ghb.ts[1].filename, ghb.ts[1].interpolation_methodrecord.get_data()[0][0]
    )
)
print(
    "{} is using {} interpolation".format(
        ghb.ts[2].filename, ghb.ts[2].interpolation_methodrecord.get_data()[0][0]
    )
)

try:
    temp_dir.cleanup()
except PermissionError:
    # can occur on windows: https://docs.python.org/3/library/tempfile.html#tempfile.TemporaryDirectory
    pass
