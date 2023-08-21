# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   metadata:
#     section: mf6
# ---

# # Observations, time series and time array series
# This code sets up a simulation and creates some data for the simulation.

# +
import os
import sys
from tempfile import TemporaryDirectory

import numpy as np

try:
    import flopy
except:
    fpth = os.path.abspath(os.path.join("..", ".."))
    sys.path.append(fpth)
    import flopy

# init paths
exe_name = "mf6"

# temporary directory
temp_dir = TemporaryDirectory()
sim_path = os.path.join(temp_dir.name, "obs_ts_tas_ex")
# make the directory if it does not exist
if not os.path.isdir(sim_path):
    os.makedirs(sim_path, exist_ok=True)

# init paths
test_ex_name = "child_pkgs_test"
model_name = "child_pkgs"

print(sys.version)
print(f"numpy version: {np.__version__}")
print(f"flopy version: {flopy.__version__}")

# +
# create simulation
sim = flopy.mf6.MFSimulation(
    sim_name=test_ex_name, version="mf6", exe_name="mf6", sim_ws=sim_path
)

tdis_rc = [(1.0, 1, 1.0), (10.0, 120, 1.0), (10.0, 120, 1.0), (10.0, 120, 1.0)]
tdis_package = flopy.mf6.modflow.mftdis.ModflowTdis(
    sim, time_units="DAYS", nper=4, perioddata=tdis_rc
)
model = flopy.mf6.ModflowGwf(
    sim, modelname=model_name, model_nam_file=f"{model_name}.nam"
)
ims_package = flopy.mf6.modflow.mfims.ModflowIms(
    sim,
    print_option="SUMMARY",
    complexity="SIMPLE",
    outer_hclose=0.0001,
    outer_maximum=500,
    under_relaxation="NONE",
    inner_maximum=100,
    inner_hclose=0.0001,
    rcloserecord=0.001,
    linear_acceleration="CG",
    scaling_method="NONE",
    reordering_method="NONE",
    relaxation_factor=0.97,
)
sim.register_ims_package(ims_package, [model.name])
bot_data = [-100 for x in range(150)]
dis_package = flopy.mf6.modflow.mfgwfdis.ModflowGwfdis(
    model,
    nlay=3,
    nrow=15,
    ncol=10,
    delr=500.0,
    delc=500.0,
    top=50.0,
    botm=[5.0, -10.0, {"factor": 1.0, "data": bot_data}],
    filename=f"{model_name}.dis",
)
ic_package = flopy.mf6.modflow.mfgwfic.ModflowGwfic(
    model, strt=50.0, filename=f"{model_name}.ic"
)
npf_package = flopy.mf6.modflow.mfgwfnpf.ModflowGwfnpf(
    model,
    save_flows=True,
    icelltype=[1, 0, 0],
    k=[5.0, 0.1, 4.0],
    k33=[0.5, 0.005, 0.1],
)
oc_package = flopy.mf6.modflow.mfgwfoc.ModflowGwfoc(
    model,
    budget_filerecord="child_pkgs.cbc",
    head_filerecord="child_pkgs.hds",
    headprintrecord=["COLUMNS", 10, "WIDTH", 15, "DIGITS", 6, "GENERAL"],
    saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    printrecord=[("HEAD", "FIRST"), ("HEAD", "LAST"), ("BUDGET", "LAST")],
)
sto_package = flopy.mf6.modflow.mfgwfsto.ModflowGwfsto(
    model,
    save_flows=True,
    iconvert=1,
    ss=0.000001,
    sy=0.2,
    steady_state={0: True},
    transient={1: True},
)
# -

# ## Observations
#
# Observations can be set for any package through the package.obs object, and each package.obs
# object has several attributes that can be set:
#
# package.obs.filename : str
#     Name of observations file to create. The default is packagename + '.obs',
#     e.g. mymodel.ghb.obs.
#
# package.obs.continuous : dict
#     A dictionary that has file names as keys and a list of
#     observations as the dictionary values. Default should probably be None.
#     package.obs.observations = {'fname1': [(obsname, obstype, cellid), ...],
#     'fname2': [(obsname, obstype, cellid), ...]}
#
# package.obs.digits : int
#     Number of digits to write the observation values. Default is 10.
#
# package.obs.print_input : bool
#     Flag indicating whether or not observations are written to listing file.

# ### Method 1: Pass obs to package constructor

# +
# build ghb stress period data
ghb_spd = {}
ghb_period = []
for layer, cond in zip(range(1, 3), [15.0, 1500.0]):
    for row in range(0, 15):
        ghb_period.append(((layer, row, 9), 1.0, cond, "Estuary-L2"))
ghb_spd[0] = ghb_period

# build obs data
ghb_obs = {
    ("ghb_obs.csv", "binary"): [
        ("ghb-2-6-10", "GHB", (1, 5, 9)),
        ("ghb-3-6-10", "GHB", (2, 5, 9)),
    ],
    "ghb_flows.csv": [
        ("Estuary2", "GHB", "Estuary-L2"),
        ("Estuary3", "GHB", "Estuary-L3"),
    ],
}
# build ghb package
ghb = flopy.mf6.modflow.mfgwfghb.ModflowGwfghb(
    model,
    print_input=True,
    print_flows=True,
    save_flows=True,
    boundnames=True,
    observations=ghb_obs,
    pname="ghb",
    maxbound=30,
    stress_period_data=ghb_spd,
)
ghb.obs.print_input = True

sim.write_simulation()
success, buff = sim.run_simulation(silent=True, report=True)
if success:
    for line in buff:
        print(line)
else:
    raise ValueError("Failed to run.")

# clean up for next example
model.remove_package("ghb")
# -

# ### Method 2: Initialize obs through ghb.obs.initialize

# +
# build ghb stress period data
ghb_spd = {}
ghb_period = []
for layer, cond in zip(range(1, 3), [15.0, 1500.0]):
    for row in range(0, 15):
        ghb_period.append(((layer, row, 9), 1.0, cond, "Estuary-L2"))
ghb_spd[0] = ghb_period

# build ghb package
ghb = flopy.mf6.modflow.mfgwfghb.ModflowGwfghb(
    model,
    print_input=True,
    print_flows=True,
    save_flows=True,
    boundnames=True,
    maxbound=30,
    stress_period_data=ghb_spd,
    pname="ghb",
)

# build obs data
ghb_obs = {
    ("ghb_obs.csv", "binary"): [
        ("ghb-2-6-10", "GHB", (1, 5, 9)),
        ("ghb-3-6-10", "GHB", (2, 5, 9)),
    ],
    "ghb_flows.csv": [
        ("Estuary2", "GHB", "Estuary-L2"),
        ("Estuary3", "GHB", "Estuary-L3"),
    ],
}

# initialize obs package
ghb.obs.initialize(
    filename="child_pkgs_test.ghb.obs",
    digits=9,
    print_input=True,
    continuous=ghb_obs,
)

sim.write_simulation()
success, buff = sim.run_simulation(silent=True, report=True)
if success:
    for line in buff:
        print(line)
else:
    raise ValueError("Failed to run.")

# clean up for next example
model.remove_package("ghb")
# -

# ### Method 3: Pass observations a dictionary of anything that could be passed to ghb.obs.initialize

# +
# build ghb stress period data
ghb_spd = {}
ghb_period = []
for layer, cond in zip(range(1, 3), [15.0, 1500.0]):
    for row in range(0, 15):
        ghb_period.append(((layer, row, 9), 1.0, cond, "Estuary-L2"))
ghb_spd[0] = ghb_period

# build obs data
ghb_obs = {
    ("ghb_obs.csv", "binary"): [
        ("ghb-2-6-10", "GHB", (1, 5, 9)),
        ("ghb-3-6-10", "GHB", (2, 5, 9)),
    ],
    "ghb_flows.csv": [
        ("Estuary2", "GHB", "Estuary-L2"),
        ("Estuary3", "GHB", "Estuary-L3"),
    ],
}
# append additional obs attributes to obs dictionary
ghb_obs["digits"] = 7
ghb_obs["print_input"] = False
ghb_obs["filename"] = "method_3.obs"

# build ghb package
ghb_package = flopy.mf6.modflow.mfgwfghb.ModflowGwfghb(
    model,
    print_input=True,
    print_flows=True,
    save_flows=True,
    boundnames=True,
    observations=ghb_obs,
    pname="ghb",
    maxbound=30,
    stress_period_data=ghb_spd,
)

sim.write_simulation()
success, buff = sim.run_simulation(silent=True, report=True)
if success:
    for line in buff:
        print(line)
else:
    raise ValueError("Failed to run.")

# clean up for next example
model.remove_package("ghb")
# -

# ## Time Series
#
# Time series can be set for any package through the package.ts object, and each package.ts object
# has several attributes that can be set:
#
# package.ts.filename : str
#     Name of time series file to create. The default is packagename + '.ts',
#     e.g. mymodel.ghb.ts.
#
# package.ts.timeseries : recarray
#     Array containing the time series information.
#     timeseries = [(t, np.sin(t)) for t in np.linspace(0, 100., 10)]
#
# package.ts.time_series_namerecord : str or list (of strings)
#     List of names of the time series data columns. Default is to use names from
#     timeseries.dtype.names[1:].
#
# package.ts.interpolation_methodrecord_single : str
#     Interpolation method. Must be only one time series record. If there are multiple time
#     series records, then the methods attribute must be used. Default is 'linear'.
#
# package.ts.interpolation_methodrecord : list (of strings)
#     List of interpolation methods to use for each time series data column. Method must be
#     either 'stepwise', 'linear', or 'linearend'.
#
# package.ts.sfacrecord_single : float
#     Scale factor to multiply the time series data column. Can only be used if there is
#     one time series data column.
#
# package.ts.sfacrecord : list (of floats)
#     Scale factors to multiply the time series data columns.

# ### Method 1: Pass time series to package constructor

# +
# build ghb stress period data
ghb_spd_ts = {}
ghb_period = []
for layer, cond in zip(range(1, 3), [15.0, 1500.0]):
    for row in range(0, 15):
        ghb_period.append(((layer, row, 9), "tides", cond, "Estuary-L2"))
ghb_spd_ts[0] = ghb_period

# build ts data
ts_data = []
for n in range(0, 365):
    ts_data.append((float(n / 11.73), float(n / 60.0)))

# build obs data
ghb_obs = {
    ("ghb_obs.csv", "binary"): [
        ("ghb-2-6-10", "GHB", (1, 5, 9)),
        ("ghb-3-6-10", "GHB", (2, 5, 9)),
    ],
    "ghb_flows.csv": [
        ("Estuary2", "GHB", "Estuary-L2"),
        ("Estuary3", "GHB", "Estuary-L3"),
    ],
}
# build ghb package
ghb = flopy.mf6.modflow.mfgwfghb.ModflowGwfghb(
    model,
    print_input=True,
    print_flows=True,
    save_flows=True,
    boundnames=True,
    timeseries=ts_data,
    pname="ghb",
    maxbound=30,
    stress_period_data=ghb_spd_ts,
)

# set required time series attributes
ghb.ts.time_series_namerecord = "tides"
ghb.ts.interpolation_methodrecord = "stepwise"

sim.write_simulation()
success, buff = sim.run_simulation(silent=True, report=True)
if success:
    for line in buff:
        print(line)
else:
    raise ValueError("Failed to run.")

# clean up for next example
model.remove_package("ghb")
# -

# ### Method 2: Initialize time series through ghb.ts.initialize

# +
# build ghb stress period data
ghb_spd_ts = {}
ghb_period = []
for layer, cond in zip(range(1, 3), [15.0, 1500.0]):
    for row in range(0, 15):
        ghb_period.append(((layer, row, 9), "tides", cond, "Estuary-L2"))
ghb_spd_ts[0] = ghb_period

# build ts data
ts_data = []
for n in range(0, 365):
    ts_data.append((float(n / 11.73), float(n / 60.0)))

# build obs data
ghb_obs = {
    ("ghb_obs.csv", "binary"): [
        ("ghb-2-6-10", "GHB", (1, 5, 9)),
        ("ghb-3-6-10", "GHB", (2, 5, 9)),
    ],
    "ghb_flows.csv": [
        ("Estuary2", "GHB", "Estuary-L2"),
        ("Estuary3", "GHB", "Estuary-L3"),
    ],
}
# build ghb package
ghb = flopy.mf6.modflow.mfgwfghb.ModflowGwfghb(
    model,
    print_input=True,
    print_flows=True,
    save_flows=True,
    boundnames=True,
    pname="ghb",
    maxbound=30,
    stress_period_data=ghb_spd_ts,
)
# initialize time series
ghb.ts.initialize(
    filename="method2.ts",
    timeseries=ts_data,
    time_series_namerecord="tides",
    interpolation_methodrecord="linearend",
    sfacrecord=1.1,
)

sim.write_simulation()
success, buff = sim.run_simulation(silent=True, report=True)
if success:
    for line in buff:
        print(line)
else:
    raise ValueError("Failed to run.")

# clean up for next example
model.remove_package("ghb")
# -

# ### Method 3: Pass timeseries a dictionary of anything that could be passed to ghb.ts.initialize

# +
# build ghb stress period data
ghb_spd_ts = {}
ghb_period = []
for layer, cond in zip(range(1, 3), [15.0, 1500.0]):
    for row in range(0, 15):
        ghb_period.append(((layer, row, 9), "tides", cond, "Estuary-L2"))
ghb_spd_ts[0] = ghb_period

# build ts data
ts_data = []
for n in range(0, 365):
    ts_data.append((float(n / 11.73), float(n / 60.0)))
ts_dict = {
    "timeseries": ts_data,
    "time_series_namerecord": "tides",
    "interpolation_methodrecord": "linear",
    "filename": "method3.ts",
}
# build obs data
ghb_obs = {
    ("ghb_obs.csv", "binary"): [
        ("ghb-2-6-10", "GHB", (1, 5, 9)),
        ("ghb-3-6-10", "GHB", (2, 5, 9)),
    ],
    "ghb_flows.csv": [
        ("Estuary2", "GHB", "Estuary-L2"),
        ("Estuary3", "GHB", "Estuary-L3"),
    ],
}
# build ghb package
ghb = flopy.mf6.modflow.mfgwfghb.ModflowGwfghb(
    model,
    print_input=True,
    print_flows=True,
    save_flows=True,
    boundnames=True,
    pname="ghb",
    timeseries=ts_dict,
    maxbound=30,
    stress_period_data=ghb_spd_ts,
)

sim.write_simulation()
success, buff = sim.run_simulation(silent=True, report=True)
if success:
    for line in buff:
        print(line)
else:
    raise ValueError("Failed to run.")

# clean up for next example
model.remove_package("ghb")
# -

# ### Multiple time series packages

# +
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

# build ts data
ts_data = []
for n in range(0, 365):
    ts_data.append((float(n / 11.73), float(n / 60.0)))
ts_data2 = []
for n in range(0, 365):
    ts_data2.append((float(0.0 + (n / 11.73)), 2 * float(n / 60.0)))
ts_data3 = []
for n in range(0, 365):
    ts_data3.append((float(0.0 + (n / 11.73)), 1.5 * float(n / 60.0)))

# build obs data
ghb_obs = {
    ("ghb_obs.csv", "binary"): [
        ("ghb-2-6-10", "GHB", (1, 5, 9)),
        ("ghb-3-6-10", "GHB", (2, 5, 9)),
    ],
    "ghb_flows.csv": [
        ("Estuary2", "GHB", "Estuary-L2"),
        ("Estuary3", "GHB", "Estuary-L3"),
    ],
}
# build ghb package
ghb = flopy.mf6.modflow.mfgwfghb.ModflowGwfghb(
    model,
    print_input=True,
    print_flows=True,
    save_flows=True,
    boundnames=True,
    pname="ghb",
    maxbound=30,
    stress_period_data=ghb_spd_ts,
)

# initialize time series
ghb.ts.initialize(
    filename="tides.ts",
    timeseries=ts_data,
    time_series_namerecord="tides",
    interpolation_methodrecord="linearend",
    sfacrecord=1.1,
)
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

# retreive information from each time series
print(
    "{} is using {} interpolation".format(
        ghb.ts[0].filename,
        ghb.ts[0].interpolation_methodrecord.get_data()[0][0],
    )
)
print(
    "{} is using {} interpolation".format(
        ghb.ts[1].filename,
        ghb.ts[1].interpolation_methodrecord.get_data()[0][0],
    )
)
print(
    "{} is using {} interpolation".format(
        ghb.ts[2].filename,
        ghb.ts[2].interpolation_methodrecord.get_data()[0][0],
    )
)

sim.write_simulation()
success, buff = sim.run_simulation(silent=True, report=True)
if success:
    for line in buff:
        print(line)
else:
    raise ValueError("Failed to run.")
# -

# ## Time Array Series
# Time array series can be set for any package through the package.tas object, and each package.tas object
# has several attributes that can be set:
#
# package.tas.filename : str
#     Name of time series file to create. The default is packagename + '.tas',
#     e.g. mymodel.rcha.tas.
#
# package.tas.tas_array : {double:[double]}
#     Array containing the time array series information for specific times.
#     tas_array = {0.0: 0.0001, 200.0: [0.01, 0.02...]}
#
# package.tas.time_series_namerecord : str
#     Name by which a package references a particular time-array series.
#     The name must be unique among all time-array series used in a package
#
# package.tas.interpolation_methodrecord : list (of strings)
#     List of interpolation methods to use for time array series. Method must be
#     either 'stepwise' or 'linear'.
#
# package.tas.sfacrecord_single : float
#     Scale factor to multiply the time array series data column. Can only be used if
#     there is one time series data column.
#

# ### Method 1: Pass time array series to package constructor

# +
tas = {0.0: 0.000002, 200.0: 0.0000001}
# create recharge package with time array series data
# flopy will generate a warning that there is not yet a time series name
# record for recharray_1
rcha = flopy.mf6.modflow.mfgwfrcha.ModflowGwfrcha(
    model, timearrayseries=tas, recharge="TIMEARRAYSERIES rcharray_1"
)
# finish defining the time array series properties
rcha.tas.time_series_namerecord = "rcharray_1"
rcha.tas.interpolation_methodrecord = "LINEAR"

sim.write_simulation()
success, buff = sim.run_simulation(silent=True, report=True)
if success:
    for line in buff:
        print(line)
else:
    raise ValueError("Failed to run.")

# clean up for next example
model.remove_package("rcha")
# -

# ### Method 2: Initialize time array series through rcha.tas.initialize

# +
# create recharge package with recharge pointing to a time array series
# not yet defined. flopy will generate a warning that there is not yet a
# time series name record for recharray_1
rcha = flopy.mf6.modflow.mfgwfrcha.ModflowGwfrcha(
    model, recharge="TIMEARRAYSERIES rcharray_1"
)
rch_array = 0.000002 * np.ones((15, 10))
rch_array[0, 0] = 0.0001
tas = {0.0: rch_array, 200.0: 0.0000001}
# initialize the time array series
rcha.tas.initialize(
    filename="method2.tas",
    tas_array=tas,
    time_series_namerecord="rcharray_1",
    interpolation_methodrecord="LINEAR",
)

sim.write_simulation()
success, buff = sim.run_simulation(silent=True, report=True)
if success:
    for line in buff:
        print(line)
else:
    raise ValueError("Failed to run.")

# clean up for next example
model.remove_package("rcha")
# -

# ### Method 3: Pass timearrayseries a dictionary of anything that could be passed to rcha.tas.initialize

# +
rch_array = 0.0000001 * np.ones((15, 10))
rch_array[0, 0] = 0.0001
tas = {
    0.0: 0.000002,
    200.0: rch_array,
    "filename": "method3.tas",
    "time_series_namerecord": "rcharray_1",
    "interpolation_methodrecord": "LINEAR",
}
rcha = flopy.mf6.modflow.mfgwfrcha.ModflowGwfrcha(
    model, timearrayseries=tas, recharge="TIMEARRAYSERIES rcharray_1"
)

sim.write_simulation()
success, buff = sim.run_simulation(silent=True, report=True)
if success:
    for line in buff:
        print(line)
else:
    raise ValueError("Failed to run.")

# clean up for next example
model.remove_package("rcha")
