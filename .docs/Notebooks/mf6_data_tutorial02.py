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

# # MODFLOW 6: Observation packages
#

# ## Introduction to Observations
#
# Observations can be set for any package through the `package.obs` object, and
# each package.obs object has several attributes that can be set:
#
# | Attribute | Type | Description |
# | :---      | :---- | :----      |
# | package.obs.filename | str | Name of observations file to create. The default is packagename + '.obs'.|
# | package.obs.continuous | dict | A dictionary that has file names as keys and a list of observations as the dictionary values. |
# | package.obs.digits | int | Number of digits to write the observation values. Default is 10. |
# | package.obs.print_input | bool | Flag indicating whether or not observations are written to listing file. |
#
# The following code sets up a simulation used in the observation examples.

# package import
from tempfile import TemporaryDirectory

import numpy as np

import flopy

# set up where simulation workspace will be stored
temp_dir = TemporaryDirectory()
workspace = temp_dir.name
name = "tutorial02_mf6_data"

# create the flopy simulation and tdis objects
sim = flopy.mf6.MFSimulation(
    sim_name=name, exe_name="mf6", version="mf6", sim_ws=workspace
)
tdis_rc = [(1.0, 1, 1.0), (10.0, 5, 1.0), (10.0, 5, 1.0), (10.0, 1, 1.0)]
tdis_package = flopy.mf6.modflow.mftdis.ModflowTdis(
    sim, time_units="DAYS", nper=4, perioddata=tdis_rc
)
# create the flopy groundwater flow (gwf) model object
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

# ## Observation Example 1
#
# One method to build the observation package is to pass a dictionary with
# the observations containing "observations" parameters of the parent package.
#
# This example uses the observation package in a `GHB` package.  First the
# stress period data for a ghb package is built.

# build ghb stress period data
ghb_spd = {}
ghb_period = []
for layer, cond in zip(range(1, 3), [15.0, 1500.0]):
    for row in range(0, 15):
        ghb_period.append(((layer, row, 9), 1.0, cond, "Estuary-L2"))
ghb_spd[0] = ghb_period

# The next step is to build the observation data in a dictionary.  The
# dictionary key is the filename of the observation output file and
# optionally a "binary" keyword to make the file binary.  When the optional
# "binary" keyword is used the dictionary key is a tuple, otherwise it is a
# string.  The dictionary value is a list of tuples containing the contents
# of the observation package's continuous block, with each tuple containing
# one line of information.

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

# The ghb package is now constructed with observations by setting the
# `observations` parameter to `ghb_obs` on construction of the ghb package.

# build ghb package passing obs dictionary to package constructor
ghb = flopy.mf6.modflow.mfgwfghb.ModflowGwfghb(
    gwf,
    print_input=True,
    print_flows=True,
    save_flows=True,
    boundnames=True,
    observations=ghb_obs,
    pname="ghb",
    maxbound=30,
    stress_period_data=ghb_spd,
)

# Observation information such as the print_input option can then be set using
# the package's `obs` parameter.

ghb.obs.print_input = True

# clean up for next example
gwf.remove_package("ghb")

# ## Observation Example 2
#
# Alternatively, an obs package can be built by initializing obs
# through `ghb.obs.initialize`.

# First, a `GHB` package is built without defining observations.

# build ghb package
ghb = flopy.mf6.modflow.mfgwfghb.ModflowGwfghb(
    gwf,
    print_input=True,
    print_flows=True,
    save_flows=True,
    boundnames=True,
    maxbound=30,
    stress_period_data=ghb_spd,
    pname="ghb",
)

# Then the ghb observations are defined in a dictionary similar to example 1.

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

# The observations can then be added to the ghb package using the obs
# attribute's initialize method.  The observation package's file name,
# digits, and print_input options, along with the continuous block data
# are set in the initialize method.

# initialize obs package
ghb.obs.initialize(
    filename="child_pkgs_test.ghb.obs",
    digits=9,
    print_input=True,
    continuous=ghb_obs,
)

try:
    temp_dir.cleanup()
except PermissionError:
    # can occur on windows: https://docs.python.org/3/library/tempfile.html#tempfile.TemporaryDirectory
    pass
