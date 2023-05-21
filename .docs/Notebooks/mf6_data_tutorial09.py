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

# # MODFLOW 6: Multiple Models - How to create multiple models in a simulation
#
# This tutorial shows a simulation using two models, demonstrating how to use
# exchanges and exchange subpackages.
#
# ## Introduction to Multiple Models
# MODFLOW-6 simulations can contain multiple models, which can be linked
# through the groundwater exchange package, which can contain mover and
# ghost node correction subpackages.
#
# The following code sets up a basic simulation.

# package import
from pathlib import Path
from tempfile import TemporaryDirectory

import flopy

# set up where simulation workspace will be stored
temp_dir = TemporaryDirectory()
workspace = temp_dir.name
name = "tutorial09_mf6_data"

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

# ## Groundwater Flow Model Setup
# We will start by setting up two groundwater flow models that are part of the
# same simulation.

# set up first groundwater flow model
name_1 = "ex_1_mod_1"
model_nam_file = f"{name_1}.nam"
gwf = flopy.mf6.ModflowGwf(
    sim, modelname=name_1, model_nam_file=model_nam_file
)
# create the discretization package
bot = [-10.0, -50.0, -200.0]
delrow = delcol = 4.0
nlay = 3
nrow = 10
ncol = 10
dis = flopy.mf6.modflow.mfgwfdis.ModflowGwfdis(
    gwf,
    pname="dis-1",
    nogrb=True,
    nlay=nlay,
    nrow=nrow,
    ncol=ncol,
    delr=delrow,
    delc=delcol,
    top=0.0,
    botm=bot,
)
# create npf package
npf = flopy.mf6.ModflowGwfnpf(
    gwf,
    pname="npf-1",
    save_flows=True,
    icelltype=[1, 1, 1],
    k=10.0,
    k33=5.0,
    xt3doptions="xt3d rhs",
    # rewet_record="REWET WETFCT 1.0 IWETIT 1 IHDWET 0",
)
# create ic package
ic_package = flopy.mf6.modflow.mfgwfic.ModflowGwfic(gwf, strt=0.0)

# create ghb package
ghb_spd = {0: [((0, 0, 0), -1.0, 1000.0)]}
ghb = flopy.mf6.modflow.mfgwfghb.ModflowGwfghb(
    gwf,
    print_input=True,
    print_flows=True,
    pname="ghb-1",
    maxbound=1,
    stress_period_data=ghb_spd,
)

# create wel package
welspd = {0: [((0, 5, ncol - 1), -100.0)]}
wel = flopy.mf6.ModflowGwfwel(
    gwf,
    print_input=True,
    print_flows=True,
    mover=True,
    stress_period_data=welspd,
    save_flows=False,
    pname="WEL-1",
)

# set up second groundwater flow model with a finer grid
name_1 = "ex_1_mod_2"
model_nam_file = f"{name_1}.nam"
gwf_2 = flopy.mf6.ModflowGwf(
    sim, modelname=name_1, model_nam_file=model_nam_file
)
# create the flopy iterative model solver (ims) package object
# by default flopy will register both models with the ims package.
ims = flopy.mf6.modflow.mfims.ModflowIms(
    sim, pname="ims", complexity="SIMPLE", linear_acceleration="BICGSTAB"
)
# no need to create a new ims package.  flopy will automatically register
# create the discretization package
bot = [-10.0, -50.0, -200.0]
dis_2 = flopy.mf6.modflow.mfgwfdis.ModflowGwfdis(
    gwf_2,
    pname="dis-2",
    nogrb=True,
    nlay=nlay,
    nrow=nrow * 2,
    ncol=ncol * 2,
    delr=delrow / 2.0,
    delc=delcol / 2.0,
    top=0.0,
    botm=bot,
)
# create npf package
npf_2 = flopy.mf6.ModflowGwfnpf(
    gwf_2,
    pname="npf-2",
    save_flows=True,
    icelltype=[1, 1, 1],
    k=10.0,
    k33=5.0,
    xt3doptions="xt3d rhs",
    # rewet_record="REWET WETFCT 1.0 IWETIT 1 IHDWET 0",
)
# create ic package
ic_package_2 = flopy.mf6.modflow.mfgwfic.ModflowGwfic(gwf_2, strt=0.0)

# create ghb package
ghb_spd = {0: [((0, 0, 19), -10.0, 1000.0)]}
ghb_2 = flopy.mf6.modflow.mfgwfghb.ModflowGwfghb(
    gwf_2,
    print_input=True,
    print_flows=True,
    pname="ghb-2",
    maxbound=1,
    stress_period_data=ghb_spd,
)

# create lak package
lakpd = [(0, -2.0, 1)]
lakecn = [(0, 0, (0, 5, 0), "HORIZONTAL", 1.0, -5.0, 0.0, 10.0, 10.0)]
lak_2 = flopy.mf6.ModflowGwflak(
    gwf_2,
    pname="lak-2",
    print_input=True,
    mover=True,
    nlakes=1,
    noutlets=0,
    ntables=0,
    packagedata=lakpd,
    connectiondata=lakecn,
)

# ## Connecting the Flow Models
# The two groundwater flow models created above are now part of the same
# simulation, but they are not connected. To connect them we will use the
# gwfgwf package and two of its subpackages, gnc and mvr.

# Use exchangedata to define how the two models are connected.  In this
# example we are connecting the right edge of the first model to the left edge
# of the second model. The second model is 2x the discretization, so each cell
# in the first model is connected to two cells in the second model.

gwfgwf_data = []
row_2 = 0
for row in range(0, nrow):
    gwfgwf_data.append([(0, ncol - 1, row), (0, 0, row_2), 1, 2.03, 1.01, 2.0])
    row_2 += 1
    gwfgwf_data.append([(0, ncol - 1, row), (0, 0, row_2), 1, 2.03, 1.01, 2.0])
    row_2 += 1

# create the gwfgwf package
gwfgwf = flopy.mf6.ModflowGwfgwf(
    sim,
    exgtype="GWF6-GWF6",
    nexg=len(gwfgwf_data),
    exgmnamea=gwf.name,
    exgmnameb=gwf_2.name,
    exchangedata=gwfgwf_data,
    filename="mod1_mod2.gwfgwf",
)

# Due to the two model's different cell sizes, the cell centers of the first
# model do not align with the cell centers in the second model. To correct for
# this we will use the ghost node correction package (gnc).

gnc_data = []
col_2 = 0
weight_close = 1.0 / 1.25
weight_far = 0.25 / 1.25
for col in range(0, ncol):
    if col == 0:
        gnc_data.append(
            (
                (0, nrow - 1, col),
                (0, 0, col_2),
                (0, nrow - 1, col),
                (0, nrow - 1, 0),
                1.00,
                0.0,
            )
        )
    else:
        gnc_data.append(
            (
                (0, nrow - 1, col),
                (0, 0, col_2),
                (0, nrow - 1, col),
                (0, nrow - 1, col - 1),
                weight_close,
                weight_far,
            )
        )
    col_2 += 1
    if col == ncol - 1:
        gnc_data.append(
            (
                (0, nrow - 1, col),
                (0, 0, col_2),
                (0, nrow - 1, col),
                (0, nrow - 1, 0),
                1.00,
                0.0,
            )
        )
    else:
        gnc_data.append(
            (
                (0, nrow - 1, col),
                (0, 0, col_2),
                (0, nrow - 1, col),
                (0, nrow - 1, col + 1),
                weight_close,
                weight_far,
            )
        )
    col_2 += 1

# set up gnc package
fname = "gwfgwf.input.gnc"
gwfgwf.gnc.initialize(
    filename=fname,
    print_input=True,
    print_flows=True,
    numgnc=ncol * 2,
    numalphaj=2,
    gncdata=gnc_data,
)

# The extraction well at the right-hand side of the first model is pumping
# the water it extracts into a nearby lake at the left-hand side of the
# second model. Using the mover (mvr) package, water extracted from the first
# model's wel package is moved to the second model's lak package.

package_data = [(gwf.name, "WEL-1"), (gwf_2.name, "lak-2")]
period_data = [(gwf.name, "WEL-1", 0, gwf_2.name, "lak-2", 0, "FACTOR", 1.0)]
fname = "gwfgwf.input.mvr"
gwfgwf.mvr.initialize(
    filename=fname,
    modelnames=True,
    print_input=True,
    print_flows=True,
    maxpackages=2,
    maxmvr=1,
    packages=package_data,
    perioddata=period_data,
)

sim.write_simulation()
sim.run_simulation()

try:
    temp_dir.cleanup()
except PermissionError:
    # can occur on windows: https://docs.python.org/3/library/tempfile.html#tempfile.TemporaryDirectory
    pass
