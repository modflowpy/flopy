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
# ---

# # FloPy plugin Tutorial 1: Using existing FloPy plugins.
#
# This tutorial demonstrates use of FloPy plugins in a simple MODFLOW 6 model.
#
# In this tutorial the rvp FloPy plugin is used with a simple model.  The rvp
# FloPy plugin behaves like the riv package with the added feature of allowing
# the modeler to specify a different stream bed conductance depending on the
# direction of flow.  Using FloPy plugins is very similar to using MODFLOW-6
# packages.

# ## Creating a simple MODFLOW-6 simulation

from tempfile import TemporaryDirectory
import flopy
import matplotlib.pyplot as plt


temp_dir = TemporaryDirectory()
workspace = temp_dir.name
name = "t01_mf6_plugin"

# ### Create the FloPy simulation object

# In order to run a simulation with a FloPy plug-in, the MODFLOW-6 BMI
# interface must be installed.  To help FloPy find the MODFLOW-6 BMI dll on
# your computer, specify the "dll_name" parameter when constructing your
# MFSimulation object.

sim = flopy.mf6.MFSimulation(
    sim_name=name,
    exe_name="mf6",
    dll_name="libmf6",
    version="mf6",
    sim_ws=workspace,
)

# ### Create the Flopy `TDIS` object

tdis = flopy.mf6.ModflowTdis(
    sim, pname="tdis", time_units="DAYS", nper=1, perioddata=[(1.0, 1, 1.0)]
)

# ### Create the Flopy `IMS` Package object

ims = flopy.mf6.ModflowIms(
    sim,
    pname="ims",
    complexity="SIMPLE",
    linear_acceleration="BICGSTAB",
)

# Create the Flopy groundwater flow (gwf) model object

model_nam_file = f"{name}.nam"
gwf = flopy.mf6.ModflowGwf(
    sim,
    modelname=name,
    model_nam_file=model_nam_file,
    save_flows=True,
    newtonoptions="NEWTON UNDER_RELAXATION",
)

# ### Create the discretization (`DIS`) Package

dis = flopy.mf6.ModflowGwfdis(
    gwf,
    nlay=2,
    nrow=10,
    ncol=10,
    delr=500,
    delc=500,
    top=100.0,
    botm=[50.0, 0.0],
    idomain=1,
)

# ### Create the initial conditions (`IC`) Package

ic = flopy.mf6.ModflowGwfic(gwf, pname="ic", strt=[80.0, 40.0])

# ### Create the output control (`OC`) Package

headfile = f"{name}.hds"
head_filerecord = [headfile]
budgetfile = f"{name}.cbb"
budget_filerecord = [budgetfile]
saverecord = [("HEAD", "ALL"), ("BUDGET", "ALL")]
printrecord = [("HEAD", "LAST")]
oc = flopy.mf6.ModflowGwfoc(
    gwf,
    saverecord=saverecord,
    head_filerecord=head_filerecord,
    budget_filerecord=budget_filerecord,
    printrecord=printrecord,
)

# ### Create the node property flow (`NPF`) Package

npf = flopy.mf6.ModflowGwfnpf(
    gwf,
    save_specific_discharge=True,
    icelltype=1,
    k=10.0,
    k33=1.0,
)

# ### Create the constant-head (`CHD`) Package
#
# Create constant head boundaries along the left and right edges of the model.
chd_spd = []
for row in range(0, 10):
    chd_spd.append(((0, row, 0), 80.0))
    chd_spd.append(((0, row, 9), 70.0))
chd = flopy.mf6.ModflowGwfchd(
    gwf,
    print_input=True,
    save_flows=True,
    stress_period_data=chd_spd,
)

# ## Adding the FloPy plugin to a simulation
#
# ### Create FloPy river plugin (`RVP`) stress period data
#
# Create a river that runs parallel to the constant head boundaries down the
# middle of the model.  The upstream portion of the river has a high head
# (>90 ft), above the head of either chd boundary, while the downstream
# portion has a low head (<60 ft), below either chd boundary.  Conductance
# into the river (cond_up) is very high (10,000) while conductance out of the
# river is much lower (500.0).
# Stress period data fields are:
#    cellid, stage, conductance_up, conductance_down, river_bottom

spd = []
for row in range(0, 5):
    spd.append(((0, row, 5), 95.0 - row, 10000.0, 500.0, 70.0 - row))
for row in range(5, 10):
    spd.append(((0, row, 5), 60.0 - row, 10000.0, 500.0, 40.0 - row))

# ### Create a FloPy river plugin with duel conductances (`RVP`)
#
# FloPy plugins are accessible from the same location as MODFLOW-6 packages,
# and are added to a model in the same way as a MODFLOW-6 package. FloPy also
# stores FloPy plugin data in the same file format as MODFLOW-6 package data
# is stored.

rvp = flopy.mf6.ModflowGwffp_Rvp(
    gwf,
    pname="rvp_0",
    print_input=True,
    save_flows=True,
    stress_period_data=spd,
)

# Note that FloPy plugins can also be installed as separate python packages.
# In this case you will need to install the flopy plugin's python package
# and import the FloPy plugin from that package.  FloPy plugins that are not
# part of the FloPy distribution have not necessarily gone through any
# approval process, so use at your own risk.

# ## Writing and running the simulation
#
# Writing the simulation writes out both the MODFLOW-6 files and the FloPy
# plug-in files.  Both FloPy plugins and MODFLOW-6 packages have user-defined
# data which is stored in formatted files.  FloPy also writes plugin
# information to the model name file on (commented) lines that MODFLOW-6
# ignores.

sim.write_simulation()

# When FloPy detects a flopy plug-in in your simulation it uses the modflowapi
# python package to run your simulation through the MODFLOW-6 BMI interface
# (using the libmf6 dll instead of using the mf6 executable).

success = sim.run_simulation()

# ## Post-processing model output
#
# ### Get and process specific discharge and head output

flow_rvp = gwf.oc.output.budget().get_data(
    text="API",
    kstpkper=(0, 0),
)[0]
fjf = gwf.oc.output.budget().get_data(text="FLOW-JA-FACE", kstpkper=(0, 0))[0]
spdis = gwf.oc.output.budget().get_data(text="SPDIS", kstpkper=(0, 0))[0]
head = gwf.output.head().get_data(kstpkper=(0, 0))
qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(spdis, gwf)

# ### Plot the volumetric discharge
#
# The volumetric discharge plot shows that the lower portion of the river,
# where flow is into the river, has a much greater effect on the direction and
# amount of flow than the upper portion.  This is primarily due to the
# differing river conductance ("cond_down" being much lower than "cond_up").

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(1, 2, 1, aspect="equal")
ax.set_title("Volumetric discharge (" + r"$L^3/T$" + ")")
mapview = flopy.plot.PlotMapView(model=gwf)
quadmesh = mapview.plot_ibound()
quadmesh_2 = mapview.plot_array(head, alpha=0.5)
quiver = mapview.plot_vector(qx, qy)
linecollection = mapview.plot_grid()
