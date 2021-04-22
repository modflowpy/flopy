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

# # MODFLOW 6 Tutorial: Accessing MODFLOW 6 output
#
# This tutorial shows how to access output from MODFLOW 6 models and packages
# by using the built in `.output` attribute on any MODFLOW 6 model or
# package object

# ## Package import
import flopy
import os

# ## Load a simple demonstration model
ws = os.path.abspath(os.path.dirname(__file__))
if os.path.split(ws)[-1] == "modflow6":
    sim_ws = os.path.join(ws, "..", "..", 'data', 'mf6', 'test001e_UZF_3lay')
else:
    sim_ws = os.path.join(ws, '..', '..', 'examples', 'data', 'mf6', 'test001e_UZF_3lay')
sim = flopy.mf6.MFSimulation.load(sim_ws=sim_ws)
sim.run_simulation(silent=True)

# ## Get output using the `.output` attribute
# The output attribute dynamically generates methods for each package based on
# the available output options within that package. A list of all available
# outputs are:
#
# +-----------------------+------------------------------------------------+
# | head()                | Method to get the `HeadFile` object for the    |
# |                       | model. Accessed from the model object or the   |
# |                       | OC package object                              |
# +-----------------------+------------------------------------------------+
# | budget()              | Method to get the `CellBudgetFile` object for  |
# |                       | the model. Accessed from the model object or   |
# |                       | the OC package object                          |
# +-----------------------+------------------------------------------------+
# | obs()                 | Method to get observation file data in the     |
# |                       | form of a `MF6Obs` object. Accessed from any   |
# |                       | package that allows observations.              |
# +-----------------------+------------------------------------------------+
# | csv()                 | Method to get csv output data in the form of a |
# |                       | `CsvFile` object. Example files are inner and  |
# |                       | outer iteration files from IMS                 |
# +-----------------------+------------------------------------------------+
# | package_convergence() | Method to get csv based package convergence    |
# |                       | information from packages such as SFR, LAK,    |
# |                       | UZF, and MAW. Returns a `CsvFile` object       |
# +-----------------------+------------------------------------------------+
# | stage()               | Method to get binary stage file output from    |
# |                       | the SFR and LAK packages                       |
# +-----------------------+------------------------------------------------+
# | concentration()       | Method to get the binary concentration file    |
# |                       | output from a groundwater transport model.     |
# |                       | Accessed from the model object or the OC       |
# |                       | package object                                 |
# +-----------------------+------------------------------------------------+
# | cim()                 | Method to get immobile concentration output    |
# |                       | from the CIM package                           |
# +-----------------------+------------------------------------------------+
# | density()             | Method to get density file output from the     |
# |                       | BUY package                                    |
# +-----------------------+------------------------------------------------+

# ## Get head file and cell budget file outputs
# The head file output and cell budget file output can be loaded from either
# the model object or the OC package object.

ml = sim.get_model("gwf_1")

bud = ml.output.budget()
bud.get_data(idx=0, full3D=True)

hds = ml.output.head()
hds.get_data()

bud = ml.oc.output.budget()
bud.get_data(idx=0, full3D=True)

hds = ml.oc.output.head()
hds.get_data()

# ## Get output associated with a specific package
# The `.output` attribute is tied to the package object and allows the user
# to get the output types specified in the MODFLOW 6 package. Here is an
# example with a UZF package that has UZF budget file output,
# package convergence output, and observation data.

uzf = ml.uzf
uzf_bud = uzf.output.budget()
uzf_bud.get_data(idx=0)

uzf_conv = uzf.output.package_convergence()
uzf_conv.data[0:10]

uzf_obs = uzf.output.obs()
uzf_obs.data[0:10]

# ## Check which output types are available in a package
# The `.output` attribute also has a `methods()` function that returns a list
# of available output functions for a given package. Here are a couple of
# examples

print("UZF package: ", uzf.output.methods())
print("Model object: ", ml.output.methods())
print("OC package: ", ml.oc.output.methods())
print("DIS package: ", ml.dis.output.methods())

# ## Managing multiple observation and csv file outputs in the same package
# For many packages, multiple observation output files can be used. The
# `obs()` and `csv()` functions allow the user to specify a observation file
# or csv file name. If no name is specified, the `obs()` and `csv()` methods
# will return the first file that is listed in the package.

output = ml.obs[0].output
obs_names = output.obs_names
output.obs(f=obs_names[0]).data[0:10]
