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

# # Accessing MODFLOW 6 Output
#
# This tutorial shows how to access output from MODFLOW 6 models and packages
# by using the built in `.output` attribute on any MODFLOW 6 model or
# package object

from pathlib import Path
from shutil import copytree
from tempfile import TemporaryDirectory

import numpy as np
import pooch

# ## Package import
import flopy

# ## Load a simple demonstration model

exe_name = "mf6"
sim_name = "test001e_UZF_3lay"

temp_dir = TemporaryDirectory()
sim_ws = Path(temp_dir.name)

files = {
    "chd_spd.txt": "4d87f60022832372981caa2bd162681d5c4b8b3fcf8bc7f5de533c96ad1ed03c",
    "mfsim.nam": "2f7889dedb9e7befb45251f08f015bd5531a4952f4141295ebad9e550be365fd",
    "simulation.tdis": "d466787698c88b7f229cf244f2c9f226a87628c0a5748819e4e34fd4edc48d4c",
    "test001e_UZF_3lay.chd": "96a00121e7004b152a03d0759bf4abfd70f8a1ea21cbce6b9441f18ce4d89b45",
    "test001e_UZF_3lay.dis": "d2f879dcba84ec4be8883d6e29ea9197dd0e67c4058fdde7b9e1de737d1e0639",
    "test001e_UZF_3lay.ic": "6e434a9d42ffe1b126b26890476f6893e9ab526f3a4ee96e63d443fd9008e1df",
    "test001e_UZF_3lay.ims": "c4ef9ebe359def38f0e9ed810b61af0aae9a437c57d54b1db00b8dda20e5b67d",
    "test001e_UZF_3lay.nam": "078ea7b0a774546a93c2cedfb98dc686395332ece7df6493653b072d43b4b834",
    "test001e_UZF_3lay.npf": "89181af1fd91fe59ea931aae02fe64f855f27c705ee9200c8a9c23831aa7bace",
    "test001e_UZF_3lay.obs": "b9857f604c0594a466255f040bd5a47a1687a69ae3be749488bd8736ead7d106",
    "test001e_UZF_3lay.oc": "5eb327ead17588a1faa8b5c7dd37844f7a63d98351ef8bb41df1162c87a94d02",
    "test001e_UZF_3lay.sto": "8d808d0c2ae4edc114455db3f1766446f9f9d6d3775c46a70369a57509bff811",
    "test001e_UZF_3lay.uzf": "97624f1102abef4985bb40f432523df76bd94e069ac8a4aa17455d0d5b8f146e",
    "test001e_UZF_3lay_obs.hed": "78c67035fc6f0c5c1d6090c1ce1e50dcab75b361e4ed44dc951f11fd3915a388",
}

for fname, fhash in files.items():
    pooch.retrieve(
        url=f"https://github.com/modflowpy/flopy/raw/develop/examples/data/mf6/{sim_name}/{fname}",
        fname=fname,
        path=sim_ws,
        known_hash=fhash,
    )

# load the model
sim = flopy.mf6.MFSimulation.load(
    sim_ws=sim_ws,
    exe_name=exe_name,
    verbosity_level=0,
)
# change the simulation path, rewrite the files, and run the model
sim.set_sim_path(sim_ws)
sim.write_simulation(silent=True)
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
# | budgetcsv()           | Method to get the MODFLOW-6 budget csv as a    |
# |                       | `CsvFile` object. Valid for model, oc, and     |
# |                       | advanced packages such as MAW, UZF, LAK        |
# +-----------------------+------------------------------------------------+
# | zonebudget()          | Method to get the `ZoneBudget6` object for     |
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

budcsv = ml.output.budgetcsv()
budcsv.data

hds = ml.output.head()
hds.get_data()

bud = ml.oc.output.budget()
bud.get_data(idx=0, full3D=True)

hds = ml.oc.output.head()
hds.get_data()

# ## Get output associated with a specific package
# The `.output` attribute is tied to the package object and allows the user
# to get the output types specified in the MODFLOW 6 package. Here is an
# example with a UZF package that has UZF budget file output, budgetcsv
# file output, package convergence output, and observation data.

uzf = ml.uzf
uzf_bud = uzf.output.budget()
uzf_bud.get_data(idx=0)

uzf_budcsv = uzf.output.budgetcsv()
uzf_budcsv.data

uzf_conv = uzf.output.package_convergence()
if uzf_conv is not None:
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

# ## Creating and running ZoneBudget for MF6
# For the model and many packages, zonebudget can be run on the cell budget
# file. The `.output` method allows the user to easily build a ZoneBudget6
# instance, then run the model, and view output. First we'll build a layered
# zone array, then build and run zonebudget

zarr = np.ones(ml.modelgrid.shape, dtype=int)
for i in range(1, 4):
    zarr[i - 1] *= i

zonbud = ml.output.zonebudget(zarr)
zonbud.change_model_ws(sim_ws)
zonbud.write_input()
zonbud.run_model()

df = zonbud.get_dataframes(net=True)
df = df.reset_index()
df

try:
    temp_dir.cleanup()
except:
    # prevent windows permission error
    pass
