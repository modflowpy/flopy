# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: all
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
#     section: misc
#     authors:
#       - name: Jason Bellino
# ---

# # ZoneBudget Example
#
# This notebook demonstrates how to use the `ZoneBudget` class to extract budget information from the cell by cell budget file using an array of zones.
#
# First set the path and import the required packages. The flopy path doesn't have to be set if you install flopy from a binary installer. If you want to run this notebook, you have to set the path to your own flopy path.

# +
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import git
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pooch

proj_root = Path.cwd().parent.parent

import flopy

print(sys.version)
print(f"numpy version: {np.__version__}")
print(f"matplotlib version: {mpl.__version__}")
print(f"pandas version: {pd.__version__}")
print(f"flopy version: {flopy.__version__}")

# +
# temporary workspace
temp_dir = TemporaryDirectory()
workspace = Path(temp_dir.name)

# Check if we are in the repository and define the data path.

try:
    root = Path(git.Repo(".", search_parent_directories=True).working_dir)
except:
    root = None

data_path = root / "examples" / "data" if root else Path.cwd()

folder_name = "zonbud_examples"

fname = "freyberg.gitcbc"
pooch.retrieve(
    url=f"https://github.com/modflowpy/flopy/raw/develop/examples/data/{folder_name}/{fname}",
    fname=fname,
    path=data_path / folder_name,
    known_hash=None,
)

# Set path to example datafiles
loadpth = data_path / "zonbud_examples"
cbc_f = loadpth / fname
# -

# ### Read File Containing Zones
# Using the `ZoneBudget.read_zone_file()` utility, we can import zonebudget-style array files.

# +
from flopy.utils import ZoneBudget

fname = "zonef_mlt.zbr"
pooch.retrieve(
    url=f"https://github.com/modflowpy/flopy/raw/develop/examples/data/{folder_name}/{fname}",
    fname=fname,
    path=data_path / folder_name,
    known_hash=None,
)

zone_file = loadpth / "zonef_mlt.zbr"
zon = ZoneBudget.read_zone_file(zone_file)
nlay, nrow, ncol = zon.shape

fig = plt.figure(figsize=(10, 4))

for lay in range(nlay):
    ax = fig.add_subplot(1, nlay, lay + 1)
    im = ax.pcolormesh(zon[lay, ::-1, :])
    cbar = plt.colorbar(im)
    plt.gca().set_aspect("equal")

plt.show()
# -

# ### Extract Budget Information from ZoneBudget Object
#
# At the core of the `ZoneBudget` object is a numpy structured array. The class provides some wrapper functions to help us interogate the array and save it to disk.

# Create a ZoneBudget object and get the budget record array
zb = flopy.utils.ZoneBudget(cbc_f, zon, kstpkper=(0, 1096))
zb.get_budget()

# Get a list of the unique budget record names
zb.get_record_names()

# Look at a subset of fluxes
names = ["FROM_RECHARGE", "FROM_ZONE_1", "FROM_ZONE_3"]
zb.get_budget(names=names)

# Look at fluxes in from zone 2
names = ["FROM_RECHARGE", "FROM_ZONE_1", "FROM_ZONE_3"]
zones = ["ZONE_2"]
zb.get_budget(names=names, zones=zones)

# Look at all of the mass-balance records
names = ["TOTAL_IN", "TOTAL_OUT", "IN-OUT", "PERCENT_DISCREPANCY"]
zb.get_budget(names=names)

# ### Convert Units
# The `ZoneBudget` class supports the use of mathematical operators and returns a new copy of the object.

# +
cmd = flopy.utils.ZoneBudget(cbc_f, zon, kstpkper=(0, 0))
cfd = cmd / 35.3147
inyr = (cfd / (250 * 250)) * 365 * 12

cmdbud = cmd.get_budget()
cfdbud = cfd.get_budget()
inyrbud = inyr.get_budget()

names = ["FROM_RECHARGE"]
rowidx = np.isin(cmdbud["name"], names)
colidx = "ZONE_1"

print(f"{cmdbud[rowidx][colidx][0]:,.1f} cubic meters/day")
print(f"{cfdbud[rowidx][colidx][0]:,.1f} cubic feet/day")
print(f"{inyrbud[rowidx][colidx][0]:,.1f} inches/year")
# -

cmd is cfd

# ### Alias Names
# A dictionary of {zone: "alias"} pairs can be passed to replace the typical "ZONE_X" fieldnames of the `ZoneBudget` structured array with more descriptive names.

aliases = {1: "SURF", 2: "CONF", 3: "UFA"}
zb = flopy.utils.ZoneBudget(cbc_f, zon, totim=[1097.0], aliases=aliases)
zb.get_budget()

# ### Return the Budgets as a Pandas DataFrame
# Set `kstpkper` and `totim` keyword args to `None` (or omit) to return all times.
# The `get_dataframes()` method will return a DataFrame multi-indexed on `totim` and `name`.

aliases = {1: "SURF", 2: "CONF", 3: "UFA"}
times = list(range(1092, 1097 + 1))
zb = flopy.utils.ZoneBudget(cbc_f, zon, totim=times, aliases=aliases)
zb.get_dataframes()

# Slice the multi-index dataframe to retrieve a subset of the budget.
# NOTE: We can pass "names" directly to the `get_dataframes()` method to return a subset of reocrds. By omitting the `"FROM_"` or `"TO_"` prefix we get both.

dateidx1 = 1095.0
dateidx2 = 1097.0
names = ["FROM_RECHARGE", "TO_WELLS", "CONSTANT_HEAD"]
zones = ["SURF", "CONF"]
df = zb.get_dataframes(names=names)
df.loc[(slice(dateidx1, dateidx2), slice(None)), :][zones]

# Look at pumpage (`TO_WELLS`) as a percentage of recharge (`FROM_RECHARGE`)

# +
dateidx1 = 1095.0
dateidx2 = 1097.0
zones = ["SURF"]

# Pull out the individual records of interest
rech = df.loc[(slice(dateidx1, dateidx2), ["FROM_RECHARGE"]), :][zones]
pump = df.loc[(slice(dateidx1, dateidx2), ["TO_WELLS"]), :][zones]

# Remove the "record" field from the index so we can
# take the difference of the two DataFrames
rech = rech.reset_index()
rech = rech.set_index(["totim"])
rech = rech[zones]
pump = pump.reset_index()
pump = pump.set_index(["totim"])
pump = pump[zones] * -1

# Compute pumping as a percentage of recharge
(pump / rech) * 100.0
# -

# Pass `start_datetime` and `timeunit` keyword arguments to return a dataframe with a datetime multi-index

dateidx1 = pd.Timestamp("1972-12-29")
dateidx2 = pd.Timestamp("1972-12-30")
names = ["FROM_RECHARGE", "TO_WELLS", "CONSTANT_HEAD"]
zones = ["SURF", "CONF"]
df = zb.get_dataframes(start_datetime="1970-01-01", timeunit="D", names=names)
df.loc[(slice(dateidx1, dateidx2), slice(None)), :][zones]

# Pass `index_key` to indicate which fields to use in the multi-index (default is "totim"; valid keys are "totim" and "kstpkper")

df = zb.get_dataframes(index_key="kstpkper")
df.head()

# ### Write Budget Output to CSV
#
# We can write the resulting recarray to a csv file with the `.to_csv()` method of the `ZoneBudget` object.

# +
zb = flopy.utils.ZoneBudget(cbc_f, zon, kstpkper=[(0, 0), (0, 1096)])
f_out = workspace / "Example_output.csv"
zb.to_csv(f_out)

# Read the file in to see the contents
try:
    import pandas as pd

    print(pd.read_csv(f_out).to_string(index=False))
except:
    with open(f_out) as f:
        for line in f.readlines():
            print("\t".join(line.split(",")))
# -

# ### Net Budget
# Using the "net" keyword argument, we can request a net budget for each zone/record name or for a subset of zones and record names. Note that we can identify the record names we want without the added `"_IN"` or `"_OUT"` string suffix.

# +
zon = np.ones((nlay, nrow, ncol), int)
zon[1, :, :] = 2
zon[2, :, :] = 3

aliases = {1: "SURF", 2: "CONF", 3: "UFA"}
times = list(range(1092, 1097 + 1))
zb = flopy.utils.ZoneBudget(cbc_f, zon, totim=times, aliases=aliases)
zb.get_budget(names=["STORAGE", "WELLS"], zones=["SURF", "UFA"], net=True)
# -

df = zb.get_dataframes(names=["STORAGE", "WELLS"], zones=["SURF", "UFA"], net=True)
df.head(6)


# ### Plot Budget Components
# The following is a function that can be used to better visualize the budget components using matplotlib.


# +
def tick_label_formatter_comma_sep(x, pos):
    return f"{x:,.0f}"


def volumetric_budget_bar_plot(values_in, values_out, labels, **kwargs):
    if "ax" in kwargs:
        ax = kwargs.pop("ax")
    else:
        ax = plt.gca()

    x_pos = np.arange(len(values_in))
    rects_in = ax.bar(x_pos, values_in, align="center", alpha=0.5)

    x_pos = np.arange(len(values_out))
    rects_out = ax.bar(x_pos, values_out, align="center", alpha=0.5)

    plt.xticks(list(x_pos), labels)
    ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=90)
    ax.get_yaxis().set_major_formatter(
        mpl.ticker.FuncFormatter(tick_label_formatter_comma_sep)
    )

    ymin, ymax = ax.get_ylim()
    if ymax != 0:
        if abs(ymin) / ymax < 0.33:
            ymin = -(ymax * 0.5)
        else:
            ymin *= 1.35
    else:
        ymin *= 1.35
    plt.ylim([ymin, ymax * 1.25])

    for i, rect in enumerate(rects_in):
        label = f"{values_in[i]:,.0f}"
        height = values_in[i]
        x = rect.get_x() + rect.get_width() / 2
        y = height + (0.02 * ymax)
        vertical_alignment = "bottom"
        horizontal_alignment = "center"
        ax.text(
            x, y, label, ha=horizontal_alignment, va=vertical_alignment, rotation=90
        )

    for i, rect in enumerate(rects_out):
        label = f"{values_out[i]:,.0f}"
        height = values_out[i]
        x = rect.get_x() + rect.get_width() / 2
        y = height + (0.02 * ymin)
        vertical_alignment = "top"
        horizontal_alignment = "center"
        ax.text(
            x, y, label, ha=horizontal_alignment, va=vertical_alignment, rotation=90
        )

    # horizontal line indicating zero
    ax.plot(
        [
            rects_in[0].get_x() - rects_in[0].get_width() / 2,
            rects_in[-1].get_x() + rects_in[-1].get_width(),
        ],
        [0, 0],
        "k",
    )

    return rects_in, rects_out


# +
fig = plt.figure(figsize=(16, 5))

times = [2.0, 500.0, 1000.0, 1095.0]

for idx, t in enumerate(times):
    ax = fig.add_subplot(1, len(times), idx + 1)

    zb = flopy.utils.ZoneBudget(cbc_f, zon, kstpkper=None, totim=t, aliases=aliases)

    recname = "STORAGE"
    values_in = zb.get_dataframes(names=f"FROM_{recname}").T.squeeze()
    values_out = zb.get_dataframes(names=f"TO_{recname}").T.squeeze() * -1
    labels = values_in.index.tolist()

    rects_in, rects_out = volumetric_budget_bar_plot(
        values_in, values_out, labels, ax=ax
    )

    plt.ylabel("Volumetric rate, in Mgal/d")
    plt.title(f"{recname} @ totim = {t}")

plt.tight_layout()
plt.show()
# -

# ## Zonebudget for Modflow 6 (`ZoneBudget6`)
#
# This section shows how to build and run a Zonebudget when working with a MODFLOW 6 model.
#
# First let's load a model

# +
mf6_exe = "mf6"
zb6_exe = "zbud6"

sim_name = "mf6-freyberg"
sim_ws = data_path / sim_name
file_names = {
    "bot.asc": "3107f907cb027460fd40ffc16cb797a78babb31988c7da326c9f500fba855b62",
    "description.txt": "94093335eec6a24711f86d4d217ccd5a7716dd9e01cb6b732bc7757d41675c09",
    "freyberg.cbc": "c8ad843b1da753eb58cf6c462ac782faf0ca433d6dcb067742d8bd698db271e3",
    "freyberg.chd": "d8b8ada8d3978daea1758b315be983b5ca892efc7d69bf6b367ceec31e0dd156",
    "freyberg.dis": "cac230a207cc8483693f7ba8ae29ce40c049036262eac4cebe17a4e2347a8b30",
    "freyberg.dis.grb": "c8c26fb1fa4b210208134b286d895397cf4b3131f66e1d9dda76338502c7e96a",
    "freyberg.hds": "926a06411ca658a89db6b5686f51ddeaf5b74ced81239cab1d43710411ba5f5b",
    "freyberg.ic": "6efb56ee9cdd704b9a76fb9efd6dae750facc5426b828713f2d2cf8d35194120",
    "freyberg.ims": "6dddae087d85417e3cdaa13e7b24165afb7f9575ab68586f3adb6c1b2d023781",
    "freyberg.nam": "cee9b7b000fe35d2df26e878d09d465250a39504f87516c897e3fa14dcda081e",
    "freyberg.npf": "81104d3546045fff0eddf5059465e560b83b492fa5a5acad1907ce18c2b9c15f",
    "freyberg.oc": "c0715acd75eabcc42c8c47260a6c1abd6c784350983f7e2e6009ddde518b80b8",
    "freyberg.rch": "a6ec1e0eda14fd2cdf618a5c0243a9caf82686c69242b783410d5abbcf971954",
    "freyberg.riv": "a8cafc8c317cbe2acbb43e2f0cfe1188cb2277a7a174aeb6f3e6438013de8088",
    "freyberg.sto": "74d748c2f0adfa0a32ee3f2912115c8f35b91011995b70c1ec6ae1c627242c41",
    "freyberg.tdis": "9965cbb17caf5b865ea41a4ec04bcb695fe15a38cb539425fdc00abbae385cbe",
    "freyberg.wel": "f19847de455598de52c05a4be745698c8cb589e5acfb0db6ab1f06ded5ff9310",
    "k11.asc": "b6a8aa46ef17f7f096d338758ef46e32495eb9895b25d687540d676744f02af5",
    "mfsim.nam": "6b8d6d7a56c52fb2bff884b3979e3d2201c8348b4bbfd2b6b9752863cbc9975e",
    "top.asc": "3ad2b131671b9faca7f74c1dd2b2f41875ab0c15027764021a89f9c95dccaa6a",
}
for fname, fhash in file_names.items():
    pooch.retrieve(
        url=f"https://github.com/modflowpy/flopy/raw/develop/examples/data/{sim_name}/{fname}",
        fname=fname,
        path=sim_ws,
        known_hash=fhash,
    )


cpth = workspace / "zbud6"
cpth.mkdir()

sim = flopy.mf6.MFSimulation.load(sim_ws=sim_ws, exe_name=mf6_exe)
sim.simulation_data.mfpath.set_sim_path(cpth)
sim.write_simulation()
success, buff = sim.run_simulation(silent=True, report=True)
assert success, "Failed to run"
for line in buff:
    print(line)
# -

# ### Use the the `.output` model attribute to create a zonebudget model
#
# The `.output` attribute allows the user to access model output and create zonebudget models easily. The user only needs to pass in a zone array to create a zonebudget model!

# +
# let's get our idomain array from the model, split it into two zones, and use it as a zone array
ml = sim.get_model("freyberg")
zones = ml.modelgrid.idomain
zones[0, 20:] = np.where(zones[0, 20:] != 0, 2, 0)

plt.imshow(zones[0])
plt.colorbar()
# -

# now let's build a zonebudget model and run it!
zonbud = ml.output.zonebudget(zones)
zonbud.change_model_ws(cpth)
zonbud.write_input()
success, buff = zonbud.run_model(exe_name=zb6_exe, silent=True)

# ### Getting the zonebudget output
#
# We can then get the output as a recarray using the `.get_budget()` method or as a pandas dataframe using the `.get_dataframes()` method.

zonbud.get_budget()

# get the net flux using net=True flag
zonbud.get_dataframes(net=True)

# we can also pivot the data into a spreadsheet like format
zonbud.get_dataframes(net=True, pivot=True)

# +
# or get a volumetric budget by supplying modeltime
mt = ml.modeltime

# budget recarray must be pivoted to get volumetric budget!
zonbud.get_volumetric_budget(mt, recarray=zonbud.get_budget(net=True, pivot=True))
# -

try:
    # ignore PermissionError on Windows
    temp_dir.cleanup()
except:
    pass
