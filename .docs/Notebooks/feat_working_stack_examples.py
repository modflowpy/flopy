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
#     section: flopy
# ---

# # FloPy working stack demo
#
# A short demonstration of core `flopy` functionality

import sys
from pathlib import Path
from pprint import pformat
from tempfile import TemporaryDirectory

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pooch
from IPython.display import clear_output, display

# First create a temporary workspace.

# create a temporary workspace
temp_dir = TemporaryDirectory()
workspace = Path(temp_dir.name)

# run installed version of flopy or add local path
import flopy

print(sys.version)
print(f"numpy version: {np.__version__}")
print(f"matplotlib version: {mpl.__version__}")
print(f"pandas version: {pd.__version__}")
print(f"flopy version: {flopy.__version__}")
# -

# ### Model Inputs

# first lets load an existing model

sim_name = "freyberg_multilayer_transient"
file_names = {
    "freyberg.bas": "781585c140d40a27bce9369baee262c621bcf969de82361ad8d6b4d8c253ee02",
    "freyberg.cbc": "d4e18e968cabde8470fcb7cb8a1c4cc57fcd643bd63b23e7751460bfdb651ea4",
    "freyberg.ddn": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "freyberg.dis": "1ef61a467a219c036e58902ce11297e06b4eeb5f2f9d2ea40245b421a248a471",
    "freyberg.drn": "93c22ab27d599938a8c2fc5b420ec03e5251b11b050d6ae1cb23ce2aa1b77997",
    "freyberg.hds": "0b3e911ef35f625d2d046e05a20bc1300341b41028220c5b25ace6f5a267ceef",
    "freyberg.list": "14ec36c22b48d253d6b82c44f36c5bad4f0785b3a3384b386f6b69c4ee2e31bf",
    "freyberg.nam": "9e3747ce6d6229caec55a9357285a96cb4608dae11d90dd165a23e0bb394a2bd",
    "freyberg.nwt": "d66c5cc255d050a0f871639af4af0cef8d48fa59c1c64217de65fc6e7fd78cb1",
    "freyberg.oc": "faefd462d11b9a21c4579420b2156fb616ca642bc1e66fc5eb5e1b9046449e43",
    "freyberg.rch": "93a12742a2d37961d53df0405e39cbecf0e6f14d45b5ca8cbba84a2d90828258",
    "freyberg.upw": "80838be7af2f97c92965bad1d121c252b69d9c66e4885c5f3f49a6e99582deac",
    "freyberg.wel": "dd322655eadff3f618f0835c9277af30720197bd48328aae2d6772f26eef2686",
}
for fname, fhash in file_names.items():
    pooch.retrieve(
        url=f"https://github.com/modflowpy/flopy/raw/develop/examples/data/{sim_name}/{fname}",
        fname=fname,
        path=workspace,
        known_hash=fhash,
    )

ml = flopy.modflow.Modflow.load(
    "freyberg.nam",
    model_ws=workspace,
    verbose=False,
    check=False,
    exe_name="mfnwt",
)

ml.modelgrid

# Let's looks at some plots

ml.upw.plot()

ml.dis.plot()

ml.drn.plot(key="cond")
ml.drn.plot(key="elev")


# Write a shapefile of the DIS package.

# write the shapefile
ml.dis.export(workspace / "freyberg_dis.shp")

# Write a netCDF file with all model inputs.

ml.export(workspace / "freyberg.nc")

# Change model directory and external path, modify inputs and write new input files.

ml.external_path = workspace / "ref"
ml.model_ws = workspace
ml.write_input()

# Now run the model.

success, buff = ml.run_model(silent=True)
assert success, pformat(buff)

# ### Inspecting outputs
#
# First, let's look at the list file. The list file summarizes the model's results.

mfl = flopy.utils.MfListBudget(workspace / "freyberg.list")
df_flux, df_vol = mfl.get_dataframes(start_datetime="10-21-2015")
df_flux

# +
groups = df_flux.groupby(lambda x: x.split("_")[-1], axis=1).groups
df_flux_in = df_flux.loc[:, groups["IN"]]
df_flux_in.columns = df_flux_in.columns.map(lambda x: x.split("_")[0])

df_flux_out = df_flux.loc[:, groups["OUT"]]
df_flux_out.columns = df_flux_out.columns.map(lambda x: x.split("_")[0])


df_flux_delta = df_flux_in - df_flux_out
df_flux_delta.iloc[-1, :].plot(kind="bar", figsize=(10, 10), grid=True)
# -

# Now let's look at the simulated head.

# if you pass the model instance, then the plots will be offset and rotated
h = flopy.utils.HeadFile(workspace / "freyberg.hds", model=ml)
h.times

h.plot(totim=900, contour=True, grid=True, colorbar=True, figsize=(10, 10))

# We can write the heads to a shapefile.

h.to_shapefile(ml.model_ws / "freyburg_head.shp", verbose=False)

# Finally, let's make an animation of the simulated head over the time domain.

f = plt.figure(figsize=(10, 10))
ax = plt.subplot(1, 1, 1, aspect="equal")
for t in h.times[0:-1:10]:
    ax.cla()

    ax.set_title(f"totim: {t:4.0f} days")
    mm = flopy.plot.PlotMapView(model=ml, ax=ax)
    mm.plot_array(h.get_data(totim=t), vmin=0, vmax=20)
    mm.plot_grid(lw=0.5, color="black")

    display(f)
    clear_output(wait=True)
    plt.pause(0.1)

try:
    # ignore PermissionError on Windows
    temp_dir.cleanup()
except:
    pass
