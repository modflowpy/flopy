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
from tempfile import TemporaryDirectory

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# +
from IPython.display import clear_output, display

proj_root = Path.cwd().parent.parent

# run installed version of flopy or add local path
try:
    import flopy
except:
    sys.path.append(proj_root)
    import flopy

print(sys.version)
print("numpy version: {}".format(np.__version__))
print("matplotlib version: {}".format(mpl.__version__))
print("pandas version: {}".format(pd.__version__))
print("flopy version: {}".format(flopy.__version__))
# -

# ### Model Inputs

# first lets load an existing model
model_ws = proj_root / "examples" / "data" / "freyberg_multilayer_transient"
ml = flopy.modflow.Modflow.load(
    "freyberg.nam",
    model_ws=model_ws,
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

# First create a temporary workspace.

# create a temporary workspace
temp_dir = TemporaryDirectory()
workspace = Path(temp_dir.name)

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

ml.run_model(silent=True)

# ### Inspecting outputs
#
# First, let's look at the list file. The list file summarizes the model's results.

mfl = flopy.utils.MfListBudget(model_ws / "freyberg.list")
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
h = flopy.utils.HeadFile(model_ws / "freyberg.hds", model=ml)
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
