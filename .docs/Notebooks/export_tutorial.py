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
#     section: flopy
# ---

# # Exporting to netCDF and shapefile

# +
import os
import sys
from tempfile import TemporaryDirectory

import flopy

print(sys.version)
print(f"flopy version: {flopy.__version__}")
# -

# Load our old friend...the Freyberg model

nam_file = "freyberg.nam"
model_ws = os.path.join("..", "..", "examples", "data", "freyberg_multilayer_transient")
ml = flopy.modflow.Modflow.load(nam_file, model_ws=model_ws, check=False)

# We can see the ``Modelgrid`` instance has generic entries, as does ``start_datetime``

ml.modelgrid

ml.modeltime.start_datetime

# Setting the attributes of the ``ml.modelgrid`` is easy:

ml.modelgrid.set_coord_info(xoff=123456.7, yoff=765432.1, angrot=15.0, crs=3070)
ml.dis.start_datetime = "7/4/1776"

ml.modeltime.start_datetime

# ### Basic netCDF export capabilities

# temporary directory
temp_dir = TemporaryDirectory()
pth = temp_dir.name

# export the whole model (inputs and outputs)
fnc = ml.export(os.path.join(pth, f"{ml.name}.in.nc"))

# export outputs using spatial reference info
hds = flopy.utils.HeadFile(os.path.join(model_ws, "freyberg.hds"))
flopy.export.utils.output_helper(
    os.path.join(pth, f"{ml.name}.out.nc"), ml, {"hds": hds}
)

# ### Export an array to netCDF or shapefile

# export a 2d array
ml.dis.top.export(os.path.join(pth, "top.nc"))
ml.dis.top.export(os.path.join(pth, "top.shp"))

# #### sparse export of stress period data for a boundary condition package
# * excludes cells that aren't in the package (aren't in `package.stress_period_data`)
# * by default, stress periods with duplicate parameter values (e.g., stage, conductance, etc.) are omitted
# (`squeeze=True`); only stress periods with different values are exported
# * argue `squeeze=False` to export all stress periods

ml.drn.stress_period_data.export(os.path.join(pth, "drn.shp"), sparse=True)

# #### Export a 3d array

# export a 3d array
ml.upw.hk.export(os.path.join(pth, "hk.nc"))
ml.upw.hk.export(os.path.join(pth, "hk.shp"))

# #### Export a number of things to the same netCDF file

# +
# export lots of things to the same nc file
fnc = ml.dis.botm.export(os.path.join(pth, "test.nc"))
ml.upw.hk.export(fnc)
ml.dis.top.export(fnc)

# export transient 2d
ml.rch.rech.export(fnc)
# -

# ### Export a package to netCDF

# export mflist
fnc = ml.wel.export(os.path.join(pth, "packages.nc"))
ml.upw.export(fnc)

# ### Export an entire model to netCDF

fnc = ml.export(os.path.join(pth, "model.nc"))

# ### Export model outputs to netCDF
#
# FloPy has utilities to export model outputs to a netcdf file. Valid output types for export are MODFLOW binary head files, formatted head files, cell budget files, seawat concentration files, and zonebudget output.
#
# Let's use output from the Freyberg model as an example of these functions

# +
# load binary head and cell budget files
fhead = os.path.join(model_ws, "freyberg.hds")
fcbc = os.path.join(model_ws, "freyberg.cbc")

hds = flopy.utils.HeadFile(fhead)
cbc = flopy.utils.CellBudgetFile(fcbc)

export_dict = {"hds": hds, "cbc": cbc}

# export head and cell budget outputs to netcdf
fnc = flopy.export.utils.output_helper(os.path.join(pth, "output.nc"), ml, export_dict)
# -

try:
    # ignore PermissionError on Windows
    temp_dir.cleanup()
except:
    pass
