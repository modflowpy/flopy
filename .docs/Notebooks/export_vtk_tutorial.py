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
#     authors:
#       - url: https://github.com/martindjm
# ---

# # Exporting to VTK
#
# The `Vtk()` class in FloPy allows users to export Structured, Vertex, and Unstructured Grid based models to Visualization ToolKit files for display. This notebook demonstrates how to use FloPy to export to vtk (.vtu) files. This example will cover:
#
#    - basic exporting of information for a model, individual package, or array to `Vtk()`
#    - example usage of the `Vtk()` class object to output data
#    - exporting heads and model output data
#    - exporting modpath pathlines to `Vtk()`

# +
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import flopy
from flopy.export import vtk

sys.path.append(os.path.join("..", "common"))
import notebook_utils

print(sys.version)
print("flopy version: {}".format(flopy.__version__))
# -

# load model for examples
nam_file = "freyberg.nam"
prj_root = notebook_utils.get_project_root_path()
model_ws = prj_root / "examples" / "data" / "freyberg_multilayer_transient"
ml = flopy.modflow.Modflow.load(nam_file, model_ws=model_ws, check=False)

# Create a temporary workspace.

tempdir = TemporaryDirectory()
workspace = Path(tempdir.name)

# ## Using the `.export()` method
#
# For all exports **a folder path must be provided** and the `fmt` flag should be set to 'vtk'.

# ### Exporting FloPy arrays to .vtu files
#
# All array exports have the following optional keyword arguments:
#    - `smooth`: True creates a smooth surface, default is False
#    - `point_scalars`: True outputs point scalar values as well as cell values, default is False.
#    - `name`: A name can be specified to use for the output filename and array scalar name, by default the FloPy array name is used
#    - `binary`: argument that can be specified to switch between binary and ASCII, default is True
#    - `xml`: True will write an xml base vtk file, default is False
#    - `masked_values`: list or tuple of values to mask (set to nan) when writing a array
#    - `vertical_exageration`: floating point value that can be used to scale the vertical exageration of the vtk points. Default is 1.
#
# Tranient type array exports ("stress_period_data"; ex. recharge data, well flux, etc ...) have additional optional keyword arguments:
#    - `pvd`: True will write a paraview data file with simulation time for animations. Default is False
#    - `kper`: a list, tuple, or integer value of specific stess periods to output

# ### Export model top

# +
# create output folder
output_dir = workspace / "arrays_test"
output_dir.mkdir(exist_ok=True)

# export model top
model_top_dir = output_dir / "TOP"
ml.dis.top.export(model_top_dir, fmt="vtk")
# -

# ### Export model bottoms

# 3D Array export
# export model bottoms
model_bottom_dir = output_dir / "BOTM"
ml.dis.botm.export(model_bottom_dir, fmt="vtk")

# ### Export transient array recharge

# transient 2d array
# export recharge
model_recharge_dir = output_dir / "RECH"
ml.rch.rech.export(model_recharge_dir, fmt="vtk", pvd=True)

# ### Export HK with point scalars
#

# 3D Array export
# hk export, with points
model_hk_dir = output_dir / "HK"
ml.upw.hk.export(
    model_hk_dir, smooth=True, fmt="vtk", name="HK", point_scalars=True
)

# ### Package export to .vtu files
#
# Package export has the following keyword arguments:
#    - `smooth`: True creates a smooth surface, default is False
#    - `point_scalars`: True outputs point scalar values as well as cell values, default is False.
#    - `name`: A name can be specified to use for the output filename and array scalar name, by default the FloPy array name is used
#    - `binary`: argument that can be specified to switch between binary and ASCII, default is True
#    - `xml`: True will write an xml base vtk file, default is False
#    - `masked_values`: list or tuple of values to mask (set to nan) when writing a array
#    - `vertical_exageration`: floating point value that can be used to scale the vertical exageration of the vtk points. Default is 1.
#    - `pvd`: True will write a paraview data file with simulation time for animations. Default is False
#    - `kper`: a list, tuple, or integer value of specific stess periods to output

# ### Export dis and upw package

# +
# package export
# set up package export folder
output_dir = workspace / "package_output_test"
output_dir.mkdir(exist_ok=True)

# export dis
dis_output_dir = output_dir / "DIS"
ml.dis.export(dis_output_dir, fmt="vtk")

# export upw with point scalars as a binary xml based vtk file
upw_output_dir = output_dir / "UPW"
ml.upw.export(upw_output_dir, fmt="vtk", point_scalars=True, xml=True)
# -

# ### Model export to .vtu files
#
# Model export has the following optional keyword arguments:
#
#    - `package_names`: a list of package names to export, default is None and will export all packages in the model.
#    - `smooth`: True creates a smooth surface, default is False
#    - `point_scalars`: True outputs point scalar values as well as cell values, default is False.
#    - `name`: A name can be specified to use for the output filename and array scalar name, by default the FloPy array name is used
#    - `binary`: argument that can be specified to switch between binary and ASCII, default is True
#    - `xml`: True will write an xml base vtk file, default is False
#    - `masked_values`: list or tuple of values to mask (set to nan) when writing a array
#    - `vertical_exageration`: floating point value that can be used to scale the vertical exageration of the vtk points. Default is 1.
#    - `pvd`: True will write a paraview data file with simulation time for animations. Default is False
#    - `kper`: a list, tuple, or integer value of specific stess periods to output

# ### Export model as a binary unstructured vtk file

model_output_dir = workspace / "model_output_test"
ml.export(model_output_dir, fmt="vtk")

# ## Using the `Vtk` class
#
# To export custom arrays, or choose a custom combination of model inputs to view, the user first needs to instantiate a new `Vtk()` object. The `Vtk()` object has a single required parameter and a number of optional parameters that the user can take advantage of. These parameters are as follows:
#
#    - `model`: any flopy model object can be supplied to create the vtk geometry. Either the model (recommended!) or modelgrid parameter must be supplied to the Vtk() object.
#    - `modelgrid`: any flopy modelgrid object (StructuredGrid, VertexGrid, UnstructuredGrid) can be supplied, in leiu of a model object, to create the vtk geometery.
#    - `vertical_exageration`: floating point value that can be used to scale the vertical exageration of the vtk points. Default is 1.
#    - `binary`: boolean flag to switch between binary and ASCII vtk files. Default is True.
#    - `xml`: boolean flag to write xml based vtk files. Default is False
#    - `pvd`: boolean flag to write a paraview data file for transient series of vtu files. This file relates model time to vtu file for animations. Default is False. If set to True Vtk() will automatically write xml based vtu files.
#    - `shared_points`: boolean flag to write shared vertices within the vtk file. Default is False.
#    - `smooth`: boolean flag to interpolate vertex elevations using IDW based on shared cell elevations. Default is False.
#    - `point_scalars`: boolean flag to write interpolated data at each point.

# create a binary XML VTK object and enable PVD file writing
vtkobj = vtk.Vtk(ml, xml=True, pvd=True, vertical_exageration=10)
vtkobj

# ### Adding array data to the `Vtk` object
#
# The `Vtk()` object has an `add_array()` method that lets the user add array data to the Field data section of the VTK file.
#
# `add_array()` has a few parameters for the user:
#    - `array` : numpy array that has a size equal to the number of cells in the model (modelgrid.nnodes).
#    - `name` : array name (string)
#    - `masked_values` : list of array values to mask/set to NaN

# +
# Create a vtk object
vtkobj = vtk.Vtk(ml, vertical_exageration=10)

## create some random array data
r_array = np.random.random(ml.modelgrid.nnodes) * 100

## add random data to the VTK object
vtkobj.add_array(r_array, "random_data")

## add the model botom data to the VTK object
vtkobj.add_array(ml.dis.botm.array, "botm")

## write the vtk object to file
vtkobj.write(output_dir / "Array_example" / "model.vtu")
# -

# ### Adding transient array data to the `Vtk` object
#
# The `Vtk` class has an `add_transient_array()` method that allows the user to create a series of time varying VTK files that can be used for animation in VTK viewers.
#
# The `add_transient_array()` method accepts a dictionary of array2d, array3d, or numpy array objects. Parameters include:
#    - `d`: dictionary of array2d, array3d, or numpy array objects
#    - `name`: parameter name, required when user provides a dictionary of numpy arrays
#    - `masked_values`: optional list of values to set equal to NaN.

# +
# create a vtk object
vtkobj = vtk.Vtk(ml, xml=True, pvd=True, vertical_exageration=10)

## add recharge to the VTK object
recharge = ml.rch.rech.transient_2ds
vtkobj.add_transient_array(
    recharge,
    "recharge",
    masked_values=[
        0,
    ],
)

## write vtk files
vtkobj.write(output_dir / "tr_array_example" / "recharge.vtu")
# -

# ### Adding transient list data to the `Vtk` object
#
# The `Vtk` class has an `add_transient_list()` method that allows the user to create a series of time varying VTK files that can be used for animation in VTK viewers.
#
# The `add_transient_list()` method accepts a FloPy mflist (transient list) type object. Parameters include:
#    - `mflist`: flopy transient list object
#    - `masked_values`: list of values to set equal to NaN

# +
# create the vtk object
vtkobj = vtk.Vtk(ml, xml=True, pvd=True, vertical_exageration=10)

## add well fluxes to the VTK object
spd = ml.wel.stress_period_data
vtkobj.add_transient_list(
    spd,
    masked_values=[
        0,
    ],
)

## write vtk files
vtkobj.write(output_dir / "tr_list_example" / "wel_flux.vtu")
# -

# ### Adding packages to the `Vtk` object
#
# The `Vtk` class has a method for adding package data to a VTK file as Field Data. The `add_package()` method allows the user to add packages for subsequent export. `add_package()` takes the following parameters:
#
#    - `pkg`: flopy package object
#    - `masked_values`: optional list of values to set to NaN.
#

# In the following example, a HFB package is added to the existing freyberg model and then exported with the WEL package.

# +
# create a HFB package for the example
hfb_data = []
for k in range(3):
    for i in range(20):
        rec = [k, i, 6, i, 7, 1e-06]
        hfb_data.append(rec)

hfb = flopy.modflow.ModflowHfb(ml, hfb_data=hfb_data)

# +
# export HFB and WEL packages using Vtk()
vtkobj = vtk.Vtk(ml, vertical_exageration=10)

vtkobj.add_package(hfb)
vtkobj.add_package(ml.wel)

vtkobj.write(output_dir / "package_example" / "package_export.vtu")
# -

# ### Exporting heads to binary .vtu files
#
# Once a `Vtk` object is instantiated (see above), the `add_heads()` method can be used to add head data. This method has a few parameters:
#   - `hds`: a flopy FormattedHeadFile or HeadFile object. This method also accepts DrawdownFile, and ConcentrationFile objects.
#   - `kstpkper`: optional list of zero based (timestep, stress period) tuples to output. Default is None and will output all data to a series of vtu files
#   - `masked_values`: optional list of values to set to NaN, default is None.
#

# +
# import the HeadFile reader and read in the head file
from flopy.utils import HeadFile

head_file = model_ws / "freyberg.hds"
hds = HeadFile(head_file)

# create the vtk object and export heads
vtkobj = vtk.Vtk(ml, xml=True, pvd=True, vertical_exageration=10)
vtkobj.add_heads(hds)
vtkobj.write(workspace / "heads_output_test" / "freyberg_head.vtu")
# -

# ### Export heads as point scalar arrays

# +
# export heads as point scalars
vtkobj = vtk.Vtk(
    ml, xml=True, pvd=True, point_scalars=True, vertical_exageration=10
)

# export heads for time step 1, stress periods 1, 50, 100, 1000
vtkobj.add_heads(hds, kstpkper=[(0, 0), (0, 49), (0, 99), (0, 999)])
vtkobj.write(workspace / "heads_output_test_parameters" / "freyberg_head.vtu")
# -

# ### Export cell budget information
#
# Once a `Vtk` object is instantiated (see above), the `add_cell_budget()` method can be used to export cell budget data. This method has a few parameters:
#    - `cbc`: flopy CellBudgetFile object
#    - `text`: Optional text identifier for a record type. Examples include 'RIVER LEAKAGE', 'STORAGE', etc... Default is None and will export all cell budget information to vtk files
#    - `kstpkper`: optional list of zero based (timestep, stress period) tuples to output. Default is None and will output all data to a series of vtu files
#    - `masked_values`: optional list of values to set to NaN, default is None.

# +
# import the CellBudgetFile reader and read the CBC file
from flopy.utils import CellBudgetFile

cbc_file = model_ws / "freyberg.cbc"
cbc = CellBudgetFile(cbc_file)

# export the cbc file to a series of Vtu files with a PVD file for animation
vtkobj = vtk.Vtk(ml, xml=True, pvd=True, vertical_exageration=10)
vtkobj.add_cell_budget(cbc, kstpkper=[(0, 0), (0, 9), (0, 10), (0, 11)])
vtkobj.write(workspace / "cbc_output_test_parameters" / "freyberg_cbc.vtu")
# -

# ### Export vectors from the cell budget file
#
# The `Vtk` class has an `add_vector()` method that allows the user to write vector information to VTK files. This method can be used to export information such as cell centered specific discharge.
#
# The `add_vector()` method accepts a numpy array of vector data. The array size must be 3 * the number of model cells (3 * modelgrid.nnodes). Parameters include:
#    - `vector`: numpy array of size 3 * nnodes
#    - `name`: name of the vector
#    - `masked_values`: list of values to set equal to NaN
#

# +
# get frf, fff, flf from the Cell Budget file (or SPDIS when using MF6)
from flopy.utils import postprocessing

frf = cbc.get_data(text="FLOW RIGHT FACE", kstpkper=(0, 9), full3D=True)[0]
fff = cbc.get_data(text="FLOW FRONT FACE", kstpkper=(0, 9), full3D=True)[0]
flf = cbc.get_data(text="FLOW LOWER FACE", kstpkper=(0, 9), full3D=True)[0]

spdis = postprocessing.get_specific_discharge((frf, fff, flf), ml)

# create the Vtk() object
vtkobj = vtk.Vtk(ml, vertical_exageration=10)

# add the vector
vtkobj.add_vector(spdis, name="spdis")

# write to file
vtkobj.write(output_dir / "vector_example" / "spdis_vector.vtu")
# -

# ### Exporting MODPATH timeseries or pathline data
#
# The `Vtk` class supports writing MODPATH pathline/timeseries data to a VTK file. To start the example, let's first load and run a MODPATH simulation (see flopy3_modpath7_unstructured_example for details) and then add the output to a `Vtk` object.

# +
# load and run the vertex grid model and modpath7
notebook_utils.run(workspace)

# check if model ran properly
modelpth = workspace / "mp7_ex2" / "mf6"
files = ["mp7p2.hds", "mp7p2.cbb"]
for f in files:
    if os.path.isfile(modelpth / f):
        msg = "Output file located: {}".format(f)
        print(msg)
    else:
        errmsg = "Error. Output file cannot be found: {}".format(f)
        print(errmsg)

# +
# load the simulation and get the model
vertex_sim_name = "mfsim.nam"
vertex_sim = flopy.mf6.MFSimulation.load(
    sim_name=vertex_sim_name, exe_name="mf6", sim_ws=modelpth
)
vertex_ml6 = vertex_sim.get_model("mp7p2")

# load the MODPATH-7 results
mp_namea = "mp7p2a_mp"
fpth = modelpth / f"{mp_namea}.mppth"
p = flopy.utils.PathlineFile(fpth)
p0 = p.get_alldata()
# -

# Create the `Vtk()` object and add all of the model data to it

vtkobj = vtk.Vtk(vertex_ml6, xml=True, pvd=True, vertical_exageration=10)
vtkobj.add_model(vertex_ml6)

# Add modpath data to the `Vtk()` object.
#
# *Note: this will create a second vtk file that has the file signature: `myfilename)_pathline.vtu*`

vtkobj.add_pathline_points(p0, timeseries=False)
vtkobj.write(output_dir / "mp7_vertex_example" / "vertex_ex.vtu")

# Clean up the temporary workspace.

try:
    # ignore PermissionError on Windows
    tempdir.cleanup()
except:
    pass
