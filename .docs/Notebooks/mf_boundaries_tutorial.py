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
#     section: mf2005
# ---

# # MODFLOW-2005 Basic stress packages
#
# Flopy has a new way to enter boundary conditions for some MODFLOW packages.  These changes are substantial.  Boundary conditions can now be entered as a list of boundaries, as a numpy recarray, or as a dictionary.  These different styles are described in this notebook.
#
# Flopy also now requires zero-based input.  This means that **all boundaries are entered in zero-based layer, row, and column indices**.  This means that older Flopy scripts will need to be modified to account for this change.  If you are familiar with Python, this should be natural, but if not, then it may take some time to get used to zero-based numbering.  Flopy users submit all information in zero-based form, and Flopy converts this to the one-based form required by MODFLOW.
#
# The following MODFLOW-2005 packages are affected by this change:
#
#   * Well
#   * Drain
#   * River
#   * General-Head Boundary
#   * Time-Variant Constant Head
#
# This notebook explains the different ways to enter these types of boundary conditions.
#

# +
# begin by importing flopy
import os
import sys
from tempfile import TemporaryDirectory

import numpy as np

# run installed version of flopy or add local path
try:
    import flopy
except:
    fpth = os.path.abspath(os.path.join("..", ".."))
    sys.path.append(fpth)
    import flopy

# temporary directory
temp_dir = TemporaryDirectory()
workspace = os.path.join(temp_dir.name)

print(sys.version)
print(f"numpy version: {np.__version__}")
print(f"flopy version: {flopy.__version__}")
# -

# ## List of Boundaries

# Boundary condition information is passed to a package constructor as stress_period_data.  In its simplest form, stress_period_data can be a list of individual boundaries, which themselves are lists.  The following shows a simple example for a MODFLOW River Package boundary:

stress_period_data = [
    [
        2,
        3,
        4,
        10.7,
        5000.0,
        -5.7,
    ],  # layer, row, column, stage, conductance, river bottom
    [
        2,
        3,
        5,
        10.7,
        5000.0,
        -5.7,
    ],  # layer, row, column, stage, conductance, river bottom
    [
        2,
        3,
        6,
        10.7,
        5000.0,
        -5.7,
    ],  # layer, row, column, stage, conductance, river bottom
]
m = flopy.modflow.Modflow(modelname="test", model_ws=workspace)
riv = flopy.modflow.ModflowRiv(m, stress_period_data=stress_period_data)
m.write_input()

# If we look at the River Package created here, you see that the layer, row, and column numbers have been increased by one.

# !head -n 10 '../../examples/data/test.riv'

# If this model had more than one stress period, then Flopy will assume that this boundary condition information applies until the end of the simulation

m = flopy.modflow.Modflow(modelname="test", model_ws=workspace)
dis = flopy.modflow.ModflowDis(m, nper=3)
riv = flopy.modflow.ModflowRiv(m, stress_period_data=stress_period_data)
m.write_input()
# !head -n 10 '../../examples/data/test.riv'

# ## Recarray of Boundaries
#
# Numpy allows the use of recarrays, which are numpy arrays in which each column of the array may be given a different type.  Boundary conditions can be entered as recarrays.  Information on the structure of the recarray for a boundary condition package can be obtained from that particular package.  The structure of the recarray is contained in the dtype.

riv_dtype = flopy.modflow.ModflowRiv.get_default_dtype()
print(riv_dtype)

# Now that we know the structure of the recarray that we want to create, we can create a new one as follows.

stress_period_data = np.zeros((3), dtype=riv_dtype)
stress_period_data = stress_period_data.view(np.recarray)
print("stress_period_data: ", stress_period_data)
print("type is: ", type(stress_period_data))

# We can then fill the recarray with our boundary conditions.

stress_period_data[0] = (2, 3, 4, 10.7, 5000.0, -5.7)
stress_period_data[1] = (2, 3, 5, 10.7, 5000.0, -5.7)
stress_period_data[2] = (2, 3, 6, 10.7, 5000.0, -5.7)
print(stress_period_data)

m = flopy.modflow.Modflow(modelname="test", model_ws=workspace)
riv = flopy.modflow.ModflowRiv(m, stress_period_data=stress_period_data)
m.write_input()
# !head -n 10 '../../examples/data/test.riv'

# As before, if we have multiple stress periods, then this recarray will apply to all of them.

m = flopy.modflow.Modflow(modelname="test", model_ws=workspace)
dis = flopy.modflow.ModflowDis(m, nper=3)
riv = flopy.modflow.ModflowRiv(m, stress_period_data=stress_period_data)
m.write_input()
# !head -n 10 '../../examples/data/test.riv'

# ## Dictionary of Boundaries
#
# The power of the new functionality in Flopy3 is the ability to specify a dictionary for stress_period_data.  If specified as a dictionary, the key is the stress period number (**as a zero-based number**), and the value is either a nested list, an integer value of 0 or -1, or a recarray for that stress period.
#
# Let's say that we want to use the following schedule for our rivers:
#   0. No rivers in stress period zero
#   1. Rivers specified by a list in stress period 1
#   2. No rivers
#   3. No rivers
#   4. No rivers
#   5. Rivers specified by a recarray
#   6. Same recarray rivers
#   7. Same recarray rivers
#   8. Same recarray rivers
#

sp1 = [
    [
        2,
        3,
        4,
        10.7,
        5000.0,
        -5.7,
    ],  # layer, row, column, stage, conductance, river bottom
    [
        2,
        3,
        5,
        10.7,
        5000.0,
        -5.7,
    ],  # layer, row, column, stage, conductance, river bottom
    [
        2,
        3,
        6,
        10.7,
        5000.0,
        -5.7,
    ],  # layer, row, column, stage, conductance, river bottom
]
print(sp1)

riv_dtype = flopy.modflow.ModflowRiv.get_default_dtype()
sp5 = np.zeros((3), dtype=riv_dtype)
sp5 = sp5.view(np.recarray)
sp5[0] = (2, 3, 4, 20.7, 5000.0, -5.7)
sp5[1] = (2, 3, 5, 20.7, 5000.0, -5.7)
sp5[2] = (2, 3, 6, 20.7, 5000.0, -5.7)
print(sp5)

sp_dict = {0: 0, 1: sp1, 2: 0, 5: sp5}
m = flopy.modflow.Modflow(modelname="test", model_ws=workspace)
dis = flopy.modflow.ModflowDis(m, nper=8)
riv = flopy.modflow.ModflowRiv(m, stress_period_data=sp_dict)
m.write_input()
# !head -n 10 '../../examples/data/test.riv'

# ## MODFLOW Auxiliary Variables
#
# Flopy works with MODFLOW auxiliary variables by allowing the recarray to contain additional columns of information.  The auxiliary variables must be specified as package options as shown in the example below.
#
# In this example, we also add a string in the last column of the list in order to name each boundary condition.  In this case, however, we do not include boundname as an auxiliary variable as MODFLOW would try to read it as a floating point number.

# create an empty array with an iface auxiliary variable at the end
riva_dtype = [
    ("k", "<i8"),
    ("i", "<i8"),
    ("j", "<i8"),
    ("stage", "<f4"),
    ("cond", "<f4"),
    ("rbot", "<f4"),
    ("iface", "<i4"),
    ("boundname", object),
]
riva_dtype = np.dtype(riva_dtype)
stress_period_data = np.zeros((3), dtype=riva_dtype)
stress_period_data = stress_period_data.view(np.recarray)
print("stress_period_data: ", stress_period_data)
print("type is: ", type(stress_period_data))

stress_period_data[0] = (2, 3, 4, 10.7, 5000.0, -5.7, 1, "riv1")
stress_period_data[1] = (2, 3, 5, 10.7, 5000.0, -5.7, 2, "riv2")
stress_period_data[2] = (2, 3, 6, 10.7, 5000.0, -5.7, 3, "riv3")
print(stress_period_data)

m = flopy.modflow.Modflow(modelname="test", model_ws=workspace)
riv = flopy.modflow.ModflowRiv(
    m,
    stress_period_data=stress_period_data,
    dtype=riva_dtype,
    options=["aux iface"],
)
m.write_input()
# !head -n 10 '../../examples/data/test.riv'

# ## Working with Unstructured Grids
#
# Flopy can create an unstructured grid boundary condition package for MODFLOW-USG.  This can be done by specifying a custom dtype for the recarray.  The following shows an example of how that can be done.

# create an empty array based on nodenumber instead of layer, row, and column
rivu_dtype = [
    ("nodenumber", "<i8"),
    ("stage", "<f4"),
    ("cond", "<f4"),
    ("rbot", "<f4"),
]
rivu_dtype = np.dtype(rivu_dtype)
stress_period_data = np.zeros((3), dtype=rivu_dtype)
stress_period_data = stress_period_data.view(np.recarray)
print("stress_period_data: ", stress_period_data)
print("type is: ", type(stress_period_data))

stress_period_data[0] = (77, 10.7, 5000.0, -5.7)
stress_period_data[1] = (245, 10.7, 5000.0, -5.7)
stress_period_data[2] = (450034, 10.7, 5000.0, -5.7)
print(stress_period_data)

m = flopy.modflow.Modflow(modelname="test", model_ws=workspace)
riv = flopy.modflow.ModflowRiv(
    m, stress_period_data=stress_period_data, dtype=rivu_dtype
)
m.write_input()
# !head -n 10 '../../examples/data/test.riv'

# ## Combining two boundary condition packages

ml = flopy.modflow.Modflow(modelname="test", model_ws=workspace)
dis = flopy.modflow.ModflowDis(ml, 10, 10, 10, 10)
sp_data1 = {3: [1, 1, 1, 1.0], 5: [1, 2, 4, 4.0]}
wel1 = flopy.modflow.ModflowWel(ml, stress_period_data=sp_data1)
ml.write_input()
# !head -n 10 '../../examples/data/test.wel'

sp_data2 = {0: [1, 1, 3, 3.0], 8: [9, 2, 4, 4.0]}
wel2 = flopy.modflow.ModflowWel(ml, stress_period_data=sp_data2)
ml.write_input()
# !head -n 10 '../../examples/data/test.wel'

# Now we create a third wel package, using the ```MfList.append()``` method:

# + pycharm={"name": "#%%\n"}
wel3 = flopy.modflow.ModflowWel(
    ml,
    stress_period_data=wel2.stress_period_data.append(wel1.stress_period_data),
)
ml.write_input()
# !head -n 10 '../../examples/data/test.wel'
