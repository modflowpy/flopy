# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   metadata:
#     section: pest
# ---

# # Support for PEST
# This notebook demonstrates the current parameter estimation functionality that is available with FloPy.  The capability to write a simple template file for PEST is the only capability implemented so far.  The plan is to develop functionality for creating PEST instruction files as well as the PEST control file.

# +
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

print(sys.version)
print("numpy version: {}".format(np.__version__))
print("flopy version: {}".format(flopy.__version__))
# -

# This notebook will work with a simple model using the dimensions below

# +
# temporary directory
temp_dir = TemporaryDirectory()
workspace = temp_dir.name

# Define the model dimensions
nlay = 3
nrow = 20
ncol = 20

# Create the flopy model object and add the dis and lpf packages
m = flopy.modflow.Modflow(modelname="mymodel", model_ws=workspace)
dis = flopy.modflow.ModflowDis(m, nlay, nrow, ncol)
lpf = flopy.modflow.ModflowLpf(m, hk=10.0)
# -

# ### Simple One Parameter Example
# In order to create a PEST template file, we first need to define a parameter.  For example, let's say we want to parameterize hydraulic conductivity, which is a static variable in flopy and MODFLOW.  As a first step, let's define a parameter called HK_LAYER_1 and assign it to all of layer 1.  We will not parameterize hydraulic conductivity for layers 2 and 3 and instead leave HK at its value of 10. (as assigned in the block above this one). We can do this as follows.

# +
mfpackage = "lpf"
partype = "hk"
parname = "HK_LAYER_1"
idx = np.empty((nlay, nrow, ncol), dtype=bool)
idx[0] = True
idx[1:] = False

# The span variable defines how the parameter spans the package
span = {"idx": idx}

# These parameters have not affect yet, but may in the future
startvalue = 10.0
lbound = 0.001
ubound = 1000.0
transform = "log"

p = flopy.pest.Params(
    mfpackage, partype, parname, startvalue, lbound, ubound, span
)
# -

# At this point, we have enough information to the write a PEST template file for the LPF package.  We can do this using the following statement:

tw = flopy.pest.TemplateWriter(m, [p])
tw.write_template()

# At this point, the lpf template file will have been created.  The following block will print the template file.

lines = open(os.path.join(workspace, "mymodel.lpf.tpl"), "r").readlines()
for l in lines:
    print(l.strip())

# The span variable will also accept 'layers', in which the parameter applies to the list of layers, as shown next.  When 'layers' is specified in the span dictionary, then the original hk value of 10. remains in the array, and the multiplier is specified on the array control line.

# +
mfpackage = "lpf"
partype = "hk"
parname = "HK_LAYER_1-3"

# Span indicates that the hk parameter applies as a multiplier to layers 0 and 2 (MODFLOW layers 1 and 3)
span = {"layers": [0, 2]}

# These parameters have not affect yet, but may in the future
startvalue = 10.0
lbound = 0.001
ubound = 1000.0
transform = "log"

p = flopy.pest.Params(
    mfpackage, partype, parname, startvalue, lbound, ubound, span
)
tw = flopy.pest.templatewriter.TemplateWriter(m, [p])
tw.write_template()
# -

lines = open(os.path.join(workspace, "mymodel.lpf.tpl"), "r").readlines()
for l in lines:
    print(l.strip())

# ### Multiple Parameter Zoned Approach
#
# The params module has a helper function called zonearray2params that will take a zone array and some other information and create a list of parameters, which can then be passed to the template writer.  This next example shows how to create a slightly more complicated LPF template file in which both HK and VKA are parameterized.

# Create a zone array
zonearray = np.ones((nlay, nrow, ncol), dtype=int)
zonearray[0, 10:, 7:] = 2
zonearray[0, 15:, 9:] = 3
zonearray[1] = 4

# Create a list of parameters for HK
mfpackage = "lpf"
parzones = [2, 3, 4]
parvals = [56.777, 78.999, 99.0]
lbound = 5
ubound = 500
transform = "log"
plisthk = flopy.pest.zonearray2params(
    mfpackage, "hk", parzones, lbound, ubound, parvals, transform, zonearray
)

# In this case, Flopy will create three parameters: hk_2, hk_3, and hk_4, which will apply to the horizontal hydraulic conductivity for cells in zones 2, 3, and 4, respectively.  Only those zone numbers listed in parzones will be parameterized.  For example, many cells in zonearray have a value of 1.  Those cells will not be parameterized.  Instead, their hydraulic conductivity values will remain fixed at the value that was specified when the Flopy LPF package was created.

# Create a list of parameters for VKA
parzones = [1, 2]
parvals = [0.001, 0.0005]
zonearray = np.ones((nlay, nrow, ncol), dtype=int)
zonearray[1] = 2
plistvk = flopy.pest.zonearray2params(
    mfpackage, "vka", parzones, lbound, ubound, parvals, transform, zonearray
)

# Combine the HK and VKA parameters together
plist = plisthk + plistvk
for p in plist:
    print(p.name, p.mfpackage, p.startvalue)

# Write the template file
tw = flopy.pest.templatewriter.TemplateWriter(m, plist)
tw.write_template()

# Print contents of template file
lines = open(os.path.join(workspace, "mymodel.lpf.tpl"), "r").readlines()
for l in lines:
    print(l.strip())

# ## Two-Dimensional Transient Arrays
#
# Flopy supports parameterization of transient two dimensional arrays, like recharge.  This is similar to the approach for three dimensional static arrays, but there are some important differences in how span is specified.  The parameter span here is also a dictionary, and it must contain a 'kper' key, which corresponds to a list of stress periods (zero based, of course) for which the parameter applies.  The span dictionary must also contain an 'idx' key.  If span['idx'] is None, then the parameter is a multiplier for those stress periods.  If span['idx'] is a tuple (iarray, jarray), where iarray and jarray are a list of array indices, or a boolean array of shape (nrow, ncol), then the parameter applies only to the cells specified in idx.

# +
# Define the model dimensions (made smaller for easier viewing)
nlay = 3
nrow = 5
ncol = 5
nper = 3

# Create the flopy model object and add the dis and lpf packages
m = flopy.modflow.Modflow(modelname="mymodel", model_ws=workspace)
dis = flopy.modflow.ModflowDis(m, nlay, nrow, ncol, nper=nper)
lpf = flopy.modflow.ModflowLpf(m, hk=10.0)
rch = flopy.modflow.ModflowRch(m, rech={0: 0.001, 2: 0.003})
# -

# Next, we create the parameters

plist = []

# +
# Create a multiplier parameter for recharge
mfpackage = "rch"
partype = "rech"
parname = "RECH_MULT"
startvalue = None
lbound = None
ubound = None
transform = None

# For a recharge multiplier, span['idx'] must be None
idx = None
span = {"kpers": [0, 1, 2], "idx": idx}
p = flopy.pest.Params(
    mfpackage, partype, parname, startvalue, lbound, ubound, span
)
plist.append(p)
# -

# Write the template file
tw = flopy.pest.TemplateWriter(m, plist)
tw.write_template()

# Print the results
lines = open(os.path.join(workspace, "mymodel.rch.tpl"), "r").readlines()
for l in lines:
    print(l.strip())

# Multiplier parameters can also be combined with index parameters as follows.

# +
plist = []

# Create a multiplier parameter for recharge
mfpackage = "rch"
partype = "rech"
parname = "RECH_MULT"
startvalue = None
lbound = None
ubound = None
transform = None

# For a recharge multiplier, span['idx'] must be None
span = {"kpers": [1, 2], "idx": None}
p = flopy.pest.Params(
    mfpackage, partype, parname, startvalue, lbound, ubound, span
)
plist.append(p)

# +
# Now create an index parameter
mfpackage = "rch"
partype = "rech"
parname = "RECH_ZONE"
startvalue = None
lbound = None
ubound = None
transform = None

# For a recharge index parameter, span['idx'] must be a boolean array or tuple of array indices
idx = np.empty((nrow, ncol), dtype=bool)
idx[0:3, 0:3] = True
span = {"kpers": [1], "idx": idx}
p = flopy.pest.Params(
    mfpackage, partype, parname, startvalue, lbound, ubound, span
)
plist.append(p)

# +
# Write the template file
tw = flopy.pest.templatewriter.TemplateWriter(m, plist)
tw.write_template()

# Print the results
lines = open(os.path.join(workspace, "mymodel.rch.tpl"), "r").readlines()
for l in lines:
    print(l.strip())
# -

try:
    # ignore PermissionError on Windows
    temp_dir.cleanup()
except:
    pass
