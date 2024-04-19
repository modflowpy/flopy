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
#     section: nwt
#     authors:
#       - name: Joshua Larsen
# ---

# # Working with MODFLOW-NWT v 1.1 option blocks
#
# In MODFLOW-NWT an option block is present for the WEL file, UZF file, and SFR file. This block takes keyword arguments that are supplied in an option line in other versions of MODFLOW.
#
# The `OptionBlock` class was created to provide combatibility with the MODFLOW-NWT option block and allow the user to easily edit values within the option block

# +
import os
import sys
from tempfile import TemporaryDirectory

import flopy
from flopy.utils import OptionBlock

print(sys.version)
print(f"flopy version: {flopy.__version__}")

# +
load_ws = os.path.join("..", "..", "examples", "data", "options", "sagehen")

# temporary directory
temp_dir = TemporaryDirectory()
model_ws = os.path.join(temp_dir.name, "nwt_options", "output")
# -

# ## Loading a MODFLOW-NWT model that has option block options
#
# It is critical to set the `version` flag in `flopy.modflow.Modflow.load()` to `version='mfnwt'`
#
# We are going to load a modified version of the Sagehen test problem from GSFLOW to illustrate compatibility

# +
mfexe = "mfnwt"

ml = flopy.modflow.Modflow.load(
    "sagehen.nam", model_ws=load_ws, exe_name=mfexe, version="mfnwt"
)
ml.change_model_ws(new_pth=model_ws)
ml.write_input()
# -

# ### Let's look at the options attribute of the UZF object
#
# The `uzf.options` attribute is an `OptionBlock` object. The representation of this object is the option block that will be written to output, which allows the user to easily check to make sure the block has the options they want.

uzf = ml.get_package("UZF")
uzf.options

# The `OptionBlock` object also has attributes which correspond to the option names listed in the online guide to modflow
#
# The user can call and edit the options within the option block

print(uzf.options.nosurfleak)
print(uzf.options.savefinf)

uzf.options.etsquare = False
uzf.options

uzf.options.etsquare = True
uzf.options

# ### The user can also see the single line representation of the options

uzf.options.single_line_options

# ### And the user can easily change to single line options writing

# +
uzf.options.block = False

# write out only the uzf file
uzf_name = "uzf_opt.uzf"
uzf.write_file(os.path.join(model_ws, uzf_name))
# -

# Now let's examine the first few lines of the new UZF file

f = open(os.path.join(model_ws, uzf_name))
for ix, line in enumerate(f):
    if ix == 3:
        break
    else:
        print(line)

# And let's load the new UZF file

uzf2 = flopy.modflow.ModflowUzf1.load(
    os.path.join(model_ws, uzf_name), ml, check=False
)

# ### Now we can look at the options object, and check if it's block or line format
#
# `block=False` indicates that options will be written as line format

print(uzf2.options)
print(uzf2.options.block)

# ### Finally we can convert back to block format

# +
uzf2.options.block = True
uzf2.write_file(os.path.join(model_ws, uzf_name))
ml.remove_package("UZF")

uzf3 = flopy.modflow.ModflowUzf1.load(
    os.path.join(model_ws, uzf_name), ml, check=False
)
print("\n")
print(uzf3.options)
print(uzf3.options.block)
# -

# ### We can also look at the WEL object

wel = ml.get_package("WEL")
wel.options

# Let's write this out as a single line option block and examine the first few lines

# +
wel_name = "wel_opt.wel"
wel.options.block = False

wel.write_file(os.path.join(model_ws, wel_name))


f = open(os.path.join(model_ws, wel_name))
for ix, line in enumerate(f):
    if ix == 4:
        break
    else:
        print(line)
# -

# And we can load the new single line options WEL file and confirm that it is being read as an option line

# +
ml.remove_package("WEL")
wel2 = flopy.modflow.ModflowWel.load(
    os.path.join(model_ws, wel_name), ml, nper=ml.nper, check=False
)

wel2.options
wel2.options.block
# -

# ## Building an OptionBlock from scratch
#
# The user can also build an `OptionBlock` object from scratch to add to a `ModflowSfr2`, `ModflowUzf1`, or `ModflowWel` file.
#
# The `OptionBlock` class has two required parameters and one optional parameter
#
# `option_line`: a one line, string based representation of the options
#
# `package`: a modflow package object
#
# `block`: boolean flag for line based or block based options

opt_line = "specify 0.1 20"
options = OptionBlock(opt_line, flopy.modflow.ModflowWel, block=True)
options

# from here we can set the noprint flag by using `options.noprint`

options.noprint = True

# and the user can also add auxiliary variables by using `options.auxiliary`

options.auxiliary = ["aux", "iface"]

# ### Now we can create a new wel file using this `OptionBlock`
#
# and write it to output

# +
wel3 = flopy.modflow.ModflowWel(
    ml,
    stress_period_data=wel.stress_period_data,
    options=options,
    unitnumber=99,
)

wel3.write_file(os.path.join(model_ws, wel_name))
# -

# And now let's examine the first few lines of the file

f = open(os.path.join(model_ws, wel_name))
for ix, line in enumerate(f):
    if ix == 8:
        break
    else:
        print(line)

# We can see that everything that the OptionBlock class writes out options in the correct location.

# ### The user can also switch the options over to option line style and write out the output too!

# +
wel3.options.block = False
wel3.write_file(os.path.join(model_ws, wel_name))

f = open(os.path.join(model_ws, wel_name))
for ix, line in enumerate(f):
    if ix == 6:
        break
    else:
        print(line)
# -

try:
    # ignore PermissionError on Windows
    temp_dir.cleanup()
except:
    pass
