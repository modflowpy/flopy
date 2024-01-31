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
#       - name: Jeremy White
# ---

# # External file handling
#

# +
import os
import sys
from tempfile import TemporaryDirectory

import numpy as np

import flopy
from flopy.utils import flopy_io

print(sys.version)
print(f"numpy version: {np.__version__}")
print(f"flopy version: {flopy.__version__}")

# +
# make a model
nlay, nrow, ncol = 10, 20, 5
temp_dir = TemporaryDirectory()
model_ws = temp_dir.name

# the place for all of your hand made and costly model inputs
array_dir = os.path.join(temp_dir.name, "array_dir")
os.mkdir(array_dir)

ml = flopy.modflow.Modflow(model_ws=model_ws)
dis = flopy.modflow.ModflowDis(
    ml, nlay=nlay, nrow=nrow, ncol=ncol, steady=False, nper=2
)
# -

# make an ``hk`` and ```vka``` array.  We'll save ```hk``` to files - pretent that you spent months making this important model property.  Then make an ```lpf```

hk = np.zeros((nlay, nrow, ncol)) + 5.0
vka = np.zeros_like(hk)
fnames = []
for i, h in enumerate(hk):
    fname = os.path.join(array_dir, f"hk_{i + 1}.ref")
    fnames.append(fname)
    np.savetxt(fname, h)
    vka[i] = i + 1
lpf = flopy.modflow.ModflowLpf(ml, hk=fnames, vka=vka)

# Let's also have some recharge with mixed args as well.  Pretend the recharge in the second stress period is very important and precise

warmup_recharge = np.ones((nrow, ncol))
important_recharge = np.random.random((nrow, ncol))
fname = os.path.join(array_dir, "important_recharge.ref")
np.savetxt(fname, important_recharge)
rch = flopy.modflow.ModflowRch(ml, rech={0: warmup_recharge, 1: fname})

ml.write_input()

# Let's look at the files that were created

# +
from pprint import pprint

print("model_ws:", flopy_io.scrub_login(ml.model_ws))
pprint([flopy_io.scrub_login(p) for p in os.listdir(ml.model_ws)])
# -

# We see that a copy of the ``hk`` files as well as the important recharge file were made in the ```model_ws```.Let's looks at the ```lpf``` file

open(os.path.join(ml.model_ws, f"{ml.name}.lpf")).readlines()[:20]

# We see that the ```open/close``` approach was used - this is because ``ml.array_free_format`` is ``True``.  Notice that ```vka``` is written internally

ml.array_free_format

# Now change ```model_ws```

ml.model_ws = os.path.join(model_ws, "new_external_demo_dir")

# Now when we call ``write_input()``, a copy of external files are made in the current ```model_ws```

ml.write_input()

# list the files in model_ws that have 'hk' in the name
print(
    "\n".join(
        [
            name
            for name in os.listdir(ml.model_ws)
            if "hk" in name or "impor" in name
        ]
    )
)

# Now we see that the external files were copied to the new ```model_ws```
#
# ### Using ```external_path```
#
# It is sometimes useful when first building a model to write the model arrays as external files for processing and parameter estimation.  The ```model``` attribute ```external_path``` triggers this behavior

# +
# make a model - same code as before except for the model constructor
nlay, nrow, ncol = 10, 20, 5
model_ws = os.path.join(model_ws, "external_demo")
os.mkdir(model_ws)

# the place for all of your hand made and costly model inputs
array_dir = os.path.join(model_ws, "array_dir")
os.mkdir(array_dir)

# lets make an external path relative to the model_ws
ml = flopy.modflow.Modflow(
    model_ws=model_ws, external_path=os.path.join(model_ws, "ref")
)
dis = flopy.modflow.ModflowDis(
    ml, nlay=nlay, nrow=nrow, ncol=ncol, steady=False, nper=2
)

hk = np.zeros((nlay, nrow, ncol)) + 5.0
vka = np.zeros_like(hk)
fnames = []
for i, h in enumerate(hk):
    fname = os.path.join(array_dir, f"hk_{i + 1}.ref")
    fnames.append(fname)
    np.savetxt(fname, h)
    vka[i] = i + 1
lpf = flopy.modflow.ModflowLpf(ml, hk=fnames, vka=vka)

warmup_recharge = np.ones((nrow, ncol))
important_recharge = np.random.random((nrow, ncol))
fname = os.path.join(array_dir, "important_recharge.ref")
np.savetxt(fname, important_recharge)
rch = flopy.modflow.ModflowRch(ml, rech={0: warmup_recharge, 1: fname})
# -

# We can see that the model constructor created both ```model_ws``` and ```external_path``` which is _relative to the model_ws_

os.listdir(ml.model_ws)

# Now, when we call ```write_input()```, any array properties that were specified as ```np.ndarray``` will be written externally.  If a scalar was passed as the argument, the value remains internal to the model input files

ml.write_input()
# open(os.path.join(ml.model_ws, ml.name + ".lpf"), "r").readlines()[:20]

# Now, ```vka``` was also written externally, but not the storage properties.Let's verify the contents of the external path directory. We see our hard-fought ```hk``` and ```important_recharge``` arrays, as well as the ``vka`` arrays.

ml.lpf.ss.how = "internal"
ml.write_input()
# open(os.path.join(ml.model_ws, ml.name + ".lpf"), "r").readlines()[:20]

print("\n".join(os.listdir(os.path.join(ml.model_ws, ml.external_path))))

# ### Fixed format
#
# All of this behavior also works for fixed-format type models (really, really old models - I mean OLD!)

# +
# make a model - same code as before except for the model constructor
nlay, nrow, ncol = 10, 20, 5

# lets make an external path relative to the model_ws
ml = flopy.modflow.Modflow(model_ws=model_ws, external_path="ref")

# explicitly reset the free_format flag BEFORE ANY PACKAGES ARE MADE!!!
ml.array_free_format = False

dis = flopy.modflow.ModflowDis(
    ml, nlay=nlay, nrow=nrow, ncol=ncol, steady=False, nper=2
)

hk = np.zeros((nlay, nrow, ncol)) + 5.0
vka = np.zeros_like(hk)
fnames = []
for i, h in enumerate(hk):
    fname = os.path.join(array_dir, f"hk_{i + 1}.ref")
    fnames.append(fname)
    np.savetxt(fname, h)
    vka[i] = i + 1
lpf = flopy.modflow.ModflowLpf(ml, hk=fnames, vka=vka)
ml.lpf.ss.how = "internal"
warmup_recharge = np.ones((nrow, ncol))
important_recharge = np.random.random((nrow, ncol))
fname = os.path.join(array_dir, "important_recharge.ref")
np.savetxt(fname, important_recharge)
rch = flopy.modflow.ModflowRch(ml, rech={0: warmup_recharge, 1: fname})

ml.write_input()
# -

# We see that now the external arrays are being handled through the name file.  Let's look at the name file

open(os.path.join(ml.model_ws, f"{ml.name}.nam")).readlines()

# ### Free and binary format

ml.dis.botm[0].format.binary = True
ml.write_input()

open(os.path.join(ml.model_ws, f"{ml.name}.nam")).readlines()

open(os.path.join(ml.model_ws, f"{ml.name}.dis")).readlines()

# ### The ```.how``` attribute
# ```Util2d``` includes a ```.how``` attribute that gives finer grained control of how arrays will written

ml.lpf.hk[0].how

# This will raise an error since our model does not support free format...

try:
    ml.lpf.hk[0].how = "openclose"
    ml.lpf.hk[0].how
    ml.write_input()
except Exception as e:
    print("\n", e, "\n")

# So let's reset hk layer 1 back to external...

ml.lpf.hk[0].how = "external"
ml.lpf.hk[0].how

ml.dis.top.how = "external"

ml.write_input()

open(os.path.join(ml.model_ws, f"{ml.name}.dis")).readlines()

open(os.path.join(ml.model_ws, f"{ml.name}.lpf")).readlines()

open(os.path.join(ml.model_ws, f"{ml.name}.nam")).readlines()

try:
    # ignore PermissionError on Windows
    temp_dir.cleanup()
except:
    pass
