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
#     display_name: Python 3
#     language: python
#     name: python3
#   metadata:
#     section: mt3dms
#     authors:
#       - name: Jeremy White
# ---

# # Using FloPy to simplify the use of the MT3DMS ```SSM``` package
#
# A multi-component transport demonstration

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
print(f"numpy version: {np.__version__}")
print(f"flopy version: {flopy.__version__}")
# -

# First, we will create a simple model structure

# +
nlay, nrow, ncol = 10, 10, 10
perlen = np.zeros((10), dtype=float) + 10
nper = len(perlen)

ibound = np.ones((nlay, nrow, ncol), dtype=int)

botm = np.arange(-1, -11, -1)
top = 0.0
# -

# ## Create the ```MODFLOW``` packages

# +
# temporary directory
temp_dir = TemporaryDirectory()
model_ws = temp_dir.name

modelname = "ssmex"
mf = flopy.modflow.Modflow(modelname, model_ws=model_ws)
dis = flopy.modflow.ModflowDis(
    mf,
    nlay=nlay,
    nrow=nrow,
    ncol=ncol,
    perlen=perlen,
    nper=nper,
    botm=botm,
    top=top,
    steady=False,
)
bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=top)
lpf = flopy.modflow.ModflowLpf(mf, hk=100, vka=100, ss=0.00001, sy=0.1)
oc = flopy.modflow.ModflowOc(mf)
pcg = flopy.modflow.ModflowPcg(mf)
rch = flopy.modflow.ModflowRch(mf)
# -

# We'll track the cell locations for the ```SSM``` data using the ```MODFLOW``` boundary conditions.
#
#
# Get a dictionary (```dict```) that has the ```SSM``` ```itype``` for each of the boundary types.

itype = flopy.mt3d.Mt3dSsm.itype_dict()
print(itype)
print(flopy.mt3d.Mt3dSsm.get_default_dtype())
ssm_data = {}

# Add a general head boundary (```ghb```).  The general head boundary head (```bhead```) is 0.1 for the first 5 stress periods with a component 1 (comp_1) concentration of 1.0 and a component 2 (comp_2) concentration of 100.0.  Then ```bhead``` is increased to 0.25 and comp_1 concentration is reduced to 0.5 and comp_2 concentration is increased to 200.0

# +
ghb_data = {}
print(flopy.modflow.ModflowGhb.get_default_dtype())
ghb_data[0] = [(4, 4, 4, 0.1, 1.5)]
ssm_data[0] = [(4, 4, 4, 1.0, itype["GHB"], 1.0, 100.0)]
ghb_data[5] = [(4, 4, 4, 0.25, 1.5)]
ssm_data[5] = [(4, 4, 4, 0.5, itype["GHB"], 0.5, 200.0)]

for k in range(nlay):
    for i in range(nrow):
        ghb_data[0].append((k, i, 0, 0.0, 100.0))
        ssm_data[0].append((k, i, 0, 0.0, itype["GHB"], 0.0, 0.0))

ghb_data[5] = [(4, 4, 4, 0.25, 1.5)]
ssm_data[5] = [(4, 4, 4, 0.5, itype["GHB"], 0.5, 200.0)]
for k in range(nlay):
    for i in range(nrow):
        ghb_data[5].append((k, i, 0, -0.5, 100.0))
        ssm_data[5].append((k, i, 0, 0.0, itype["GHB"], 0.0, 0.0))
# -

# Add an injection ```well```.  The injection rate (```flux```) is 10.0 with a comp_1 concentration of 10.0 and a comp_2 concentration of 0.0 for all stress periods.  WARNING: since we changed the ```SSM``` data in stress period 6, we need to add the well to the ssm_data for stress period 6.

wel_data = {}
print(flopy.modflow.ModflowWel.get_default_dtype())
wel_data[0] = [(0, 4, 8, 10.0)]
ssm_data[0].append((0, 4, 8, 10.0, itype["WEL"], 10.0, 0.0))
ssm_data[5].append((0, 4, 8, 10.0, itype["WEL"], 10.0, 0.0))

# Add the ```GHB``` and ```WEL``` packages to the ```mf``` ```MODFLOW``` object instance.

ghb = flopy.modflow.ModflowGhb(mf, stress_period_data=ghb_data)
wel = flopy.modflow.ModflowWel(mf, stress_period_data=wel_data)

# ## Create the ```MT3DMS``` packages

mt = flopy.mt3d.Mt3dms(modflowmodel=mf, modelname=modelname, model_ws=model_ws)
btn = flopy.mt3d.Mt3dBtn(mt, sconc=0, ncomp=2, sconc2=50.0)
adv = flopy.mt3d.Mt3dAdv(mt)
ssm = flopy.mt3d.Mt3dSsm(mt, stress_period_data=ssm_data)
gcg = flopy.mt3d.Mt3dGcg(mt)

# Let's verify that ```stress_period_data``` has the right ```dtype```

print(ssm.stress_period_data.dtype)

# ## Create the ```SEAWAT``` packages

swt = flopy.seawat.Seawat(
    modflowmodel=mf,
    mt3dmodel=mt,
    modelname=modelname,
    namefile_ext="nam_swt",
    model_ws=model_ws,
)
vdf = flopy.seawat.SeawatVdf(swt, mtdnconc=0, iwtable=0, indense=-1)

mf.write_input()
mt.write_input()
swt.write_input()

# And finally, modify the ```vdf``` package to fix ```indense```.

fname = f"{modelname}.vdf"
f = open(os.path.join(model_ws, fname))
lines = f.readlines()
f.close()
f = open(os.path.join(model_ws, fname), "w")
for line in lines:
    f.write(line)
for kper in range(nper):
    f.write("-1\n")
f.close()

try:
    # ignore PermissionError on Windows
    temp_dir.cleanup()
except:
    pass
