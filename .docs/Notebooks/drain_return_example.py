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

# # Flopy Drain Return (DRT) capabilities

# +
import sys
import os
from tempfile import TemporaryDirectory

import numpy as np
import flopy
import matplotlib as mpl
import matplotlib.pyplot as plt


print(sys.version)
print("numpy version: {}".format(np.__version__))
print("matplotlib version: {}".format(mpl.__version__))
print("flopy version: {}".format(flopy.__version__))

# +
# temporary directory
temp_dir = TemporaryDirectory()
modelpth = temp_dir.name

# creat the model package
m = flopy.modflow.Modflow(
    "drt_test",
    model_ws=modelpth,
    exe_name="mfnwt",
    version="mfnwt",
)
d = flopy.modflow.ModflowDis(
    m,
    nlay=1,
    nrow=10,
    ncol=10,
    nper=1,
    perlen=1,
    top=10,
    botm=0,
    steady=True,
)
b = flopy.modflow.ModflowBas(m, strt=10, ibound=1)
u = flopy.modflow.ModflowUpw(m, hk=10)
n = flopy.modflow.ModflowNwt(m)
o = flopy.modflow.ModflowOc(m)
# -

# create the drt package
spd = []
for i in range(m.nrow):
    spd.append([0, i, m.ncol - 1, 5.0, 50.0, 1, 1, 1, 1.0])
d = flopy.modflow.ModflowDrt(m, stress_period_data={0: spd})

# run the drt model
m.write_input()
success, buff = m.run_model(silent=True, report=True)
if success:
    for line in buff:
        print(line)
else:
    raise ValueError("Failed to run.")

# plot heads for the drt model
hds = flopy.utils.HeadFile(os.path.join(m.model_ws, m.name + ".hds"))
hds.plot(colorbar=True)

# remove the drt package and create a standard drain file
spd = []
for i in range(m.nrow):
    spd.append([0, i, m.ncol - 1, 5.0, 1.0])
m.remove_package("DRT")
d = flopy.modflow.ModflowDrn(m, stress_period_data={0: spd})

# run the drain model
m.write_input()
success, buff = m.run_model(silent=True, report=True)
if success:
    for line in buff:
        print(line)
else:
    raise ValueError("Failed to run.")

# plot the heads for the model with the drain
hds = flopy.utils.HeadFile(os.path.join(m.model_ws, m.name + ".hds"))
hds.plot(colorbar=True)

try:
    # ignore PermissionError on Windows
    temp_dir.cleanup()
except:
    pass
