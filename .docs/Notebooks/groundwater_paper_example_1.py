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
#     section: 2016gw-paper
# ---

# ## Basic Flopy example
#
# From:
# Bakker, Mark, Post, Vincent, Langevin, C. D., Hughes, J. D., White, J. T., Starn, J. J. and Fienen, M. N., 2016, Scripting MODFLOW Model Development Using Python and FloPy: Groundwater, v. 54, p. 733-739, https://doi.org/10.1111/gwat.12413.

# Import the `modflow` and `utils` subpackages of FloPy and give them the aliases `fpm` and `fpu`, respectively

# +
import os
import sys
from pprint import pformat

import matplotlib as mpl
import numpy as np

# run installed version of flopy or add local path
import flopy
import flopy.modflow as fpm
import flopy.utils as fpu

print(sys.version)
print(f"numpy version: {np.__version__}")
print(f"matplotlib version: {mpl.__version__}")
print(f"flopy version: {flopy.__version__}")
# -

# Create a MODFLOW model object. Here, the MODFLOW model object is stored in a Python variable called {\tt model}, but this can be an arbitrary name. This object name is important as it will be used as a reference to the model in the remainder of the FloPy script. In addition, a {\tt modelname} is specified when the MODFLOW model object is created. This {\tt modelname} is used for all the files that are created by FloPy for this model.

exe = "mf2005"
ws = os.path.join("temp")
model = fpm.Modflow(modelname="gwexample", exe_name=exe, model_ws=ws)

# The discretization of the model is specified with the discretization file (DIS) of MODFLOW. The aquifer is divided into 201 cells of length 10 m and width 1 m.  The first input of the discretization package is the name of the model object. All other input arguments are self explanatory.

fpm.ModflowDis(model, nlay=1, nrow=1, ncol=201, delr=10, delc=1, top=50, botm=0)

# Active cells and the like are defined with the Basic package (BAS), which is required for every MODFLOW model. It contains the {\tt ibound} array, which is used to specify which cells are active (value is positive), inactive (value is 0), or fixed head (value is negative). The {\tt numpy} package (aliased as {\tt np}) can be used to quickly initialize the {\tt ibound} array with values of 1, and then set the {\tt ibound} value for the first and last columns to -1. The {\tt numpy} package (and Python, in general) uses zero-based indexing and supports negative indexing so that row 1 and column 1, and row 1 and column 201, can be referenced as [0, 0], and [0, -1], respectively.  Although this simulation is for steady flow, starting heads still need to be specified. They are used as the head for fixed-head cells (where {\tt ibound} is negative), and as a starting point to compute the saturated thickness for cases of unconfined flow.

ibound = np.ones((1, 201))
ibound[0, 0] = ibound[0, -1] = -1
fpm.ModflowBas(model, ibound=ibound, strt=20)

# The hydraulic properties of the aquifer are specified with the Layer Properties Flow (LPF) package (alternatively, the Block Centered Flow (BCF) package may be used). Only the hydraulic conductivity of the aquifer and the layer type ({\tt laytyp}) need to be specified. The latter is set to 1, which means that MODFLOW will calculate the saturated thickness differently depending on whether or not the head is above the top of the aquifer.
#

fpm.ModflowLpf(model, hk=10, laytyp=1)

# Aquifer recharge is simulated with the Recharge package (RCH) and the extraction of water at the two ditches is simulated with the Well package (WEL). The latter requires specification of the layer, row, column, and injection rate of the well for each stress period. The layers, rows, columns, and the stress period are numbered (consistent with Python's zero-based numbering convention) starting at 0. The required data are stored in a Python dictionary ({\tt lrcQ} in the code below), which is used in FloPy to store data that can vary by stress period. The {\tt lrcQ} dictionary specifies that two wells (one in cell 1, 1, 51 and one in cell 1, 1, 151), each with a rate of -1 m$^3$/m/d, will be active for the first stress period.  Because this is a steady-state model, there is only one stress period and therefore only one entry in the dictionary.

fpm.ModflowRch(model, rech=0.001)
lrcQ = {0: [[0, 0, 50, -1], [0, 0, 150, -1]]}
fpm.ModflowWel(model, stress_period_data=lrcQ)

# The Preconditioned Conjugate-Gradient (PCG) solver, using the default settings, is specified to solve the model.

fpm.ModflowPcg(model)

# The frequency and type of output that MODFLOW writes to an output file is specified with the Output Control (OC) package.  In this case, the budget is printed and heads are saved (the default), so no arguments are needed.

fpm.ModflowOc(model)

# Finally the MODFLOW input files are written (eight files for this model) and the model is run. This requires, of course, that MODFLOW is installed on your computer and FloPy can find the executable in your path.

model.write_input()
success, buff = model.run_model(silent=True, report=True)
assert success, pformat(buff)

# After MODFLOW has responded with the positive {\tt Normal termination of simulation}, the calculated heads can be read from the binary output file. First a file object is created. As the modelname used for all MODFLOW files was specified as {\tt gwexample} in step 1, the file with the heads is called {\tt gwexample.hds}. FloPy includes functions to read data from the file object, including heads for specified layers or time steps, or head time series at individual cells. For this simple mode, all computed heads are read.

fpth = os.path.join(ws, "gwexample.hds")
hfile = fpu.HeadFile(fpth)
h = hfile.get_data(totim=1.0)

# The heads are now stored in the Python variable {\tt h}. FloPy includes powerful plotting functions to plot the grid, boundary conditions, head, etc. This functionality is demonstrated later. For this simple one-dimensional example, a plot is created with the matplotlib package

# +
import matplotlib.pyplot as plt

ax = plt.subplot(111)
x = model.modelgrid.xcellcenters[0]
ax.plot(x, h[0, 0, :])
ax.set_xlim(0, x.max())
ax.set_xlabel("x(m)")
ax.set_ylabel("head(m)")
plt.show()
