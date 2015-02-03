__author__ = 'langevin'

import numpy as np
import flopy

# Assign name and create modflow model object
modelname = 'tutorial1'
mf = flopy.modflow.Modflow(modelname, exe_name='mf2005')

# Model domain and grid definition
Lx = 1000.
Ly = 1000.
ztop = 0.
zbot = -50.
nlay = 1
nrow = 10
ncol = 10
delr = Lx/ncol
delc = Ly/nrow
delv = (ztop - zbot) / nlay
botm = np.linspace(ztop, zbot, nlay + 1)

# Create the discretization object
dis = flopy.modflow.ModflowDis(mf, nlay, nrow, ncol, delr=delr, delc=delc,
                               top=ztop, botm=botm[1:])

# Variables for the BAS package
ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)
ibound[:, :, 0] = -1
ibound[:, :, -1] = -1
strt = np.ones((nlay, nrow, ncol), dtype=np.float32)
strt[:, :, 0] = 10.
strt[:, :, -1] = 0.
bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)

# Add LPF package to the MODFLOW model
lpf = flopy.modflow.ModflowLpf(mf, hk=10., vka=10.)

# Add OC package to the MODFLOW model
oc = flopy.modflow.ModflowOc(mf)

# Add PCG package to the MODFLOW model
pcg = flopy.modflow.ModflowPcg(mf)

# Write the MODFLOW model input files
mf.write_input()

# Run the MODFLOW model
success, buff = mf.run_model()

# Post process the results
import matplotlib.pyplot as plt
import flopy.utils.binaryfile as bf
plt.subplot(1, 1, 1, aspect='equal')
hds = bf.HeadFile(modelname+'.hds')
head = hds.get_data(totim=1.0)
levels = np.arange(1, 10, 1)
extent = (delr/2., Lx - delr/2., Ly - delc/2., delc/2.)
plt.contour(head[0, :, :], levels=levels, extent=extent)
plt.show()
