__author__ = 'langevin'

import numpy as np
import flopy

# Model domain and grid definition
Lx = 1000.
Ly = 1000.
ztop = 0.
zbot = -50.
nlay = 1
nrow = 10
ncol = 10
delr = Lx / ncol
delc = Ly / nrow
delv = (ztop - zbot) / nlay
botm = np.linspace(ztop, zbot, nlay + 1)
hk = 1.
vka = 1.
sy = 0.1
ss = 1.e-4
laytyp = 1

# Variables for the BAS package
# Note that changes from the previous tutorial!
ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)
strt = 10. * np.ones((nlay, nrow, ncol), dtype=np.float32)

# Time step parameters
nper = 3
perlen = [1, 100, 100]
nstp = [1, 100, 100]
steady = [True, False, False]

# Flopy objects
modelname = 'tutorial2'
mf = flopy.modflow.Modflow(modelname, exe_name='mf2005')
dis = flopy.modflow.ModflowDis(mf, nlay, nrow, ncol, delr=delr, delc=delc,
                               top=ztop, botm=botm[1:],
                               nper=nper, perlen=perlen, nstp=nstp,
                               steady=steady)
bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)
lpf = flopy.modflow.ModflowLpf(mf, hk=hk, vka=vka, sy=sy, ss=ss, laytyp=laytyp)
pcg = flopy.modflow.ModflowPcg(mf)

# Make list for stress period 1
stageleft = 10.
stageright = 10.
bound_sp1 = []
for il in xrange(nlay):
    condleft = hk * (stageleft - zbot) * delc
    condright = hk * (stageright - zbot) * delc
    for ir in xrange(nrow):
        bound_sp1.append([il, ir, 0, stageleft, condleft])
        bound_sp1.append([il, ir, ncol - 1, stageright, condright])
print 'Adding ', len(bound_sp1), 'GHBs for stress period 1.'

# Make list for stress period 2
stageleft = 10.
stageright = 0.
condleft = hk * (stageleft - zbot) * delc
condright = hk * (stageright - zbot) * delc
bound_sp2 = []
for il in xrange(nlay):
    for ir in xrange(nrow):
        bound_sp2.append([il, ir, 0, stageleft, condleft])
        bound_sp2.append([il, ir, ncol - 1, stageright, condright])
print 'Adding ', len(bound_sp2), 'GHBs for stress period 2.'

# We do not need to add a dictionary entry for stress period 3.
# Flopy will automatically take the list from stess period 2 and apply it
# to the end of the simulation, if necessary
stress_period_data = {0: bound_sp1, 1: bound_sp2}

# Create the flopy ghb object
ghb = flopy.modflow.ModflowGhb(mf, stress_period_data=stress_period_data)

# Create the well package
# Remember to use zero-based layer, row, column indices!
pumping_rate = -100.
wel_sp1 = [[0, nrow/2 - 1, nrow/2, 0.]]
wel_sp2 = [[0, nrow/2 - 1, nrow/2, 0.]]
wel_sp3 = [[0, nrow/2 - 1, nrow/2, pumping_rate]]
stress_period_data = {0: wel_sp1, 1: wel_sp2, 2: wel_sp3}
wel = flopy.modflow.ModflowWel(mf, stress_period_data=stress_period_data)

# Output control
words = ['head','drawdown','budget', 'phead', 'pbudget']
save_head_every = 1
oc = flopy.modflow.ModflowOc(mf, words=words, save_head_every=save_head_every)

# Write the model input files
mf.write_input()

# Run the model
success, mfoutput = mf.run_model(silent=False, pause=False)
if not success:
    raise Exception('MODFLOW did not terminate normally.')


# Imports
import matplotlib.pyplot as plt
import flopy.utils.binaryfile as bf

# Create the headfile object
headobj = bf.HeadFile(modelname+'.hds')
times = headobj.get_times()

# Setup contour parameters
levels = np.arange(1, 10, 1)
extent = (delr/2., Lx - delr/2., delc/2., Ly - delc/2.)
print 'Levels: ', levels
print 'Extent: ', extent

# Make the plots
mytimes = [1.0, 101.0, 201.0]
for iplot, time in enumerate(mytimes):
    print '*****Processing time: ', time
    head = headobj.get_data(totim=time)
    #Print statistics
    print 'Head statistics'
    print '  min: ', head.min()
    print '  max: ', head.max()
    print '  std: ', head.std()

    #Create the plot
    #plt.subplot(1, len(mytimes), iplot + 1, aspect='equal')
    plt.subplot(1, 1, 1, aspect='equal')
    plt.title('stress period ' + str(iplot + 1))
    plt.imshow(head[0, :, :], extent=extent, cmap='BrBG', vmin=0., vmax=10., origin='lower')
    plt.colorbar()
    CS = plt.contour(head[0, :, :], levels=levels, extent=extent)
    plt.clabel(CS, inline=1, fontsize=10, fmt='%1.1f')
    plt.show()

plt.show()

# Plot the head versus time
idx = (0, nrow/2 - 1, ncol/2 - 1)
ts = headobj.get_ts(idx)
plt.subplot(1, 1, 1)
ttl = 'Head at cell ({0},{1},{2})'.format(idx[0] + 1, idx[1] + 1, idx[2] + 1)
plt.title(ttl)
plt.xlabel('time')
plt.ylabel('head')
plt.plot(ts[:, 0], ts[:, 1])
plt.show()
