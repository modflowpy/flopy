import os
import platform
import sys
import math

import numpy as np

import flopy.modflow as mf
import flopy.utils as fu

import matplotlib.pyplot as plt

# --modify default matplotlib settings
updates = {'font.family':['Univers 57 Condensed', 'Arial'], 
           'mathtext.default':'regular',
           'pdf.compression':0,
           'pdf.fonttype':42,
           'legend.fontsize':7,
           'axes.labelsize':8,
           'xtick.labelsize':7,
           'ytick.labelsize':7}
plt.rcParams.update(updates)

def LegBar(ax, x0, y0, t0, dx, dy, dt, cc):
    for c in cc:
        ax.plot([x0, x0 + dx], [y0, y0], color=c, linewidth=4)
        ctxt = '{0:=3d} years'.format(t0)
        ax.text(x0 + 2. * dx, y0 + dy / 2., ctxt, size=5)
        y0 += dy
        t0 += dt
    return


cleanFiles = False
skipRuns = False
fext = 'png'
narg = len(sys.argv)
iarg = 0
if narg > 1:
    while iarg < narg - 1:
        iarg += 1
        basearg = sys.argv[iarg].lower()
        if basearg == '-clean':
            cleanFiles = True
        elif basearg == '-skipruns':
            skipRuns = True
        elif basearg == '-pdf':
            fext = 'pdf'

if cleanFiles:
    print 'cleaning all files'
    print 'excluding *.py files'
    files = os.listdir('.')
    for f in files:
        if '.py' != os.path.splitext(f)[1].lower():
            print '  removing...{}'.format(os.path.basename(f))
            os.remove(f)
    sys.exit(1)

#Set path and name of MODFLOW exe
exe_name = 'mf2005'
if platform.system() == 'Windows':
    exe_name = 'mf2005'

workspace = '.'

ncol = 61
nrow = 61
nlay = 2


nper = 3
perlen = [365.25 * 200., 365.25 * 12., 365.25 * 18.]
nstp = [1000, 120, 180]
save_head = [200, 60, 60]
steady=True

#--dis data
delr, delc = 50.0, 50.0
botm = np.array([-10., -30., -50.])

#--oc data
savewords = []
for i in xrange(0, nper):
    icnt = 0
    for j in xrange(0, nstp[i]):
        icnt += 1
        savebudget = False
        savehead = False
        if icnt == save_head[i]:
            savebudget = True
            savehead = True
            icnt = 0
        if (j + 1) == nstp[i]:
            savebudget = True
        if savebudget == True or savehead == True:
            twords = [i + 1, j + 1]
            if savebudget == True:
                twords.append('pbudget')
            if savehead == True:
                twords.append('head')
            savewords.append(twords)
            
            
modelname = 'swiex4_2d_2layer'

#--bas data
#--ibound - active except for the corners
ibound = np.ones((nlay, nrow, ncol), dtype= np.int)
ibound[:, 0, 0] = 0
ibound[:, 0, -1] = 0
ibound[:, -1, 0] = 0
ibound[:, -1, -1] = 0
#--initial head data
ihead = np.zeros((nlay, nrow, ncol), dtype=np.float)

#--lpf data
laytyp=0
hk=10.
vka=0.2

#--boundary condition data
#--ghb data
colcell, rowcell = np.meshgrid(np.arange(0, ncol), np.arange(0, nrow))
index = np.zeros((nrow, ncol), dtype=np.int)
index[:, :10] = 1
index[:, -10:] = 1
index[:10, :] = 1
index[-10:, :] = 1
nghb = np.sum(index)
lrchc = np.zeros((nghb, 5))
lrchc[:, 0] = 0
lrchc[:, 1] = rowcell[index == 1]
lrchc[:, 2] = colcell[index == 1]
lrchc[:, 3] = 0.
lrchc[:, 4] = 50.0 * 50.0 / 40.0
#--create ghb dictionary
ghb_data = {0:lrchc}

#--recharge data
rch = np.zeros((nrow, ncol), dtype=np.float)
rch[index == 0] = 0.0004
#--create recharge dictionary
rch_data = {0: rch}

#--well data
nwells = 2
lrcq = np.zeros((nwells, 4))
lrcq[0, :] = np.array((0, 30, 35, 0))
lrcq[1, :] = np.array([1, 30, 35, 0])
lrcqw = lrcq.copy()
lrcqw[0, 3] = -250
lrcqsw = lrcq.copy()
lrcqsw[0, 3] = -250.
lrcqsw[1, 3] = -25.
#--create well dictionary
base_well_data = {0:lrcq, 1:lrcqw}
swwells_well_data = {0:lrcq, 1:lrcqw, 2:lrcqsw}
    
#--swi2 data
adaptive = False
nadptmx = 10
nadptmn = 1
nu = [0, 0.025]
numult = 5.0
toeslope = nu[1] / numult  #0.005
tipslope = nu[1] / numult  #0.005
z1 = -10.0 * np.ones((nrow, ncol))
z1[index == 0] = -11.0
z = np.array([[z1, z1]])
iso = np.zeros((nlay, nrow, ncol), np.int)
iso[0, :, :][index == 0] = 1
iso[0, :, :][index == 1] = -2
iso[1, 30, 35] = 2
ssz=0.2
#--swi2 observations
obsnam = ['layer1_', 'layer2_']
obslrc=[[1, 31, 36], [2, 31, 36]]
nobs = len(obsnam)
iswiobs = 1051 

modelname = 'swiex4_s1'
if not skipRuns:
    ml = mf.Modflow(modelname, version='mf2005', exe_name=exe_name, model_ws=workspace)
    
    discret = mf.ModflowDis(ml, nlay=nlay, nrow=nrow, ncol=ncol, laycbd=0,
                            delr=delr, delc=delc, top=botm[0], botm=botm[1:],
                            nper=nper, perlen=perlen, nstp=nstp, steady=steady)
    bas = mf.ModflowBas(ml, ibound=ibound, strt=ihead)
    lpf = mf.ModflowLpf(ml, laytyp=laytyp, hk=hk, vka=vka)
    wel = mf.ModflowWel(ml, stress_period_data=base_well_data)
    ghb = mf.ModflowGhb(ml, stress_period_data=ghb_data)
    rch = mf.ModflowRch(ml, rech=rch_data)
    swi = mf.ModflowSwi2(ml, nsrf=1, istrat=1, toeslope=toeslope, tipslope=tipslope, nu=nu,
                         zeta=z, ssz=ssz, isource=iso, nsolver=1,
                         adaptive=adaptive, nadptmx=nadptmx, nadptmn=nadptmn, 
                         nobs=nobs, iswiobs=iswiobs, obsnam=obsnam, obslrc=obslrc)
    oc = mf.ModflowOc(ml, words=savewords)
    pcg = mf.ModflowPcg(ml, hclose=1.0e-6, rclose=3.0e-3, mxiter=100, iter1=50)
    #--create model files
    ml.write_input()
    #--run the model
    m = ml.run_model(silent=False)

#--model with saltwater wells
modelname2 = 'swiex4_s2'
if not skipRuns:
    ml2 = mf.Modflow(modelname2, version='mf2005', exe_name=exe_name, model_ws=workspace)
    
    discret = mf.ModflowDis(ml2, nlay=nlay, nrow=nrow, ncol=ncol, laycbd=0,
                            delr=delr, delc=delc, top=botm[0], botm=botm[1:],
                            nper=nper, perlen=perlen, nstp=nstp, steady=steady)
    bas = mf.ModflowBas(ml2, ibound=ibound, strt=ihead)
    lpf = mf.ModflowLpf(ml2, laytyp=laytyp, hk=hk, vka=vka)
    wel = mf.ModflowWel(ml2, stress_period_data=swwells_well_data)
    ghb = mf.ModflowGhb(ml2, stress_period_data=ghb_data)
    rch = mf.ModflowRch(ml2, rech=rch_data)
    swi = mf.ModflowSwi2(ml2, nsrf=1, istrat=1, toeslope=toeslope, tipslope=tipslope, nu=nu,
                         zeta=z, ssz=ssz, isource=iso, nsolver=1,
                         adaptive=adaptive, nadptmx=nadptmx, nadptmn=nadptmn,
                         nobs=nobs, iswiobs=iswiobs, obsnam=obsnam, obslrc=obslrc)
    oc = mf.ModflowOc(ml2, words=savewords)
    pcg = mf.ModflowPcg(ml2, hclose=1.0e-6, rclose=3.0e-3, mxiter=100, iter1=50)
    #--create model files
    ml2.write_input()
    #--run the model
    m = ml2.run_model(silent=False)

# Load the simulation 1 `ZETA` data and `ZETA` observations.
#--read base model zeta
zfile = fu.CellBudgetFile(os.path.join(ml.model_ws, modelname+'.zta'))
kstpkper = zfile.get_kstpkper()
zeta = []
for kk in kstpkper:
    zeta.append(zfile.get_data(kstpkper=kk, text='ZETASRF  1')[0])
zeta = np.array(zeta)
#--read swi obs
zobs = np.genfromtxt(os.path.join(ml.model_ws, modelname+'.zobs'), names=True)


# Load the simulation 2 `ZETA` data and `ZETA` observations.
#--read saltwater well model zeta
zfile2 = fu.CellBudgetFile(os.path.join(ml2.model_ws, modelname2+'.zta'))
kstpkper = zfile2.get_kstpkper()
zeta2 = []
for kk in kstpkper:
    zeta2.append(zfile2.get_data(kstpkper=kk, text='ZETASRF  1')[0])
zeta2 = np.array(zeta2)
#--read swi obs
zobs2 = np.genfromtxt(os.path.join(ml2.model_ws, modelname2+'.zobs'), names=True)


# Create arrays for the x-coordinates and the output years

x = np.linspace(-1500, 1500, 61)
xcell = np.linspace(-1500, 1500, 61) + delr / 2.
xedge = np.linspace(-1525, 1525, 62)
years = [40, 80, 120, 160, 200, 6, 12, 18, 24, 30]


# Define figure dimensions and colors used for plotting `ZETA` surfaces

#--figure dimensions
fwid, fhgt = 8.00, 5.50
flft, frgt, fbot, ftop = 0.125, 0.95, 0.125, 0.925

#--line color definition
icolor = 5
colormap = plt.cm.jet  #winter
cc = []
cr = np.linspace(0.9, 0.0, icolor)
for idx in cr:
    cc.append(colormap(idx))


# Recreate **Figure 9** from the SWI2 documentation (http://pubs.usgs.gov/tm/6a46/).

plt.rcParams.update({'legend.fontsize': 6, 'legend.frameon' : False})
fig = plt.figure(figsize=(fwid, fhgt), facecolor='w')
fig.subplots_adjust(wspace=0.25, hspace=0.25, left=flft, right=frgt, bottom=fbot, top=ftop)
#--first plot
ax = fig.add_subplot(2, 2, 1)
#--axes limits
ax.set_xlim(-1500, 1500)
ax.set_ylim(-50, -10)
for idx in xrange(5):
    #--layer 1
    ax.plot(xcell, zeta[idx, 0, 30, :], drawstyle='steps-mid', 
            linewidth=0.5, color=cc[idx], label='{:2d} years'.format(years[idx]))
    #--layer 2
    ax.plot(xcell, zeta[idx, 1, 30, :], drawstyle='steps-mid',
            linewidth=0.5, color=cc[idx], label='_None')
ax.plot([-1500, 1500], [-30, -30], color='k', linewidth=1.0)
#--legend
plt.legend(loc='lower left')
#--axes labels and text
ax.set_xlabel('Horizontal distance, in meters')
ax.set_ylabel('Elevation, in meters')
ax.text(0.025, .55, 'Layer 1', transform=ax.transAxes, va='center', ha='left', size='7')
ax.text(0.025, .45, 'Layer 2', transform=ax.transAxes, va='center', ha='left', size='7')
ax.text(0.975, .1, 'Recharge conditions', transform=ax.transAxes, va='center', ha='right', size='8')

#--second plot
ax = fig.add_subplot(2, 2, 2)
#--axes limits
ax.set_xlim(-1500, 1500)
ax.set_ylim(-50, -10)
for idx in xrange(5, len(years)):
    #--layer 1
    ax.plot(xcell, zeta[idx, 0, 30, :], drawstyle='steps-mid', 
            linewidth=0.5, color=cc[idx-5], label='{:2d} years'.format(years[idx]))
    #--layer 2
    ax.plot(xcell, zeta[idx, 1, 30, :], drawstyle='steps-mid',
            linewidth=0.5, color=cc[idx-5], label='_None')
ax.plot([-1500, 1500], [-30, -30], color='k', linewidth=1.0)
#--legend
plt.legend(loc='lower left')
#--axes labels and text
ax.set_xlabel('Horizontal distance, in meters')
ax.set_ylabel('Elevation, in meters')
ax.text(0.025, .55, 'Layer 1', transform=ax.transAxes, va='center', ha='left', size='7')
ax.text(0.025, .45, 'Layer 2', transform=ax.transAxes, va='center', ha='left', size='7')
ax.text(0.975, .1, 'Freshwater well withdrawal', transform=ax.transAxes, va='center', ha='right', size='8')

#--third plot
ax = fig.add_subplot(2, 2, 3)
#--axes limits
ax.set_xlim(-1500, 1500)
ax.set_ylim(-50, -10)
for idx in xrange(5, len(years)):
    #--layer 1
    ax.plot(xcell, zeta2[idx, 0, 30, :], drawstyle='steps-mid', 
            linewidth=0.5, color=cc[idx-5], label='{:2d} years'.format(years[idx]))
    #--layer 2
    ax.plot(xcell, zeta2[idx, 1, 30, :], drawstyle='steps-mid',
            linewidth=0.5, color=cc[idx-5], label='_None')
ax.plot([-1500, 1500], [-30, -30], color='k', linewidth=1.0)
#--legend
plt.legend(loc='lower left')
#--axes labels and text
ax.set_xlabel('Horizontal distance, in meters')
ax.set_ylabel('Elevation, in meters')
ax.text(0.025, .55, 'Layer 1', transform=ax.transAxes, va='center', ha='left', size='7')
ax.text(0.025, .45, 'Layer 2', transform=ax.transAxes, va='center', ha='left', size='7')
ax.text(0.975, .1, 'Freshwater and saltwater\nwell withdrawals', transform=ax.transAxes,
        va='center', ha='right', size='8')

#--fourth plot
ax = fig.add_subplot(2, 2, 4)
#--axes limits
ax.set_xlim(0, 30)
ax.set_ylim(-50, -10)
t = zobs['TOTIM'][999:] / 365 - 200.
tz2 = zobs['layer1_001'][999:]
tz3 = zobs2['layer1_001'][999:]
for i in xrange(len(t)):
    if zobs['layer2_001'][i+999] < -30. - 0.1:
        tz2[i] = zobs['layer2_001'][i+999]
    if zobs2['layer2_001'][i+999] < 20. - 0.1:
        tz3[i] = zobs2['layer2_001'][i+999]
ax.plot(t, tz2, linestyle='solid', color='r', linewidth=0.75, label='Freshwater well')
ax.plot(t, tz3, linestyle='dotted', color='r', linewidth=0.75, label='Freshwater and saltwater well')
ax.plot([0, 30], [-30, -30], 'k', linewidth=1.0, label='_None')
#--legend
leg = plt.legend(loc='lower right', numpoints=1)
#--axes labels and text
ax.set_xlabel('Time, in years')
ax.set_ylabel('Elevation, in meters')
ax.text(0.025, .55, 'Layer 1', transform=ax.transAxes, va='center', ha='left', size='7')
ax.text(0.025, .45, 'Layer 2', transform=ax.transAxes, va='center', ha='left', size='7')


outfig = 'Figure09_swi2ex4.{0}'.format(fext)
fig.savefig(outfig, dpi=300)
print 'created...', outfig
