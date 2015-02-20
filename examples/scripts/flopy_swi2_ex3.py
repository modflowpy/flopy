import os
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

def MergeData(ndim, zdata, tb):
    sv = 0.05
    md = np.empty((ndim), np.float)
    md.fill(np.nan)
    found = np.empty((ndim), np.bool)
    found.fill(False)
    for idx, layer in enumerate(zdata):
        for jdx, z in enumerate(layer):
            if found[jdx] == True:
                continue
            t0 = tb[idx][0] - sv
            t1 = tb[idx][1] + sv
            if z < t0 and z > t1:
                md[jdx] = z
                found[jdx] = True
    return md


def LegBar(ax, x0, y0, t0, dx, dy, dt, cc):
    for c in cc:
        ax.plot([x0, x0 + dx], [y0, y0], color=c, linewidth=4)
        ctxt = '{0:=3d} years'.format(t0)
        ax.text(x0 + 2. * dx, y0 + dy / 2., ctxt, size=5)
        y0 += dy
        t0 += dt
    return


cleanFiles = False
fext = 'png'
narg = len(sys.argv)
iarg = 0
if narg > 1:
    while iarg < narg - 1:
        iarg += 1
        basearg = sys.argv[iarg].lower()
        if basearg == '-clean':
            cleanFiles = True
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

modelname = 'swiex3'
exe_name = 'mf2005'

nlay = 3
nrow = 1
ncol = 200
delr = 20.0
delc = 1.
#--well data
lrcQ1 = np.array([(0, 0, 199, 0.01), (2, 0, 199, 0.02)])
lrcQ2 = np.array([(0, 0, 199, 0.01 * 0.5), (2, 0, 199, 0.02 * 0.5)])
#--ghb data
lrchc = np.zeros((30, 5))
lrchc[:, [0, 1, 3, 4]] = [0, 0, 0., 0.8 / 2.0]
lrchc[:, 2] = np.arange(0, 30)
#--swi2 data
zini = np.hstack(( -9 * np.ones(24), np.arange(-9, -50, -0.5), -50 * np.ones(94)))[np.newaxis, :]
iso = np.zeros((1, 200), dtype=np.int)
iso[:, :30] = -2
#--model objects
ml = mf.Modflow(modelname, version='mf2005', exe_name=exe_name)
discret = mf.ModflowDis(ml, nrow=nrow, ncol=ncol, nlay=3, delr=delr, delc=delc,
                        laycbd=[0, 0, 0], top=-9.0, botm=[-29, -30, -50],
                        nper=2, perlen=[365 * 1000, 1000 * 365], nstp=[500, 500])
bas = mf.ModflowBas(ml, ibound=1, strt=1.0)
bcf = mf.ModflowBcf(ml, laycon=[0, 0, 0], tran=[40.0, 1, 80.0], vcont=[0.005, 0.005])
wel = mf.ModflowWel(ml, stress_period_data={0:lrcQ1, 1:lrcQ2})
ghb = mf.ModflowGhb(ml, stress_period_data={0:lrchc})
swi = mf.ModflowSwi2(ml, nsrf=1, istrat=1, toeslope=0.01, tipslope=0.04, nu=[0, 0.025],
                     zeta=[zini, zini, zini], ssz=0.2, isource=iso, nsolver=1)
oc = mf.ModflowOc(ml, save_head_every=100)
pcg = mf.ModflowPcg(ml)
#--write the model files
ml.write_input()
#--run the model
m = ml.run_model(silent=True)

headfile = '{}.hds'.format(modelname)
hdobj = fu.HeadFile(headfile)
head = hdobj.get_data(totim=3.65000E+05)

zetafile = '{}.zta'.format(modelname)
zobj = fu.CellBudgetFile(zetafile)
zkstpkper = zobj.get_kstpkper()
zeta = []
for kk in zkstpkper:
    zeta.append(zobj.get_data(kstpkper=kk, text='      ZETASRF  1')[0])
zeta = np.array(zeta)

fwid, fhgt = 7.00, 4.50
flft, frgt, fbot, ftop = 0.125, 0.95, 0.125, 0.925

colormap = plt.cm.spectral  #winter
cc = []
icolor = 11
cr = np.linspace(0.0, 0.9, icolor)
for idx in cr:
    cc.append(colormap(idx))
lw = 0.5

x = np.arange(-30 * delr + 0.5 * delr, (ncol - 30) * delr, delr)
xedge = np.linspace(-30. * delr, (ncol - 30.) * delr, len(x) + 1)
zedge = [[-9., -29.], [-29., -30.], [-30., -50.]]

fig = plt.figure(figsize=(fwid, fhgt), facecolor='w')
fig.subplots_adjust(wspace=0.25, hspace=0.25, left=flft, right=frgt, bottom=fbot, top=ftop)

ax = fig.add_subplot(311)
ax.text(-0.075, 1.05, 'A', transform=ax.transAxes, va='center', ha='center', size='8')
#--confining unit
ax.fill([-600, 3400, 3400, -600], [-29, -29, -30, -30], fc=[.8, .8, .8], ec=[.8, .8, .8])
#--
z = np.copy(zini[0, :])
zr = z.copy()
p = (zr < -9.) & (zr > -50.0)
ax.plot(x[p], zr[p], color=cc[0], linewidth=lw, drawstyle='steps-mid')
#--
for i in range(5):
    zt = MergeData(ncol, [zeta[i, 0, 0, :], zeta[i, 1, 0, :], zeta[i, 2, 0, :]], zedge)
    dr = zt.copy()
    ax.plot(x, dr, color=cc[i + 1], linewidth=lw, drawstyle='steps-mid')
#--Manufacture a legend bar
LegBar(ax, -200., -33.75, 0, 25, -2.5, 200, cc[0:6])
#--axes
ax.set_ylim(-50, -9)
ax.set_ylabel('Elevation, in meters')
ax.set_xlim(-250., 2500.)

ax = fig.add_subplot(312)
ax.text(-0.075, 1.05, 'B', transform=ax.transAxes, va='center', ha='center', size='8')
#--confining unit
ax.fill([-600, 3400, 3400, -600], [-29, -29, -30, -30], fc=[.8, .8, .8], ec=[.8, .8, .8])
#--
for i in range(4, 10):
    zt = MergeData(ncol, [zeta[i, 0, 0, :], zeta[i, 1, 0, :], zeta[i, 2, 0, :]], zedge)
    dr = zt.copy()
    ax.plot(x, dr, color=cc[i + 1], linewidth=lw, drawstyle='steps-mid')
#--Manufacture a legend bar
LegBar(ax, -200., -33.75, 1000, 25, -2.5, 200, cc[5:11])
#--axes
ax.set_ylim(-50, -9)
ax.set_ylabel('Elevation, in meters')
ax.set_xlim(-250., 2500.)

ax = fig.add_subplot(313)
ax.text(-0.075, 1.05, 'C', transform=ax.transAxes, va='center', ha='center', size='8')
#--confining unit
ax.fill([-600, 3400, 3400, -600], [-29, -29, -30, -30], fc=[.8, .8, .8], ec=[.8, .8, .8])
#--
zt = MergeData(ncol, [zeta[4, 0, 0, :], zeta[4, 1, 0, :], zeta[4, 2, 0, :]], zedge)
ax.plot(x, zt, marker='o', markersize=3, linewidth=0.0, markeredgecolor='blue', markerfacecolor='None')
#--ghyben herzberg
zeta1 = -9 - 40. * (head[0, 0, :])
gbh = np.empty(len(zeta1), np.float)
gbho = np.empty(len(zeta1), np.float)
for idx, z1 in enumerate(zeta1):
    if z1 >= -9.0 or z1 <= -50.0:
        gbh[idx] = np.nan
        gbho[idx] = 0.
    else:
        gbh[idx] = z1
        gbho[idx] = z1
ax.plot(x, gbh, 'r')
np.savetxt('Ghyben-Herzberg.out', gbho)
#--fake figures    
ax.plot([-100., -100], [-100., -100], 'r', label='Ghyben-Herzberg')
ax.plot([-100., -100], [-100., -100], 'bo', markersize=3, markeredgecolor='blue', markerfacecolor='None', label='SWI2')
#--legend
leg = ax.legend(loc='lower left', numpoints=1)
leg._drawFrame = False
#--axes
ax.set_ylim(-50, -9)
ax.set_xlabel('Horizontal distance, in meters')
ax.set_ylabel('Elevation, in meters')
ax.set_xlim(-250., 2500.)

outfig = 'Figure08_swi2ex3.{0}'.format(fext)
fig.savefig(outfig, dpi=300)
print 'created...', outfig
