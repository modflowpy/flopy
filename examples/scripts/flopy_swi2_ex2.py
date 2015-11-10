from __future__ import print_function

import os
import sys
import math

import numpy as np

import flopy.modflow as mf
import flopy.mt3d as mt3
import flopy.seawat as swt
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

dirs = [os.path.join('SWI2'), os.path.join('SEAWAT')]

if cleanFiles:
    print('cleaning all files')
    print('excluding *.py files')
    file_dict = {}
    file_dict['.'] = os.listdir('.')
    file_dict[dirs[0]] = os.listdir(dirs[0])
    file_dict[dirs[1]] = os.listdir(dirs[1])
    for key, files in list(file_dict.items()):
        for f in files:
            if os.path.isdir(f):
                continue
            if '.py' != os.path.splitext(f)[1].lower():
                print('  removing...{}'.format(os.path.basename(f)))
                os.remove(os.path.join(key, f))
    for d in dirs:
        if os.path.exists(d):
            os.rmdir(d)
    sys.exit(1)

# make working directories
for d in dirs:
    if not os.path.exists(d):
        os.mkdir(d)

modelname = 'swiex2'
mf_name = 'mf2005'

# problem data
nper = 1
perlen = 2000
nstp = 1000
nlay, nrow, ncol = 1, 1, 60
delr = 5.
nsurf = 2
x = np.arange(0.5 * delr, ncol * delr, delr)
xedge = np.linspace(0, float(ncol) * delr, len(x) + 1)
ibound = np.ones((nrow, ncol), np.int)
ibound[0, 0] = -1
# swi2 data
z0 = np.zeros((nlay, nrow, ncol), np.float)
z1 = np.zeros((nlay, nrow, ncol), np.float)
z0[0, 0, 30:38] = np.arange(-2.5, -40, -5)
z0[0, 0, 38:] = -40
z1[0, 0, 22:30] = np.arange(-2.5, -40, -5)
z1[0, 0, 30:] = -40
z = []
z.append(z0)
z.append(z1)
ssz = 0.2
isource = np.ones((nrow, ncol), 'int')
isource[0, 0] = 2

# stratified model
modelname = 'swiex2_strat'
print('creating...', modelname)
ml = mf.Modflow(modelname, version='mf2005', exe_name=mf_name, model_ws=dirs[0])
discret = mf.ModflowDis(ml, nlay=1, ncol=ncol, nrow=nrow, delr=delr, delc=1, top=0, botm=[-40.0],
                        nper=nper, perlen=perlen, nstp=nstp)
bas = mf.ModflowBas(ml, ibound=ibound, strt=0.05)
bcf = mf.ModflowBcf(ml, laycon=0, tran=2 * 40)
swi = mf.ModflowSwi2(ml, nsrf=nsurf, istrat=1, toeslope=0.2, tipslope=0.2, nu=[0, 0.0125, 0.025],
                     zeta=z, ssz=ssz, isource=isource, nsolver=1)
oc = mf.ModflowOc88(ml, save_head_every=1000)
pcg = mf.ModflowPcg(ml)
ml.write_input()
# run stratified model
if not skipRuns:
    m = ml.run_model(silent=False)
# read stratified results
zetafile = os.path.join(dirs[0], '{}.zta'.format(modelname))
zobj = fu.CellBudgetFile(zetafile)
zkstpkper = zobj.get_kstpkper()
zeta = zobj.get_data(kstpkper=zkstpkper[-1], text='      ZETASRF  1')[0]
zeta2 = zobj.get_data(kstpkper=zkstpkper[-1], text='      ZETASRF  2')[0]
#
# vd model
modelname = 'swiex2_vd'
print('creating...', modelname)
ml = mf.Modflow(modelname, version='mf2005', exe_name=mf_name, model_ws=dirs[0])
discret = mf.ModflowDis(ml, nlay=1, ncol=ncol, nrow=nrow, delr=delr, delc=1, top=0, botm=[-40.0],
                        nper=nper, perlen=perlen, nstp=nstp)
bas = mf.ModflowBas(ml, ibound=ibound, strt=0.05)
bcf = mf.ModflowBcf(ml, laycon=0, tran=2 * 40)
swi = mf.ModflowSwi2(ml, nsrf=nsurf, istrat=0, toeslope=0.2, tipslope=0.2, nu=[0, 0, 0.025, 0.025],
                     zeta=z, ssz=ssz, isource=isource, nsolver=1)
oc = mf.ModflowOc88(ml, save_head_every=1000)
pcg = mf.ModflowPcg(ml)
ml.write_input()
# run vd model
if not skipRuns:
    m = ml.run_model(silent=False)
# read vd model data
zetafile = os.path.join(dirs[0], '{}.zta'.format(modelname))
zobj = fu.CellBudgetFile(zetafile)
zkstpkper = zobj.get_kstpkper()
zetavd = zobj.get_data(kstpkper=zkstpkper[-1], text='      ZETASRF  1')[0]
zetavd2 = zobj.get_data(kstpkper=zkstpkper[-1], text='      ZETASRF  2')[0]
#
# seawat model
swtexe_name = 'swt_v4'
modelname = 'swiex2_swt'
print('creating...', modelname)
swt_xmax = 300.0
swt_zmax = 40.0
swt_delr = 1.0
swt_delc = 1.0
swt_delz = 0.5
swt_ncol = int(swt_xmax / swt_delr)  #300
swt_nrow = 1
swt_nlay = int(swt_zmax / swt_delz)  #80
print(swt_nlay, swt_nrow, swt_ncol)
swt_ibound = np.ones((swt_nlay, swt_nrow, swt_ncol), np.int)
#swt_ibound[0, swt_ncol-1, 0] = -1
swt_ibound[0, 0, 0] = -1
swt_x = np.arange(0.5 * swt_delr, swt_ncol * swt_delr, swt_delr)
swt_xedge = np.linspace(0, float(ncol) * delr, len(swt_x) + 1)
swt_top = 0.
z0 = swt_top
swt_botm = np.zeros((swt_nlay), np.float)
swt_z = np.zeros((swt_nlay), np.float)
zcell = -swt_delz / 2.0
for ilay in range(0, swt_nlay):
    z0 -= swt_delz
    swt_botm[ilay] = z0
    swt_z[ilay] = zcell
    zcell -= swt_delz
#swt_X, swt_Z = np.meshgrid(swt_x, swt_botm)
swt_X, swt_Z = np.meshgrid(swt_x, swt_z)
# mt3d
# mt3d boundary array set to all active
icbund = np.ones((swt_nlay, swt_nrow, swt_ncol), np.int)
# create initial concentrations for MT3D
sconc = np.ones((swt_nlay, swt_nrow, swt_ncol), np.float)
sconcp = np.zeros((swt_nlay, swt_ncol), np.float)
xsb = 110
xbf = 150
for ilay in range(0, swt_nlay):
    for icol in range(0, swt_ncol):
        if swt_x[icol] > xsb:
            sconc[ilay, 0, icol] = 0.5
        if swt_x[icol] > xbf:
            sconc[ilay, 0, icol] = 0.0
    for icol in range(0, swt_ncol):
        sconcp[ilay, icol] = sconc[ilay, 0, icol]
    xsb += swt_delz
    xbf += swt_delz

# ssm data
itype = mt3.Mt3dSsm.itype_dict()
ssm_data = {0: [0, 0, 0, 35., itype['BAS6']]}

#print sconcp
#mt3d print times
timprs = (np.arange(5) + 1) * 2000.
nprs = len(timprs)
# MODFLOW files
ml = []
ml = mf.Modflow(modelname, version='mf2005', exe_name=swtexe_name, model_ws=dirs[1])
discret = mf.ModflowDis(ml, nrow=swt_nrow, ncol=swt_ncol, nlay=swt_nlay,
                        delr=swt_delr, delc=swt_delc, laycbd=0, top=swt_top, botm=swt_botm,
                        nper=nper, perlen=perlen, nstp=1, steady=False)
bas = mf.ModflowBas(ml, ibound=swt_ibound, strt=0.05)
lpf = mf.ModflowLpf(ml, hk=2.0, vka=2.0, ss=0.0, sy=0.0, laytyp=0, layavg=0)
oc = mf.ModflowOc88(ml, save_head_every=1, item2=[[0, 1, 0, 0]])
pcg = mf.ModflowPcg(ml)
ml.write_input()
# Create the basic MT3DMS model structure
mt = mt3.Mt3dms(modelname, 'nam_mt3dms', modflowmodel=ml, 
                model_ws=dirs[1])  # Coupled to modflow model 'mf'
adv = mt3.Mt3dAdv(mt, mixelm=-1,  #-1 is TVD
                  percel=0.05,
                  nadvfd=0,  #0 or 1 is upstream; 2 is central in space
                  #particle based methods
                  nplane=4,
                  mxpart=1e7,
                  itrack=2,
                  dceps=1e-4,
                  npl=16,
                  nph=16,
                  npmin=8,
                  npmax=256)
btn = mt3.Mt3dBtn(mt, icbund=1, prsity=ssz, sconc=sconc, ifmtcn=-1,
                  chkmas=False, nprobs=10, nprmas=10, dt0=0.0, ttsmult=1.2, ttsmax=100.0,
                  ncomp=1, nprs=nprs, timprs=timprs, mxstrn=1e8)
dsp = mt3.Mt3dDsp(mt, al=0., trpt=1., trpv=1., dmcoef=0.)
gcg = mt3.Mt3dGcg(mt, mxiter=1, iter1=50, isolve=3, cclose=1e-6, iprgcg=5)
ssm = mt3.Mt3dSsm(mt, stress_period_data=ssm_data)
mt.write_input()
# Create the SEAWAT model structure
mswtf = swt.Seawat(modelname, 'nam_swt', modflowmodel=ml, mt3dmsmodel=mt,
                   exe_name=swtexe_name, model_ws=dirs[1])  # Coupled to modflow model mf and mt3dms model mt
vdf = swt.SeawatVdf(mswtf, nswtcpl=1, iwtable=0, densemin=0, densemax=0, denseref=1000., denseslp=25., firstdt=1.0e-03)
mswtf.write_input()
# run seawat model
if not skipRuns:
    m = mswtf.run_model(silent=False)
# read seawat model data
ucnfile = os.path.join(dirs[1], 'MT3D001.UCN')
uobj = fu.UcnFile(ucnfile)
times = uobj.get_times()
print(times)
ukstpkper = uobj.get_kstpkper()
print(ukstpkper)
c = uobj.get_data(totim=times[-1])
conc = np.zeros((swt_nlay, swt_ncol), np.float)
for icol in range(0, swt_ncol):
    for ilay in range(0, swt_nlay):
        conc[ilay, icol] = c[ilay, 0, icol]
#
# figure
fwid = 7.0  #6.50
fhgt = 4.5  #6.75
flft = 0.125
frgt = 0.95
fbot = 0.125
ftop = 0.925

print('creating  cross-section figure...')
xsf = plt.figure(figsize=(fwid, fhgt), facecolor='w')
xsf.subplots_adjust(wspace=0.25, hspace=0.25, left=flft, right=frgt, bottom=fbot, top=ftop)
# plot initial conditions
ax = xsf.add_subplot(3, 1, 1)
ax.text(-0.075, 1.05, 'A', transform=ax.transAxes, va='center', ha='center', size='8')
#text(.975, .1, '(a)', transform = ax.transAxes, va = 'center', ha = 'center')
ax.plot([110, 150], [0, -40], 'k')
ax.plot([150, 190], [0, -40], 'k')
ax.set_xlim(0, 300)
ax.set_ylim(-40, 0)
ax.set_yticks(np.arange(-40, 1, 10))
ax.text(50, -20, 'salt', va='center', ha='center')
ax.text(150, -20, 'brackish', va='center', ha='center')
ax.text(250, -20, 'fresh', va='center', ha='center')
ax.set_ylabel('Elevation, in meters')
# plot stratified swi2 and seawat results
ax = xsf.add_subplot(3, 1, 2)
ax.text(-0.075, 1.05, 'B', transform=ax.transAxes, va='center', ha='center', size='8')
#
zp = zeta[0, 0, :]
p = (zp < 0.0) & (zp > -40.)
ax.plot(x[p], zp[p], 'b', linewidth=1.5, drawstyle='steps-mid')
zp = zeta2[0, 0, :]
p = (zp < 0.0) & (zp > -40.)
ax.plot(x[p], zp[p], 'b', linewidth=1.5, drawstyle='steps-mid')
# seawat data
cc = ax.contour(swt_X, swt_Z, conc, levels=[0.25, 0.75], colors='k', linestyles='solid', linewidths=0.75, zorder=101)
# fake figures
ax.plot([-100., -100], [-100., -100], 'b', linewidth=1.5, label='SWI2')
ax.plot([-100., -100], [-100., -100], 'k', linewidth=0.75, label='SEAWAT')
# legend
leg = ax.legend(loc='lower left', numpoints=1)
leg._drawFrame = False
# axes
ax.set_xlim(0, 300)
ax.set_ylim(-40, 0)
ax.set_yticks(np.arange(-40, 1, 10))
ax.set_ylabel('Elevation, in meters')
# plot vd model
ax = xsf.add_subplot(3, 1, 3)
ax.text(-0.075, 1.05, 'C', transform=ax.transAxes, va='center', ha='center', size='8')
dr = zeta[0, 0, :]
ax.plot(x, dr, 'b', linewidth=1.5, drawstyle='steps-mid')
dr = zeta2[0, 0, :]
ax.plot(x, dr, 'b', linewidth=1.5, drawstyle='steps-mid')
dr = zetavd[0, 0, :]
ax.plot(x, dr, 'r', linewidth=0.75, drawstyle='steps-mid')
dr = zetavd2[0, 0, :]
ax.plot(x, dr, 'r', linewidth=0.75, drawstyle='steps-mid')
# fake figures
ax.plot([-100., -100], [-100., -100], 'b', linewidth=1.5, label='SWI2 stratified option')
ax.plot([-100., -100], [-100., -100], 'r', linewidth=0.75, label='SWI2 continuous option')
# legend
leg = ax.legend(loc='lower left', numpoints=1)
leg._drawFrame = False
# axes
ax.set_xlim(0, 300)
ax.set_ylim(-40, 0)
ax.set_yticks(np.arange(-40, 1, 10))
ax.set_xlabel('Horizontal distance, in meters')
ax.set_ylabel('Elevation, in meters')

outfig = 'Figure07_swi2ex2.{0}'.format(fext)
xsf.savefig(outfig, dpi=300)
print('created...', outfig)
