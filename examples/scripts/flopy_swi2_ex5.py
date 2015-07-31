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
    print 'cleaning all files'
    print 'excluding *.py files'
    file_dict = {}
    file_dict['.'] = os.listdir('.')
    file_dict[dirs[0]] = os.listdir(dirs[0])
    file_dict[dirs[1]] = os.listdir(dirs[1])
    for key, files in file_dict.iteritems():
        for f in files:
            if os.path.isdir(f):
                continue
            if '.py' != os.path.splitext(f)[1].lower():
                print '  removing...{}'.format(os.path.basename(f))
                os.remove(os.path.join(key, f))
    for d in dirs:
        if os.path.exists(d):
            os.rmdir(d)
    sys.exit(1)

# --make working directories
for d in dirs:
    if not os.path.exists(d):
        os.mkdir(d)

# --problem data
nlay = 6
nrow = 1
ncol = 113
delr = np.zeros((ncol), np.float)
delc = 1.
r = np.zeros((ncol), np.float)
x = np.zeros((ncol), np.float)
edge = np.zeros((ncol), np.float)
dx = 25.0
for i in xrange(0, ncol):
    delr[i] = dx
r[0] = delr[0] / 2.0
for i in xrange(1, ncol):
    r[i] = r[i - 1] + ( delr[i - 1] + delr[i] ) / 2.0
x[0] = delr[0] / 2.0
for i in xrange(1, ncol):
    x[i] = x[i - 1] + ( delr[i - 1] + delr[i] ) / 2.0
edge[0] = delr[0]
for i in xrange(1, ncol):
    edge[i] = edge[i - 1] + delr[i]

# constant data for all simulations
nper = 2
perlen = [1460, 1460]
nstp = [1460, 1460]
steady = True

nsave_zeta = 8
ndecay = 4
ibound = np.ones((nlay, nrow, ncol), np.int)
for k in xrange(0, nlay):
    ibound[k, 0, ncol - 1] = -1
bot = np.zeros((nlay, nrow, ncol), np.float)
dz = 100. / float(nlay - 1)
zall = -np.arange(0, 100 + dz, dz)
zall = np.append(zall, -120.)
tb = -np.arange(dz, 100 + dz, dz)
tb = np.append(tb, -120.)
for k in xrange(0, nlay):
    for i in xrange(0, ncol):
        bot[k, 0, i] = tb[k]
isource = np.zeros((nlay, nrow, ncol), np.int)
isource[:, 0, ncol - 1] = 1
isource[nlay - 1, 0, ncol - 1] = 2

khb = (0.0000000000256 * 1000. * 9.81 / 0.001) * 60 * 60 * 24
kvb = (0.0000000000100 * 1000. * 9.81 / 0.001) * 60 * 60 * 24
ssb = 1e-5
sszb = 0.2
kh = np.zeros((nlay, nrow, ncol), np.float)
kv = np.zeros((nlay, nrow, ncol), np.float)
ss = np.zeros((nlay, nrow, ncol), np.float)
ssz = np.zeros((nlay, nrow, ncol), np.float)
for k in xrange(0, nlay):
    for i in range(0, ncol):
        f = r[i] * 2.0 * math.pi
        kh[k, 0, i] = khb * f
        kv[k, 0, i] = kvb * f
        ss[k, 0, i] = ssb * f
        ssz[k, 0, i] = sszb * f
z = np.ones((nlay), np.float)
z = -100. * z

nwell = 1
for k in xrange(0, nlay):
    if zall[k] > -20. and zall[k + 1] <= -20:
        nwell = k + 1
print 'nlay={} dz={} nwell={}'.format(nlay, dz, nwell)
wellQ = -2400.
wellbtm = -20.0
wellQpm = wellQ / abs(wellbtm)
well_data = {}
for ip in xrange(0, nper):
    welllist = np.zeros((nwell, 4), np.float)
    for iw in xrange(0, nwell):
        if ip == 0:
            b = zall[iw] - zall[iw + 1]
            if zall[iw + 1] < wellbtm:
                b = zall[iw] - wellbtm
            q = wellQpm * b
        else:
            q = 0.0
        welllist[iw, 0] = iw
        welllist[iw, 1] = 0
        welllist[iw, 2] = 0
        welllist[iw, 3] = q
    well_data[ip] = welllist.copy()

ihead = np.zeros((nlay), np.float)

savewords = []
for i in xrange(0, nper):
    icnt = 0
    for j in xrange(0, nstp[i]):
        icnt += 1
        savebudget = False
        savehead = False
        if icnt == 365:
            savebudget = True
            savehead = True
            icnt = 0
        if j < 10:
            savebudget = True
        if savebudget == True or savehead == True:
            twords = [i + 1, j + 1]
            if savebudget == True:
                twords.append('pbudget')
            if savehead == True:
                twords.append('head')
            savewords.append(twords)
solver2params = {'mxiter': 100, 'iter1': 20, 'npcond': 1, 'zclose': 1.0e-6, 'rclose': 3e-3, 'relax': 1.0,
                 'nbpol': 2, 'damp': 1.0, 'dampt': 1.0}

# --create model file and run model
modelname = 'swi2ex5'
mf_name = 'mf2005'
if not skipRuns:
    ml = mf.Modflow(modelname, version='mf2005', exe_name=mf_name, model_ws=dirs[0])
    discret = mf.ModflowDis(ml, nrow=nrow, ncol=ncol, nlay=nlay, delr=delr, delc=delc,
                            top=0, botm=bot, laycbd=0, nper=nper,
                            perlen=perlen, nstp=nstp, steady=steady)
    bas = mf.ModflowBas(ml, ibound=ibound, strt=ihead)
    lpf = mf.ModflowLpf(ml, hk=kh, vka=kv, ss=ss, sy=ssz, vkcb=0, laytyp=0, layavg=1)
    wel = mf.ModflowWel(ml, stress_period_data=well_data)
    swi = mf.ModflowSwi2(ml, npln=1, istrat=1, toeslope=0.025, tipslope=0.025, nu=[0, 0.025], \
                         zeta=z, ssz=ssz, isource=isource, nsolver=2, solver2params=solver2params)
    oc = mf.ModflowOc88(ml, words=savewords)
    pcg = mf.ModflowPcg(ml, hclose=1.0e-6, rclose=3.0e-3, mxiter=100, iter1=50)
    # --write the modflow files
    ml.write_input()
    m = ml.run_model(silent=False)

# --read model zeta
get_stp = [364, 729, 1094, 1459, 364, 729, 1094, 1459]
get_per = [0, 0, 0, 0, 1, 1, 1, 1]
nswi_times = len(get_per)
zetafile = os.path.join(dirs[0], '{}.zta'.format(modelname))
zobj = fu.CellBudgetFile(zetafile)
zeta = []
for kk in zip(get_stp, get_per):
    zeta.append(zobj.get_data(kstpkper=kk, text='      ZETASRF  1')[0])
zeta = np.array(zeta)

# --seawat input - redefine input data that differ from SWI2
nlay_swt = 120
# --mt3d print times
timprs = (np.arange(8) + 1) * 365.
nprs = len(timprs)
# --
ndecay = 4
ibound = np.ones((nlay_swt, nrow, ncol), 'int')
for k in xrange(0, nlay_swt):
    ibound[k, 0, ncol - 1] = -1
bot = np.zeros((nlay_swt, nrow, ncol), np.float)
zall = [0, -20., -40., -60., -80., -100., -120.]
dz = 120. / nlay_swt
tb = np.arange(nlay_swt) * -dz - dz
sconc = np.zeros((nlay_swt, nrow, ncol), np.float)
icbund = np.ones((nlay_swt, nrow, ncol), np.int)
strt = np.zeros((nlay_swt, nrow, ncol), np.float)
pressure = 0.
g = 9.81
z = - dz / 2.  #cell center
for k in xrange(0, nlay_swt):
    for i in xrange(0, ncol):
        bot[k, 0, i] = tb[k]
    if bot[k, 0, 0] >= -100.:
        sconc[k, 0, :] = 0. / 3. * .025 * 1000. / .7143
    else:
        sconc[k, 0, :] = 3. / 3. * .025 * 1000. / .7143
        icbund[k, 0, -1] = -1

    dense = 1000. + 0.7143 * sconc[k, 0, 0]
    pressure += 0.5 * dz * dense * g
    if k > 0:
        z = z - dz
        denseup = 1000. + 0.7143 * sconc[k - 1, 0, 0]
        pressure += 0.5 * dz * denseup * g
    strt[k, 0, :] = z + pressure / dense / g
    #print z, pressure, strt[k, 0, 0], sconc[k, 0, 0]

khb = (0.0000000000256 * 1000. * 9.81 / 0.001) * 60 * 60 * 24
kvb = (0.0000000000100 * 1000. * 9.81 / 0.001) * 60 * 60 * 24
ssb = 1e-5
sszb = 0.2
kh = np.zeros((nlay_swt, nrow, ncol), np.float)
kv = np.zeros((nlay_swt, nrow, ncol), np.float)
ss = np.zeros((nlay_swt, nrow, ncol), np.float)
ssz = np.zeros((nlay_swt, nrow, ncol), np.float)
for k in xrange(0, nlay_swt):
    for i in xrange(0, ncol):
        f = r[i] * 2.0 * math.pi
        kh[k, 0, i] = khb * f
        kv[k, 0, i] = kvb * f
        ss[k, 0, i] = ssb * f
        ssz[k, 0, i] = sszb * f
# wells and ssm data
itype = mt3.Mt3dSsm.itype_dict()
nwell = 1
for k in xrange(0, nlay_swt):
    if bot[k, 0, 0] >= -20.:
        nwell = k + 1
print 'nlay_swt={} dz={} nwell={}'.format(nlay_swt, dz, nwell)
well_data = {}
ssm_data = {}
wellQ = -2400.
wellbtm = -20.0
wellQpm = wellQ / abs(wellbtm)
for ip in xrange(0, nper):
    welllist = np.zeros((nwell, 4), np.float)
    ssmlist = []
    for iw in xrange(0, nwell):
        if ip == 0:
            q = wellQpm * dz
        else:
            q = 0.0
        welllist[iw, 0] = iw
        welllist[iw, 1] = 0
        welllist[iw, 2] = 0
        welllist[iw, 3] = q
        ssmlist.append([iw, 0, 0, 0., itype['WEL']])
    well_data[ip] = welllist.copy()
    ssm_data[ip] = ssmlist

# Define model name for SEAWAT model
modelname = 'swi2ex5_swt'
swtexe_name = 'swt_v4'
# Create the MODFLOW model structure
if not skipRuns:
    ml = mf.Modflow(modelname, version='mf2005', exe_name=swtexe_name, model_ws=dirs[1])
    discret = mf.ModflowDis(ml, nrow=nrow, ncol=ncol, nlay=nlay_swt, delr=delr, delc=delc,
                            top=0, botm=bot,
                            laycbd=0, nper=nper, perlen=perlen, nstp=nstp, steady=True)
    bas = mf.ModflowBas(ml, ibound=ibound, strt=strt)
    lpf = mf.ModflowLpf(ml, hk=kh, vka=kv, ss=ss, sy=ssz, vkcb=0, laytyp=0, layavg=1)
    wel = mf.ModflowWel(ml, stress_period_data=well_data)
    oc = mf.ModflowOc88(ml, save_head_every=365)
    pcg = mf.ModflowPcg(ml, hclose=1.0e-5, rclose=3.0e-3, mxiter=100, iter1=50)
    ml.write_input()
    # Create the basic MT3DMS model structure
    mt = mt3.Mt3dms(modelname, 'nam_mt3dms', ml, model_ws=dirs[1])  # Coupled to modflow model 'mf'
    adv = mt3.Mt3dAdv(mt, mixelm=-1,
                      percel=0.5,
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
    btn = mt3.Mt3dBtn(mt, icbund=icbund, prsity=ssz, ncomp=1, sconc=sconc, ifmtcn=-1,
                      chkmas=False, nprobs=10, nprmas=10, dt0=1.0, ttsmult=1.0,
                      nprs=nprs, timprs=timprs, mxstrn=1e8)
    dsp = mt3.Mt3dDsp(mt, al=0., trpt=1., trpv=1., dmcoef=0.)
    gcg = mt3.Mt3dGcg(mt, mxiter=1, iter1=50, isolve=1, cclose=1e-7)
    ssm = mt3.Mt3dSsm(mt, stress_period_data=ssm_data)
    mt.write_input()
    # Create the SEAWAT model structure
    mswt = swt.Seawat(modelname, 'nam_swt', ml, mt, 
                      exe_name=swtexe_name, model_ws=dirs[1])  # Coupled to modflow model mf and mt3dms model mt
    vdf = swt.SeawatVdf(mswt, iwtable=0, densemin=0, densemax=0, denseref=1000., denseslp=0.7143, firstdt=1e-3)
    mswt.write_input()
    # Run SEAWAT
    m = mswt.run_model(silent=False)

# plot the results
timprs
# read seawat model data
ucnfile = os.path.join(dirs[1], 'MT3D001.UCN')
uobj = fu.UcnFile(ucnfile)
times = uobj.get_times()
print times
conc = np.zeros((len(times), nlay_swt, ncol), np.float)
for idx, tt in enumerate(times):
    c = uobj.get_data(totim=tt)
    for ilay in xrange(0, nlay_swt):
        for jcol in xrange(0, ncol):
            conc[idx, ilay, jcol] = c[ilay, 0, jcol]

# spatial data
# swi2
bot = np.zeros((1, ncol, nlay), np.float)
dz = 100. / float(nlay - 1)
zall = -np.arange(0, 100 + dz, dz)
zall = np.append(zall, -120.)
tb = -np.arange(dz, 100 + dz, dz)
tb = np.append(tb, -120.)
for k in xrange(0, nlay):
    for i in xrange(0, ncol):
        bot[0, i, k] = tb[k]
# seawat
swt_dz = 120. / nlay_swt
swt_tb = np.zeros((nlay_swt), np.float)
zc = -swt_dz / 2.0
for klay in range(0, nlay_swt):
    swt_tb[klay] = zc
    zc -= swt_dz
X, Z = np.meshgrid(x, swt_tb)


# Make figure
fwid, fhgt = 6.5, 6.5
flft, frgt, fbot, ftop = 0.125, 0.95, 0.125, 0.925

eps = 1.0e-3

lc = ['r', 'c', 'g', 'b', 'k']
cfig = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
inc = 1.0e-3

xsf = plt.figure(figsize=(fwid, fhgt), facecolor='w')
xsf.subplots_adjust(wspace=0.25, hspace=0.25, left=flft, right=frgt, bottom=fbot, top=ftop)
# withdrawal and recovery titles
ax = xsf.add_subplot(4, 2, 1)
ax.text(0.0, 1.03, 'Withdrawal', transform=ax.transAxes, va='bottom', ha='left', size='8')
ax = xsf.add_subplot(4, 2, 2)
ax.text(0.0, 1.03, 'Recovery', transform=ax.transAxes, va='bottom', ha='left', size='8')
# dummy items for legend
ax = xsf.add_subplot(4, 2, 1)
ax.plot([-1, -1], [-1, -1], 'bo', markersize=3, markeredgecolor='blue', markerfacecolor='None', label='SWI2 interface')
ax.plot([-1, -1], [-1, -1], color='k', linewidth=0.75, linestyle='solid', label='SEAWAT 50% seawater')
ax.plot([-1, -1], [-1, -1], marker='s', color='k', linewidth=0, linestyle='none', markeredgecolor='w',
        markerfacecolor='0.75', label='SEAWAT 5-95% seawater')
leg = ax.legend(loc='upper left', numpoints=1, ncol=1, labelspacing=0.5, borderaxespad=1, handlelength=3)
leg._drawFrame = False
# data items
for itime in xrange(0, nswi_times):
    zb = np.zeros((ncol), np.float)
    zs = np.zeros((ncol), np.float)
    for icol in range(0, ncol):
        for klay in range(0, nlay):
            #top and bottom of layer
            ztop = float('{0:10.3e}'.format(zall[klay]))
            zbot = float('{0:10.3e}'.format(zall[klay + 1]))
            #fresh-salt zeta surface
            zt = zeta[itime, klay, 0, icol]
            if ( ztop - zt ) > eps:
                zs[icol] = zt
    if itime < ndecay:
        ic = itime
        isp = ic * 2 + 1
        ax = xsf.add_subplot(4, 2, isp)
    else:
        ic = itime - ndecay
        isp = ( ic * 2 ) + 2
        ax = xsf.add_subplot(4, 2, isp)
    # figure title
    ax.text(-0.15, 1.025, cfig[itime], transform=ax.transAxes, va='center', ha='center', size='8')

    # swi2
    ax.plot(x, zs, 'bo', markersize=3, markeredgecolor='blue', markerfacecolor='None', label='_None')

    # seawat
    sc = ax.contour(X, Z, conc[itime, :, :], levels=[17.5], colors='k', linestyles='solid', linewidths=0.75, zorder=30)
    cc = ax.contourf(X, Z, conc[itime, :, :], levels=[0.0, 1.75, 33.250], colors=['w', '0.75', 'w'])
    # set graph limits
    ax.set_xlim(0, 500)
    ax.set_ylim(-100, -65)
    if itime < ndecay:
        ax.set_ylabel('Elevation, in meters')


# x labels
ax = xsf.add_subplot(4, 2, 7)
ax.set_xlabel('Horizontal distance, in meters')
ax = xsf.add_subplot(4, 2, 8)
ax.set_xlabel('Horizontal distance, in meters')

# simulation time titles
for itime in range(0, nswi_times):
    if itime < ndecay:
        ic = itime
        isp = ic * 2 + 1
        ax = xsf.add_subplot(4, 2, isp)
    else:
        ic = itime - ndecay
        isp = ( ic * 2 ) + 2
        ax = xsf.add_subplot(4, 2, isp)
    iyr = itime + 1
    if iyr > 1:
        ctxt = '{} years'.format(iyr)
    else:
        ctxt = '{} year'.format(iyr)
    ax.text(0.95, 0.925, ctxt, transform=ax.transAxes, va='top', ha='right', size='8')

outfig = 'Figure11_swi2ex5.{0}'.format(fext)
xsf.savefig(outfig, dpi=300)
print 'created...', outfig
