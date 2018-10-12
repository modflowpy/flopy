from __future__ import print_function

import os
import sys
import math

import numpy as np

import flopy

import matplotlib.pyplot as plt

# --modify default matplotlib settings
updates = {'font.family': ['Univers 57 Condensed', 'Arial'],
           'mathtext.default': 'regular',
           'pdf.compression': 0,
           'pdf.fonttype': 42,
           'legend.fontsize': 7,
           'axes.labelsize': 8,
           'xtick.labelsize': 7,
           'ytick.labelsize': 7}
plt.rcParams.update(updates)


def run():
    workspace = 'swiex1'

    cleanFiles = False
    fext = 'png'
    narg = len(sys.argv)
    iarg = 0
    if narg > 1:
        while iarg < narg - 1:
            iarg += 1
            basearg = sys.argv[iarg].lower()
            if basearg == '--clean':
                cleanFiles = True
            elif basearg == '--pdf':
                fext = 'pdf'

    if cleanFiles:
        print('cleaning all files')
        print('excluding *.py files')
        files = os.listdir(workspace)
        for f in files:
            if os.path.isdir(f):
                continue
            if '.py' != os.path.splitext(f)[1].lower():
                print('  removing...{}'.format(os.path.basename(f)))
                os.remove(os.path.join(workspace, f))
        return 1

    modelname = 'swiex1'
    exe_name = 'mf2005'

    nlay = 1
    nrow = 1
    ncol = 50

    delr = 5.
    delc = 1.

    ibound = np.ones((nrow, ncol), np.int)
    ibound[0, -1] = -1

    # create initial zeta surface
    z = np.zeros((nrow, ncol), np.float)
    z[0, 16:24] = np.arange(-2.5, -40, -5)
    z[0, 24:] = -40
    z = [z]
    # create isource for SWI2
    isource = np.ones((nrow, ncol), np.int)
    isource[0, 0] = 2

    ocdict = {}
    for idx in range(49, 200, 50):
        key = (0, idx)
        ocdict[key] = ['save head', 'save budget']
        key = (0, idx + 1)
        ocdict[key] = []

    # create flopy modflow object
    ml = flopy.modflow.Modflow(modelname, version='mf2005', exe_name=exe_name,
                               model_ws=workspace)
    # create flopy modflow package objects
    discret = flopy.modflow.ModflowDis(ml, nlay=nlay, nrow=nrow, ncol=ncol,
                                       delr=delr, delc=delc,
                                       top=0, botm=[-40.0],
                                       perlen=400, nstp=200)
    bas = flopy.modflow.ModflowBas(ml, ibound=ibound, strt=0.0)
    lpf = flopy.modflow.ModflowLpf(ml, hk=2., vka=2.0, vkcb=0, laytyp=0,
                                   layavg=0)
    wel = flopy.modflow.ModflowWel(ml, stress_period_data={0: [(0, 0, 0, 1)]})
    swi = flopy.modflow.ModflowSwi2(ml, iswizt=55, npln=1, istrat=1,
                                    toeslope=0.2, tipslope=0.2, nu=[0, 0.025],
                                    zeta=z, ssz=0.2, isource=isource,
                                    nsolver=1)
    oc = flopy.modflow.ModflowOc(ml, stress_period_data=ocdict)
    pcg = flopy.modflow.ModflowPcg(ml)
    # create model files
    ml.write_input()
    # run the model
    m = ml.run_model(silent=False)
    # read model heads
    headfile = os.path.join(workspace, '{}.hds'.format(modelname))
    hdobj = flopy.utils.HeadFile(headfile)
    head = hdobj.get_alldata()
    head = np.array(head)
    # read model zeta
    zetafile = os.path.join(workspace, '{}.zta'.format(modelname))
    zobj = flopy.utils.CellBudgetFile(zetafile)
    zkstpkper = zobj.get_kstpkper()
    zeta = []
    for kk in zkstpkper:
        zeta.append(zobj.get_data(kstpkper=kk, text='      ZETASRF  1')[0])
    zeta = np.array(zeta)

    x = np.arange(0.5 * delr, ncol * delr, delr)

    # Wilson and Sa Da Costa
    k = 2.0
    n = 0.2
    nu = 0.025
    H = 40.0
    tzero = H * n / (k * nu) / 4.0
    Ltoe = np.zeros(4)
    v = 0.125
    t = np.arange(100, 500, 100)

    fwid = 7.00
    fhgt = 3.50
    flft = 0.125
    frgt = 0.95
    fbot = 0.125
    ftop = 0.925

    fig = plt.figure(figsize=(fwid, fhgt), facecolor='w')
    fig.subplots_adjust(wspace=0.25, hspace=0.25, left=flft, right=frgt,
                        bottom=fbot, top=ftop)

    ax = fig.add_subplot(211)
    ax.text(-0.075, 1.05, 'A', transform=ax.transAxes, va='center',
            ha='center',
            size='8')
    ax.plot([80, 120], [0, -40], 'k')
    ax.set_xlim(0, 250)
    ax.set_ylim(-40, 0)
    ax.set_yticks(np.arange(-40, 1, 10))
    ax.text(50, -10, 'salt')
    ax.text(130, -10, 'fresh')
    a = ax.annotate("", xy=(50, -25), xytext=(30, -25),
                    arrowprops=dict(arrowstyle='->', fc='k'))
    ax.text(40, -22, 'groundwater flow velocity=0.125 m/d', ha='center',
            size=7)
    ax.set_ylabel('Elevation, in meters')

    ax = fig.add_subplot(212)
    ax.text(-0.075, 1.05, 'B', transform=ax.transAxes, va='center',
            ha='center',
            size='8')

    for i in range(4):
        Ltoe[i] = H * math.sqrt(k * nu * (t[i] + tzero) / n / H)
        ax.plot([100 - Ltoe[i] + v * t[i], 100 + Ltoe[i] + v * t[i]], [0, -40],
                'k', label='_None')

    for i in range(4):
        zi = zeta[i, 0, 0, :]
        p = (zi < 0) & (zi > -39.9)
        ax.plot(x[p], zeta[i, 0, 0, p], 'bo',
                markersize=3, markeredgecolor='blue', markerfacecolor='None',
                label='_None')
        ipos = 0
        for jdx, t in enumerate(zeta[i, 0, 0, :]):
            if t > -39.9:
                ipos = jdx
        ax.text(x[ipos], -37.75, '{0} days'.format(((i + 1) * 100)), size=5,
                ha='left', va='center')

    # fake items for labels
    ax.plot([-100., -100], [-100., -100], 'k', label='Analytical solution')
    ax.plot([-100., -100], [-100., -100], 'bo', markersize=3,
            markeredgecolor='blue', markerfacecolor='None', label='SWI2')
    # legend
    leg = ax.legend(loc='upper right', numpoints=1)
    leg._drawFrame = False
    # axes
    ax.set_xlim(0, 250)
    ax.set_ylim(-40, 0)
    ax.set_yticks(np.arange(-40, 1, 10))
    a = ax.annotate("", xy=(50, -25), xytext=(30, -25),
                    arrowprops=dict(arrowstyle='->', fc='k'))
    ax.text(40, -22, 'groundwater flow velocity=0.125 m/d', ha='center',
            size=7)
    ax.set_ylabel('Elevation, in meters')
    ax.set_xlabel('Horizontal distance, in meters')

    outfig = os.path.join(workspace, 'Figure06_swi2ex1.{0}'.format(fext))
    fig.savefig(outfig, dpi=300)
    print('created...', outfig)

    return 0


if __name__ == "__main__":
    success = run()
    sys.exit(success)
