"""
test UZF package
"""
import sys
sys.path.insert(0, '..')
import os
import shutil
import glob
import numpy as np
import flopy
from flopy.utils.util_array import Util2d
import numpy as np

cpth = os.path.join('temp/uzf')
if not os.path.isdir(cpth):
    os.mkdirs(cpth)

def test_load_and_write():

    # load in the test problem
    m = flopy.modflow.Modflow('UZFtest2', model_ws=cpth, verbose=True)
    m.model_ws = 'temp'
    path = os.path.join('..', 'examples', 'data', 'mf2005_test')
    dis = flopy.modflow.ModflowDis.load(path + '/UZFtest2.dis', m)
    uzf = flopy.modflow.ModflowUzf1.load(path + '/UZFtest2.uzf', m)
    assert np.sum(uzf.iuzfbnd.array) == 116
    assert np.array_equal(np.unique(uzf.irunbnd.array), np.arange(9))
    assert np.abs(np.sum(uzf.vks.array)/uzf.vks.cnstnt - 116.) < 1e-5
    assert uzf.eps._Util2d__value == 3.5
    assert np.abs(uzf.thts._Util2d__value - .30) < 1e-5
    assert np.abs(np.sum(uzf.extwc[0].array) / uzf.extwc[0].cnstnt - 176.0) < 1e4
    for per in [0, 1]:
        assert np.abs(uzf.pet[per]._Util2d__value - 5e-8) < 1e-10
    for per in range(m.nper):
        assert np.abs(np.sum(uzf.finf[per].array) / uzf.finf[per].cnstnt - 339.0) < 1e4
        assert True
    m.model_ws = cpth
    uzf.write_file()
    m2 = flopy.modflow.Modflow('UZFtest2_2', model_ws=cpth)
    dis = flopy.modflow.ModflowDis(nrow=m.nrow, ncol=m.ncol, model=m2)
    uzf2 = flopy.modflow.ModflowUzf1.load(cpth + '/UZFtest2.uzf', m2)
    attrs = dir(uzf)
    for attr in attrs:
        a1 = uzf.__getattribute__(attr)
        if isinstance(a1, Util2d):
            a2 = uzf2.__getattribute__(attr)
            assert a1 == a2

def test_create():


    gpth = os.path.join('..', 'examples', 'data', 'mf2005_test', 'UZFtest2.*')
    for f in glob.glob(gpth):
        shutil.copy(f, cpth)
    m = flopy.modflow.Modflow.load('UZFtest2.nam', version='mf2005', exe_name='mf2005',
                                   model_ws=cpth, load_only=['ghb', 'dis', 'bas6', 'oc', 'sip', 'lpf', 'sfr'])

    datpth = os.path.join('..', 'examples', 'data', 'uzf_examples')
    irnbndpth = os.path.join(datpth, 'irunbnd.dat')
    irunbnd = np.loadtxt(irnbndpth)

    vksbndpth = os.path.join(datpth, 'vks.dat')
    vks = np.loadtxt(vksbndpth)

    finf = np.loadtxt(os.path.join(datpth, 'finf.dat'))
    finf = np.reshape(finf, (m.nper, m.nrow, m.ncol))

    uzgag = {-68: [-68],
             65: [3, 6, 65, 1],
             66: [6, 3, 66, 2],
             67: [10, 5, 67, 3]}

    uzf = flopy.modflow.ModflowUzf1(m,
                                    nuztop=1, iuzfopt=1, irunflg=1, ietflg=1,
                                    iuzfcb1=0,
                                    iuzfcb2=61,  # binary output of recharge and groundwater discharge
                                    ntrail2=25, nsets=20, nuzgag=4,
                                    surfdep=1.0, uzgag=uzgag,
                                    iuzfbnd=m.bas6.ibound.array,
                                    irunbnd=irunbnd,
                                    vks=vks,
                                    finf=finf,
                                    eps=3.5,
                                    thts=0.35,
                                    pet=5.000000E-08,
                                    extdp=15.,
                                    extwc=0.1
                                    )
    m.write_input()

if __name__ == '__main__':
    #test_load_and_write()
    test_create()