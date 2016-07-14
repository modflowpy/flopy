"""
test UZF package
"""
import sys
sys.path.insert(0, '..')
import os
import flopy
import numpy as np

cpth = os.path.join('temp')

def test_load():

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
    assert np.sum(uzf.extwc[0].array)/uzf.extwc[0].cnstnt - 176.0 < 1e4
    assert np.sum(uzf.finf[0].array)/uzf.finf[0].cnstnt - 339.0 < 1e4






    uzf.write_file()
    j=2

if __name__ == '__main__':
    test_load()
    #test_make_package()