"""
Test postprocessing utilties
"""

import sys
sys.path.append('/Users/aleaf/Documents/GitHub/flopy3')
import numpy as np
import matplotlib.pyplot as plt
import flopy
import flopy.utils.binaryfile as bf
from flopy.utils.postprocessing import get_transmissivities, get_water_table

mf = flopy.modflow


def test_get_transmissivities():
    sctop = [-.25, .5, 1.7, 1.5, 3., 2.5]
    scbot = [-1., -.5, 1.2, 0.5, 1.5, -.2]
    heads = np.array([[1., 2.0, 2.05, 3., 4., 2.5],
                      [1.1, 2.1, 2.2, 2., 3.5, 3.],
                      [1.2, 2.3, 2.4, 0.6, 3.4, 3.2]
            ])
    nl, nr = heads.shape
    nc = nr
    botm = np.ones((nl, nr, nc), dtype=float)
    top = np.ones((nr, nc), dtype=float) * 2.1
    hk = np.ones((nl, nr, nc), dtype=float) * 2.
    for i in range(nl):
        botm[nl-i-1, :, :] = i

    m = mf.Modflow('junk', version='mfnwt', model_ws='temp')
    dis = mf.ModflowDis(m, nlay=nl, nrow=nr, ncol=nc, botm=botm, top=top)
    upw = mf.ModflowUpw(m, hk=hk)

    # test with open intervals
    r, c = np.arange(nr), np.arange(nc)
    T = get_transmissivities(heads, m, r=r, c=c, sctop=sctop, scbot=scbot)
    assert (T - np.array([[0., 0, 0., 0., 0.2, 0.2],
                          [0., 0., 1., 1., 1., 2.],
                          [0., 1., 0., 0.2, 0., 2.]])).sum() < 1e-3

    # test without specifying open intervals
    T = get_transmissivities(heads, m, r=r, c=c)
    assert (T - np.array([[0., 0., 0.1, 0.2, 0.2, 0.2],
                          [0.2, 2., 2., 2., 2., 2.],
                          [2., 2., 2., 1.2, 2., 2.]])).sum() < 1e-3

def test_get_water_table():
    nodata = -9999.
    hds = np.ones ((3, 3, 3), dtype=float) * nodata
    hds[-1, :, :] = 2.
    hds[1, 1, 1] = 1.
    wt = get_water_table(hds, nodata=nodata)
    assert wt.shape == (3, 3)
    assert wt[1, 1] == 1.
    assert np.sum(wt) == 17.

    hds = np.array([hds, hds])
    wt = get_water_table(hds, nodata)
    assert wt.shape == (2, 3, 3)
    assert np.sum(wt[:, 1, 1]) == 2.
    assert np.sum(wt) == 34.



if __name__ == '__main__':
    #test_get_transmissivities()
    test_get_water_table()
