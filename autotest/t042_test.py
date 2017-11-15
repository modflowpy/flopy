"""
Test postprocessing utilties
"""

import sys
sys.path.append('/Users/aleaf/Documents/GitHub/flopy3')
import numpy as np
import flopy
from flopy.utils.postprocessing import get_transmissivities, get_water_table, get_gradients, get_saturated_thickness

mf = flopy.modflow


def test_get_transmissivities():
    sctop = [-.25, .5, 1.7, 1.5, 3., 2.5, 3., -10.]
    scbot = [-1., -.5, 1.2, 0.5, 1.5, -.2, 2.5, -11.]
    heads = np.array([[1., 2.0, 2.05, 3., 4., 2.5, 2.5, 2.5],
                      [1.1, 2.1, 2.2, 2., 3.5, 3., 3., 3.],
                      [1.2, 2.3, 2.4, 0.6, 3.4, 3.2, 3.2, 3.2]
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
    assert (T - np.array([[0., 0, 0., 0., 0.2, 0.2, 2., 0.],
                          [0., 0., 1., 1., 1., 2., 0., 0.],
                          [2., 1., 0., 0.2, 0., 2., 0., 2.]])).sum() < 1e-3

    # test without specifying open intervals
    T = get_transmissivities(heads, m, r=r, c=c)
    assert (T - np.array([[0., 0., 0.1, 0.2, 0.2, 0.2, .2, .2],
                          [0.2, 2., 2., 2., 2., 2., 2., 2.],
                          [2., 2., 2., 1.2, 2., 2., 2., 2.]])).sum() < 1e-3

def test_get_water_table():
    nodata = -9999.
    hds = np.ones ((3, 3, 3), dtype=float) * nodata
    hds[-1, :, :] = 2.
    hds[1, 1, 1] = 1.
    wt = get_water_table(hds, nodata=nodata)
    assert wt.shape == (3, 3)
    assert wt[1, 1] == 1.
    assert np.sum(wt) == 17.

    hds2 = np.array([hds, hds])
    wt = get_water_table(hds2, nodata=nodata)
    assert wt.shape == (2, 3, 3)
    assert np.sum(wt[:, 1, 1]) == 2.
    assert np.sum(wt) == 34.

    wt = get_water_table(hds2, nodata=nodata, per_idx=0)
    assert wt.shape == (3, 3)
    assert wt[1, 1] == 1.
    assert np.sum(wt) == 17.

def test_get_sat_thickness_gradients():
    nodata = -9999.
    hds = np.ones ((3, 3, 3), dtype=float) * nodata
    hds[1, :, :] = 2.4
    hds[0, 1, 1] = 3.2
    hds[2, :, :] = 2.5
    hds[1, 1, 1] = 3.
    hds[2, 1, 1] = 2.6

    nl, nr, nc = hds.shape
    botm = np.ones((nl, nr, nc), dtype=float)
    top = np.ones((nr, nc), dtype=float) * 4.
    botm[0, :, :] = 3.
    botm[1, :, :] = 2.

    m = mf.Modflow('junk', version='mfnwt', model_ws='temp')
    dis = mf.ModflowDis(m, nlay=nl, nrow=nr, ncol=nc, botm=botm, top=top)

    grad = get_gradients(hds, m, nodata=nodata)
    dh = np.diff(hds[:, 1, 1])
    dz = np.array([-.7, -1.])
    assert np.abs(dh/dz - grad[:, 1, 1]).sum() < 1e-6
    dh = np.diff(hds[:, 1, 0])
    dz = np.array([np.nan, -.9])
    assert np.nansum(np.abs(dh / dz - grad[:, 1, 0])) < 1e-6

    sat_thick = get_saturated_thickness(hds, m, nodata)
    assert np.abs(np.sum(sat_thick[:, 1, 1] - np.array([0.2, 1., 1.]))) < 1e-6

if __name__ == '__main__':
    #test_get_transmissivities()
    #test_get_water_table()
    test_get_sat_thickness_gradients()