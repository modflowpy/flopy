import os
import numpy as np
import flopy
from flopy.utils.lgrutil import Lgr


tpth = os.path.join('temp', 't063')
# make the directory if it does not exist
if not os.path.isdir(tpth):
    os.makedirs(tpth)


def test_lgrutil():
    nlayp = 5
    nrowp = 5
    ncolp = 5
    delrp = 100.
    delcp = 100.
    topp = 100.
    botmp = [-100, -200, -300, -400, -500]
    idomainp = np.ones((nlayp, nrowp, ncolp), dtype=np.int)
    idomainp[0:2, 1:4, 1:4] = 0
    ncpp = 3
    ncppl = [1, 1, 0, 0, 0]

    lgr = Lgr(nlayp, nrowp, ncolp, delrp, delcp, topp, botmp,
              idomainp, ncpp=ncpp, ncppl=ncppl, xllp=100., yllp=100.)

    # child shape
    assert lgr.get_shape() == (2, 9, 9), 'child shape is not (2, 9, 9)'

    # child delr/delc
    delr, delc = lgr.get_delr_delc()
    assert np.allclose(delr, delrp / ncpp), 'child delr not correct'
    assert np.allclose(delc, delcp / ncpp), 'child delc not correct'

    # child idomain
    idomain = lgr.get_idomain()
    assert idomain.min() == idomain.max() == 1
    assert idomain.shape == (2, 9, 9)

    # replicated parent array
    ap = np.arange(nrowp * ncolp).reshape((nrowp, ncolp))
    ac = lgr.get_replicated_parent_array(ap)
    assert ac[0, 0] == 6
    assert ac[-1, -1] == 18

    # top/bottom
    top, botm = lgr.get_top_botm()
    assert top.shape == (9, 9)
    assert botm.shape == (2, 9, 9)
    assert top.min() == top.max() == 100.
    assert botm.min() == botm.max() == 0.

    # exchange data
    exchange_data = lgr.get_exchange_data(angldegx=True, cdist=True)
    ans1 = [(0, 1, 0), (0, 0, 0), 1, 50.0, 16.666666666666668,
            33.333333333333336, 0.0, 354.33819375782156]
    ans2 = [(1, 4, 3), (1, 8, 8), 1, 50.0, 16.666666666666668,
            33.333333333333336, 90.0, 354.3381937578216]
    assert exchange_data[0] == ans1
    assert exchange_data[-1] == ans2
    assert len(exchange_data) == 72

    # list of parent cells connected to a child cell
    assert lgr.get_parent_connections(0, 0, 0) == [((0, 1, 0), -1),
                                                   ((0, 0, 1), 2)]
    assert lgr.get_parent_connections(1, 8, 8) == [((1, 3, 4), 1),
                                                   ((1, 4, 3), -2)]

    return


if __name__ == '__main__':
    test_lgrutil()

