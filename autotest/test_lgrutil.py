import numpy as np

from flopy.utils.lgrutil import Lgr


def test_lgrutil():
    nlayp = 5
    nrowp = 5
    ncolp = 5
    delrp = 100.0
    delcp = 100.0
    topp = 100.0
    botmp = [-100, -200, -300, -400, -500]
    idomainp = np.ones((nlayp, nrowp, ncolp), dtype=int)
    idomainp[0:2, 1:4, 1:4] = 0
    ncpp = 3
    ncppl = [1, 1, 0, 0, 0]

    lgr = Lgr(
        nlayp,
        nrowp,
        ncolp,
        delrp,
        delcp,
        topp,
        botmp,
        idomainp,
        ncpp=ncpp,
        ncppl=ncppl,
        xllp=100.0,
        yllp=100.0,
    )

    # child shape
    assert lgr.get_shape() == (2, 9, 9), "child shape is not (2, 9, 9)"

    # child delr/delc
    delr, delc = lgr.get_delr_delc()
    assert np.allclose(delr, delrp / ncpp), "child delr not correct"
    assert np.allclose(delc, delcp / ncpp), "child delc not correct"

    # child idomain
    idomain = lgr.get_idomain()
    assert idomain.min() == idomain.max() == 1
    assert idomain.shape == (2, 9, 9)

    # replicated parent array
    ap = np.arange(nrowp * ncolp).reshape((nrowp, ncolp))
    ac = lgr.get_replicated_parent_array(ap)
    assert ac[0, 0] == 6
    assert ac[-1, -1] == 18

    # child top/bottom
    topc, botmc = lgr.get_top_botm()
    assert topc.shape == (9, 9)
    assert botmc.shape == (2, 9, 9)
    assert topc.min() == topc.max() == 100.0
    errmsg = f"{botmc[:, 0, 0]} /= {np.array(botmp[:2])}"
    assert np.allclose(botmc[:, 0, 0], np.array(botmp[:2])), errmsg

    # exchange data
    exchange_data = lgr.get_exchange_data(angldegx=True, cdist=True)

    ans1 = [
        (0, 1, 0),
        (0, 0, 0),
        1,
        50.0,
        16.666666666666668,
        33.333333333333336,
        0.0,
        354.33819375782156,
    ]
    errmsg = f"{ans1} /= {exchange_data[0]}"
    assert exchange_data[0] == ans1, errmsg

    ans2 = [
        (2, 3, 3),
        (1, 8, 8),
        0,
        50.0,
        50,
        1111.1111111111113,
        180.0,
        100.0,
    ]
    errmsg = f"{ans2} /= {exchange_data[-1]}"
    assert exchange_data[-1] == ans2, errmsg

    errmsg = "exchanges should be 71 horizontal plus 81 vertical"
    assert len(exchange_data) == 72 + 81, errmsg

    # list of parent cells connected to a child cell
    assert lgr.get_parent_connections(0, 0, 0) == [
        ((0, 1, 0), -1),
        ((0, 0, 1), 2),
    ]
    assert lgr.get_parent_connections(1, 8, 8) == [
        ((1, 3, 4), 1),
        ((1, 4, 3), -2),
        ((2, 3, 3), -3),
    ]


def test_lgrutil2():
    # Define parent grid information
    xoffp = 0.0
    yoffp = 0.0
    nlayp = 1
    nrowp = 5
    ncolp = 5
    dx = 100.0
    dy = 100.0
    dz = 100.0
    delrp = dx * np.array([1.0, 0.75, 0.5, 0.75, 1.0], dtype=float)
    delcp = dy * np.array([1.0, 0.75, 0.5, 0.75, 1.0], dtype=float)
    topp = dz * np.ones((nrowp, ncolp), dtype=float)
    botmp = np.empty((nlayp, nrowp, ncolp), dtype=float)
    for k in range(nlayp):
        botmp[k] = -(k + 1) * dz

    # Define relation of child to parent
    idomainp = np.ones((nlayp, nrowp, ncolp), dtype=int)
    idomainp[:, 1:4, 1:4] = 0
    ncpp = 3
    ncppl = nlayp * [1]

    lgr = Lgr(
        nlayp,
        nrowp,
        ncolp,
        delrp,
        delcp,
        topp,
        botmp,
        idomainp,
        ncpp=ncpp,
        ncppl=ncppl,
        xllp=xoffp,
        yllp=yoffp,
    )

    # check to make sure child delr and delc are calculated correctly for
    # the case where the parent grid has variable row and column spacings
    answer = [
        25.0,
        25.0,
        25.0,
        50.0 / 3.0,
        50.0 / 3.0,
        50.0 / 3.0,
        25.0,
        25.0,
        25.0,
    ]
    assert np.allclose(lgr.delr, answer), f"{lgr.delr} /= {answer}"
    assert np.allclose(lgr.delc, answer), f"{lgr.delc} /= {answer}"
