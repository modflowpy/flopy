# Test reference class
import os
import numpy as np
try:
    import matplotlib
    if os.getenv('TRAVIS'):  # are we running https://travis-ci.org/ automated tests ?
        matplotlib.use('Agg')  # Force matplotlib  not to use any Xwindows backend
except:
    matplotlib = None

import flopy
import shapefile


cpth = os.path.join('temp', 't006')
# make the directory if it does not exist
if not os.path.isdir(cpth):
    os.makedirs(cpth)


def test_binaryfile_reference():
    h = flopy.utils.HeadFile(
        os.path.join('..', 'examples', 'data', 'freyberg', 'freyberg.githds'))
    assert isinstance(h, flopy.utils.HeadFile)
    h.sr.xul = 1000.0
    h.sr.yul = 200.0
    h.sr.rotation = 15.0
    if matplotlib is not None:
        assert isinstance(h.plot(), matplotlib.axes.Axes)
    return


def test_formattedfile_reference():
    h = flopy.utils.FormattedHeadFile(
        os.path.join('..', 'examples', 'data', 'mf2005_test',
                     'test1tr.githds'))
    assert isinstance(h, flopy.utils.FormattedHeadFile)
    h.sr.xul = 1000.0
    h.sr.yul = 200.0
    h.sr.rotation = 15.0
    if matplotlib is not None:
        assert isinstance(h.plot(masked_values=[6999.000]), matplotlib.axes.Axes)
    return


def test_mflist_reference():
    # make the model
    ml = flopy.modflow.Modflow()
    assert isinstance(ml, flopy.modflow.Modflow)
    perlen = np.arange(1, 20, 1)
    nstp = np.flipud(perlen) + 3
    tsmult = 1.2
    nlay = 10
    nrow, ncol = 50, 40
    botm = np.arange(0, -100, -10)
    hk = np.random.random((nrow, ncol))
    dis = flopy.modflow.ModflowDis(ml, delr=100.0, delc=100.0,
                                   nrow=nrow, ncol=ncol, nlay=nlay,
                                   nper=perlen.shape[0], perlen=perlen,
                                   nstp=nstp, tsmult=tsmult,
                                   top=10, botm=botm, steady=False)
    assert isinstance(dis, flopy.modflow.ModflowDis)
    lpf = flopy.modflow.ModflowLpf(ml, hk=hk, vka=10.0, laytyp=1)
    assert isinstance(lpf, flopy.modflow.ModflowLpf)
    pcg = flopy.modflow.ModflowPcg(ml)
    assert isinstance(pcg, flopy.modflow.ModflowPcg)
    oc = flopy.modflow.ModflowOc(ml)
    assert isinstance(oc, flopy.modflow.ModflowOc)
    ibound = np.ones((nrow, ncol))
    ibound[:, 0] = -1
    ibound[25:30, 30:39] = 0
    bas = flopy.modflow.ModflowBas(ml, strt=5.0, ibound=ibound)
    assert isinstance(bas, flopy.modflow.ModflowBas)
    rch = flopy.modflow.ModflowRch(ml, rech={0: 0.00001, 5: 0.0001, 6: 0.0})
    assert isinstance(rch, flopy.modflow.ModflowRch)
    wel_dict = {}
    wel_data = [[9, 25, 20, -200], [0, 0, 0, -400], [5, 20, 32, 500]]
    wel_dict[0] = wel_data
    wel_data2 = [[45, 20, 200], [9, 49, 39, 400], [5, 20, 32, 500]]
    wel_dict[10] = wel_data2
    wel = flopy.modflow.ModflowWel(ml, stress_period_data={0: wel_data})
    assert isinstance(wel, flopy.modflow.ModflowWel)
    ghb_dict = {0: [1, 10, 10, 400, 300]}
    ghb = flopy.modflow.ModflowGhb(ml, stress_period_data=ghb_dict)
    assert isinstance(ghb, flopy.modflow.ModflowGhb)

    test = os.path.join(cpth, 'test3.shp')
    ml.export(test, kper=0)
    shp = shapefile.Reader(test)
    assert shp.numRecords == nrow * ncol


if __name__ == '__main__':
    # test_mbase_sr()
    # test_sr()
    # test_dis_reference()
    test_mflist_reference()
    # test_formattedfile_reference()
    # test_binaryfile_reference()
