# Test reference class
import matplotlib
matplotlib.use('agg')


def test_dis_reference():
    import os
    import numpy as np
    import flopy.modflow as fmf
    ml = fmf.Modflow(modelname="dis_test", model_ws="temp")
    assert isinstance(ml, fmf.Modflow)
    perlen = np.arange(1, 20, 1)
    nstp = np.flipud(perlen) + 3
    tsmult = 1.2
    nlay = 10
    nrow, ncol = 50, 40
    botm = np.arange(0, -100, -10)
    hk = np.random.random((nrow, ncol))
    dis = fmf.ModflowDis(ml, delr=100.0, delc=100.0,
                         nrow=nrow, ncol=ncol, nlay=nlay,
                         nper=perlen.shape[0], perlen=perlen,
                         nstp=nstp, tsmult=tsmult,
                         top=10, botm=botm, steady=False, rotation=45,
                         xul=999.9,yul=-999.9,proj4_str="some_proj4_str")
    ml.write_input()
    ml1 = fmf.Modflow.load(ml.namefile,model_ws=ml.model_ws)
    assert ml1.dis.sr == ml.dis.sr


def test_binaryfile_reference():
    import os
    import flopy

    h = flopy.utils.HeadFile(os.path.join('..', 'examples', 'data', 'freyberg', 'freyberg.githds'))
    assert isinstance(h, flopy.utils.HeadFile)
    h.sr.xul = 1000.0
    h.sr.yul = 200.0
    h.sr.rotation = 15.0
    assert isinstance(h.plot(), matplotlib.axes.Axes)
    return

def test_formattedfile_reference():
    import os
    import flopy
    h = flopy.utils.FormattedHeadFile(os.path.join('..', 'examples', 'data', 'mf2005_test', 'test1tr.githds'))
    assert isinstance(h, flopy.utils.FormattedHeadFile)
    h.sr.xul = 1000.0
    h.sr.yul = 200.0
    h.sr.rotation = 15.0
    assert isinstance(h.plot(masked_values=[6999.000]), matplotlib.axes.Axes)
    return


def test_mflist_reference():
    import os
    import numpy as np
    import shapefile
    import flopy.modflow as fmf

    # model_ws = os.path.join('..', 'data', 'freyberg')
    # ml = fmf.Modflow.load('freyberg.nam', model_ws=model_ws)
    # make the model
    ml = fmf.Modflow()
    assert isinstance(ml, fmf.Modflow)
    perlen = np.arange(1, 20, 1)
    nstp = np.flipud(perlen) + 3
    tsmult = 1.2
    nlay = 10
    nrow, ncol = 50, 40
    botm = np.arange(0, -100, -10)
    hk = np.random.random((nrow, ncol))
    dis = fmf.ModflowDis(ml, delr=100.0, delc=100.0,
                         nrow=nrow, ncol=ncol, nlay=nlay,
                         nper=perlen.shape[0], perlen=perlen,
                         nstp=nstp, tsmult=tsmult,
                         top=10, botm=botm, steady=False, rotation=45)
    assert isinstance(dis, fmf.ModflowDis)
    lpf = fmf.ModflowLpf(ml, hk=hk, vka=10.0, laytyp=1)
    assert isinstance(lpf, fmf.ModflowLpf)
    pcg = fmf.ModflowPcg(ml)
    assert isinstance(pcg, fmf.ModflowPcg)
    oc = fmf.ModflowOc(ml)
    assert isinstance(oc, fmf.ModflowOc)
    ibound = np.ones((nrow, ncol))
    ibound[:, 0] = -1
    ibound[25:30, 30:39] = 0
    bas = fmf.ModflowBas(ml, strt=5.0, ibound=ibound)
    assert isinstance(bas, fmf.ModflowBas)
    rch = fmf.ModflowRch(ml, rech={0: 0.00001, 5: 0.0001, 6: 0.0})
    assert isinstance(rch, fmf.ModflowRch)
    wel_dict = {}
    wel_data = [[9, 25, 20, -200], [0, 0, 0, -400], [5, 20, 32, 500]]
    wel_dict[0] = wel_data
    wel_data2 = [[45, 20, 200], [9, 49, 39, 400], [5, 20, 32, 500]]
    wel_dict[10] = wel_data2
    wel = fmf.ModflowWel(ml, stress_period_data={0: wel_data})
    assert isinstance(wel, fmf.ModflowWel)
    ghb_dict = {0: [1, 10, 10, 400, 300]}
    ghb = fmf.ModflowGhb(ml, stress_period_data=ghb_dict)
    assert isinstance(ghb, fmf.ModflowGhb)

    test = os.path.join('temp', 'test3.shp')
    ml.export(test, kper=0)
    shp = shapefile.Reader(test)
    assert shp.numRecords == nrow * ncol


if __name__ == '__main__':
    test_dis_reference()
    #test_mflist_reference()
    #test_formattedfile_reference()
    #test_binaryfile_reference()
