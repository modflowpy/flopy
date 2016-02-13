# Test reference class
import matplotlib
matplotlib.use('agg')


def test_sr():
    import numpy as np
    import flopy
    sr1 = flopy.utils.SpatialReference(1.0,1.0,1)
    print(sr1.xcentergrid)
    sr1.delr = np.ones((100))
    print(sr1.xcentergrid)
    sr1.rotation = 2.1
    print(sr1.xcentergrid)

    sr1.reset(delr=np.ones((20)),rotation=-5.0,lenuni=2,xul=200.0)
    print(sr1.xcentergrid)

    sr2 = flopy.utils.SpatialReference([1.0],[1.0],1)
    print(sr2.nrow,sr2.ncol)

    sr3 = flopy.utils.SpatialReference(np.ones((10)),np.ones((10)),1)
    print(sr3.nrow,sr3.ncol)


def test_mbase_sr():
    import numpy as np
    import flopy

    ml = flopy.modflow.Modflow(modelname="test",xul=1000.0,yul=50.0,
                               rotation=12.5,start_datetime="1/1/2016")
    print(ml.sr.xcentergrid)

    dis = flopy.modflow.ModflowDis(ml,nrow=10,ncol=5,delr=np.arange(5))
    print(ml.sr.xcentergrid)

    ml.model_ws = "temp"


    ml.write_input()
    ml1 = flopy.modflow.Modflow.load("test.nam",model_ws="temp")
    assert ml1.sr == ml.sr
    assert ml1.start_datetime == ml.start_datetime

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
                         top=10, botm=botm, steady=False)
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
    #test_mbase_sr()
    #test_sr()
    #test_dis_reference()
    test_mflist_reference()
    #test_formattedfile_reference()
    #test_binaryfile_reference()
