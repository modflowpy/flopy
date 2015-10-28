# Test reference class
import matplotlib
matplotlib.use('agg')


# def test_binaryfile_reference():
#     import os
#     import flopy
#
#     h = flopy.utils.HeadFile(os.path.join('..', 'examples', 'data', 'freyberg', 'freyberg.githds'))
#     h.sr.xul = 1000.0
#     h.sr.yul = 200.0
#     h.sr.rotation = 15.0
#     h.plot(filename_base=os.path.join('temp', 't006'))
#     matplotlib.pyplot.close('all')
#     return


def test_mflist_reference():
    import os
    import numpy as np
    import shapefile
    import flopy.modflow as fmf

    # model_ws = os.path.join('..', 'data', 'freyberg')
    # ml = fmf.Modflow.load('freyberg.nam', model_ws=model_ws)
    # make the model
    ml = fmf.Modflow()
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
    lpf = fmf.ModflowLpf(ml, hk=hk, vka=10.0, laytyp=1)
    pcg = fmf.ModflowPcg(ml)
    oc = fmf.ModflowOc(ml)
    ibound = np.ones((nrow, ncol))
    ibound[:, 0] = -1
    ibound[25:30, 30:39] = 0
    bas = fmf.ModflowBas(ml, strt=5.0, ibound=ibound)
    rch = fmf.ModflowRch(ml, rech={0: 0.00001, 5: 0.0001, 6: 0.0})
    wel_dict = {}
    wel_data = [[9, 25, 20, -200], [0, 0, 0, -400], [5, 20, 32, 500]]
    wel_dict[0] = wel_data
    wel_data2 = [[45, 20, 200], [9, 49, 39, 400], [5, 20, 32, 500]]
    wel_dict[10] = wel_data2
    wel = fmf.ModflowWel(ml, stress_period_data={0: wel_data})
    ghb_dict = {0: [1, 10, 10, 400, 300]}
    ghb = fmf.ModflowGhb(ml, stress_period_data=ghb_dict)

    test = os.path.join('temp', 'test3.shp')
    ml.to_shapefile(test, kper=0)
    shp = shapefile.Reader(test)
    assert shp.numRecords == nrow * ncol


if __name__ == '__main__':
    test_mflist_reference()
    test_binaryfile_reference()
