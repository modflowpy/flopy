# Test SWR binary read functionality
import os
import flopy

pth = os.path.join('..', 'examples', 'data', 'swr_test')
files = ('SWR004.stg', 'SWR004.flow', 'SWR004.vel', 'swr005.qaq',
         'SWR004.str', 'SWR004.obs')


def test_swr_binary_stage(ipos=0):
    fpth = os.path.join(pth, files[ipos])

    sobj = flopy.utils.SwrStage(fpth)
    assert isinstance(sobj, flopy.utils.SwrStage), \
        'SwrStage object not created'

    nrecords = sobj.get_nrecords()
    assert nrecords == (18, 0), 'SwrStage records does not equal (18, 0)'

    ntimes = sobj.get_ntimes()
    assert ntimes == 336, 'SwrStage ntimes does not equal 336'

    for idx in range(ntimes):
        r = sobj.get_data(idx=idx)
        assert r is not None, \
            'SwrStage could not read data with get_data(idx=)'
        assert r.shape == (18,), \
            'SwrStage stage data shape does not equal (18,)'
        assert len(r.dtype.names) == 2, \
            'SwrStage stage data dtype does not have 2 entries'

    kswrkstpkper = sobj.get_kswrkstpkper()
    assert kswrkstpkper.shape == (336, 3), \
        'SwrStage kswrkstpkper shape does not equal (336, 3)'

    for kkk in kswrkstpkper:
        r = sobj.get_data(kswrkstpkper=kkk)
        assert r is not None, \
            'SwrStage could not read data with get_data(kswrkstpkper=)'
        assert r.shape == (18,), \
            'SwrStage stage data shape does not equal (18,)'
        assert len(r.dtype.names) == 2, \
            'SwrStage stage data dtype does not have 2 entries'

    times = sobj.get_times()
    assert len(times) == 336, 'SwrStage times length does not equal 336'

    for time in times:
        r = sobj.get_data(totim=time)
        assert r is not None, \
            'SwrStage could not read data with get_data(tottim=)'
        assert r.shape == (18,), \
            'SwrStage stage data shape does not equal (18,)'
        assert len(r.dtype.names) == 2, \
            'SwrStage stage data dtype does not have 2 entries'

    ts = sobj.get_ts(irec=17)
    assert ts.shape == (336,), \
        'SwrStage stage timeseries shape does not equal (336,)'
    assert len(ts.dtype.names) == 2, \
        'SwrStage stage time series stage data dtype does not have 2 entries'

    # plt.plot(ts['totim'], ts['stage'])
    # plt.show()

    return


def test_swr_binary_budget(ipos=1):
    fpth = os.path.join(pth, files[ipos])

    sobj = flopy.utils.SwrBudget(fpth)
    assert isinstance(sobj, flopy.utils.SwrBudget), \
        'SwrBudget object not created'

    nrecords = sobj.get_nrecords()
    assert nrecords == (18, 0), 'SwrBudget records does not equal (18, 0)'

    ntimes = sobj.get_ntimes()
    assert ntimes == 336, 'SwrBudget ntimes does not equal 336'

    for idx in range(ntimes):
        r = sobj.get_data(idx=idx)
        assert r is not None, \
            'SwrBudget could not read data with get_data(idx=)'
        assert r.shape == (18,), \
            'SwrBudget budget data shape does not equal (18,)'
        assert len(r.dtype.names) == 15, \
            'SwrBudget data dtype does not have 15 entries'

    # plt.bar(range(18), r['inf-out'])
    # plt.show()

    kswrkstpkper = sobj.get_kswrkstpkper()
    assert kswrkstpkper.shape == (336, 3), \
        'SwrBudget kswrkstpkper shape does not equal (336, 3)'

    for kkk in kswrkstpkper:
        r = sobj.get_data(kswrkstpkper=kkk)
        assert r is not None, \
            'SwrBudget could not read data with get_data(kswrkstpkper=)'
        assert r.shape == (18,), \
            'SwrBudget budget data shape does not equal (18,)'
        assert len(r.dtype.names) == 15, \
            'SwrBudget budget data dtype does not have 15 entries'

    times = sobj.get_times()
    assert len(times) == 336, 'SwrBudget times length does not equal 336'

    for time in times:
        r = sobj.get_data(totim=time)
        assert r is not None, \
            'SwrBudget could not read data with get_data(tottim=)'
        assert r.shape == (18,), \
            'SwrBudget budget data shape does not equal (18,)'
        assert len(r.dtype.names) == 15, \
            'SwrBudget budget data dtype does not have 15 entries'

    ts = sobj.get_ts(irec=17)
    assert ts.shape == (336,), \
        'SwrBudget budget timeseries shape does not equal (336,)'
    assert len(ts.dtype.names) == 15, \
        'SwrBudget time series budget data dtype does not have 15 entries'

    # plt.plot(ts['totim'], ts['qbcflow'])
    # plt.show()

    return


def test_swr_binary_qm(ipos=2):
    fpth = os.path.join(pth, files[ipos])

    sobj = flopy.utils.SwrFlow(fpth)
    assert isinstance(sobj, flopy.utils.SwrFlow), 'SwrFlow object not created'

    nrecords = sobj.get_nrecords()
    assert nrecords == (40, 18), 'SwrFlow records does not equal (40, 18)'

    connect = sobj.get_connectivity()
    assert connect.shape == (
        40, 3), 'SwrFlow connectivity shape does not equal (40, 3)'

    ntimes = sobj.get_ntimes()
    assert ntimes == 336, 'SwrFlow ntimes does not equal 336'

    for idx in range(ntimes):
        r = sobj.get_data(idx=idx)
        assert r is not None, \
            'SwrFlow could not read data with get_data(idx=)'
        assert r.shape == (40,), 'SwrFlow qm data shape does not equal (40,)'
        assert len(r.dtype.names) == 3, \
            'SwrFlow qm data dtype does not have 3 entries'

    # plt.bar(range(40), r['flow'])
    # plt.show()

    kswrkstpkper = sobj.get_kswrkstpkper()
    assert kswrkstpkper.shape == (336, 3), \
        'SwrFlow kswrkstpkper shape does not equal (336, 3)'

    for kkk in kswrkstpkper:
        r = sobj.get_data(kswrkstpkper=kkk)
        assert r is not None, \
            'SwrFlow could not read data with get_data(kswrkstpkper=)'
        assert r.shape == (40,), 'SwrFlow qm data shape does not equal (40,)'
        assert len(r.dtype.names) == 3, \
            'SwrFlow qm data dtype does not have 3 entries'

    times = sobj.get_times()
    assert len(times) == 336, 'SwrFlow times length does not equal 336'

    for time in times:
        r = sobj.get_data(totim=time)
        assert r is not None, \
            'SwrFlow could not read data with get_data(tottim=)'
        assert r.shape == (40,), 'SwrFlow qm data shape does not equal (40,)'
        assert len(r.dtype.names) == 3, \
            'SwrFlow qm data dtype does not have 3 entries'

    ts = sobj.get_ts(irec=17, iconn=16)
    assert ts.shape == (336,), \
        'SwrFlow qm timeseries shape does not equal (336,)'
    assert len(ts.dtype.names) == 3, \
        'SwrFlow time series qm data dtype does not have 3 entries'

    ts2 = sobj.get_ts(irec=16, iconn=17)
    assert ts2.shape == (336,), \
        'SwrFlow qm timeseries shape does not equal (336,)'
    assert len(ts2.dtype.names) == 3, \
        'SwrFlow time series qm data dtype does not have 3 entries'

    # plt.plot(ts['totim'], ts['velocity'])
    # plt.plot(ts2['totim'], ts2['velocity'])
    # plt.show()

    return


def test_swr_binary_qaq(ipos=3):
    fpth = os.path.join(pth, files[ipos])

    sobj = flopy.utils.SwrExchange(fpth, verbose=True)
    assert isinstance(sobj,
                      flopy.utils.SwrExchange), 'SwrExchange object not created'

    nrecords = sobj.get_nrecords()
    assert nrecords == (19, 0), 'SwrExchange records does not equal (19, 0)'

    ntimes = sobj.get_ntimes()
    assert ntimes == 350, 'SwrExchange ntimes does not equal 350'

    for idx in range(ntimes):
        r = sobj.get_data(idx=idx)
        assert r is not None, \
            'SwrExchange could not read data with get_data(idx=)'
        assert r.shape == (21,), \
            'SwrExchange qaq data shape does not equal (21,)'
        assert len(r.dtype.names) == 11, \
            'SwrExchange qaq data dtype does not have 11 entries'

    # plt.bar(range(21), r['qaq'])
    # plt.show()

    kswrkstpkper = sobj.get_kswrkstpkper()
    assert kswrkstpkper.shape == (
        350, 3), 'SwrExchange kswrkstpkper shape does not equal (350, 3)'

    for kkk in kswrkstpkper:
        r = sobj.get_data(kswrkstpkper=kkk)
        assert r is not None, \
            'SwrExchange could not read data with get_data(kswrkstpkper=)'
        assert r.shape == (21,), \
            'SwrExchange qaq data shape does not equal (21,)'
        assert len(r.dtype.names) == 11, \
            'SwrExchange qaq data dtype does not have 11 entries'

    times = sobj.get_times()
    assert len(times) == 350, 'SwrExchange times length does not equal 350'

    for time in times:
        r = sobj.get_data(totim=time)
        assert r is not None, \
            'SwrExchange could not read data with get_data(tottim=)'
        assert r.shape == (21,), \
            'SwrExchange qaq data shape does not equal (21,)'
        assert len(r.dtype.names) == 11, \
            'SwrExchange qaq data dtype does not have 11 entries'

    ts = sobj.get_ts(irec=17, klay=0)
    assert ts.shape == (350,), \
        'SwrExchange timeseries shape does not equal (350,)'
    assert len(ts.dtype.names) == 11, \
        'SwrExchange time series qaq data dtype does not have 11 entries'

    # plt.plot(ts['totim'], ts['qaq'])
    # plt.show()

    return


def test_swr_binary_structure(ipos=4):
    fpth = os.path.join(pth, files[ipos])

    sobj = flopy.utils.SwrStructure(fpth, verbose=True)
    assert isinstance(sobj, flopy.utils.SwrStructure), \
        'SwrStructure object not created'

    nrecords = sobj.get_nrecords()
    assert nrecords == (18, 0), 'SwrStructure records does not equal (18, 0)'

    ntimes = sobj.get_ntimes()
    assert ntimes == 336, 'SwrStructure ntimes does not equal 336'

    for idx in range(ntimes):
        r = sobj.get_data(idx=idx)
        assert r is not None, \
            'SwrStructure could not read data with get_data(idx=)'
        assert r.shape == (2,), \
            'SwrStructure structure data shape does not equal (2,)'
        assert len(r.dtype.names) == 8, \
            'SwrStructure structure data dtype does not have 8 entries'

    kswrkstpkper = sobj.get_kswrkstpkper()
    assert kswrkstpkper.shape == (336, 3), \
        'SwrStructure kswrkstpkper shape does not equal (336, 3)'

    for kkk in kswrkstpkper:
        r = sobj.get_data(kswrkstpkper=kkk)
        assert r is not None, \
            'SwrStructure could not read data with get_data(kswrkstpkper=)'
        assert r.shape == (2,), \
            'SwrStructure structure data shape does not equal (2,)'
        assert len(r.dtype.names) == 8, \
            'SwrStructure structure data dtype does not have 8 entries'

    times = sobj.get_times()
    assert len(times) == 336, 'SwrStructure times length does not equal 336'

    for time in times:
        r = sobj.get_data(totim=time)
        assert r is not None, \
            'SwrStructure could not read data with get_data(tottim=)'
        assert r.shape == (2,), \
            'SwrStructure structure data shape does not equal (2,)'
        assert len(r.dtype.names) == 8, \
            'SwrStructure structure data dtype does not have 8 entries'

    ts = sobj.get_ts(irec=17, istr=0)
    assert ts.shape == (336,), \
        'SwrStructure timeseries shape does not equal (336,)'
    assert len(ts.dtype.names) == 8, \
        'SwrStructure time series structure data dtype does not have 8 entries'

    # plt.plot(ts['totim'], ts['strflow'])
    # plt.show()


    obs3 = sobj.get_ts(irec=17, istr=0)

    return


def test_swr_binary_obs(ipos=5):
    fpth = os.path.join(pth, files[ipos])

    sobj = flopy.utils.SwrObs(fpth)
    assert isinstance(sobj, flopy.utils.SwrObs), 'SwrObs object not created'

    nobs = sobj.get_nobs()
    assert nobs == 9, 'SwrObs numobs does not equal 9'

    obsnames = sobj.get_obsnames()
    assert len(obsnames) == 9, 'SwrObs number of obsnames does not equal 9'

    ntimes = sobj.get_ntimes()
    assert ntimes == 336, 'SwrObs numtimes does not equal 336'

    times = sobj.get_times()
    assert len(times) == 336, 'SwrFile times length does not equal 336'

    ts = sobj.get_data()
    assert ts.shape == (336,), \
        'SwrObs length of data array does not equal (336,)'
    assert len(ts.dtype.names) == 10, \
        'SwrObs data does not have totim + 9 observations'

    ts = sobj.get_data(obsname='OBS5')
    assert ts.shape == (336,), \
        'SwrObs length of data array does not equal (336,)'
    assert len(ts.dtype.names) == 2, \
        'SwrObs data does not have totim + 1 observation'

    # plt.plot(ts['totim'], ts['OBS5'])
    # plt.show()

    for idx in range(ntimes):
        d = sobj.get_data(idx=idx)
        assert d.shape == (1,), \
            'SwrObs length of data array does not equal (1,)'
        assert len(d.dtype.names) == nobs+1, \
            'SwrObs data does not have nobs + 1'


    for time in times:
        d = sobj.get_data(totim=time)
        assert d.shape == (1,), \
            'SwrObs length of data array does not equal (1,)'
        assert len(d.dtype.names) == nobs+1, \
            'SwrObs data does not have nobs + 1'

    # test get_dataframes()
    try:
        import pandas as pd

        for idx in range(ntimes):
            df = sobj.get_dataframe(idx=idx, timeunit='S')
            assert isinstance(df, pd.DataFrame), 'A DataFrame was not returned'
            assert df.shape == (1, nobs + 1), 'data shape is not (1, 10)'

        for time in times:
            df = sobj.get_dataframe(totim=time, timeunit='S')
            assert isinstance(df, pd.DataFrame), 'A DataFrame was not returned'
            assert df.shape == (1, nobs + 1), 'data shape is not (1, 10)'

        df = sobj.get_dataframe(timeunit='S')
        assert isinstance(df, pd.DataFrame), 'A DataFrame was not returned'
        assert df.shape == (336, nobs + 1), 'data shape is not (336, 10)'
    except ImportError:
        print('pandas not available...')

    return


if __name__ == '__main__':
    test_swr_binary_obs()
    test_swr_binary_stage()
    test_swr_binary_budget()
    test_swr_binary_qm()
    test_swr_binary_qaq()
    test_swr_binary_structure()
