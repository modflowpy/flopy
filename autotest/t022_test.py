# Test modflow write adn run
import os
import matplotlib.pyplot as plt

pth = os.path.join('..', 'examples', 'data', 'swr_test')
files = [('SWR004.stg', 'stage'),
         ('SWR004.flow', 'budget'),
         ('SWR004.vel', 'qm'),
         ('swr005.qaq', 'qaq')]


def test_swr_binary_stage(ipos=0):
    import flopy

    fpth = os.path.join(pth, files[ipos][0])
    swrtype = files[ipos][1]

    sobj = flopy.utils.SwrFile(fpth, swrtype=swrtype)
    assert isinstance(sobj, flopy.utils.SwrFile), 'SwrFile object not created'

    nrecords = sobj.get_nrecords()
    assert nrecords == (0, 18), 'SwrFile records does not equal (0, 18)'

    ntimes = sobj.get_ntimes()
    assert ntimes == 336, 'SwrFile ntimes does not equal 336'

    for idx in range(ntimes):
        r = sobj.get_data(idx=idx)
        assert r is not None, 'SwrFile could not read data with get_data(idx=)'
        assert r.shape == (18, 1), 'SwrFile data shape does not equal (18, 1)'
        assert len(r.dtype.names) == 2, 'SwrFile stage data dtype does not have 2 entries'

    kswrkstpkper = sobj.get_kswrkstpkper()
    assert kswrkstpkper.shape == (336, 3), 'SwrFile kswrkstpkper shape does not equal (336, 3)'

    for kkk in kswrkstpkper:
        r = sobj.get_data(kswrkstpkper=kkk)
        assert r is not None, 'SwrFile could not read data with get_data(kswrkstpkper=)'
        assert r.shape == (18, 1), 'SwrFile data shape does not equal (18, 1)'
        assert len(r.dtype.names) == 2, 'SwrFile stage data dtype does not have 2 entries'

    times = sobj.get_times()
    assert times.shape == (336,), 'SwrFile times shape does not equal (336,)'

    for time in times:
        r = sobj.get_data(totim=time)
        assert r is not None, 'SwrFile could not read data with get_data(tottim=)'
        assert r.shape == (18, 1), 'SwrFile data shape does not equal (18, 1)'
        assert len(r.dtype.names) == 2, 'SwrFile stage data dtype does not have 2 entries'

    ts = sobj.get_ts(irec=17)
    assert ts.shape == (336, 1), 'SwrFile timeseries shape does not equal (336, 1)'
    assert len(ts.dtype.names) == 2, 'SwrFile time series stage data dtype does not have 2 entries'

    #plt.plot(ts['totim'], ts['stage'])
    #plt.show()

    return

def test_swr_binary_budget(ipos=1):
    import flopy

    fpth = os.path.join(pth, files[ipos][0])
    swrtype = files[ipos][1]

    sobj = flopy.utils.SwrFile(fpth, swrtype=swrtype)
    assert isinstance(sobj, flopy.utils.SwrFile), 'SwrFile object not created'

    nrecords = sobj.get_nrecords()
    assert nrecords == (0, 18), 'SwrFile records does not equal (0, 18)'

    ntimes = sobj.get_ntimes()
    assert ntimes == 336, 'SwrFile ntimes does not equal 336'

    for idx in range(ntimes):
        r = sobj.get_data(idx=idx)
        assert r is not None, 'SwrFile could not read data with get_data(idx=)'
        assert r.shape == (18, 1), 'SwrFile budget data shape does not equal (18, 1)'
        assert len(r.dtype.names) == 15, 'SwrFile data dtype does not have 15 entries'

    #plt.bar(range(18), r['inf-out'])
    #plt.show()

    kswrkstpkper = sobj.get_kswrkstpkper()
    assert kswrkstpkper.shape == (336, 3), 'SwrFile kswrkstpkper shape does not equal (336, 3)'

    for kkk in kswrkstpkper:
        r = sobj.get_data(kswrkstpkper=kkk)
        assert r is not None, 'SwrFile could not read data with get_data(kswrkstpkper=)'
        assert r.shape == (18, 1), 'SwrFile data shape does not equal (18, 1)'
        assert len(r.dtype.names) == 15, 'SwrFile budget data dtype does not have 15 entries'

    times = sobj.get_times()
    assert times.shape == (336,), 'SwrFile times shape does not equal (336,)'

    for time in times:
        r = sobj.get_data(totim=time)
        assert r is not None, 'SwrFile could not read data with get_data(tottim=)'
        assert r.shape == (18, 1), 'SwrFile data shape does not equal (18, 1)'
        assert len(r.dtype.names) == 15, 'SwrFile budget data dtype does not have 15 entries'

    ts = sobj.get_ts(irec=17)
    assert ts.shape == (336, 1), 'SwrFile timeseries shape does not equal (336, 1)'
    assert len(ts.dtype.names) == 15, 'SwrFile time series budget data dtype does not have 15 entries'

    #plt.plot(ts['totim'], ts['qbcflow'])
    #plt.show()

    return

def test_swr_binary_qm(ipos=2):
    import flopy

    fpth = os.path.join(pth, files[ipos][0])
    swrtype = files[ipos][1]

    sobj = flopy.utils.SwrFile(fpth, swrtype=swrtype)
    assert isinstance(sobj, flopy.utils.SwrFile), 'SwrFile object not created'

    nrecords = sobj.get_nrecords()
    assert nrecords == (18, 40), 'SwrFile records does not equal (18, 40)'

    connect = sobj.get_connectivity()
    assert connect.shape == (40, 3), 'SwrFile connectivity shape does not equal (40, 3)'

    ntimes = sobj.get_ntimes()
    assert ntimes == 336, 'SwrFile ntimes does not equal 336'

    for idx in range(ntimes):
        r = sobj.get_data(idx=idx)
        assert r is not None, 'SwrFile could not read data with get_data(idx=)'
        assert r.shape == (40, 1), 'SwrFile data shape does not equal (40, 1)'
        assert len(r.dtype.names) == 3, 'SwrFile qm data dtype does not have 3 entries'

    #plt.bar(range(18), r['inf-out'])
    #plt.show()

    kswrkstpkper = sobj.get_kswrkstpkper()
    assert kswrkstpkper.shape == (336, 3), 'SwrFile kswrkstpkper shape does not equal (336, 3)'

    for kkk in kswrkstpkper:
        r = sobj.get_data(kswrkstpkper=kkk)
        assert r is not None, 'SwrFile could not read data with get_data(kswrkstpkper=)'
        assert r.shape == (40, 1), 'SwrFile data shape does not equal (40, 1)'
        assert len(r.dtype.names) == 3, 'SwrFile qm data dtype does not have 3 entries'

    times = sobj.get_times()
    assert times.shape == (336,), 'SwrFile times shape does not equal (336,)'

    for time in times:
        r = sobj.get_data(totim=time)
        assert r is not None, 'SwrFile could not read data with get_data(tottim=)'
        assert r.shape == (40, 1), 'SwrFile data shape does not equal (40, 1)'
        assert len(r.dtype.names) == 3, 'SwrFile qm data dtype does not have 3 entries'

    ts = sobj.get_ts(irec=17, iconn=16)
    assert ts.shape == (336, 1), 'SwrFile timeseries shape does not equal (336, 1)'
    assert len(ts.dtype.names) == 3, 'SwrFile time series qm data dtype does not have 3 entries'

    ts2 = sobj.get_ts(irec=16, iconn=17)
    assert ts2.shape == (336, 1), 'SwrFile timeseries shape does not equal (336, 1)'
    assert len(ts2.dtype.names) == 3, 'SwrFile time series qm data dtype does not have 3 entries'

    #plt.plot(ts['totim'], ts['velocity'])
    #plt.plot(ts2['totim'], ts2['velocity'])
    #plt.show()

    return


def test_swr_binary_qaq(ipos=3):
    import flopy

    fpth = os.path.join(pth, files[ipos][0])
    swrtype = files[ipos][1]

    sobj = flopy.utils.SwrFile(fpth, swrtype=swrtype, verbose=True)
    assert isinstance(sobj, flopy.utils.SwrFile), 'SwrFile object not created'

    nrecords = sobj.get_nrecords()
    assert nrecords == (0, 19), 'SwrFile records does not equal (0, 19)'

    ntimes = sobj.get_ntimes()
    assert ntimes == 350, 'SwrFile ntimes does not equal 331'

    for idx in range(ntimes):
        r = sobj.get_data(idx=idx)
        assert r is not None, 'SwrFile could not read data with get_data(idx=)'
        assert r.shape == (21, 1), 'SwrFile qaq data shape does not equal (21, 1)'
        assert len(r.dtype.names) == 11, 'SwrFile qaq data dtype does not have 11 entries'

    #plt.bar(range(18), r['inf-out'])
    #plt.show()

    kswrkstpkper = sobj.get_kswrkstpkper()
    assert kswrkstpkper.shape == (350, 3), 'SwrFile kswrkstpkper shape does not equal (350, 3)'

    for kkk in kswrkstpkper:
        r = sobj.get_data(kswrkstpkper=kkk)
        assert r is not None, 'SwrFile could not read data with get_data(kswrkstpkper=)'
        assert r.shape == (21, 1), 'SwrFile qaq data shape does not equal (21, 1)'
        assert len(r.dtype.names) == 11, 'SwrFile qaq data dtype does not have 11 entries'

    times = sobj.get_times()
    assert times.shape == (350,), 'SwrFile times shape does not equal (350,)'

    for time in times:
        r = sobj.get_data(totim=time)
        assert r is not None, 'SwrFile could not read data with get_data(tottim=)'
        assert r.shape == (21, 1), 'SwrFile qaq data shape does not equal (21, 1)'
        assert len(r.dtype.names) == 11, 'SwrFile qaq data dtype does not have 11 entries'

    ts = sobj.get_ts(irec=17, klay=0)
    assert ts.shape == (350, 1), 'SwrFile timeseries shape does not equal (350, 1)'
    assert len(ts.dtype.names) == 11, 'SwrFile time series qaq data dtype does not have 11 entries'

    #plt.plot(ts['totim'], ts['qaq'])
    #plt.show()

    return

if __name__ == '__main__':
    test_swr_binary_qaq()
    test_swr_binary_qm()
    test_swr_binary_budget()
    test_swr_binary_stage()
