# Test binary and formatted data readers
import numpy as np


def test_formattedfile_read():
    import os
    import flopy
    h = flopy.utils.FormattedHeadFile(os.path.join('..', 'examples', 'data', 'mf2005_test', 'test1tr.githds'))
    assert isinstance(h, flopy.utils.FormattedHeadFile)

    times = h.get_times()
    assert abs(times[0] - 1577880064.0) < 1e-6, 'times[0] != {}'.format(times[0])

    kstpkper = h.get_kstpkper()
    assert kstpkper[0] == (49, 0), 'kstpkper[0] != (49, 0)'

    h0 = h.get_data(totim=times[0])
    h1 = h.get_data(kstpkper=kstpkper[0])
    h2 = h.get_data(idx=0)
    assert np.array_equal(h0, h1), 'formatted head read using totim != head read using kstpkper'
    assert np.array_equal(h0, h2), 'formatted head read using totim != head read using idx'

    ts = h.get_ts((0, 7, 5))
    assert abs(ts[0, 1] - 976.801025390625) < 1e-6, \
        'time series value ({}) != 976.801 - difference = {}'.format(ts[0, 1], abs(ts[0, 1] - 976.801025390625))
    return


def test_binaryfile_read():
    import os
    import flopy

    h = flopy.utils.HeadFile(os.path.join('..', 'examples', 'data', 'freyberg', 'freyberg.githds'))
    assert isinstance(h, flopy.utils.HeadFile)

    times = h.get_times()
    assert abs(times[0] - 10.0) < 1e-6, 'times[0] != {}'.format(times[0])

    kstpkper = h.get_kstpkper()
    assert kstpkper[0] == (0, 0), 'kstpkper[0] != (0, 0)'

    h0 = h.get_data(totim=times[0])
    h1 = h.get_data(kstpkper=kstpkper[0])
    h2 = h.get_data(idx=0)
    assert np.array_equal(h0, h1), 'binary head read using totim != head read using kstpkper'
    assert np.array_equal(h0, h2), 'binary head read using totim != head read using idx'

    ts = h.get_ts((0, 7, 5))
    assert abs(ts[0, 1] - 26.00697135925293) < 1e-6, \
        'time series value ({}) != 976.801 - difference = {}'.format(ts[0, 1], abs(ts[0, 1] - 26.00697135925293))
    return


def test_cellbudgetfile_read():
    import os
    import flopy

    v = flopy.utils.CellBudgetFile(os.path.join('..', 'examples', 'data', 'mf2005_test', 'mnw1.gitcbc'))
    assert isinstance(v, flopy.utils.CellBudgetFile)

    kstpkper = v.get_kstpkper()
    assert len(kstpkper) == 5, 'length of kstpkper != 5'

    records = v.unique_record_names()
    idx = 0
    for t in kstpkper:
        for record in records:
            t0 = v.get_data(kstpkper=t, text=record, full3D=True)[0]
            t1 = v.get_data(idx=idx, text=record, full3D=True)[0]
            assert np.array_equal(t0, t1), \
                'binary budget item {0} read using kstpkper != binary budget item {0} read using idx'.format(record)
            idx += 1

    return


def test_cellbudgetfile_readrecord():
    import os
    import flopy

    v = flopy.utils.CellBudgetFile(os.path.join('..', 'examples', 'data', 'mf2005_test', 'test1tr.gitcbc'))
    assert isinstance(v, flopy.utils.CellBudgetFile)

    kstpkper = v.get_kstpkper()
    assert len(kstpkper) == 30, 'length of kstpkper != 30'

    t = v.get_data(text='STREAM LEAKAGE')
    assert len(t) == 30, 'length of stream leakage data != 30'
    assert t[0].shape[0] == 36, 'sfr budget data does not have 36 reach entries'

    t = v.get_data(text='STREAM LEAKAGE', full3D=True)
    assert t[0].shape == (1, 15, 10), '3D sfr budget data does not have correct shape (1, 15,10) - ' + \
                                      'returned shape {}'.format(t[0].shape)

    for kk in kstpkper:
        t = v.get_data(kstpkper=kk, text='STREAM LEAKAGE', full3D=True)[0]
        assert t.shape == (1, 15, 10), '3D sfr budget data for kstpkper {} '.format(kk) + \
                                       'does not have correct shape (1, 15,10) - ' + \
                                       'returned shape {}'.format(t[0].shape)

    idx = v.get_indices()
    assert idx is None, 'get_indices() without record did not return None'

    records = v.unique_record_names()
    for record in records:
        indices = v.get_indices(text=record.decode().strip())
        for idx, kk in enumerate(kstpkper):
            t0 = v.get_data(kstpkper=kk, text=record.decode().strip())[0]
            t1 = v.get_data(idx=indices[idx], text=record)[0]
            assert np.array_equal(t0, t1), \
                'binary budget item {0} read using kstpkper != binary budget item {0} read using idx'.format(record)

    return


def test_cellbudgetfile_readrecord_waux():
    import os
    import flopy

    v = flopy.utils.CellBudgetFile(os.path.join('..', 'examples', 'data', 'mf2005_test', 'test1tr.gitcbc'))
    assert isinstance(v, flopy.utils.CellBudgetFile)

    kstpkper = v.get_kstpkper()
    assert len(kstpkper) == 30, 'length of kstpkper != 30'

    t = v.get_data(text='WELLS')
    assert len(t) == 30, 'length of well data != 30'
    assert t[0].shape[0] == 10, 'wel budget data does not have 10 well entries'

    t = v.get_data(text='WELLS', full3D=True)
    assert t[0].shape == (1, 15, 10), '3D wel budget data does not have correct shape (1, 15,10) - ' + \
                                      'returned shape {}'.format(t[0].shape)

    for kk in kstpkper:
        t = v.get_data(kstpkper=kk, text='wells', full3D=True)[0]
        assert t.shape == (1, 15, 10), '3D wel budget data for kstpkper {} '.format(kk) + \
                                       'does not have correct shape (1, 15,10) - ' + \
                                       'returned shape {}'.format(t[0].shape)

    idx = v.get_indices()
    assert idx is None, 'get_indices() without record did not return None'

    records = v.unique_record_names()
    for record in records:
        indices = v.get_indices(text=record.decode().strip())
        for idx, kk in enumerate(kstpkper):
            t0 = v.get_data(kstpkper=kk, text=record.decode().strip())[0]
            t1 = v.get_data(idx=indices[idx], text=record)[0]
            assert np.array_equal(t0, t1), \
                'binary budget item {0} read using kstpkper != binary budget item {0} read using idx'.format(record)

    return


if __name__ == '__main__':
    test_formattedfile_read()
    test_binaryfile_read()
    test_cellbudgetfile_read()
    test_cellbudgetfile_readrecord()
    test_cellbudgetfile_readrecord_waux()
