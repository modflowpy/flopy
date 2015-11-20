# Test binary and formatted data readers
import numpy as np

def test_formattedfile_read():
    import os
    import flopy
    h = flopy.utils.FormattedHeadFile(os.path.join('..', 'examples', 'data', 'mf2005_test', 'test1tr.githds'))
    assert isinstance(h, flopy.utils.FormattedHeadFile)

    times = h.get_times()
    assert abs(times[0]-1577880064.0) < 1e-6, 'times[0] != {}'.format(times[0])

    kstpkper = h.get_kstpkper()
    assert kstpkper[0] == (49, 0), 'kstpkper[0] != (49, 0)'

    h0 = h.get_data(totim=times[0])
    h1 = h.get_data(kstpkper=kstpkper[0])
    h2 = h.get_data(idx=0)
    assert np.array_equal(h0, h1), 'formatted head read using totim != head read using kstpkper'
    assert np.array_equal(h0, h2), 'formatted head read using totim != head read using idx'

    ts = h.get_ts((0, 7, 5))
    assert abs(ts[0, 1]-976.801025390625) < 1e-6, \
        'time series value ({}) != 976.801 - difference = {}'.format(ts[0, 1], abs(ts[0, 1]-976.801025390625))
    return

def test_binaryfile_read():
    import os
    import flopy

    h = flopy.utils.HeadFile(os.path.join('..', 'examples', 'data', 'freyberg', 'freyberg.githds'))
    assert isinstance(h, flopy.utils.HeadFile)

    times = h.get_times()
    assert abs(times[0]-10.0) < 1e-6, 'times[0] != {}'.format(times[0])

    kstpkper = h.get_kstpkper()
    assert kstpkper[0] == (0, 0), 'kstpkper[0] != (0, 0)'

    h0 = h.get_data(totim=times[0])
    h1 = h.get_data(kstpkper=kstpkper[0])
    h2 = h.get_data(idx=0)
    assert np.array_equal(h0, h1), 'binary head read using totim != head read using kstpkper'
    assert np.array_equal(h0, h2), 'binary head read using totim != head read using idx'

    ts = h.get_ts((0, 7, 5))
    assert abs(ts[0, 1]-26.00697135925293) < 1e-6, \
        'time series value ({}) != 976.801 - difference = {}'.format(ts[0, 1], abs(ts[0, 1]-26.00697135925293))
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



if __name__ == '__main__':
    test_formattedfile_read()
    test_binaryfile_read()
    test_cellbudgetfile_read()
