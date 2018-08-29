import os
import numpy as np
import flopy

mpth = os.path.join('temp', 't019')
# make the directory if it does not exist
if not os.path.isdir(mpth):
    os.makedirs(mpth)


# Test hydmod data readers
def test_hydmodfile_create():
    model_ws = os.path.join(mpth)
    if not os.path.exists(model_ws):
        os.makedirs(model_ws)
    m = flopy.modflow.Modflow('test', model_ws=model_ws)
    hyd = flopy.modflow.ModflowHyd(m)
    m.hyd.write_file()
    pth = os.path.join(model_ws, 'test.hyd')
    hydload = flopy.modflow.ModflowHyd.load(pth, m)
    assert np.array_equal(hyd.obsdata,
                          hydload.obsdata), 'Written hydmod data not equal to loaded hydmod data'

    # test obsdata as recarray
    obsdata = np.array(
        [(3208, 'BAS', 'HD', 'I', 4, 630486.19, 5124733.18, 'well1')],
        dtype=[('index', '<i8'),
               ('pckg', 'O'),
               ('arr', 'O'),
               ('intyp', 'O'),
               ('klay', '<i8'),
               ('xl', '<f8'),
               ('yl', '<f8'),
               ('hydlbl', 'O')]).view(np.recarray)
    hyd = flopy.modflow.ModflowHyd(m, obsdata=obsdata)

    # test obsdata as object array
    obsdata = np.array([('BAS', 'HD', 'I', 4, 630486.19, 5124733.18, 'well1')],
                       dtype=object)
    hyd = flopy.modflow.ModflowHyd(m, obsdata=obsdata)
    assert True
    return


def test_hydmodfile_load():
    model = 'test1tr.nam'
    pth = os.path.join('..', 'examples', 'data', 'hydmod_test')
    m = flopy.modflow.Modflow.load(model, version='mf2005', model_ws=pth,
                                   verbose=True)
    hydref = m.hyd
    assert isinstance(hydref,
                      flopy.modflow.ModflowHyd), 'Did not load hydmod package...test1tr.hyd'

    model_ws = os.path.join(mpth)
    if not os.path.exists(model_ws):
        os.makedirs(model_ws)

    m.change_model_ws(model_ws)
    m.hyd.write_file()

    pth = os.path.join('..', 'examples', 'data', 'hydmod_test', 'test1tr.hyd')
    hydload = flopy.modflow.ModflowHyd.load(pth, m)
    assert np.array_equal(hydref.obsdata, hydload.obsdata), \
        'Written hydmod data not equal to loaded hydmod data'

    return


def test_hydmodfile_read():
    import os
    import flopy

    pth = os.path.join('..', 'examples', 'data', 'hydmod_test',
                       'test1tr.hyd.gitbin')
    h = flopy.utils.HydmodObs(pth)
    assert isinstance(h, flopy.utils.HydmodObs)

    ntimes = h.get_ntimes()
    assert ntimes == 101, \
        'Not enough times in hydmod file ()...'.format(os.path.basename(pth))

    times = h.get_times()
    assert len(times) == 101, \
        'Not enough times in hydmod file ()...'.format(os.path.basename(pth))

    nitems = h.get_nobs()
    assert nitems == 8, \
        'Not enough records in hydmod file ()...'.format(os.path.basename(pth))

    labels = h.get_obsnames()
    assert len(labels) == 8, \
        'Not enough labels in hydmod file ()...'.format(os.path.basename(pth))
    print(labels)

    for idx in range(ntimes):
        data = h.get_data(idx=idx)
        assert data.shape == (1,), 'data shape is not (1,)'

    for time in times:
        data = h.get_data(totim=time)
        assert data.shape == (1,), 'data shape is not (1,)'

    for label in labels:
        data = h.get_data(obsname=label)
        assert data.shape == (len(times),), \
            'data shape is not ({},)'.format(len(times))

    data = h.get_data()
    assert data.shape == (len(times),), \
        'data shape is not ({},)'.format(len(times))
    assert len(data.dtype.names) == nitems + 1, \
        'data column length is not {}'.format(len(nitems + 1))

    try:
        import pandas as pd

        for idx in range(ntimes):
            df = h.get_dataframe(idx=idx, timeunit='S')
            assert isinstance(df, pd.DataFrame), 'A DataFrame was not returned'
            assert df.shape == (1, 9), 'data shape is not (1, 9)'

        for time in times:
            df = h.get_dataframe(totim=time, timeunit='S')
            assert isinstance(df, pd.DataFrame), 'A DataFrame was not returned'
            assert df.shape == (1, 9), 'data shape is not (1, 9)'

        df = h.get_dataframe(timeunit='S')
        assert isinstance(df, pd.DataFrame), 'A DataFrame was not returned'
        assert df.shape == (101, 9), 'data shape is not (101, 9)'
    except:
        print('pandas not available...')
        pass

    return


def test_mf6obsfile_read():
    import os
    import flopy
    try:
        import pandas as pd
    except:
        print('pandas not available...')
        pd = None

    txt = 'binary mf6 obs'

    files = ['maw_obs.gitbin', 'maw_obs.gitcsv']
    binfile = [True, False]

    for idx in range(len(files)):
        pth = os.path.join('..', 'examples', 'data', 'mf6_obs',
                           files[idx])
        h = flopy.utils.Mf6Obs(pth, isBinary=binfile[idx])
        assert isinstance(h, flopy.utils.Mf6Obs)

        ntimes = h.get_ntimes()
        assert ntimes == 3, \
            'Not enough times in {} file...{}'.format(txt,
                                                      os.path.basename(pth))

        times = h.get_times()
        assert len(times) == 3, \
            'Not enough times in {} file...{}'.format(txt,
                                                      os.path.basename(pth))

        nitems = h.get_nobs()
        assert nitems == 1, \
            'Not enough records in {} file...{}'.format(txt,
                                                        os.path.basename(pth))

        labels = h.get_obsnames()
        assert len(labels) == 1, \
            'Not enough labels in {} file...{}'.format(txt,
                                                       os.path.basename(pth))
        print(labels)

        for idx in range(ntimes):
            data = h.get_data(idx=idx)
            assert data.shape == (1,), 'data shape is not (1,)'

        for time in times:
            data = h.get_data(totim=time)
            assert data.shape == (1,), 'data shape is not (1,)'

        for label in labels:
            data = h.get_data(obsname=label)
            assert data.shape == (len(times),), \
                'data shape is not ({},)'.format(len(times))

        data = h.get_data()
        assert data.shape == (len(times),), \
            'data shape is not ({},)'.format(len(times))
        assert len(data.dtype.names) == nitems + 1, \
            'data column length is not {}'.format(len(nitems + 1))

        if pd is not None:
            for idx in range(ntimes):
                df = h.get_dataframe(idx=idx, timeunit='S')
                assert isinstance(df, pd.DataFrame), 'A DataFrame was not returned'
                assert df.shape == (1, 2), 'data shape is not (1, 2)'

            for time in times:
                df = h.get_dataframe(totim=time, timeunit='S')
                assert isinstance(df, pd.DataFrame), 'A DataFrame was not returned'
                assert df.shape == (1, 2), 'data shape is not (1, 2)'

            df = h.get_dataframe(timeunit='S')
            assert isinstance(df, pd.DataFrame), 'A DataFrame was not returned'
            assert df.shape == (3, 2), 'data shape is not (3, 2)'

    return


if __name__ == '__main__':
    test_mf6obsfile_read()
    test_hydmodfile_create()
    test_hydmodfile_load()
    test_hydmodfile_read()
