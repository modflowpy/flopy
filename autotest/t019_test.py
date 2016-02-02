# Test hydmod data readers
def test_hydmodfile_create():
    import os
    import numpy as np
    import flopy

    model_ws = os.path.join('temp')
    if not os.path.exists(model_ws):
        os.makedirs(model_ws)
    m = flopy.modflow.Modflow('test', model_ws=model_ws)
    hyd = flopy.modflow.ModflowHyd(m)
    m.hyd.write_file()
    pth = os.path.join(model_ws, 'test.hyd')
    hydload = flopy.modflow.ModflowHyd.load(pth, m)
    assert np.array_equal(hyd.obsdata, hydload.obsdata), 'Written hydmod data not equal to loaded hydmod data'

    return


def test_hydmodfile_load():
    import os
    import numpy as np
    import flopy

    model = 'test1tr.nam'
    pth = os.path.join('..', 'examples', 'data', 'hydmod_test')
    m = flopy.modflow.Modflow.load(model, version='mf2005', model_ws=pth, verbose=True)
    hydref = m.hyd
    assert isinstance(hydref, flopy.modflow.ModflowHyd), 'Did not load hydmod package...test1tr.hyd'

    model_ws = os.path.join('temp')
    if not os.path.exists(model_ws):
        os.makedirs(model_ws)

    m.change_model_ws(model_ws)
    m.hyd.write_file()

    pth = os.path.join('..', 'examples', 'data', 'hydmod_test', 'test1tr.hyd')
    hydload = flopy.modflow.ModflowHyd.load(pth, m)
    assert np.array_equal(hydref.obsdata, hydload.obsdata), 'Written hydmod data not equal to loaded hydmod data'

    return


def test_hydmodfile_read():
    import os
    import flopy

    pth = os.path.join('..', 'examples', 'data', 'hydmod_test',
                       'test1tr.hyd.gitbin')
    h = flopy.utils.HydmodObs(pth)
    assert isinstance(h, flopy.utils.HydmodObs)

    ntimes = h.get_ntimes()
    assert ntimes == 101, 'Not enough times in hydmod file ()...'.format(os.path.basename(pth))

    times = h.get_times()
    assert len(times) == 101, 'Not enough times in hydmod file ()...'.format(os.path.basename(pth))

    nitems = h.get_nobs()
    assert nitems == 8, 'Not enough records in hydmod file ()...'.format(os.path.basename(pth))

    labels = h.get_obsnames()
    assert len(labels) == 8, 'Not enough labels in hydmod file ()...'.format(os.path.basename(pth))
    print(labels)

    for idx in range(nitems):
        data = h.get_data(idx=idx)
        assert data.shape == (len(times), 1), 'data shape is not ({}, 1)'.format(len(times))

    for label in labels:
        data = h.get_data(obsname=label)
        assert data.shape == (len(times), 1), 'data shape is not ({}, 1)'.format(len(times))

    data = h.get_data()
    assert data.shape == (len(times), 1), 'data shape is not ({}, 1)'.format(len(times))
    assert len(data.dtype.names) == nitems + 1, 'data column length is not {}'.format(len(nitems+1))

    return


if __name__ == '__main__':
    test_hydmodfile_create()
    test_hydmodfile_load()
    test_hydmodfile_read()
