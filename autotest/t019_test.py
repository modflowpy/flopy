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


def test_hydmodfile_slurp():
    import os
    import flopy

    pth = os.path.join('..', 'examples', 'data', 'hydmod_test', 'test1tr.hyd.gitbin')
    h = flopy.utils.HydmodObs(pth, slurp=True)
    assert isinstance(h, flopy.utils.HydmodObs)

    nitems = h.get_num_items()
    assert nitems == 8, 'Not enough records in hydmod file ()...'.format(os.path.basename(pth))

    data = h.slurp()
    assert len(data.dtype.names) == nitems + 1, 'Not enough records in hydmod file ()...'.format(os.path.basename(pth))
    assert data.shape[0] == 101, 'Not enough times in hydmod file ()...'.format(os.path.basename(pth))

    return


def test_hydmodfile_read():
    import os
    import flopy

    pth = os.path.join('..', 'examples', 'data', 'hydmod_test', 'test1tr.hyd.gitbin')
    h = flopy.utils.HydmodObs(pth)
    assert isinstance(h, flopy.utils.HydmodObs)

    times = h.get_time_list()
    assert len(times) == 101, 'Not enough times in hydmod file ()...'.format(os.path.basename(pth))

    nitems = h.get_num_items()
    assert nitems == 8, 'Not enough records in hydmod file ()...'.format(os.path.basename(pth))

    labels = h.get_hyd_labels()
    assert len(labels) == 8, 'Not enough labels in hydmod file ()...'.format(os.path.basename(pth))

    for idx in range(nitems):
        data = h.get_time_gage(idx=idx, lblstrip=0)
        assert data.shape == (len(times), 2), 'data shape is not ({}, 2)'.format(len(times))

    for label in labels:
        data = h.get_time_gage(record=label, lblstrip=0)
        assert data.shape == (len(times), 2), 'data shape is not ({}, 2)'.format(len(times))

    for time in times:
        t, data, success = h.get_values(totim=time)
        assert success, 'Could not access data for time {}'.format(time)
        assert t == time, 'Data time ({}) does not passed time ({})'.format(t, time)
        assert data.shape[0] == nitems, 'Data does not have {} items'.format(nitems)

    for idx in range(len(times)):
        t, data, success = h.get_values(idx=idx)
        assert success, 'Could not access data for time {}'.format(time)
        assert data.shape[0] == nitems, 'Data does not have {} items'.format(nitems)

    return


if __name__ == '__main__':
    test_hydmodfile_create()
    test_hydmodfile_load()
    test_hydmodfile_slurp()
    test_hydmodfile_read()
