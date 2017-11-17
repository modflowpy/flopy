# test unstructured binary head file

import os
import numpy as np
import flopy

tpth = os.path.join('temp', 't053')
# make the directory if it does not exist
if not os.path.isdir(tpth):
    os.makedirs(tpth)


def test_headu_file():
    fname = os.path.join('..', 'examples', 'data', 'unstructured',
                         'headu.githds')
    headobj = flopy.utils.HeadUFile(fname)
    assert isinstance(headobj, flopy.utils.HeadUFile)
    assert headobj.nlay == 3

    # ensure recordarray is has correct data
    ra = headobj.recordarray
    assert ra['kstp'].min() == 1
    assert ra['kstp'].max() == 1
    assert ra['kper'].min() == 1
    assert ra['kper'].max() == 5
    assert ra['ncol'].min() == 1
    assert ra['ncol'].max() == 14001
    assert ra['nrow'].min() == 7801
    assert ra['nrow'].max() == 19479

    # read the heads for the last time and make sure they are correct
    data = headobj.get_data()
    assert len(data) == 3
    minmaxtrue = [np.array([-1.4783, -1.0]), np.array([-2.0, -1.0]),
                  np.array([-2.0, -1.01616])]
    for i, d in enumerate(data):
        t1 = np.array([d.min(), d.max()])
        assert np.allclose(t1, minmaxtrue[i])

    return


if __name__ == '__main__':
    test_headu_file()
