import os
import numpy as np
import flopy
from flopy.utils.util_array import util_2d, util_3d


def test_util2d():
    ml = flopy.modflow.Modflow()
    u2d = util_2d(ml, (10, 10), np.float32, 10.)
    a1 = u2d.array
    a2 = np.ones((10, 10), dtype=np.float32) * 10.
    assert np.array_equal(a1, a2)
    # bin read write test
    fname = os.path.join('temp', 'test.bin')
    u2d.write_bin((10, 10), fname, u2d.array)
    a3 = u2d.load_bin((10, 10), fname, u2d.dtype)[1]
    assert np.array_equal(a3, a1)
    # ascii read write test
    fname = os.path.join('temp', 'text.dat')
    u2d.write_txt((10, 10), fname, u2d.array)
    a4 = u2d.load_txt((10, 10), fname, u2d.dtype, "(FREE)")
    assert np.array_equal(a1, a4)

    # util_2d.__mul__() overload
    new_2d = u2d * 2
    assert np.array_equal(new_2d.array, u2d.array * 2)

    return


def test_util3d():
    ml = flopy.modflow.Modflow()
    u3d = util_3d(ml, (10, 10, 10), np.float32, 10., 'test')
    a1 = u3d.array
    a2 = np.ones((10, 10, 10), dtype=np.float32) * 10.
    assert np.array_equal(a1, a2)

    new_3d = u3d * 2.0
    assert np.array_equal(new_3d.array, u3d.array * 2)
    return


if __name__ == '__main__':
    test_util2d()
    test_util3d()
