import os
import numpy as np
import flopy
from flopy.utils.util_array import Util2d, Util3d, Transient2d

out_dir = "temp"
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

def test_transient2d():
    ml = flopy.modflow.Modflow()
    dis = flopy.modflow.ModflowDis(ml,nlay=10,nrow=10,ncol=10,nper=3)
    t2d = Transient2d(ml, (10, 10), np.float32, 10., "fake")
    a1 = t2d.array
    assert a1.shape == (3,10,10), a1.shape
    t2d.cnstnt = 2.0
    assert np.array_equal(t2d.array,np.zeros((3,10,10))+20.0)


def test_util2d():
    ml = flopy.modflow.Modflow()
    u2d = Util2d(ml, (10, 10), np.float32, 10.)
    a1 = u2d.array
    a2 = np.ones((10, 10), dtype=np.float32) * 10.
    assert np.array_equal(a1, a2)
    # bin read write test
    fname = os.path.join(out_dir, 'test.bin')
    u2d.write_bin((10, 10), fname, u2d.array)
    a3 = u2d.load_bin((10, 10), fname, u2d.dtype)[1]
    assert np.array_equal(a3, a1)
    # ascii read write test
    fname = os.path.join(out_dir, 'text.dat')
    u2d.write_txt((10, 10), fname, u2d.array)
    a4 = u2d.load_txt((10, 10), fname, u2d.dtype, "(FREE)")
    assert np.array_equal(a1, a4)

    # test view vs copy with .array
    a5 = u2d.array
    a5 += 1
    assert not np.array_equal(a5,u2d.array)

    # Util2d.__mul__() overload
    new_2d = u2d * 2
    assert np.array_equal(new_2d.array, u2d.array * 2)

    # test the cnstnt application
    u2d.cnstnt = 2.0
    a6 = u2d.array
    assert not np.array_equal(a1,a6)
    u2d.write_txt((10, 10), fname, u2d.array)
    a7 = u2d.load_txt((10, 10), fname, u2d.dtype, "(FREE)")
    assert np.array_equal(u2d.array,a7)

    return


def test_util3d():
    ml = flopy.modflow.Modflow()
    u3d = Util3d(ml, (10, 10, 10), np.float32, 10., 'test')
    a1 = u3d.array
    a2 = np.ones((10, 10, 10), dtype=np.float32) * 10.
    assert np.array_equal(a1, a2)

    new_3d = u3d * 2.0
    assert np.array_equal(new_3d.array, u3d.array * 2)

    #test the mult list-based overload for Util3d
    mult = [2.0] * 10
    mult_array = (u3d * mult).array
    assert np.array_equal(mult_array,np.zeros((10,10,10))+20.0)
    u3d.cnstnt = 2.0
    assert not np.array_equal(a1,u3d.array)

    return


if __name__ == '__main__':
    test_transient2d()
    test_util2d()
    test_util3d()
