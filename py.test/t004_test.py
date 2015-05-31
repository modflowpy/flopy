import numpy as np
import flopy
from flopy.utils.util_array import util_2d

def test_util2d():
    ml = flopy.modflow.Modflow()
    u2d = util_2d(ml, (10,10), np.float32, 10.)
    a1 = u2d.array
    a2 = np.ones((10, 10), dtype=np.float32) * 10.
    assert np.array_equal(a1, a2)
    return

if __name__ == '__main__':
    test_util2d()
