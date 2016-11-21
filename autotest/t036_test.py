"""
Test loading and preserving existing unit numbers
"""
import os
import flopy

cpth = os.path.join('temp', 'preserve_unitnums')
if not os.path.isdir(cpth):
    os.makedirs(cpth)

def test_unitnums_load_and_write():
    pth = os.path.join('..', 'examples', 'data', 'preserve_unitnums')
    ml = flopy.modflow.Modflow.load('testsfr2_tab.nam', verbose=True, model_ws=pth)

    msg = 'modflow-2005 testsfr2_tab does not have 1 layer, 7 rows, and 100 colummns'
    v = (ml.nlay, ml.nrow, ml.ncol, ml.nper)
    assert v == (1, 7, 100, 50), msg

    ival = 1


if __name__ == '__main__':
    test_unitnums_load_and_write()
