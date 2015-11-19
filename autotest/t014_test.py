"""
Some basic tests for STR load.
"""

import os
import flopy
import numpy as np

path = os.path.join('..', 'examples', 'data', 'mf2005_test')
pthgw = os.path.join('..', 'examples', 'groundwater_paper', 'uspb', 'flopy')
cpth = os.path.join('temp')

mf_items = ['str.nam', 'DG.nam']
pths = [path, pthgw]

def load_str(mfnam, pth):
    m = flopy.modflow.Modflow.load(mfnam, model_ws=pth, verbose=True)
    assert m.load_fail is False

    # rewrite files
    m.change_model_ws(cpth)
    m.write_input()

    # load files
    pth = os.path.join(cpth, '{}.str'.format(m.name))
    str2 = flopy.modflow.ModflowStr.load(pth, m)
    for name in str2.dtype.names:
        assert np.array_equal(str2.stress_period_data[0][name], m.str.stress_period_data[0][name]) is True
    for name in str2.dtype2.names:
        assert np.array_equal(str2.segment_data[0][name], m.str.segment_data[0][name]) is True
    return




def test_mf2005load():
    for namfile, pth in zip(mf_items, pths):
        yield load_str, namfile, pth
    return


if __name__ == '__main__':
    for namfile, pth in zip(mf_items, pths):
        load_str(namfile, pth)
