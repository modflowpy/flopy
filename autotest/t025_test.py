"""
Some basic tests for LAKE load.
"""

import os
import flopy
import numpy as np

path = os.path.join('..', 'examples', 'data', 'mf2005_test')
cpth = os.path.join('temp')

mf_items = ['l1b2k_bath.nam', 'lakeex3.nam', 'l1b2k.nam', 'l1a2k.nam']
pths = [path, path, path, path]

#mf_items = ['l1b2k_bath.nam']
#pths = [path]

run = False

def load_lak(mfnam, pth):
    m = flopy.modflow.Modflow.load(mfnam, model_ws=pth, verbose=True)
    assert m.load_fail is False
    m.exe_name = 'mf2005'
    if run:
        try:
            success, buff = m.run_model(silent=True)
        except:
            pass
        assert success, 'base model run did not terminate successfully'

    # rewrite files
    #m.array_free_format = True
    m.change_model_ws(cpth, reset_external=True)
    m.write_input()
    if run:
        try:
            success, buff = m.run_model(silent=True)
        except:
            pass
        assert success, 'base model run did not terminate successfully'


    # load files
    pth = os.path.join(cpth, '{}.lak'.format(m.name))
    #lak = flopy.modflow.ModflowLak.load(pth, m)
    #for name in str2.dtype.names:
    #    assert np.array_equal(str2.stress_period_data[0][name], m.str.stress_period_data[0][name]) is True
    #for name in str2.dtype2.names:
    #    assert np.array_equal(str2.segment_data[0][name], m.str.segment_data[0][name]) is True
    return




def test_mf2005load():
    for namfile, pth in zip(mf_items, pths):
        yield load_lak, namfile, pth
    return


if __name__ == '__main__':
    for namfile, pth in zip(mf_items, pths):
        load_lak(namfile, pth)
