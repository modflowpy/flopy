"""
Some basic tests for STR load.
"""

import sys
import os
import flopy


def load_check(mfnam, model_ws, new_pth):
    m = flopy.modflow.Modflow.load(mfnam, model_ws=model_ws,
                                   verbose=True)
    m.change_model_ws(new_pth)
    m.write_input()
    return m


def test_mf2005load():
    path = os.path.join('..', 'examples', 'data', 'mf2005_test')
    pthgw = os.path.join('..', 'examples', 'groundwater_paper', 'uspb', 'flopy')
    cpth = os.path.join('temp')
    
    mf_items = ['str.nam',
                'twri.nam',
                'twrip.nam']
    
    for i, namf in enumerate(mf_items):
        m = load_check(namf, model_ws=path, new_pth=cpth)
        assert m.load_fail is False



if __name__ == '__main__':
    test_mf2005load()
