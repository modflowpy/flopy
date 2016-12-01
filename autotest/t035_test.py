"""
Test the lgr model
"""
import os
import flopy

cpth = os.path.join('temp', 't035')


def test_load_and_write():
    pth = os.path.join('..', 'examples', 'data', 'mflgr_v2', 'ex3')
    lgr = flopy.modflowlgr.ModflowLgr.load('ex3.lgr', verbose=True,
                                           model_ws=pth)

    msg = 'modflow-lgr ex3 does not have 2 grids'
    assert lgr.ngrids == 2, msg


if __name__ == '__main__':
    test_load_and_write()
