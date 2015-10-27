"""
Try to load all of the MODFLOW examples in ../examples/data/mf2005_test.
These are the examples that are distributed with MODFLOW-2005.
"""

import os
import flopy

def test_modflow_load():
    pth = os.path.join('..', 'examples', 'data', 'mf2005_test')
    namfiles = [namfile for namfile in os.listdir(pth) if namfile.endswith('.nam')]
    for namfile in namfiles:
        m = flopy.modflow.Modflow.load(namfile, model_ws=pth, 
                                       version='mf2005', verbose=True)
        assert m, 'Could not load namefile {}'.format(namfile)
        assert m.load_fail is False
    return


if __name__ == '__main__':
    test_modflow_load()
