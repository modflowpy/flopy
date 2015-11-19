"""
Try to load all of the MODFLOW examples in ../examples/data/mf2005_test.
These are the examples that are distributed with MODFLOW-2005.
"""

import os
import flopy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

pth = os.path.join('..', 'examples', 'data', 'mf2005_test')
namfiles = [namfile for namfile in os.listdir(pth) if namfile.endswith('.nam')]


def load_model(namfile):
    m = flopy.modflow.Modflow.load(namfile, model_ws=pth,
                                   version='mf2005', verbose=True)
    assert m, 'Could not load namefile {}'.format(namfile)
    assert m.load_fail is False
    #m.plot()
    #plt.close("all")


def test_modflow_load():
    for namfile in namfiles:
        yield load_model, namfile
    return


if __name__ == '__main__':
    for namfile in namfiles:
        load_model(namfile)
