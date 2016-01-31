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
                                   version='mf2005', verbose=True,load_only=["bas6"])
    assert m, 'Could not load namefile {}'.format(namfile)
    assert m.load_fail is False
    #m.plot()
    #plt.close("all")

def load_only_bas6_model(namfile):
    m = flopy.modflow.Modflow.load(namfile, model_ws=pth,
                                   version='mf2005', verbose=True,
                                   load_only=["bas6"])
    assert m, 'Could not load namefile {}'.format(namfile)
    assert m.load_fail is False


def test_modflow_load():
    for namfile in namfiles:
        yield load_model, namfile
    return

def test_modflow_loadonly():
    for namfile in namfiles:
        yield load_only_bas6_model, namfile
    return


def test_nwt_load():
    ml = flopy.modflow.Modflow(model_ws="temp")
    test_nwt_pth = os.path.join("..","examples","data","nwt_test")
    nwt_files = [os.path.join(test_nwt_pth,f) for f in os.listdir(test_nwt_pth)]
    for nwt_file in nwt_files:
        nwt = flopy.modflow.ModflowNwt.load(nwt_file,ml)
        nwt.write_file()
if __name__ == '__main__':
    test_nwt_load()
    # for namfile in namfiles:
    #     load_model(namfile)
    #     load_only_bas6_model(namfile)
