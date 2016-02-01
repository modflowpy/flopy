"""
Try to load all of the MODFLOW examples in ../examples/data/mf2005_test.
These are the examples that are distributed with MODFLOW-2005.
"""

import os
import flopy
import matplotlib

matplotlib.use('Agg')

pth = os.path.join('..', 'examples', 'data', 'mf2005_test')
namfiles = [namfile for namfile in os.listdir(pth) if namfile.endswith('.nam')]

test_nwt_pth = os.path.join("..", "examples", "data", "nwt_test")
nwt_files = [os.path.join(test_nwt_pth, f) for f in os.listdir(test_nwt_pth)]


def load_model(namfile):
    m = flopy.modflow.Modflow.load(namfile, model_ws=pth,
                                   version='mf2005', verbose=True,
                                   load_only=["bas6"])
    assert m, 'Could not load namefile {}'.format(namfile)
    assert m.load_fail is False
    # m.plot()
    # plt.close("all")


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
    for nwt_file in nwt_files:
        yield load_nwt, nwt_file


def load_nwt(nwtfile):
    ml = flopy.modflow.Modflow(model_ws="temp", version='mfnwt')
    fn = os.path.join('temp', '{}.nwt'.format(ml.name))
    if os.path.isfile(fn):
        os.remove(fn)
    if 'fmt.' in nwtfile.lower():
        ml.set_free_format(value=False)
    else:
        ml.set_free_format(value=True)
    nwt = flopy.modflow.ModflowNwt.load(nwtfile, ml)
    assert isinstance(nwt,
                      flopy.modflow.ModflowNwt), '{} load unsuccessful'.format(
            os.path.basename(nwtfile))
    nwt.write_file()
    assert os.path.isfile(fn), '{} write unsuccessful'.format(
            os.path.basename(nwtfile))
    nwt2 = flopy.modflow.ModflowNwt.load(fn, ml)
    lst = [a for a in dir(nwt) if
           not a.startswith('__') and not callable(getattr(nwt, a))]
    for l in lst:
        assert nwt2[l] == nwt[l], '{} data '.format(l) + \
                                  'instantiated from {} load '.format(
                                          os.path.basename(nwtfile)) + \
                                  ' is not the same as written to {}'.format(
                                          os.path.basename(fn))


if __name__ == '__main__':
    for fnwt in nwt_files:
        load_nwt(fnwt)
        # for namfile in namfiles:
        #     load_model(namfile)
        #     load_only_bas6_model(namfile)
