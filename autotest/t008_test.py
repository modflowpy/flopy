"""
Try to load all of the MODFLOW examples in ../examples/data/mf2005_test.
These are the examples that are distributed with MODFLOW-2005.
"""

import os
import flopy

tpth = os.path.join('temp', 't008')

# make the directory if it does not exist
if not os.path.isdir(tpth):
    os.makedirs(tpth)

pth = os.path.join('..', 'examples', 'data', 'mf2005_test')
namfiles = [namfile for namfile in os.listdir(pth) if namfile.endswith('.nam')]

test_nwt_pth = os.path.join("..", "examples", "data", "nwt_test")
nwt_files = [os.path.join(test_nwt_pth, f) for f in os.listdir(test_nwt_pth)
             if f.endswith('.nwt')]

nwt_nam = [os.path.join(test_nwt_pth, f) for f in os.listdir(test_nwt_pth)
           if f.endswith('.nam')]


def load_model(namfile):
    m = flopy.modflow.Modflow.load(namfile, model_ws=pth,
                                   version='mf2005', verbose=True)
    assert m, 'Could not load namefile {}'.format(namfile)
    assert m.load_fail is False


def load_only_bas6_model(namfile):
    m = flopy.modflow.Modflow.load(namfile, model_ws=pth,
                                   version='mf2005', verbose=True,
                                   load_only=["bas6"], check=False)
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


def test_nwt_model_load():
    for f in nwt_nam:
        yield load_nwt_model, f


def load_nwt(nwtfile):
    ml = flopy.modflow.Modflow(model_ws=tpth, version='mfnwt')
    fn = os.path.join(tpth, '{}.nwt'.format(ml.name))
    if os.path.isfile(fn):
        os.remove(fn)
    if 'fmt.' in nwtfile.lower():
        ml.array_free_format = False
    else:
        ml.array_free_format = True

    nwt = flopy.modflow.ModflowNwt.load(nwtfile, ml)
    msg = '{} load unsuccessful'.format(os.path.basename(nwtfile))
    assert isinstance(nwt, flopy.modflow.ModflowNwt), msg

    nwt.write_file()
    msg = '{} write unsuccessful'.format(os.path.basename(nwtfile))
    assert os.path.isfile(fn), msg

    ml2 = flopy.modflow.Modflow(model_ws=tpth, version='mfnwt')
    nwt2 = flopy.modflow.ModflowNwt.load(fn, ml2)
    lst = [a for a in dir(nwt) if
           not a.startswith('__') and not callable(getattr(nwt, a))]
    for l in lst:
        msg = '{} data '.format(l) + 'instantiated from ' + \
              '{} load '.format(os.path.basename(nwtfile)) + \
              ' is not the same as written to {}'.format(os.path.basename(fn))
        assert nwt2[l] == nwt[l], msg


def load_nwt_model(nfile):
    f = os.path.basename(nfile)
    model_ws = os.path.dirname(nfile)
    ml = flopy.modflow.Modflow.load(f, model_ws=model_ws)
    msg = 'Error: flopy model instance was not created'
    assert isinstance(ml, flopy.modflow.Modflow), msg

    # change the model work space and rewrite the files
    ml.change_model_ws(tpth)
    ml.write_input()

    # reload the model that was just written
    ml2 = flopy.modflow.Modflow.load(f, model_ws=tpth)

    # check that the data are the same
    for pn in ml.get_package_list():
        p = ml.get_package(pn)
        p2 = ml2.get_package(pn)
        lst = [a for a in dir(p) if
               not a.startswith('__') and not callable(getattr(p, a))]
        for l in lst:
            msg = '{}.{} data '.format(pn, l) + \
                  'instantiated from {} load '.format(model_ws) + \
                  ' is not the same as written to {}'.format(tpth)
            assert p[l] == p2[l], msg


if __name__ == '__main__':
    for namfile in namfiles:
        load_model(namfile)
    for namfile in namfiles:
        load_only_bas6_model(namfile)
    for fnwt in nwt_nam:
        load_nwt_model(fnwt)
    for fnwt in nwt_files:
        load_nwt(fnwt)
