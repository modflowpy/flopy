import flopy
import os
from nose.tools import raises


def test_loadfreyberg():
    cwd = os.getcwd()
    pth = os.path.join('..', 'examples', 'data', 'freyberg')
    print(pth)
    assert (os.path.isdir(pth))
    os.chdir(pth)
    namefile = 'freyberg.nam'
    ml = flopy.modflow.Modflow.load(namefile, verbose=True)
    os.chdir(cwd)
    assert isinstance(ml, flopy.modflow.Modflow)
    assert ml.load_fail is False

    return


def test_loadoahu():
    cwd = os.getcwd()
    pth = os.path.join('..', 'examples', 'data', 'parameters')
    assert (os.path.isdir(pth))
    os.chdir(pth)
    namefile = 'Oahu_01.nam'
    ml = flopy.modflow.Modflow.load(namefile, verbose=True)
    os.chdir(cwd)
    assert isinstance(ml, flopy.modflow.Modflow)
    assert ml.load_fail is False

    return


def test_loadtwrip():
    cwd = os.getcwd()
    pth = os.path.join('..', 'examples', 'data', 'parameters')
    assert (os.path.isdir(pth))
    os.chdir(pth)
    namefile = 'twrip.nam'
    ml = flopy.modflow.Modflow.load(namefile, verbose=True)
    os.chdir(cwd)
    assert isinstance(ml, flopy.modflow.Modflow)
    assert ml.load_fail is False

    return


def test_loadoc():
    ws = os.path.join('temp', 't003')
    ml = flopy.modflow.Modflow(model_ws=ws)
    fpth = os.path.join('..', 'examples', 'data', 'mf2005_test', 'fhb.dis')
    dis = flopy.modflow.ModflowDis.load(fpth, ml, check=False)
    fpth = os.path.join('..', 'examples', 'data', 'mf2005_test', 'fhb.oc')
    oc = flopy.modflow.ModflowOc.load(fpth, ml, ext_unit_dict=None)

    return


@raises(IOError)
def test_loadoc_lenfail():
    ws = os.path.join('temp', 't003')
    ml = flopy.modflow.Modflow(model_ws=ws)
    fpth = os.path.join('..', 'examples', 'data', 'mf2005_test', 'fhb.oc')
    oc = flopy.modflow.ModflowOc.load(fpth, ml, nper=3, nstp=1, nlay=1)

    return


@raises(ValueError)
def test_loadoc_nperfail():
    ws = os.path.join('temp', 't003')
    ml = flopy.modflow.Modflow(model_ws=ws)
    fpth = os.path.join('..', 'examples', 'data', 'mf2005_test', 'fhb.oc')
    oc = flopy.modflow.ModflowOc.load(fpth, ml, nper=0, nlay=1)

    return


@raises(ValueError)
def test_loadoc_nlayfail():
    ws = os.path.join('temp', 't003')
    ml = flopy.modflow.Modflow(model_ws=ws)
    fpth = os.path.join('..', 'examples', 'data', 'mf2005_test', 'fhb.oc')
    oc = flopy.modflow.ModflowOc.load(fpth, ml, nper=3, nlay=0)

    return


@raises(ValueError)
def test_loadoc_nstpfail():
    ws = os.path.join('temp', 't003')
    ml = flopy.modflow.Modflow(model_ws=ws)
    fpth = os.path.join('..', 'examples', 'data', 'mf2005_test', 'fhb.oc')
    oc = flopy.modflow.ModflowOc.load(fpth, ml, nper=3, nlay=1)

    return


if __name__ == '__main__':
    test_loadoc()
    test_loadoc_nstpfail()
    # test_loadoc_lenfail()
    # test_loadfreyberg()
    # test_loadoahu()
    # test_loadtwrip()
