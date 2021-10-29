import pytest
import flopy
import os

from ci_framework import baseTestDir, flopyTest

baseDir = baseTestDir(__file__, relPath="temp", verbose=True)


def test_loadfreyberg():
    cwd = os.getcwd()
    pth = os.path.join("..", "examples", "data", "freyberg")
    print(pth)
    assert os.path.isdir(pth)
    os.chdir(pth)
    namefile = "freyberg.nam"
    ml = flopy.modflow.Modflow.load(namefile, verbose=True)
    os.chdir(cwd)
    assert isinstance(ml, flopy.modflow.Modflow)
    assert ml.load_fail is False

    return


def test_loadoahu():
    pth = os.path.join("..", "examples", "data", "parameters")
    assert os.path.isdir(pth), f"'{pth}' does not exist"

    namefile = "Oahu_01.nam"
    ml = flopy.modflow.Modflow.load(namefile, verbose=True, model_ws=pth)
    assert isinstance(ml, flopy.modflow.Modflow)
    assert ml.load_fail is False

    return


def test_loadtwrip():
    pth = os.path.join("..", "examples", "data", "parameters")
    assert os.path.isdir(pth), f"'{pth}' does not exist"

    namefile = "twrip.nam"
    ml = flopy.modflow.Modflow.load(namefile, verbose=True, model_ws=pth)

    assert isinstance(ml, flopy.modflow.Modflow)
    assert not ml.load_fail

    return


def test_loadtwrip_upw():
    pth = os.path.join("..", "examples", "data", "parameters")
    assert os.path.isdir(pth), f"'{pth}' does not exist"

    namefile = "twrip_upw.nam"
    ml = flopy.modflow.Modflow.load(namefile, verbose=True, model_ws=pth)

    assert isinstance(ml, flopy.modflow.Modflow)
    assert ml.load_fail is False

    return


def test_loadoc():
    ml = flopy.modflow.Modflow()

    fpth = os.path.join("..", "examples", "data", "mf2005_test", "fhb.dis")
    dis = flopy.modflow.ModflowDis.load(fpth, ml, check=False)

    fpth = os.path.join("..", "examples", "data", "mf2005_test", "fhb.oc")
    oc = flopy.modflow.ModflowOc.load(fpth, ml, ext_unit_dict=None)

    return


def test_loadoc_lenfail():
    ml = flopy.modflow.Modflow()

    fpth = os.path.join("..", "examples", "data", "mf2005_test", "fhb.oc")
    with pytest.raises(OSError):
        oc = flopy.modflow.ModflowOc.load(fpth, ml, nper=3, nstp=1, nlay=1)

    return


def test_loadoc_nperfail():
    ml = flopy.modflow.Modflow()

    fpth = os.path.join("..", "examples", "data", "mf2005_test", "fhb.oc")
    with pytest.raises(ValueError):
        oc = flopy.modflow.ModflowOc.load(fpth, ml, nper=0, nlay=1)

    return


def test_loadoc_nlayfail():
    ml = flopy.modflow.Modflow()

    fpth = os.path.join("..", "examples", "data", "mf2005_test", "fhb.oc")
    with pytest.raises(ValueError):
        oc = flopy.modflow.ModflowOc.load(fpth, ml, nper=3, nlay=0)

    return


def test_loadoc_nstpfail():
    ml = flopy.modflow.Modflow()

    fpth = os.path.join("..", "examples", "data", "mf2005_test", "fhb.oc")
    with pytest.raises(ValueError):
        oc = flopy.modflow.ModflowOc.load(fpth, ml, nper=3, nlay=1)

    return


def test_load_nam_mf_nonexistant_file():
    with pytest.raises(OSError):
        ml = flopy.modflow.Modflow.load("nonexistant.nam")


def test_load_nam_mt_nonexistant_file():
    with pytest.raises(OSError):
        ml = flopy.mt3d.Mt3dms.load("nonexistant.nam")


if __name__ == "__main__":
    test_loadoc()
    test_loadoc_nstpfail()
    test_load_nam_mf_nonexistant_file()
    test_load_nam_mt_nonexistant_file()
    test_loadoc_lenfail()
    test_loadfreyberg()
    test_loadoahu()
    test_loadtwrip()
    test_loadtwrip_upw()
