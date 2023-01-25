import numpy as np

from flopy.modflow import Modflow, ModflowAg, ModflowDis, ModflowUzf1
from flopy.pakbase import Package


def test_empty_ag_package(function_tmpdir):
    ml = Modflow("agtest", version="mfnwt")
    ag = ModflowAg(ml)
    assert isinstance(ag, Package)


def test_load_write_agwater(function_tmpdir, example_data_path):
    agfile = "Agwater1.ag"
    ml = Modflow("Agwater1", version="mfnwt")
    mpath = example_data_path / "ag_test"
    ag1 = ModflowAg.load(str(mpath / agfile), ml, nper=49, ext_unit_dict={})

    loaded = False
    for pak in ml.packagelist:
        if isinstance(pak, ModflowAg):
            loaded = True
            break

    assert loaded

    ml.change_model_ws(str(function_tmpdir))
    ag1.write_file()

    ml2 = Modflow(
        "Agwater1",
        version="mfnwt",
        model_ws=str(function_tmpdir),
    )
    ag2 = ModflowAg.load(str(function_tmpdir / agfile), ml2, nper=49)

    assert repr(ag1) == repr(ag2), "comparison failed"


def test_load_write_agwater_uzf(function_tmpdir, example_data_path):
    uzffile = "Agwater1.uzf"
    ml = Modflow("Agwater1", version="mfnwt")
    dis = ModflowDis(ml, nlay=1, nrow=15, ncol=10, nper=49)
    mpath = example_data_path / "ag_test"
    uzf1 = ModflowUzf1.load(str(mpath / uzffile), ml)

    loaded = False
    for pak in ml.packagelist:
        if isinstance(pak, ModflowUzf1):
            loaded = True
            break

    assert loaded

    ml.change_model_ws(str(function_tmpdir))
    uzf1.write_file()

    ml2 = Modflow(
        "Agwater1",
        version="mfnwt",
        model_ws=str(function_tmpdir),
    )
    dis2 = ModflowDis(ml2, nlay=1, nrow=15, ncol=10, nper=49)
    uzf2 = ModflowUzf1.load(str(function_tmpdir / uzffile), ml2)

    assert np.allclose(
        uzf1.air_entry.array, uzf2.air_entry.array
    ), "Air entry pressure array comparison failed"
    assert np.allclose(
        uzf1.hroot.array, uzf2.hroot.array
    ), "root pressure array comparison failed"
    assert np.allclose(
        uzf1.rootact.array, uzf2.rootact.array
    ), "root activity array comparison failed"
