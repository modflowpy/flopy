import os
import flopy
import platform
import shutil
import numpy as np


mpth = os.path.join('..', 'examples', 'data', 'ag_test')
opth = os.path.join('temp', 't071')
if not os.path.exists(opth):
    os.makedirs(opth)


def test_empty_ag_package():
    ml = flopy.modflow.Modflow("agtest", version='mfnwt')
    ag = flopy.modflow.ModflowAg(ml)

    if not isinstance(ag, flopy.pakbase.Package):
        raise Exception


def test_load_write_agwater():
    agfile = "Agwater1.ag"
    ml = flopy.modflow.Modflow("Agwater1", version='mfnwt')
    ag1 = flopy.modflow.ModflowAg.load(os.path.join(mpth, agfile),
                                       ml, nper=49, ext_unit_dict={})

    loaded = False
    for pak in ml.packagelist:
        if isinstance(pak, flopy.modflow.ModflowAg):
            loaded = True
            break

    if not loaded:
        raise AssertionError("ModflowAg package not loaded")

    ml.change_model_ws(opth)
    ag1.write_file()

    ml2 = flopy.modflow.Modflow("Agwater1", version='mfnwt', model_ws=opth)
    ag2 = flopy.modflow.ModflowAg.load(os.path.join(opth, agfile),
                                       ml2, nper=49)

    if repr(ag1) != repr(ag2):
        raise AssertionError("Ag package comparison failed")

def test_load_write_agwater_uzf():
    uzffile = "Agwater1.uzf"
    ml = flopy.modflow.Modflow("Agwater1", version='mfnwt')
    dis = flopy.modflow.ModflowDis(ml, nlay=1, nrow=15, ncol=10, nper=49)
    uzf1 = flopy.modflow.ModflowUzf1.load(os.path.join(mpth, uzffile), ml)

    loaded = False
    for pak in ml.packagelist:
        if isinstance(pak, flopy.modflow.ModflowUzf1):
            loaded = True
            break

    if not loaded:
        raise AssertionError("ModflowUzf1 package not loaded")

    ml.change_model_ws(opth)
    uzf1.write_file()

    ml2 = flopy.modflow.Modflow("Agwater1", version='mfnwt', model_ws=opth)
    dis2 = flopy.modflow.ModflowDis(ml2, nlay=1, nrow=15, ncol=10, nper=49)
    uzf2 = flopy.modflow.ModflowUzf1.load(os.path.join(opth, uzffile), ml2)

    if not np.allclose(uzf1.air_entry.array, uzf2.air_entry.array):
        raise AssertionError("Air entry pressure array comparison failed")

    if not np.allclose(uzf1.hroot.array, uzf2.hroot.array):
        raise AssertionError("root pressure array comparison failed")

    if not np.allclose(uzf1.rootact.array, uzf2.rootact.array):
        raise AssertionError("root activity array comparison failed")


if __name__ == "__main__":
    test_empty_ag_package()
    test_load_write_agwater()
    test_load_write_agwater_uzf()