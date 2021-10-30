import os
import flopy
from ci_framework import baseTestDir, flopyTest

baseDir = baseTestDir(__file__, relPath="temp", verbose=True)


def test_vdf_vsc():
    model_ws = f"{baseDir}_test_vdf_vsc"
    testFramework = flopyTest(verbose=True, testDirs=model_ws)

    nlay = 3
    nrow = 4
    ncol = 5
    nper = 3
    m = flopy.seawat.Seawat(modelname="vdftest", model_ws=model_ws)
    dis = flopy.modflow.ModflowDis(
        m, nlay=nlay, nrow=nrow, ncol=ncol, nper=nper
    )
    vdf = flopy.seawat.SeawatVdf(m)

    # Test different variations of instantiating vsc
    vsc = flopy.seawat.SeawatVsc(m)
    m.write_input()
    m.remove_package("VSC")

    vsc = flopy.seawat.SeawatVsc(m, mt3dmuflg=0)
    m.write_input()
    m.remove_package("VSC")

    vsc = flopy.seawat.SeawatVsc(m, mt3dmuflg=0, mtmutempspec=0)
    m.write_input()
    m.remove_package("VSC")

    vsc = flopy.seawat.SeawatVsc(m, mt3dmuflg=-1)
    m.write_input()
    m.remove_package("VSC")

    vsc = flopy.seawat.SeawatVsc(m, mt3dmuflg=-1, nsmueos=1)
    m.write_input()
    m.remove_package("VSC")

    vsc = flopy.seawat.SeawatVsc(m, mt3dmuflg=1)
    m.write_input()
    m.remove_package("VSC")

    return


if __name__ == "__main__":
    test_vdf_vsc()
