import os
import numpy as np
import flopy
from flopy.utils.util_array import Util2d
from ci_framework import base_test_dir, FlopyTestSetup

base_dir = base_test_dir(__file__, rel_path="temp", verbose=True)


def test_rchload():
    model_ws = f"{base_dir}_test_rchload"
    test_setup = FlopyTestSetup(verbose=True, test_dirs=model_ws)

    nlay = 2
    nrow = 3
    ncol = 4
    nper = 2

    # create model 1
    m1 = flopy.modflow.Modflow("rchload1", model_ws=model_ws)
    dis1 = flopy.modflow.ModflowDis(
        m1, nlay=nlay, nrow=nrow, ncol=ncol, nper=nper
    )
    a = np.random.random((nrow, ncol))
    rech1 = Util2d(
        m1, (nrow, ncol), np.float32, a, "rech", cnstnt=1.0, how="openclose"
    )
    rch1 = flopy.modflow.ModflowRch(m1, rech={0: rech1})
    m1.write_input()

    # load model 1
    m1l = flopy.modflow.Modflow.load("rchload1.nam", model_ws=model_ws)
    a1 = rech1.array
    a2 = m1l.rch.rech[0].array
    assert np.allclose(a1, a2)
    a2 = m1l.rch.rech[1].array
    assert np.allclose(a1, a2)

    m2 = flopy.modflow.Modflow("rchload2", model_ws=model_ws)
    dis2 = flopy.modflow.ModflowDis(
        m2, nlay=nlay, nrow=nrow, ncol=ncol, nper=nper
    )
    a = np.random.random((nrow, ncol))
    rech2 = Util2d(
        m2, (nrow, ncol), np.float32, a, "rech", cnstnt=2.0, how="openclose"
    )
    rch2 = flopy.modflow.ModflowRch(m2, rech={0: rech2})
    m2.write_input()

    # load model 2
    m2l = flopy.modflow.Modflow.load("rchload2.nam", model_ws=model_ws)
    a1 = rech2.array
    a2 = m2l.rch.rech[0].array
    assert np.allclose(a1, a2)
    a2 = m2l.rch.rech[1].array
    assert np.allclose(a1, a2)


if __name__ == "__main__":
    test_rchload()
