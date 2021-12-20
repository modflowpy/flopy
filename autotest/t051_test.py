import os
import numpy as np
import flopy
from ci_framework import base_test_dir, FlopyTestSetup

base_dir = base_test_dir(__file__, rel_path="temp", verbose=True)


def test_default_oc_stress_period_data():
    model_ws = f"{base_dir}_test_default_oc_stress_period_data"
    test_setup = FlopyTestSetup(verbose=True, test_dirs=model_ws)

    m = flopy.modflow.Modflow(model_ws=model_ws, verbose=True)
    dis = flopy.modflow.ModflowDis(m, nper=10, perlen=10.0, nstp=5)
    bas = flopy.modflow.ModflowBas(m)
    lpf = flopy.modflow.ModflowLpf(m, ipakcb=100)
    wel_data = {0: [[0, 0, 0, -1000.0]]}
    wel = flopy.modflow.ModflowWel(m, ipakcb=101, stress_period_data=wel_data)
    # spd = {(0, 0): ['save head', 'save budget']}
    oc = flopy.modflow.ModflowOc(m, stress_period_data=None)
    spd_oc = oc.stress_period_data
    tups = list(spd_oc.keys())
    kpers = [t[0] for t in tups]
    assert len(kpers) == m.nper
    kstps = [t[1] for t in tups]
    assert max(kstps) == 4
    assert min(kstps) == 4
    m.write_input()


def test_mfcbc():
    model_ws = f"{base_dir}_test_mfcbc"
    test_setup = FlopyTestSetup(verbose=True, test_dirs=model_ws)

    m = flopy.modflow.Modflow(verbose=True, model_ws=model_ws)
    dis = flopy.modflow.ModflowDis(m)
    bas = flopy.modflow.ModflowBas(m)
    lpf = flopy.modflow.ModflowLpf(m, ipakcb=100)
    wel_data = {0: [[0, 0, 0, -1000.0]]}
    wel = flopy.modflow.ModflowWel(m, ipakcb=101, stress_period_data=wel_data)
    spd = {(0, 0): ["save head", "save budget"]}
    oc = flopy.modflow.ModflowOc(m, stress_period_data=spd)
    t = oc.get_budgetunit()
    assert t == [100, 101], f"budget units are {t} not [100, 101]"

    nlay = 3
    nrow = 3
    ncol = 3
    ml = flopy.modflow.Modflow(modelname="t1", model_ws=model_ws, verbose=True)
    dis = flopy.modflow.ModflowDis(
        ml, nlay=nlay, nrow=nrow, ncol=ncol, top=0, botm=[-1.0, -2.0, -3.0]
    )
    ibound = np.ones((nlay, nrow, ncol), dtype=int)
    ibound[0, 1, 1] = 0
    ibound[0, 0, -1] = -1
    bas = flopy.modflow.ModflowBas(ml, ibound=ibound)
    lpf = flopy.modflow.ModflowLpf(ml, ipakcb=102)
    wel_data = {0: [[2, 2, 2, -1000.0]]}
    wel = flopy.modflow.ModflowWel(ml, ipakcb=100, stress_period_data=wel_data)
    oc = flopy.modflow.ModflowOc(ml)

    oc.reset_budgetunit(budgetunit=1053, fname="big.bin")

    msg = (
        f"wel ipakcb ({wel.ipakcb}) "
        "not set correctly to 1053 using oc.resetbudgetunit()"
    )
    assert wel.ipakcb == 1053, msg

    ml.write_input()


if __name__ == "__main__":
    test_mfcbc()
    test_default_oc_stress_period_data()
