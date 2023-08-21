import os

import numpy as np
import pytest
from modflow_devtools.markers import requires_exe, requires_pkg

from flopy.modflow import (
    Modflow,
    ModflowBas,
    ModflowDis,
    ModflowLpf,
    ModflowOc,
    ModflowPcg,
    ModflowWel,
)
from flopy.utils.compare import compare_budget, compare_heads


@requires_exe("mf2005")
@pytest.mark.regression
def test_binary_well(function_tmpdir):
    nlay = 3
    nrow = 3
    ncol = 3
    mfnam = "t1"
    ml = Modflow(
        modelname=mfnam,
        model_ws=function_tmpdir,
        verbose=True,
        exe_name="mf2005",
    )
    dis = ModflowDis(
        ml, nlay=nlay, nrow=nrow, ncol=ncol, top=0, botm=[-1.0, -2.0, -3.0]
    )
    ibound = np.ones((nlay, nrow, ncol), dtype=int)
    ibound[0, 1, 1] = 0
    ibound[0, 0, -1] = -1
    bas = ModflowBas(ml, ibound=ibound)
    lpf = ModflowLpf(ml, ipakcb=102)
    wd = ModflowWel.get_empty(ncells=2, aux_names=["v1", "v2"])
    wd["k"][0] = 2
    wd["i"][0] = 2
    wd["j"][0] = 2
    wd["flux"][0] = -1000.0
    wd["v1"][0] = 1.0
    wd["v2"][0] = 2.0
    wd["k"][1] = 2
    wd["i"][1] = 1
    wd["j"][1] = 1
    wd["flux"][1] = -500.0
    wd["v1"][1] = 200.0
    wd["v2"][1] = 100.0
    wel_data = {0: wd}
    wel = ModflowWel(ml, stress_period_data=wel_data, dtype=wd.dtype)
    oc = ModflowOc(ml)
    pcg = ModflowPcg(ml)

    ml.write_input()

    # run the modflow-2005 model
    success, buff = ml.run_model(silent=False)
    assert success, "could not run MODFLOW-2005 model"
    fn0 = os.path.join(function_tmpdir, f"{mfnam}.nam")

    # load the model
    m = Modflow.load(
        f"{mfnam}.nam",
        model_ws=function_tmpdir,
        verbose=True,
        exe_name="mf2005",
    )

    wl = m.wel.stress_period_data[0]
    assert np.array_equal(wel.stress_period_data[0], wl), (
        "previous well package stress period data does not match "
        "stress period data loaded."
    )

    # change model work space
    pth = os.path.join(function_tmpdir, "flopy")
    m.change_model_ws(new_pth=pth)

    # remove the existing well package
    m.remove_package("WEL")

    # recreate well package with binary output
    wel = ModflowWel(
        m, stress_period_data=wel_data, binary=True, dtype=wd.dtype
    )

    # write the model to the new path
    m.write_input()

    # run the new modflow-2005 model
    success, buff = m.run_model()
    assert success, "could not run the new MODFLOW-2005 model"
    fn1 = os.path.join(pth, f"{mfnam}.nam")

    # compare the files
    fsum = os.path.join(
        function_tmpdir, f"{os.path.splitext(mfnam)[0]}.head.out"
    )
    assert compare_heads(fn0, fn1, outfile=fsum), "head comparison failure"

    fsum = os.path.join(
        function_tmpdir, f"{os.path.splitext(mfnam)[0]}.budget.out"
    )
    assert compare_budget(
        fn0, fn1, max_incpd=0.1, max_cumpd=0.1, outfile=fsum
    ), "budget comparison failure"
