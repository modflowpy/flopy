import os

import numpy as np
import pytest
from modflow_devtools.markers import requires_exe, requires_pkg

from flopy.mf6.utils import MfGrdFile
from flopy.modflow import (
    Modflow,
    ModflowBas,
    ModflowDis,
    ModflowLpf,
    ModflowOc,
    ModflowPcg,
    ModflowWel,
)
from flopy.utils.compare import (
    _diffmax,
    _difftol,
    compare_budget,
    compare_heads,
)


def test_diffmax():
    a1 = np.array([1, 2, 3])
    a2 = np.array([4, 5, 7])
    d, indices = _diffmax(a1, a2)
    indices = indices[0]  # return value is a tuple of arrays (1 for each dimension)
    assert d == 4
    assert list(indices) == [2]


def test_difftol():
    a1 = np.array([1, 2, 3])
    a2 = np.array([3, 5, 7])
    d, indices = _difftol(a1, a2, 2.5)
    indices = indices[0]  # return value is a tuple of arrays (1 for each dimension)
    assert d == 4
    print(d, indices)
    assert list(indices) == [1, 2]


@pytest.mark.skip(reason="todo")
def test_eval_bud_diff(example_data_path):
    # get ia from grdfile
    mfgrd_test_path = example_data_path / "mfgrd_test"
    grb_path = mfgrd_test_path / "nwtp3.dis.grb"
    grb = MfGrdFile(grb_path, verbose=True)
    ia = grb._datadict["IA"] - 1

    # TODO: create/run minimal model, then compare budget files


@pytest.fixture
def comparison_model_1(function_tmpdir):
    nlay = 3
    nrow = 3
    ncol = 3
    model_name = "t1"

    ml = Modflow(
        modelname=model_name,
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

    # load the model
    m = Modflow.load(
        f"{model_name}.nam",
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
    wel = ModflowWel(m, stress_period_data=wel_data, binary=True, dtype=wd.dtype)

    m.write_input()

    fn1 = function_tmpdir / "flopy" / f"{model_name}.nam"
    fn0 = function_tmpdir / f"{model_name}.nam"
    fhsum = function_tmpdir / f"{os.path.splitext(model_name)[0]}.head.out"
    fbsum = function_tmpdir / f"{os.path.splitext(model_name)[0]}.budget.out"

    return m, fn1, fn0, fhsum, fbsum


@requires_exe("mf2005")
def test_compare_budget_and_heads(comparison_model_1):
    m, fn1, fn0, fhsum, fbsum = comparison_model_1
    success, buff = m.run_model()
    assert success, "could not run the new MODFLOW-2005 model"

    # compare the files
    assert compare_heads(
        fn0, fn1, outfile=fhsum
    ), "head comparison failure (pathlib.Path)"
    assert compare_heads(
        str(fn0), str(fn1), outfile=fhsum
    ), "head comparison failure (str path)"
    assert compare_budget(
        fn0, fn1, max_incpd=0.1, max_cumpd=0.1, outfile=fbsum
    ), "budget comparison failure (pathlib.Path)"
    assert compare_budget(
        str(fn0), str(fn1), max_incpd=0.1, max_cumpd=0.1, outfile=str(fbsum)
    ), "budget comparison failure (str path)"

    # todo test with files1 and files2 arguments


@pytest.mark.skip(reason="todo")
def test_compare_swrbudget():
    pass


@pytest.mark.skip(reason="todo")
def test_compare_heads():
    pass


@pytest.mark.skip(reason="todo")
def test_compare_concs():
    pass


@pytest.mark.skip(reason="todo")
def test_compare_stages():
    pass


@pytest.mark.skip(reason="todo")
def test_compare():
    pass
