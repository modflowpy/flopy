import os
import shutil

import pytest
from modflow_devtools.markers import requires_exe

from flopy.modflow import Modflow
from flopy.utils.compare import compare_budget


@pytest.fixture
def swi_path(example_data_path):
    return example_data_path / "mf2005_test"


@requires_exe("mf2005")
@pytest.mark.slow
@pytest.mark.regression
@pytest.mark.parametrize(
    "namfile", ["swiex1.nam", "swiex2_strat.nam", "swiex3.nam"]
)
def test_mf2005swi2(function_tmpdir, swi_path, namfile):
    name = namfile.replace(".nam", "")
    ws = function_tmpdir / "ws"
    shutil.copytree(swi_path, ws)

    m = Modflow.load(namfile, model_ws=ws, verbose=True, exe_name="mf2005")
    assert m.load_fail is False

    success, buff = m.run_model(silent=False)
    assert success, "base model run did not terminate successfully"
    fn0 = os.path.join(ws, namfile)

    # write free format files -
    # won't run without resetting to free format - evt external file issue
    m.free_format_input = True

    # rewrite files
    model_ws2 = os.path.join(ws, "flopy")
    m.change_model_ws(
        model_ws2, reset_external=True
    )  # l1b2k_bath won't run without this
    m.write_input()

    success, buff = m.run_model()
    assert success, "base model run did not terminate successfully"
    fn1 = os.path.join(model_ws2, namfile)

    fsum = os.path.join(ws, f"{os.path.splitext(namfile)[0]}.budget.out")
    success = compare_budget(
        fn0, fn1, max_incpd=0.1, max_cumpd=0.1, outfile=fsum
    )

    assert success, "budget comparison failure"
