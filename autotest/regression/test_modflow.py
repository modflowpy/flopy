import filecmp
from os.path import join, splitext
from pathlib import Path
from shutil import copytree

import pytest
from modflow_devtools.markers import requires_exe, requires_pkg

from autotest.conftest import get_example_data_path
from flopy.modflow import Modflow, ModflowOc
from flopy.utils.compare import compare_budget, compare_heads


@pytest.fixture
def mf2005_test_path(example_data_path):
    return example_data_path / "mf2005_test"


@pytest.fixture
def uzf_example_path(example_data_path):
    return example_data_path / "uzf_examples"


@requires_exe("mf2005")
@pytest.mark.slow
@pytest.mark.regression
def test_uzf_unit_numbers(function_tmpdir, uzf_example_path):
    mfnam = "UZFtest2.nam"
    ws = function_tmpdir / "ws"
    copytree(uzf_example_path, ws)

    m = Modflow.load(
        mfnam,
        verbose=True,
        model_ws=ws,
        forgive=False,
        exe_name="mf2005",
    )
    assert m.load_fail is False, "failed to load all packages"

    # reset the oc file
    m.remove_package("OC")
    output = ["save head", "print budget"]
    spd = {}
    for iper in range(1, m.dis.nper):
        for istp in [0, 4, 9, 14]:
            spd[(iper, istp)] = output
    spd[(0, 0)] = output
    spd[(1, 1)] = output
    spd[(1, 2)] = output
    spd[(1, 3)] = output
    oc = ModflowOc(m, stress_period_data=spd)
    oc.write_file()

    success, buff = m.run_model(silent=False)
    assert success, "base model run did not terminate successfully"
    fn0 = join(ws, mfnam)

    # change uzf iuzfcb2 and add binary uzf output file
    m.uzf.iuzfcb2 = 61
    m.add_output_file(m.uzf.iuzfcb2, extension="uzfcb2.bin", package="UZF")

    # change the model work space
    model_ws2 = join(ws, "flopy")
    m.change_model_ws(model_ws2, reset_external=True)

    # rewrite files
    m.write_input()

    # run and compare the output files
    success, buff = m.run_model(silent=False)
    assert success, "new model run did not terminate successfully"
    fn1 = join(model_ws2, mfnam)

    # compare budget terms
    fsum = join(function_tmpdir, f"{splitext(mfnam)[0]}.budget.out")
    success = compare_budget(
        fn0, fn1, max_incpd=0.1, max_cumpd=0.1, outfile=fsum
    )
    assert success, "budget comparison failure"


@requires_exe("mf2005")
@pytest.mark.slow
@pytest.mark.regression
def test_unitnums(function_tmpdir, mf2005_test_path):
    mfnam = "testsfr2_tab.nam"
    ws = function_tmpdir / "ws"
    copytree(mf2005_test_path, ws)

    m = Modflow.load(mfnam, verbose=True, model_ws=ws, exe_name="mf2005")
    assert m.load_fail is False, "failed to load all packages"

    v = (m.nlay, m.nrow, m.ncol, m.nper)
    assert v == (1, 7, 100, 50), (
        "modflow-2005 testsfr2_tab does not have "
        "1 layer, 7 rows, and 100 columns"
    )

    success, buff = m.run_model(silent=False)
    assert success, "base model run did not terminate successfully"
    fn0 = join(ws, mfnam)

    # rewrite files
    model_ws2 = join(ws, "flopy")
    m.change_model_ws(model_ws2, reset_external=True)

    m.write_input()

    success, buff = m.run_model(silent=False)
    assert success, "base model run did not terminate successfully"
    fn1 = join(model_ws2, mfnam)

    fsum = join(ws, f"{splitext(mfnam)[0]}.budget.out")
    success = compare_budget(
        fn0, fn1, max_incpd=0.1, max_cumpd=0.1, outfile=fsum
    )
    assert success, "budget comparison failure"


@requires_exe("mf2005")
@pytest.mark.slow
@pytest.mark.regression
def test_gage(function_tmpdir, example_data_path):
    """
    test043 load and write of MODFLOW-2005 GAGE example problem
    """
    pth = example_data_path / "mf2005_test"
    fpth = join(pth, "testsfr2_tab.nam")
    ws = function_tmpdir / "ws"
    copytree(pth, ws)

    # load the modflow model
    mf = Modflow.load(
        "testsfr2_tab.nam", verbose=True, model_ws=ws, exe_name="mf2005"
    )

    # run the modflow-2005 model
    success, buff = mf.run_model()
    assert success, "could not run original MODFLOW-2005 model"

    files = mf.gage.files

    model_ws2 = join(ws, "flopy")
    mf.change_model_ws(new_pth=model_ws2, reset_external=True)

    # write the modflow model in to the new path
    mf.write_input()

    # run the modflow-2005 model
    success, buff = mf.run_model()
    assert success, "could not run new MODFLOW-2005 model"

    # compare the two results
    for f in files:
        pth0 = join(ws, f)
        pth1 = join(model_ws2, f)
        assert filecmp.cmp(
            pth0, pth1
        ), f'new and original gage file "{f}" are not binary equal.'


__example_data_path = get_example_data_path()


@requires_exe("mf2005")
@pytest.mark.slow
@pytest.mark.regression
@pytest.mark.parametrize(
    "namfile",
    [
        __example_data_path / "pcgn_test" / nf
        for nf in ["twri.nam", "MNW2.nam"]
    ],
)
def test_mf2005pcgn(function_tmpdir, namfile):
    ws = function_tmpdir / "ws"
    copytree(Path(namfile).parent, ws)
    nf = Path(namfile).name

    m = Modflow.load(
        nf,
        model_ws=ws,
        verbose=True,
        exe_name="mf2005",
    )
    assert m.load_fail is False
    if nf in ["twri.nam"]:  # update this list for fixed models
        assert m.free_format_input is False
    else:
        assert m.free_format_input is True

    success, buff = m.run_model(silent=False)
    assert success, "base model run did not terminate successfully"
    fn0 = ws / nf

    # rewrite files
    ws2 = function_tmpdir / "ws2"
    m.change_model_ws(ws2)
    m.write_input()

    success, buff = m.run_model(silent=False)
    assert success, "new model run did not terminate successfully"
    fn1 = ws2 / nf

    fsum = function_tmpdir / f"{Path(namfile).stem}.head.out"
    success = compare_heads(fn0, fn1, outfile=fsum, htol=0.005)
    assert success, "head comparison failure"

    fsum = function_tmpdir / f"{Path(namfile).stem}.budget.out"
    success = compare_budget(
        fn0, fn1, max_incpd=0.1, max_cumpd=0.1, outfile=fsum
    )
    assert success, "budget comparison failure"


@requires_exe("mf2005")
@pytest.mark.slow
@pytest.mark.regression
@pytest.mark.parametrize(
    "namfile", [__example_data_path / "secp" / nf for nf in ["secp.nam"]]
)
def test_mf2005gmg(function_tmpdir, namfile):
    ws = function_tmpdir / "ws"
    copytree(Path(namfile).parent, ws)
    nf = Path(namfile).name

    m = Modflow.load(
        namfile,
        model_ws=ws,
        verbose=True,
        exe_name="mf2005",
    )
    assert m.load_fail is False

    success, buff = m.run_model(silent=False)
    assert success, "base model run did not terminate successfully"
    fn0 = ws / nf

    # rewrite files
    m.change_model_ws(function_tmpdir)
    m.write_input()

    success, buff = m.run_model(silent=False)
    assert success, "new model run did not terminate successfully"
    fn1 = function_tmpdir / nf

    fsum = function_tmpdir / f"{Path(namfile).stem}.head.out"
    success = compare_heads(fn0, fn1, outfile=fsum)
    assert success, "head comparison failure"

    fsum = function_tmpdir / f"{Path(namfile).stem}.budget.out"
    success = compare_budget(
        fn0, fn1, max_incpd=0.1, max_cumpd=0.1, outfile=fsum
    )
    assert success, "budget comparison failure"


@requires_exe("mf2005")
@pytest.mark.regression
@pytest.mark.parametrize(
    "namfile",
    [__example_data_path / "freyberg" / nf for nf in ["freyberg.nam"]],
)
def test_mf2005(function_tmpdir, namfile):
    """
    test045 load and write of MODFLOW-2005 GMG example problem
    """
    compth = function_tmpdir / "flopy"
    ws = function_tmpdir / "ws"
    copytree(Path(namfile).parent, ws)

    m = Modflow.load(
        Path(namfile).name,
        model_ws=ws,
        verbose=True,
        exe_name="mf2005",
    )
    assert m.load_fail is False

    success, buff = m.run_model(silent=False)
    assert success, "base model run did not terminate successfully"
    fn0 = ws / Path(namfile).name

    # change model workspace
    m.change_model_ws(compth)

    # recreate oc file
    oc = m.oc
    unitnumber = [oc.unit_number[0], oc.iuhead, oc.iuddn, 0, 0]
    spd = {(0, 0): ["save head", "save drawdown"]}
    chedfm = "(10(1X1PE13.5))"
    cddnfm = "(10(1X1PE13.5))"
    oc = ModflowOc(
        m,
        stress_period_data=spd,
        chedfm=chedfm,
        cddnfm=cddnfm,
        unitnumber=unitnumber,
    )

    # rewrite files
    m.write_input()

    success, buff = m.run_model()
    assert success, "new model run did not terminate successfully"
    fn1 = compth / Path(namfile).name

    # compare heads
    fsum = ws / f"{Path(namfile).stem}.head.out"
    success = compare_heads(fn0, fn1, outfile=fsum)
    assert success, "head comparison failure"

    # compare heads
    fsum = ws / f"{Path(namfile).stem}.ddn.out"
    success = compare_heads(fn0, fn1, outfile=fsum, text="drawdown")
    assert success, "head comparison failure"

    # compare budgets
    fsum = ws / f"{Path(namfile).stem}.budget.out"
    success = compare_budget(
        fn0, fn1, max_incpd=0.1, max_cumpd=0.1, outfile=fsum
    )
    assert success, "budget comparison failure"


mf2005_namfiles = [
    __example_data_path / "mf2005_test" / nf
    for nf in [
        "fhb.nam",
        "l1a2k.nam",
        "l1b2k.nam",
        "l1b2k_bath.nam",
        "lakeex3.nam",
    ]
]


@requires_exe("mf2005")
@pytest.mark.slow
@pytest.mark.regression
@pytest.mark.parametrize("namfile", mf2005_namfiles)
def test_mf2005fhb(function_tmpdir, namfile):
    ws = function_tmpdir / "ws"
    copytree(Path(namfile).parent, ws)

    m = Modflow.load(
        Path(namfile).name, model_ws=ws, verbose=True, exe_name="mf2005"
    )
    assert m.load_fail is False

    success, buff = m.run_model(silent=False)
    assert success, "base model run did not terminate successfully"
    fn0 = join(ws, Path(namfile).name)

    # rewrite files
    m.change_model_ws(function_tmpdir, reset_external=True)
    m.write_input()

    success, buff = m.run_model()
    assert success, "new model run did not terminate successfully"
    fn1 = join(function_tmpdir, Path(namfile).name)

    fsum = join(ws, f"{Path(namfile).stem}.head.out")
    success = compare_heads(fn0, fn1, outfile=fsum)
    assert success, "head comparison failure"

    fsum = join(ws, f"{Path(namfile).stem}.budget.out")
    success = compare_budget(
        fn0, fn1, max_incpd=0.1, max_cumpd=0.1, outfile=fsum
    )
    assert success, "budget comparison failure"


@requires_exe("mf2005")
@pytest.mark.slow
@pytest.mark.regression
@pytest.mark.parametrize("namfile", mf2005_namfiles)
def test_mf2005_lake(function_tmpdir, namfile, mf2005_test_path):
    ws = function_tmpdir / "ws"

    copytree(mf2005_test_path, ws)
    m = Modflow.load(
        Path(namfile).name,
        model_ws=ws,
        verbose=True,
        forgive=False,
        exe_name="mf2005",
    )
    assert m.load_fail is False

    success, buff = m.run_model(silent=True)
    assert success

    fn0 = join(ws, Path(namfile).name)

    # write free format files - won't run without resetting to free format - evt external file issue
    m.free_format_input = True

    # rewrite files
    model_ws2 = join(ws, "external")
    m.change_model_ws(
        model_ws2, reset_external=True
    )  # l1b2k_bath won't run without this
    m.write_input()

    success, buff = m.run_model()
    assert success
    fn1 = join(model_ws2, Path(namfile).name)

    fsum = join(ws, f"{Path(namfile).stem}.budget.out")

    success = compare_budget(
        fn0, fn1, max_incpd=0.1, max_cumpd=0.1, outfile=fsum
    )
    assert success, "budget comparison failure"
