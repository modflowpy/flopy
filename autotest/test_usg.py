import os
from pathlib import Path

import numpy as np
import pytest
from autotest.conftest import get_example_data_path
from flaky import flaky
from modflow_devtools.markers import requires_exe

from flopy.mfusg import MfUsg, MfUsgDisU, MfUsgLpf, MfUsgSms, MfUsgWel
from flopy.modflow import (
    ModflowBas,
    ModflowDis,
    ModflowDrn,
    ModflowGhb,
    ModflowOc,
)
from flopy.utils import TemporalReference, Util2d, Util3d


@pytest.fixture
def mfusg_01A_nestedgrid_nognc_model_path(example_data_path):
    return example_data_path / "mfusg_test" / "01A_nestedgrid_nognc"


@pytest.fixture
def mfusg_rch_evt_model_path(example_data_path):
    return example_data_path / "mfusg_test" / "rch_evt_tests"


@pytest.fixture
def freyberg_usg_model_path(example_data_path):
    return example_data_path / "freyberg_usg"


@requires_exe("mfusg")
def test_usg_disu_load(function_tmpdir, mfusg_01A_nestedgrid_nognc_model_path):
    fname = str(mfusg_01A_nestedgrid_nognc_model_path / "flow.disu")
    assert os.path.isfile(fname), f"disu file not found {fname}"

    # Create the model
    m = MfUsg(modelname="usgload", verbose=True)

    # Load the disu file
    disu = MfUsgDisU.load(fname, m)
    assert isinstance(disu, MfUsgDisU)

    # Change where model files are written
    m.model_ws = str(function_tmpdir)

    # Write the disu file
    disu.write_file()
    assert Path(function_tmpdir / f"{m.name}.{m.disu.extension[0]}").is_file()

    # Load disu file
    disu2 = MfUsgDisU.load(fname, m)
    for (key1, value1), (key2, value2) in zip(
        disu2.__dict__.items(), disu.__dict__.items()
    ):
        if isinstance(value1, (Util2d, Util3d)):
            assert np.array_equal(value1.array, value2.array)
        elif isinstance(
            value1, list
        ):  # this is for the jagged _get_neighbours list
            assert np.all([np.all(v1 == v2) for v1, v2 in zip(value1, value2)])
        elif not isinstance(value1, TemporalReference):
            assert value1 == value2


@requires_exe("mfusg")
def test_usg_sms_load(function_tmpdir, mfusg_01A_nestedgrid_nognc_model_path):
    fname = str(mfusg_01A_nestedgrid_nognc_model_path / "flow.sms")
    assert os.path.isfile(fname), f"sms file not found {fname}"

    # Create the model
    m = MfUsg(modelname="usgload", verbose=True)

    # Load the sms file
    sms = MfUsgSms.load(fname, m)
    assert isinstance(sms, MfUsgSms)

    # Change where model files are written
    m.model_ws = str(function_tmpdir)

    # Write the sms file
    sms.write_file()
    assert Path(function_tmpdir / f"{m.name}.{m.sms.extension[0]}").is_file()

    # Load sms file
    sms2 = MfUsgSms.load(fname, m)
    for (key1, value1), (key2, value2) in zip(
        sms2.__dict__.items(), sms.__dict__.items()
    ):
        assert (
            value1 == value2
        ), f"key1 {key1}, value 1 {value1} != key2 {key2} value 2 {value2}"


@requires_exe("mfusg")
def test_usg_model(function_tmpdir):
    mf = MfUsg(
        version="mfusg",
        structured=True,
        model_ws=str(function_tmpdir),
        modelname="simple",
        exe_name="mfusg",
    )
    dis = ModflowDis(mf, nlay=1, nrow=11, ncol=11)
    bas = ModflowBas(mf)
    lpf = MfUsgLpf(mf)
    wel = MfUsgWel(mf, stress_period_data={0: [[0, 5, 5, -1.0]]})
    ghb = ModflowGhb(
        mf,
        stress_period_data={
            0: [
                [0, 0, 0, 1.0, 1000.0],
                [0, 9, 9, 0.0, 1000.0],
            ]
        },
    )
    oc = ModflowOc(mf)
    sms = MfUsgSms(mf, options="complex")

    # run with defaults
    mf.write_input()
    success, buff = mf.run_model()
    assert success

    # try different complexity options; all should run successfully
    for complexity in ["simple", "moderate", "complex"]:
        print(f"testing MFUSG with sms complexity: {complexity}")
        sms = MfUsgSms(mf, options=complexity)
        sms.write_file()
        success, buff = mf.run_model()
        assert success, f"{mf.name} did not run"


@requires_exe("mfusg")
def test_usg_load_01B(function_tmpdir, mfusg_01A_nestedgrid_nognc_model_path):
    print(
        "testing 1-layer unstructured mfusg model "
        "loading: 01A_nestedgrid_nognc.nam"
    )

    fname = str(mfusg_01A_nestedgrid_nognc_model_path / "flow.nam")
    assert os.path.isfile(fname), f"nam file not found {fname}"

    # Create the model
    m = MfUsg(
        modelname="usgload_1b",
        verbose=True,
        model_ws=str(function_tmpdir),
    )

    # Load the model, with checking
    m = m.load(fname, check=True, model_ws=str(function_tmpdir))

    # assert disu, lpf, bas packages have been loaded
    msg = "flopy failed on loading mfusg disu package"
    assert isinstance(m.disu, MfUsgDisU), msg
    msg = "flopy failed on loading mfusg lpf package"
    assert isinstance(m.lpf, MfUsgLpf), msg
    msg = "flopy failed on loading mfusg bas package"
    assert isinstance(m.bas6, ModflowBas), msg
    msg = "flopy failed on loading mfusg oc package"
    assert isinstance(m.oc, ModflowOc), msg
    msg = "flopy failed on loading mfusg sms package"
    assert isinstance(m.sms, MfUsgSms), msg


@requires_exe("mfusg")
def test_usg_load_45usg(function_tmpdir, example_data_path):
    print("testing 3-layer unstructured mfusg model loading: 45usg.nam")

    pthusgtest = str(example_data_path / "mfusg_test" / "45usg")
    fname = os.path.abspath(os.path.join(pthusgtest, "45usg.nam"))
    assert os.path.isfile(fname), f"nam file not found {fname}"

    # Create the model
    m = MfUsg(modelname="45usg", verbose=True, model_ws=str(function_tmpdir))

    # Load the model, with checking.
    m = m.load(fname, check=True, model_ws=str(function_tmpdir))

    # assert disu, lpf, bas packages have been loaded
    msg = "flopy failed on loading mfusg disu package"
    assert isinstance(m.disu, MfUsgDisU), msg
    msg = "flopy failed on loading mfusg lpf package"
    assert isinstance(m.lpf, MfUsgLpf), msg
    msg = "flopy failed on loading mfusg bas package"
    assert isinstance(m.bas6, ModflowBas), msg
    msg = "flopy failed on loading mfusg oc package"
    assert isinstance(m.oc, ModflowOc), msg
    msg = "flopy failed on loading mfusg sms package"
    assert isinstance(m.sms, MfUsgSms), msg
    msg = "flopy failed on loading mfusg drn package"
    assert isinstance(m.drn, ModflowDrn), msg
    msg = "flopy failed on loading mfusg wel package"
    assert isinstance(m.wel, MfUsgWel), msg


@requires_exe("mfusg")
def test_usg_rch_evt_models01(function_tmpdir, mfusg_rch_evt_model_path):
    # this test has RCH nrchop == 1, and EVT nevtop == 1
    print(
        "testing unstructured mfusg RCH nrchop == 1, and "
        "EVT nevtop == 1: usg_rch_evt.nam"
    )

    nam = "usg_rch_evt.nam"
    m = MfUsg.load(
        nam, model_ws=str(mfusg_rch_evt_model_path), exe_name="mfusg"
    )
    m.riv.check()

    m.model_ws = str(function_tmpdir)
    m.write_input()
    success, buff = m.run_model()
    assert success


@requires_exe("mfusg")
def test_usg_rch_evt_models02(function_tmpdir, mfusg_rch_evt_model_path):
    # this test has RCH nrchop == 2, and EVT nevtop == 2
    print(
        "testing unstructured mfusg RCH nrchop == 2, "
        "and EVT nevtop == 2: usg_rch_evt_nrchop2.nam"
    )

    nam = "usg_rch_evt_nrchop2.nam"
    m = MfUsg.load(
        nam, model_ws=str(mfusg_rch_evt_model_path), exe_name="mfusg"
    )

    m.model_ws = str(function_tmpdir)
    m.write_input()
    success, buff = m.run_model()
    assert success


@requires_exe("mfusg")
def test_usg_rch_evt_models02a(function_tmpdir, mfusg_rch_evt_model_path):
    # this test has RCH nrchop == 2, and EVT nevtop == 2
    print(
        "testing unstructured mfusg RCH nrchop == 2, "
        "and EVT nevtop == 2, but with fewer irch nodes: "
        "than in nodelay[0] usg_rch_evt_nrchop2.nam"
    )

    nam = "usg_rch_evt_nrchop2a.nam"
    m = MfUsg.load(
        nam, model_ws=str(mfusg_rch_evt_model_path), exe_name="mfusg"
    )

    m.model_ws = str(function_tmpdir)
    m.write_input()
    success, buff = m.run_model()
    assert success


@requires_exe("mfusg")
def test_usg_ss_to_tr(function_tmpdir, mfusg_01A_nestedgrid_nognc_model_path):
    # Test switching steady model to transient
    # https://github.com/modflowpy/flopy/issues/1187

    nam = "flow.nam"
    m = MfUsg.load(
        nam,
        model_ws=str(mfusg_01A_nestedgrid_nognc_model_path),
        exe_name="mfusg",
    )

    m.model_ws = str(function_tmpdir)
    m.disu.steady = [False]
    m.write_input()
    success, buff = m.run_model()
    assert success

    m = MfUsg.load(nam, model_ws=str(function_tmpdir), exe_name="mfusg")
    success, buff = m.run_model()
    assert success


@requires_exe("mfusg")
def test_usg_str(function_tmpdir, mfusg_rch_evt_model_path):
    # test mfusg model with str package
    print("testing unstructured mfusg with STR: usg_rch_evt_str.nam")

    nam = "usg_rch_evt_str.nam"
    m = MfUsg.load(
        nam, model_ws=str(mfusg_rch_evt_model_path), exe_name="mfusg"
    )

    m.model_ws = str(function_tmpdir)
    m.write_input()
    success, buff = m.run_model()
    assert success


@requires_exe("mfusg")
def test_usg_lak(function_tmpdir, mfusg_rch_evt_model_path):
    # test mfusg model with lak package
    print("testing unstructured mfusg with LAK: usg_rch_evt_lak.nam")

    nam = "usg_rch_evt_lak.nam"
    m = MfUsg.load(
        nam, model_ws=str(mfusg_rch_evt_model_path), exe_name="mfusg"
    )

    m.model_ws = str(function_tmpdir)
    m.write_input()
    success, buff = m.run_model()
    assert success


# occasional forrtl: error (72): floating overflow
@flaky
@requires_exe("mfusg")
@pytest.mark.slow
def test_freyburg_usg(function_tmpdir, freyberg_usg_model_path):
    # test mfusg model with rch nrchop 3 / freyburg.usg
    print("testing usg nrchop 3: freyburg.usg.nam")

    nam = "freyberg.usg.nam"
    m = MfUsg.load(
        nam, model_ws=str(freyberg_usg_model_path), exe_name="mfusg"
    )

    m.model_ws = str(function_tmpdir)
    m.write_input()
    success, buff = m.run_model()
    assert success


# occasional forrtl: error (72): floating overflow
@flaky
@requires_exe("mfusg")
@pytest.mark.slow
def test_freyburg_usg_external(function_tmpdir, freyberg_usg_model_path):
    # test mfusg model after setting all files to external form
    print("testing usg external files: freyburg.usg.nam")

    nam = "freyberg.usg.nam"
    m = MfUsg.load(
        nam, model_ws=str(freyberg_usg_model_path), exe_name="mfusg"
    )
    # convert to all open/close
    ext_model_ws = str(function_tmpdir)
    m.external_path = "."
    # reduce nper to speed this test up a bit
    m.disu.nper = 3
    # change dir and write
    m.change_model_ws(ext_model_ws, reset_external=True)
    m.write_input()
    success, buff = m.run_model()
    assert success


@requires_exe("mfusg")
def test_flat_array_to_util3d_usg(function_tmpdir, freyberg_usg_model_path):
    # test mfusg model package constructor with flat arrays
    # for layer-based properties
    print("testing usg flat arrays to layer property constructor")

    nam = "freyberg.usg.nam"
    m = MfUsg.load(
        nam, model_ws=str(freyberg_usg_model_path), exe_name="mfusg"
    )

    custom_array = m.lpf.hk.array

    msg = "lpf.hk.array should return a flat array disu.nodes long"
    assert custom_array.ndim == 1 and custom_array.size == m.disu.nodes, msg

    # modify hk array and check updates values are in the lpf
    custom_array[m.disu.nodelay[1] : m.disu.nodelay[1] + 2] = 999.9
    lpf_new = MfUsgLpf(m, hk=custom_array)

    msg = "modified flat array provided to lpf constructor is not updated as expected."
    assert (lpf_new.hk[1][:2] == 999.9).all(), msg

    # ensure we can still write the lpf file
    m.model_ws = str(function_tmpdir)
    m.write_input()


@requires_exe("mfusg")
@pytest.mark.slow
@pytest.mark.parametrize(
    "fpth",
    [str(p) for p in (get_example_data_path() / "mfusg_test").rglob("*.nam")],
)
def test_load_usg(function_tmpdir, fpth):
    namfile = Path(fpth)

    m = MfUsg.load(
        namfile,
        model_ws=str(namfile.parent),
        verbose=True,
        check=False,
    )
    assert m, f"Could not load namefile {namfile}"
    assert m.load_fail is False

    m.change_model_ws(str(function_tmpdir))
    m.write_input()
