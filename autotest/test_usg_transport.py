import os
from pathlib import Path

import numpy as np
import pytest
from flaky import flaky
from modflow_devtools.markers import requires_exe

from autotest.conftest import get_example_data_path
from flopy.mfusg import (
    MfUsg,
    MfUsgBcf,
    MfUsgBct,
    MfUsgCln,
    MfUsgDdf,
    MfUsgDisU,
    MfUsgDpf,
    MfUsgDpt,
    MfUsgEvt,
    MfUsgGnc,
    MfUsgLak,
    MfUsgLpf,
    MfUsgMdt,
    MfUsgOc,
    MfUsgPcb,
    MfUsgRch,
    MfUsgSms,
    MfUsgWel,
)
from flopy.modflow import (
    ModflowBas,
    ModflowChd,
    ModflowDis,
    ModflowFhb,
    ModflowGhb,
)
from flopy.utils import TemporalReference, Util2d, Util3d


@pytest.fixture
def mfusg_transport_Ex1_model_path(example_data_path: Path):
    return example_data_path / "mfusg_transport" / "Ex1_1D"

@pytest.fixture
def mfusg_transport_Ex2_model_path(example_data_path: Path):
    return example_data_path / "mfusg_transport" / "Ex2_Radial_2D"

@pytest.fixture
def mfusg_transport_Ex3_model_path(example_data_path: Path):
    return example_data_path / "mfusg_transport" / "Ex3_CLN_Conduit"

@pytest.fixture
def mfusg_transport_Ex4_model_path(example_data_path: Path):
    return example_data_path / "mfusg_transport" / "Ex4_Dual_Domain"

@pytest.fixture
def mfusg_transport_Ex5_model_path(example_data_path: Path):
    return example_data_path / "mfusg_transport" / "Ex5_Henry"

@pytest.fixture
def mfusg_transport_Ex6_model_path(example_data_path: Path):
    return example_data_path / "mfusg_transport" / "Ex6_Stallman"

@pytest.fixture
def mfusg_transport_Ex7_model_path(example_data_path: Path):
    return example_data_path / "mfusg_transport" / "Ex7_Matrix_Diffusion"

@pytest.fixture
def mfusg_transport_Ex8_model_path(example_data_path: Path):
    return example_data_path / "mfusg_transport" / "Ex8_Lake"

@pytest.fixture
def mfusg_transport_Ex9_model_path(example_data_path: Path):
    return example_data_path / "mfusg_transport" / "Ex9_PFAS"


@requires_exe("mfusg_gsi")
def test_usg_load_Ex1_1D(function_tmpdir, mfusg_transport_Ex1_model_path):
    print("testing mfusg transport model loading: BTN_Test1.nam")

    fname = mfusg_transport_Ex1_model_path / "BTN_Test1.nam"
    assert os.path.isfile(fname), f"nam file not found {fname}"

    # Create the model
    m = MfUsg.load(
        fname,
        exe_name = "mfusg_gsi",
        verbose=True,
        model_ws=function_tmpdir,
        check=True)

    # assert disu, lpf, bas packages have been loaded
    msg = "flopy failed on loading modflow dis package"
    assert isinstance(m.dis, ModflowDis), msg
    msg = "flopy failed on loading mfusg lpf package"
    assert isinstance(m.lpf, MfUsgLpf), msg
    msg = "flopy failed on loading modflow bas package"
    assert isinstance(m.bas6, ModflowBas), msg
    msg = "flopy failed on loading mfusg oc package"
    assert isinstance(m.oc, MfUsgOc), msg
    msg = "flopy failed on loading mfusg sms package"
    assert isinstance(m.sms, MfUsgSms), msg
    msg = "flopy failed on loading mfusg bct package"
    assert isinstance(m.bct, MfUsgBct), msg
    msg = "flopy failed on loading mfusg pcb package"
    assert isinstance(m.pcb, MfUsgPcb), msg

    m.write_input()
    success, buff = m.run_model()
    msg = "flopy failed on running BTN_Test1.nam"
    assert success, msg

@requires_exe("mfusg_gsi")
def test_usg_load_Ex2_Radial_adv(function_tmpdir, mfusg_transport_Ex2_model_path):
    print("testing mfusg transport model loading: Radial-adv.nam")

    fname = mfusg_transport_Ex2_model_path / "Radial_adv.nam"

    assert os.path.isfile(fname), f"nam file not found {fname}"

    # Create the model
    m = MfUsg.load(
        fname,
        exe_name = "mfusg_gsi",
        verbose=True, 
        model_ws=function_tmpdir,
        check=True)

    # assert disu, lpf, bas packages have been loaded
    msg = "flopy failed on loading modflow dis package"
    assert isinstance(m.dis, ModflowDis), msg
    msg = "flopy failed on loading mfusg lpf package"
    assert isinstance(m.lpf, MfUsgLpf), msg
    msg = "flopy failed on loading mfusg bas package"
    assert isinstance(m.bas6, ModflowBas), msg
    msg = "flopy failed on loading mfusg oc package"
    assert isinstance(m.oc, MfUsgOc), msg
    msg = "flopy failed on loading mfusg sms package"
    assert isinstance(m.sms, MfUsgSms), msg
    msg = "flopy failed on loading modflow chd package"
    assert isinstance(m.chd, ModflowChd), msg
    msg = "flopy failed on loading mfusg bct package"
    assert isinstance(m.bct, MfUsgBct), msg

    m.write_input()
    success, buff = m.run_model()
    assert success

@requires_exe("mfusg_gsi")
def test_usg_load_Ex2_Radial_Disp(function_tmpdir, mfusg_transport_Ex2_model_path):
    print("testing mfusg transport model loading: Radial-dis.nam")

    fname = mfusg_transport_Ex2_model_path / "Radial_dis.nam"

    assert os.path.isfile(fname), f"nam file not found {fname}"

    # Create the model
    m = MfUsg.load(
        fname,
        exe_name = "mfusg_gsi",
        verbose=True, 
        model_ws=function_tmpdir,
        check=True)

    # assert disu, lpf, bas packages have been loaded
    msg = "flopy failed on loading modflow dis package"
    assert isinstance(m.dis, ModflowDis), msg
    msg = "flopy failed on loading mfusg lpf package"
    assert isinstance(m.lpf, MfUsgLpf), msg
    msg = "flopy failed on loading mfusg bas package"
    assert isinstance(m.bas6, ModflowBas), msg
    msg = "flopy failed on loading mfusg oc package"
    assert isinstance(m.oc, MfUsgOc), msg
    msg = "flopy failed on loading mfusg sms package"
    assert isinstance(m.sms, MfUsgSms), msg
    msg = "flopy failed on loading modflow chd package"
    assert isinstance(m.chd, ModflowChd), msg
    msg = "flopy failed on loading mfusg bct package"
    assert isinstance(m.bct, MfUsgBct), msg

    m.write_input()
    success, buff = m.run_model()
    assert success

@requires_exe("mfusg_gsi")
def test_usg_load_Ex3_CLN_Conduit(function_tmpdir, mfusg_transport_Ex3_model_path):
    print("testing mfusg transport model loading: Conduit.nam")

    fname = mfusg_transport_Ex3_model_path / "Conduit/Conduit.nam"
    assert os.path.isfile(fname), f"nam file not found {fname}"

    # Create the model
    m = MfUsg.load(
        fname,
        exe_name = "mfusg_gsi",
        verbose=True, 
        model_ws=function_tmpdir,
        check=True)

    # assert disu, lpf, bas packages have been loaded
    msg = "flopy failed on loading modflow dis package"
    assert isinstance(m.dis, ModflowDis), msg
    msg = "flopy failed on loading modflow bas package"
    assert isinstance(m.bas6, ModflowBas), msg
    msg = "flopy failed on loading modflow chd package"
    assert isinstance(m.chd, ModflowChd), msg
    msg = "flopy failed on loading mfusg bcf package"
    assert isinstance(m.bcf6, MfUsgBcf), msg
    msg = "flopy failed on loading mfusg oc package"
    assert isinstance(m.oc, MfUsgOc), msg
    msg = "flopy failed on loading mfusg sms package"
    assert isinstance(m.sms, MfUsgSms), msg
    msg = "flopy failed on loading mfusg bct package"
    assert isinstance(m.bct, MfUsgBct), msg
    msg = "flopy failed on loading mfusg wel package"
    assert isinstance(m.wel, MfUsgWel), msg
    msg = "flopy failed on loading mfusg cln package"
    assert isinstance(m.cln, MfUsgCln), msg

    m.write_input()
    success, buff = m.run_model()
    msg = "flopy failed on running CLN Conduit.nam"
    assert success, msg

@requires_exe("mfusg_gsi")
def test_usg_load_Ex3_CLN_Conduit_Dispersion(
    function_tmpdir, mfusg_transport_Ex3_model_path):
    print("testing mfusg transport model loading: Conduit.nam")

    fname = mfusg_transport_Ex3_model_path / "Dispersion/Conduit_Dispersion.nam"
    assert os.path.isfile(fname), f"nam file not found {fname}"

    # Create the model
    m = MfUsg.load(
        fname,
        exe_name = "mfusg_gsi",
        verbose=True, 
        model_ws=function_tmpdir,
        check=True)

    # assert disu, lpf, bas packages have been loaded
    msg = "flopy failed on loading modflow dis package"
    assert isinstance(m.dis, ModflowDis), msg
    msg = "flopy failed on loading modflow bas package"
    assert isinstance(m.bas6, ModflowBas), msg
    msg = "flopy failed on loading modflow chd package"
    assert isinstance(m.chd, ModflowChd), msg
    msg = "flopy failed on loading mfusg bcf package"
    assert isinstance(m.bcf6, MfUsgBcf), msg
    msg = "flopy failed on loading mfusg oc package"
    assert isinstance(m.oc, MfUsgOc), msg
    msg = "flopy failed on loading mfusg sms package"
    assert isinstance(m.sms, MfUsgSms), msg
    msg = "flopy failed on loading mfusg bct package"
    assert isinstance(m.bct, MfUsgBct), msg
    msg = "flopy failed on loading mfusg wel package"
    assert isinstance(m.wel, MfUsgWel), msg
    msg = "flopy failed on loading mfusg cln package"
    assert isinstance(m.cln, MfUsgCln), msg

    m.write_input()
    success, buff = m.run_model()
    msg = "flopy failed on running CLN Conduit_Dispersion.nam"
    assert success, msg

@requires_exe("mfusg_gsi")
def test_usg_load_Ex3_CLN_Conduit_Nest(function_tmpdir, mfusg_transport_Ex3_model_path):
    print("testing mfusg transport model loading: Conduit.nam")

    fname = mfusg_transport_Ex3_model_path / "Nest/Conduit_Nest.nam"
    assert os.path.isfile(fname), f"nam file not found {fname}"

    # Create the model
    m = MfUsg.load(
        fname,
        exe_name = "mfusg_gsi",
        verbose=True, 
        model_ws=function_tmpdir,
        check=True)

    # assert disu, lpf, bas packages have been loaded
    msg = "flopy failed on loading mfusg dis package"
    assert isinstance(m.disu, MfUsgDisU), msg
    msg = "flopy failed on loading modflow bas package"
    assert isinstance(m.bas6, ModflowBas), msg
    msg = "flopy failed on loading modflow chd package"
    assert isinstance(m.chd, ModflowChd), msg
    msg = "flopy failed on loading mfusg bcf package"
    assert isinstance(m.bcf6, MfUsgBcf), msg
    msg = "flopy failed on loading mfusg oc package"
    assert isinstance(m.oc, MfUsgOc), msg
    msg = "flopy failed on loading mfusg sms package"
    assert isinstance(m.sms, MfUsgSms), msg
    msg = "flopy failed on loading mfusg bct package"
    assert isinstance(m.bct, MfUsgBct), msg
    msg = "flopy failed on loading mfusg wel package"
    assert isinstance(m.wel, MfUsgWel), msg
    msg = "flopy failed on loading mfusg cln package"
    assert isinstance(m.cln, MfUsgCln), msg

    m.write_input()
    success, buff = m.run_model()
    msg = "flopy failed on running CLN Conduit_Nest.nam"
    assert success, msg

@requires_exe("mfusg_gsi")
def test_usg_load_Ex4_Dual_Domain(function_tmpdir, mfusg_transport_Ex4_model_path):
    print("testing mfusg transport model loading: Conduit.nam")

    fname = mfusg_transport_Ex4_model_path / "DualDomain.nam"
    assert os.path.isfile(fname), f"nam file not found {fname}"

    # Create the model
    m = MfUsg.load(
        fname,
        exe_name = "mfusg_gsi",
        verbose=True, 
        model_ws=function_tmpdir,
        check=True)

    # assert disu, lpf, bas packages have been loaded
    msg = "flopy failed on loading modflow dis package"
    assert isinstance(m.dis, ModflowDis), msg
    msg = "flopy failed on loading modflow bas package"
    assert isinstance(m.bas6, ModflowBas), msg
    msg = "flopy failed on loading modflow chd package"
    assert isinstance(m.chd, ModflowChd), msg
    msg = "flopy failed on loading mfusg bcf package"
    assert isinstance(m.bcf6, MfUsgBcf), msg
    msg = "flopy failed on loading mfusg oc package"
    assert isinstance(m.oc, MfUsgOc), msg
    msg = "flopy failed on loading mfusg sms package"
    assert isinstance(m.sms, MfUsgSms), msg
    msg = "flopy failed on loading mfusg bct package"
    assert isinstance(m.bct, MfUsgBct), msg
    msg = "flopy failed on loading mfusg pcb package"
    assert isinstance(m.pcb, MfUsgPcb), msg
    msg = "flopy failed on loading mfusg dpt package"
    assert isinstance(m.dpt, MfUsgDpt), msg

    m.write_input()
    success, buff = m.run_model()
    msg = "flopy failed on running DualDomain.nam"
    assert success, msg

@requires_exe("mfusg_gsi")
def test_usg_load_Ex5_Henry(function_tmpdir, mfusg_transport_Ex5_model_path):
    print("testing mfusg transport model loading: Conduit.nam")

    fname = mfusg_transport_Ex5_model_path / "Henry.nam"
    assert os.path.isfile(fname), f"nam file not found {fname}"

    # Create the model
    m = MfUsg.load(
        fname,
        exe_name = "mfusg_gsi",
        verbose=True, 
        model_ws=function_tmpdir,
        check=True)

    # assert disu, lpf, bas packages have been loaded
    msg = "flopy failed on loading mfusg disu package"
    assert isinstance(m.disu, MfUsgDisU), msg
    msg = "flopy failed on loading modflow bas package"
    assert isinstance(m.bas6, ModflowBas), msg
    msg = "flopy failed on loading modflow chd package"
    assert isinstance(m.chd, ModflowChd), msg
    msg = "flopy failed on loading mfusg lpf package"
    assert isinstance(m.lpf, MfUsgLpf), msg
    msg = "flopy failed on loading mfusg oc package"
    assert isinstance(m.oc, MfUsgOc), msg
    msg = "flopy failed on loading mfusg sms package"
    assert isinstance(m.sms, MfUsgSms), msg
    msg = "flopy failed on loading mfusg bct package"
    assert isinstance(m.bct, MfUsgBct), msg
    msg = "flopy failed on loading mfusg pcb package"
    assert isinstance(m.pcb, MfUsgPcb), msg
    msg = "flopy failed on loading mfusg ddf package"
    assert isinstance(m.ddf, MfUsgDdf), msg
    msg = "flopy failed on loading mfusg wel package"
    assert isinstance(m.wel, MfUsgWel), msg

    m.write_input()
    success, buff = m.run_model()
    msg = "flopy failed on running Henry.nam"
    assert success, msg

@requires_exe("mfusg_gsi")
def test_usg_load_Ex6_Stallman_Heat(function_tmpdir, mfusg_transport_Ex6_model_path):
    print("testing mfusg transport model loading: Stallman_Heat.nam")

    fname = mfusg_transport_Ex6_model_path / "Heat/Stallman_Heat.nam"
    assert os.path.isfile(fname), f"nam file not found {fname}"

    # Create the model
    m = MfUsg.load(
        fname,
        exe_name = "mfusg_gsi",
        verbose=True, 
        model_ws=function_tmpdir,
        check=True)

    # assert disu, lpf, bas packages have been loaded
    msg = "flopy failed on loading mfusg disu package"
    assert isinstance(m.disu, MfUsgDisU), msg
    msg = "flopy failed on loading modflow bas package"
    assert isinstance(m.bas6, ModflowBas), msg
    msg = "flopy failed on loading modflow chd package"
    assert isinstance(m.chd, ModflowChd), msg
    msg = "flopy failed on loading mfusg lpf package"
    assert isinstance(m.lpf, MfUsgLpf), msg
    msg = "flopy failed on loading mfusg oc package"
    assert isinstance(m.oc, MfUsgOc), msg
    msg = "flopy failed on loading mfusg sms package"
    assert isinstance(m.sms, MfUsgSms), msg
    msg = "flopy failed on loading mfusg bct package"
    assert isinstance(m.bct, MfUsgBct), msg
    msg = "flopy failed on loading mfusg pcb package"
    assert isinstance(m.pcb, MfUsgPcb), msg

    m.write_input()
    success, buff = m.run_model()
    msg = "flopy failed on running Stallman_Heat.nam"
    assert success, msg

@requires_exe("mfusg_gsi")
def test_usg_load_Ex6_Stallman_Solute(function_tmpdir, mfusg_transport_Ex6_model_path):
    print("testing mfusg transport model loading: Stallman_Solute.nam")

    fname = mfusg_transport_Ex6_model_path / "Solute/Stallman_Solute.nam"
    assert os.path.isfile(fname), f"nam file not found {fname}"

    # Create the model
    m = MfUsg.load(
        fname,
        exe_name = "mfusg_gsi",
        verbose=True, 
        model_ws=function_tmpdir,
        check=True)

    # assert disu, lpf, bas packages have been loaded
    msg = "flopy failed on loading mfusg disu package"
    assert isinstance(m.disu, MfUsgDisU), msg
    msg = "flopy failed on loading modflow bas package"
    assert isinstance(m.bas6, ModflowBas), msg
    msg = "flopy failed on loading modflow chd package"
    assert isinstance(m.chd, ModflowChd), msg
    msg = "flopy failed on loading mfusg lpf package"
    assert isinstance(m.lpf, MfUsgLpf), msg
    msg = "flopy failed on loading mfusg oc package"
    assert isinstance(m.oc, MfUsgOc), msg
    msg = "flopy failed on loading mfusg sms package"
    assert isinstance(m.sms, MfUsgSms), msg
    msg = "flopy failed on loading mfusg bct package"
    assert isinstance(m.bct, MfUsgBct), msg
    msg = "flopy failed on loading mfusg pcb package"
    assert isinstance(m.pcb, MfUsgPcb), msg

    m.write_input()
    success, buff = m.run_model()
    msg = "flopy failed on running Stallman_Solute.nam"
    assert success, msg

@requires_exe("mfusg_gsi")
def test_usg_load_Ex6_Stallman_Solute_Heat(
    function_tmpdir, mfusg_transport_Ex6_model_path):

    print("testing mfusg transport model loading: Stallman.nam")

    fname = mfusg_transport_Ex6_model_path / "Solute_Heat/Stallman.nam"
    assert os.path.isfile(fname), f"nam file not found {fname}"

    # Create the model
    m = MfUsg.load(
        fname,
        exe_name = "mfusg_gsi",
        verbose=True, 
        model_ws=function_tmpdir,
        check=True)

    # assert disu, lpf, bas packages have been loaded
    msg = "flopy failed on loading mfusg disu package"
    assert isinstance(m.disu, MfUsgDisU), msg
    msg = "flopy failed on loading modflow bas package"
    assert isinstance(m.bas6, ModflowBas), msg
    msg = "flopy failed on loading modflow chd package"
    assert isinstance(m.chd, ModflowChd), msg
    msg = "flopy failed on loading mfusg lpf package"
    assert isinstance(m.lpf, MfUsgLpf), msg
    msg = "flopy failed on loading mfusg oc package"
    assert isinstance(m.oc, MfUsgOc), msg
    msg = "flopy failed on loading mfusg sms package"
    assert isinstance(m.sms, MfUsgSms), msg
    msg = "flopy failed on loading mfusg bct package"
    assert isinstance(m.bct, MfUsgBct), msg
    msg = "flopy failed on loading mfusg pcb package"
    assert isinstance(m.pcb, MfUsgPcb), msg

    m.write_input()
    success, buff = m.run_model()
    msg = "flopy failed on running Stallman.nam"
    assert success, msg

@requires_exe("mfusg_gsi")
def test_usg_load_Ex7_Matrix_Diffusion_DiscreteFracture(
    function_tmpdir, mfusg_transport_Ex7_model_path):

    print("testing mfusg transport model loading: USG_discrete_fracture.nam")

    fname = mfusg_transport_Ex7_model_path/"DiscreteFracture/USG_discrete_fracture.nam"
    assert os.path.isfile(fname), f"nam file not found {fname}"

    # Create the model
    m = MfUsg.load(
        fname,
        exe_name = "mfusg_gsi",
        verbose=True, 
        model_ws=function_tmpdir,
        check=True)

    # assert disu, lpf, bas packages have been loaded
    msg = "flopy failed on loading mfusg disu package"
    assert isinstance(m.disu, MfUsgDisU), msg
    msg = "flopy failed on loading modflow bas package"
    assert isinstance(m.bas6, ModflowBas), msg
    msg = "flopy failed on loading modflow chd package"
    assert isinstance(m.chd, ModflowChd), msg
    msg = "flopy failed on loading mfusg lpf package"
    assert isinstance(m.lpf, MfUsgLpf), msg
    msg = "flopy failed on loading mfusg oc package"
    assert isinstance(m.oc, MfUsgOc), msg
    msg = "flopy failed on loading mfusg sms package"
    assert isinstance(m.sms, MfUsgSms), msg
    msg = "flopy failed on loading mfusg bct package"
    assert isinstance(m.bct, MfUsgBct), msg
    msg = "flopy failed on loading mfusg pcb package"
    assert isinstance(m.pcb, MfUsgPcb), msg
    msg = "flopy failed on loading mfusg mdt package"
    assert isinstance(m.mdt, MfUsgMdt), msg

    m.write_input()
    success, buff = m.run_model()
    msg = "flopy failed on running USG_discrete_fracture.nam"
    assert success, msg

@requires_exe("mfusg_gsi")
def test_usg_load_Ex7_Matrix_Diffusion(function_tmpdir, mfusg_transport_Ex7_model_path):
    print("testing mfusg transport model loading: USG_Multispecies.nam")

    fname = mfusg_transport_Ex7_model_path / "Multispecies/USG_Multispecies.nam"
    assert os.path.isfile(fname), f"nam file not found {fname}"

    # Create the model
    m = MfUsg.load(
        fname,
        exe_name = "mfusg_gsi",
        verbose=True, 
        model_ws=function_tmpdir,
        check=True)

    # assert disu, lpf, bas packages have been loaded
    msg = "flopy failed on loading mfusg disu package"
    assert isinstance(m.disu, MfUsgDisU), msg
    msg = "flopy failed on loading modflow bas package"
    assert isinstance(m.bas6, ModflowBas), msg
    msg = "flopy failed on loading modflow chd package"
    assert isinstance(m.chd, ModflowChd), msg
    msg = "flopy failed on loading mfusg lpf package"
    assert isinstance(m.lpf, MfUsgLpf), msg
    msg = "flopy failed on loading mfusg oc package"
    assert isinstance(m.oc, MfUsgOc), msg
    msg = "flopy failed on loading mfusg sms package"
    assert isinstance(m.sms, MfUsgSms), msg
    msg = "flopy failed on loading mfusg bct package"
    assert isinstance(m.bct, MfUsgBct), msg
    msg = "flopy failed on loading mfusg pcb package"
    assert isinstance(m.pcb, MfUsgPcb), msg
    msg = "flopy failed on loading mfusg mdt package"
    assert isinstance(m.mdt, MfUsgMdt), msg

    m.write_input()
    success, buff = m.run_model()
    msg = "flopy failed on running USG_Multispecies.nam"
    assert success, msg

@requires_exe("mfusg_gsi")
def test_usg_load_Ex7_SandTank(function_tmpdir, mfusg_transport_Ex7_model_path):
    print("testing mfusg transport model loading: usg_sand_tank.nam")

    fname = mfusg_transport_Ex7_model_path / "SandTank/usg_sand_tank.nam"
    assert os.path.isfile(fname), f"nam file not found {fname}"

    # Create the model
    m = MfUsg.load(
        fname,
        exe_name = "mfusg_gsi",
        verbose=True, 
        model_ws=function_tmpdir,
        check=True)

    # assert disu, lpf, bas packages have been loaded
    msg = "flopy failed on loading mfusg disu package"
    assert isinstance(m.disu, MfUsgDisU), msg
    msg = "flopy failed on loading modflow bas package"
    assert isinstance(m.bas6, ModflowBas), msg
    msg = "flopy failed on loading modflow chd package"
    assert isinstance(m.chd, ModflowChd), msg
    msg = "flopy failed on loading mfusg lpf package"
    assert isinstance(m.lpf, MfUsgLpf), msg
    msg = "flopy failed on loading mfusg oc package"
    assert isinstance(m.oc, MfUsgOc), msg
    msg = "flopy failed on loading mfusg sms package"
    assert isinstance(m.sms, MfUsgSms), msg
    msg = "flopy failed on loading mfusg bct package"
    assert isinstance(m.bct, MfUsgBct), msg
    msg = "flopy failed on loading mfusg pcb package"
    assert isinstance(m.pcb, MfUsgPcb), msg
    msg = "flopy failed on loading mfusg mdt package"
    assert isinstance(m.mdt, MfUsgMdt), msg
    msg = "flopy failed on loading mfusg wel package"
    assert isinstance(m.wel, MfUsgWel), msg

    m.write_input()
    success, buff = m.run_model()
    msg = "flopy failed on running usg_sand_tank.nam"
    assert success, msg

@requires_exe("mfusg_gsi")
def test_usg_load_Ex8_Lake(function_tmpdir, mfusg_transport_Ex8_model_path):
    print("testing mfusg transport model loading: lak_usg_01.nam")

    fname = mfusg_transport_Ex8_model_path / "lak_usg_01.nam"
    assert os.path.isfile(fname), f"nam file not found {fname}"

    # Create the model
    m = MfUsg.load(
        fname,
        exe_name = "mfusg_gsi",
        verbose=True, 
        model_ws=function_tmpdir,
        check=True)

    # assert disu, lpf, bas packages have been loaded
    msg = "flopy failed on loading mfusg disu package"
    assert isinstance(m.disu, MfUsgDisU), msg
    msg = "flopy failed on loading modflow bas package"
    assert isinstance(m.bas6, ModflowBas), msg
    msg = "flopy failed on loading modflow fhb package"
    assert isinstance(m.fhb, ModflowFhb), msg
    msg = "flopy failed on loading mfusg bcf package"
    assert isinstance(m.bcf6, MfUsgBcf), msg
    msg = "flopy failed on loading mfusg oc package"
    assert isinstance(m.oc, MfUsgOc), msg
    msg = "flopy failed on loading mfusg sms package"
    assert isinstance(m.sms, MfUsgSms), msg
    msg = "flopy failed on loading mfusg bct package"
    assert isinstance(m.bct, MfUsgBct), msg
    msg = "flopy failed on loading mfusg pcb package"
    assert isinstance(m.pcb, MfUsgPcb), msg
    msg = "flopy failed on loading mfusg rch package"
    assert isinstance(m.rch, MfUsgRch), msg
    msg = "flopy failed on loading mfusg evt package"
    assert isinstance(m.evt, MfUsgEvt), msg
    msg = "flopy failed on loading mfusg lak package"
    assert isinstance(m.lak, MfUsgLak), msg

    m.write_input()
    success, buff = m.run_model()
    # Issue with the executable mfusg 2.3.0 deallocate GWF2EVT8U1DA line 661
    # msg = "flopy failed on running lak_usg_01.nam"
    # assert success, msg

@requires_exe("mfusg_gsi")
def test_usg_load_Ex9_PFAS(function_tmpdir, mfusg_transport_Ex9_model_path):
    print("testing mfusg transport model loading: PFAS_C1.nam")

    fname = mfusg_transport_Ex9_model_path / "C1/PFAS_C1.nam"
    assert os.path.isfile(fname), f"nam file not found {fname}"

    # Create the model
    m = MfUsg.load(
        fname,
        exe_name = "mfusg_gsi",
        verbose=True, 
        model_ws=function_tmpdir,
        check=True)

    # assert disu, lpf, bas packages have been loaded
    msg = "flopy failed on loading mfusg disu package"
    assert isinstance(m.disu, MfUsgDisU), msg
    msg = "flopy failed on loading modflow bas package"
    assert isinstance(m.bas6, ModflowBas), msg
    msg = "flopy failed on loading mfusg lpf package"
    assert isinstance(m.lpf, MfUsgLpf), msg
    msg = "flopy failed on loading mfusg oc package"
    assert isinstance(m.oc, MfUsgOc), msg
    msg = "flopy failed on loading mfusg sms package"
    assert isinstance(m.sms, MfUsgSms), msg
    msg = "flopy failed on loading mfusg bct package"
    assert isinstance(m.bct, MfUsgBct), msg
    msg = "flopy failed on loading mfusg rch package"
    assert isinstance(m.rch, MfUsgRch), msg

    m.write_input()
    success, buff = m.run_model()
    msg = "flopy failed on running PFAS_C1.nam"
    assert success, msg
