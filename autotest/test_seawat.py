from pathlib import Path

import numpy as np
import pytest
from autotest.conftest import get_example_data_path, requires_exe

from flopy.modflow import (
    Modflow,
    ModflowBas,
    ModflowDis,
    ModflowLpf,
    ModflowOc,
    ModflowPcg,
    ModflowWel,
)
from flopy.mt3d import Mt3dAdv, Mt3dBtn, Mt3dDsp, Mt3dGcg, Mt3dms, Mt3dSsm
from flopy.seawat import Seawat, SeawatVdf, SeawatVsc

# Setup problem parameters
Lx = 2.0
Lz = 1.0
nlay = 50
nrow = 1
ncol = 100
delr = Lx / ncol
delc = 1.0
delv = Lz / nlay
henry_top = 1.0
henry_botm = np.linspace(henry_top - delv, 0.0, nlay)
qinflow = 5.702  # m3/day
dmcoef = 0.57024  # m2/day  Could also try 1.62925
hk = 864.0  # m/day

ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)
ibound[:, :, -1] = -1
itype = Mt3dSsm.itype_dict()
wel_data = {}
ssm_data = {}
wel_sp1 = []
ssm_sp1 = []
for k in range(nlay):
    wel_sp1.append([k, 0, 0, qinflow / nlay])
    ssm_sp1.append([k, 0, 0, 0.0, itype["WEL"]])
    ssm_sp1.append([k, 0, ncol - 1, 35.0, itype["BAS6"]])
wel_data[0] = wel_sp1
ssm_data[0] = ssm_sp1


@pytest.mark.slow
@requires_exe("swtv4")
def test_seawat_henry(tmpdir):
    # SEAWAT model from a modflow model and an mt3d model
    modelname = "henry"
    mf = Modflow(modelname, exe_name="swtv4", model_ws=str(tmpdir))
    # shortened perlen to 0.1 to make this run faster -- should be about 0.5
    dis = ModflowDis(
        mf,
        nlay,
        nrow,
        ncol,
        nper=1,
        delr=delr,
        delc=delc,
        laycbd=0,
        top=henry_top,
        botm=henry_botm,
        perlen=0.1,
        nstp=15,
    )
    bas = ModflowBas(mf, ibound, 0)
    lpf = ModflowLpf(mf, hk=hk, vka=hk)
    pcg = ModflowPcg(mf, hclose=1.0e-8)
    wel = ModflowWel(mf, stress_period_data=wel_data)
    oc = ModflowOc(
        mf,
        stress_period_data={(0, 0): ["save head", "save budget"]},
        compact=True,
    )

    # Create the basic MT3DMS model structure
    mt = Mt3dms(modelname, "nam_mt3dms", mf, model_ws=str(tmpdir))
    btn = Mt3dBtn(
        mt,
        nprs=-5,
        prsity=0.35,
        sconc=35.0,
        ifmtcn=0,
        chkmas=False,
        nprobs=10,
        nprmas=10,
        dt0=0.001,
    )
    adv = Mt3dAdv(mt, mixelm=0)
    dsp = Mt3dDsp(mt, al=0.0, trpt=1.0, trpv=1.0, dmcoef=dmcoef)
    gcg = Mt3dGcg(mt, iter1=500, mxiter=1, isolve=1, cclose=1e-7)
    ssm = Mt3dSsm(mt, stress_period_data=ssm_data)

    # Create the SEAWAT model structure
    mswt = Seawat(
        modelname,
        "nam_swt",
        mf,
        mt,
        model_ws=str(tmpdir),
        exe_name="swtv4",
    )
    vdf = SeawatVdf(
        mswt,
        iwtable=0,
        densemin=0,
        densemax=0,
        denseref=1000.0,
        denseslp=0.7143,
        firstdt=1e-3,
    )

    # Write the input files
    mf.write_input()
    mt.write_input()
    mswt.write_input()

    success, buff = mswt.run_model(silent=False)
    assert success


@pytest.mark.slow
@requires_exe("swtv4")
def test_seawat2_henry(tmpdir):
    # SEAWAT model directly by adding packages
    modelname = "henry2"
    m = Seawat(
        modelname,
        "nam",
        model_ws=str(tmpdir),
        exe_name="swtv4",
    )
    dis = ModflowDis(
        m,
        nlay,
        nrow,
        ncol,
        nper=1,
        delr=delr,
        delc=delc,
        laycbd=0,
        top=henry_top,
        botm=henry_botm,
        perlen=0.1,
        nstp=15,
    )
    bas = ModflowBas(m, ibound, 0)
    lpf = ModflowLpf(m, hk=hk, vka=hk)
    pcg = ModflowPcg(m, hclose=1.0e-8)
    wel = ModflowWel(m, stress_period_data=wel_data)
    oc = ModflowOc(
        m,
        stress_period_data={(0, 0): ["save head", "save budget"]},
        compact=True,
    )

    # Create the basic MT3DMS model structure
    btn = Mt3dBtn(
        m,
        nprs=-5,
        prsity=0.35,
        sconc=35.0,
        ifmtcn=0,
        chkmas=False,
        nprobs=10,
        nprmas=10,
        dt0=0.001,
    )
    adv = Mt3dAdv(m, mixelm=0)
    dsp = Mt3dDsp(m, al=0.0, trpt=1.0, trpv=1.0, dmcoef=dmcoef)
    gcg = Mt3dGcg(m, iter1=500, mxiter=1, isolve=1, cclose=1e-7)
    ssm = Mt3dSsm(m, stress_period_data=ssm_data)

    # Create the SEAWAT model structure
    vdf = SeawatVdf(
        m,
        iwtable=0,
        densemin=0,
        densemax=0,
        denseref=1000.0,
        denseslp=0.7143,
        firstdt=1e-3,
    )

    # Write the input files
    m.write_input()

    success, buff = m.run_model(silent=False)
    assert success


def swt4_namfiles():
    return [
        str(p) for p in (get_example_data_path() / "swtv4_test").rglob("*.nam")
    ]


@requires_exe("swtv4")
@pytest.mark.parametrize("namfile", swt4_namfiles())
@pytest.mark.parametrize("binary", [True, False])
def test_seawat_load_and_write(tmpdir, namfile, binary):
    model_name = Path(namfile).name
    m = Seawat.load(
        model_name, model_ws=str(Path(namfile).parent), verbose=True
    )
    m.change_model_ws(str(tmpdir), reset_external=True)

    if binary:
        skip_bcf6 = False
        if m.bcf6 is not None:
            m.bcf6.hy[0].fmtin = "(BINARY)"
            skip_bcf6 = True

        skip_btn = False
        if m.btn is not None:
            m.btn.prsity[0].fmtin = "(BINARY)"
            skip_btn = True

        if skip_bcf6 and skip_btn:
            pytest.skip(
                f"no basic transport or block centered flow packages in {model_name}, "
                f"skipping binary array format test"
            )

    m.write_input()

    # TODO: run models in separate CI workflow?
    #   with regression testing & benchmarking?
    run = False

    if run:
        success, buff = m.run_model(silent=False)
        assert success


def test_vdf_vsc(tmpdir):
    nlay = 3
    nrow = 4
    ncol = 5
    nper = 3
    m = Seawat(modelname="vdftest", model_ws=str(tmpdir))
    dis = ModflowDis(m, nlay=nlay, nrow=nrow, ncol=ncol, nper=nper)
    vdf = SeawatVdf(m)

    # Test different variations of instantiating vsc
    vsc = SeawatVsc(m)
    m.write_input()
    m.remove_package("VSC")

    vsc = SeawatVsc(m, mt3dmuflg=0)
    m.write_input()
    m.remove_package("VSC")

    vsc = SeawatVsc(m, mt3dmuflg=0, mtmutempspec=0)
    m.write_input()
    m.remove_package("VSC")

    vsc = SeawatVsc(m, mt3dmuflg=-1)
    m.write_input()
    m.remove_package("VSC")

    vsc = SeawatVsc(m, mt3dmuflg=-1, nsmueos=1)
    m.write_input()
    m.remove_package("VSC")

    vsc = SeawatVsc(m, mt3dmuflg=1)
    m.write_input()
    m.remove_package("VSC")
