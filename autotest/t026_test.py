"""
Some basic tests for SEAWAT Henry create and run.

"""

import os
import numpy as np
import flopy
from ci_framework import base_test_dir, FlopyTestSetup

base_dir = base_test_dir(__file__, rel_path="temp", verbose=True)

seawat_exe = "swtv4"
isseawat = flopy.which(seawat_exe)

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
itype = flopy.mt3d.Mt3dSsm.itype_dict()
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


def test_seawat_henry():
    # SEAWAT model from a modflow model and an mt3d model
    model_ws = f"{base_dir}_test_seawat_henry"
    test_setup = FlopyTestSetup(verbose=True, test_dirs=model_ws)

    modelname = "henry"
    mf = flopy.modflow.Modflow(modelname, exe_name="swtv4", model_ws=model_ws)
    # shortened perlen to 0.1 to make this run faster -- should be about 0.5
    dis = flopy.modflow.ModflowDis(
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
    bas = flopy.modflow.ModflowBas(mf, ibound, 0)
    lpf = flopy.modflow.ModflowLpf(mf, hk=hk, vka=hk)
    pcg = flopy.modflow.ModflowPcg(mf, hclose=1.0e-8)
    wel = flopy.modflow.ModflowWel(mf, stress_period_data=wel_data)
    oc = flopy.modflow.ModflowOc(
        mf,
        stress_period_data={(0, 0): ["save head", "save budget"]},
        compact=True,
    )

    # Create the basic MT3DMS model structure
    mt = flopy.mt3d.Mt3dms(modelname, "nam_mt3dms", mf, model_ws=model_ws)
    btn = flopy.mt3d.Mt3dBtn(
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
    adv = flopy.mt3d.Mt3dAdv(mt, mixelm=0)
    dsp = flopy.mt3d.Mt3dDsp(mt, al=0.0, trpt=1.0, trpv=1.0, dmcoef=dmcoef)
    gcg = flopy.mt3d.Mt3dGcg(mt, iter1=500, mxiter=1, isolve=1, cclose=1e-7)
    ssm = flopy.mt3d.Mt3dSsm(mt, stress_period_data=ssm_data)

    # Create the SEAWAT model structure
    mswt = flopy.seawat.Seawat(
        modelname,
        "nam_swt",
        mf,
        mt,
        model_ws=model_ws,
        exe_name="swtv4",
    )
    vdf = flopy.seawat.SeawatVdf(
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

    if isseawat is not None:
        success, buff = mswt.run_model(silent=False)
        assert success, f"{mswt.name} did not run"

    return


def test_seawat2_henry():
    # SEAWAT model directly by adding packages
    model_ws = f"{base_dir}_test_seawat2_henry"
    test_setup = FlopyTestSetup(verbose=True, test_dirs=model_ws)

    modelname = "henry2"
    m = flopy.seawat.swt.Seawat(
        modelname,
        "nam",
        model_ws=model_ws,
        exe_name="swtv4",
    )
    dis = flopy.modflow.ModflowDis(
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
    bas = flopy.modflow.ModflowBas(m, ibound, 0)
    lpf = flopy.modflow.ModflowLpf(m, hk=hk, vka=hk)
    pcg = flopy.modflow.ModflowPcg(m, hclose=1.0e-8)
    wel = flopy.modflow.ModflowWel(m, stress_period_data=wel_data)
    oc = flopy.modflow.ModflowOc(
        m,
        stress_period_data={(0, 0): ["save head", "save budget"]},
        compact=True,
    )

    # Create the basic MT3DMS model structure
    btn = flopy.mt3d.Mt3dBtn(
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
    adv = flopy.mt3d.Mt3dAdv(m, mixelm=0)
    dsp = flopy.mt3d.Mt3dDsp(m, al=0.0, trpt=1.0, trpv=1.0, dmcoef=dmcoef)
    gcg = flopy.mt3d.Mt3dGcg(m, iter1=500, mxiter=1, isolve=1, cclose=1e-7)
    ssm = flopy.mt3d.Mt3dSsm(m, stress_period_data=ssm_data)

    # Create the SEAWAT model structure
    vdf = flopy.seawat.SeawatVdf(
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

    if isseawat is not None:
        success, buff = m.run_model(silent=False)
        assert success, f"{m.name} did not run"

    return


if __name__ == "__main__":
    test_seawat_henry()
    test_seawat2_henry()
