"""
Some basic tests for SEAWAT Henry create and run.

"""

import os
import numpy as np
import flopy


workspace = os.path.join('temp')
seawat_exe = 'swt_v4'
isseawat = flopy.which(seawat_exe)


def test_seawat_henry():
    # Setup problem parameters
    Lx = 2.
    Lz = 1.
    nlay = 50
    nrow = 1
    ncol = 100
    delr = Lx / ncol
    delc = 1.0
    delv = Lz / nlay
    henry_top = 1.
    henry_botm = np.linspace(henry_top - delv, 0., nlay)
    qinflow = 5.702  #m3/day
    dmcoef = 0.57024 #m2/day  Could also try 1.62925
    hk = 864.  #m/day

    modelname = 'henry'
    ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)
    ibound[:, :, -1] = -1
    itype = flopy.mt3d.Mt3dSsm.itype_dict()
    wel_data = {}
    ssm_data = {}
    wel_sp1 = []
    ssm_sp1 = []
    for k in range(nlay):
        wel_sp1.append([k, 0, 0, qinflow / nlay])
        ssm_sp1.append([k, 0, 0, 0., itype['WEL']])
        ssm_sp1.append([k, 0, ncol - 1, 35., itype['BAS6']])
    wel_data[0] = wel_sp1
    ssm_data[0] = ssm_sp1

    mf = flopy.modflow.Modflow(modelname, exe_name='swt_v4',
                               model_ws=workspace)
    #shortened perlen to 0.1 to make this run faster -- should be about 0.5
    dis = flopy.modflow.ModflowDis(mf, nlay, nrow, ncol, nper=1, delr=delr,
                                   delc=delc, laycbd=0, top=henry_top,
                                   botm=henry_botm, perlen=0.1, nstp=15)
    bas = flopy.modflow.ModflowBas(mf, ibound, 0)
    lpf = flopy.modflow.ModflowLpf(mf, hk=hk, vka=hk)
    pcg = flopy.modflow.ModflowPcg(mf, hclose=1.e-8)
    wel = flopy.modflow.ModflowWel(mf, stress_period_data=wel_data)
    oc = flopy.modflow.ModflowOc(mf,
                                 stress_period_data={(0, 0): ['save head',
                                                              'save budget']},
                                 compact=True)

    # Create the basic MT3DMS model structure
    mt = flopy.mt3d.Mt3dms(modelname, 'nam_mt3dms', mf, model_ws=workspace)
    btn = flopy.mt3d.Mt3dBtn(mt, nprs=-5, prsity=0.35, sconc=35., ifmtcn=0,
                             chkmas=False, nprobs=10, nprmas=10, dt0=0.001)
    adv = flopy.mt3d.Mt3dAdv(mt, mixelm=0)
    dsp = flopy.mt3d.Mt3dDsp(mt, al=0., trpt=1., trpv=1., dmcoef=dmcoef)
    gcg = flopy.mt3d.Mt3dGcg(mt, iter1=500, mxiter=1, isolve=1, cclose=1e-7)
    ssm = flopy.mt3d.Mt3dSsm(mt, stress_period_data=ssm_data)

    # Create the SEAWAT model structure
    mswt = flopy.seawat.Seawat(modelname, 'nam_swt', mf, mt,
                               model_ws=workspace, exe_name='swt_v4')
    vdf = flopy.seawat.SeawatVdf(mswt, iwtable=0, densemin=0, densemax=0,
                                 denseref=1000., denseslp=0.7143, firstdt=1e-3)

    # Write the input files
    mf.write_input()
    mt.write_input()
    mswt.write_input()

    if isseawat is not None:
        success, buff = mswt.run_model(silent=True)
        assert success, '{} did not run'.format(mswt.name)

    return


if __name__ == '__main__':
    test_seawat_henry()
