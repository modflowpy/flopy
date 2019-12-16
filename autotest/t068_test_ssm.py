"""
Test MT3D model creation and file writing
"""

import os
import flopy
import numpy as np

mf_exe_name = 'mf2005'
mt_exe_name = 'mt3dms'
v1 = flopy.which(mf_exe_name)
v2 = flopy.which(mt_exe_name)
run = True
if v1 is None or v2 is None:
    run = False


def test_mt3d_ssm_with_nodata_in_1st_sp():

    nlay, nrow, ncol = 3, 5, 5
    perlen = np.zeros((10), dtype=np.float) + 10
    nper = len(perlen)
    
    ibound = np.ones((nlay,nrow,ncol), dtype=np.int)
    
    botm = np.arange(-1,-4,-1)
    top = 0.
    
    # creating MODFLOW model
    
    model_ws = os.path.join('.', 'temp', 't068a')
    modelname = 'model_mf'
    
    mf = flopy.modflow.Modflow(modelname, model_ws=model_ws,
                               exe_name=mf_exe_name)
    dis = flopy.modflow.ModflowDis(mf, nlay=nlay, nrow=nrow, ncol=ncol, 
                                   perlen=perlen, nper=nper, botm=botm, top=top, 
                                   steady=False)
    
    bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=top)
    lpf = flopy.modflow.ModflowLpf(mf, hk=100, vka=100, ss=0.00001, sy=0.1)
    oc = flopy.modflow.ModflowOc(mf)
    pcg = flopy.modflow.ModflowPcg(mf)
    lmt = flopy.modflow.ModflowLmt(mf)
    
    # recharge
    rchrate = {}
    rchrate[0] = 0.0
    rchrate[5] = 0.001
    rchrate[6] = 0.0
    
    rch = flopy.modflow.ModflowRch(mf, rech=rchrate, nrchop=3)
    
    # define itype
    itype = flopy.mt3d.Mt3dSsm.itype_dict()
    ssm_data = {}
    
    # Now for the point of this test: Enter SSM data sometime
    # after the first stress period (this was crashing flopy
    # version 3.2.13
    ghb_data = {}
    ssm_data[2] = [(nlay - 1, 4, 4, 1.0, itype['GHB'], 1.0, 100.0)]
    ghb_data[2] = [(nlay - 1, 4, 4, 0.1, 1.5)]
    ghb_data[5] = [(nlay - 1, 4, 4, 0.25, 1.5)]
    ssm_data[5] = [(nlay - 1, 4, 4, 0.5, itype['GHB'], 0.5, 200.0)]
    
    for k in range(nlay):
        for i in range(nrow):
            ghb_data[2].append((k, i, 0, 0.0, 100.0))
            ssm_data[2].append((k, i, 0, 0.0, itype['GHB'], 0.0, 0.0))
    
    ghb_data[5] = [(nlay - 1, 4, 4, 0.25, 1.5)]
    ssm_data[5] = [(nlay - 1, 4, 4, 0.5, itype['GHB'], 0.5, 200.0)]
    for k in range(nlay):
        for i in range(nrow):
            ghb_data[5].append((k, i, 0, -0.5, 100.0))
            ssm_data[5].append((k, i, 0, 0.0, itype['GHB'], 0.0, 0.0))
    
    
    ghb = flopy.modflow.ModflowGhb(mf, stress_period_data=ghb_data)
    
    # create MT3D-USGS model
    modelname = 'model_mt'
    mt = flopy.mt3d.Mt3dms(modflowmodel=mf, modelname=modelname,
                           model_ws=model_ws, exe_name='mt3dms')
    btn = flopy.mt3d.Mt3dBtn(mt, sconc=0, ncomp=2, sconc2=50.0)
    adv = flopy.mt3d.Mt3dAdv(mt)
    ssm = flopy.mt3d.Mt3dSsm(mt, stress_period_data=ssm_data)
    gcg = flopy.mt3d.Mt3dGcg(mt)
    
    # Write the output
    mf.write_input()
    mt.write_input()

    # run the models
    if run:
        success, buff = mf.run_model(report=True)
        assert success, 'MODFLOW did not run'
        success, buff = mt.run_model(report=True, normal_msg='Program completed.')
        assert success, 'MT3D did not run'

        model_ws2 = os.path.join('.', 'temp', 't068b')
        mf2 = flopy.modflow.Modflow.load('model_mf.nam', model_ws=model_ws,
                                         exe_name='mf2005')
        mf2.change_model_ws(model_ws2)
        mt2 = flopy.mt3d.Mt3dms.load('model_mt.nam', model_ws=model_ws,
                                     verbose=True, exe_name='mt3dms')
        mt2.change_model_ws(model_ws2)
        mf2.write_input()
        mt2.write_input()
        success, buff = mf2.run_model(report=True)
        assert success, 'MODFLOW did not run'
        success, buff = mt2.run_model(report=True, normal_msg='Program completed.')
        assert success, 'MT3D did not run'

        fname = os.path.join(model_ws, 'MT3D001.UCN')
        ucnobj = flopy.utils.UcnFile(fname)
        conca = ucnobj.get_alldata()

        fname = os.path.join(model_ws2, 'MT3D001.UCN')
        ucnobj = flopy.utils.UcnFile(fname)
        concb = ucnobj.get_alldata()

        assert np.allclose(conca, concb)

    return


def test_none_spdtype():
    # ensure that -1 and None work as valid list entries in the
    # stress period dictionary
    model_ws = os.path.join('.', 'temp', 't068c')
    mf = flopy.modflow.Modflow(model_ws=model_ws, exe_name=mf_exe_name)
    dis = flopy.modflow.ModflowDis(mf, nper=2)
    bas = flopy.modflow.ModflowBas(mf)
    lpf = flopy.modflow.ModflowLpf(mf)
    spd = {0: -1, 1: None}
    wel = flopy.modflow.ModflowWel(mf, stress_period_data=spd)
    pcg = flopy.modflow.ModflowPcg(mf)
    mf.write_input()
    mf2 = flopy.modflow.Modflow.load('modflowtest.nam', model_ws=model_ws,
                                     verbose=True)
    if run:
        success, buff = mf.run_model(report=True)
        assert success


if __name__ == '__main__':
    test_mt3d_ssm_with_nodata_in_1st_sp()
    test_none_spdtype()


