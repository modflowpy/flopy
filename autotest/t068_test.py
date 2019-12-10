"""
Test MT3D model creation and file writing
"""

import os
import warnings
import flopy


def test_mt3d_ssm_with_nodata_in_1st_sp():
    model_ws = os.path.join('.', 'temp', 't068')
    
    nlay, nrow, ncol = 10, 10, 10
    perlen = np.zeros((10), dtype=np.float) + 10
    nper = len(perlen)
    
    ibound = np.ones((nlay,nrow,ncol), dtype=np.int)
    
    botm = np.arange(-1,-11,-1)
    top = 0.
    
    # creating MODFLOW model
    
    model_ws = 'data'
    modelname = 'ssm_ex2'
    
    mf = flopy.modflow.Modflow(modelname, model_ws=model_ws, version='mfnwt')
    dis = flopy.modflow.ModflowDis(mf, nlay=nlay, nrow=nrow, ncol=ncol, 
                                   perlen=perlen, nper=nper, botm=botm, top=top, 
                                   steady=False)
    
    bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=top)
    upw = flopy.modflow.ModflowUpw(mf, hk=100, vka=100, ss=0.00001, sy=0.1)
    oc = flopy.modflow.ModflowOc(mf)
    nwt = flopy.modflow.ModflowNwt(mf)
    
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
    ghb_data[2] = [(4, 4, 4, 0.1, 1.5)]
    ssm_data[2] = [(4, 4, 4, 1.0, itype['GHB'], 1.0, 100.0)]
    ghb_data[5] = [(4, 4, 4, 0.25, 1.5)]
    ssm_data[5] = [(4, 4, 4, 0.5, itype['GHB'], 0.5, 200.0)]
    
    for k in range(nlay):
        for i in range(nrow):
            ghb_data[2].append((k, i, 0, 0.0, 100.0))
            ssm_data[2].append((k, i, 0, 0.0, itype['GHB'], 0.0, 0.0))
    
    ghb_data[5] = [(4, 4, 4, 0.25, 1.5)]
    ssm_data[5] = [(4, 4, 4, 0.5, itype['GHB'], 0.5, 200.0)]
    for k in range(nlay):
        for i in range(nrow):
            ghb_data[5].append((k, i, 0, -0.5, 100.0))
            ssm_data[5].append((k, i, 0, 0.0, itype['GHB'], 0.0, 0.0))
    
    
    ghb = flopy.modflow.ModflowGhb(mf, stress_period_data=ghb_data)
    
    # create MT3D-USGS model
    mt = flopy.mt3d.Mt3dms(modflowmodel=mf, modelname=modelname, model_ws=model_ws)
    btn = flopy.mt3d.Mt3dBtn(mt, sconc=0, ncomp=2, sconc2=50.0)
    adv = flopy.mt3d.Mt3dAdv(mt)
    ssm = flopy.mt3d.Mt3dSsm(mt, stress_period_data=ssm_data)
    gcg = flopy.mt3d.Mt3dGcg(mt)
    
    # Write the output
    mf.write_input()
    mt.write_input()
    
    # confirm that MT3D files exist
    assert os.path.isfile(os.path.join(model_ws, '{}.{}'.format(mt.name, btn.extension[0]))) is True
    assert os.path.isfile(os.path.join(model_ws, '{}.{}'.format(mt.name, adv.extension[0]))) is True
    assert os.path.isfile(os.path.join(model_ws, '{}.{}'.format(mt.name, ssm.extension[0]))) is True
    assert os.path.isfile(os.path.join(model_ws, '{}.{}'.format(mt.name, gcg.extension[0]))) is True
    
    return


if __name__ == '__main__':
    test_mt3d_ssm_with_nodata_in_1st_sp()


