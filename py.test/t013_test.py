"""
Test MT3D model creation and file writing
"""

import os
import flopy

def test_mt3d_create_withmfmodel():
    model_ws = os.path.join('.', 'temp')

    # Create a MODFLOW model
    mf = flopy.modflow.Modflow(model_ws=model_ws)
    dis = flopy.modflow.ModflowDis(mf)
    lpf = flopy.modflow.ModflowLpf(mf)

    # Create MT3D model
    mt = flopy.mt3d.Mt3dms(modflowmodel=mf, model_ws=model_ws)
    btn = flopy.mt3d.Mt3dBtn(mt)
    adv = flopy.mt3d.Mt3dAdv(mt)
    dsp = flopy.mt3d.Mt3dDsp(mt)
    ssm = flopy.mt3d.Mt3dSsm(mt)
    gcg = flopy.mt3d.Mt3dRct(mt)
    rct = flopy.mt3d.Mt3dGcg(mt)
    tob = flopy.mt3d.Mt3dTob(mt)

    # Write the output
    mt.write_input()
    return

def test_mt3d_create_woutmfmodel():
    model_ws = os.path.join('.', 'temp')

    # Create MT3D model
    mt = flopy.mt3d.Mt3dms(model_ws=model_ws)
    btn = flopy.mt3d.Mt3dBtn(mt, nlay=1, nrow=2, ncol=2, nper=1, delr=1.,
                             delc=1., htop=1., dz=1., laycon=0, perlen=1.,
                             nstp=1, tsmult=1.)
    adv = flopy.mt3d.Mt3dAdv(mt)
    dsp = flopy.mt3d.Mt3dDsp(mt)
    ssm = flopy.mt3d.Mt3dSsm(mt)
    gcg = flopy.mt3d.Mt3dRct(mt)
    rct = flopy.mt3d.Mt3dGcg(mt)
    tob = flopy.mt3d.Mt3dTob(mt)

    # Write the output
    mt.write_input()
    return


if __name__ == '__main__':
    test_mt3d_create_withmfmodel()
    test_mt3d_create_woutmfmodel()

