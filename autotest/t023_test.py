# Test multi-species options in mt3d
import os
import numpy as np
import flopy

testpth = os.path.join('.', 'temp')

def test_mt3d_multispecies():
    # modflow model
    modelname = 'multispecies'
    nlay = 1
    nrow = 20
    ncol = 20
    nper = 10
    mf = flopy.modflow.Modflow(modelname=modelname, model_ws=testpth)
    dis = flopy.modflow.ModflowDis(mf, nlay=nlay, nrow=nrow, ncol=ncol,
                                   nper=nper)
    lpf = flopy.modflow.ModflowLpf(mf)
    rch = flopy.modflow.ModflowRch(mf)
    evt = flopy.modflow.ModflowEvt(mf)
    mf.write_input()

    # Create a 5-component mt3d model and write the files
    ncomp = 5
    mt = flopy.mt3d.Mt3dms(modelname=modelname, modflowmodel=mf,
                           model_ws=testpth, verbose=True)
    sconc3 = np.random.random((nrow, ncol))
    btn = flopy.mt3d.Mt3dBtn(mt, ncomp=ncomp, sconc=1., sconc2=2.,
                             sconc3=sconc3, sconc5=5.)
    crch32 = np.random.random((nrow, ncol))
    cevt33 = np.random.random((nrow, ncol))
    ssm = flopy.mt3d.Mt3dSsm(mt, crch=1., crch2=2., crch3={2:crch32}, crch5=5.,
                             cevt=1., cevt2=2., cevt3={3:cevt33}, cevt5=5.)
    crch2 = ssm.crch[1].array
    assert(crch2.max() == 2.)
    cevt2 = ssm.cevt[1].array
    assert(cevt2.max() == 2.)
    mt.write_input()

    # Create a second MODFLOW model
    modelname2 = 'multispecies2'
    mf2 = flopy.modflow.Modflow(modelname=modelname2, model_ws=testpth)
    dis2 = flopy.modflow.ModflowDis(mf2, nlay=nlay, nrow=nrow, ncol=ncol,
                                    nper=nper)

    # Load the MT3D model into mt2 and then write it out
    fname = modelname + '.nam'
    mt2 = flopy.mt3d.Mt3dms.load(fname, model_ws=testpth, verbose=True)
    mt2.name = modelname2
    mt2.write_input()

    return


if __name__ == '__main__':
    test_mt3d_multispecies()
