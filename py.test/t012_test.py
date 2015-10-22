# Test loading of MODFLOW and MT3D models that come with MT3D distribution
import os
import flopy

pthtest = os.path.join('..', 'examples', 'data', 'mt3d_test')
pth2005 = os.path.join(pthtest, 'mf2005mt3d')
pth2000 = os.path.join(pthtest, 'mf2kmt3d')

def test_mf2005_p07():
    pth = os.path.join(pth2005, 'P07')
    namfile = 'p7mf2005.nam'
    mf = flopy.modflow.Modflow.load(namfile, model_ws=pth, verbose=True)
    namfile = 'p7mt.nam'
    mt = flopy.mt3d.mt.Mt3dms2.load(namfile, modflowmodel=mf, model_ws=pth, verbose=True)
    return

def test_mf2000_p07():
    pth = os.path.join(pth2000, 'P07')
    namfile = 'p7mf2k.nam'
    mf = flopy.modflow.Modflow.load(namfile, model_ws=pth, verbose=True)
    namfile = 'p7mt.nam'
    mt = flopy.mt3d.mt.Mt3dms2.load(namfile, modflowmodel=mf, model_ws=pth, verbose=True)
    return

def test_mf2000_HSSTest():
    pth = os.path.join(pth2000, 'HSSTest')
    namfile = 'hsstest_mf2k.nam'
    mf = flopy.modflow.Modflow.load(namfile, model_ws=pth, verbose=True)
    namfile = 'hsstest_mt.nam'
    mt = flopy.mt3d.mt.Mt3dms2.load(namfile, modflowmodel=mf, model_ws=pth, verbose=True)
    return

if __name__ == '__main__':
    #test_mf2005_p07()
    #test_mf2000_p07()
    test_mf2000_HSSTest()
