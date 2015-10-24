# Test loading of MODFLOW and MT3D models that come with MT3D distribution
import os
import flopy

pthtest = os.path.join('..', 'examples', 'data', 'mt3d_test')
pth2005 = os.path.join(pthtest, 'mf2005mt3d')
pth2000 = os.path.join(pthtest, 'mf2kmt3d')
newpth = os.path.join('.', 'temp')

def test_mf2005_p07():
    pth = os.path.join(pth2005, 'P07')
    namfile = 'p7mf2005.nam'
    mf = flopy.modflow.Modflow.load(namfile, model_ws=pth, verbose=True)
    namfile = 'p7mt.nam'
    mt = flopy.mt3d.mt.Mt3dms.load(namfile, model_ws=pth, verbose=True)
    mt.change_model_ws(newpth)
    mt.write_input()
    return

def test_mf2000_p07():
    pth = os.path.join(pth2000, 'P07')
    namfile = 'p7mf2k.nam'
    mf = flopy.modflow.Modflow.load(namfile, model_ws=pth, verbose=True)
    namfile = 'p7mt.nam'
    mt = flopy.mt3d.mt.Mt3dms.load(namfile, model_ws=pth, verbose=True)
    mt.change_model_ws(newpth)
    mt.write_input()
    return

def test_mf2000_HSSTest():
    pth = os.path.join(pth2000, 'HSSTest')
    namfile = 'hsstest_mf2k.nam'
    mf = flopy.modflow.Modflow.load(namfile, model_ws=pth, verbose=True)
    namfile = 'hsstest_mt.nam'
    mt = flopy.mt3d.mt.Mt3dms.load(namfile, model_ws=pth, verbose=True)
    mt.change_model_ws(newpth)
    mt.write_input()
    return

"""
This problem doesn't work.  File looks messed up.
def test_mf2000_mnw():
    pth = os.path.join(pth2000, 'mnw')
    namfile = 't5mf2k.nam'
    mf = flopy.modflow.Modflow.load(namfile, model_ws=pth, verbose=True)
    namfile = 't5mt.nam'
    mt = flopy.mt3d.mt.Mt3dms.load(namfile, model_ws=pth, verbose=True)
    mt.change_model_ws(newpth)
    mt.write_input()
    return
"""

def test_mf2000_MultiDiffusion():
    pth = os.path.join(pth2000, 'MultiDiffusion')
    namfile = 'p7mf2k.nam'
    mf = flopy.modflow.Modflow.load(namfile, model_ws=pth, verbose=True)
    namfile = 'p7MT.NAM'
    mt = flopy.mt3d.mt.Mt3dms.load(namfile, model_ws=pth, verbose=True)
    mt.change_model_ws(newpth)
    mt.write_input()
    return

def test_mf2000_P07():
    pth = os.path.join(pth2000, 'P07')
    namfile = 'p7mf2k.nam'
    mf = flopy.modflow.Modflow.load(namfile, model_ws=pth, verbose=True)
    namfile = 'p7mt.nam'
    mt = flopy.mt3d.mt.Mt3dms.load(namfile, model_ws=pth, verbose=True)
    mt.change_model_ws(newpth)
    mt.write_input()
    return

def test_mf2000_reinject():
    pth = os.path.join(pth2000, 'reinject')
    namfile = 'P3MF2K.NAM'
    mf = flopy.modflow.Modflow.load(namfile, model_ws=pth, verbose=True)
    namfile = 'P3MT.NAM'
    mt = flopy.mt3d.mt.Mt3dms.load(namfile, model_ws=pth, verbose=True)
    mt.change_model_ws(newpth)
    mt.write_input()
    return

def test_mf2000_SState():
    pth = os.path.join(pth2000, 'SState')
    namfile = 'SState_mf2k.nam'
    mf = flopy.modflow.Modflow.load(namfile, model_ws=pth, verbose=True)
    namfile = 'SState_mt.nam'
    mt = flopy.mt3d.mt.Mt3dms.load(namfile, model_ws=pth, verbose=True)
    mt.change_model_ws(newpth)
    mt.write_input()
    return

def test_mf2000_tob():
    pth = os.path.join(pth2000, 'tob')
    namfile = 'p7mf2k.nam'
    mf = flopy.modflow.Modflow.load(namfile, model_ws=pth, verbose=True)
    namfile = 'p7mt.nam'
    mt = flopy.mt3d.mt.Mt3dms.load(namfile, model_ws=pth, verbose=True)
    mt.change_model_ws(newpth)
    mt.write_input()
    return

def test_mf2000_zeroth():
    pth = os.path.join(pth2000, 'zeroth')
    namfile = 'z0mf2k.nam'
    mf = flopy.modflow.Modflow.load(namfile, model_ws=pth, verbose=True)
    namfile = 'z0mt.nam'
    mt = flopy.mt3d.mt.Mt3dms.load(namfile, model_ws=pth, verbose=True)
    mt.change_model_ws(newpth)
    mt.write_input()
    return



if __name__ == '__main__':
    test_mf2005_p07()
    test_mf2000_p07()
    test_mf2000_HSSTest()
    test_mf2000_MultiDiffusion()
    test_mf2000_P07()
    test_mf2000_reinject()
    test_mf2000_SState()
    test_mf2000_tob()
    test_mf2000_zeroth()
