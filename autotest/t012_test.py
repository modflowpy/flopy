# Test loading of MODFLOW and MT3D models that come with MT3D distribution
import os
import flopy


pthtest = os.path.join('..', 'examples', 'data', 'mt3d_test')
pth2005 = os.path.join(pthtest, 'mf2005mt3d')
pth2000 = os.path.join(pthtest, 'mf2kmt3d')
pthNWT = os.path.join(pthtest, 'mfnwt_mt3dusgs')
newpth = os.path.join('.', 'temp', 't012')

mf2k_exe = 'mf2000'
mf2005_exe = 'mf2005'
mfnwt_exe = 'mfnwt'
mt3d_exe = 'mt3dms'
mt3d_usgs_exe = 'mt3d-usgs'

ismf2k = flopy.which(mf2k_exe)
ismf2005 = flopy.which(mf2005_exe)
ismfnwt = flopy.which(mfnwt_exe)
ismt3d = flopy.which(mt3d_exe)
ismt3dusgs = flopy.which(mt3d_usgs_exe)

def test_mf2005_p07():
    pth = os.path.join(pth2005, 'P07')
    namfile = 'p7mf2005.nam'
    mf = flopy.modflow.Modflow.load(namfile, model_ws=pth, verbose=True,
                                    exe_name=mf2005_exe)
    mf.model_ws = newpth
    mf.write_input()
    if ismf2005 is not None:
        success, buff = mf.run_model(silent=False)
        assert success, '{} did not run'.format(mf.name)

    namfile = 'p7mt.nam'
    mt = flopy.mt3d.mt.Mt3dms.load(namfile, model_ws=pth, verbose=True,
                                   exe_name=mt3d_exe)
    mt.model_ws = newpth
    ftlfile = 'p7.ftl'
    mt.ftlfilename = ftlfile
    mt.write_input()
    if ismt3d is not None and ismf2005 is not None:
        success, buff = mt.run_model(silent=False,
                                     normal_msg='program completed.')
        assert success, '{} did not run'.format(mt.name)
        os.remove(os.path.join(newpth, ftlfile))
    return

def test_mf2000_p07():
    pth = os.path.join(pth2000, 'P07')
    namfile = 'p7mf2k.nam'
    mf = flopy.modflow.Modflow.load(namfile, model_ws=pth,
                                    version='mf2k', verbose=True,
                                    exe_name=mf2k_exe)
    mf.model_ws = newpth
    mf.write_input()
    if ismf2k is not None:
        success, buff = mf.run_model(silent=True)
        assert success, '{} did not run'.format(mf.name)

    namfile = 'p7mt.nam'
    mt = flopy.mt3d.mt.Mt3dms.load(namfile, model_ws=pth, verbose=True,
                                   exe_name=mt3d_exe)
    mt.model_ws = newpth
    ftlfile = 'p7.ftl'
    mt.ftlfilename = ftlfile
    mt.write_input()
    if ismt3d is not None and ismf2k is not None:
        success, buff = mt.run_model(silent=False,
                                     normal_msg='program completed.')
        assert success, '{} did not run'.format(mt.name)
        os.remove(os.path.join(newpth, ftlfile))
    return

def test_mf2000_HSSTest():
    pth = os.path.join(pth2000, 'HSSTest')
    namfile = 'hsstest_mf2k.nam'
    mf = flopy.modflow.Modflow.load(namfile, model_ws=pth,
                                    version='mf2k', verbose=True,
                                    exe_name=mf2k_exe)
    mf.model_ws = newpth
    mf.write_input()
    if ismf2k is not None:
        success, buff = mf.run_model(silent=True)
        assert success, '{} did not run'.format(mf.name)
    namfile = 'hsstest_mt.nam'
    mt = flopy.mt3d.mt.Mt3dms.load(namfile, model_ws=pth, verbose=True,
                                   exe_name=mt3d_exe)
    mt.model_ws = newpth
    ftlfile = 'hsstest.FTL'
    mt.ftlfilename = ftlfile
    mt.write_input()
    if ismt3d is not None and ismf2k is not None:
        success, buff = mt.run_model(silent=False,
                                     normal_msg='program completed.')
        assert success, '{} did not run'.format(mt.name)
        os.remove(os.path.join(newpth, ftlfile))
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
    mf = flopy.modflow.Modflow.load(namfile, model_ws=pth,
                                    version='mf2k', verbose=True,
                                    exe_name=mf2k_exe)
    mf.model_ws = newpth
    mf.write_input()
    if ismf2k is not None:
        success, buff = mf.run_model(silent=True)
        assert success, '{} did not run'.format(mf.name)
    namfile = 'P7MT.NAM'
    mt = flopy.mt3d.mt.Mt3dms.load(namfile, model_ws=pth, verbose=True,
                                   exe_name=mt3d_exe)
    mt.model_ws = newpth
    ftlfile = 'p7.ftl'
    mt.ftlfilename = ftlfile
    mt.write_input()
    if ismt3d is not None and ismf2k is not None:
        success, buff = mt.run_model(silent=False,
                                     normal_msg='program completed.')
        assert success, '{} did not run'.format(mt.name)
        os.remove(os.path.join(newpth, ftlfile))
    return

def test_mf2000_P07():
    pth = os.path.join(pth2000, 'P07')
    namfile = 'p7mf2k.nam'
    mf = flopy.modflow.Modflow.load(namfile, model_ws=pth,
                                    version='mf2k', verbose=True,
                                    exe_name=mf2k_exe)
    mf.model_ws = newpth
    mf.write_input()
    if ismf2k is not None:
        success, buff = mf.run_model(silent=True)
        assert success, '{} did not run'.format(mf.name)

    namfile = 'p7mt.nam'
    mt = flopy.mt3d.mt.Mt3dms.load(namfile, model_ws=pth, verbose=True,
                                   exe_name=mt3d_exe)
    mt.model_ws = newpth
    ftlfile = 'p7.ftl'
    mt.ftlfilename = ftlfile
    mt.write_input()
    if ismt3d is not None and ismf2k is not None:
        success, buff = mt.run_model(silent=False,
                                     normal_msg='program completed.')
        assert success, '{} did not run'.format(mt.name)
        os.remove(os.path.join(newpth, ftlfile))
    return

def test_mf2000_reinject():
    pth = os.path.join(pth2000, 'reinject')
    namfile = 'P3MF2K.NAM'
    mf = flopy.modflow.Modflow.load(namfile, model_ws=pth,
                                    version='mf2k', verbose=True,
                                    exe_name=mf2k_exe)
    mf.model_ws = newpth
    mf.write_input()
    if ismf2k is not None:
        success, buff = mf.run_model(silent=True)
        assert success, '{} did not run'.format(mf.name)

    namfile = 'P3MT.NAM'
    mt = flopy.mt3d.mt.Mt3dms.load(namfile, model_ws=pth, verbose=True,
                                   exe_name=mt3d_exe)
    mt.model_ws = newpth
    ftlfile = 'p3.FTL'
    mt.ftlfilename = ftlfile
    mt.write_input()
    if ismt3d is not None and ismf2k is not None:
        success, buff = mt.run_model(silent=False,
                                     normal_msg='program completed.')
        assert success, '{} did not run'.format(mt.name)
        os.remove(os.path.join(newpth, ftlfile))
    return

def test_mf2000_SState():
    pth = os.path.join(pth2000, 'SState')
    namfile = 'SState_mf2k.nam'
    mf = flopy.modflow.Modflow.load(namfile, model_ws=pth,
                                    version='mf2k', verbose=True,
                                    exe_name=mf2k_exe)
    mf.model_ws = newpth
    mf.write_input()
    if ismf2k is not None:
        success, buff = mf.run_model(silent=True)
        assert success, '{} did not run'.format(mf.name)

    namfile = 'SState_mt.nam'
    mt = flopy.mt3d.mt.Mt3dms.load(namfile, model_ws=pth, verbose=True,
                                   exe_name=mt3d_exe)
    mt.model_ws = newpth
    ftlfile = 'SState.ftl'
    mt.ftlfilename = ftlfile
    mt.write_input()
    if ismt3d is not None and ismf2k is not None:
        success, buff = mt.run_model(silent=False,
                                     normal_msg='program completed.')
        assert success, '{} did not run'.format(mt.name)
        os.remove(os.path.join(newpth, ftlfile))
    return

def test_mf2000_tob():
    pth = os.path.join(pth2000, 'tob')
    namfile = 'p7mf2k.nam'
    mf = flopy.modflow.Modflow.load(namfile, model_ws=pth,
                                    version='mf2k', verbose=True,
                                    exe_name=mf2k_exe)
    mf.model_ws = newpth
    mf.lmt6.output_file_header = 'extended'
    mf.lmt6.output_file_format = 'formatted'
    mf.write_input()
    if ismf2k is not None:
        success, buff = mf.run_model(silent=True)
        assert success, '{} did not run'.format(mf.name)

    namfile = 'p7mt.nam'
    mt = flopy.mt3d.mt.Mt3dms.load(namfile, model_ws=pth, verbose=True,
                                   exe_name=mt3d_exe)
    mt.model_ws = newpth
    ftlfile = 'p7.ftl'
    mt.ftlfilename = ftlfile
    mt.write_input()
    if ismt3d is not None and ismf2k is not None:
        success, buff = mt.run_model(silent=False,
                                     normal_msg='program completed.')
        assert success, '{} did not run'.format(mt.name)
        os.remove(os.path.join(newpth, ftlfile))
    return

def test_mf2000_zeroth():
    pth = os.path.join(pth2000, 'zeroth')
    namfile = 'z0mf2k.nam'
    mf = flopy.modflow.Modflow.load(namfile, model_ws=pth,
                                    version='mf2k', verbose=True,
                                    exe_name=mf2k_exe)
    mf.model_ws = newpth
    mf.write_input()
    if ismf2k is not None:
        success, buff = mf.run_model(silent=True)
        assert success, '{} did not run'.format(mf.name)

    namfile = 'z0mt.nam'
    mt = flopy.mt3d.mt.Mt3dms.load(namfile, model_ws=pth, verbose=True,
                                   exe_name=mt3d_exe)
    mt.model_ws = newpth
    ftlfile = 'zeroth.FTL'
    mt.ftlfilename = ftlfile
    mt.write_input()
    if ismt3d is not None and ismf2k is not None:
        success, buff = mt.run_model(silent=False,
                                     normal_msg='program completed.')
        assert success, '{} did not run'.format(mt.name)
        os.remove(os.path.join(newpth, ftlfile))
    return

def test_mfnwt_CrnkNic():
    pth = os.path.join(pthNWT, 'sft_crnkNic')
    namefile = 'CrnkNic.nam'
    mf = flopy.modflow.Modflow.load(namefile, model_ws=pth,
                                    version='mfnwt', verbose=True,
                                    exe_name=mfnwt_exe)
    mf.model_ws = newpth
    mf.write_input()
    if ismfnwt is not None:
        success, buff = mf.run_model(silent=True)
        assert success, '{} did not run'.format(mf.name)

    namefile = 'CrnkNic.mtnam'
    mt = flopy.mt3d.mt.Mt3dms.load(namefile, model_ws=pth, verbose=True,
                                   version='mt3d-usgs', exe_name=mt3d_usgs_exe)
    mt.model_ws = newpth
    ftlfile = 'CrnkNic.ftl'
    mt.ftlfilename = ftlfile
    mt.write_input()
    if ismt3dusgs is not None and ismfnwt is not None:
        success, buff = mt.run_model(silent=False,
                                     normal_msg='program completed.')
        assert success, '{} did not run'.format(mt.name)
        os.remove(os.path.join(newpth, ftlfile))
    return

def test_mfnwt_LKT():
    pth = os.path.join(pthNWT, 'LKT')
    namefile = 'lkt_mf.nam'
    mf = flopy.modflow.Modflow.load(namefile, model_ws=pth,
                                    version='mfnwt', verbose=True,
                                    exe_name=mfnwt_exe)
    mf.model_ws = newpth
    mf.write_input()
    if ismfnwt is not None:
        success, buff = mf.run_model(silent=True)
        assert success, '{} did not run'.format(mf.name)

    namefile = 'lkt_mt.nam'
    mt = flopy.mt3d.mt.Mt3dms.load(namefile, model_ws=pth, verbose=True,
                                   version='mt3d-usgs', exe_name=mt3d_usgs_exe)
    mt.model_ws = newpth
    ftlfile = 'lkt.ftl'
    mt.ftlfilename = ftlfile
    mt.write_input()
    if ismt3dusgs is not None and ismfnwt is not None:
        success, buff = mt.run_model(silent=False,
                                     normal_msg='program completed.')
        assert success, '{} did not run'.format(mt.name)
        os.remove(os.path.join(newpth, ftlfile))
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
#    test_mfnwt_CrnkNic()
#    test_mfnwt_LKT()
