# Test loading of MODFLOW and MT3D models that come with MT3D distribution
import os
import sys
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
mt3d_usgs_exe = 'mt3dusgs'

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
    cpth = os.path.join(newpth, 'P07')
    mf.model_ws = cpth

    mf.write_input()

    if ismf2005 is not None:
        success, buff = mf.run_model(silent=False)
        assert success, '{} did not run'.format(mf.name)

    namfile = 'p7mt.nam'
    mt = flopy.mt3d.mt.Mt3dms.load(namfile, model_ws=pth, verbose=True,
                                   exe_name=mt3d_exe)
    # Optional keyword line is absent in this example, ensure defaults are kept
    assert mt.btn.DRYCell is False
    assert mt.btn.Legacy99Stor is False
    assert mt.btn.MFStyleArr is False
    assert mt.btn.AltWTSorb is False

    mt.model_ws = cpth
    ftlfile = 'p7.ftl'
    mt.ftlfilename = ftlfile
    mt.write_input()
    if ismt3d is not None and ismf2005 is not None:
        success, buff = mt.run_model(silent=False,
                                     normal_msg='program completed.')
        assert success, '{} did not run'.format(mt.name)
        os.remove(os.path.join(cpth, ftlfile))
    return


def test_mf2000_p07():
    pth = os.path.join(pth2000, 'P07')
    namfile = 'p7mf2k.nam'
    mf = flopy.modflow.Modflow.load(namfile, model_ws=pth,
                                    verbose=True,
                                    exe_name=mf2k_exe)

    cpth = os.path.join(newpth, 'P07_2K')
    mf.model_ws = cpth

    mf.write_input()

    if ismf2k is not None:
        success, buff = mf.run_model(silent=True)
        assert success, '{} did not run'.format(mf.name)

    namfile = 'p7mt.nam'
    mt = flopy.mt3d.mt.Mt3dms.load(namfile, model_ws=pth, verbose=True,
                                   exe_name=mt3d_exe)
    mt.model_ws = cpth
    ftlfile = 'p7.ftl'
    mt.ftlfilename = ftlfile
    mt.write_input()
    if ismt3d is not None and ismf2k is not None:
        success, buff = mt.run_model(silent=False,
                                     normal_msg='program completed.')
        assert success, '{} did not run'.format(mt.name)
        os.remove(os.path.join(cpth, ftlfile))
    return


def test_mf2000_HSSTest():
    pth = os.path.join(pth2000, 'HSSTest')
    namfile = 'hsstest_mf2k.nam'
    mf = flopy.modflow.Modflow.load(namfile, model_ws=pth,
                                    version='mf2k', verbose=True,
                                    exe_name=mf2k_exe)

    cpth = os.path.join(newpth, 'HSSTest')
    mf.model_ws = cpth

    mf.write_input()
    if ismf2k is not None:
        success, buff = mf.run_model(silent=True)
        assert success, '{} did not run'.format(mf.name)
    namfile = 'hsstest_mt.nam'
    mt = flopy.mt3d.mt.Mt3dms.load(namfile, model_ws=pth, verbose=True,
                                   exe_name=mt3d_exe)

    mt.model_ws = cpth
    ftlfile = 'hsstest.FTL'
    mt.ftlfilename = ftlfile
    mt.write_input()

    if ismt3d is not None and ismf2k is not None:
        success, buff = mt.run_model(silent=False,
                                     normal_msg='program completed.')
        assert success, '{} did not run'.format(mt.name)
        os.remove(os.path.join(cpth, ftlfile))
    return


# cannot run this model because it uses mnw1 and there is no load for mnw1
# this model includes block format data in the btn file
def test_mf2000_mnw():
    pth = os.path.join(pth2000, 'mnw')
    namfile = 't5mf2k.nam'
    mf = flopy.modflow.Modflow.load(namfile, model_ws=pth, verbose=True)

    cpth = os.path.join(newpth, 'MNW')
    mf.model_ws = cpth

    namfile = 't5mt.nam'
    mt = flopy.mt3d.mt.Mt3dms.load(namfile, model_ws=pth, verbose=True)

    mt.change_model_ws(cpth)
    mt.write_input()
    return


def test_mf2000_MultiDiffusion():
    pth = os.path.join(pth2000, 'MultiDiffusion')
    namfile = 'p7mf2k.nam'
    mf = flopy.modflow.Modflow.load(namfile, model_ws=pth,
                                    version='mf2k', verbose=True,
                                    exe_name=mf2k_exe)

    cpth = os.path.join(newpth, 'MultiDiffusion')
    mf.model_ws = cpth

    mf.write_input()
    if ismf2k is not None:
        success, buff = mf.run_model(silent=True)
        assert success, '{} did not run'.format(mf.name)
    namfile = 'P7MT.NAM'
    mt = flopy.mt3d.mt.Mt3dms.load(namfile, model_ws=pth, verbose=True,
                                   exe_name=mt3d_exe)
    mt.model_ws = cpth
    ftlfile = 'p7.ftl'
    mt.ftlfilename = ftlfile
    mt.write_input()
    if ismt3d is not None and ismf2k is not None:
        success, buff = mt.run_model(silent=False,
                                     normal_msg='program completed.')
        assert success, '{} did not run'.format(mt.name)
        os.remove(os.path.join(cpth, ftlfile))
    return


def test_mf2000_reinject():
    pth = os.path.join(pth2000, 'reinject')
    namfile = 'p3mf2k.nam'
    mf = flopy.modflow.Modflow.load(namfile, model_ws=pth,
                                    version='mf2k', verbose=True,
                                    exe_name=mf2k_exe)

    cpth = os.path.join(newpth, 'reinject')
    mf.model_ws = cpth

    mf.write_input()
    if ismf2k is not None:
        success, buff = mf.run_model(silent=True)
        assert success, '{} did not run'.format(mf.name)

    namfile = 'p3mt.nam'
    mt = flopy.mt3d.mt.Mt3dms.load(namfile, model_ws=pth, verbose=True,
                                   exe_name=mt3d_exe)

    mt.model_ws = cpth
    ftlfile = 'p3.ftl'
    mt.ftlfilename = ftlfile
    mt.write_input()
    if ismt3d is not None and ismf2k is not None:
        success, buff = mt.run_model(silent=False,
                                     normal_msg='program completed.')
        assert success, '{} did not run'.format(mt.name)
        os.remove(os.path.join(cpth, ftlfile))
    return


def test_mf2000_SState():
    pth = os.path.join(pth2000, 'SState')
    namfile = 'SState_mf2k.nam'
    mf = flopy.modflow.Modflow.load(namfile, model_ws=pth,
                                    version='mf2k', verbose=True,
                                    exe_name=mf2k_exe)

    cpth = os.path.join(newpth, 'SState')
    mf.model_ws = cpth

    mf.write_input()
    if ismf2k is not None:
        success, buff = mf.run_model(silent=True)
        assert success, '{} did not run'.format(mf.name)

    namfile = 'SState_mt.nam'
    mt = flopy.mt3d.mt.Mt3dms.load(namfile, model_ws=pth, verbose=True,
                                   exe_name=mt3d_exe)

    mt.model_ws = cpth
    ftlfile = 'SState.ftl'
    mt.ftlfilename = ftlfile
    mt.write_input()
    if ismt3d is not None and ismf2k is not None:
        success, buff = mt.run_model(silent=False,
                                     normal_msg='program completed.')
        assert success, '{} did not run'.format(mt.name)
        os.remove(os.path.join(cpth, ftlfile))
    return


def test_mf2000_tob():
    pth = os.path.join(pth2000, 'tob')
    namfile = 'p7mf2k.nam'
    mf = flopy.modflow.Modflow.load(namfile, model_ws=pth,
                                    version='mf2k', verbose=True,
                                    exe_name=mf2k_exe)

    cpth = os.path.join(newpth, 'tob')
    mf.model_ws = cpth

    mf.lmt6.output_file_header = 'extended'
    mf.lmt6.output_file_format = 'formatted'
    mf.write_input()
    if ismf2k is not None:
        success, buff = mf.run_model(silent=True)
        assert success, '{} did not run'.format(mf.name)

    namfile = 'p7mt.nam'
    mt = flopy.mt3d.mt.Mt3dms.load(namfile, model_ws=pth, verbose=True,
                                   exe_name=mt3d_exe, forgive=True)
    mt.model_ws = cpth
    ftlfile = 'p7.ftl'
    mt.ftlfilename = ftlfile
    mt.write_input()
    if ismt3d is not None and ismf2k is not None:
        success, buff = mt.run_model(silent=False,
                                     normal_msg='program completed.')
        assert success, '{} did not run'.format(mt.name)
        os.remove(os.path.join(cpth, ftlfile))
    return


def test_mf2000_zeroth():
    pth = os.path.join(pth2000, 'zeroth')
    namfile = 'z0mf2k.nam'
    mf = flopy.modflow.Modflow.load(namfile, model_ws=pth,
                                    version='mf2k', verbose=True,
                                    exe_name=mf2k_exe)

    cpth = os.path.join(newpth, 'zeroth')
    mf.model_ws = cpth

    mf.write_input()
    if ismf2k is not None:
        success, buff = mf.run_model(silent=True)
        assert success, '{} did not run'.format(mf.name)

    namfile = 'z0mt.nam'
    mt = flopy.mt3d.mt.Mt3dms.load(namfile, model_ws=pth, verbose=True,
                                   exe_name=mt3d_exe)
    mt.model_ws = cpth
    ftlfile = 'zeroth.ftl'
    mt.ftlfilename = ftlfile
    mt.write_input()
    if ismt3d is not None and ismf2k is not None:
        success, buff = mt.run_model(silent=False,
                                     normal_msg='program completed.')
        assert success, '{} did not run'.format(mt.name)
        os.remove(os.path.join(cpth, ftlfile))
    return


def test_mfnwt_CrnkNic():
    # fix for CI failures on GitHub actions - remove once fixed in MT3D-USGS
    if 'CI' in os.environ:
        if sys.platform.lower() in ("win32", "darwin"):
            runTest = False
    else:
        runTest = True

    if runTest:
        pth = os.path.join(pthNWT, 'sft_crnkNic')
        namefile = 'CrnkNic.nam'
        mf = flopy.modflow.Modflow.load(namefile, model_ws=pth,
                                        version='mfnwt', verbose=True,
                                        exe_name=mfnwt_exe)

        cpth = os.path.join(newpth, 'SFT_CRNKNIC')
        mf.model_ws = cpth

        mf.write_input()
        if ismfnwt is not None:
            success, buff = mf.run_model(silent=False)
            assert success, '{} did not run'.format(mf.name)

        namefile = 'CrnkNic.mtnam'
        mt = flopy.mt3d.mt.Mt3dms.load(namefile, model_ws=pth, verbose=True,
                                       version='mt3d-usgs',
                                       exe_name=mt3d_usgs_exe)

        mt.model_ws = cpth
        ftlfile = 'CrnkNic.ftl'
        mt.ftlfilename = ftlfile
        mt.ftlfree = True
        mt.write_input()
        if ismt3dusgs is not None and ismfnwt is not None:
            success, buff = mt.run_model(silent=False,
                                         normal_msg='program completed.')
            assert success, '{} did not run'.format(mt.name)
            os.remove(os.path.join(cpth, ftlfile))
    return


def test_mfnwt_LKT():
    pth = os.path.join(pthNWT, 'lkt')
    namefile = 'lkt_mf.nam'
    mf = flopy.modflow.Modflow.load(namefile, model_ws=pth,
                                    version='mfnwt', verbose=True,
                                    forgive=False,
                                    exe_name=mfnwt_exe)

    assert not mf.load_fail, 'MODFLOW model did not load'

    cpth = os.path.join(newpth, 'LKT')
    mf.model_ws = cpth

    # write modflow-nwt files
    mf.write_input()

    success = False
    if ismfnwt is not None:
        success, buff = mf.run_model(silent=False)
        assert success, '{} did not run'.format(mf.name)

    namefile = 'lkt_mt.nam'
    mt = flopy.mt3d.mt.Mt3dms.load(namefile, model_ws=pth, verbose=True,
                                   version='mt3d-usgs', exe_name=mt3d_usgs_exe,
                                   modflowmodel=mf)
    mt.model_ws = cpth
    ftlfile = 'lkt.ftl'
    mt.ftlfilename = ftlfile
    mt.ftlfree = True

    # write mt3d files
    mt.write_input()

    if ismt3dusgs is not None and ismfnwt is not None and success:
        success, buff = mt.run_model(silent=False,
                                     normal_msg='program completed.')
        assert success, '{} did not run'.format(mt.name)
        os.remove(os.path.join(cpth, ftlfile))
    return


def test_mfnwt_keat_uzf():
    pth = os.path.join(pthNWT, 'keat_uzf')
    namefile = 'Keat_UZF_mf.nam'
    mf = flopy.modflow.Modflow.load(namefile, model_ws=pth,
                                    version='mfnwt', verbose=True,
                                    exe_name=mfnwt_exe)

    cpth = os.path.join(newpth, 'KEAT_UZF')
    mf.model_ws = cpth

    mf.write_input()
    if ismfnwt is not None:
        success, buff = mf.run_model(silent=True)
        assert success, '{} did not run'.format(mf.name)

    namefile = 'Keat_UZF_mt.nam'
    mt = flopy.mt3d.mt.Mt3dms.load(namefile, model_ws=pth, verbose=True,
                                   version='mt3d-usgs', exe_name=mt3d_usgs_exe)
    # Check a few options specified as optional keywords on line 3
    assert mt.btn.DRYCell is True
    assert mt.btn.Legacy99Stor is False
    assert mt.btn.MFStyleArr is True
    assert mt.btn.AltWTSorb is False

    mt.model_ws = cpth
    ftlfile = 'Keat_UZF.ftl'
    mt.ftlfilename = ftlfile
    mt.ftlfree = True
    mt.write_input()
    if ismt3dusgs is not None and ismfnwt is not None:
        success, buff = mt.run_model(silent=False,
                                     normal_msg='program completed.')
        assert success, '{} did not run'.format(mt.name)
        os.remove(os.path.join(cpth, ftlfile))
    return


if __name__ == '__main__':
    # test_mf2000_mnw()
    # test_mf2005_p07()
    # test_mf2000_p07()
    # test_mf2000_HSSTest()
    # test_mf2000_MultiDiffusion()
    # test_mf2000_reinject()
    # test_mf2000_SState()
    # test_mf2000_tob()
    # test_mf2000_zeroth()
    test_mfnwt_CrnkNic()
    # test_mfnwt_LKT()
    # test_mfnwt_keat_uzf()
