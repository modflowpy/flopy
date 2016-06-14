import os
import shutil
import flopy


pthtest = os.path.join('..', 'examples', 'data', 'swtv4_test')
newpth = os.path.join('.', 'temp')
swtv4_exe = 'swt_v4'
isswtv4 = flopy.which(swtv4_exe)
runmodel = False
verbose = False


def test_1_swtv4_ex():
    d = '1_box'
    subds = ['case1', 'case2']
    for subd in subds:
        pth = os.path.join(pthtest, d, subd)
        testpth = os.path.join(newpth, d + '-' + subd)
        if os.path.isdir(testpth):
            shutil.rmtree(testpth)
        os.mkdir(testpth)
        namfile = 'seawat.nam'
        m = flopy.seawat.swt.Seawat.load(namfile, model_ws=pth,
                                          verbose=verbose)
        m.model_ws = testpth
        m.write_input()
        if isswtv4 is not None and runmodel:
            success, buff = m.run_model(silent=False)
            assert success, '{} did not run'.format(m.name)
    return


def test_2_swtv4_ex():
    d = '2_henry'
    subds = ['1_classic_case1', '2_classic_case2', '3_VDF_no_Trans',
             '4_VDF_uncpl_Trans', '5_VDF_DualD_Trans', '6_age_simulation']
    for subd in subds:
        pth = os.path.join(pthtest, d, subd)
        testpth = os.path.join(newpth, d + '-' + subd)
        if os.path.isdir(testpth):
            shutil.rmtree(testpth)
        os.mkdir(testpth)
        namfile = 'seawat.nam'
        if subd == '6_age_simulation':
            namfile = 'henry_mod.nam'
        m = flopy.seawat.swt.Seawat.load(namfile, model_ws=pth,
                                          verbose=verbose)
        m.model_ws = testpth
        m.write_input()
        if isswtv4 is not None and runmodel:
            success, buff = m.run_model(silent=False)
            assert success, '{} did not run'.format(m.name)
    return


def test_3_swtv4_ex():
    d = '3_elder'
    subds = ['']
    for subd in subds:
        pth = os.path.join(pthtest, d, subd)
        testpth = os.path.join(newpth, d + '-' + subd)
        if os.path.isdir(testpth):
            shutil.rmtree(testpth)
        os.mkdir(testpth)
        namfile = 'seawat.nam'
        m = flopy.seawat.swt.Seawat.load(namfile, model_ws=pth,
                                          verbose=verbose)
        m.model_ws = testpth
        m.write_input()
        if isswtv4 is not None and runmodel:
            success, buff = m.run_model(silent=False)
            assert success, '{} did not run'.format(m.name)
    return


def test_4_swtv4_ex():
    d = '4_hydrocoin'
    subds = ['']
    for subd in subds:
        pth = os.path.join(pthtest, d, subd)
        testpth = os.path.join(newpth, d + '-' + subd)
        if os.path.isdir(testpth):
            shutil.rmtree(testpth)
        os.mkdir(testpth)
        namfile = 'seawat.nam'
        m = flopy.seawat.swt.Seawat.load(namfile, model_ws=pth,
                                          verbose=verbose)
        m.model_ws = testpth
        m.write_input()
        if isswtv4 is not None and runmodel:
            success, buff = m.run_model(silent=False)
            assert success, '{} did not run'.format(m.name)
    return


def test_5_swtv4_ex():
    d = '5_saltlake'
    subds = ['']
    for subd in subds:
        pth = os.path.join(pthtest, d, subd)
        testpth = os.path.join(newpth, d + '-' + subd)
        if os.path.isdir(testpth):
            shutil.rmtree(testpth)
        os.mkdir(testpth)
        namfile = 'seawat.nam'
        m = flopy.seawat.swt.Seawat.load(namfile, model_ws=pth,
                                          verbose=verbose)
        m.model_ws = testpth
        m.write_input()
        if isswtv4 is not None and runmodel:
            success, buff = m.run_model(silent=False)
            assert success, '{} did not run'.format(m.name)
    return


def test_6_swtv4_ex():
    d = '6_rotation'
    subds = ['1_symmetric', '2_asymmetric']
    for subd in subds:
        pth = os.path.join(pthtest, d, subd)
        testpth = os.path.join(newpth, d + '-' + subd)
        if os.path.isdir(testpth):
            shutil.rmtree(testpth)
        os.mkdir(testpth)
        namfile = 'seawat.nam'
        m = flopy.seawat.swt.Seawat.load(namfile, model_ws=pth,
                                          verbose=verbose)
        m.model_ws = testpth
        m.write_input()
        if isswtv4 is not None and runmodel:
            success, buff = m.run_model(silent=False)
            assert success, '{} did not run'.format(m.name)
    return


def test_7_swtv4_ex():
    d = '7_swtv4_ex'
    subds = ['case1', 'case2', 'case3', 'case4', 'case5', 'case6', 'case7']
    for subd in subds:
        pth = os.path.join(pthtest, d, subd)
        testpth = os.path.join(newpth, d + '-' + subd)
        if not os.path.isdir(testpth):
            os.mkdir(testpth)
        namfile = 'seawat.nam'
        m = flopy.seawat.swt.Seawat.load(namfile, model_ws=pth,
                                          verbose=verbose)
        m.model_ws = testpth
        m.write_input()
        if isswtv4 is not None and runmodel:
            success, buff = m.run_model(silent=False)
            assert success, '{} did not run'.format(m.name)
    return


if __name__ == '__main__':
    test_1_swtv4_ex()
    test_2_swtv4_ex()
    test_3_swtv4_ex()
    test_4_swtv4_ex()
    test_5_swtv4_ex()
    test_6_swtv4_ex()
    test_7_swtv4_ex()
