"""
Test the observation process load and write
"""
import os
import shutil
import filecmp
import flopy
try:
    import pymake
except:
    print('could not import pymake')

cpth = os.path.join('temp', 't043')
# delete the directory if it exists
if os.path.isdir(cpth):
    shutil.rmtree(cpth)

exe_name = 'mf2005'
v = flopy.which(exe_name)

run = True
if v is None:
    run = False


def test_gage_load_and_write():
    """
    test043 load and write of MODFLOW-2005 GAGE example problem
    """
    pth = os.path.join('..', 'examples', 'data', 'mf2005_test')
    opth = os.path.join(cpth, 'testsfr2_tab', 'orig')
    # delete the directory if it exists
    if os.path.isdir(opth):
        shutil.rmtree(opth)
    os.makedirs(opth)
    # copy the original files
    fpth = os.path.join(pth,'testsfr2_tab.nam')
    try:
        pymake.setup(fpth, opth)
    except:
        opth = pth

    # load the modflow model
    mf = flopy.modflow.Modflow.load('testsfr2_tab.nam', verbose=True,
                                    model_ws=opth, exe_name=exe_name)

    # run the modflow-2005 model
    if run:
        success, buff = mf.run_model(silent=False)
        assert success, 'could not run original MODFLOW-2005 model'

        try:
            files = mf.gage.files
        except:
            raise ValueError('could not load original GAGE output files')

    npth = os.path.join(cpth, 'testsfr2_tab', 'new')
    mf.change_model_ws(new_pth=npth, reset_external=True)

    # write the modflow model in to the new path
    mf.write_input()

    # run the modflow-2005 model
    if run:
        success, buff = mf.run_model(silent=False)
        assert success, 'could not run new MODFLOW-2005 model'

        # compare the two results
        try:
            for f in files:
                pth0 = os.path.join(opth, f)
                pth1 = os.path.join(npth, f)
                msg = 'new and original gage file "{}" '.format(f) + \
                      'are not binary equal.'
                assert filecmp.cmp(pth0, pth1), msg
        except:
            raise ValueError('could not load new GAGE output files')



if __name__ == '__main__':
    test_gage_load_and_write()
