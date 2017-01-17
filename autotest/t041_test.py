"""
Test the observation process load and write
"""
import os
import shutil
import flopy
try:
    import pymake
except:
    print('could not import pymake')

cpth = os.path.join('temp', 't041')
# delete the directory if it exists
if os.path.isdir(cpth):
    shutil.rmtree(cpth)

exe_name = 'mf2005'
v = flopy.which(exe_name)

run = True
if v is None:
    run = False


def test_obs_load_and_write():
    """
    Test load and write of distributed MODFLOW-LGR example problem
    """
    pth = os.path.join('..', 'examples', 'data', 'mf2005_obs')
    opth = os.path.join(cpth, 'tc1-true', 'orig')
    # delete the directory if it exists
    if os.path.isdir(opth):
        shutil.rmtree(opth)
    os.makedirs(opth)
    # copy the original files
    files = os.listdir(pth)
    for file in files:
        src = os.path.join(pth, file)
        dst = os.path.join(opth, file)
        shutil.copyfile(src, dst)

    # load the modflow model
    mf = flopy.modflow.Modflow.load('tc1-true.nam', verbose=True,
                                    model_ws=opth, exe_name=exe_name)

    # run the modflow-2005 model
    if run:
        success, buff = mf.run_model(silent=False)
        assert success, 'could not run original modflow-2005 model'

    npth = os.path.join(cpth, 'tc1-true', 'new')
    mf.change_model_ws(new_pth=npth, reset_external=True)

    # write the lgr model in to the new path
    mf.write_input()

    # run the modflow-2005 model
    if run:
        success, buff = mf.run_model(silent=False)
        assert success, 'could not run new modflow-2005 model'

        # compare parent results
        pth0 = os.path.join(opth, 'tc1-true.nam')
        pth1 = os.path.join(npth, 'tc1-true.nam')
        # try:
        #     msg = 'parent heads do not match'
        #     success = pymake.compare_heads(pth0, pth1)
        #     assert success, msg
        # except:
        #     pass


if __name__ == '__main__':
    test_obs_load_and_write()
