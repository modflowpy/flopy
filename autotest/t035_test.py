"""
Test the lgr model
"""
import os
import shutil
import flopy
try:
    import pymake
except:
    print('could not import pymake')

cpth = os.path.join('temp', 't035')
# delete the directory if it exists
if os.path.isdir(cpth):
    shutil.rmtree(cpth)

exe_name = 'mflgr'
v = flopy.which(exe_name)

run = True
if v is None:
    run = False


def test_simplelgr_load_and_write():
    """
    Test load and write of distributed MODFLOW-LGR example problem
    """
    pth = os.path.join('..', 'examples', 'data', 'mflgr_v2', 'ex3')
    opth = os.path.join(cpth, 'ex3', 'orig')
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

    # load the lgr model
    lgr = flopy.modflowlgr.ModflowLgr.load('ex3.lgr', verbose=True,
                                           model_ws=opth, exe_name=exe_name)

    # run the lgr model
    if run:
        success, buff = lgr.run_model(silent=False)
        assert success, 'could not run original modflow-lgr model'

    # check that a parent and child were read
    msg = 'modflow-lgr ex3 does not have 2 grids'
    assert lgr.ngrids == 2, msg

    npth = os.path.join(cpth, 'ex3', 'new')
    lgr.change_model_ws(new_pth=npth, reset_external=True)

    # write the lgr model in to the new path
    lgr.write_input()

    # run the lgr model
    if run:
        success, buff = lgr.run_model(silent=False)
        assert success, 'could not run new modflow-lgr model'

        # compare parent results
        pth0 = os.path.join(opth, 'ex3_parent.nam')
        pth1 = os.path.join(npth, 'ex3_parent.nam')
        try:
            msg = 'parent heads do not match'
            success = pymake.compare_heads(pth0, pth1)
            assert success, msg
        except:
            pass

        # compare child results
        pth0 = os.path.join(opth, 'ex3_child.nam')
        pth1 = os.path.join(npth, 'ex3_child.nam')
        try:
            msg = 'child heads do not match'
            success = pymake.compare_heads(pth0, pth1)
            assert success, msg
        except:
            pass


if __name__ == '__main__':
    test_simplelgr_load_and_write()
