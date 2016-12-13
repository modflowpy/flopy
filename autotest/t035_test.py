"""
Test the lgr model
"""
import os
import shutil
import flopy

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
        lgr.run_model(silent=False)

    # check that a parent and child were read
    msg = 'modflow-lgr ex3 does not have 2 grids'
    assert lgr.ngrids == 2, msg

    npth = os.path.join(cpth, 'ex3', 'new')
    lgr.change_model_ws(new_pth=npth, reset_external=True)

    # write the lgr model in to the new path
    lgr.write_input()

    # run the lgr model
    if run:
        lgr.run_model(silent=False)

def test_complexlgr_load_and_write():
    pth = os.path.join('..', 'examples', 'data', 'mflgr_v2', 'ex3sd')
    opth = os.path.join(cpth, 'ex3sd', 'orig')
    # delete the directory if it exists
    if os.path.isdir(opth):
        shutil.rmtree(opth)
    os.makedirs(opth)
    # copy the original files
    dirs = os.listdir(pth)
    for dir in dirs:
        # copy files in root directory
        src = os.path.join(pth, dir)
        if os.path.isfile(src):
            dst = os.path.join(opth, dir)
            shutil.copyfile(src, dst)
        # copy files in subdirectories
        tpth = os.path.join(pth, dir)
        if os.path.isdir(tpth):
            # make the dst directory if it does not exist
            dpth = os.path.join(opth, dir)
            if not os.path.isdir(dpth):
                os.makedirs(dpth)
            # copy the original files
            files = os.listdir(tpth)
            for file in files:
                src = os.path.join(tpth, file)
                dst = os.path.join(opth, dir, file)
                shutil.copyfile(src, dst)
    return

    # load the lgr model
    lgr = flopy.modflowlgr.ModflowLgr.load('ex3.lgr', verbose=True,
                                           model_ws=opth, exe_name=exe_name)

    # run the lgr model
    if run:
        lgr.run_model(silent=False)

    # check that a parent and child were read
    msg = 'modflow-lgr ex3 does not have 2 grids'
    assert lgr.ngrids == 2, msg

    npth = os.path.join(cpth, 'ex3sd', 'new')
    lgr.change_model_ws(new_pth=npth, reset_external=True)

    # write the lgr model in to the new path
    lgr.write_input()

    # run the lgr model
    if run:
        lgr.run_model(silent=False)

if __name__ == '__main__':
    test_complexlgr_load_and_write()
    test_simplelgr_load_and_write()
