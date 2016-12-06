"""
Test the lgr model
"""
import os
import shutil
import flopy

cpth = os.path.join('temp', 't035')

exe_name = 'mflgr'
v = flopy.which(exe_name)

run = True
if v is None:
    run = False


def test_load_and_write():
    pth = os.path.join('..', 'examples', 'data', 'mflgr_v2', 'ex3')
    opth = os.path.join(cpth, 'orig')
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

    npth = os.path.join(cpth, 'new')
    lgr.change_model_ws(new_pth=npth)

    # write the lgr model in to the new path
    lgr.write_input()

    # run the lgr model
    #if run:
    #    lgr.run_model(silent=False)

    print('write lgr model files')


if __name__ == '__main__':
    test_load_and_write()
