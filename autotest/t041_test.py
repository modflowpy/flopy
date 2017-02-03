"""
Test the observation process load and write
"""
import os
import shutil
import numpy as np
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
    test041 load and write of MODFLOW-2005 OBS example problem
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
        assert success, 'could not run original MODFLOW-2005 model'

        try:
            iu = mf.hob.iuhobsv
            fpth = mf.get_output(unit=iu)
            pth0 = os.path.join(opth, fpth)
            obs0 = np.genfromtxt(pth0, skip_header=1)
        except:
            raise ValueError('could not load original HOB output file')

    npth = os.path.join(cpth, 'tc1-true', 'new')
    mf.change_model_ws(new_pth=npth, reset_external=True)

    # write the lgr model in to the new path
    mf.write_input()

    # run the modflow-2005 model
    if run:
        success, buff = mf.run_model(silent=False)
        assert success, 'could not run new MODFLOW-2005 model'

        # compare parent results
        try:
            pth1 = os.path.join(npth, fpth)
            obs1 = np.genfromtxt(pth1, skip_header=1)

            msg = 'new simulated heads are not approximately equal'
            assert np.allclose(obs0[:, 0], obs1[:, 0], atol=1e-4), msg

            msg = 'new observed heads are not approximately equal'
            assert np.allclose(obs0[:, 1], obs1[:, 1], atol=1e-4), msg
        except:
            raise ValueError('could not load new HOB output file')



if __name__ == '__main__':
    test_obs_load_and_write()
