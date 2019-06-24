"""
Test the observation process load and write
"""
import os
import shutil
import numpy as np
import flopy
from nose.tools import raises

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


def test_hob_simple():
    """
    test041 create and run a simple MODFLOW-2005 OBS example
    """
    pth = os.path.join(cpth, 'simple')
    modelname = 'hob_simple'
    nlay, nrow, ncol = 1, 11, 11
    shape3d = (nlay, nrow, ncol)
    shape2d = (nrow, ncol)
    ib = np.ones(shape3d, dtype=np.int)
    ib[0, 0, 0] = -1
    m = flopy.modflow.Modflow(modelname=modelname, model_ws=pth,
                              verbose=False, exe_name=exe_name, )
    dis = flopy.modflow.ModflowDis(m, nlay=1, nrow=11, ncol=11, nper=2,
                                   perlen=[1, 1])

    bas = flopy.modflow.ModflowBas(m, ibound=ib, strt=10.)
    lpf = flopy.modflow.ModflowLpf(m)
    pcg = flopy.modflow.ModflowPcg(m)
    obs = flopy.modflow.HeadObservation(m, layer=0, row=5, column=5,
                                        time_series_data=[[1., 54.4],
                                                          [2., 55.2]])
    hob = flopy.modflow.ModflowHob(m, iuhobsv=51, hobdry=-9999.,
                                   obs_data=[obs])

    # Write the model input files
    m.write_input()

    # run the modflow-2005 model
    if run:
        success, buff = m.run_model(silent=False)
        assert success, 'could not run simple MODFLOW-2005 model'

    return


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


def test_obs_create_and_write():
    """
    test041 create and write of MODFLOW-2005 OBS example problem
    """
    pth = os.path.join('..', 'examples', 'data', 'mf2005_obs')
    opth = os.path.join(cpth, 'create', 'orig')
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
                                    model_ws=opth, exe_name=exe_name,
                                    forgive=False)
    # remove the existing hob package
    iuhob = mf.hob.unit_number[0]
    mf.remove_package('HOB')

    # create a new hob object
    obs_data = []

    # observation location 1
    tsd = [[1., 1.], [87163., 2.], [348649., 3.],
           [871621., 4.], [24439070., 5.], [24439072., 6.]]
    names = ['o1.1', 'o1.2', 'o1.3', 'o1.4', 'o1.5', 'o1.6']
    obs_data.append(flopy.modflow.HeadObservation(mf, layer=0, row=2, column=0,
                                                  time_series_data=tsd,
                                                  names=names, obsname='o1'))
    # observation location 2
    tsd = [[0., 126.938], [87163., 126.904], [871621., 126.382],
           [871718.5943, 115.357], [871893.7713, 112.782]]
    names = ['o2.1', 'o2.2', 'o2.3', 'o2.4', 'o2.5']
    obs_data.append(flopy.modflow.HeadObservation(mf, layer=0, row=3, column=3,
                                                  time_series_data=tsd,
                                                  names=names, obsname='o2'))
    hob = flopy.modflow.ModflowHob(mf, iuhobsv=51, obs_data=obs_data,
                                   unitnumber=iuhob)
    # write the hob file
    hob.write_file()

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

    npth = os.path.join(cpth, 'create', 'new')
    mf.change_model_ws(new_pth=npth, reset_external=True)

    # write the model at the new path
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


def test_filenames():
    """
    test041 load and run a simple MODFLOW-2005 OBS example with specified
    filenames
    """
    print('test041 load and run a simple MODFLOW-2005 OBS example with'
          ' specified filenames')
    pth = os.path.join(cpth, 'simple')
    modelname = 'hob_simple'
    pkglst = ['dis', 'bas6', 'pcg', 'lpf']
    m = flopy.modflow.Modflow.load(modelname + '.nam', model_ws=pth,
                                   check=False, load_only=pkglst,
                                   verbose=False, exe_name=exe_name,
                                   forgive=False)

    obs = flopy.modflow.HeadObservation(m, layer=0, row=5, column=5,
                                        time_series_data=[[1., 54.4],
                                                          [2., 55.2]])
    f_in = modelname + '_custom_fname.hob'
    f_out = modelname + '_custom_fname.hob.out'
    filenames = [f_in, f_out]
    hob = flopy.modflow.ModflowHob(m, iuhobsv=51, hobdry=-9999.,
                                   obs_data=[obs], options=['NOPRINT'],
                                   filenames=filenames)

    # Write the model input files
    m.write_input()

    s = 'output filename ({}) does ' \
        'not match specified name'.format(m.get_output(unit=51))
    assert m.get_output(unit=51) == f_out, s
    s = 'specified HOB input file not found'
    assert os.path.isfile(os.path.join(pth, f_in)), s

    # run the modflow-2005 model
    if run:
        success, buff = m.run_model(silent=False)
        assert success, 'could not run simple MODFLOW-2005 model'

    return


def test_multilayerhob_pr():
    """
    test041 test multilayer obs PR == 1 criteria with problematic PRs
    """
    ml = flopy.modflow.Modflow()
    dis = flopy.modflow.ModflowDis(ml, nlay=3, nrow=1, ncol=1, nper=1,
                                   perlen=[1])
    flopy.modflow.HeadObservation(ml,layer=-3,row=0,column=0,
                                  time_series_data=[[1.0, 0]],
                                  mlay={0:0.19, 1:0.69, 2:0.12})
    return


@raises(ValueError)
def test_multilayerhob_prfail():
    """
    test041 failure of multilayer obs PR == 1 criteria
    """
    ml = flopy.modflow.Modflow()
    dis = flopy.modflow.ModflowDis(ml, nlay=3, nrow=1, ncol=1, nper=1,
                                   perlen=[1])
    flopy.modflow.HeadObservation(ml,layer=-3,row=0,column=0,
                                  time_series_data=[[1.0, 0]],
                                  mlay={0:0.50, 1:0.50, 2:0.01})
    return


def test_flwob_load():
    """
    test041 create, write, and load ModflowFlwob package.
    """
    # load the modflow model
    opth = os.path.join(cpth, 'tc1-true', 'orig')
    m = flopy.modflow.Modflow.load('tc1-true.nam', verbose=True,
                                   model_ws=opth, exe_name=exe_name)

    npth = os.path.join(cpth, 'tc1-true', 'flwob')
    m.change_model_ws(new_pth=npth, reset_external=True)

    # write the lgr model in to the new path
    m.write_input()

    # add DRN package
    spd = {0: [[0, 5, 5, .5, 8e6],
               [0, 8, 8, .7, 8e6]]}
    drn = flopy.modflow.ModflowDrn(m, 53, stress_period_data=spd)

    # flow observation

    # Lists of length nqfb
    nqobfb = [1, 1]
    nqclfb = [1, 1]

    # Lists of length nqtfb
    obsnam = ['drob_1', 'drob_2']
    irefsp = [0, 0]
    toffset = [0, 0]
    flwobs = [-5.678, -6.874]

    # Lists of length (nqfb, nqclfb)
    layer = [[0], [0]]
    row = [[5], [8]]
    column = [[5], [8]]
    factor = [[1.], [1.]]

    drob = flopy.modflow.ModflowFlwob(m,
                                      nqfb=len(nqclfb),
                                      nqcfb=np.sum(nqclfb),
                                      nqtfb=np.sum(nqobfb),
                                      nqobfb=nqobfb,
                                      nqclfb=nqclfb,
                                      obsnam=obsnam,
                                      irefsp=irefsp,
                                      toffset=toffset,
                                      flwobs=flwobs,
                                      layer=layer,
                                      row=row,
                                      column=column,
                                      factor=factor,
                                      flowtype='drn',
                                      options=['NOPRINT'])
    # Write the model input files
    m.write_input(check=False)

    # Load the DROB package and compare it to the original
    pkglst = ['drob']
    m = flopy.modflow.Modflow.load('tc1-true.nam', model_ws=npth,
                                   check=False, load_only=pkglst,
                                   verbose=False, exe_name=exe_name,
                                   forgive=False)

    # check variables were read properly
    s = 'nqfb loaded from {} read incorrectly'.format(m.drob.fn_path)
    assert(drob.nqfb == m.drob.nqfb), s
    s = 'nqcfb loaded from {} read incorrectly'.format(m.drob.fn_path)
    assert (drob.nqcfb == m.drob.nqcfb), s
    s = 'nqtfb loaded from {} read incorrectly'.format(m.drob.fn_path)
    assert (drob.nqtfb == m.drob.nqtfb), s
    s = 'obsnam loaded from {} read incorrectly'.format(m.drob.fn_path)
    assert (list([n for n in drob.obsnam]) ==
            list([n for n in m.drob.obsnam])), s
    s = 'flwobs loaded from {} read incorrectly'.format(m.drob.fn_path)
    assert np.array_equal(drob.flwobs, m.drob.flwobs), s
    s = 'layer loaded from {} read incorrectly'.format(m.drob.fn_path)
    assert np.array_equal(drob.layer, m.drob.layer), s
    s = 'row loaded from {} read incorrectly'.format(m.drob.fn_path)
    assert np.array_equal(drob.row, m.drob.row), s
    s = 'column loaded from {} read incorrectly'.format(m.drob.fn_path)
    assert np.array_equal(drob.column, m.drob.column), s

    return


if __name__ == '__main__':
    test_hob_simple()
    test_obs_create_and_write()
    test_obs_load_and_write()
    test_filenames()
    test_flwob_load()
