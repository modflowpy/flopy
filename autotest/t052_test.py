import shutil
import os
import numpy as np
import flopy
try:
    import pymake
except:
    print('could not import pymake')


cpth = os.path.join('temp', 't052')
# delete the directory if it exists
if os.path.isdir(cpth):
    shutil.rmtree(cpth)
# make the directory
os.makedirs(cpth)

exe_name = 'mf2005'
v = flopy.which(exe_name)

run = True
if v is None:
    run = False


def test_binary_well():

    nlay = 3
    nrow = 3
    ncol = 3
    mfnam = 't1'
    ml = flopy.modflow.Modflow(modelname=mfnam, model_ws=cpth, verbose=True,
                               exe_name=exe_name)
    dis = flopy.modflow.ModflowDis(ml, nlay=nlay, nrow=nrow, ncol=ncol, top=0,
                                   botm=[-1., -2., -3.])
    ibound = np.ones((nlay, nrow, ncol), dtype=np.int)
    ibound[0, 1, 1] = 0
    ibound[0, 0, -1] = -1
    bas = flopy.modflow.ModflowBas(ml, ibound=ibound)
    lpf = flopy.modflow.ModflowLpf(ml, ipakcb=102)
    wd = flopy.modflow.ModflowWel.get_empty(ncells=2,
                                            aux_names=['v1', 'v2'])
    wd['k'][0] = 2
    wd['i'][0] = 2
    wd['j'][0] = 2
    wd['flux'][0] = -1000.
    wd['v1'][0] = 1.
    wd['v2'][0] = 2.
    wd['k'][1] = 2
    wd['i'][1] = 1
    wd['j'][1] = 1
    wd['flux'][1] = -500.
    wd['v1'][1] = 200.
    wd['v2'][1] = 100.
    wel_data = {0: wd}
    wel = flopy.modflow.ModflowWel(ml, stress_period_data=wel_data,
                                   dtype=wd.dtype)
    oc = flopy.modflow.ModflowOc(ml)
    pcg = flopy.modflow.ModflowPcg(ml)

    ml.write_input()

    # run the modflow-2005 model
    if run:
        success, buff = ml.run_model(silent=False)
        assert success, 'could not run MODFLOW-2005 model'
        fn0 = os.path.join(cpth, mfnam+'.nam')


    # load the model
    m = flopy.modflow.Modflow.load(mfnam+'.nam', model_ws=cpth,
                                   verbose=True, exe_name=exe_name)

    wl = m.wel.stress_period_data[0]
    msg = 'previous well package stress period data does not match ' + \
          'stress period data loaded.'
    assert np.array_equal(wel.stress_period_data[0], wl), msg

    # change model work space
    pth = os.path.join(cpth, 'flopy')
    m.change_model_ws(new_pth=pth)

    # remove the existing well package
    m.remove_package('WEL')

    # recreate well package with binary output
    wel = flopy.modflow.ModflowWel(m, stress_period_data=wel_data,
                                   binary=True, dtype=wd.dtype)

    # write the model to the new path
    m.write_input()

    # run the new modflow-2005 model
    if run:
        success, buff = m.run_model(silent=False)
        assert success, 'could not run the new MODFLOW-2005 model'
        fn1 = os.path.join(pth, mfnam+'.nam')

    # compare the files
    if run:
        fsum = os.path.join(cpth,
                            '{}.head.out'.format(os.path.splitext(mfnam)[0]))
        success = False
        try:
            success = pymake.compare_heads(fn0, fn1, outfile=fsum)
        except:
            print('could not perform head comparison')

        assert success, 'head comparison failure'

        fsum = os.path.join(cpth,
                            '{}.budget.out'.format(os.path.splitext(mfnam)[0]))
        success = False
        try:
            success = pymake.compare_budget(fn0, fn1,
                                            max_incpd=0.1, max_cumpd=0.1,
                                            outfile=fsum)
        except:
            print('could not perform budget comparison')

        assert success, 'budget comparison failure'

    # clean up
    shutil.rmtree(cpth)

if __name__ == '__main__':
    test_binary_well()
