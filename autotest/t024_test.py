import sys
sys.path.insert(0, '..')
import os
import glob
import numpy as np
import flopy

def test_checker_on_load():
    # load all of the models in the mf2005_test folder
    # model level checks are performed by default on load()
    model_ws = '../examples/data/mf2005_test/'
    testmodels = [os.path.split(f)[-1] for f in glob.glob(model_ws + '*.nam')]

    for mfnam in testmodels:
        m = flopy.modflow.Modflow.load(mfnam, model_ws=model_ws)

def test_bcs_check():
    mf = flopy.modflow.Modflow(version='mf2005',
                               model_ws='temp')

    # test check for isolated cells
    dis = flopy.modflow.ModflowDis(mf, nlay=2, nrow=3, ncol=3, top=100, botm=95)
    bas = flopy.modflow.ModflowBas(mf, ibound=np.ones((2, 3, 3), dtype=int))
    chk = bas.check()

    dis = flopy.modflow.ModflowDis(mf, nlay=3, nrow=5, ncol=5, top=100, botm=95)
    ibound = np.zeros((3, 5, 5), dtype=int)
    ibound[1, 1, 1] = 1 # fully isolated cell
    ibound[0:2, 4, 4] = 1 # cell connected vertically to one other cell
    bas = flopy.modflow.ModflowBas(mf, ibound=ibound)
    chk = bas.check()
    assert chk.summary_array['desc'][0] == 'isolated cells in ibound array'
    assert chk.summary_array.i[0] == 1 and chk.summary_array.i[0] == 1 and chk.summary_array.j[0] == 1
    assert len(chk.summary_array) == 1

    ghb = flopy.modflow.ModflowGhb(mf, stress_period_data={0: [0, 0, 0, 100, 1]})
    riv = flopy.modflow.ModflowRiv(mf, stress_period_data={0: [[0, 0, 0, 101, 10, 100],
                                                               [0, 0, 1, 80, 10, 90]]})
    chk = ghb.check()
    assert chk.summary_array['desc'][0] == 'BC in inactive cell'
    chk = riv.check()
    assert chk.summary_array['desc'][4] == 'RIV stage below rbots'
    assert np.array_equal(chk.summary_array['j'], np.array([0, 1, 1, 1, 1]))

def test_properties_check():

    # test that storage values ignored for steady state
    mf = flopy.modflow.Modflow(version='mf2005',
                               model_ws='temp')
    dis = flopy.modflow.ModflowDis(mf, nrow=2, ncol=2, nper=3, steady=True)
    lpf = flopy.modflow.ModflowLpf(mf, sy=np.ones((2, 2)), ss=np.ones((2, 2)))
    chk = lpf.check()
    assert len(chk.summary_array) == 0

    # test k values check
    lpf = flopy.modflow.ModflowLpf(mf,
                                   hk=np.array([[1, 1e10], [1, -1]]),
                                   hani=np.array([[1, 1], [1, 0]]),
                                   vka=np.array([[1, 0], [1, 1e-20]]))
    chk = lpf.check()
    ind1 = np.array([True if list(inds) == [0, 1, 1]
                     else False for inds in chk.summary_array[['k', 'i', 'j']]])
    ind1_errors = chk.summary_array[ind1]['desc']
    ind2 = np.array([True if list(inds) == [0, 0, 1]
                     else False for inds in chk.summary_array[['k', 'i', 'j']]])
    ind2_errors = chk.summary_array[ind1]['desc']
    assert 'zero or negative horizontal hydraulic conductivity values' in ind1_errors
    assert 'horizontal hydraulic conductivity values below checker threshold of 1e-11' in ind1_errors
    assert 'vertical hydraulic conductivity values below checker threshold of 1e-11' in ind1_errors
    assert 'horizontal hydraulic conductivity values above checker threshold of 100000.0' in ind2_errors
    assert 'zero or negative vertical hydraulic conductivity values' in ind2_errors

    j=2

if __name__ == '__main__':
    #test_checker_on_load()
    test_bcs_check()
    test_properties_check()
