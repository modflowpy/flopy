import os
import numpy as np
import flopy

model_ws = os.path.join('..', 'examples', 'data', 'mf2005_test')
testmodels = [os.path.join(model_ws, f) for f in os.listdir(model_ws)
              if f.endswith('.nam')]


def txest_checker_on_load():
    # load all of the models in the mf2005_test folder
    # model level checks are performed by default on load()
    for mfnam in testmodels:
        yield checker_on_load, mfnam


def checker_on_load(mfnam):
    f = os.path.basename(mfnam)
    d = os.path.dirname(mfnam)
    m = flopy.modflow.Modflow.load(f, model_ws=d)
    assert isinstance(m,
                      flopy.modflow.Modflow), 'Not a flopy.modflow.Modflow instance'


def test_bcs_check():
    mf = flopy.modflow.Modflow(version='mf2005',
                               model_ws='temp')
    dis = flopy.modflow.ModflowDis(mf, top=100, botm=95)
    bas = flopy.modflow.ModflowBas(mf, ibound=np.array([[0, 1], [1, 1]]))
    ghb = flopy.modflow.ModflowGhb(mf,
                                   stress_period_data={0: [0, 0, 0, 100, 1]})
    riv = flopy.modflow.ModflowRiv(mf, stress_period_data={
        0: [[0, 0, 0, 101, 10, 100],
            [0, 0, 1, 80, 10, 90]]})
    chk = ghb.check()
    assert chk.summary_array['desc'][0] == 'BC in inactive cell'
    chk = riv.check()
    assert chk.summary_array['desc'][3] == 'RIV stage below rbots'
    assert np.array_equal(chk.summary_array['j'], np.array([0, 1, 1, 1]))


if __name__ == '__main__':
    for mfnam in testmodels:
        checker_on_load(mfnam)
    test_bcs_check()
