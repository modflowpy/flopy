import os
import flopy
import numpy as np


tpth = os.path.abspath(os.path.join('temp', 't016'))
if not os.path.isdir(tpth):
    os.makedirs(tpth)


exe_name = 'mfusg'
v = flopy.which(exe_name)

run = True
if v is None:
    run = False


def test_usg_disu_load():

    pthusgtest = os.path.join('..', 'examples', 'data', 'mfusg_test',
                              '01A_nestedgrid_nognc')
    fname = os.path.join(pthusgtest, 'flow.disu')
    assert os.path.isfile(fname), 'disu file not found {}'.format(fname)

    # Create the model
    m = flopy.modflow.Modflow(modelname='usgload', verbose=True)

    # Load the disu file
    disu = flopy.modflow.ModflowDisU.load(fname, m)
    assert isinstance(disu, flopy.modflow.ModflowDisU)

    # Change where model files are written
    model_ws = tpth
    m.model_ws = model_ws

    # Write the disu file
    disu.write_file()
    assert os.path.isfile(os.path.join(model_ws,
                                       '{}.{}'.format(m.name,
                                                      m.disu.extension[0])))

    # Load disu file
    disu2 = flopy.modflow.ModflowDisU.load(fname, m)
    for (key1, value1), (key2, value2) in zip(disu2.__dict__.items(),
                                              disu.__dict__.items()):
        if isinstance(value1, flopy.utils.Util2d) or isinstance(value1, flopy.utils.Util3d):
            assert np.array_equal(value1.array, value2.array)
        else:
            assert value1 == value2

    return


def test_usg_sms_load():

    pthusgtest = os.path.join('..', 'examples', 'data', 'mfusg_test',
                              '01A_nestedgrid_nognc')
    fname = os.path.join(pthusgtest, 'flow.sms')
    assert os.path.isfile(fname), 'sms file not found {}'.format(fname)

    # Create the model
    m = flopy.modflow.Modflow(modelname='usgload', verbose=True)

    # Load the sms file
    sms = flopy.modflow.ModflowSms.load(fname, m)
    assert isinstance(sms, flopy.modflow.ModflowSms)

    # Change where model files are written
    model_ws = tpth
    m.model_ws = model_ws

    # Write the sms file
    sms.write_file()
    assert os.path.isfile(os.path.join(model_ws,
                                       '{}.{}'.format(m.name,
                                                      m.sms.extension[0])))

    # Load sms file
    sms2 = flopy.modflow.ModflowSms.load(fname, m)
    for (key1, value1), (key2, value2) in zip(sms2.__dict__.items(),
                                              sms.__dict__.items()):
        assert value1 == value2, 'key1 {}, value 1 {} != key2 {} value 2 {}'.format(key1, value1, key2, value2)

    return


def test_usg_model():
    mf = flopy.modflow.Modflow(version='mfusg', structured=True,
                               model_ws=tpth, modelname='simple',
                               exe_name=v)
    dis = flopy.modflow.ModflowDis(mf, nlay=1, nrow=11, ncol=11)
    bas = flopy.modflow.ModflowBas(mf)
    lpf = flopy.modflow.ModflowLpf(mf)
    wel = flopy.modflow.ModflowWel(mf, stress_period_data={0: [[0, 5, 5, -1.]]})
    ghb = flopy.modflow.ModflowGhb(mf,
                                   stress_period_data={
                                       0: [[0, 0, 0, 1.0, 1000.],
                                           [0, 9, 9, 0.0, 1000.], ]})
    oc = flopy.modflow.ModflowOc(mf)
    sms = flopy.modflow.ModflowSms(mf, options='complex')

    # run with defaults
    mf.write_input()
    if run:
        success, buff = mf.run_model()
        assert success

    # try different complexity options; all should run successfully
    for complexity in ['simple', 'moderate', 'complex']:
        print('testing MFUSG with sms complexity: ' + complexity)
        sms = flopy.modflow.ModflowSms(mf, options=complexity)
        sms.write_file()
        if run:
            success, buff = mf.run_model()
            assert success


if __name__ == '__main__':
    test_usg_disu_load()
    test_usg_sms_load()
    test_usg_model()
