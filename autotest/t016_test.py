import os
import flopy
import numpy as np

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
    model_ws = 'temp'
    m.change_model_ws(model_ws)

    # Write the disu file
    disu.write_file()
    assert os.path.isfile(os.path.join(model_ws,
                                       '{}.{}'.format(m.name,
                                                      m.disu.extension[0])))

    # Load disu file
    disu2 = flopy.modflow.ModflowDisU.load(fname, m)
    for (key1, value1), (key2, value2) in zip(disu2.__dict__.items(),
                                              disu.__dict__.items()):
        if isinstance(value1, flopy.utils.util_2d) or isinstance(value1, flopy.utils.util_3d):
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
    model_ws = 'temp'
    m.change_model_ws(model_ws)

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

if __name__ == '__main__':
    test_usg_disu_load()
    test_usg_sms_load()
