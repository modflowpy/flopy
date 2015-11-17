import os
import flopy

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
    assert os.path.isfile(os.path.join(model_ws, '{}.{}'.format(m.name, m.disu.extension[0]))) is True

    # Load disu file
    disu2 = flopy.modflow.ModflowDisU.load(fname, m)
    for key in disu2.__dict__.keys():
        assert disu2[key] == disu[key]

    return

if __name__ == '__main__':
    test_usg_disu_load()
