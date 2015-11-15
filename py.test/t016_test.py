import os
import flopy

def test_usgload():

    pthusgtest = os.path.join('..', 'examples', 'data', 'mfusg_test',
                              '01A_nestedgrid_nognc')
    fname = os.path.join(pthusgtest, 'flow.disu')
    assert os.path.isfile(fname), 'disu file not found {}'.format(fname)

    # Create the model
    m = flopy.modflow.Modflow(modelname='usgload', verbose=True)

    # Load the disu file
    disu = flopy.modflow.ModflowDisU.load(fname, m)

    # Change where model files are written
    model_ws = 'temp'
    m.change_model_ws(model_ws)

    # Write the disu file
    disu.write_file()

    return

if __name__ == '__main__':
    test_usgload()
