import sys
sys.path.insert(0, '..')
import os
import glob
import flopy

def test_checker_on_load():
    # load all of the models in the mf2005_test folder
    # model level checks are performed by default on load()
    model_ws = '../examples/data/mf2005_test/'
    testmodels = [os.path.split(f)[-1] for f in glob.glob(model_ws + '*.nam')]

    for mfnam in testmodels:
        m = flopy.modflow.Modflow.load(mfnam, model_ws=model_ws)

if __name__ == '__main__':
    test_checker_on_load()