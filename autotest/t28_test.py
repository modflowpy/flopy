"""
test UZF package
"""
import os
import flopy
import numpy as np

cpth = os.path.join('temp')

def test_load():

    # load in the test problem
    m = flopy.modflow.Modflow('UZFtest2', model_ws=cpth)
    path = os.path.join('..', 'examples', 'data', 'mf2005_test')
    dis = flopy.modflow.ModflowDis.load(path + '/UZFtest2.dis', m)
    uzf = flopy.modflow.ModflowUzf1.load(path + '/UZFtest2.mnw2', m)

if __name__ == '__main__':
    test_load()
    #test_make_package()