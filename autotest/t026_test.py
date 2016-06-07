"""
test MNW2 package
"""
import os
import flopy
import numpy as np

path = os.path.join('..', 'examples', 'data', 'mf2005_test')
cpth = os.path.join('temp')
m = flopy.modflow.Modflow('MNW2-Fig28', model_ws=cpth)
dis = flopy.modflow.ModflowDis.load(path + '/MNW2-Fig28.dis', m)
mnw2 = flopy.modflow.ModflowMnw2.load(path + '/MNW2-Fig28.mnw2', m)