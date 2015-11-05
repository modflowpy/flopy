import os
import numpy as np
import flopy

pth = os.path.join('..', 'data', 'mf2005_test')
opth = os.path.join('data')
mname = 'swtex4'

ml = flopy.modflow.Modflow.load(mname, version='mf2005', model_ws=pth, verbose=True)

ml.change_model_ws(opth)

ml.write_input()

ml.set_exename('mf2005dbl')

ml.run_model()

print('finished...')