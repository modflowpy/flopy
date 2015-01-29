import sys
import os
import platform
import numpy as np

# -- add development version of flopy to system path
flopypath = os.path.join('..', '..')
if flopypath not in sys.path:
    print 'Adding to sys.path: ', flopypath
    sys.path.append(flopypath)

import flopy
import flopy.utils as fputl

mname = 'twrip.nam'
#mname = 'Oahu_01.nam'

model_ws = os.path.join('..', 'data', 'parameters')
omodel_ws = os.path.join('..', 'basic', 'data')

exe_name = 'mf2005'
version = 'mf2005'

# -- load the model
ml = flopy.modflow.Modflow.load(mname, version=version, exe_name=exe_name, 
                                verbose=False, model_ws=model_ws)

# -- change model workspace
ml.change_model_ws(new_pth=omodel_ws)

# -- add pcg package
if mname == 'twrip.nam':
    ml.remove_package('SIP')
    pcg = flopy.modflow.ModflowPcg(ml)

# -- save the model
ml.write_input()

print 'finished...'