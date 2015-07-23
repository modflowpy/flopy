import sys
import os
import platform
import numpy as np
import matplotlib.pyplot as plt

import flopy

#Set name of MODFLOW exe
#  assumes executable is in users path statement
version = 'mf2005'
exe_name = 'mf2005'
if platform.system() == 'Windows':
    exe_name = 'mf2005.exe'
mfexe = exe_name

#Set the paths
loadpth = os.path.join('..', 'data', 'secp')
modelpth = os.path.join('data')

#make sure modelpth directory exists
if not os.path.exists(modelpth):
    os.makedirs(modelpth)

ml = flopy.modflow.Modflow.load('secp.nam', model_ws=loadpth, 
                                exe_name=exe_name, version=version, verbose=True)
ml.change_model_ws(new_pth=modelpth)
ml.write_input()

print '...end'
