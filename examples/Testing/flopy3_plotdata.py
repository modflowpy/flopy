from __future__ import print_function
import os

import numpy as np
import matplotlib.pyplot as plt

import flopy


fb = flopy.modflow.Modflow.load('freyberg', version='mf2005', model_ws=os.path.join('..', 'data', 'freyberg'), verbose=True)

dis = fb.dis

top = fb.dis.top

fb.dis.top.plot(grid=True, colorbar=True)
fb.dis.botm.plot(grid=True, colorbar=True)
#plt.show()


fb.dis.plot()
plt.show()

fb.dis.plot()
plt.show()

print('this is the end my friend')