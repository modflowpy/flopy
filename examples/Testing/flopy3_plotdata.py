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

fb.dis.plot()
plt.show()

fb.dis.plot()
plt.show()


fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1,2,1, aspect='equal')
fb.dis.top.plot(grid=True, axes=ax,  colorbar=True)
ax = fig.add_subplot(1,2,2, aspect='equal')
fb.dis.botm.plot(grid=True, axes=ax, colorbar=True)
plt.show()

print('this is the end my friend')