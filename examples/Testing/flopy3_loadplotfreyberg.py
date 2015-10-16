import os
import matplotlib.pyplot as plt
import flopy

ml = flopy.modflow.Modflow.load('freyberg.nam', version='mf2005', verbose=True, model_ws='data')

ml.check()

fb = os.path.join('data', 'ml')
ml.plot(filename_base=fb)

#ml.plot()
#plt.show()

binobj = ml.load_results()

print('then end my friend')