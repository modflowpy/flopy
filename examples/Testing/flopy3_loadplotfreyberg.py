import matplotlib.pyplot as plt
import flopy

ml = flopy.modflow.Modflow.load('freyberg.nam', version='mf2005', verbose=True, model_ws='data')

ml.plot()
plt.show()

print('then end my friend')