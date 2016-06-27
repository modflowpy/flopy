import flopy
import numpy as np


ml = flopy.modflow.Modflow.load("fas.nam",check=False,verbose=True)
#ml.change_model_ws("temp2")
arr = np.ones((ml.nlay,ml.nrow,ml.ncol))

ml.bas6.strt = arr