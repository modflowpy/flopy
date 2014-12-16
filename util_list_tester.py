import os
import sys
import warnings
import numpy as np
import flopy
from flopy.utils import mflist

#--instance testing
#dtype = np.dtype([("layer",np.int),("row",np.int),("column",np.int),("stage",np.float32),("cond",np.float32),("rbot",np.float32)])
data = {}
data[0] = 0
data[1] = -1
d = []
for k in range(0,2):
    for i in range(1,3):
        for j in range(4,6):
            dd = np.array([k,i,j,10,1,2])
            d.append(dd)
#d1 = np.core.records.fromarrays(np.array(d).transpose(),dtype=dtype)
data[3] = d
data[4] = np.array((9,8,7,6,5,4))
f = open("some_list.dat",'w')
f.write(' 0  0  0  1.0  1.0  1.0\n')
f.write(' 0  0  0  1.0  1.0  1.0\n')
f.write(' 0  0  0  1.0  1.0  1.0\n')
f.close()
data[9] = "some_list.dat"

dtype = flopy.modflow.ModflowRiv.get_default_dtype()

model = flopy.modflow.Modflow()
dis = flopy.modflow.ModflowDis(model,nlay=1,nrow=10,ncol=10,nper=3)
riv = flopy.modflow.ModflowRiv(model,layer_row_column_data=data)
#riv.add_field_to_dtype("aux1",np.float32)
riv.list_data.check_kij()
riv.write_file()











