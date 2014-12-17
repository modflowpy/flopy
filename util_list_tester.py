import os
import sys
import warnings
import numpy as np
import flopy
from flopy.utils import mflist

#--instance testing
#override the default data type
dtype = np.dtype([("k",int),("i",np.int),("j",np.int),\
                  ("stage",np.float32),("cond",np.float32),("rbot",np.float32),\
    ("aux1",np.float32),("comments",object)])

#a dict to store data
data = {}

#sp 2 = -1
data[1] = -1


#build a nested list for sp 4
d = []
for k in range(0,2):
    for i in range(1,3):
        for j in range(4,6):
            dd = np.array([k,i,j,10,1,2,999,"# this is 1"])
            d.append(dd)
data[3] = d

#a single entry for sp 5
data[4] = np.array((9,8,7,6,5,4,999,'1'))


#an external file for sp 10
f = open("some_list.dat",'w')
f.write(' 0  0  0  1.0  1.0  1.0 999 2 \n')
f.write(' 0  0  0  1.0  1.0  1.0 999 2 \n')
f.write(' 0  0  0  1.0  1.0  1.0 999 2 \n')
f.close()
data[9] = "some_list.dat"


model = flopy.modflow.Modflow()
dis = flopy.modflow.ModflowDis(model,nlay=1,nrow=1,ncol=1,nper=3)
riv = flopy.modflow.ModflowRiv(model,layer_row_column_data=data,dtype=dtype)

#get the recarray from sp 10
sp10_rec = riv[9]
#check bounds
riv.list_data.check_kij()
#write file
riv.write_file()











