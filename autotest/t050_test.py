import shutil
import os
import numpy as np
import flopy
from flopy.export.vtk import Vtk

# create output directory
cpth = os.path.join('temp', 't050')
if os.path.isdir(cpth):
    shutil.rmtree(cpth)
os.makedirs(cpth)


def test_vtkoutput():
    nlay = 3
    nrow = 3
    ncol = 3
    ml = flopy.modflow.Modflow()
    dis = flopy.modflow.ModflowDis(ml, nlay=nlay, nrow=nrow, ncol=ncol, top=0,
                                   botm=[-1., -2., -3.])
    ibound = np.ones((nlay, nrow, ncol), dtype=np.int)
    ibound[0, 1, 1] = 0
    bas = flopy.modflow.ModflowBas(ml, ibound=ibound)

    fvtkout = os.path.join(cpth, 'test.vtu')
    vtkfile = Vtk(fvtkout, ml)
    a = np.arange(nlay * nrow * ncol).reshape((nlay, nrow, ncol))
    vtkfile.add_array('testarray', a)
    vtkfile.write(shared_vertex=False, ibound_filter=True)
    return


if __name__ == '__main__':
    test_vtkoutput()
