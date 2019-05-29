import flopy
import numpy as np
import matplotlib.pyplot as plt

# grid properties
nlay = 3
nrow = 21
ncol = 20
delr = 500.
delc = 500.
top = 400.
botm = [220., 200., 0.]
xorigin=3000
yorigin=1000
angrot=10


def dis_model():
    sim = flopy.mf6.MFSimulation()
    gwf = flopy.mf6.ModflowGwf(sim)

    # dis
    flopy.mf6.ModflowGwfdis(gwf, nlay=nlay, nrow=nrow, ncol=ncol,
                            delr=delr, delc=delc,
                            top=top, botm=botm,
                            xorigin=xorigin,yorigin=yorigin,angrot=angrot)
    return gwf

def disv_model():
    sim = flopy.mf6.MFSimulation()
    gwf = flopy.mf6.ModflowGwf(sim)

    ncpl = nrow * ncol
    xv = np.linspace(0, delr * ncol, ncol + 1)
    yv = np.linspace(delc * nrow, 0, nrow + 1)
    xv, yv = np.meshgrid(xv, yv)
    xv = xv.ravel()
    yv = yv.ravel()

    def get_vlist(i, j, nrow, ncol):
        v1 = i * (ncol + 1) + j
        v2 = v1 + 1
        v3 = v2 + ncol + 1
        v4 = v3 - 1
        return [v1, v2, v3, v4]

    iverts = []
    for i in range(nrow):
        for j in range(ncol):
            iverts.append(get_vlist(i, j, nrow, ncol))

    nvert = xv.shape[0]
    verts = np.hstack((xv.reshape(nvert, 1), yv.reshape(nvert, 1)))

    cellxy = np.empty((nvert, 2))
    for icpl in range(ncpl):
        iv = iverts[icpl]
        cellxy[icpl, 0] = (xv[iv[0]] + xv[iv[1]]) / 2.
        cellxy[icpl, 1] = (yv[iv[1]] + yv[iv[2]]) / 2.

    # need to create cell2d, which is [[icpl, xc, yc, nv, iv1, iv2, iv3, iv4]]
    cell2d = [[icpl, cellxy[icpl, 0], cellxy[icpl, 1], 4]  + iverts[icpl] for
              icpl in range(ncpl)]
    vertices = [[ivert, verts[ivert, 0], verts[ivert, 1]] for ivert in
                range(nvert)]
    # disv
    flopy.mf6.ModflowGwfdisv(gwf, nlay=nlay, ncpl=ncpl,
                             top=top, botm=botm,
                             nvert=nvert, vertices=vertices,
                             cell2d=cell2d,
                             xorigin=xorigin,yorigin=yorigin,angrot=angrot)
    gwf.modelgrid.set_coord_info(xoff=xorigin,yoff=yorigin,angrot=angrot)
    return gwf


def test_intersection():
    ml_dis = dis_model()
    ml_disv = disv_model()

    if False:
        plt.subplots()
        ml_dis.modelgrid.plot()
        plt.subplots()
        ml_disv.modelgrid.plot()

    for i in range(5):
        if i==0:
            # inside a cell, in real-world coordinates
            x = 4000
            y = 4000
            local = False
            forgive = False
        elif i==1:
            # on the cell-edge, in local coordinates
            x = 4000
            y = 4000
            local = True
            forgive = False
        elif i==2:
            # inside a cell, in local coordinates
            x = 4001
            y = 4001
            local = True
            forgive = False
        elif i==3:
            # inside a cell, in local coordinates
            x = 4001
            y = 4001
            local = False
            forgive = False
        elif i==4:
            # inside a cell, in local coordinates
            x = 999
            y = 4001
            local = False
            forgive = True
        if local:
            print('In local coordinates:')
        else:
            print('In real_world coordinates:')
        try:
            row, col = ml_dis.modelgrid.intersect(x, y, local, forgive=forgive)
            cell2d_disv = ml_disv.modelgrid.intersect(x, y, local,
                                                      forgive=forgive)
        except Exception as e:
            if not forgive and any(['outside of the model area' 
                                    in k for k in e.args]):
                pass
            else:  # should be forgiving x,y out of grid
                raise e
        print('x={},y={} in dis  is in row {} and col {}, so...'.format(
            x, y, row, col))
        cell2d_dis = row * ml_dis.modelgrid.ncol + col
        print('x={},y={} in dis  is in cell2d-number {}'.format(
            x, y, cell2d_dis))
        print('x={},y={} in disv is in cell2d-number {}'.format(
            x, y, cell2d_disv))

        if not forgive:
            assert cell2d_dis == cell2d_disv
        else: 
            assert all(np.isnan([row, col, cell2d_disv]))


if __name__ == '__main__':
    test_intersection()