import numpy as np

from flopy.mf6 import MFSimulation, ModflowGwf, ModflowGwfdis, ModflowGwfdisv


def case_dis():
    sim = MFSimulation()
    gwf = ModflowGwf(sim)
    dis = ModflowGwfdis(
        gwf,
        nlay=3,
        nrow=21,
        ncol=20,
        delr=500.0,
        delc=500.0,
        top=400.0,
        botm=[220.0, 200.0, 0.0],
        xorigin=3000,
        yorigin=1000,
        angrot=10,
    )

    return gwf


def case_disv():
    sim = MFSimulation()
    gwf = ModflowGwf(sim)

    nrow, ncol = 21, 20
    delr, delc = 500.0, 500.0
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
        cellxy[icpl, 0] = (xv[iv[0]] + xv[iv[1]]) / 2.0
        cellxy[icpl, 1] = (yv[iv[1]] + yv[iv[2]]) / 2.0

    # need to create cell2d, which is [[icpl, xc, yc, nv, iv1, iv2, iv3, iv4]]
    cell2d = [
        [icpl, cellxy[icpl, 0], cellxy[icpl, 1], 4] + iverts[icpl]
        for icpl in range(ncpl)
    ]
    vertices = [
        [ivert, verts[ivert, 0], verts[ivert, 1]] for ivert in range(nvert)
    ]
    xorigin = 3000
    yorigin = 1000
    angrot = 10
    ModflowGwfdisv(
        gwf,
        nlay=3,
        ncpl=ncpl,
        top=400.0,
        botm=[220.0, 200.0, 0.0],
        nvert=nvert,
        vertices=vertices,
        cell2d=cell2d,
        xorigin=xorigin,
        yorigin=yorigin,
        angrot=angrot,
    )
    gwf.modelgrid.set_coord_info(xoff=xorigin, yoff=yorigin, angrot=angrot)
    return gwf


def case_disu():
    pass
