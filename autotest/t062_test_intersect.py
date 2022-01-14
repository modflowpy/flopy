import os
import flopy
import numpy as np
import matplotlib.pyplot as plt

# grid properties
nlay = 3
nrow = 21
ncol = 20
delr = 500.0
delc = 500.0
top = 400.0
botm = [220.0, 200.0, 0.0]
xorigin = 3000
yorigin = 1000
angrot = 10


def dis_model():
    sim = flopy.mf6.MFSimulation()
    gwf = flopy.mf6.ModflowGwf(sim)

    # dis
    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
        xorigin=xorigin,
        yorigin=yorigin,
        angrot=angrot,
    )
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
    # disv
    flopy.mf6.ModflowGwfdisv(
        gwf,
        nlay=nlay,
        ncpl=ncpl,
        top=top,
        botm=botm,
        nvert=nvert,
        vertices=vertices,
        cell2d=cell2d,
        xorigin=xorigin,
        yorigin=yorigin,
        angrot=angrot,
    )
    gwf.modelgrid.set_coord_info(xoff=xorigin, yoff=yorigin, angrot=angrot)
    return gwf


# simple functions to load vertices and index lists for unstructured grid
def load_verts(fname):
    verts = np.genfromtxt(fname, dtype=[int, float, float],
                          names=['iv', 'x', 'y'])
    verts['iv'] -= 1  # zero based
    return verts


def load_iverts(fname):
    f = open(fname, 'r')
    iverts = []
    xc = []
    yc = []
    for line in f:
        ll = line.strip().split()
        iverts.append([int(i) - 1 for i in ll[4:]])
        xc.append(float(ll[1]))
        yc.append(float(ll[2]))
    return iverts, np.array(xc), np.array(yc)


def test_intersection():
    ml_dis = dis_model()
    ml_disv = disv_model()

    if False:
        plt.subplots()
        ml_dis.modelgrid.plot()
        plt.subplots()
        ml_disv.modelgrid.plot()

    for i in range(5):
        if i == 0:
            # inside a cell, in real-world coordinates
            x = 4000
            y = 4000
            local = False
            forgive = False
        elif i == 1:
            # on the cell-edge, in local coordinates
            x = 4000
            y = 4000
            local = True
            forgive = False
        elif i == 2:
            # inside a cell, in local coordinates
            x = 4001
            y = 4001
            local = True
            forgive = False
        elif i == 3:
            # inside a cell, in local coordinates
            x = 4001
            y = 4001
            local = False
            forgive = False
        elif i == 4:
            # inside a cell, in local coordinates
            x = 999
            y = 4001
            local = False
            forgive = True
        if local:
            print("In local coordinates:")
        else:
            print("In real_world coordinates:")
        try:
            row, col = ml_dis.modelgrid.intersect(
                x, y, local=local, forgive=forgive
            )
            cell2d_disv = ml_disv.modelgrid.intersect(
                x, y, local=local, forgive=forgive
            )
        except Exception as e:
            if not forgive and any(
                ["outside of the model area" in k for k in e.args]
            ):
                pass
            else:  # should be forgiving x,y out of grid
                raise e
        print(f"x={x},y={y} in dis  is in row {row} and col {col}, so...")
        cell2d_dis = row * ml_dis.modelgrid.ncol + col
        print(f"x={x},y={y} in dis  is in cell2d-number {cell2d_dis}")
        print(f"x={x},y={y} in disv is in cell2d-number {cell2d_disv}")

        if not forgive:
            assert cell2d_dis == cell2d_disv
        else:
            assert all(np.isnan([row, col, cell2d_disv]))


def test_structured_xyz_intersect():
    model_ws = os.path.join(
        "..", "examples", "data", "freyberg_multilayer_transient"
    )
    fname = "freyberg.nam"

    ml = flopy.modflow.Modflow.load(fname, model_ws=model_ws)
    mg = ml.modelgrid
    top_botm = ml.modelgrid.top_botm
    xc, yc, zc = mg.xyzcellcenters

    for _ in range(10):
        k = np.random.randint(0, mg.nlay, 1)[0]
        i = np.random.randint(0, mg.nrow, 1)[0]
        j = np.random.randint(0, mg.ncol, 1)[0]
        x = xc[i, j]
        y = yc[i, j]
        z = zc[k, i, j]
        k2, i2, j2 = ml.modelgrid.intersect(x, y, z)
        if (k, i, j) != (k2, i2, j2):
            raise AssertionError("Structured grid intersection failed")


def test_vertex_xyz_intersect():
    sim_ws = os.path.join("..", "examples", "data", "mf6", "test003_gwfs_disv")

    sim = flopy.mf6.MFSimulation.load(sim_ws=sim_ws)
    ml = sim.get_model(list(sim.model_names)[0])
    mg = ml.modelgrid

    xc, yc, zc = mg.xyzcellcenters
    for _ in range(10):
        icell = np.random.randint(0, mg.ncpl, 1)[0]
        lay = np.random.randint(0, mg.nlay, 1)[0]
        x = xc[icell]
        y = yc[icell]
        z = zc[lay, icell]
        lay1, icell1 = mg.intersect(x, y, z)

        if (lay, icell) != (lay1, icell1):
            raise AssertionError("Vertex grid intersection failed")


def test_unstructured_xyz_intersect():
    ws = os.path.join("..", "examples", "data", "unstructured")
    # usg example
    name = os.path.join(ws, "ugrid_verts.dat")
    verts = load_verts(name)

    name = os.path.join(ws, "ugrid_iverts.dat")
    iverts, xc, yc = load_iverts(name)

    # create a 3 layer model grid
    ncpl = np.array(3 * [len(iverts)])
    nnodes = np.sum(ncpl)

    top = np.ones((nnodes), )
    botm = np.ones((nnodes), )

    # set top and botm elevations
    i0 = 0
    i1 = ncpl[0]
    elevs = [100, 0, -100, -200]
    for ix, cpl in enumerate(ncpl):
        top[i0:i1] *= elevs[ix]
        botm[i0:i1] *= elevs[ix + 1]
        i0 += cpl
        i1 += cpl

    # create the modelgrid
    mg = flopy.discretization.UnstructuredGrid(vertices=verts,
                                               iverts=iverts,
                                               xcenters=xc,
                                               ycenters=yc, top=top,
                                               botm=botm, ncpl=ncpl)

    xc, yc, zc = mg.xyzcellcenters
    zc = zc[0].reshape(mg.nlay, mg.ncpl[0])
    for _ in range(10):
        icell = np.random.randint(0, mg.ncpl[0], 1)[0]
        lay = np.random.randint(0, mg.nlay, 1)[0]
        x = xc[icell]
        y = yc[icell]
        z = zc[lay, icell]
        icell1 = mg.intersect(x, y, z)
        icell = icell + (mg.ncpl[0] * lay)
        if icell != icell1:
            raise AssertionError("Unstructured grid intersection failed")


if __name__ == "__main__":
    test_intersection()
    test_structured_xyz_intersect()
    test_vertex_xyz_intersect()
    test_unstructured_xyz_intersect()
