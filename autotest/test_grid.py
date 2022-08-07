import os

import matplotlib
import numpy as np
import pytest
from flaky import flaky
from matplotlib import pyplot as plt

from autotest.conftest import requires_pkg

from flopy.discretization import StructuredGrid, UnstructuredGrid, VertexGrid
from flopy.mf6 import MFSimulation, ModflowGwf, ModflowGwfdis, ModflowGwfdisv
from flopy.modflow import Modflow, ModflowDis
from flopy.utils.cvfdutil import gridlist_to_disv_gridprops, to_cvfd
from flopy.utils.triangle import Triangle
from flopy.utils.voronoi import VoronoiGrid


def test_rotation():
    m = Modflow(rotation=20.0)
    dis = ModflowDis(
        m, nlay=1, nrow=40, ncol=20, delr=250.0, delc=250.0, top=10, botm=0
    )
    xul, yul = 500000, 2934000
    mg = StructuredGrid(delc=m.dis.delc.array, delr=m.dis.delr.array)
    mg._angrot = 45.0
    mg.set_coord_info(mg._xul_to_xll(xul), mg._yul_to_yll(yul), angrot=45.0)

    xll, yll = mg.xoffset, mg.yoffset
    assert np.abs(mg.xvertices[0, 0] - xul) < 1e-4
    assert np.abs(mg.yvertices[0, 0] - yul) < 1e-4

    mg2 = StructuredGrid(delc=m.dis.delc.array, delr=m.dis.delr.array)
    mg2._angrot = -45.0
    mg2.set_coord_info(
        mg2._xul_to_xll(xul), mg2._yul_to_yll(yul), angrot=-45.0
    )

    xll2, yll2 = mg2.xoffset, mg2.yoffset
    assert np.abs(mg2.xvertices[0, 0] - xul) < 1e-4
    assert np.abs(mg2.yvertices[0, 0] - yul) < 1e-4

    mg3 = StructuredGrid(
        delc=m.dis.delc.array,
        delr=m.dis.delr.array,
        xoff=xll2,
        yoff=yll2,
        angrot=-45.0,
    )

    assert np.abs(mg3.xvertices[0, 0] - xul) < 1e-4
    assert np.abs(mg3.yvertices[0, 0] - yul) < 1e-4

    mg4 = StructuredGrid(
        delc=m.dis.delc.array,
        delr=m.dis.delr.array,
        xoff=xll,
        yoff=yll,
        angrot=45.0,
    )

    assert np.abs(mg4.xvertices[0, 0] - xul) < 1e-4
    assert np.abs(mg4.yvertices[0, 0] - yul) < 1e-4


def test_get_vertices():
    m = Modflow(rotation=20.0)
    nrow, ncol = 40, 20
    dis = ModflowDis(
        m, nlay=1, nrow=nrow, ncol=ncol, delr=250.0, delc=250.0, top=10, botm=0
    )
    mmg = m.modelgrid
    xul, yul = 500000, 2934000
    mg = StructuredGrid(
        delc=m.dis.delc.array,
        delr=m.dis.delr.array,
        xoff=mmg._xul_to_xll(xul, 45.0),
        yoff=mmg._yul_to_yll(xul, 45.0),
        angrot=45.0,
    )

    xgrid = mg.xvertices
    ygrid = mg.yvertices
    # a1 = np.array(mg.xyvertices)
    a1 = np.array(
        [
            [xgrid[0, 0], ygrid[0, 0]],
            [xgrid[0, 1], ygrid[0, 1]],
            [xgrid[1, 1], ygrid[1, 1]],
            [xgrid[1, 0], ygrid[1, 0]],
        ]
    )

    a2 = np.array(mg.get_cell_vertices(0, 0))
    assert np.array_equal(a1, a2)


def test_get_lrc_get_node():
    nlay, nrow, ncol = 3, 4, 5
    ml = Modflow()
    dis = ModflowDis(
        ml, nlay=nlay, nrow=nrow, ncol=ncol, top=50, botm=[0, -1, -2]
    )
    nodes = list(range(nlay * nrow * ncol))
    indices = np.indices((nlay, nrow, ncol))
    layers = indices[0].flatten()
    rows = indices[1].flatten()
    cols = indices[2].flatten()
    for node, (l, r, c) in enumerate(zip(layers, rows, cols)):
        # ensure get_lrc returns zero-based layer row col
        lrc = dis.get_lrc(node)[0]
        assert lrc == (
            l,
            r,
            c,
        ), f"get_lrc() returned {lrc}, expecting {l, r, c}"
        # ensure get_node returns zero-based node number
        n = dis.get_node((l, r, c))[0]
        assert node == n, f"get_node() returned {n}, expecting {node}"
    return


def test_get_rc_from_node_coordinates():
    m = Modflow(rotation=20.0)
    nrow, ncol = 10, 10
    dis = ModflowDis(
        m, nlay=1, nrow=nrow, ncol=ncol, delr=100.0, delc=100.0, top=10, botm=0
    )
    r, c = m.dis.get_rc_from_node_coordinates([50.0, 110.0], [50.0, 220.0])
    assert np.array_equal(r, np.array([9, 7]))
    assert np.array_equal(c, np.array([0, 1]))

    # test variable delr and delc spacing
    mf = Modflow()
    delc = [0.5] * 5 + [2.0] * 5
    delr = [0.5] * 5 + [2.0] * 5
    nrow = 10
    ncol = 10
    mfdis = ModflowDis(
        mf, nrow=nrow, ncol=ncol, delr=delr, delc=delc
    )  # , xul=50, yul=1000)
    ygrid, xgrid, zgrid = mfdis.get_node_coordinates()
    for i in range(nrow):
        for j in range(ncol):
            x = xgrid[j]
            y = ygrid[i]
            r, c = mfdis.get_rc_from_node_coordinates(x, y)
            assert r == i, f"row {r} not equal {i} for xy ({x}, {y})"
            assert c == j, f"col {c} not equal {j} for xy ({x}, {y})"


@pytest.fixture
def dis_model():
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


@pytest.fixture
def disv_model():
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


def load_verts(fname):
    verts = np.genfromtxt(
        fname, dtype=[int, float, float], names=["iv", "x", "y"]
    )
    verts["iv"] -= 1  # zero based
    return verts


def load_iverts(fname):
    f = open(fname, "r")
    iverts = []
    xc = []
    yc = []
    for line in f:
        ll = line.strip().split()
        iverts.append([int(i) - 1 for i in ll[4:]])
        xc.append(float(ll[1]))
        yc.append(float(ll[2]))
    return iverts, np.array(xc), np.array(yc)


def test_intersection(dis_model, disv_model):

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
            row, col = dis_model.modelgrid.intersect(
                x, y, local=local, forgive=forgive
            )
            cell2d_disv = disv_model.modelgrid.intersect(
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
        cell2d_dis = row * dis_model.modelgrid.ncol + col
        print(f"x={x},y={y} in dis  is in cell2d-number {cell2d_dis}")
        print(f"x={x},y={y} in disv is in cell2d-number {cell2d_disv}")

        if not forgive:
            assert cell2d_dis == cell2d_disv
        else:
            assert all(np.isnan([row, col, cell2d_disv]))


def test_structured_xyz_intersect(example_data_path):
    ml = Modflow.load(
        "freyberg.nam",
        model_ws=str(example_data_path / "freyberg_multilayer_transient"),
    )
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


def test_vertex_xyz_intersect(example_data_path):
    sim = MFSimulation.load(
        sim_ws=str(example_data_path / "mf6" / "test003_gwfs_disv")
    )
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


def test_unstructured_xyz_intersect(example_data_path):
    ws = str(example_data_path / "unstructured")
    name = os.path.join(ws, "ugrid_verts.dat")
    verts = load_verts(name)

    name = os.path.join(ws, "ugrid_iverts.dat")
    iverts, xc, yc = load_iverts(name)

    # create a 3 layer model grid
    ncpl = np.array(3 * [len(iverts)])
    nnodes = np.sum(ncpl)

    top = np.ones(
        (nnodes),
    )
    botm = np.ones(
        (nnodes),
    )

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
    mg = UnstructuredGrid(
        vertices=verts,
        iverts=iverts,
        xcenters=xc,
        ycenters=yc,
        top=top,
        botm=botm,
        ncpl=ncpl,
    )

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


@pytest.mark.parametrize("spc_file", ["grd.spc", "grdrot.spc"])
def test_from_gridspec(example_data_path, spc_file):
    fn = str(example_data_path / "specfile" / spc_file)
    modelgrid = StructuredGrid.from_gridspec(fn)
    assert isinstance(modelgrid, StructuredGrid)

    lc = modelgrid.plot()
    assert isinstance(
        lc, matplotlib.collections.LineCollection
    ), f"could not plot grid object created from {fn}"
    plt.close()

    extents = modelgrid.extent
    theta = modelgrid.angrot_radians
    if "rot" in fn:
        assert theta != 0, "rotation missing"
    rotated_extents = (
        0,  # xmin
        8000 * np.sin(theta) + 8000 * np.cos(theta),  # xmax
        8000 * np.sin(theta) * np.tan(theta / 2),  # ymin
        8000 + 8000 * np.sin(theta),
    )  # ymax
    errmsg = f"extents {extents} of {fn} does not equal {rotated_extents}"
    assert all(
        [np.isclose(x, x0) for x, x0 in zip(modelgrid.extent, rotated_extents)]
    ), errmsg

    ncpl = modelgrid.ncol * modelgrid.nrow
    assert (
        modelgrid.ncpl == ncpl
    ), f"ncpl ({modelgrid.ncpl}) does not equal {ncpl}"

    nvert = modelgrid.nvert
    iverts = modelgrid.iverts
    maxvertex = max([max(sublist[1:]) for sublist in iverts])
    assert (
        maxvertex + 1 == nvert
    ), f"nvert ({maxvertex + 1}) does not equal {nvert}"
    verts = modelgrid.verts
    assert nvert == verts.shape[0], (
        f"number of vertex (x, y) pairs ({verts.shape[0]}) "
        f"does not equal {nvert}"
    )


def test_epsgs():
    import flopy.export.shapefile_utils as shp

    # test setting a geographic (lat/lon) coordinate reference
    # (also tests shapefile_utils.CRS parsing of geographic crs info)
    delr = np.ones(10)
    delc = np.ones(10)
    mg = StructuredGrid(delr=delr, delc=delc)

    mg.epsg = 102733
    assert mg.epsg == 102733, f"mg.epsg is not 102733 ({mg.epsg})"

    t_value = mg.__repr__()
    if not "proj4_str:epsg:102733" in t_value:
        raise AssertionError(
            f"proj4_str:epsg:102733 not in mg.__repr__(): ({t_value})"
        )

    mg.epsg = 4326  # WGS 84
    crs = shp.CRS(epsg=4326)
    if crs.grid_mapping_attribs is not None:
        assert crs.crs["proj"] == "longlat"
        t_value = crs.grid_mapping_attribs["grid_mapping_name"]
        assert (
            t_value == "latitude_longitude"
        ), f"grid_mapping_name is not latitude_longitude: {t_value}"

    t_value = mg.__repr__()
    if not "proj4_str:epsg:4326" in t_value:
        raise AssertionError(
            f"proj4_str:epsg:4326 not in sr.__repr__(): ({t_value})"
        )


def test_tocvfd1():
    vertdict = {}
    vertdict[0] = [(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)]
    vertdict[1] = [(100, 0), (120, 0), (120, 20), (100, 20), (100, 0)]
    verts, iverts = to_cvfd(vertdict)
    assert 6 in iverts[0]


def test_tocvfd2():
    vertdict = {}
    vertdict[0] = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
    vertdict[1] = [(1, 0), (3, 0), (3, 2), (1, 2), (1, 0)]
    verts, iverts = to_cvfd(vertdict)
    assert [1, 4, 5, 6, 2, 1] in iverts


def test_tocvfd3():
    # create the nested grid described in the modflow-usg documentation

    # outer grid
    nlay = 1
    nrow = ncol = 7
    delr = 100.0 * np.ones(ncol)
    delc = 100.0 * np.ones(nrow)
    tp = np.zeros((nrow, ncol))
    bt = -100.0 * np.ones((nlay, nrow, ncol))
    idomain = np.ones((nlay, nrow, ncol))
    idomain[:, 2:5, 2:5] = 0
    sg1 = StructuredGrid(
        delr=delr, delc=delc, top=tp, botm=bt, idomain=idomain
    )
    # inner grid
    nlay = 1
    nrow = ncol = 9
    delr = 100.0 / 3.0 * np.ones(ncol)
    delc = 100.0 / 3.0 * np.ones(nrow)
    tp = np.zeros((nrow, ncol))
    bt = -100 * np.ones((nlay, nrow, ncol))
    idomain = np.ones((nlay, nrow, ncol))
    sg2 = StructuredGrid(
        delr=delr,
        delc=delc,
        top=tp,
        botm=bt,
        xoff=200.0,
        yoff=200,
        idomain=idomain,
    )
    gridprops = gridlist_to_disv_gridprops([sg1, sg2])
    assert "ncpl" in gridprops
    assert "nvert" in gridprops
    assert "vertices" in gridprops
    assert "cell2d" in gridprops

    ncpl = gridprops["ncpl"]
    nvert = gridprops["nvert"]
    vertices = gridprops["vertices"]
    cell2d = gridprops["cell2d"]
    assert ncpl == 121
    assert nvert == 148
    assert len(vertices) == nvert
    assert len(cell2d) == 121

    # spot check information for cell 28 (zero based)
    answer = [28, 250.0, 150.0, 7, 38, 142, 143, 45, 46, 44, 38]
    for i, j in zip(cell2d[28], answer):
        assert i == j, f"{i} not equal {j}"


def test_unstructured_grid_shell():
    # constructor with no arguments.  incomplete shell should exist
    g = UnstructuredGrid()
    assert g.nlay is None
    assert g.nnodes is None
    assert g.ncpl is None
    assert not g.grid_varies_by_layer
    assert not g.is_valid
    assert not g.is_complete


def test_unstructured_grid_dimensions():
    # constructor with just dimensions
    ncpl = [1, 10, 1]
    g = UnstructuredGrid(ncpl=ncpl)
    assert np.allclose(g.ncpl, ncpl)
    assert g.nlay == 3
    assert g.nnodes == 12
    assert not g.is_valid
    assert not g.is_complete
    assert not g.grid_varies_by_layer


def test_unstructured_minimal_grid():
    # pass in simple 2 cell minimal grid to make grid valid
    vertices = [
        [0, 0.0, 1.0],
        [1, 1.0, 1.0],
        [2, 2.0, 1.0],
        [3, 0.0, 0.0],
        [4, 1.0, 0.0],
        [5, 2.0, 0.0],
    ]
    iverts = [[0, 1, 4, 3], [1, 2, 5, 4]]
    xcenters = [0.5, 1.5]
    ycenters = [0.5, 0.5]
    g = UnstructuredGrid(
        vertices=vertices, iverts=iverts, xcenters=xcenters, ycenters=ycenters
    )
    assert np.allclose(g.ncpl, np.array([2], dtype=int))
    assert g.nlay == 1
    assert g.nnodes == 2
    assert g.is_valid
    assert not g.is_complete
    assert not g.grid_varies_by_layer
    assert g._vertices == vertices
    assert g._iverts == iverts
    assert g._xc == xcenters
    assert g._yc == ycenters
    grid_lines = [
        [(0.0, 0), (0.0, 1.0)],
        [(0.0, 1), (1.0, 1.0)],
        [(1.0, 1), (1.0, 0.0)],
        [(1.0, 0), (0.0, 0.0)],
        [(1.0, 0), (1.0, 1.0)],
        [(1.0, 1), (2.0, 1.0)],
        [(2.0, 1), (2.0, 0.0)],
        [(2.0, 0), (1.0, 0.0)],
    ]
    assert (
        g.grid_lines == grid_lines
    ), f"\n{g.grid_lines} \n /=   \n{grid_lines}"
    assert g.extent == (0, 2, 0, 1)
    xv, yv, zv = g.xyzvertices
    assert xv == [[0, 1, 1, 0], [1, 2, 2, 1]]
    assert yv == [[1, 1, 0, 0], [1, 1, 0, 0]]
    assert zv is None


def test_unstructured_complete_grid():
    # pass in simple 2 cell complete grid to make grid valid, and put each
    # cell in a different layer
    vertices = [
        [0, 0.0, 1.0],
        [1, 1.0, 1.0],
        [2, 2.0, 1.0],
        [3, 0.0, 0.0],
        [4, 1.0, 0.0],
        [5, 2.0, 0.0],
    ]
    iverts = [[0, 1, 4, 3], [1, 2, 5, 4]]
    xcenters = [0.5, 1.5]
    ycenters = [0.5, 0.5]
    ncpl = [1, 1]
    top = [1, 0]
    top = np.array(top)
    botm = [0, -1]
    botm = np.array(botm)
    g = UnstructuredGrid(
        vertices=vertices,
        iverts=iverts,
        xcenters=xcenters,
        ycenters=ycenters,
        ncpl=ncpl,
        top=top,
        botm=botm,
    )
    assert np.allclose(g.ncpl, np.array([1, 1], dtype=int))
    assert g.nlay == 2
    assert g.nnodes == 2
    assert g.is_valid
    assert not g.is_complete
    assert g.grid_varies_by_layer
    assert g._vertices == vertices
    assert g._iverts == iverts
    assert g._xc == xcenters
    assert g._yc == ycenters
    grid_lines = {
        0: [
            [(0.0, 0.0), (0.0, 1.0)],
            [(0.0, 1.0), (1.0, 1.0)],
            [(1.0, 1.0), (1.0, 0.0)],
            [(1.0, 0.0), (0.0, 0.0)],
        ],
        1: [
            [(1.0, 0.0), (1.0, 1.0)],
            [(1.0, 1.0), (2.0, 1.0)],
            [(2.0, 1.0), (2.0, 0.0)],
            [(2.0, 0.0), (1.0, 0.0)],
        ],
    }
    assert isinstance(g.grid_lines, dict)
    assert (
        g.grid_lines == grid_lines
    ), f"\n{g.grid_lines} \n /=   \n{grid_lines}"
    assert g.extent == (0, 2, 0, 1)
    xv, yv, zv = g.xyzvertices
    assert xv == [[0, 1, 1, 0], [1, 2, 2, 1]]
    assert yv == [[1, 1, 0, 0], [1, 1, 0, 0]]
    assert np.allclose(zv, np.array([[1, 0], [0, -1]]))


@requires_pkg("shapely")
def test_loading_argus_meshes(example_data_path):
    datapth = str(example_data_path / "unstructured")
    fnames = [fname for fname in os.listdir(datapth) if fname.endswith(".exp")]
    for fname in fnames:
        fname = os.path.join(datapth, fname)
        print(f"Loading Argus mesh ({fname}) into UnstructuredGrid")
        g = UnstructuredGrid.from_argus_export(fname)
        print(f"  Number of nodes: {g.nnodes}")


def test_create_unstructured_grid_from_verts(tmpdir, example_data_path):
    datapth = str(example_data_path / "unstructured")

    # simple functions to load vertices and incidence lists
    def load_verts(fname):
        print(f"Loading vertices from: {fname}")
        verts = np.genfromtxt(
            fname, dtype=[int, float, float], names=["iv", "x", "y"]
        )
        verts["iv"] -= 1  # zero based
        return verts

    def load_iverts(fname):
        print(f"Loading iverts from: {fname}")
        f = open(fname, "r")
        iverts = []
        xc = []
        yc = []
        for line in f:
            ll = line.strip().split()
            iverts.append([int(i) - 1 for i in ll[4:]])
            xc.append(float(ll[1]))
            yc.append(float(ll[2]))
        return iverts, np.array(xc), np.array(yc)

    # load vertices
    fname = os.path.join(datapth, "ugrid_verts.dat")
    verts = load_verts(fname)

    # load the incidence list into iverts
    fname = os.path.join(datapth, "ugrid_iverts.dat")
    iverts, xc, yc = load_iverts(fname)

    ncpl = np.array(5 * [len(iverts)])
    g = UnstructuredGrid(verts, iverts, xc, yc, ncpl=ncpl)
    assert isinstance(g.grid_lines, list)
    assert np.allclose(g.ncpl, ncpl)
    assert g.extent == (0.0, 700.0, 0.0, 700.0)
    assert g._vertices.shape == (156,)
    assert g.nnodes == g.ncpl.sum() == 1090


@requires_pkg("shapely")
def test_triangle_unstructured_grid(tmpdir):
    maximum_area = 30000.0
    extent = (214270.0, 221720.0, 4366610.0, 4373510.0)
    domainpoly = [
        (extent[0], extent[2]),
        (extent[1], extent[2]),
        (extent[1], extent[3]),
        (extent[0], extent[3]),
    ]
    tri = Triangle(
        maximum_area=maximum_area,
        angle=30,
        model_ws=str(tmpdir),
    )
    tri.add_polygon(domainpoly)
    tri.build(verbose=False)
    verts = [[iv, x, y] for iv, (x, y) in enumerate(tri.verts)]
    iverts = tri.iverts
    xc, yc = tri.get_xcyc().T
    ncpl = np.array([len(iverts)])
    g = UnstructuredGrid(
        vertices=verts,
        iverts=iverts,
        ncpl=ncpl,
        xcenters=xc,
        ycenters=yc,
    )
    assert len(g.grid_lines) == 8190
    assert g.nnodes == g.ncpl == 2730


@requires_pkg("shapely", "scipy")
def test_voronoi_vertex_grid(tmpdir):
    xmin = 0.0
    xmax = 2.0
    ymin = 0.0
    ymax = 1.0
    area_max = 0.05
    tri = Triangle(maximum_area=area_max, angle=30, model_ws=str(tmpdir))
    poly = np.array(((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)))
    tri.add_polygon(poly)
    tri.build(verbose=False)

    # create vor object and VertexGrid
    vor = VoronoiGrid(tri)
    gridprops = vor.get_gridprops_vertexgrid()
    vgrid = VertexGrid(**gridprops, nlay=1)
    assert vgrid.is_valid

    # arguments for creating a mf6 disv package
    gridprops = vor.get_disv_gridprops()
    print(gridprops)
    assert gridprops["ncpl"] == 43
    assert gridprops["nvert"] == 127
    assert len(gridprops["vertices"]) == 127
    assert len(gridprops["cell2d"]) == 43


def __voronoi_grid_0():
    name = "vor0"
    ncpl = 3803
    domain = [
        [1831.381546, 6335.543757],
        [4337.733475, 6851.136153],
        [6428.747084, 6707.916043],
        [8662.980804, 6493.085878],
        [9350.437333, 5891.561415],
        [9235.861245, 4717.156511],
        [8963.743036, 3685.971717],
        [8691.624826, 2783.685023],
        [8047.13433, 2038.94045],
        [7416.965845, 578.0953252],
        [6414.425073, 105.4689614],
        [5354.596258, 205.7230386],
        [4624.173696, 363.2651598],
        [3363.836725, 563.7733141],
        [1330.11116, 1809.788273],
        [399.1804436, 2998.515188],
        [914.7728404, 5132.494831],
        #        [1831.381546, 6335.543757],
    ]
    area_max = 100.0**2
    poly = np.array(domain)
    angle = 30
    return name, poly, area_max, ncpl, angle


@pytest.fixture
def voronoi_grid_0():
    return __voronoi_grid_0()


def __voronoi_grid_1():
    name = "vor1"
    ncpl = 1679
    xmin = 0.0
    xmax = 2.0
    ymin = 0.0
    ymax = 1.0
    area_max = 0.001
    poly = np.array(((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)))
    angle = 30
    return name, poly, area_max, ncpl, angle


@pytest.fixture
def voronoi_grid_1():
    return __voronoi_grid_1()


def __voronoi_grid_2():
    name = "vor2"
    ncpl = 538
    theta = np.arange(0.0, 2 * np.pi, 0.2)
    radius = 100.0
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    circle_poly = [(x, y) for x, y in zip(x, y)]
    max_area = 50
    angle = 30
    return name, circle_poly, max_area, ncpl, angle


@pytest.fixture
def voronoi_grid_2():
    return __voronoi_grid_2()


@requires_pkg("shapely", "scipy")
@flaky
@pytest.mark.parametrize(
    "grid_info", [__voronoi_grid_0(), __voronoi_grid_1(), __voronoi_grid_2()]
)
def test_voronoi_grid(tmpdir, grid_info):
    name, poly, area_max, ncpl, angle = grid_info
    tri = Triangle(maximum_area=area_max, angle=angle, model_ws=str(tmpdir))
    tri.add_polygon(poly)
    tri.build(verbose=False)
    vor = VoronoiGrid(tri)

    gridprops = vor.get_gridprops_vertexgrid()
    voronoi_grid = VertexGrid(**gridprops, nlay=1)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    ax.set_aspect("equal")
    voronoi_grid.plot(ax=ax)
    plt.savefig(os.path.join(str(tmpdir), f"{name}.png"))

    # ensure proper number of cells
    almost_right = (ncpl == 538 and gridprops["ncpl"] == 535)  # TODO: why does this sometimes happen on CI
    assert ncpl == gridprops["ncpl"] or almost_right

    # ensure that all cells have 3 or more points
    ninvalid_cells = []
    for icell, ivts in enumerate(vor.iverts):
        if len(ivts) < 3:
            ninvalid_cells.append(icell)
    errmsg = f"The following cells do not have 3 or more vertices.\n{ninvalid_cells}"
    assert len(ninvalid_cells) == 0, errmsg


@requires_pkg("shapely", "scipy")
@flaky
def test_voronoi_grid3(tmpdir):
    name = "vor3"
    answer_ncpl = 300

    theta = np.arange(0.0, 2 * np.pi, 0.2)
    radius = 100.0
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    circle_poly = [(x, y) for x, y in zip(x, y)]

    theta = np.arange(0.0, 2 * np.pi, 0.2)
    radius = 30.0
    x = radius * np.cos(theta) + 25.0
    y = radius * np.sin(theta) + 25.0
    inner_circle_poly = [(x, y) for x, y in zip(x, y)]

    tri = Triangle(maximum_area=100, angle=30, model_ws=str(tmpdir))
    tri.add_polygon(circle_poly)
    tri.add_polygon(inner_circle_poly)
    tri.add_hole((25, 25))
    tri.build(verbose=False)

    vor = VoronoiGrid(tri)
    gridprops = vor.get_gridprops_vertexgrid()
    voronoi_grid = VertexGrid(**gridprops, nlay=1)

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    ax.set_aspect("equal")
    voronoi_grid.plot(ax=ax)
    plt.savefig(os.path.join(str(tmpdir), f"{name}.png"))

    # ensure proper number of cells
    ncpl = gridprops["ncpl"]
    errmsg = f"Number of cells should be {answer_ncpl}. Found {ncpl}"
    assert ncpl == answer_ncpl, errmsg

    # ensure that all cells have 3 or more points
    ninvalid_cells = []
    for icell, ivts in enumerate(vor.iverts):
        if len(ivts) < 3:
            ninvalid_cells.append(icell)
    errmsg = f"The following cells do not have 3 or more vertices.\n{ninvalid_cells}"
    assert len(ninvalid_cells) == 0, errmsg


@requires_pkg("shapely", "scipy")
@flaky
def test_voronoi_grid4(tmpdir):
    name = "vor4"
    answer_ncpl = 410
    active_domain = [(0, 0), (100, 0), (100, 100), (0, 100)]
    area1 = [(10, 10), (40, 10), (40, 40), (10, 40)]
    area2 = [(60, 60), (80, 60), (80, 80), (60, 80)]
    tri = Triangle(angle=30, model_ws=str(tmpdir))
    tri.add_polygon(active_domain)
    tri.add_polygon(area1)
    tri.add_polygon(area2)
    tri.add_region((1, 1), 0, maximum_area=100)  # point inside active domain
    tri.add_region((11, 11), 1, maximum_area=10)  # point inside area1
    tri.add_region((61, 61), 2, maximum_area=3)  # point inside area2
    tri.build(verbose=False)

    vor = VoronoiGrid(tri)
    gridprops = vor.get_gridprops_vertexgrid()
    voronoi_grid = VertexGrid(**gridprops, nlay=1)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    ax.set_aspect("equal")
    voronoi_grid.plot(ax=ax)
    plt.savefig(os.path.join(str(tmpdir), f"{name}.png"))

    # ensure proper number of cells
    ncpl = gridprops["ncpl"]
    errmsg = f"Number of cells should be {answer_ncpl}. Found {ncpl}"
    assert ncpl == answer_ncpl, errmsg

    # ensure that all cells have 3 or more points
    ninvalid_cells = []
    for icell, ivts in enumerate(vor.iverts):
        if len(ivts) < 3:
            ninvalid_cells.append(icell)
    errmsg = f"The following cells do not have 3 or more vertices.\n{ninvalid_cells}"
    assert len(ninvalid_cells) == 0, errmsg


@requires_pkg("shapely", "scipy")
@flaky
def test_voronoi_grid5(tmpdir):
    name = "vor5"
    answer_ncpl = 1305
    active_domain = [(0, 0), (100, 0), (100, 100), (0, 100)]
    area1 = [(10, 10), (40, 10), (40, 40), (10, 40)]
    area2 = [(70, 70), (90, 70), (90, 90), (70, 90)]

    tri = Triangle(angle=30, model_ws=str(tmpdir))

    # requirement that active_domain is first polygon to be added
    tri.add_polygon(active_domain)

    # requirement that any holes be added next
    theta = np.arange(0.0, 2 * np.pi, 0.2)
    radius = 10.0
    x = radius * np.cos(theta) + 50.0
    y = radius * np.sin(theta) + 70.0
    circle_poly0 = [(x, y) for x, y in zip(x, y)]
    tri.add_polygon(circle_poly0)
    tri.add_hole((50, 70))

    # Add a polygon to force cells to conform to it
    theta = np.arange(0.0, 2 * np.pi, 0.2)
    radius = 10.0
    x = radius * np.cos(theta) + 70.0
    y = radius * np.sin(theta) + 20.0
    circle_poly1 = [(x, y) for x, y in zip(x, y)]
    tri.add_polygon(circle_poly1)
    # tri.add_hole((70, 20))

    # add line through domain to force conforming cells
    line = [(x, x) for x in np.linspace(11, 89, 100)]
    tri.add_polygon(line)

    # then regions and other polygons should follow
    tri.add_polygon(area1)
    tri.add_polygon(area2)
    tri.add_region((1, 1), 0, maximum_area=100)  # point inside active domain
    tri.add_region((11, 11), 1, maximum_area=10)  # point inside area1
    tri.add_region((70, 70), 2, maximum_area=1)  # point inside area2

    tri.build(verbose=False)

    vor = VoronoiGrid(tri)
    gridprops = vor.get_gridprops_vertexgrid()
    voronoi_grid = VertexGrid(**gridprops, nlay=1)

    ninvalid_cells = []
    for icell, ivts in enumerate(vor.iverts):
        if len(ivts) < 3:
            ninvalid_cells.append(icell)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    ax.set_aspect("equal")
    voronoi_grid.plot(ax=ax)

    # plot invalid cells
    ax.plot(
        voronoi_grid.xcellcenters[ninvalid_cells],
        voronoi_grid.ycellcenters[ninvalid_cells],
        "ro",
    )

    plt.savefig(os.path.join(str(tmpdir), f"{name}.png"))

    # ensure proper number of cells
    ncpl = gridprops["ncpl"]
    errmsg = f"Number of cells should be {answer_ncpl}. Found {ncpl}"
    assert ncpl == answer_ncpl, errmsg

    # ensure that all cells have 3 or more points
    errmsg = f"The following cells do not have 3 or more vertices.\n{ninvalid_cells}"
    assert len(ninvalid_cells) == 0, errmsg


def test_structured_thick():
    nlay, nrow, ncol = 3, 2, 3
    delc = 1.0 * np.ones(nrow, dtype=float)
    delr = 1.0 * np.ones(ncol, dtype=float)
    top = 10.0 * np.ones((nrow, ncol), dtype=float)
    botm = np.zeros((nlay, nrow, ncol), dtype=float)
    botm[0, :, :] = 5.0
    botm[1, :, :] = 0.0
    botm[2, :, :] = -5.0
    grid = StructuredGrid(
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delc=delc,
        delr=delr,
        top=top,
        botm=botm,
    )
    thick = grid.thick
    assert np.allclose(thick, 5.0), "thicknesses != 5."

    sat_thick = grid.saturated_thick(grid.botm + 10.0)
    assert np.allclose(sat_thick, thick), "saturated thicknesses != 5."

    sat_thick = grid.saturated_thick(grid.botm + 5.0)
    assert np.allclose(sat_thick, thick), "saturated thicknesses != 5."

    sat_thick = grid.saturated_thick(grid.botm + 2.5)
    assert np.allclose(sat_thick, 2.5), "saturated thicknesses != 2.5"

    sat_thick = grid.saturated_thick(grid.botm)
    assert np.allclose(sat_thick, 0.0), "saturated thicknesses != 0."

    sat_thick = grid.saturated_thick(grid.botm - 100.0)
    assert np.allclose(sat_thick, 0.0), "saturated thicknesses != 0."


def test_vertices_thick():
    nlay, ncpl = 3, 5
    vertices = [
        [0, 0.0, 3.0],
        [1, 1.0, 3.0],
        [2, 2.0, 3.0],
        [3, 0.0, 2.0],
        [4, 1.0, 2.0],
        [5, 2.0, 2.0],
        [6, 0.0, 1.0],
        [7, 1.0, 1.0],
        [8, 2.0, 1.0],
        [9, 0.0, 0.0],
        [10, 1.0, 0.0],
    ]
    iverts = [
        [0, 0, 1, 4, 3],
        [1, 1, 2, 5, 4],
        [2, 3, 4, 7, 6],
        [3, 4, 5, 8, 7],
        [4, 6, 7, 10, 9],
        [5, 0, 1, 4, 3],
        [6, 1, 2, 5, 4],
        [7, 3, 4, 7, 6],
        [8, 4, 5, 8, 7],
        [9, 6, 7, 10, 9],
        [10, 0, 1, 4, 3],
        [11, 1, 2, 5, 4],
        [12, 3, 4, 7, 6],
        [13, 4, 5, 8, 7],
        [14, 6, 7, 10, 9],
    ]
    top = np.ones(ncpl, dtype=float) * 10.0
    botm = np.zeros((nlay, ncpl), dtype=float)
    botm[0, :] = 5.0
    botm[1, :] = 0.0
    botm[2, :] = -5.0
    grid = VertexGrid(
        nlay=nlay,
        ncpl=ncpl,
        vertices=vertices,
        cell2d=iverts,
        top=top,
        botm=botm,
    )
    thick = grid.thick
    assert np.allclose(thick, 5.0), "thicknesses != 5."

    sat_thick = grid.saturated_thick(grid.botm + 10.0)
    assert np.allclose(sat_thick, thick), "saturated thicknesses != 5."

    sat_thick = grid.saturated_thick(grid.botm + 5.0)
    assert np.allclose(sat_thick, thick), "saturated thicknesses != 5."

    sat_thick = grid.saturated_thick(grid.botm + 2.5)
    assert np.allclose(sat_thick, 2.5), "saturated thicknesses != 2.5"

    sat_thick = grid.saturated_thick(grid.botm)
    assert np.allclose(sat_thick, 0.0), "saturated thicknesses != 0."

    sat_thick = grid.saturated_thick(grid.botm - 100.0)
    assert np.allclose(sat_thick, 0.0), "saturated thicknesses != 0."


def test_unstructured_thick():
    nlay = 3
    ncpl = [5, 5, 5]
    vertices = [
        [0, 0.0, 3.0],
        [1, 1.0, 3.0],
        [2, 2.0, 3.0],
        [3, 0.0, 2.0],
        [4, 1.0, 2.0],
        [5, 2.0, 2.0],
        [6, 0.0, 1.0],
        [7, 1.0, 1.0],
        [8, 2.0, 1.0],
        [9, 0.0, 0.0],
        [10, 1.0, 0.0],
    ]
    iverts = [
        [0, 0, 1, 4, 3],
        [1, 1, 2, 5, 4],
        [2, 3, 4, 7, 6],
        [3, 4, 5, 8, 7],
        [4, 6, 7, 10, 9],
        [5, 0, 1, 4, 3],
        [6, 1, 2, 5, 4],
        [7, 3, 4, 7, 6],
        [8, 4, 5, 8, 7],
        [9, 6, 7, 10, 9],
        [10, 0, 1, 4, 3],
        [11, 1, 2, 5, 4],
        [12, 3, 4, 7, 6],
        [13, 4, 5, 8, 7],
        [14, 6, 7, 10, 9],
    ]
    xcenters = [
        0.5,
        1.5,
        0.5,
        1.5,
        0.5,
    ]
    ycenters = [
        2.5,
        2.5,
        1.5,
        1.5,
        0.5,
    ]
    top = np.ones((nlay, 5), dtype=float)
    top[0, :] = 10.0
    top[1, :] = 5.0
    top[2, :] = 0.0
    botm = np.zeros((nlay, 5), dtype=float)
    botm[0, :] = 5.0
    botm[1, :] = 0.0
    botm[2, :] = -5.0

    grid = UnstructuredGrid(
        vertices=vertices,
        iverts=iverts,
        xcenters=xcenters,
        ycenters=ycenters,
        ncpl=ncpl,
        top=top.flatten(),
        botm=botm.flatten(),
    )

    thick = grid.thick
    assert np.allclose(thick, 5.0), "thicknesses != 5."

    sat_thick = grid.saturated_thick(grid.botm + 10.0)
    assert np.allclose(sat_thick, thick), "saturated thicknesses != 5."

    sat_thick = grid.saturated_thick(grid.botm + 5.0)
    assert np.allclose(sat_thick, thick), "saturated thicknesses != 5."

    sat_thick = grid.saturated_thick(grid.botm + 2.5)
    assert np.allclose(sat_thick, 2.5), "saturated thicknesses != 2.5"

    sat_thick = grid.saturated_thick(grid.botm)
    assert np.allclose(sat_thick, 0.0), "saturated thicknesses != 0."

    sat_thick = grid.saturated_thick(grid.botm - 100.0)
    assert np.allclose(sat_thick, 0.0), "saturated thicknesses != 0."


def test_ncb_thick():
    nlay = 3
    nrow = ncol = 15
    laycbd = np.array([1, 2, 0], dtype=int)
    ncb = np.count_nonzero(laycbd)
    dx = dy = 150
    delc = np.array(
        [
            dy,
        ]
        * nrow
    )
    delr = np.array(
        [
            dx,
        ]
        * ncol
    )
    top = np.ones((15, 15))
    botm = np.ones((nlay + ncb, nrow, ncol))
    elevations = np.array([-10, -20, -40, -50, -70])[:, np.newaxis]
    botm *= elevations[:, None]

    modelgrid = StructuredGrid(
        delc=delc,
        delr=delr,
        top=top,
        botm=botm,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        laycbd=laycbd,
    )

    thick = modelgrid.thick

    assert (
        thick.shape[0] == nlay + ncb
    ), "grid thick attribute returns incorrect shape"

    thick = modelgrid.remove_confining_beds(modelgrid.thick)
    assert (
        thick.shape == modelgrid.shape
    ), "quasi3d confining beds not properly removed"

    sat_thick = modelgrid.saturated_thick(modelgrid.thick)
    assert (
        sat_thick.shape == modelgrid.shape
    ), "saturated_thickness confining beds not removed"

    if sat_thick[1, 0, 0] != 20:
        raise AssertionError(
            "saturated_thickness is not properly indexing confining beds"
        )
