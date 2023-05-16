import os
from warnings import warn

import matplotlib
import numpy as np
import pytest
from autotest.test_dis_cases import case_dis, case_disv
from autotest.test_grid_cases import GridCases
from flaky import flaky
from matplotlib import pyplot as plt
from modflow_devtools.markers import requires_exe, requires_pkg
from packaging import version
from pytest_cases import parametrize_with_cases

import flopy
from flopy.discretization import StructuredGrid, UnstructuredGrid, VertexGrid
from flopy.mf6 import MFSimulation
from flopy.modflow import Modflow, ModflowDis
from flopy.utils.crs import get_authority_crs
from flopy.utils.cvfdutil import gridlist_to_disv_gridprops, to_cvfd
from flopy.utils.triangle import Triangle
from flopy.utils.voronoi import VoronoiGrid


@pytest.fixture
def minimal_unstructured_grid_info():
    d = {
        # pass in simple 2 cell minimal grid to make grid valid
        "vertices": [
            [0, 0.0, 1.0],
            [1, 1.0, 1.0],
            [2, 2.0, 1.0],
            [3, 0.0, 0.0],
            [4, 1.0, 0.0],
            [5, 2.0, 0.0],
        ],
        "iverts": [[0, 1, 4, 3], [1, 2, 5, 4]],
        "xcenters": [0.5, 1.5],
        "ycenters": [0.5, 0.5],
    }
    return d


@pytest.fixture
def minimal_vertex_grid_info(minimal_unstructured_grid_info):
    usg_info = minimal_unstructured_grid_info
    d = {}
    d["vertices"] = minimal_unstructured_grid_info["vertices"]
    cell2d = []
    for n in range(len(usg_info["iverts"])):
        cell2d_n = [
            n,
            usg_info["xcenters"][n],
            usg_info["ycenters"][n],
        ] + usg_info["iverts"][n]
        cell2d.append(cell2d_n)
    d["cell2d"] = cell2d
    d["ncpl"] = len(cell2d)
    d["nlay"] = 1
    return d


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
    nnodes = nlay * nrow * ncol
    ml = Modflow()
    dis = ModflowDis(
        ml, nlay=nlay, nrow=nrow, ncol=ncol, top=50, botm=[0, -1, -2]
    )
    nodes = list(range(nnodes))
    indices = np.indices((nlay, nrow, ncol))
    layers = indices[0].flatten()
    rows = indices[1].flatten()
    cols = indices[2].flatten()
    for node, (l, r, c) in enumerate(zip(layers, rows, cols)):
        # ensure get_lrc returns zero-based layer row col
        assert dis.get_lrc(node)[0] == (l, r, c)
        # ensure get_node returns zero-based node number
        assert dis.get_node((l, r, c))[0] == node

    # check full list
    lrc_list = list(zip(layers, rows, cols))
    assert dis.get_lrc(nodes) == lrc_list
    assert dis.get_node(lrc_list) == nodes

    # check array-like input
    assert dis.get_lrc(np.arange(nnodes)) == lrc_list
    # dis.get_node does not accept array-like inputs, just tuple or list

    # check out of bounds errors
    with pytest.raises(ValueError, match="index 60 is out of bounds for"):
        dis.get_lrc(nnodes)
    with pytest.raises(ValueError, match="invalid entry in coordinates array"):
        dis.get_node((4, 4, 4))


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


@pytest.fixture
def dis_model():
    return case_dis()


@pytest.fixture
def disv_model():
    return case_disv()


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
        model_ws=example_data_path / "freyberg_multilayer_transient",
    )
    mg = ml.modelgrid

    assert mg.size == np.prod((mg.nlay, mg.nrow, mg.ncol))

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
        sim_ws=example_data_path / "mf6" / "test003_gwfs_disv"
    )
    ml = sim.get_model(list(sim.model_names)[0])
    mg = ml.modelgrid

    assert mg.size == np.prod((mg.nlay, mg.ncpl))

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
    ws = example_data_path / "unstructured"
    name = ws / "ugrid_verts.dat"
    verts = load_verts(name)

    name = ws / "ugrid_iverts.dat"
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

    assert mg.size == mg.nnodes

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
def test_structured_from_gridspec(example_data_path, spc_file):
    fn = example_data_path / "specfile" / spc_file
    modelgrid = StructuredGrid.from_gridspec(fn)
    assert isinstance(modelgrid, StructuredGrid)

    lc = modelgrid.plot()
    assert isinstance(
        lc, matplotlib.collections.LineCollection
    ), f"could not plot grid object created from {fn}"
    plt.close()

    extents = modelgrid.extent
    theta = modelgrid.angrot_radians
    if fn.name == "grdrot.spc":
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


@requires_pkg("shapely")
def test_unstructured_from_argus_mesh(example_data_path):
    datapth = example_data_path / "unstructured"
    fnames = [fname for fname in os.listdir(datapth) if fname.endswith(".exp")]
    for fname in fnames:
        fname = datapth / fname
        print(f"Loading Argus mesh ({fname}) into UnstructuredGrid")
        g = UnstructuredGrid.from_argus_export(fname)
        print(f"  Number of nodes: {g.nnodes}")


def test_unstructured_from_verts_and_iverts(
    function_tmpdir, example_data_path
):
    datapth = example_data_path / "unstructured"

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
    fname = datapth / "ugrid_verts.dat"
    verts = load_verts(fname)

    # load the incidence list into iverts
    fname = datapth / "ugrid_iverts.dat"
    iverts, xc, yc = load_iverts(fname)

    ncpl = np.array(5 * [len(iverts)])
    g = UnstructuredGrid(verts, iverts, xc, yc, ncpl=ncpl)
    assert isinstance(g.grid_lines, list)
    assert np.allclose(g.ncpl, ncpl)
    assert g.extent == (0.0, 700.0, 0.0, 700.0)
    assert g._vertices.shape == (156,)
    assert g.nnodes == g.ncpl.sum() == 1090


def test_unstructured_from_gridspec(example_data_path):
    model_path = example_data_path / "freyberg_usg"
    spec_path = model_path / "freyberg.usg.gsf"
    grid = UnstructuredGrid.from_gridspec(spec_path)

    with open(spec_path) as file:
        lines = file.readlines()
        split = [line.strip().split() for line in lines]

        # check number of nodes
        nnodes = int(split[1][0])
        assert len(grid.iverts) == nnodes
        assert len(split[1]) == 4

        # check number of vertices
        nverts = int(split[2][0])
        assert len(grid.verts) == nverts
        assert len(split[2]) == 1

        # check vertices
        expected_verts = [
            (float(s[0]), float(s[1]), float(s[2]))
            for s in split[3 : (3 + nverts)]
        ]
        for i, ev in enumerate(expected_verts[:10]):
            assert grid.verts[i][0] == ev[0]
            assert grid.verts[i][1] == ev[1]
        for i, ev in enumerate(expected_verts[-10:-1]):
            ii = nverts - 10 + i
            assert grid.verts[ii][0] == ev[0]
            assert grid.verts[ii][1] == ev[1]

        # check nodes
        expected_nodes = [
            (
                int(s[0]),
                float(s[1]),
                float(s[2]),
                float(s[3]),
                int(s[4]),
                int(s[5]),
            )
            for s in split[(3 + nverts) : -1]
        ]
        for i, en in enumerate(expected_nodes):
            assert any(xcc == en[1] for xcc in grid.xcellcenters)
            assert any(ycc == en[2] for ycc in grid.ycellcenters)

        # check elevation
        assert max(grid.top) == max([xyz[2] for xyz in expected_verts])
        assert min(grid.botm) == min([xyz[2] for xyz in expected_verts])


@requires_pkg("pyproj")
@pytest.mark.parametrize(
    "crs,expected_srs",
    (
        (None, None),
        (26916, "EPSG:26916"),
        ("epsg:5070", "EPSG:5070"),
        (
            "+proj=tmerc +lat_0=0 +lon_0=-90 +k=0.9996 +x_0=520000 +y_0=-4480000 +datum=NAD83 +units=m +no_defs ",
            "EPSG:3070",
        ),
        pytest.param(4269, None, marks=pytest.mark.xfail),
    ),
)
def test_grid_crs(
    minimal_unstructured_grid_info, crs, expected_srs, function_tmpdir
):
    import pyproj

    d = minimal_unstructured_grid_info
    delr = np.ones(10)
    delc = np.ones(10)
    sg = StructuredGrid(delr=delr, delc=delc, crs=crs)
    if crs is not None:
        assert isinstance(sg.crs, pyproj.CRS)
        assert sg.crs.srs == expected_srs

    usg = UnstructuredGrid(**d, crs=crs)
    assert getattr(sg.crs, "srs", None) == expected_srs

    vg = VertexGrid(vertices=d["vertices"], crs=crs)
    assert getattr(sg.crs, "srs", None) == expected_srs

    # test input of pyproj.CRS object
    if crs == 26916:
        sg2 = StructuredGrid(delr=delr, delc=delc, crs=sg.crs)

        if crs is not None:
            assert isinstance(sg2.crs, pyproj.CRS)
        assert getattr(sg2.crs, "srs", None) == expected_srs

        # test input of projection file
        prjfile = function_tmpdir / "grid_crs.prj"
        with open(prjfile, "w", encoding="utf-8") as dest:
            write_text = sg.crs.to_wkt()
            dest.write(write_text)

        sg3 = StructuredGrid(delr=delr, delc=delc, prjfile=prjfile)
        if crs is not None:
            assert isinstance(sg3.crs, pyproj.CRS)
        assert getattr(sg3.crs, "srs", None) == expected_srs


@requires_pkg("pyproj")
@pytest.mark.parametrize(
    "crs,expected_srs",
    (
        (None, None),
        (26916, "EPSG:26916"),
        ("epsg:5070", "EPSG:5070"),
        (
            "+proj=tmerc +lat_0=0 +lon_0=-90 +k=0.9996 +x_0=520000 +y_0=-4480000 +datum=NAD83 +units=m +no_defs ",
            "EPSG:3070",
        ),
        ("ESRI:102733", "ESRI:102733"),
        pytest.param(4269, None, marks=pytest.mark.xfail),
    ),
)
def test_grid_set_crs(crs, expected_srs, function_tmpdir):
    import pyproj

    delr = np.ones(10)
    delc = np.ones(10)
    sg = StructuredGrid(delr=delr, delc=delc)
    sg.crs = crs
    if crs is not None:
        assert isinstance(sg.crs, pyproj.CRS)
    assert getattr(sg.crs, "srs", None) == expected_srs

    # test setting the crs via set_coord_info
    sg.set_coord_info()
    if crs is not None:
        assert isinstance(sg.crs, pyproj.CRS)
    assert getattr(sg.crs, "srs", None) == expected_srs
    sg.set_coord_info(crs=crs)
    if crs is not None:
        assert isinstance(sg.crs, pyproj.CRS)
    assert getattr(sg.crs, "srs", None) == expected_srs
    sg.set_coord_info(crs=26915, merge_coord_info=False)
    assert getattr(sg.crs, "srs", None) == "EPSG:26915"
    # reset back to test case crs
    sg = StructuredGrid(delr=delr, delc=delc, crs=crs)

    # test input of projection file
    if crs is not None:
        prjfile = function_tmpdir / "grid_crs.prj"
        with open(prjfile, "w", encoding="utf-8") as dest:
            write_text = sg.crs.to_wkt()
            dest.write(write_text)
        sg = StructuredGrid(delr=delr, delc=delc)
        sg.prjfile = prjfile
        if crs is not None:
            assert isinstance(sg.crs, pyproj.CRS)
        assert getattr(sg.crs, "srs", None) == expected_srs
        assert sg.prjfile == prjfile
        sg.set_coord_info(prjfile=prjfile)
        if crs is not None:
            assert isinstance(sg.crs, pyproj.CRS)
            assert getattr(sg.crs, "srs", None) == expected_srs
        assert sg.prjfile == prjfile

    # test setting another crs
    sg.crs = 26915
    assert sg.crs == get_authority_crs(26915)

    if (
        version.parse(flopy.__version__) < version.parse("3.4")
        and crs is not None
    ):
        pyproj_crs = get_authority_crs(crs)
        epsg = pyproj_crs.to_epsg()
        if epsg is not None:
            sg = StructuredGrid(delr=delr, delc=delc, crs=crs)
            sg.epsg = epsg
            if crs is not None:
                assert isinstance(sg.crs, pyproj.CRS)
            assert getattr(sg.crs, "srs", None) == expected_srs
            assert sg.epsg == epsg

        sg = StructuredGrid(delr=delr, delc=delc)
        sg.proj4 = pyproj_crs.to_proj4()
        if crs is not None:
            assert isinstance(sg.crs, pyproj.CRS)
        assert getattr(sg.crs, "srs", None) == expected_srs
        assert sg.proj4 == pyproj_crs.to_proj4()

        if crs is not None:
            sg = StructuredGrid(delr=delr, delc=delc)
            sg.prj = prjfile
            if crs is not None:
                assert isinstance(sg.crs, pyproj.CRS)
            assert getattr(sg.crs, "srs", None) == expected_srs
            assert sg.prj == prjfile


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


def test_unstructured_minimal_grid_ctor(minimal_unstructured_grid_info):
    # pass in simple 2 cell minimal grid to make grid valid
    d = minimal_unstructured_grid_info
    g = UnstructuredGrid(**d)
    assert np.allclose(g.ncpl, np.array([2], dtype=int))
    assert g.nlay == 1
    assert g.nnodes == 2
    assert g.is_valid
    assert not g.is_complete
    assert not g.grid_varies_by_layer
    assert g._vertices == d["vertices"]
    assert g._iverts == d["iverts"]
    assert g._xc == d["xcenters"]
    assert g._yc == d["ycenters"]
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


def test_unstructured_complete_grid_ctor(minimal_unstructured_grid_info):
    # pass in simple 2 cell complete grid to make grid valid, and put each
    # cell in a different layer
    d = minimal_unstructured_grid_info
    ncpl = [1, 1]
    top = [1, 0]
    top = np.array(top)
    botm = [0, -1]
    botm = np.array(botm)
    g = UnstructuredGrid(
        ncpl=ncpl, top=top, botm=botm, **minimal_unstructured_grid_info
    )
    assert np.allclose(g.ncpl, np.array([1, 1], dtype=int))
    assert g.nlay == 2
    assert g.nnodes == 2
    assert g.is_valid
    assert not g.is_complete
    assert g.grid_varies_by_layer
    assert g._vertices == d["vertices"]
    assert g._iverts == d["iverts"]
    assert g._xc == d["xcenters"]
    assert g._yc == d["ycenters"]
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
@requires_exe("triangle")
def test_triangle_unstructured_grid(function_tmpdir):
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
        model_ws=function_tmpdir,
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
@requires_exe("triangle")
def test_voronoi_vertex_grid(function_tmpdir):
    xmin = 0.0
    xmax = 2.0
    ymin = 0.0
    ymax = 1.0
    area_max = 0.05
    tri = Triangle(maximum_area=area_max, angle=30, model_ws=function_tmpdir)
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


@flaky
@requires_exe("triangle")
@requires_pkg("shapely", "scipy")
@parametrize_with_cases("grid_info", cases=GridCases, prefix="voronoi")
def test_voronoi_grid(request, function_tmpdir, grid_info):
    name = (
        request.node.name.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )
    ncpl, vor, gridprops, grid = grid_info

    # TODO: debug off-by-3 issue
    #  could be a rounding error as described here:
    #  https://github.com/modflowpy/flopy/issues/1492#issuecomment-1210596349

    # ensure proper number of cells
    almost_right = ncpl == 538 and gridprops["ncpl"] == 535
    if almost_right:
        warn(f"off-by-3")

    # ensure that all cells have 3 or more points
    invalid_cells = [i for i, ivts in enumerate(vor.iverts) if len(ivts) < 3]

    # make a plot including invalid cells
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    ax.set_aspect("equal")
    grid.plot(ax=ax)
    ax.plot(
        grid.xcellcenters[invalid_cells],
        grid.ycellcenters[invalid_cells],
        "ro",
    )
    plt.savefig(function_tmpdir / f"{name}.png")

    assert ncpl == gridprops["ncpl"] or almost_right
    assert (
        len(invalid_cells) == 0
    ), f"The following cells do not have 3 or more vertices.\n{invalid_cells}"


@pytest.fixture
def structured_grid():
    return GridCases().structured_small()


@pytest.fixture
def vertex_grid():
    return GridCases().vertex_small()


@pytest.fixture
def unstructured_grid():
    return GridCases().unstructured_small()


def test_structured_thickness(structured_grid):
    thickness = structured_grid.cell_thickness
    assert np.allclose(thickness, 5.0), "thicknesses != 5."

    sat_thick = structured_grid.saturated_thickness(
        structured_grid.botm + 10.0
    )
    assert np.allclose(sat_thick, thickness), "saturated thicknesses != 5."

    sat_thick = structured_grid.saturated_thickness(structured_grid.botm + 5.0)
    assert np.allclose(sat_thick, thickness), "saturated thicknesses != 5."

    sat_thick = structured_grid.saturated_thickness(structured_grid.botm + 2.5)
    assert np.allclose(sat_thick, 2.5), "saturated thicknesses != 2.5"

    sat_thick = structured_grid.saturated_thickness(structured_grid.botm)
    assert np.allclose(sat_thick, 0.0), "saturated thicknesses != 0."

    sat_thick = structured_grid.saturated_thickness(
        structured_grid.botm - 100.0
    )
    assert np.allclose(sat_thick, 0.0), "saturated thicknesses != 0."


def test_vertices_thickness(vertex_grid):
    thickness = vertex_grid.cell_thickness
    assert np.allclose(thickness, 5.0), "thicknesses != 5."

    sat_thick = vertex_grid.saturated_thickness(vertex_grid.botm + 10.0)
    assert np.allclose(sat_thick, thickness), "saturated thicknesses != 5."

    sat_thick = vertex_grid.saturated_thickness(vertex_grid.botm + 5.0)
    assert np.allclose(sat_thick, thickness), "saturated thicknesses != 5."

    sat_thick = vertex_grid.saturated_thickness(vertex_grid.botm + 2.5)
    assert np.allclose(sat_thick, 2.5), "saturated thicknesses != 2.5"

    sat_thick = vertex_grid.saturated_thickness(vertex_grid.botm)
    assert np.allclose(sat_thick, 0.0), "saturated thicknesses != 0."

    sat_thick = vertex_grid.saturated_thickness(vertex_grid.botm - 100.0)
    assert np.allclose(sat_thick, 0.0), "saturated thicknesses != 0."


def test_unstructured_thickness(unstructured_grid):
    thickness = unstructured_grid.cell_thickness
    assert np.allclose(thickness, 5.0), "thicknesses != 5."

    sat_thick = unstructured_grid.saturated_thickness(
        unstructured_grid.botm + 10.0
    )
    assert np.allclose(sat_thick, thickness), "saturated thicknesses != 5."

    sat_thick = unstructured_grid.saturated_thickness(
        unstructured_grid.botm + 5.0
    )
    assert np.allclose(sat_thick, thickness), "saturated thicknesses != 5."

    sat_thick = unstructured_grid.saturated_thickness(
        unstructured_grid.botm + 2.5
    )
    assert np.allclose(sat_thick, 2.5), "saturated thicknesses != 2.5"

    sat_thick = unstructured_grid.saturated_thickness(unstructured_grid.botm)
    assert np.allclose(sat_thick, 0.0), "saturated thicknesses != 0."

    sat_thick = unstructured_grid.saturated_thickness(
        unstructured_grid.botm - 100.0
    )
    assert np.allclose(sat_thick, 0.0), "saturated thicknesses != 0."


def test_structured_neighbors(structured_grid):
    rook_neighbors = structured_grid.neighbors(1)
    assert np.allclose(rook_neighbors, [0, 2, 4])

    queen_neighbors = structured_grid.neighbors(1, method="queen", reset=True)
    assert np.allclose(queen_neighbors, [0, 3, 4, 2, 5])


def test_vertex_neighbors(vertex_grid):
    rook_neighbors = vertex_grid.neighbors(2)
    assert np.allclose(rook_neighbors, [0, 3, 4])

    queen_neighbors = vertex_grid.neighbors(2, method="queen", reset=True)
    assert np.allclose(queen_neighbors, [0, 1, 3, 4])


def test_unstructured_neighbors(unstructured_grid):
    rook_neighbors = unstructured_grid.neighbors(5)
    assert np.allclose(rook_neighbors, [0, 10, 1, 2, 3, 7, 12])

    queen_neighbors = unstructured_grid.neighbors(
        5, method="queen", reset=True
    )
    assert np.allclose(
        queen_neighbors, [0, 10, 1, 6, 11, 2, 3, 4, 7, 8, 12, 13]
    )


@parametrize_with_cases("grid", cases=GridCases, prefix="structured_cbd")
def test_structured_ncb_thickness(grid):
    thickness = grid.cell_thickness

    assert thickness.shape[0] == grid.nlay + np.count_nonzero(
        grid.laycbd
    ), "grid cell_thickness attribute returns incorrect shape"

    thickness = grid.remove_confining_beds(grid.cell_thickness)
    assert (
        thickness.shape == grid.shape
    ), "quasi3d confining beds not properly removed"

    sat_thick = grid.saturated_thickness(grid.cell_thickness)
    assert (
        sat_thick.shape == grid.shape
    ), "saturated_thickness confining beds not removed"

    assert (
        sat_thick[1, 0, 0] == 20
    ), "saturated_thickness is not properly indexing confining beds"


@parametrize_with_cases("grid", cases=GridCases, prefix="unstructured")
def test_unstructured_iverts(grid):
    iverts = grid.iverts
    assert not any(
        None in l for l in iverts
    ), "None type should not be returned in iverts list"


@parametrize_with_cases("grid", cases=GridCases, prefix="structured")
def test_get_lni_structured(grid):
    for nn in range(0, grid.nnodes):
        layer, i = grid.get_lni([nn])[0]
        assert layer * grid.ncpl + i == nn


@parametrize_with_cases("grid", cases=GridCases, prefix="vertex")
def test_get_lni_vertex(grid):
    for nn in range(0, grid.nnodes):
        layer, i = grid.get_lni([nn])[0]
        assert layer * grid.ncpl + i == nn


@parametrize_with_cases("grid", cases=GridCases, prefix="unstructured")
def test_get_lni_unstructured(grid):
    for nn in range(0, grid.nnodes):
        layer, i = grid.get_lni([nn])[0]
        csum = [0] + list(
            np.cumsum(
                list(grid.ncpl)
                if not isinstance(grid.ncpl, int)
                else [grid.ncpl for _ in range(grid.nlay)]
            )
        )
        assert csum[layer] + i == nn
