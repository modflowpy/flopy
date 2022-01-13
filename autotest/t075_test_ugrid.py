"""
Unstructured grid tests

"""

import os
import shutil

import numpy as np
from flopy.discretization import UnstructuredGrid, VertexGrid
from flopy.utils.triangle import Triangle
from flopy.utils.voronoi import VoronoiGrid
from ci_framework import base_test_dir, FlopyTestSetup

base_dir = base_test_dir(__file__, rel_path="temp", verbose=True)


def test_unstructured_grid_shell():
    # constructor with no arguments.  incomplete shell should exist
    g = UnstructuredGrid()
    assert g.nlay is None
    assert g.nnodes is None
    assert g.ncpl is None
    assert not g.grid_varies_by_layer
    assert not g.is_valid
    assert not g.is_complete
    return


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
    return


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

    return


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

    return


def test_loading_argus_meshes():
    datapth = os.path.join("..", "examples", "data", "unstructured")
    fnames = [fname for fname in os.listdir(datapth) if fname.endswith(".exp")]
    for fname in fnames:
        fname = os.path.join(datapth, fname)
        print(f"Loading Argus mesh ({fname}) into UnstructuredGrid")
        g = UnstructuredGrid.from_argus_export(fname)
        print(f"  Number of nodes: {g.nnodes}")


def test_create_unstructured_grid_from_verts():

    datapth = os.path.join("..", "examples", "data", "unstructured")

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
    return


def test_triangle_unstructured_grid():
    model_ws = f"{base_dir}_test_triangle_unstructured_grid"
    test_setup = FlopyTestSetup(verbose=True, test_dirs=model_ws)

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
        model_ws=model_ws,
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

    return


def test_voronoi_vertex_grid():
    model_ws = f"{base_dir}_test_voronoi_vertex_grid"
    test_setup = FlopyTestSetup(verbose=True, test_dirs=model_ws)

    xmin = 0.0
    xmax = 2.0
    ymin = 0.0
    ymax = 1.0
    area_max = 0.05
    tri = Triangle(maximum_area=area_max, angle=30, model_ws=model_ws)
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

    return


def test_voronoi_grid0(plot=False):
    model_ws = f"{base_dir}_test_voronoi_grid0"
    test_setup = FlopyTestSetup(verbose=True, test_dirs=model_ws)

    name = "vor0"
    answer_ncpl = 3803
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
    area_max = 100.0 ** 2
    tri = Triangle(maximum_area=area_max, angle=30, model_ws=model_ws)
    poly = np.array(domain)
    tri.add_polygon(poly)
    tri.build(verbose=False)

    vor = VoronoiGrid(tri)
    gridprops = vor.get_gridprops_vertexgrid()
    voronoi_grid = VertexGrid(**gridprops, nlay=1)

    if plot:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot()
        ax.set_aspect("equal")
        voronoi_grid.plot(ax=ax)
        plt.savefig(os.path.join(model_ws, f"{name}.png"))

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

    return


def test_voronoi_grid1(plot=False):
    model_ws = f"{base_dir}_test_voronoi_grid1"
    test_setup = FlopyTestSetup(verbose=True, test_dirs=model_ws)

    name = "vor1"
    answer_ncpl = 1679
    xmin = 0.0
    xmax = 2.0
    ymin = 0.0
    ymax = 1.0
    area_max = 0.001
    tri = Triangle(maximum_area=area_max, angle=30, model_ws=model_ws)
    poly = np.array(((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)))
    tri.add_polygon(poly)
    tri.build(verbose=False)

    vor = VoronoiGrid(tri)
    gridprops = vor.get_gridprops_vertexgrid()
    voronoi_grid = VertexGrid(**gridprops, nlay=1)

    if plot:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot()
        ax.set_aspect("equal")
        voronoi_grid.plot(ax=ax)
        plt.savefig(os.path.join(model_ws, f"{name}.png"))

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

    return


def test_voronoi_grid2(plot=False):
    model_ws = f"{base_dir}_test_voronoi_grid2"
    test_setup = FlopyTestSetup(verbose=True, test_dirs=model_ws)

    name = "vor2"
    answer_ncpl = 540
    #theta = np.arange(0.0, 2 * np.pi, 0.2)
    theta = np.linspace(0.0, 2 * np.pi, 35)[:-1]
    radius = 100.0
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    circle_poly = [(x, y) for x, y in zip(x, y)]
    tri = Triangle(maximum_area=50, angle=30, model_ws=model_ws)
    tri.add_polygon(circle_poly)
    tri.build(verbose=True)

    vor = VoronoiGrid(tri)
    gridprops = vor.get_gridprops_vertexgrid()
    voronoi_grid = VertexGrid(**gridprops, nlay=1)

    # Check for success
    success = True
    final_error_message = ""

    # ensure proper number of cells
    ncpl = gridprops["ncpl"]
    errmsg = f"Number of cells should be {answer_ncpl}. Found {ncpl}"
    if ncpl != answer_ncpl:
        final_error_message += errmsg + "\n"
        success = False

    # ensure that all cells have 3 or more points
    ninvalid_cells = []
    for icell, ivts in enumerate(vor.iverts):
        if len(ivts) < 3:
            ninvalid_cells.append(icell)
    errmsg = f"The following cells do not have 3 or more vertices.\n{ninvalid_cells}"
    if len(ninvalid_cells) > 0:
        final_error_message += errmsg + "\n"
        success = False

    plot = True
    if plot:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot()
        ax.set_aspect("equal")
        voronoi_grid.plot(ax=ax)
        plt.savefig(os.path.join(model_ws, f"{name}.png"))

    # copy folder to ./failedTests folder
    test_setup.save_as_artifact()

    assert success, final_error_message
    return


def test_voronoi_grid3(plot=False):
    model_ws = f"{base_dir}_test_voronoi_grid3"
    test_setup = FlopyTestSetup(verbose=True, test_dirs=model_ws)

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

    tri = Triangle(maximum_area=100, angle=30, model_ws=model_ws)
    tri.add_polygon(circle_poly)
    tri.add_polygon(inner_circle_poly)
    tri.add_hole((25, 25))
    tri.build(verbose=False)

    vor = VoronoiGrid(tri)
    gridprops = vor.get_gridprops_vertexgrid()
    voronoi_grid = VertexGrid(**gridprops, nlay=1)

    if plot:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot()
        ax.set_aspect("equal")
        voronoi_grid.plot(ax=ax)
        plt.savefig(os.path.join(model_ws, f"{name}.png"))

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

    return


def test_voronoi_grid4(plot=False):
    model_ws = f"{base_dir}_test_voronoi_grid4"
    test_setup = FlopyTestSetup(verbose=True, test_dirs=model_ws)

    name = "vor4"
    answer_ncpl = 410
    active_domain = [(0, 0), (100, 0), (100, 100), (0, 100)]
    area1 = [(10, 10), (40, 10), (40, 40), (10, 40)]
    area2 = [(60, 60), (80, 60), (80, 80), (60, 80)]
    tri = Triangle(angle=30, model_ws=model_ws)
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

    if plot:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot()
        ax.set_aspect("equal")
        voronoi_grid.plot(ax=ax)
        plt.savefig(os.path.join(model_ws, f"{name}.png"))

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

    return


def test_voronoi_grid5(plot=False):
    model_ws = f"{base_dir}_test_voronoi_grid5"
    test_setup = FlopyTestSetup(verbose=True, test_dirs=model_ws)

    name = "vor5"
    answer_ncpl = 1305
    active_domain = [(0, 0), (100, 0), (100, 100), (0, 100)]
    area1 = [(10, 10), (40, 10), (40, 40), (10, 40)]
    area2 = [(70, 70), (90, 70), (90, 90), (70, 90)]

    tri = Triangle(angle=30, model_ws=model_ws)

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

    if plot:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot()
        ax.set_aspect("equal")
        voronoi_grid.plot(ax=ax)

        # plot invalid cells
        ax.plot(voronoi_grid.xcellcenters[ninvalid_cells], voronoi_grid.ycellcenters[ninvalid_cells], 'ro')

        plt.savefig(os.path.join(model_ws, f"{name}.png"))

    # ensure proper number of cells
    ncpl = gridprops["ncpl"]
    errmsg = f"Number of cells should be {answer_ncpl}. Found {ncpl}"
    assert ncpl == answer_ncpl, errmsg

    # ensure that all cells have 3 or more points
    errmsg = f"The following cells do not have 3 or more vertices.\n{ninvalid_cells}"
    assert len(ninvalid_cells) == 0, errmsg

    return


if __name__ == "__main__":
    # test_unstructured_grid_shell()
    # test_unstructured_grid_dimensions()
    # test_unstructured_minimal_grid()
    # test_unstructured_complete_grid()
    # test_loading_argus_meshes()
    # test_create_unstructured_grid_from_verts()
    # test_triangle_unstructured_grid()
    # test_voronoi_vertex_grid()
    #test_voronoi_grid0(plot=True)
    #test_voronoi_grid1(plot=True)
    test_voronoi_grid2(plot=True)
    #test_voronoi_grid3(plot=True)
    #test_voronoi_grid4(plot=True)
    #test_voronoi_grid5(plot=True)
