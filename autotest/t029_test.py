import os

import numpy as np

import flopy
import matplotlib
import matplotlib.pyplot as plt

pthtest = os.path.join("..", "examples", "data", "mfgrd_test")


def test_mfgrddis():
    fn = os.path.join(pthtest, "nwtp3.dis.grb")
    grid = flopy.mf6.utils.MfGrdFile(fn, verbose=True)

    ncpl = grid.ncol * grid.nrow
    assert grid.ncpl == ncpl, "ncpl ({}) does not equal {}".format(
        grid.ncpl, ncpl
    )

    iverts, verts = grid.get_vertices()
    nvert = grid.nvert
    node = np.array(iverts, dtype=int).max()
    assert node + 1 == nvert, "nvert ({}) does not equal {}".format(
        node + 1, nvert
    )

    assert (
        nvert == verts.shape[0]
    ), "number of vertex (x, y) pairs ({}) ".format(
        verts.shape[0]
    ) + "does not equal {}".format(
        nvert
    )

    connections = grid.connectivity
    errmsg = "number of connections ({}) is not equal to {}".format(
        len(connections), grid.nodes
    )
    assert len(connections) == grid.nodes, errmsg

    cell_conn = grid.cellconnections
    errmsg = "number of cell connections ({}) is not equal to {}".format(
        cell_conn.shape[0], grid.nconnections
    )
    assert cell_conn.shape[0] == grid.nconnections, errmsg

    mg = grid.modelgrid
    assert isinstance(
        mg, flopy.discretization.StructuredGrid
    ), "invalid grid type"

    lc = mg.plot()
    assert isinstance(
        lc, matplotlib.collections.LineCollection
    ), "could not plot grid object created from {}".format(fn)
    plt.close()

    extents = mg.extent
    errmsg = (
        "extents {} of {} ".format(extents, fn)
        + "does not equal (0.0, 8000.0, 0.0, 8000.0)"
    )
    assert extents == (0.0, 8000.0, 0.0, 8000.0), errmsg


def test_mfgrddisv():
    fn = os.path.join(pthtest, "flow.disv.grb")
    grid = flopy.mf6.utils.MfGrdFile(fn, verbose=True)

    ncpl = 218
    assert grid.ncpl == ncpl, "ncpl ({}) does not equal {}".format(
        grid.ncpl, ncpl
    )

    iverts, verts = grid.get_vertices()
    nvert = grid.nvert
    node = max([max(sublist) for sublist in iverts])
    assert node + 1 == nvert, "nvert ({}) does not equal {}".format(
        node + 1, nvert
    )

    assert (
        nvert == verts.shape[0]
    ), "number of vertex (x, y) pairs ({}) ".format(
        verts.shape[0]
    ) + "does not equal {}".format(
        nvert
    )

    connections = grid.connectivity
    errmsg = "number of connections ({}) is not equal to {}".format(
        len(connections), grid.nodes
    )
    assert len(connections) == grid.nodes, errmsg

    cell_conn = grid.cellconnections
    errmsg = "number of cell connections ({}) is not equal to {}".format(
        cell_conn.shape[0], grid.nconnections
    )
    assert cell_conn.shape[0] == grid.nconnections, errmsg

    mg = grid.modelgrid
    assert isinstance(
        mg, flopy.discretization.VertexGrid
    ), "invalid grid type ({})".format(type(mg))

    lc = mg.plot()
    assert isinstance(
        lc, matplotlib.collections.LineCollection
    ), "could not plot grid object created from {}".format(fn)
    plt.close("all")

    extents = mg.extent
    extents0 = (0.0, 700.0, 0.0, 700.0)
    errmsg = "extents {} of {} ".format(
        extents, fn
    ) + "does not equal {}".format(extents0)
    assert extents == extents0, errmsg

    cellxy = grid.get_centroids
    errmsg = "shape of flow.disv centroids {} not equal to (218, 2).".format(
        cellxy.shape
    )
    assert cellxy.shape == (218, 2), errmsg
    return


def test_mfgrddisu():
    fn = os.path.join(pthtest, "flow.disu.grb")
    grid = flopy.mf6.utils.MfGrdFile(fn, verbose=True)

    iverts, verts = grid.get_vertices()
    assert iverts is None, "iverts and verts should be None for {}".format(fn)

    connections = grid.connectivity
    errmsg = "number of connections ({}) is not equal to {}".format(
        len(connections), grid.nodes
    )
    assert len(connections) == grid.nodes, errmsg

    mg = grid.modelgrid
    assert mg is None, "model grid is not None"

    fn = os.path.join(pthtest, "keating.disu.grb")
    grid = flopy.mf6.utils.MfGrdFile(fn, verbose=True)

    iverts, verts = grid.get_vertices()
    nvert = grid.nvert
    node = max([max(sublist) for sublist in iverts])
    assert node + 1 == nvert, "nvert ({}) does not equal {}".format(
        node + 1, nvert
    )

    assert (
        nvert == len(verts)
    ), "number of vertex (x, y) pairs ({}) ".format(
        len(verts)
    ) + "does not equal {}".format(
        nvert
    )

    connections = grid.connectivity
    errmsg = "number of connections ({}) is not equal to {}".format(
        len(connections), grid.nodes
    )
    assert len(connections) == grid.nodes, errmsg

    mg = grid.modelgrid
    assert isinstance(
        mg, flopy.discretization.UnstructuredGrid
    ), "invalid grid type ({})".format(type(mg))

    lc = mg.plot()
    assert isinstance(
        lc, matplotlib.collections.LineCollection
    ), "could not plot grid object created from {}".format(fn)
    plt.close("all")

    extents = mg.extent
    extents0 = (0.0, 10000.0, 0.0, 1.0)
    errmsg = "extents {} of {} ".format(
        extents, fn
    ) + "does not equal {}".format(extents0)
    assert extents == extents0, errmsg

    return


if __name__ == "__main__":
    test_mfgrddis()
    test_mfgrddisv()
    test_mfgrddisu()
