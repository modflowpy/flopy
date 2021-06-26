import os
import flopy
import matplotlib
import matplotlib.pyplot as plt

pthtest = os.path.join("..", "examples", "data", "mfgrd_test")


def test_mfgrddis():
    fn = os.path.join(pthtest, "nwtp3.dis.grb")
    grid = flopy.mf6.utils.MfGrdFile(fn, verbose=True)

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

    mg = grid.get_modelgrid
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

    mg = grid.get_modelgrid
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

    connections = grid.connectivity
    errmsg = "number of connections ({}) is not equal to {}".format(
        len(connections), grid.nodes
    )
    assert len(connections) == grid.nodes, errmsg

    mg = grid.get_modelgrid
    assert mg is None, "model grid is not None"

    fn = os.path.join(pthtest, "keating.disu.grb")
    grid = flopy.mf6.utils.MfGrdFile(fn, verbose=True)

    connections = grid.connectivity
    errmsg = "number of connections ({}) is not equal to {}".format(
        len(connections), grid.nodes
    )
    assert len(connections) == grid.nodes, errmsg

    mg = grid.get_modelgrid
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
