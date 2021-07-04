import os
import shutil

import numpy as np

import flopy
import matplotlib
import matplotlib.pyplot as plt

pthtest = os.path.join("..", "examples", "data", "mfgrd_test")
flowpth = os.path.join("..", "examples", "data", "mf6-freyberg")

tpth = os.path.join("temp", "t029")
# remove the directory if it exists
if os.path.isdir(tpth):
    shutil.rmtree(tpth)
# make the directory
os.makedirs(tpth)


def test_mfgrddis():
    fn = os.path.join(pthtest, "nwtp3.dis.grb")
    grid = flopy.mf6.utils.MfGrdFile(fn, verbose=True)

    ncpl = grid.ncol * grid.nrow
    assert grid.ncpl == ncpl, "ncpl ({}) does not equal {}".format(
        grid.ncpl, ncpl
    )

    nvert = grid.modelgrid.nvert
    iverts = grid.modelgrid.iverts
    maxvertex = max([max(sublist[1:]) for sublist in iverts])
    assert maxvertex + 1 == nvert, "nvert ({}) does not equal {}".format(
        maxvertex + 1, nvert
    )
    verts = grid.modelgrid.verts
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
    assert grid.modelgrid.ncpl == ncpl, "ncpl ({}) does not equal {}".format(
        grid.modelgrid.ncpl, ncpl
    )

    nvert = grid.modelgrid.nvert
    iverts = grid.modelgrid.iverts
    maxvertex = max([max(sublist[1:]) for sublist in iverts])
    assert maxvertex + 1 == nvert, "nvert ({}) does not equal {}".format(
        maxvertex + 1, nvert
    )
    verts = grid.modelgrid.verts
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

    cellxy = grid.xycentroids
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

    mg = grid.modelgrid
    assert mg is None, "model grid is not None"

    fn = os.path.join(pthtest, "keating.disu.grb")
    grid = flopy.mf6.utils.MfGrdFile(fn, verbose=True)

    nvert = grid.modelgrid.nvert
    iverts = grid.modelgrid.iverts
    maxvertex = max([max(sublist[1:]) for sublist in iverts])
    assert maxvertex + 1 == nvert, "nvert ({}) does not equal {}".format(
        maxvertex + 1, nvert
    )
    verts = grid.modelgrid.verts
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


def test_faceflows():
    sim = flopy.mf6.MFSimulation.load(
        sim_name="freyberg",
        exe_name="mf6",
        sim_ws=flowpth,
    )

    # change the simulation workspace
    sim.set_sim_path(tpth)

    # write the model simulation files
    sim.write_simulation()

    # run the simulation
    sim.run_simulation()

    # load the grid data
    fpth = os.path.join(tpth, "freyberg.dis.grb")
    grid = flopy.mf6.utils.MfGrdFile(fpth, verbose=True)

    # get output
    gwf = sim.get_model("gwf_1")
    head = gwf.oc.output.head().get_data()
    cbc = gwf.oc.output.budget()

    spdis = cbc.get_data(text="DATA-SPDIS")[0]
    flowja = cbc.get_data(text="FLOW-JA-FACE")[0]

    frf, fff, flf = grid.get_faceflows(flowja)
    Qx, Qy, Qz = flopy.utils.postprocessing.get_specific_discharge(
        (frf, fff, flf),
        gwf,
    )
    sqx, sqy, sqz = flopy.utils.postprocessing.get_specific_discharge(
        (frf, fff, flf),
        gwf,
        head=head,
    )
    qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(spdis, gwf)

    fig = plt.figure(figsize=(12, 12), constrained_layout=True)
    ax = fig.add_subplot(1, 3, 1)
    mm = flopy.plot.PlotMapView(model=gwf, ax=ax)
    Q0 = mm.plot_vector(Qx, Qy)
    assert isinstance(
        Q0, matplotlib.quiver.Quiver
    ), "Q0 not type matplotlib.quiver.Quiver"

    ax = fig.add_subplot(1, 3, 2)
    mm = flopy.plot.PlotMapView(model=gwf, ax=ax)
    q0 = mm.plot_vector(sqx, sqy)
    assert isinstance(
        q0, matplotlib.quiver.Quiver
    ), "q0 not type matplotlib.quiver.Quiver"

    ax = fig.add_subplot(1, 3, 3)
    mm = flopy.plot.PlotMapView(model=gwf, ax=ax)
    q1 = mm.plot_vector(qx, qy)
    assert isinstance(
        q1, matplotlib.quiver.Quiver
    ), "q1 not type matplotlib.quiver.Quiver"

    plt.close("all")

    # uv0 = np.column_stack((q0.U, q0.V))
    # uv1 = np.column_stack((q1.U, q1.V))
    # diff = uv1 - uv0
    # assert (
    #     np.allclose(uv0, uv1)
    # ), "get_faceflows quivers are not equal to specific discharge vectors"

    return


if __name__ == "__main__":
    # test_mfgrddis()
    # test_mfgrddisv()
    test_mfgrddisu()
    # test_faceflows()
