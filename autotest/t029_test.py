import pytest
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import flopy

from ci_framework import base_test_dir, FlopyTestSetup

base_dir = base_test_dir(__file__, rel_path="temp", verbose=True)

pthtest = os.path.join("..", "examples", "data", "mfgrd_test")
flowpth = os.path.join("..", "examples", "data", "mf6-freyberg")


def test_mfgrddis_MfGrdFile():
    fn = os.path.join(pthtest, "nwtp3.dis.grb")
    grb = flopy.mf6.utils.MfGrdFile(fn, verbose=True)

    nodes = grb.nodes
    ia = grb.ia
    shape = ia.shape[0]
    assert shape == nodes + 1, f"ia size ({shape}) not equal to {nodes + 1}"

    nnz = ia[-1]
    ja = grb.ja
    shape = ja.shape[0]
    assert shape == nnz, f"ja size ({shape}) not equal to {nnz}"

    modelgrid = grb.modelgrid
    assert isinstance(
        modelgrid, flopy.discretization.StructuredGrid
    ), "invalid grid type"


def test_mfgrddis_modelgrid():
    fn = os.path.join(pthtest, "nwtp3.dis.grb")
    modelgrid = flopy.discretization.StructuredGrid.from_binary_grid_file(
        fn, verbose=True
    )
    assert isinstance(
        modelgrid, flopy.discretization.StructuredGrid
    ), "invalid grid type"

    lc = modelgrid.plot()
    assert isinstance(
        lc, matplotlib.collections.LineCollection
    ), f"could not plot grid object created from {fn}"
    plt.close()

    extents = modelgrid.extent
    errmsg = (
        f"extents {extents} of {fn} does not equal (0.0, 8000.0, 0.0, 8000.0)"
    )
    assert extents == (0.0, 8000.0, 0.0, 8000.0), errmsg

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


def test_mfgrddisv_MfGrdFile():
    fn = os.path.join(pthtest, "flow.disv.grb")
    grb = flopy.mf6.utils.MfGrdFile(fn, verbose=True)

    nodes = grb.nodes
    ia = grb.ia
    shape = ia.shape[0]
    assert shape == nodes + 1, f"ia size ({shape}) not equal to {nodes + 1}"

    nnz = ia[-1]
    ja = grb.ja
    shape = ja.shape[0]
    assert shape == nnz, f"ja size ({shape}) not equal to {nnz}"

    mg = grb.modelgrid
    assert isinstance(
        mg, flopy.discretization.VertexGrid
    ), f"invalid grid type ({type(mg)})"


def test_mfgrddisv_modelgrid():
    fn = os.path.join(pthtest, "flow.disv.grb")
    mg = flopy.discretization.VertexGrid.from_binary_grid_file(
        fn, verbose=True
    )
    assert isinstance(
        mg, flopy.discretization.VertexGrid
    ), f"invalid grid type ({type(mg)})"

    ncpl = 218
    assert mg.ncpl == ncpl, f"ncpl ({mg.ncpl}) does not equal {ncpl}"

    lc = mg.plot()
    assert isinstance(
        lc, matplotlib.collections.LineCollection
    ), f"could not plot grid object created from {fn}"
    plt.close("all")

    extents = mg.extent
    extents0 = (0.0, 700.0, 0.0, 700.0)
    errmsg = f"extents {extents} of {fn} does not equal {extents0}"
    assert extents == extents0, errmsg

    nvert = mg.nvert
    iverts = mg.iverts
    maxvertex = max([max(sublist[1:]) for sublist in iverts])
    assert (
        maxvertex + 1 == nvert
    ), f"nvert ({maxvertex + 1}) does not equal {nvert}"
    verts = mg.verts
    assert nvert == verts.shape[0], (
        f"number of vertex (x, y) pairs ({verts.shape[0]}) "
        f"does not equal {nvert}"
    )

    cellxy = np.column_stack((mg.xyzcellcenters[:2]))
    errmsg = (
        f"shape of flow.disv centroids {cellxy.shape} not equal to (218, 2)."
    )
    assert cellxy.shape == (218, 2), errmsg
    return


def test_mfgrddisu_MfGrdFile():
    fn = os.path.join(pthtest, "keating.disu.grb")
    grb = flopy.mf6.utils.MfGrdFile(fn, verbose=True)

    nodes = grb.nodes
    ia = grb.ia
    shape = ia.shape[0]
    assert shape == nodes + 1, f"ia size ({shape}) not equal to {nodes + 1}"

    nnz = ia[-1]
    ja = grb.ja
    shape = ja.shape[0]
    assert shape == nnz, f"ja size ({shape}) not equal to {nnz}"

    mg = grb.modelgrid
    assert isinstance(
        mg, flopy.discretization.UnstructuredGrid
    ), f"invalid grid type ({type(mg)})"


def test_mfgrddisu_modelgrid_fail():
    fn = os.path.join(pthtest, "flow.disu.grb")
    with pytest.raises(TypeError):
        mg = flopy.discretization.UnstructuredGrid.from_binary_grid_file(
            fn, verbose=True
        )


def test_mfgrddisu_modelgrid():
    fn = os.path.join(pthtest, "keating.disu.grb")
    mg = flopy.discretization.UnstructuredGrid.from_binary_grid_file(
        fn, verbose=True
    )
    assert isinstance(
        mg, flopy.discretization.UnstructuredGrid
    ), f"invalid grid type ({type(mg)})"

    lc = mg.plot()
    assert isinstance(
        lc, matplotlib.collections.LineCollection
    ), f"could not plot grid object created from {fn}"
    plt.close("all")

    extents = mg.extent
    extents0 = (0.0, 10000.0, 0.0, 1.0)
    errmsg = f"extents {extents} of {fn} does not equal {extents0}"
    assert extents == extents0, errmsg

    nvert = mg.nvert
    iverts = mg.iverts
    maxvertex = max([max(sublist[1:]) for sublist in iverts])
    assert (
        maxvertex + 1 == nvert
    ), f"nvert ({maxvertex + 1}) does not equal {nvert}"
    verts = mg.verts
    assert nvert == verts.shape[0], (
        f"number of vertex (x, y) pairs ({verts.shape[0]}) "
        f"does not equal {nvert}"
    )

    return


def test_faceflows():
    model_ws = f"{base_dir}_test_faceflows"
    test_setup = FlopyTestSetup(verbose=True, test_dirs=model_ws)

    sim = flopy.mf6.MFSimulation.load(
        sim_name="freyberg",
        exe_name="mf6",
        sim_ws=flowpth,
    )

    # change the simulation workspace
    sim.set_sim_path(model_ws)

    # write the model simulation files
    sim.write_simulation()

    # run the simulation
    sim.run_simulation()

    # get output
    gwf = sim.get_model("freyberg")
    head = gwf.output.head().get_data()
    cbc = gwf.output.budget()

    spdis = cbc.get_data(text="DATA-SPDIS")[0]
    flowja = cbc.get_data(text="FLOW-JA-FACE")[0]

    frf, fff, flf = flopy.mf6.utils.get_structured_faceflows(
        flowja,
        grb_file=os.path.join(model_ws, "freyberg.dis.grb"),
    )
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

    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    ax = fig.add_subplot(1, 3, 1, aspect="equal")
    mm = flopy.plot.PlotMapView(model=gwf, ax=ax)
    Q0 = mm.plot_vector(Qx, Qy)
    assert isinstance(
        Q0, matplotlib.quiver.Quiver
    ), "Q0 not type matplotlib.quiver.Quiver"

    ax = fig.add_subplot(1, 3, 2, aspect="equal")
    mm = flopy.plot.PlotMapView(model=gwf, ax=ax)
    q0 = mm.plot_vector(sqx, sqy)
    assert isinstance(
        q0, matplotlib.quiver.Quiver
    ), "q0 not type matplotlib.quiver.Quiver"

    ax = fig.add_subplot(1, 3, 3, aspect="equal")
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


def test_flowja_residuals():
    model_ws = f"{base_dir}_test_flowja_residuals"
    test_setup = FlopyTestSetup(verbose=True, test_dirs=model_ws)

    sim = flopy.mf6.MFSimulation.load(
        sim_name="freyberg",
        exe_name="mf6",
        sim_ws=flowpth,
    )

    # change the simulation workspace
    sim.set_sim_path(model_ws)

    # write the model simulation files
    sim.write_simulation()

    # run the simulation
    sim.run_simulation()

    # get output
    gwf = sim.get_model("freyberg")
    grb_file = os.path.join(model_ws, "freyberg.dis.grb")
    cbc = gwf.output.budget()

    spdis = cbc.get_data(text="DATA-SPDIS")[0]
    flowja = cbc.get_data(text="FLOW-JA-FACE")[0]

    residual = flopy.mf6.utils.get_residuals(flowja, grb_file=grb_file)
    qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(spdis, gwf)

    fig = plt.figure(figsize=(6, 9), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1, aspect="equal")
    mm = flopy.plot.PlotMapView(model=gwf, ax=ax)
    r0 = mm.plot_array(residual)
    assert isinstance(
        r0, matplotlib.collections.QuadMesh
    ), "r0 not type matplotlib.collections.QuadMesh"
    q0 = mm.plot_vector(qx, qy)
    assert isinstance(
        q0, matplotlib.quiver.Quiver
    ), "q0 not type matplotlib.quiver.Quiver"
    mm.plot_grid(lw=0.5, color="black")
    mm.plot_ibound()
    plt.colorbar(r0, shrink=0.5)
    plt.title("Cell Residual, cubic meters per second")

    plt.close("all")
    return


def test_structured_faceflows_3d():
    model_ws = f"{base_dir}_test_faceflows_3d"
    test_setup = FlopyTestSetup(verbose=True, test_dirs=model_ws)
    name = 'mymodel'
    sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=model_ws, exe_name='mf6')
    tdis = flopy.mf6.ModflowTdis(sim)
    ims = flopy.mf6.ModflowIms(sim)
    gwf = flopy.mf6.ModflowGwf(sim, modelname=name, save_flows=True)
    dis = flopy.mf6.ModflowGwfdis(gwf, nlay=3, nrow=10, ncol=10, top=0, botm=[-1, -2, -3])
    ic = flopy.mf6.ModflowGwfic(gwf)
    npf = flopy.mf6.ModflowGwfnpf(gwf, save_specific_discharge=True)
    chd = flopy.mf6.ModflowGwfchd(gwf, stress_period_data=[[(0, 0, 0), 1.],
                                                           [(0, 9, 9), 0.]])
    budget_file = name + '.bud'
    head_file = name + '.hds'
    oc = flopy.mf6.ModflowGwfoc(gwf,
                                budget_filerecord=budget_file,
                                head_filerecord=head_file,
                                saverecord=[('HEAD', 'ALL'), ('BUDGET', 'ALL')])
    sim.write_simulation()
    sim.run_simulation()

    head = gwf.output.head().get_data()
    bud = gwf.output.budget()
    flowja = bud.get_data(text="FLOW-JA-FACE")[0]
    frf, fff, flf = flopy.mf6.utils.get_structured_faceflows(
        flowja,
        grb_file=os.path.join(model_ws, "mymodel.dis.grb"),
    )
    assert frf.shape == head.shape, f"frf.shape {frf.shape} != head.shape {head.shape}"
    assert fff.shape == head.shape, f"frf.shape {frf.shape} != head.shape {head.shape}"
    assert flf.shape == head.shape, f"frf.shape {frf.shape} != head.shape {head.shape}"
    return


def test_faceflows_empty():
    flowja = np.zeros(10, dtype=np.float64)
    with pytest.raises(ValueError):
        frf, fff, flf = flopy.mf6.utils.get_structured_faceflows(flowja)


def test_faceflows_jaempty():
    flowja = np.zeros(10, dtype=np.float64)
    ia = np.zeros(10, dtype=np.int32)
    with pytest.raises(ValueError):
        frf, fff, flf = flopy.mf6.utils.get_structured_faceflows(flowja, ia=ia)


def test_faceflows_iaempty():
    flowja = np.zeros(10, dtype=np.float64)
    ja = np.zeros(10, dtype=np.int32)
    with pytest.raises(ValueError):
        _v = flopy.mf6.utils.get_structured_faceflows(flowja, ja=ja)


def test_faceflows_flowja_size():
    flowja = np.zeros(10, dtype=np.float64)
    ia = np.zeros(5, dtype=np.int32)
    ja = np.zeros(5, dtype=np.int32)
    with pytest.raises(ValueError):
        _v = flopy.mf6.utils.get_structured_faceflows(flowja, ia=ia, ja=ja)


def test_residuals_jaempty():
    flowja = np.zeros(10, dtype=np.float64)
    ia = np.zeros(10, dtype=np.int32)
    with pytest.raises(ValueError):
        _v = flopy.mf6.utils.get_residuals(flowja, ia=ia)


def test_residuals_iaempty():
    flowja = np.zeros(10, dtype=np.float64)
    ja = np.zeros(10, dtype=np.int32)
    with pytest.raises(ValueError):
        _v = flopy.mf6.utils.get_residuals(flowja, ja=ja)


if __name__ == "__main__":
    # test_mfgrddis_MfGrdFile()
    # test_mfgrddis_modelgrid()
    # test_mfgrddisv_MfGrdFile()
    # test_mfgrddisv_modelgrid()
    # test_mfgrddisu_MfGrdFile()
    # test_mfgrddisu_modelgrid()
    # test_faceflows()
    test_structured_faceflows_3d()
    # test_flowja_residuals()
