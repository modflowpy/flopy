import os

import numpy as np
import pandas as pd
import pytest
from flaky import flaky
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.collections import (
    LineCollection,
    PatchCollection,
    PathCollection,
    QuadMesh,
)
from modflow_devtools.markers import requires_exe, requires_pkg

import flopy
from flopy.discretization import StructuredGrid
from flopy.mf6 import MFSimulation
from flopy.modflow import (
    Modflow,
    ModflowBas,
    ModflowDis,
    ModflowLpf,
    ModflowOc,
    ModflowPcg,
    ModflowWel,
)
from flopy.modpath import Modpath6, Modpath6Bas
from flopy.plot import PlotCrossSection, PlotMapView
from flopy.utils import CellBudgetFile, EndpointFile, HeadFile, PathlineFile


@pytest.fixture
def rng():
    # set seed so parametrized plot tests are comparable
    return np.random.default_rng(0)


@requires_pkg("shapely")
def test_map_view():
    m = flopy.modflow.Modflow(rotation=20.0)
    dis = flopy.modflow.ModflowDis(
        m, nlay=1, nrow=40, ncol=20, delr=250.0, delc=250.0, top=10, botm=0
    )
    # transformation assigned by arguments
    xll, yll, rotation = 500000.0, 2934000.0, 45.0

    def check_vertices():
        xllp, yllp = pc._paths[0].vertices[0]
        assert np.abs(xllp - xll) < 1e-6
        assert np.abs(yllp - yll) < 1e-6

    m.modelgrid.set_coord_info(xoff=xll, yoff=yll, angrot=rotation)
    modelmap = flopy.plot.PlotMapView(model=m)
    pc = modelmap.plot_grid()
    check_vertices()

    modelmap = flopy.plot.PlotMapView(modelgrid=m.modelgrid)
    pc = modelmap.plot_grid()
    check_vertices()

    mf = flopy.modflow.Modflow()

    # Model domain and grid definition
    dis = flopy.modflow.ModflowDis(
        mf,
        nlay=1,
        nrow=10,
        ncol=20,
        delr=1.0,
        delc=1.0,
    )
    xul, yul = 100.0, 210.0
    mg = mf.modelgrid
    mf.modelgrid.set_coord_info(
        xoff=mg._xul_to_xll(xul, 0.0), yoff=mg._yul_to_yll(yul, 0.0)
    )
    verts = [[101.0, 201.0], [119.0, 209.0]]
    modelxsect = flopy.plot.PlotCrossSection(model=mf, line={"line": verts})
    patchcollection = modelxsect.plot_grid()


@pytest.mark.mf6
@pytest.mark.xfail(reason="sometimes get wrong collection type")
def test_map_view_bc_gwfs_disv(example_data_path):
    mpath = example_data_path / "mf6" / "test003_gwfs_disv"
    sim = MFSimulation.load(sim_ws=mpath)
    ml6 = sim.get_model("gwf_1")
    ml6.modelgrid.set_coord_info(angrot=-14)
    mapview = flopy.plot.PlotMapView(model=ml6)
    mapview.plot_bc("CHD")
    ax = mapview.ax

    if len(ax.collections) == 0:
        raise AssertionError("Boundary condition was not drawn")

    for col in ax.collections:
        assert isinstance(col, (QuadMesh, PathCollection)), (
            f"Unexpected collection type: {type(col)}"
        )


@pytest.mark.mf2005
@pytest.mark.xfail(reason="sometimes get wrong collection type")
def test_map_view_bc_freyberg_subset(example_data_path):
    mpath = example_data_path / "freyberg"
    name_file = "freyberg.nam"
    ml = Modflow.load(name_file, model_ws=mpath, verbose=True)
    mapview = flopy.plot.PlotMapView(model=ml)
    mapview.plot_bc(
        "RIV",
        subset=[
            (0, 34, 14),
            (0, 35, 14),
            (0, 36, 14),
            (0, 37, 14),
            (0, 38, 14),
            (0, 39, 14),
        ],
    )
    ax = mapview.ax

    if len(ax.collections) == 0:
        raise AssertionError("Boundary condition was not drawn")

    for col in ax.collections:
        assert isinstance(col, (QuadMesh, PathCollection)), (
            f"Unexpected collection type: {type(col)}"
        )
        if isinstance(col, QuadMesh):
            count = col.get_array().count()
            assert count == 6, f"More than six river cells plotted ({count})"

    # plt.show(block=True)


@pytest.mark.mf2005
@pytest.mark.xfail(reason="sometimes get wrong collection type")
def test_map_view_bc_freyberg_ml_subset_plotAll(example_data_path):
    mpath = example_data_path / "freyberg_multilayer_transient"
    name_file = "freyberg.nam"
    ml = Modflow.load(name_file, model_ws=mpath, verbose=True)
    mapview = flopy.plot.PlotMapView(model=ml, layer=2)
    mapview.plot_bc(
        "WEL",
        plotAll=True,
        subset=[
            (0, 8, 15),
            (0, 28, 5),
        ],
    )

    ax = mapview.ax

    if len(ax.collections) == 0:
        raise AssertionError("Boundary condition was not drawn")

    for col in ax.collections:
        assert isinstance(col, (QuadMesh, PathCollection, LineCollection)), (
            f"Unexpected collection type: {type(col)}"
        )
        if isinstance(col, QuadMesh):
            count = col.get_array().count()
            assert count == 2, f"More than two wells plotted ({count})"

    # plt.show(block=True)


@pytest.mark.mf6
@pytest.mark.xfail(reason="sometimes get wrong collection type")
def test_map_view_bc_gwfs_disv_subset(example_data_path):
    mpath = example_data_path / "mf6" / "test003_gwfs_disv"
    sim = MFSimulation.load(sim_ws=mpath)
    ml6 = sim.get_model("gwf_1")
    ml6.modelgrid.set_coord_info(angrot=-14)
    mapview = flopy.plot.PlotMapView(model=ml6)
    mapview.plot_bc(
        "CHD", subset=[(0, 10), (0, 30), (0, 50), (0, 70), (0, 90), (0, 49)]
    )
    ax = mapview.ax

    if len(ax.collections) == 0:
        raise AssertionError("Boundary condition was not drawn")

    for col in ax.collections:
        assert isinstance(col, (QuadMesh, PathCollection)), (
            f"Unexpected collection type: {type(col)}"
        )

    # plt.show(block=True)


@pytest.mark.mf6
@pytest.mark.xfail(reason="sometimes get wrong collection type")
def test_map_view_bc_lake2tr(example_data_path):
    mpath = example_data_path / "mf6" / "test045_lake2tr"
    sim = MFSimulation.load(sim_ws=mpath)
    ml6 = sim.get_model("lakeex2a")
    mapview = flopy.plot.PlotMapView(model=ml6)
    mapview.plot_bc("LAK")
    mapview.plot_bc("SFR")

    ax = mapview.ax
    if len(ax.collections) == 0:
        raise AssertionError("Boundary condition was not drawn")

    for col in ax.collections:
        assert isinstance(col, (QuadMesh, PathCollection)), (
            f"Unexpected collection type: {type(col)}"
        )


@pytest.mark.mf6
@pytest.mark.xfail(reason="sometimes get wrong collection type")
def test_map_view_bc_2models_mvr(example_data_path):
    mpath = example_data_path / "mf6" / "test006_2models_mvr"
    sim = MFSimulation.load(sim_ws=mpath)
    ml6 = sim.get_model("parent")
    ml6c = sim.get_model("child")
    ml6c.modelgrid.set_coord_info(xoff=700, yoff=0, angrot=0)

    mapview = flopy.plot.PlotMapView(model=ml6)
    mapview.plot_bc("MAW")
    ax = mapview.ax

    assert len(ax.collections) > 0, "Boundary condition was not drawn"

    mapview2 = flopy.plot.PlotMapView(model=ml6c, ax=mapview.ax)
    mapview2.plot_bc("MAW")
    ax = mapview2.ax

    assert len(ax.collections) > 0, "Boundary condition was not drawn"

    for col in ax.collections:
        assert isinstance(col, (QuadMesh, PathCollection)), (
            f"Unexpected collection type: {type(col)}"
        )


@pytest.mark.mf6
@pytest.mark.xfail(reason="sometimes get wrong collection type")
def test_map_view_bc_UZF_3lay(example_data_path):
    mpath = example_data_path / "mf6" / "test001e_UZF_3lay"
    sim = MFSimulation.load(sim_ws=mpath)
    ml6 = sim.get_model("gwf_1")

    mapview = flopy.plot.PlotMapView(model=ml6)
    mapview.plot_bc("UZF")
    ax = mapview.ax

    if len(ax.collections) == 0:
        raise AssertionError("Boundary condition was not drawn")

    for col in ax.collections:
        assert isinstance(col, (QuadMesh, PathCollection)), (
            f"Unexpected collection type: {type(col)}"
        )


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_map_view_contour_array_structured(function_tmpdir, ndim, rng):
    nlay, nrow, ncol = 3, 10, 10
    ncpl = nrow * ncol
    delc = np.array([10] * nrow, dtype=float)
    delr = np.array([8] * ncol, dtype=float)
    top = np.ones((nrow, ncol), dtype=float)
    botm = np.ones((nlay, nrow, ncol), dtype=float)
    botm[0] = 0.75
    botm[1] = 0.5
    botm[2] = 0.25
    idomain = np.ones((nlay, nrow, ncol))
    idomain[0, 0, :] = 0

    grid = StructuredGrid(
        delc=delc,
        delr=delr,
        top=top,
        botm=botm,
        idomain=idomain,
        lenuni=1,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
    )

    # define full grid 1D array to contour
    arr = rng.random(nlay * nrow * ncol) * 100

    for l in range(nlay):
        if ndim == 1:
            # full grid 1D array
            pmv = PlotMapView(modelgrid=grid, layer=l)
            contours = pmv.contour_array(a=arr)
            fname = f"map_view_contour_{ndim}d_l{l}_full.png"
            plt.savefig(function_tmpdir / fname)
            plt.clf()

            # 1 layer slice
            pmv = PlotMapView(modelgrid=grid, layer=l)
            contours = pmv.contour_array(a=arr[(l * ncpl) : ((l + 1) * ncpl)])
            fname = f"map_view_contour_{ndim}d_l{l}_1lay.png"
            plt.savefig(function_tmpdir / fname)
            plt.clf()
        elif ndim == 2:
            # 1 layer as 2D
            pmv = PlotMapView(modelgrid=grid, layer=l)
            contours = pmv.contour_array(a=arr.reshape(nlay, nrow, ncol)[l, :, :])
            plt.savefig(function_tmpdir / f"map_view_contour_{ndim}d_l{l}.png")
            plt.clf()
        elif ndim == 3:
            # full grid as 3D
            pmv = PlotMapView(modelgrid=grid, layer=l)
            contours = pmv.contour_array(a=arr.reshape(nlay, nrow, ncol))
            plt.savefig(function_tmpdir / f"map_view_contour_{ndim}d_l{l}.png")
            plt.clf()

    # if we ever revert from standard contours to tricontours, restore this nan check
    # vmin = np.nanmin(arr)
    # vmax = np.nanmax(arr)
    # levels = np.linspace(vmin, vmax, 7)
    # for ix, lev in enumerate(contours.levels):
    #     if not np.allclose(lev, levels[ix]):
    #         raise AssertionError("TriContour NaN catch Failed")


def test_plot_limits():
    xymin, xymax = 0, 1000
    cellsize = 50
    nrow = (xymax - xymin) // cellsize
    ncol = nrow
    nlay = 1

    delc = np.full((nrow,), cellsize)
    delr = np.full((ncol,), cellsize)

    top = np.full((nrow, ncol), 100)
    botm = np.full((nlay, nrow, ncol), 0)
    idomain = np.ones(botm.shape, dtype=int)

    grid = flopy.discretization.StructuredGrid(
        delc=delc, delr=delr, top=top, botm=botm, idomain=idomain
    )

    fig, ax = plt.subplots()
    user_extent = 0, 300, 0, 100
    ax.axis(user_extent)

    pmv = flopy.plot.PlotMapView(modelgrid=grid, ax=ax)
    pmv.plot_grid()

    lims = ax.axes.viewLim
    if (lims.x0, lims.x1, lims.y0, lims.y1) != user_extent:
        raise AssertionError("PlotMapView not checking for user scaling")

    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 8))
    pmv = flopy.plot.PlotMapView(modelgrid=grid, ax=ax)
    pmv.plot_grid()

    lims = ax.axes.viewLim
    if (lims.x0, lims.x1, lims.y0, lims.y1) != pmv.extent:
        raise AssertionError("PlotMapView auto extent setting not working")

    plt.close(fig)


@requires_pkg("shapely")
def test_plot_centers():
    nlay = 1
    nrow = 10
    ncol = 10

    delc = np.ones((nrow,))
    delr = np.ones((ncol,))
    top = np.ones((nrow, ncol))
    botm = np.zeros((nlay, nrow, ncol))
    idomain = np.ones(botm.shape, dtype=int)

    idomain[0, :, 0:3] = 0
    active_cells = np.count_nonzero(idomain)

    grid = flopy.discretization.StructuredGrid(
        delc=delc, delr=delr, top=top, botm=botm, idomain=idomain
    )

    xcenters = grid.xcellcenters.ravel()
    ycenters = grid.ycellcenters.ravel()
    xycenters = list(zip(xcenters, ycenters))

    pmv = flopy.plot.PlotMapView(modelgrid=grid)
    pc = pmv.plot_centers()
    if not isinstance(pc, PathCollection):
        raise AssertionError("plot_centers() not returning PathCollection object")

    verts = pc._offsets
    if not verts.shape[0] == active_cells:
        raise AssertionError("plot_centers() not properly masking inactive cells")

    for vert in verts:
        vert = tuple(vert)
        if vert not in xycenters:
            raise AssertionError("center location not properly plotted")


@pytest.fixture(params=["dis", "disv", "disu"])
def hfb_model(request):
    """Create a MODFLOW 6 model with HFB for testing different grid types.

    Parameters
    ----------
    request.param : str
        Grid type: "dis" (structured), "disv" (vertex), or "disu" (unstructured)

    Returns
    -------
    tuple
        (gwf_model, expected_barrier_count, grid_type)
    """
    from flopy.utils.gridutil import get_disu_kwargs, get_disv_kwargs

    grid_type = request.param

    # Create simulation
    sim = flopy.mf6.MFSimulation(sim_name=f"test_hfb_{grid_type}")
    tdis = flopy.mf6.ModflowTdis(sim)
    ims = flopy.mf6.ModflowIms(sim)
    gwf = flopy.mf6.ModflowGwf(sim, modelname="test")

    # Create discretization based on grid type
    if grid_type == "dis":
        # Structured grid
        dis = flopy.mf6.ModflowGwfdis(
            gwf,
            nlay=1,
            nrow=10,
            ncol=10,
            delr=100.0,
            delc=100.0,
        )
        # HFB cellids for structured grid: (layer, row, col)
        hfb_data = [
            [(0, 3, 4), (0, 3, 5), 1e-6],
            [(0, 4, 4), (0, 4, 5), 1e-6],
            [(0, 5, 4), (0, 5, 5), 1e-6],
            [(0, 6, 4), (0, 6, 5), 1e-6],
            [(0, 7, 4), (0, 7, 5), 1e-6],
        ]
        chd_spd = [[(0, 0, 0), 100.0], [(0, 9, 9), 95.0]]
        expected_barriers = 5

    elif grid_type == "disv":
        # Vertex grid (regular grid converted to vertex format)
        disv_kwargs = get_disv_kwargs(1, 10, 10, 100.0, 100.0, 100.0, [0.0])
        disv = flopy.mf6.ModflowGwfdisv(gwf, **disv_kwargs)
        # HFB cellids for vertex grid: (layer, cell2d_id)
        # For a 10x10 grid, cell2d IDs are row*ncol + col
        # Barriers between cells in column 4 and 5 for rows 3-7
        hfb_data = [
            [(0, 3 * 10 + 4), (0, 3 * 10 + 5), 1e-6],
            [(0, 4 * 10 + 4), (0, 4 * 10 + 5), 1e-6],
            [(0, 5 * 10 + 4), (0, 5 * 10 + 5), 1e-6],
            [(0, 6 * 10 + 4), (0, 6 * 10 + 5), 1e-6],
            [(0, 7 * 10 + 4), (0, 7 * 10 + 5), 1e-6],
        ]
        chd_spd = [[(0, 0), 100.0], [(0, 99), 95.0]]
        expected_barriers = 5

    elif grid_type == "disu":
        # Unstructured grid (regular grid converted to unstructured format)
        disu_kwargs = get_disu_kwargs(
            1, 10, 10, 100.0, 100.0, 100.0, [0.0], return_vertices=True
        )
        disu = flopy.mf6.ModflowGwfdisu(gwf, **disu_kwargs)
        # HFB cellids for unstructured grid: (node,)
        # For a 10x10 grid, node = layer * nrow * ncol + row * ncol + col
        # Barriers between cells in column 4 and 5 for rows 3-7
        hfb_data = [
            [(3 * 10 + 4,), (3 * 10 + 5,), 1e-6],
            [(4 * 10 + 4,), (4 * 10 + 5,), 1e-6],
            [(5 * 10 + 4,), (5 * 10 + 5,), 1e-6],
            [(6 * 10 + 4,), (6 * 10 + 5,), 1e-6],
            [(7 * 10 + 4,), (7 * 10 + 5,), 1e-6],
        ]
        chd_spd = [[(0,), 100.0], [(99,), 95.0]]
        expected_barriers = 5

    # Add packages
    ic = flopy.mf6.ModflowGwfic(gwf, strt=100.0)
    npf = flopy.mf6.ModflowGwfnpf(gwf, save_flows=True)
    chd = flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chd_spd)
    hfb = flopy.mf6.ModflowGwfhfb(gwf, stress_period_data=hfb_data)

    return gwf, expected_barriers, grid_type


def test_plot_bc_hfb(hfb_model):
    """Test plotting HFB (Horizontal Flow Barrier) boundary conditions.

    HFB packages have cellid1/cellid2 fields instead of a single cellid field,
    representing barriers between pairs of cells. This test verifies that HFB
    can be plotted as lines on the shared faces between cells.

    Tests structured (DIS), vertex (DISV), and unstructured (DISU) grids.

    Addresses issue #2676.
    """
    gwf, expected_barriers, grid_type = hfb_model

    # Create map view and plot both CHD and HFB
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, aspect="equal")
    mapview = flopy.plot.PlotMapView(model=gwf, ax=ax)

    # Plot grid lines
    grid_result = mapview.plot_grid()
    assert isinstance(grid_result, LineCollection), (
        f"Expected LineCollection for grid ({grid_type})"
    )

    # Plot CHD (should work as before)
    chd_result = mapview.plot_bc("CHD")
    assert chd_result is not None, f"CHD plot should return a result ({grid_type})"

    # Plot HFB (the new functionality) - uses default orange color
    hfb_result = mapview.plot_bc("HFB", linewidth=3)

    assert isinstance(hfb_result, LineCollection)

    # Verify that the correct number of barrier segments were plotted
    segments = hfb_result.get_segments()
    assert len(segments) == expected_barriers

    # Verify each segment has 2 points (start and end of the barrier line)
    for seg in segments:
        assert len(seg) == 2

    # plt.show()
    plt.close(fig)


@pytest.fixture(params=["dis", "disv"])
def vertical_hfb_model(request):
    """Create a MODFLOW 6 model with vertical HFBs for testing.

    Vertical HFBs are barriers between vertically stacked cells (different layers).

    Returns
    -------
    tuple
        (gwf_model, expected_barrier_count, grid_type)
    """
    from flopy.utils.gridutil import get_disv_kwargs

    grid_type = request.param

    # Create simulation
    sim = flopy.mf6.MFSimulation(sim_name=f"test_vhfb_{grid_type}")
    tdis = flopy.mf6.ModflowTdis(sim)
    ims = flopy.mf6.ModflowIms(sim)
    gwf = flopy.mf6.ModflowGwf(sim, modelname="test")

    # Create discretization based on grid type
    if grid_type == "dis":
        # Structured grid with 3 layers
        dis = flopy.mf6.ModflowGwfdis(
            gwf,
            nlay=3,
            nrow=10,
            ncol=10,
            delr=100.0,
            delc=100.0,
            top=100.0,
            botm=[50.0, 0.0, -50.0],
        )
        # Vertical HFB cellids: barriers between layers at same row/col
        # Add some vertical barriers between layer 0 and 1
        vhfb_data = [
            [(0, 3, 4), (1, 3, 4), 1e-6],
            [(0, 4, 4), (1, 4, 4), 1e-6],
            [(0, 5, 4), (1, 5, 4), 1e-6],
        ]
        # Also add a few horizontal barriers for comparison
        hhfb_data = [
            [(1, 3, 4), (1, 3, 5), 1e-6],
            [(1, 4, 4), (1, 4, 5), 1e-6],
        ]
        hfb_data = vhfb_data + hhfb_data
        chd_spd = [[(0, 0, 0), 100.0], [(2, 9, 9), 95.0]]
        expected_vertical_barriers = 3
        expected_horizontal_barriers = 2

    elif grid_type == "disv":
        # Vertex grid with 3 layers
        import numpy as np

        disv_kwargs = get_disv_kwargs(
            3, 10, 10, 100.0, 100.0, 100.0, np.array([50.0, 0.0, -50.0])
        )
        disv = flopy.mf6.ModflowGwfdisv(gwf, **disv_kwargs)
        # Vertical HFB cellids for vertex grid: (layer, cell2d_id)
        vhfb_data = [
            [(0, 3 * 10 + 4), (1, 3 * 10 + 4), 1e-6],
            [(0, 4 * 10 + 4), (1, 4 * 10 + 4), 1e-6],
            [(0, 5 * 10 + 4), (1, 5 * 10 + 4), 1e-6],
        ]
        # Horizontal barriers
        hhfb_data = [
            [(1, 3 * 10 + 4), (1, 3 * 10 + 5), 1e-6],
            [(1, 4 * 10 + 4), (1, 4 * 10 + 5), 1e-6],
        ]
        hfb_data = vhfb_data + hhfb_data
        chd_spd = [[(0, 0), 100.0], [(2, 99), 95.0]]
        expected_vertical_barriers = 3
        expected_horizontal_barriers = 2

    # Add packages
    ic = flopy.mf6.ModflowGwfic(gwf, strt=100.0)
    npf = flopy.mf6.ModflowGwfnpf(gwf, save_flows=True)
    chd = flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chd_spd)
    hfb = flopy.mf6.ModflowGwfhfb(gwf, stress_period_data=hfb_data)

    return gwf, expected_vertical_barriers, expected_horizontal_barriers, grid_type


def test_plot_bc_vertical_hfb(vertical_hfb_model):
    """Test plotting vertical HFB (barriers between vertically stacked cells).

    Vertical HFBs should be rendered as patches (full cells) in map view,
    while horizontal HFBs are rendered as lines.

    Tests structured (DIS) and vertex (DISV) grids.
    """
    gwf, expected_vbarriers, expected_hbarriers, grid_type = vertical_hfb_model

    # Test on layer 0 - should show vertical barriers as patches
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, aspect="equal")
    mapview = flopy.plot.PlotMapView(model=gwf, layer=0, ax=ax)
    mapview.plot_grid()

    # Plot HFB
    hfb_result = mapview.plot_bc("HFB", linewidth=3)

    # Result should be a list containing both LineCollection (horizontal)
    # and PatchCollection (vertical)
    assert hfb_result is not None

    # If both types exist, result is a list
    if isinstance(hfb_result, list):
        # Should have both line and patch collections
        assert len(hfb_result) == 2, f"Expected 2 collections ({grid_type})"
        has_lines = any(isinstance(c, LineCollection) for c in hfb_result)
        has_patches = any(isinstance(c, PatchCollection) for c in hfb_result)
        assert has_lines and has_patches

    # plt.show()
    plt.close(fig)

    # Test on layer 1 - should show both vertical barriers and horizontal barriers
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, aspect="equal")
    mapview = flopy.plot.PlotMapView(model=gwf, layer=1, ax=ax)
    mapview.plot_grid()

    hfb_result = mapview.plot_bc("HFB", linewidth=3)
    assert hfb_result is not None

    # plt.show()
    plt.close(fig)
