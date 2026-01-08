import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection, PatchCollection
from modflow_devtools.markers import requires_pkg

import flopy
from flopy.mf6 import MFSimulation


@pytest.mark.mf6
@pytest.mark.xfail(reason="sometimes get LineCollections instead of PatchCollections")
def test_cross_section_bc_gwfs_disv(example_data_path):
    mpath = example_data_path / "mf6" / "test003_gwfs_disv"
    sim = MFSimulation.load(sim_ws=mpath)
    ml6 = sim.get_model("gwf_1")
    xc = flopy.plot.PlotCrossSection(ml6, line={"line": ([0, 5.5], [10, 5.5])})
    xc.plot_bc("CHD")
    ax = xc.ax

    assert len(ax.collections) != 0, "Boundary condition was not drawn"

    for col in ax.collections:
        assert isinstance(col, PatchCollection), (
            f"Unexpected collection type: {type(col)}"
        )


@pytest.mark.mf6
@pytest.mark.xfail(reason="sometimes get LineCollections instead of PatchCollections")
def test_cross_section_bc_lake2tr(example_data_path):
    mpath = example_data_path / "mf6" / "test045_lake2tr"
    sim = MFSimulation.load(sim_ws=mpath)
    ml6 = sim.get_model("lakeex2a")
    xc = flopy.plot.PlotCrossSection(ml6, line={"row": 10})
    xc.plot_bc("LAK")
    xc.plot_bc("SFR")

    ax = xc.ax
    assert len(ax.collections) != 0, "Boundary condition was not drawn"

    for col in ax.collections:
        assert isinstance(col, PatchCollection), (
            f"Unexpected collection type: {type(col)}"
        )


@pytest.mark.mf6
@pytest.mark.xfail(reason="sometimes get LineCollections instead of PatchCollections")
def test_cross_section_bc_2models_mvr(example_data_path):
    mpath = example_data_path / "mf6" / "test006_2models_mvr"
    sim = MFSimulation.load(sim_ws=mpath)
    ml6 = sim.get_model("parent")
    xc = flopy.plot.PlotCrossSection(ml6, line={"column": 1})
    xc.plot_bc("MAW")

    ax = xc.ax
    assert len(ax.collections) > 0, "Boundary condition was not drawn"

    for col in ax.collections:
        assert isinstance(col, PatchCollection), (
            f"Unexpected collection type: {type(col)}"
        )


@pytest.mark.mf6
@pytest.mark.xfail(reason="sometimes get LineCollections instead of PatchCollections")
def test_cross_section_bc_UZF_3lay(example_data_path):
    mpath = example_data_path / "mf6" / "test001e_UZF_3lay"
    sim = MFSimulation.load(sim_ws=mpath)
    ml6 = sim.get_model("gwf_1")

    xc = flopy.plot.PlotCrossSection(ml6, line={"row": 0})
    xc.plot_bc("UZF")

    ax = xc.ax
    assert len(ax.collections) != 0, "Boundary condition was not drawn"

    for col in ax.collections:
        assert isinstance(col, PatchCollection), (
            f"Unexpected collection type: {type(col)}"
        )


def structured_square_grid(side: int = 10, thick: int = 10):
    """
    Creates a basic 1-layer structured grid with the given thickness and number of
    cells per side
    Parameters
    ----------
    side : The number of cells per side
    thick : The thickness of the grid's single layer
    Returns
    -------
    A single-layer StructuredGrid of the given size and thickness
    """

    from flopy.discretization.structuredgrid import StructuredGrid

    delr = np.ones(side)
    delc = np.ones(side)
    top = np.ones((side, side)) * thick
    botm = np.ones((side, side)) * (top - thick).reshape(1, side, side)
    return StructuredGrid(delr=delr, delc=delc, top=top, botm=botm)


@requires_pkg("shapely")
@pytest.mark.parametrize(
    "line",
    [(), [], (()), [[]], (0, 0), [0, 0], [[0, 0]]],
)
def test_cross_section_invalid_lines_raise_error(line):
    grid = structured_square_grid(side=10)
    with pytest.raises(ValueError):
        flopy.plot.PlotCrossSection(modelgrid=grid, line={"line": line})


@requires_pkg("shapely")
@pytest.mark.parametrize(
    "line",
    [
        # diagonal
        [(0, 0), (10, 10)],
        ([0, 0], [10, 10]),
        # horizontal
        ([0, 5.5], [10, 5.5]),
        [(0, 5.5), (10, 5.5)],
        # vertical
        [(5.5, 0), (5.5, 10)],
        ([5.5, 0], [5.5, 10]),
        # multiple segments
        [(0, 0), (4, 6), (10, 10)],
        ([0, 0], [4, 6], [10, 10]),
    ],
)
def test_cross_section_valid_line_representations(line):
    from shapely.geometry import LineString as SLS

    from flopy.utils.geometry import LineString as FLS

    grid = structured_square_grid(side=10)

    fls = FLS(line)
    sls = SLS(line)

    # use raw, flopy.utils.geometry and shapely.geometry representations
    lxc = flopy.plot.PlotCrossSection(modelgrid=grid, line={"line": line})
    fxc = flopy.plot.PlotCrossSection(modelgrid=grid, line={"line": fls})
    sxc = flopy.plot.PlotCrossSection(modelgrid=grid, line={"line": sls})

    # make sure parsed points are identical for all line representations
    assert np.allclose(lxc.pts, fxc.pts) and np.allclose(lxc.pts, sxc.pts)
    assert set(lxc.xypts.keys()) == set(fxc.xypts.keys()) == set(sxc.xypts.keys())
    for k in lxc.xypts.keys():
        assert np.allclose(lxc.xypts[k], fxc.xypts[k]) and np.allclose(
            lxc.xypts[k], sxc.xypts[k]
        )


@pytest.mark.parametrize(
    "line",
    [
        0,
        [0],
        [0, 0],
        (0, 0),
        [(0, 0)],
        ([0, 0]),
    ],
)
@requires_pkg("shapely", "geojson")
def test_cross_section_invalid_line_representations_fail(line):
    grid = structured_square_grid(side=10)
    with pytest.raises(ValueError):
        flopy.plot.PlotCrossSection(modelgrid=grid, line={"line": line})


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
    user_extent = 0, 500, 0, 25
    ax.axis(user_extent)

    pxc = flopy.plot.PlotCrossSection(modelgrid=grid, ax=ax, line={"column": 4})
    pxc.plot_grid()

    lims = ax.axes.viewLim
    if (lims.x0, lims.x1, lims.y0, lims.y1) != user_extent:
        raise AssertionError("PlotMapView not checking for user scaling")

    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 8))
    pxc = flopy.plot.PlotCrossSection(modelgrid=grid, ax=ax, line={"column": 4})
    pxc.plot_grid()

    lims = ax.axes.viewLim
    if (lims.x0, lims.x1, lims.y0, lims.y1) != pxc.extent:
        raise AssertionError("PlotMapView auto extent setting not working")

    plt.close(fig)


@requires_pkg("shapely")
def test_plot_centers():
    from matplotlib.collections import PathCollection

    nlay = 1
    nrow = 10
    ncol = 10

    delc = np.ones((nrow,))
    delr = np.ones((ncol,))
    top = np.ones((nrow, ncol))
    botm = np.zeros((nlay, nrow, ncol))
    idomain = np.ones(botm.shape, dtype=int)

    idomain[0, :, 0:3] = 0

    grid = flopy.discretization.StructuredGrid(
        delc=delc, delr=delr, top=top, botm=botm, idomain=idomain
    )

    line = {"line": [(0, 0), (10, 10)]}
    active_xc_cells = 7

    pxc = flopy.plot.PlotCrossSection(modelgrid=grid, line=line)
    pc = pxc.plot_centers()

    if not isinstance(pc, PathCollection):
        raise AssertionError("plot_centers() not returning PathCollection object")

    verts = pc._offsets
    if not verts.shape[0] == active_xc_cells:
        raise AssertionError("plot_centers() not properly masking inactive cells")

    center_dict = pxc.projctr
    edge_dict = pxc.projpts

    for node, center in center_dict.items():
        verts = np.array(edge_dict[node]).T
        xmin = np.min(verts[0])
        xmax = np.max(verts[0])
        if xmax < center < xmin:
            raise AssertionError("Cell center not properly drawn on cross-section")


@pytest.fixture(params=["dis", "disv", "disu"])
def hfb_xc_model(request):
    """Create a MODFLOW 6 model with HFB for cross section testing.

    Parameters
    ----------
    request.param : str
        Grid type: "dis" (structured), "disv" (vertex), or "disu" (unstructured)

    Returns
    -------
    tuple
        (gwf_model, cross_section_line, grid_type)
    """
    from flopy.utils.gridutil import get_disu_kwargs, get_disv_kwargs

    grid_type = request.param

    # Create simulation
    sim = flopy.mf6.MFSimulation(sim_name=f"test_hfb_xc_{grid_type}")
    tdis = flopy.mf6.ModflowTdis(sim)
    ims = flopy.mf6.ModflowIms(sim)
    gwf = flopy.mf6.ModflowGwf(sim, modelname="test")

    # Create discretization based on grid type
    if grid_type == "dis":
        # Structured grid with 2 layers
        dis = flopy.mf6.ModflowGwfdis(
            gwf,
            nlay=2,
            nrow=10,
            ncol=10,
            delr=100.0,
            delc=100.0,
            top=100.0,
            botm=[50.0, 0.0],
        )
        # HFB cellids for structured grid: (layer, row, col)
        # Create barriers along column boundary at row 4
        hfb_data = [
            [(0, 3, 4), (0, 3, 5), 1e-6],
            [(0, 4, 4), (0, 4, 5), 1e-6],
            [(0, 5, 4), (0, 5, 5), 1e-6],
        ]
        xc_line = {"row": 4}

    elif grid_type == "disv":
        # Vertex grid (regular grid converted to vertex format)
        disv_kwargs = get_disv_kwargs(2, 10, 10, 100.0, 100.0, 100.0, [50.0, 0.0])
        disv = flopy.mf6.ModflowGwfdisv(gwf, **disv_kwargs)
        # HFB cellids for vertex grid: (layer, cell2d_id)
        # For a 10x10 grid, cell2d IDs are row*ncol + col
        hfb_data = [
            [(0, 3 * 10 + 4), (0, 3 * 10 + 5), 1e-6],
            [(0, 4 * 10 + 4), (0, 4 * 10 + 5), 1e-6],
            [(0, 5 * 10 + 4), (0, 5 * 10 + 5), 1e-6],
        ]
        # For DISV, use line coordinates instead of row
        xc_line = {"line": ([0, 450], [1000, 450])}

    elif grid_type == "disu":
        # Unstructured grid (regular grid converted to unstructured format)
        disu_kwargs = get_disu_kwargs(
            2, 10, 10, 100.0, 100.0, 100.0, [50.0, 0.0], return_vertices=True
        )
        disu = flopy.mf6.ModflowGwfdisu(gwf, **disu_kwargs)
        # HFB cellids for unstructured grid: (node,)
        # For a 10x10 grid, node = layer * nrow * ncol + row * ncol + col
        hfb_data = [
            [(0 * 10 * 10 + 3 * 10 + 4,), (0 * 10 * 10 + 3 * 10 + 5,), 1e-6],
            [(0 * 10 * 10 + 4 * 10 + 4,), (0 * 10 * 10 + 4 * 10 + 5,), 1e-6],
            [(0 * 10 * 10 + 5 * 10 + 4,), (0 * 10 * 10 + 5 * 10 + 5,), 1e-6],
        ]
        # For DISU, use line coordinates
        xc_line = {"line": ([0, 450], [1000, 450])}

    # Add packages
    ic = flopy.mf6.ModflowGwfic(gwf, strt=75.0)
    npf = flopy.mf6.ModflowGwfnpf(gwf, save_flows=True)
    hfb = flopy.mf6.ModflowGwfhfb(gwf, stress_period_data=hfb_data)

    return gwf, xc_line, grid_type


def test_cross_section_bc_hfb(hfb_xc_model):
    """Test plotting HFB (Horizontal Flow Barrier) in cross sections.

    HFB packages have cellid1/cellid2 fields instead of a single cellid field.
    Plot barriers by showing both cells that the barrier affects.

    Tests structured (DIS), vertex (DISV), and unstructured (DISU) grids.

    Addresses issue #2676.
    """
    gwf, xc_line, grid_type = hfb_xc_model

    # Create figure and cross section
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    xc = flopy.plot.PlotCrossSection(model=gwf, line=xc_line, ax=ax)

    # Plot HFB
    xc.plot_grid()
    hfb_result = xc.plot_bc("HFB", alpha=0.5)

    assert hfb_result is not None, f"HFB plot should return a result ({grid_type})"
    assert isinstance(hfb_result, PatchCollection)
    assert len(ax.collections) > 0

    # plt.show()
    plt.close(fig)


@pytest.fixture(params=["dis", "disv"])
def vertical_hfb_xc_model(request):
    """Create a MODFLOW 6 model with vertical HFBs for cross section testing.

    Vertical HFBs are barriers between vertically stacked cells (different layers).

    Parameters
    ----------
    request.param : str
        Grid type: "dis" (structured) or "disv" (vertex)

    Returns
    -------
    tuple
        (gwf_model, cross_section_line, grid_type)
    """
    import numpy as np

    from flopy.utils.gridutil import get_disv_kwargs

    grid_type = request.param

    # Create simulation
    sim = flopy.mf6.MFSimulation(sim_name=f"test_vhfb_xc_{grid_type}")
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
        # Add vertical barriers along row 4
        vhfb_data = [
            [(0, 4, 3), (1, 4, 3), 1e-6],
            [(0, 4, 4), (1, 4, 4), 1e-6],
            [(0, 4, 5), (1, 4, 5), 1e-6],
        ]
        # Also add horizontal barriers for comparison
        hhfb_data = [
            [(1, 4, 4), (1, 4, 5), 1e-6],
        ]
        hfb_data = vhfb_data + hhfb_data
        xc_line = {"row": 4}

    elif grid_type == "disv":
        # Vertex grid with 3 layers
        disv_kwargs = get_disv_kwargs(
            3, 10, 10, 100.0, 100.0, 100.0, np.array([50.0, 0.0, -50.0])
        )
        disv = flopy.mf6.ModflowGwfdisv(gwf, **disv_kwargs)
        # Vertical HFB cellids for vertex grid: (layer, cell2d_id)
        # Cross section at y=450 intersects row 5 (cells 50-59), not row 4
        vhfb_data = [
            [(0, 5 * 10 + 3), (1, 5 * 10 + 3), 1e-6],
            [(0, 5 * 10 + 4), (1, 5 * 10 + 4), 1e-6],
            [(0, 5 * 10 + 5), (1, 5 * 10 + 5), 1e-6],
        ]
        # Horizontal barriers
        hhfb_data = [
            [(1, 5 * 10 + 4), (1, 5 * 10 + 5), 1e-6],
        ]
        hfb_data = vhfb_data + hhfb_data
        # For DISV, use line coordinates
        xc_line = {"line": ([0, 450], [1000, 450])}

    # Add packages
    ic = flopy.mf6.ModflowGwfic(gwf, strt=75.0)
    npf = flopy.mf6.ModflowGwfnpf(gwf, save_flows=True)
    hfb = flopy.mf6.ModflowGwfhfb(gwf, stress_period_data=hfb_data)

    return gwf, xc_line, grid_type


def test_cross_section_vertical_hfb(vertical_hfb_xc_model):
    """Test plotting vertical HFB (barriers between vertically stacked cells).

    Vertical HFBs should be rendered as lines at layer interfaces in cross sections,
    while horizontal HFBs are rendered as patches.

    Tests structured (DIS) and vertex (DISV) grids.
    """
    gwf, xc_line, grid_type = vertical_hfb_xc_model

    # Create figure and cross section
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    xc = flopy.plot.PlotCrossSection(model=gwf, line=xc_line, ax=ax)
    xc.plot_grid()

    # Plot HFB
    hfb_result = xc.plot_bc("HFB", alpha=0.5)

    assert hfb_result is not None

    # Result should be a list containing both PatchCollection (horizontal)
    # and LineCollection (vertical)
    if isinstance(hfb_result, list):
        # Should have both types
        assert len(hfb_result) == 2, f"Expected 2 collections ({grid_type})"
        has_patches = any(isinstance(c, PatchCollection) for c in hfb_result)
        has_lines = any(isinstance(c, LineCollection) for c in hfb_result)
        assert has_patches and has_lines
    else:
        # If only one type, it should be LineCollection (only vertical barriers in view)
        # or PatchCollection (only horizontal barriers in view)
        assert isinstance(hfb_result, (PatchCollection, LineCollection))

    # plt.show()
    plt.close(fig)
