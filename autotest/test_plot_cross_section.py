import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PatchCollection
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


@pytest.mark.mf6
def test_cross_section_bc_hfb():
    """Test plotting HFB (Horizontal Flow Barrier) in cross sections.

    HFB packages have cellid1/cellid2 fields instead of a single cellid field.
    In cross sections, barriers are plotted by showing both cells that the
    barrier affects (as a simplification, since proper barrier visualization
    would require determining if the cross section plane intersects each barrier).

    Addresses issue #2676.
    """
    # Create a simple MODFLOW 6 model with multiple layers
    sim = flopy.mf6.MFSimulation(sim_name="test_hfb_xc")
    tdis = flopy.mf6.ModflowTdis(sim)
    ims = flopy.mf6.ModflowIms(sim)

    # Create gwf model
    gwf = flopy.mf6.ModflowGwf(sim, modelname="test")

    # Create structured grid with 2 layers
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

    # Add initial conditions
    ic = flopy.mf6.ModflowGwfic(gwf, strt=75.0)

    # Add npf
    npf = flopy.mf6.ModflowGwfnpf(gwf, save_flows=True)

    # Add HFB - barriers between cells in layer 0
    # Create a vertical barrier along column boundary
    hfb_data = [
        [(0, 3, 4), (0, 3, 5), 1e-6],
        [(0, 4, 4), (0, 4, 5), 1e-6],
        [(0, 5, 4), (0, 5, 5), 1e-6],
    ]
    hfb = flopy.mf6.ModflowGwfhfb(gwf, stress_period_data=hfb_data)

    # Create cross section along row 4 (which intersects the barriers)
    xc = flopy.plot.PlotCrossSection(model=gwf, line={"row": 4})

    # Plot HFB
    xc.plot_grid()
    hfb_result = xc.plot_bc("HFB", alpha=0.5)

    # Verify that something was plotted
    assert hfb_result is not None, "HFB plot should return a result"

    # For cross sections, HFB is plotted as patches (both cells affected by barrier)
    assert isinstance(hfb_result, PatchCollection), (
        f"Expected PatchCollection for HFB cross section plot, got {type(hfb_result)}"
    )

    # Verify that the axis has collections
    ax = xc.ax
    assert len(ax.collections) > 0, "HFB boundary condition was not drawn"

    # plt.show()
    plt.close()
