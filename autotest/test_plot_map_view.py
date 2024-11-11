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
        assert isinstance(
            col, (QuadMesh, PathCollection)
        ), f"Unexpected collection type: {type(col)}"


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
        assert isinstance(
            col, (QuadMesh, PathCollection)
        ), f"Unexpected collection type: {type(col)}"


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
        assert isinstance(
            col, (QuadMesh, PathCollection)
        ), f"Unexpected collection type: {type(col)}"


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
        assert isinstance(
            col, (QuadMesh, PathCollection)
        ), f"Unexpected collection type: {type(col)}"


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
