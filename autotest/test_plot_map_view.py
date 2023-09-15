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


def test_map_view_contour(function_tmpdir):
    arr = np.random.rand(10, 10) * 100
    arr[-1, :] = np.nan
    delc = np.array([10] * 10, dtype=float)
    delr = np.array([8] * 10, dtype=float)
    top = np.ones((10, 10), dtype=float)
    botm = np.ones((3, 10, 10), dtype=float)
    botm[0] = 0.75
    botm[1] = 0.5
    botm[2] = 0.25
    idomain = np.ones((3, 10, 10))
    idomain[0, 0, :] = 0
    vmin = np.nanmin(arr)
    vmax = np.nanmax(arr)
    levels = np.linspace(vmin, vmax, 7)

    grid = StructuredGrid(
        delc=delc,
        delr=delr,
        top=top,
        botm=botm,
        idomain=idomain,
        lenuni=1,
        nlay=3,
        nrow=10,
        ncol=10,
    )

    pmv = PlotMapView(modelgrid=grid, layer=0)
    contours = pmv.contour_array(a=arr)
    plt.savefig(function_tmpdir / "map_view_contour.png")

    # if we ever revert from standard contours to tricontours, restore this nan check
    # for ix, lev in enumerate(contours.levels):
    #     if not np.allclose(lev, levels[ix]):
    #         raise AssertionError("TriContour NaN catch Failed")
