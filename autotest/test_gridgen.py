import os
from pathlib import Path
from shutil import which

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import LineCollection, PathCollection, QuadMesh
from modflow_devtools.markers import requires_exe, requires_pkg
from modflow_devtools.misc import has_pkg

import flopy
from autotest.test_grid_cases import GridCases
from flopy.discretization.unstructuredgrid import UnstructuredGrid
from flopy.discretization.vertexgrid import VertexGrid
from flopy.utils.gridgen import Gridgen, get_ia_from_iac


@requires_exe("gridgen")
def test_ctor_accepts_path_or_string_model_ws(function_tmpdir):
    grid = GridCases().structured_small()

    g = Gridgen(grid, model_ws=function_tmpdir)
    assert g.model_ws == function_tmpdir

    g = Gridgen(grid, model_ws=str(function_tmpdir))
    assert g.model_ws == function_tmpdir


def get_structured_grid():
    """Get a small version of the first grid in:
    .docs/Notebooks/gridgen_example.py"""

    Lx = 100.0
    Ly = 100.0
    nlay = 2
    nrow = 11
    ncol = 11
    delr = Lx / ncol
    delc = Ly / nrow
    h0 = 10
    h1 = 5
    top = h0
    botm = np.zeros((nlay, nrow, ncol), dtype=np.float32)
    botm[1, :, :] = -10.0
    ms = flopy.modflow.Modflow(rotation=-20.0)
    dis = flopy.modflow.ModflowDis(
        ms,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
    )
    return ms.modelgrid


@requires_exe("gridgen")
@requires_pkg("pyshp", "shapely", name_map={"pyshp": "shapefile"})
@pytest.mark.parametrize("grid_type", ["vertex", "unstructured"])
def test_add_active_domain(function_tmpdir, grid_type):
    bgrid = get_structured_grid()

    # test providing active domain in various ways
    grids = []
    for feature in [
        [[[(0, 0), (0, 60), (40, 80), (60, 0), (0, 0)]]],
        function_tmpdir / "ad0.shp",
        function_tmpdir / "ad0",
        "ad0.shp",
        "ad0",
    ]:
        print(
            "Testing add_active_domain() for", grid_type, "grid with features", feature
        )
        gridgen = Gridgen(bgrid, model_ws=function_tmpdir)
        gridgen.add_active_domain(feature, range(bgrid.nlay))
        gridgen.build()
        grid = (
            VertexGrid(**gridgen.get_gridprops_vertexgrid())
            if grid_type == "vertex"
            else UnstructuredGrid(**gridgen.get_gridprops_unstructuredgrid())
        )
        grid.plot()
        grids.append(grid)
        # plt.show()

        assert grid.nnodes < bgrid.nnodes
        assert not np.array_equal(grid.ncpl, bgrid.ncpl)
        assert all(np.array_equal(grid.ncpl, g.ncpl) for g in grids)
        assert all(grid.nnodes == g.nnodes for g in grids)


@requires_exe("gridgen")
@requires_pkg("pyshp", "shapely", name_map={"pyshp": "shapefile"})
@pytest.mark.parametrize("grid_type", ["vertex", "unstructured"])
def test_add_refinement_feature(function_tmpdir, grid_type):
    bgrid = get_structured_grid()

    # test providing refinement features in various ways
    grids = []
    for features in [
        [[[(0, 0), (0, 60), (40, 80), (60, 0), (0, 0)]]],
        function_tmpdir / "rf0.shp",
        function_tmpdir / "rf0",
        "rf0.shp",
        "rf0",
    ]:
        print(
            "Testing add_refinement_feature() for",
            grid_type,
            "grid with features",
            features,
        )
        gridgen = Gridgen(bgrid, model_ws=function_tmpdir)
        gridgen.add_refinement_features(features, "polygon", 1, range(bgrid.nlay))
        gridgen.build()
        grid = (
            VertexGrid(**gridgen.get_gridprops_vertexgrid())
            if grid_type == "vertex"
            else UnstructuredGrid(**gridgen.get_gridprops_unstructuredgrid())
        )
        grid.plot()
        # plt.show()

        assert grid.nnodes > bgrid.nnodes
        assert not np.array_equal(grid.ncpl, bgrid.ncpl)
        assert all(np.array_equal(grid.ncpl, g.ncpl) for g in grids)
        assert all(grid.nnodes == g.nnodes for g in grids)


@pytest.mark.slow
@requires_exe("mf6", "gridgen")
@requires_pkg("shapely", "geopandas")
def test_mf6disv(function_tmpdir):
    from shapely.geometry import Polygon

    name = "dummy"
    nlay = 3
    nrow = 10
    ncol = 10
    delr = delc = 1.0
    top = 1
    bot = 0
    dz = (top - bot) / nlay
    botm = [top - k * dz for k in range(1, nlay + 1)]

    # Create a dummy model and regular grid to use as the base grid for gridgen
    sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=function_tmpdir, exe_name="mf6")
    gwf = flopy.mf6.ModflowGwf(sim, modelname=name)

    dis = flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
    )

    # Create and build the gridgen model with a refined area in the middle
    g = Gridgen(gwf.modelgrid, model_ws=function_tmpdir)
    polys = [Polygon([(4, 4), (6, 4), (6, 6), (4, 6)])]
    g.add_refinement_features(polys, "polygon", 3, range(nlay))
    g.build()
    disv_gridprops = g.get_gridprops_disv()

    # find the cell numbers for constant heads
    chdspd = []
    ilay = 0
    for x, y, head in [(0, 10, 1.0), (10, 0, 0.0)]:
        ra = g.intersect([(x, y)], "point", ilay)
        ic = ra["nodenumber"][0]
        chdspd.append([(ilay, ic), head])

    # build run and post-process the MODFLOW 6 model
    name = "mymodel"
    sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=function_tmpdir, exe_name="mf6")
    tdis = flopy.mf6.ModflowTdis(sim)
    ims = flopy.mf6.ModflowIms(sim, linear_acceleration="bicgstab")
    gwf = flopy.mf6.ModflowGwf(sim, modelname=name, save_flows=True)
    disv = flopy.mf6.ModflowGwfdisv(gwf, **disv_gridprops)
    ic = flopy.mf6.ModflowGwfic(gwf)
    npf = flopy.mf6.ModflowGwfnpf(gwf, xt3doptions=True, save_specific_discharge=True)
    chd = flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chdspd)
    budget_file = f"{name}.bud"
    head_file = f"{name}.hds"
    oc = flopy.mf6.ModflowGwfoc(
        gwf,
        budget_filerecord=budget_file,
        head_filerecord=head_file,
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )
    sim.write_simulation()

    gwf.modelgrid.set_coord_info(angrot=15)

    # write grid and model shapefiles
    fname = function_tmpdir / "grid.shp"
    gdf = gwf.modelgrid.to_geodataframe()
    gdf.to_file(fname)

    fname = function_tmpdir / "model.shp"
    gdf = gwf.to_geodataframe()
    gdf.to_file(fname)

    sim.run_simulation(silent=True)
    head = gwf.output.head().get_data()
    bud = gwf.output.budget()
    spdis = bud.get_data(text="DATA-SPDIS")[0]
    f = plt.figure(figsize=(10, 10))
    vmin = head.min()
    vmax = head.max()
    for ilay in range(gwf.modelgrid.nlay):
        ax = plt.subplot(1, gwf.modelgrid.nlay, ilay + 1)
        pmv = flopy.plot.PlotMapView(gwf, layer=ilay, ax=ax)
        ax.set_aspect("equal")
        pmv.plot_array(head.flatten(), cmap="jet", vmin=vmin, vmax=vmax)
        pmv.plot_grid(colors="k", alpha=0.1)
        pmv.contour_array(
            head,
            levels=[0.2, 0.4, 0.6, 0.8],
            linewidths=3.0,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"Layer {ilay + 1}")
        pmv.plot_vector(spdis["qx"], spdis["qy"], color="white")
        fname = "results.png"
        fname = function_tmpdir / fname
        plt.savefig(fname)
        plt.close("all")

    # test plotting
    # load up the vertex example problem
    name = "mymodel"
    sim = flopy.mf6.MFSimulation.load(
        sim_name=name,
        version="mf6",
        exe_name="mf6",
        sim_ws=function_tmpdir,
    )
    # get gwf model
    gwf = sim.get_model(name)

    # get the dis package
    dis = gwf.disv

    # try plotting an array
    top = dis.top
    ax = top.plot()
    assert ax
    plt.close("all")

    # try plotting a package
    ax = dis.plot()
    assert ax
    plt.close("all")

    # try plotting a model
    ax = gwf.plot()
    assert ax
    plt.close("all")


@pytest.fixture
def sim_disu_diff_layers(function_tmpdir):
    from shapely.geometry import Polygon

    name = "disu_diff_layers"
    nlay = 3
    nrow = 10
    ncol = 10
    delr = delc = 1.0
    top = 1
    bot = 0
    dz = (top - bot) / nlay
    botm = [top - k * dz for k in range(1, nlay + 1)]

    # Create a dummy model and regular grid to use as the base grid for gridgen
    sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=function_tmpdir, exe_name="mf6")
    gwf = flopy.mf6.ModflowGwf(sim, modelname=name)

    dis = flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
    )

    # Create and build the gridgen model with a refined area in the middle
    g = Gridgen(gwf.modelgrid, model_ws=function_tmpdir)
    polys = [Polygon([(4, 4), (6, 4), (6, 6), (4, 6)])]
    g.add_refinement_features(polys, "polygon", 3, layers=[0])
    g.build()
    disu_gridprops = g.get_gridprops_disu6()

    chdspd = []
    for x, y, head in [(0, 10, 1.0), (10, 0, 0.0)]:
        ra = g.intersect([(x, y)], "point", 0)
        ic = ra["nodenumber"][0]
        chdspd.append([(ic,), head])

    # build run and post-process the MODFLOW 6 model
    name = "mymodel"
    sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=function_tmpdir, exe_name="mf6")
    tdis = flopy.mf6.ModflowTdis(sim)
    ims = flopy.mf6.ModflowIms(sim, linear_acceleration="bicgstab")
    gwf = flopy.mf6.ModflowGwf(sim, modelname=name, save_flows=True)
    disu = flopy.mf6.ModflowGwfdisu(gwf, **disu_gridprops)
    ic = flopy.mf6.ModflowGwfic(gwf)
    npf = flopy.mf6.ModflowGwfnpf(gwf, xt3doptions=True, save_specific_discharge=True)
    chd = flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chdspd)
    budget_file = f"{name}.bud"
    head_file = f"{name}.hds"
    oc = flopy.mf6.ModflowGwfoc(
        gwf,
        budget_filerecord=budget_file,
        head_filerecord=head_file,
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )

    return sim


@pytest.mark.slow
@requires_exe("mf6", "gridgen")
@requires_pkg("shapely", "pyshp", name_map={"pyshp": "shapefile"})
def test_mf6disu(sim_disu_diff_layers):
    sim = sim_disu_diff_layers
    ws = sim.sim_path
    gwf = sim.get_model()
    sim.write_simulation()

    gwf.modelgrid.set_coord_info(angrot=15)

    # The flopy Gridgen object includes the plottable layer number to the
    # diagonal position in the ihc array.  This is why and how modelgrid.nlay
    # is set to 3 and ncpl has a different number of cells per layer.
    assert gwf.modelgrid.nlay == 3
    assert np.allclose(gwf.modelgrid.ncpl, np.array([436, 184, 112]))

    # write grid and model shapefiles
    fname = ws / "grid.shp"
    gdf = gwf.modelgrid.to_geodataframe()
    gdf.to_file(fname)

    fname = ws / "model.shp"
    gdf = gwf.to_geodataframe()
    gdf.to_file(fname)

    fname = ws / "chd.shp"
    gdf = gwf.chd.to_geodataframe()
    gdf.to_file(fname)

    sim.run_simulation(silent=True)
    head = gwf.output.head().get_data()
    bud = gwf.output.budget()
    spdis = bud.get_data(text="DATA-SPDIS")[0]

    f = plt.figure(figsize=(10, 10))
    vmin = head.min()
    vmax = head.max()
    for ilay in range(gwf.modelgrid.nlay):
        ax = plt.subplot(1, gwf.modelgrid.nlay, ilay + 1)
        pmv = flopy.plot.PlotMapView(gwf, layer=ilay, ax=ax)
        ax.set_aspect("equal")
        pmv.plot_array(head.flatten(), cmap="jet", vmin=vmin, vmax=vmax)
        pmv.plot_grid(colors="k", alpha=0.1)
        pmv.contour_array(
            head, levels=[0.2, 0.4, 0.6, 0.8], linewidths=3.0, vmin=vmin, vmax=vmax
        )
        ax.set_title(f"Layer {ilay + 1}")
        pmv.plot_vector(spdis["qx"], spdis["qy"], color="white")
    fname = "results.png"
    fname = ws / fname
    plt.savefig(fname)
    plt.close("all")

    # check plot_bc works for unstructured mf6 grids
    # (for each layer, and then for all layers in one plot)
    plot_ranges = [range(gwf.modelgrid.nlay), range(1)]
    plot_alls = [False, True]
    for plot_range, plot_all in zip(plot_ranges, plot_alls):
        f_bc = plt.figure(figsize=(10, 10))
        for ilay in plot_range:
            ax = plt.subplot(1, plot_range[-1] + 1, ilay + 1)
            pmv = flopy.plot.PlotMapView(gwf, layer=ilay, ax=ax)
            ax.set_aspect("equal")

            pmv.plot_bc("CHD", plotAll=plot_all, edgecolor="None", zorder=2)
            pmv.plot_grid(colors="k", linewidth=0.3, alpha=0.1, zorder=1)

            if len(ax.collections) == 0:
                raise AssertionError("Boundary condition was not drawn")

            for col in ax.collections:
                if not isinstance(col, (QuadMesh, PathCollection, LineCollection)):
                    raise AssertionError("Unexpected collection type")
        plt.close()

    # test plotting
    # load up the disu example problem
    name = "mymodel"
    sim = flopy.mf6.MFSimulation.load(
        sim_name=name,
        version="mf6",
        exe_name="mf6",
        sim_ws=ws,
    )
    gwf = sim.get_model(name)

    # check to make sure that ncpl was set properly through the diagonal
    # position of the ihc array
    assert np.allclose(gwf.modelgrid.ncpl, np.array([436, 184, 112]))

    # get the dis package
    dis = gwf.disu

    # try plotting an array
    top = dis.top
    ax = top.plot()
    assert ax
    plt.close("all")

    # try plotting a package
    ax = dis.plot()
    assert ax
    plt.close("all")

    # try plotting a model
    ax = gwf.plot()
    assert ax
    plt.close("all")


@pytest.mark.slow
@requires_exe("mfusg", "gridgen")
@requires_pkg("shapely", "pyshp", name_map={"pyshp": "shapefile"})
def test_mfusg(function_tmpdir):
    from shapely.geometry import Polygon

    name = "dummy"
    nlay = 3
    nrow = 10
    ncol = 10
    delr = delc = 1.0
    top = 1
    bot = 0
    dz = (top - bot) / nlay
    botm = [top - k * dz for k in range(1, nlay + 1)]

    # create dummy model and dis package for gridgen
    m = flopy.modflow.Modflow(modelname=name, model_ws=function_tmpdir)
    dis = flopy.modflow.ModflowDis(
        m,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
    )

    # Create and build the gridgen model with a refined area in the middle
    g = Gridgen(m.modelgrid, model_ws=function_tmpdir)
    polys = [Polygon([(4, 4), (6, 4), (6, 6), (4, 6)])]
    g.add_refinement_features(polys, "polygon", 3, layers=[0])
    g.build()

    chdspd = []
    for x, y, head in [(0, 10, 1.0), (10, 0, 0.0)]:
        ra = g.intersect([(x, y)], "point", 0)
        ic = ra["nodenumber"][0]
        chdspd.append([ic, head, head])

    # gridprops = g.get_gridprops()
    gridprops = g.get_gridprops_disu5()

    # create the mfusg modoel
    name = "mymodel"
    m = flopy.mfusg.MfUsg(
        modelname=name,
        model_ws=function_tmpdir,
        exe_name="mfusg",
        structured=False,
    )
    disu = flopy.mfusg.MfUsgDisU(m, **gridprops)
    bas = flopy.mfusg.MfUsgBas(m)
    lpf = flopy.mfusg.MfUsgLpf(m)
    chd = flopy.modflow.ModflowChd(m, stress_period_data=chdspd)
    sms = flopy.mfusg.MfUsgSms(m)
    oc = flopy.modflow.ModflowOc(m, stress_period_data={(0, 0): ["save head"]})
    m.write_input()

    # MODFLOW-USG does not have vertices, so we need to create
    # and unstructured grid and then assign it to the model. This
    # will allow plotting and other features to work properly.
    gridprops_ug = g.get_gridprops_unstructuredgrid()
    ugrid = flopy.discretization.UnstructuredGrid(**gridprops_ug, angrot=-15)
    m.modelgrid = ugrid

    m.run_model()

    # head is returned as a list of head arrays for each layer
    head_file = function_tmpdir / f"{name}.hds"
    head = flopy.utils.HeadUFile(head_file).get_data()

    f = plt.figure(figsize=(10, 10))
    vmin = 0.0
    vmax = 1.0
    for ilay in range(disu.nlay):
        ax = plt.subplot(1, g.nlay, ilay + 1)
        pmv = flopy.plot.PlotMapView(m, layer=ilay, ax=ax)
        ax.set_aspect("equal")
        pmv.plot_array(head[ilay], cmap="jet", vmin=vmin, vmax=vmax)
        pmv.plot_grid(colors="k", alpha=0.1)
        pmv.contour_array(head[ilay], levels=[0.2, 0.4, 0.6, 0.8], linewidths=3.0)
        ax.set_title(f"Layer {ilay + 1}")
        # pmv.plot_specific_discharge(spdis, color='white')
    fname = "results.png"
    fname = function_tmpdir / fname
    plt.savefig(fname)
    plt.close("all")

    # check plot_bc works for unstructured mfusg grids
    # (for each layer, and then for all layers in one plot)
    plot_ranges = [range(disu.nlay), range(1)]
    plot_alls = [False, True]
    for plot_range, plot_all in zip(plot_ranges, plot_alls):
        f_bc = plt.figure(figsize=(10, 10))
        for ilay in plot_range:
            ax = plt.subplot(1, plot_range[-1] + 1, ilay + 1)
            pmv = flopy.plot.PlotMapView(m, layer=ilay, ax=ax)
            ax.set_aspect("equal")

            pmv.plot_bc("CHD", plotAll=plot_all, edgecolor="None", zorder=2)
            pmv.plot_grid(colors="k", linewidth=0.3, alpha=0.1, zorder=1)

            if len(ax.collections) == 0:
                raise AssertionError("Boundary condition was not drawn")

            for col in ax.collections:
                if not isinstance(col, (QuadMesh, PathCollection, LineCollection)):
                    raise AssertionError("Unexpected collection type")
        plt.close()

    # re-run with an LPF keyword specified. This would have thrown an error
    # before the addition of ikcflag to mflpf.py (flopy 3.3.3 and earlier).
    lpf = flopy.mfusg.MfUsgLpf(m, novfc=True, nocvcorrection=True)
    m.write_input()
    m.run_model()

    # also test load of unstructured LPF with keywords
    lpf2 = flopy.mfusg.MfUsgLpf.load(function_tmpdir / f"{name}.lpf", m, check=False)
    msg = "NOCVCORRECTION and NOVFC should be in lpf options but at least one is not."
    assert (
        "NOVFC" in lpf2.options.upper() and "NOCVCORRECTION" in lpf2.options.upper()
    ), msg

    # test disu, bas6, lpf shapefile export for mfusg unstructured models
    gdf = m.disu.to_geodataframe()
    gdf.to_file(function_tmpdir / f"{name}_disu.shp")

    gdf = m.bas6.to_geodataframe()
    gdf.to_file(function_tmpdir / f"{name}_bas6.shp")

    gdf = m.lpf.to_geodataframe()
    gdf.to_file(function_tmpdir / f"{name}_lpf.shp")

    gdf = m.to_geodataframe()
    gdf.to_file(function_tmpdir / f"{name}.shp")


@pytest.mark.slow
@requires_exe("mfusg", "gridgen")
@requires_pkg("shapely")
def test_gridgen(function_tmpdir):
    # define the base grid and then create a couple levels of nested
    # refinement
    Lx = 10000.0
    Ly = 10500.0
    nlay = 3
    nrow = 21
    ncol = 20
    delr = Lx / ncol
    delc = Ly / nrow
    top = 400
    botm = [220, 200, np.random.random((nrow, ncol))]

    # create a dummy dis package for gridgen
    ms = flopy.modflow.Modflow()
    dis5 = flopy.modflow.ModflowDis(
        ms,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
    )

    sim = flopy.mf6.MFSimulation()
    gwf = flopy.mf6.ModflowGwf(sim)
    dis6 = flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
    )

    ms_u = flopy.mfusg.MfUsg(
        modelname="mymfusgmodel",
        model_ws=function_tmpdir,
    )
    dis_usg = flopy.modflow.ModflowDis(
        ms_u,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
    )

    gridgen = Path(which("gridgen")).name
    ws = function_tmpdir
    g = Gridgen(ms.modelgrid, model_ws=ws, exe_name=gridgen)
    g6 = Gridgen(gwf.modelgrid, model_ws=ws, exe_name=gridgen)
    gu = Gridgen(
        ms_u.modelgrid,
        model_ws=ws,
        exe_name=gridgen,
        vertical_pass_through=True,
    )

    # skip remainder if pyshp is not installed
    if not has_pkg("shapefile"):
        return

    rf0shp = os.path.join(ws, "rf0")
    xmin = 7 * delr
    xmax = 12 * delr
    ymin = 8 * delc
    ymax = 13 * delc
    rfpoly = [[[(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)]]]
    g.add_refinement_features(rfpoly, "polygon", 1, range(nlay))
    g6.add_refinement_features(rfpoly, "polygon", 1, range(nlay))
    gu.add_refinement_features(rfpoly, "polygon", 1, range(nlay))

    rf1shp = os.path.join(ws, "rf1")
    xmin = 8 * delr
    xmax = 11 * delr
    ymin = 9 * delc
    ymax = 12 * delc
    rfpoly = [[[(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)]]]
    g.add_refinement_features(rfpoly, "polygon", 2, range(nlay))
    g6.add_refinement_features(rfpoly, "polygon", 2, range(nlay))
    gu.add_refinement_features(rfpoly, "polygon", 2, range(nlay))

    rf2shp = os.path.join(ws, "rf2")
    xmin = 9 * delr
    xmax = 10 * delr
    ymin = 10 * delc
    ymax = 11 * delc
    rfpoly = [[[(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)]]]
    g.add_refinement_features(rfpoly, "polygon", 3, range(nlay))
    g6.add_refinement_features(rfpoly, "polygon", 3, range(nlay))
    gu.add_refinement_features(rfpoly, "polygon", 3, range(nlay))

    # deactivate parts of mfusg layer 2 to test vertical-pass-through option
    xmin = 0 * delr
    xmax = 18 * delr
    ymin = 0 * delc
    ymax = 18 * delc
    adpoly2 = [[[(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)]]]
    gu.add_active_domain(adpoly2, layers=[1])
    adpoly1_3 = [[[(0.0, 0.0), (Lx, 0.0), (Lx, Ly), (0.0, Ly), (0.0, 0.0)]]]
    gu.add_active_domain(adpoly1_3, layers=[0, 2])

    g.build()
    g6.build()

    # test the different gridprops dictionaries, which contain all the
    # information needed to make the different discretization packages
    gridprops = g.get_gridprops_disv()
    gridprops = g.get_gridprops_disu5()
    gridprops = g.get_gridprops_disu6()

    # test the gridgen point intersection
    points = [(4750.0, 5250.0)]
    cells = g.intersect(points, "point", 0)
    n = cells["nodenumber"][0]
    msg = f"gridgen point intersect did not identify the correct cell {n} <> 308"
    assert n == 308, msg

    # test the gridgen line intersection
    line = [[(Lx, Ly), (Lx, 0.0)]]
    cells = g.intersect(line, "line", 0)
    nlist = list(cells["nodenumber"])
    nlist2 = [
        19,
        650,
        39,
        630,
        59,
        610,
        79,
        590,
        99,
        570,
        119,
        550,
        139,
        530,
        159,
        510,
        194,
        490,
        265,
        455,
        384,
    ]
    msg = "gridgen line intersect did not identify the correct cells {} <> {}".format(
        nlist, nlist2
    )
    assert nlist == nlist2, msg

    # test getting a modflow-usg disu package
    mu = flopy.mfusg.MfUsg(structured=False)
    disu = g.get_disu(mu)

    # test mfusg with vertical pass-through (True above at instantiation)
    gu.build()
    disu_vp = gu.get_disu(ms_u)
    #  -check that node 1 (layer 1) is connected to layer 3 but not layer 2:
    ja0 = disu_vp.ja[: disu_vp.iac[0]]
    msg = (
        "MFUSG node 1 (layer 1) is not connected to layer 3 but should "
        "be (with vertical pass through activated)."
    )
    assert max(ja0) > sum(disu_vp.nodelay[:2]), msg
    #  -check that node 1 (layer 1) is not connected to any layer 2 nodes
    msg = (
        "MFUSG node 1 (layer 1) is connected to layer 2 but should not "
        "be (with vertical pass through activated)."
    )
    assert (
        len(ja0[(ja0 > disu_vp.nodelay[0]) & (ja0 <= sum(disu_vp.nodelay[:2]))]) == 0
    ), msg

    # test mfusg without vertical pass-through
    gu.vertical_pass_through = False
    gu.build()
    disu_vp = gu.get_disu(ms_u)
    #  -check that node 1 (layer 1) is connected to layer 1 only:
    ja0 = disu_vp.ja[: disu_vp.iac[0]]
    msg = (
        "MFUSG node 1 (layer 1) is connected to layer 2 or 3 but "
        "should not be (without vertical pass through activated)."
    )
    assert max(ja0) <= disu_vp.nodelay[0], msg


@requires_exe("mf6", "gridgen")
@requires_pkg("shapely", "pyshp", name_map={"pyshp": "shapefile"})
def test_flopy_issue_1492(function_tmpdir):
    """
    Submitted by David Brakenhoff in
    https://github.com/modflowpy/flopy/issues/1492
    """

    name = "issue1492"
    nlay = 3
    nrow = 10
    ncol = 10
    delr = delc = 1.1  # <-- 1.0 converges
    top = 1
    bot = 0
    dz = (top - bot) / nlay
    botm = [top - k * dz for k in range(1, nlay + 1)]

    # Create a dummy model and regular grid to use as the base grid for gridgen
    sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=function_tmpdir, exe_name="mf6")
    gwf = flopy.mf6.ModflowGwf(sim, modelname=name)
    dis = flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
    )
    og_grid = gwf.modelgrid

    # Create and build the gridgen model
    g = Gridgen(dis, model_ws=function_tmpdir)
    g.build()

    # retrieve a dictionary of arguments to be passed
    # directly into the flopy disv constructor
    disv_gridprops = g.get_gridprops_disv()

    # find the cell numbers for constant heads
    chdspd = []
    ilay = 0
    for x, y, head in [(0, 10, 1.0), (10, 0, 0.0)]:
        ra = g.intersect([(x, y)], "point", ilay)
        ic = ra["nodenumber"][0]
        chdspd.append([(ilay, ic), head])

    # build run and post-process the MODFLOW 6 model
    sim = flopy.mf6.MFSimulation(
        sim_name=name,
        sim_ws=function_tmpdir,
        exe_name="mf6",
        verbosity_level=0,
    )
    tdis = flopy.mf6.ModflowTdis(sim)
    ims = flopy.mf6.ModflowIms(sim, linear_acceleration="bicgstab")
    gwf = flopy.mf6.ModflowGwf(sim, modelname=name, save_flows=True)
    disv = flopy.mf6.ModflowGwfdisv(gwf, **disv_gridprops)
    ic = flopy.mf6.ModflowGwfic(gwf)
    npf = flopy.mf6.ModflowGwfnpf(gwf, xt3doptions=True, save_specific_discharge=True)
    chd = flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chdspd)
    budget_file = name + ".bud"
    head_file = name + ".hds"
    oc = flopy.mf6.ModflowGwfoc(
        gwf,
        budget_filerecord=budget_file,
        head_filerecord=head_file,
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )

    sim.write_simulation()
    success, _ = sim.run_simulation(silent=False)
    assert success

    # debugging duplicate vertices
    grid = gwf.modelgrid
    og_verts = pd.DataFrame(og_grid.verts, columns=["x", "y"])
    mg_verts = pd.DataFrame(grid.verts, columns=["x", "y"])

    plot_debug = False
    if plot_debug:
        head = gwf.output.head().get_data()
        bud = gwf.output.budget()
        spdis = bud.get_data(text="DATA-SPDIS")[0]
        pmv = flopy.plot.PlotMapView(gwf)
        pmv.plot_array(head)
        pmv.plot_grid(colors="white")
        ax = plt.gca()
        verts = grid.verts
        ax.plot(verts[:, 0], verts[:, 1], "bo", alpha=0.25, ms=5)
        pmv.contour_array(head, levels=[0.2, 0.4, 0.6, 0.8], linewidths=3.0)
        pmv.plot_vector(spdis["qx"], spdis["qy"], color="white")
        plt.show()


def build_gnc_gridgen(ws, layers=None, nlay=3):
    """Build a gridgen grid with a refined block in the middle"""
    from shapely.geometry import Polygon

    nrow = ncol = 10
    top = 1.0
    dz = top / nlay
    botm = [top - k * dz for k in range(1, nlay + 1)]

    sim = flopy.mf6.MFSimulation(sim_name="base", sim_ws=ws)
    gwf = flopy.mf6.ModflowGwf(sim, modelname="base")
    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=1.0,
        delc=1.0,
        top=top,
        botm=botm,
    )

    g = Gridgen(gwf.modelgrid, model_ws=ws)
    polys = [Polygon([(4, 4), (6, 4), (6, 6), (4, 6)])]
    g.add_refinement_features(
        polys, "polygon", 3, range(nlay) if layers is None else layers
    )
    g.build()
    return g


@pytest.mark.parametrize("nrec", [0, 1, 3])
def test_read_qtg_gnc_dat(function_tmpdir, nrec):
    lines = [
        "89\t125\t88\t88\t0.125\t0.125",
        "129\t128\t174\t175\t0.166667\t0.166667",
        "163\t164\t124\t124\t0.125\t0.125",
    ][:nrec]
    (function_tmpdir / "qtg.gnc.dat").write_text("\n".join(lines))

    gnc = Gridgen.read_qtg_gnc_dat(function_tmpdir)

    assert gnc.dtype.names == ("n", "m", "j0", "j1", "alpha0", "alpha1")
    assert len(gnc) == nrec

    if nrec > 0:
        # node numbers are converted to zero-based, alphas are not modified
        assert gnc["n"][0] == 88
        assert gnc["m"][0] == 124
        assert gnc["j0"][0] == gnc["j1"][0] == 87
        assert gnc["alpha0"][0] == gnc["alpha1"][0] == 0.125
    if nrec > 1:
        assert gnc["j0"][1] == 173
        assert gnc["j1"][1] == 174
        assert np.allclose(gnc["alpha1"][1], 0.166667)


@requires_exe("gridgen")
@requires_pkg("shapely", "geopandas")
def test_gnc_data(function_tmpdir):
    g = build_gnc_gridgen(function_tmpdir)
    gnc = g.get_gnc()

    # one record per line of the file gridgen wrote
    nlines = len(
        [
            line
            for line in (function_tmpdir / "qtg.gnc.dat").read_text().splitlines()
            if line.strip()
        ]
    )
    assert len(gnc) == nlines > 0

    nodes = g.get_nodes()
    for name in ("n", "m", "j0", "j1"):
        assert gnc[name].min() >= 0
        assert gnc[name].max() < nodes

    # the ghost node is always in the coarser of the two cells
    area = g.get_area()
    assert np.all(area[gnc["n"]] > area[gnc["m"]])

    # contributing factors must sum to less than one
    assert np.all(gnc["alpha0"] + gnc["alpha1"] < 1.0)

    # n must be connected to m, and each j must be connected to n
    iac = g.get_iac()
    ia = get_ia_from_iac(iac)
    ja = g.get_ja(iac.sum())
    for rec in gnc:
        neighbors = ja[ia[rec["n"]] : ia[rec["n"] + 1]]
        assert rec["m"] in neighbors
        assert rec["j0"] in neighbors
        assert rec["j1"] in neighbors


@requires_exe("gridgen")
@requires_pkg("shapely", "geopandas")
def test_gridprops_gnc6_disv(function_tmpdir):
    g = build_gnc_gridgen(function_tmpdir)
    gnc = g.get_gnc()
    gridprops = g.get_gridprops_gnc6(dis_type="disv")

    assert gridprops["numalphaj"] == 2
    assert gridprops["numgnc"] == len(gnc) == len(gridprops["gncdata"])

    ncpl = g.get_gridprops_disv()["ncpl"]
    nlay = g.get_nlay()
    for rec, (cellidn, cellidm, j0, j1, alpha0, alpha1) in zip(
        gnc, gridprops["gncdata"]
    ):
        for node, cellid in zip(
            (rec["n"], rec["m"], rec["j0"], rec["j1"]), (cellidn, cellidm, j0, j1)
        ):
            assert cellid == (node // ncpl, node % ncpl)
            assert 0 <= cellid[0] < nlay
            assert 0 <= cellid[1] < ncpl
        # gridgen only computes horizontal corrections
        assert cellidn[0] == cellidm[0] == j0[0] == j1[0]
        assert (alpha0, alpha1) == (rec["alpha0"], rec["alpha1"])


@requires_exe("gridgen")
@requires_pkg("shapely", "geopandas")
def test_gridprops_gnc6_disu(function_tmpdir):
    # refining a single layer gives a different number of nodes per layer
    g = build_gnc_gridgen(function_tmpdir, layers=[0])
    gnc = g.get_gnc()
    gridprops = g.get_gridprops_gnc6(dis_type="disu")

    assert gridprops["numalphaj"] == 2
    assert gridprops["numgnc"] == len(gnc)
    for rec, (cellidn, cellidm, j0, j1, _, _) in zip(gnc, gridprops["gncdata"]):
        assert (cellidn, cellidm, j0, j1) == (
            (rec["n"],),
            (rec["m"],),
            (rec["j0"],),
            (rec["j1"],),
        )

    # disv cellids cannot be built when nodes per layer are not constant
    nodelay = g.get_nodelay()
    assert nodelay.min() != nodelay.max()
    with pytest.raises(ValueError, match="not the same for all layers"):
        g.get_gridprops_gnc6(dis_type="disv")


@requires_exe("gridgen")
@requires_pkg("shapely", "geopandas")
def test_gridprops_gnc6_invalid(function_tmpdir):
    g = build_gnc_gridgen(function_tmpdir, nlay=1)

    with pytest.raises(ValueError, match="Unknown dis_type"):
        g.get_gridprops_gnc6(dis_type="dis")

    # n and m must be connected
    (function_tmpdir / "qtg.gnc.dat").write_text("1\t400\t2\t2\t0.125\t0.125\n")
    with pytest.raises(ValueError, match="is not connected to cell"):
        g.get_gridprops_gnc6(dis_type="disv")
    assert g.get_gridprops_gnc6(dis_type="disv", check=False)["numgnc"] == 1

    # contributing factors must sum to less than one
    (function_tmpdir / "qtg.gnc.dat").write_text("24\t34\t23\t23\t0.6\t0.6\n")
    with pytest.raises(ValueError, match="must be less than one"):
        g.get_gridprops_gnc6(dis_type="disv")


@requires_exe("gridgen")
@requires_pkg("shapely", "geopandas")
def test_gridprops_gnc5(function_tmpdir):
    g = build_gnc_gridgen(function_tmpdir, nlay=1)
    gnc = g.get_gnc()
    gridprops = g.get_gridprops_gnc5()

    assert gridprops["numalphaj"] == 2
    assert gridprops["numgnc"] == len(gnc)
    # gridgen writes contributing factors, not conductances
    assert gridprops["iflalphan"] == 0
    assert gridprops["i2kn"] == 0
    assert gridprops["isymgncn"] == 0

    gncdata = gridprops["gncdata"]
    assert gncdata.dtype == flopy.mfusg.MfUsgGnc.get_default_dtype(2, 0)
    assert np.array_equal(gncdata["NodeN"], gnc["n"])
    assert np.array_equal(gncdata["NodeM"], gnc["m"])
    assert np.array_equal(gncdata["Node0"], gnc["j0"])
    assert np.array_equal(gncdata["Node1"], gnc["j1"])
    assert np.allclose(gncdata["Alpha0"], gnc["alpha0"])
    assert np.allclose(gncdata["Alpha1"], gnc["alpha1"])

    gridprops = g.get_gridprops_gnc5(i2kn=1, isymgncn=1)
    assert gridprops["i2kn"] == 1
    assert gridprops["isymgncn"] == 1


@pytest.mark.slow
@requires_exe("mf6", "gridgen")
@requires_pkg("shapely", "geopandas")
def test_mf6disv_gnc(function_tmpdir):
    g = build_gnc_gridgen(function_tmpdir)
    disv_gridprops = g.get_gridprops_disv()
    gnc_gridprops = g.get_gridprops_gnc6(dis_type="disv")
    assert gnc_gridprops["numgnc"] > 0

    chdspd = []
    for x, y, head in [(0, 10, 1.0), (10, 0, 0.0)]:
        ra = g.intersect([(x, y)], "point", 0)
        chdspd.append([(0, ra["nodenumber"][0]), head])

    def run(tag, gnc=False, xt3d=False):
        ws = function_tmpdir / tag
        sim = flopy.mf6.MFSimulation(sim_name="m", sim_ws=ws, exe_name="mf6")
        flopy.mf6.ModflowTdis(sim)
        flopy.mf6.ModflowIms(
            sim,
            linear_acceleration="bicgstab",
            inner_dvclose=1e-9,
            outer_dvclose=1e-9,
        )
        gwf = flopy.mf6.ModflowGwf(sim, modelname="m", save_flows=True)
        flopy.mf6.ModflowGwfdisv(gwf, **disv_gridprops)
        flopy.mf6.ModflowGwfic(gwf)
        flopy.mf6.ModflowGwfnpf(gwf, xt3doptions=xt3d)
        flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chdspd)
        flopy.mf6.ModflowGwfoc(
            gwf, head_filerecord="m.hds", saverecord=[("HEAD", "ALL")]
        )
        if gnc:
            flopy.mf6.ModflowGwfgnc(gwf, **gnc_gridprops)
        sim.write_simulation()
        success, buff = sim.run_simulation(silent=True)
        assert success, "\n".join(buff[-25:])
        return gwf.output.head().get_data().flatten()

    head_none = run("none")
    head_gnc = run("gnc", gnc=True)
    head_xt3d = run("xt3d", xt3d=True)

    # the correction must move the solution toward the xt3d solution
    err_none = np.abs(head_none - head_xt3d).max()
    err_gnc = np.abs(head_gnc - head_xt3d).max()
    assert err_gnc < err_none / 5.0, f"gnc {err_gnc} vs uncorrected {err_none}"


@pytest.mark.slow
@requires_exe("mfusg", "gridgen")
@requires_pkg("shapely", "geopandas")
def test_mfusg_gnc(function_tmpdir):
    g = build_gnc_gridgen(function_tmpdir, nlay=1)
    disu_gridprops = g.get_gridprops_disu5()
    gnc_gridprops = g.get_gridprops_gnc5()
    assert gnc_gridprops["numgnc"] > 0

    chdspd = []
    for x, y, head in [(0, 10, 1.0), (10, 0, 0.0)]:
        ra = g.intersect([(x, y)], "point", 0)
        chdspd.append([ra["nodenumber"][0], head, head])

    def run(tag, gnc=False):
        ws = function_tmpdir / tag
        m = flopy.mfusg.MfUsg(
            modelname="m", model_ws=ws, exe_name="mfusg", structured=False
        )
        flopy.mfusg.MfUsgDisU(m, **disu_gridprops)
        flopy.mfusg.MfUsgBas(m)
        flopy.mfusg.MfUsgLpf(m)
        flopy.modflow.ModflowChd(m, stress_period_data=chdspd)
        flopy.mfusg.MfUsgSms(m, options="COMPLEX")
        flopy.modflow.ModflowOc(m, stress_period_data={(0, 0): ["save head"]})
        if gnc:
            flopy.mfusg.MfUsgGnc(m, **gnc_gridprops)
        m.write_input()
        success, buff = m.run_model(silent=True)
        assert success, "\n".join(buff[-25:])
        return np.concatenate(flopy.utils.HeadUFile(ws / "m.hds").get_data())

    head_none = run("none")
    head_gnc = run("gnc", gnc=True)
    assert np.abs(head_none - head_gnc).max() > 0.0

    # the written package must round trip gridgen's one-based node numbers
    written = np.genfromtxt(function_tmpdir / "gnc" / "m.gnc", skip_header=2)
    expected = np.genfromtxt(function_tmpdir / "qtg.gnc.dat")
    assert np.array_equal(written[:, :4], expected[:, :4])
    assert np.allclose(written[:, 4:], expected[:, 4:], atol=1e-6)
