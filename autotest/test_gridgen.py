import os
from pathlib import Path
from shutil import which

import matplotlib.pyplot as plt
import numpy as np
import pytest
from autotest.conftest import has_pkg, requires_exe, requires_pkg
from matplotlib.collections import LineCollection, PathCollection, QuadMesh

import flopy
from flopy.utils.gridgen import Gridgen


@pytest.mark.slow
@requires_exe("mf6", "gridgen")
@requires_pkg("shapely")
def test_mf6disv(tmpdir):
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
    sim = flopy.mf6.MFSimulation(
        sim_name=name, sim_ws=str(tmpdir), exe_name="mf6"
    )
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
    g = Gridgen(gwf.modelgrid, model_ws=str(tmpdir))
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
    sim = flopy.mf6.MFSimulation(
        sim_name=name, sim_ws=str(tmpdir), exe_name="mf6"
    )
    tdis = flopy.mf6.ModflowTdis(sim)
    ims = flopy.mf6.ModflowIms(sim, linear_acceleration="bicgstab")
    gwf = flopy.mf6.ModflowGwf(sim, modelname=name, save_flows=True)
    disv = flopy.mf6.ModflowGwfdisv(gwf, **disv_gridprops)
    ic = flopy.mf6.ModflowGwfic(gwf)
    npf = flopy.mf6.ModflowGwfnpf(
        gwf, xt3doptions=True, save_specific_discharge=True
    )
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
    fname = os.path.join(str(tmpdir), "grid.shp")
    gwf.modelgrid.write_shapefile(fname)
    fname = os.path.join(str(tmpdir), "model.shp")
    gwf.export(fname)

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
        fname = os.path.join(str(tmpdir), fname)
        plt.savefig(fname)
        plt.close("all")

    # test plotting
    # load up the vertex example problem
    name = "mymodel"
    sim = flopy.mf6.MFSimulation.load(
        sim_name=name, version="mf6", exe_name="mf6", sim_ws=str(tmpdir)
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


@pytest.mark.slow
@requires_exe("mf6", "gridgen")
@requires_pkg("shapely", "shapefile")
def test_mf6disu(tmpdir):
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
    sim = flopy.mf6.MFSimulation(
        sim_name=name, sim_ws=str(tmpdir), exe_name="mf6"
    )
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
    g = Gridgen(gwf.modelgrid, model_ws=str(tmpdir))
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
    sim = flopy.mf6.MFSimulation(
        sim_name=name, sim_ws=str(tmpdir), exe_name="mf6"
    )
    tdis = flopy.mf6.ModflowTdis(sim)
    ims = flopy.mf6.ModflowIms(sim, linear_acceleration="bicgstab")
    gwf = flopy.mf6.ModflowGwf(sim, modelname=name, save_flows=True)
    disu = flopy.mf6.ModflowGwfdisu(gwf, **disu_gridprops)
    ic = flopy.mf6.ModflowGwfic(gwf)
    npf = flopy.mf6.ModflowGwfnpf(
        gwf, xt3doptions=True, save_specific_discharge=True
    )
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

    # The flopy Gridgen object includes the plottable layer number to the
    # diagonal position in the ihc array.  This is why and how modelgrid.nlay
    # is set to 3 and ncpl has a different number of cells per layer.
    assert gwf.modelgrid.nlay == 3
    assert np.allclose(gwf.modelgrid.ncpl, np.array([436, 184, 112]))

    # write grid and model shapefiles
    fname = os.path.join(str(tmpdir), "grid.shp")
    gwf.modelgrid.write_shapefile(fname)
    fname = os.path.join(str(tmpdir), "model.shp")
    gwf.export(fname)

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
    fname = os.path.join(str(tmpdir), fname)
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
                if not isinstance(
                    col, (QuadMesh, PathCollection, LineCollection)
                ):
                    raise AssertionError("Unexpected collection type")
        plt.close()

    # test plotting
    # load up the disu example problem
    name = "mymodel"
    sim = flopy.mf6.MFSimulation.load(
        sim_name=name, version="mf6", exe_name="mf6", sim_ws=str(tmpdir)
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
@requires_pkg("shapely", "shapefile")
def test_mfusg(tmpdir):
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
    m = flopy.modflow.Modflow(modelname=name, model_ws=str(tmpdir))
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
    g = Gridgen(m.modelgrid, model_ws=str(tmpdir))
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
        model_ws=str(tmpdir),
        exe_name="mfusg",
        structured=False,
    )
    disu = flopy.mfusg.MfUsgDisU(m, **gridprops)
    bas = flopy.modflow.ModflowBas(m)
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
    head_file = os.path.join(str(tmpdir), f"{name}.hds")
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
        pmv.contour_array(
            head[ilay], levels=[0.2, 0.4, 0.6, 0.8], linewidths=3.0
        )
        ax.set_title(f"Layer {ilay + 1}")
        # pmv.plot_specific_discharge(spdis, color='white')
    fname = "results.png"
    fname = os.path.join(str(tmpdir), fname)
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
                if not isinstance(
                    col, (QuadMesh, PathCollection, LineCollection)
                ):
                    raise AssertionError("Unexpected collection type")
        plt.close()

    # re-run with an LPF keyword specified. This would have thrown an error
    # before the addition of ikcflag to mflpf.py (flopy 3.3.3 and earlier).
    lpf = flopy.mfusg.MfUsgLpf(m, novfc=True, nocvcorrection=True)
    m.write_input()
    m.run_model()

    # also test load of unstructured LPF with keywords
    lpf2 = flopy.mfusg.MfUsgLpf.load(
        os.path.join(str(tmpdir), f"{name}.lpf"), m, check=False
    )
    msg = "NOCVCORRECTION and NOVFC should be in lpf options but at least one is not."
    assert (
        "NOVFC" in lpf2.options.upper()
        and "NOCVCORRECTION" in lpf2.options.upper()
    ), msg

    # test disu, bas6, lpf shapefile export for mfusg unstructured models
    m.disu.export(os.path.join(str(tmpdir), f"{name}_disu.shp"))
    m.bas6.export(os.path.join(str(tmpdir), f"{name}_bas6.shp"))
    m.lpf.export(os.path.join(str(tmpdir), f"{name}_lpf.shp"))
    m.export(os.path.join(str(tmpdir), f"{name}.shp"))


@pytest.mark.slow
@requires_exe("mfusg", "gridgen")
@requires_pkg("shapely")
def test_gridgen(tmpdir):
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
    gwf = gwf = flopy.mf6.ModflowGwf(sim)
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
        model_ws=str(tmpdir),
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
    ws = str(tmpdir)
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
    rfpoly = [
        [
            [
                (xmin, ymin),
                (xmax, ymin),
                (xmax, ymax),
                (xmin, ymax),
                (xmin, ymin),
            ]
        ]
    ]
    g.add_refinement_features(rfpoly, "polygon", 1, range(nlay))
    g6.add_refinement_features(rfpoly, "polygon", 1, range(nlay))
    gu.add_refinement_features(rfpoly, "polygon", 1, range(nlay))

    rf1shp = os.path.join(ws, "rf1")
    xmin = 8 * delr
    xmax = 11 * delr
    ymin = 9 * delc
    ymax = 12 * delc
    rfpoly = [
        [
            [
                (xmin, ymin),
                (xmax, ymin),
                (xmax, ymax),
                (xmin, ymax),
                (xmin, ymin),
            ]
        ]
    ]
    g.add_refinement_features(rfpoly, "polygon", 2, range(nlay))
    g6.add_refinement_features(rfpoly, "polygon", 2, range(nlay))
    gu.add_refinement_features(rfpoly, "polygon", 2, range(nlay))

    rf2shp = os.path.join(ws, "rf2")
    xmin = 9 * delr
    xmax = 10 * delr
    ymin = 10 * delc
    ymax = 11 * delc
    rfpoly = [
        [
            [
                (xmin, ymin),
                (xmax, ymin),
                (xmax, ymax),
                (xmin, ymax),
                (xmin, ymin),
            ]
        ]
    ]
    g.add_refinement_features(rfpoly, "polygon", 3, range(nlay))
    g6.add_refinement_features(rfpoly, "polygon", 3, range(nlay))
    gu.add_refinement_features(rfpoly, "polygon", 3, range(nlay))

    # inactivate parts of mfusg layer 2 to test vertical-pass-through option
    xmin = 0 * delr
    xmax = 18 * delr
    ymin = 0 * delc
    ymax = 18 * delc
    adpoly2 = [
        [
            [
                (xmin, ymin),
                (xmax, ymin),
                (xmax, ymax),
                (xmin, ymax),
                (xmin, ymin),
            ]
        ]
    ]
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
    msg = (
        f"gridgen point intersect did not identify the correct cell {n} <> 308"
    )
    assert n == 308, msg

    # test the gridgen line intersection
    line = [[(Lx, Ly), (Lx, 0.0)]]
    cells = g.intersect(line, "line", 0)
    nlist = [n for n in cells["nodenumber"]]
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
    msg = (
        "gridgen line intersect did not identify the correct "
        "cells {} <> {}".format(nlist, nlist2)
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
        len(
            ja0[(ja0 > disu_vp.nodelay[0]) & (ja0 <= sum(disu_vp.nodelay[:2]))]
        )
        == 0
    ), msg
    # ms_u.disu.write_file()

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
