"""
Disv and Disu unstructured grid tests using gridgen and the underlying
flopy.discretization grid classes

"""

import os
import sys
import platform
import shutil
import numpy as np
try:
    from shapely.geometry import Polygon
except ImportWarning as e:
    print("Shapely not installed, tests cannot be run.")
    Polygon = None


import flopy
from flopy.utils.gridgen import Gridgen

try:
    import matplotlib
    import matplotlib.pyplot as plt
except:
    print("Matplotlib not installed, tests cannot be run.")
    matplotlib = None
    plt = None

# Set gridgen executable
gridgen_exe = 'gridgen'
if platform.system() in "Windows":
    gridgen_exe += '.exe'
gridgen_exe = flopy.which(gridgen_exe)

# set mf6 executable
mf6_exe = 'mf6'
if platform.system() in "Windows":
    mf6_exe += '.exe'
mf6_exe = flopy.which(mf6_exe)

# set mfusg executable
mfusg_exe = 'mfusg'
if platform.system() in "Windows":
    mfusg_exe += '.exe'
mfusg_exe = flopy.which(mfusg_exe)

# set up the example folder
tpth = os.path.join('temp', 't506')
if not os.path.isdir(tpth):
    os.makedirs(tpth)

# set up a gridgen workspace
gridgen_ws = os.path.join(tpth, 'gridgen')
if not os.path.exists(gridgen_ws):
    os.makedirs(gridgen_ws)

VERBOSITY_LEVEL = 0


def test_mf6disv():

    if gridgen_exe is None:
        print('Unable to run test_mf6disv(). Gridgen executable not available.')
        return

    if Polygon is None:
        print('Unable to run test_mf6disv(). shapely is not available.')
        return

    name = 'dummy'
    nlay = 3
    nrow = 10
    ncol = 10
    delr = delc = 1.
    top = 1
    bot = 0
    dz = (top - bot) / nlay
    botm = [top - k * dz for k in range(1, nlay + 1)]

    # Create a dummy model and regular grid to use as the base grid for gridgen
    sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=gridgen_ws,
                                 exe_name='mf6')
    gwf = flopy.mf6.ModflowGwf(sim, modelname=name)

    dis = flopy.mf6.ModflowGwfdis(gwf, nlay=nlay, nrow=nrow, ncol=ncol,
                                  delr=delr, delc=delc,
                                  top=top, botm=botm)

    # Create and build the gridgen model with a refined area in the middle
    g = Gridgen(dis, model_ws=gridgen_ws)
    polys = [Polygon([(4, 4), (6, 4), (6, 6), (4, 6)])]
    g.add_refinement_features(polys, 'polygon', 3, range(nlay))
    g.build()
    disv_gridprops = g.get_gridprops_disv()

    # find the cell numbers for constant heads
    chdspd = []
    ilay = 0
    for x, y, head in [(0, 10, 1.), (10, 0, 0.)]:
        ra = g.intersect([(x, y)], 'point', ilay)
        ic = ra['nodenumber'][0]
        chdspd.append([(ilay, ic), head])

    # build run and post-process the MODFLOW 6 model
    ws = os.path.join(tpth, 'gridgen_disv')
    name = 'mymodel'
    sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=ws, exe_name='mf6',
                                 verbosity_level=VERBOSITY_LEVEL)
    tdis = flopy.mf6.ModflowTdis(sim)
    ims = flopy.mf6.ModflowIms(sim, linear_acceleration='bicgstab')
    gwf = flopy.mf6.ModflowGwf(sim, modelname=name, save_flows=True)
    disv = flopy.mf6.ModflowGwfdisv(gwf, **disv_gridprops)
    ic = flopy.mf6.ModflowGwfic(gwf)
    npf = flopy.mf6.ModflowGwfnpf(gwf, xt3doptions=True,
                                  save_specific_discharge=True)
    chd = flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chdspd)
    budget_file = name + '.bud'
    head_file = name + '.hds'
    oc = flopy.mf6.ModflowGwfoc(gwf,
                                budget_filerecord=budget_file,
                                head_filerecord=head_file,
                                saverecord=[('HEAD', 'ALL'), ('BUDGET', 'ALL')])
    sim.write_simulation()

    gwf.modelgrid.set_coord_info(angrot=15)

    # write grid and model shapefiles
    fname = os.path.join(ws, 'grid.shp')
    gwf.modelgrid.write_shapefile(fname)
    fname = os.path.join(ws, 'model.shp')
    gwf.export(fname)

    if mf6_exe is not None:
        sim.run_simulation(silent=True)
        head = flopy.utils.HeadFile(os.path.join(ws, head_file)).get_data()
        bud = flopy.utils.CellBudgetFile(os.path.join(ws, budget_file),
                                         precision='double')
        spdis = bud.get_data(text='DATA-SPDIS')[0]

        if matplotlib is not None:
            f = plt.figure(figsize=(10, 10))
            vmin = head.min()
            vmax = head.max()
            for ilay in range(gwf.modelgrid.nlay):
                ax = plt.subplot(1, gwf.modelgrid.nlay, ilay + 1)
                pmv = flopy.plot.PlotMapView(gwf, layer=ilay, ax=ax)
                ax.set_aspect('equal')
                pmv.plot_array(head.flatten(), cmap='jet', vmin=vmin,
                               vmax=vmax)
                pmv.plot_grid(colors='k', alpha=0.1)
                pmv.contour_array(head, levels=[.2, .4, .6, .8], linewidths=3.,
                                  vmin=vmin, vmax=vmax)
                ax.set_title("Layer {}".format(ilay + 1))
                pmv.plot_specific_discharge(spdis, color='white')
            fname = 'results.png'
            fname = os.path.join(ws, fname)
            plt.savefig(fname)
            plt.close('all')

    return


def test_mf6disu():

    name = 'dummy'
    nlay = 3
    nrow = 10
    ncol = 10
    delr = delc = 1.
    top = 1
    bot = 0
    dz = (top - bot) / nlay
    botm = [top - k * dz for k in range(1, nlay + 1)]

    # Create a dummy model and regular grid to use as the base grid for gridgen
    sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=gridgen_ws,
                                 exe_name='mf6')
    gwf = flopy.mf6.ModflowGwf(sim, modelname=name)

    dis = flopy.mf6.ModflowGwfdis(gwf, nlay=nlay, nrow=nrow, ncol=ncol,
                                  delr=delr, delc=delc,
                                  top=top, botm=botm)

    # Create and build the gridgen model with a refined area in the middle
    g = Gridgen(dis, model_ws=gridgen_ws)
    polys = [Polygon([(4, 4), (6, 4), (6, 6), (4, 6)])]
    g.add_refinement_features(polys, 'polygon', 3, layers=[0])
    g.build()
    disu_gridprops = g.get_gridprops_disu6()

    chdspd = []
    for x, y, head in [(0, 10, 1.), (10, 0, 0.)]:
        ra = g.intersect([(x, y)], 'point', 0)
        ic = ra['nodenumber'][0]
        chdspd.append([(ic,), head])

    # build run and post-process the MODFLOW 6 model
    ws = os.path.join(tpth, 'gridgen_disu')
    name = 'mymodel'
    sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=ws, exe_name='mf6',
                                 verbosity_level=VERBOSITY_LEVEL)
    tdis = flopy.mf6.ModflowTdis(sim)
    ims = flopy.mf6.ModflowIms(sim, linear_acceleration='bicgstab')
    gwf = flopy.mf6.ModflowGwf(sim, modelname=name, save_flows=True)
    disu = flopy.mf6.ModflowGwfdisu(gwf, **disu_gridprops)
    ic = flopy.mf6.ModflowGwfic(gwf)
    npf = flopy.mf6.ModflowGwfnpf(gwf, xt3doptions=True,
                                  save_specific_discharge=True)
    chd = flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chdspd)
    budget_file = name + '.bud'
    head_file = name + '.hds'
    oc = flopy.mf6.ModflowGwfoc(gwf,
                                budget_filerecord=budget_file,
                                head_filerecord=head_file,
                                saverecord=[('HEAD', 'ALL'), ('BUDGET', 'ALL')])
    sim.write_simulation()

    gwf.modelgrid.set_coord_info(angrot=15)

    # The flopy Gridgen object includes the plottable layer number to the
    # diagonal position in the ihc array.  This is why and how modelgrid.nlay
    # is set to 3 and ncpl has a different number of cells per layer.
    assert gwf.modelgrid.nlay == 3
    assert np.allclose(gwf.modelgrid.ncpl, np.array([436, 184, 112]))

    # write grid and model shapefiles
    fname = os.path.join(ws, 'grid.shp')
    gwf.modelgrid.write_shapefile(fname)
    fname = os.path.join(ws, 'model.shp')
    gwf.export(fname)

    if mf6_exe is not None:
        sim.run_simulation(silent=True)
        head = flopy.utils.HeadFile(os.path.join(ws, head_file)).get_data()
        bud = flopy.utils.CellBudgetFile(os.path.join(ws, budget_file),
                                         precision='double')
        spdis = bud.get_data(text='DATA-SPDIS')[0]

        if matplotlib is not None:
            f = plt.figure(figsize=(10, 10))
            vmin = head.min()
            vmax = head.max()
            for ilay in range(gwf.modelgrid.nlay):
                ax = plt.subplot(1, gwf.modelgrid.nlay, ilay + 1)
                pmv = flopy.plot.PlotMapView(gwf, layer=ilay, ax=ax)
                ax.set_aspect('equal')
                pmv.plot_array(head.flatten(), cmap='jet', vmin=vmin,
                               vmax=vmax)
                pmv.plot_grid(colors='k', alpha=0.1)
                pmv.contour_array(head, levels=[.2, .4, .6, .8], linewidths=3.,
                                  vmin=vmin, vmax=vmax)
                ax.set_title("Layer {}".format(ilay + 1))
                pmv.plot_specific_discharge(spdis, color='white')
            fname = 'results.png'
            fname = os.path.join(ws, fname)
            plt.savefig(fname)
            plt.close('all')

    return


def test_mfusg():

    name = 'dummy'
    nlay = 3
    nrow = 10
    ncol = 10
    delr = delc = 1.
    top = 1
    bot = 0
    dz = (top - bot) / nlay
    botm = [top - k * dz for k in range(1, nlay + 1)]

    # create dummy model and dis package for gridgen
    m = flopy.modflow.Modflow(modelname=name, model_ws=gridgen_ws)
    dis = flopy.modflow.ModflowDis(m, nlay=nlay, nrow=nrow, ncol=ncol,
                                   delr=delr, delc=delc,
                                   top=top, botm=botm)

    # Create and build the gridgen model with a refined area in the middle
    g = Gridgen(dis, model_ws=gridgen_ws)
    polys = [Polygon([(4, 4), (6, 4), (6, 6), (4, 6)])]
    g.add_refinement_features(polys, 'polygon', 3, layers=[0])
    g.build()

    chdspd = []
    for x, y, head in [(0, 10, 1.), (10, 0, 0.)]:
        ra = g.intersect([(x, y)], 'point', 0)
        ic = ra['nodenumber'][0]
        chdspd.append([ic, head, head])

    #gridprops = g.get_gridprops()
    gridprops = g.get_gridprops_disu5()

    # create the mfusg modoel
    ws = os.path.join(tpth, 'gridgen_mfusg')
    name = 'mymodel'
    m = flopy.modflow.Modflow(modelname=name, model_ws=ws,
                              version='mfusg', exe_name=mfusg_exe,
                              structured=False)
    disu = flopy.modflow.ModflowDisU(m, **gridprops)
    bas = flopy.modflow.ModflowBas(m)
    lpf = flopy.modflow.ModflowLpf(m)
    chd = flopy.modflow.ModflowChd(m, stress_period_data=chdspd)
    sms = flopy.modflow.ModflowSms(m)
    oc = flopy.modflow.ModflowOc(m, stress_period_data={(0, 0): ["save head"]})
    m.write_input()

    # MODFLOW-USG does not have vertices, so we need to create
    # and unstructured grid and then assign it to the model. This
    # will allow plotting and other features to work properly.
    gridprops_ug = g.get_gridprops_unstructuredgrid()
    ugrid = flopy.discretization.UnstructuredGrid(**gridprops_ug, angrot=-15)
    m.modelgrid = ugrid

    if mfusg_exe is not None:
        m.run_model()

        # head is returned as a list of head arrays for each layer
        head_file = os.path.join(ws, name + '.hds')
        head = flopy.utils.HeadUFile(head_file).get_data()

        if matplotlib is not None:
            f = plt.figure(figsize=(10, 10))
            vmin = 0.
            vmax = 1.
            for ilay in range(disu.nlay):
                ax = plt.subplot(1, g.nlay, ilay + 1)
                pmv = flopy.plot.PlotMapView(m, layer=ilay, ax=ax)
                ax.set_aspect('equal')
                pmv.plot_array(head[ilay], cmap='jet', vmin=vmin, vmax=vmax)
                pmv.plot_grid(colors='k', alpha=0.1)
                pmv.contour_array(head[ilay], levels=[.2, .4, .6, .8],
                                  linewidths=3.)
                ax.set_title("Layer {}".format(ilay + 1))
                # pmv.plot_specific_discharge(spdis, color='white')
            fname = 'results.png'
            fname = os.path.join(ws, fname)
            plt.savefig(fname)
            plt.close('all')

        # re-run with an LPF keyword specified. This would have thrown an error
        # before the addition of ikcflag to mflpf.py (flopy 3.3.3 and earlier).
        lpf = flopy.modflow.ModflowLpf(m, novfc = True)
        m.write_input()
        m.run_model()


def test_disv_dot_plot():
    # load up the vertex example problem
    name = "mymodel"
    sim_path = os.path.join(tpth, 'gridgen_disv')
    sim = flopy.mf6.MFSimulation.load(sim_name=name, version="mf6",
                                      exe_name="mf6",
                                      sim_ws=sim_path)
    # get gwf model
    gwf = sim.get_model(name)

    # get the dis package
    dis = gwf.disv

    # try plotting an array
    top = dis.top
    ax = top.plot()
    assert ax
    plt.close('all')

    # try plotting a package
    ax = dis.plot()
    assert ax
    plt.close('all')

    # try plotting a model
    ax = gwf.plot()
    assert ax
    plt.close('all')

    return


def test_disu_dot_plot():
    # load up the disu example problem
    name = "mymodel"
    sim_path = os.path.join(tpth, 'gridgen_disu')
    sim = flopy.mf6.MFSimulation.load(sim_name=name, version="mf6",
                                      exe_name="mf6",
                                      sim_ws=sim_path)
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
    plt.close('all')

    # try plotting a package
    ax = dis.plot()
    assert ax
    plt.close('all')

    # try plotting a model
    ax = gwf.plot()
    assert ax
    plt.close('all')

    return


if __name__ == "__main__":
    test_mf6disv()
    test_mf6disu()
    test_disv_dot_plot()
    test_disu_dot_plot()
    test_mfusg()
