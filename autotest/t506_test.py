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

    if mf6_exe is not None:
        sim.run_simulation(silent=True)
        head = flopy.utils.HeadFile(os.path.join(ws, head_file)).get_data()
        bud = flopy.utils.CellBudgetFile(os.path.join(ws, budget_file),
                                         precision='double')
        spdis = bud.get_data(text='DATA-SPDIS')[0]

        if matplotlib is not None:
            pmv = flopy.plot.PlotMapView(gwf)
            pmv.plot_array(head)
            pmv.plot_grid(colors='white')
            pmv.contour_array(head, levels=[.2, .4, .6, .8], linewidths=3.)
            pmv.plot_specific_discharge(spdis, color='white')
            fname = 'results.png'
            fname = os.path.join(ws, fname)
            plt.savefig(fname)

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

    if mf6_exe is not None:
        sim.run_simulation(silent=True)
        head = flopy.utils.HeadFile(os.path.join(ws, head_file)).get_data()
        bud = flopy.utils.CellBudgetFile(os.path.join(ws, budget_file),
                                         precision='double')
        spdis = bud.get_data(text='DATA-SPDIS')[0]

        if matplotlib is not None:
            pmv = flopy.plot.PlotMapView(gwf)
            pmv.plot_array(head.flatten(), cmap='jet')
            pmv.plot_grid(colors='k', alpha=0.1)
            pmv.contour_array(head, levels=[.2, .4, .6, .8], linewidths=3.)
            fname = 'results.png'
            fname = os.path.join(ws, fname)
            plt.savefig(fname)

    return


if __name__ == "__main__":
    test_mf6disv()
    test_mf6disu()
