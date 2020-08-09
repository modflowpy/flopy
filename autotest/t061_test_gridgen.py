import shutil
import os
import numpy as np
import flopy
from flopy.utils.gridgen import Gridgen

cpth = os.path.join('temp', 't061')
# delete the directory if it exists
if os.path.isdir(cpth):
    shutil.rmtree(cpth)
# make the directory
os.makedirs(cpth)

exe_name = 'gridgen'
v = flopy.which(exe_name)

run = True
if v is None:
    run = False


def test_gridgen():

    # define the base grid and then create a couple levels of nested
    # refinement
    Lx = 10000.
    Ly = 10500.
    nlay = 3
    nrow = 21
    ncol = 20
    delr = Lx / ncol
    delc = Ly / nrow
    top = 400
    botm = [220, 200, np.random.random((nrow, ncol))]

    # create a dummy dis package for gridgen
    ms = flopy.modflow.Modflow()
    dis5 = flopy.modflow.ModflowDis(ms, nlay=nlay, nrow=nrow, ncol=ncol,
                                    delr=delr,
                                    delc=delc, top=top, botm=botm)

    sim = flopy.mf6.MFSimulation()
    gwf = gwf = flopy.mf6.ModflowGwf(sim)
    dis6 = flopy.mf6.ModflowGwfdis(gwf, nlay=nlay, nrow=nrow, ncol=ncol,
                                    delr=delr,
                                    delc=delc, top=top, botm=botm)

    ms_u = flopy.modflow.Modflow(modelname = 'mymfusgmodel', model_ws = cpth,
                                 version = 'mfusg')
    dis_usg = flopy.modflow.ModflowDis(ms_u, nlay=nlay, nrow=nrow, ncol=ncol,
                                    delr=delr,
                                    delc=delc, top=top, botm=botm)

    gridgen_ws = cpth
    g = Gridgen(dis5, model_ws=gridgen_ws, exe_name=exe_name)
    g6 = Gridgen(dis6, model_ws=gridgen_ws, exe_name=exe_name)
    gu = Gridgen(dis_usg, model_ws=gridgen_ws, exe_name=exe_name,
                 vertical_pass_through=True)

    rf0shp = os.path.join(gridgen_ws, 'rf0')
    xmin = 7 * delr
    xmax = 12 * delr
    ymin = 8 * delc
    ymax = 13 * delc
    rfpoly = [[[(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax),
                (xmin, ymin)]]]
    g.add_refinement_features(rfpoly, 'polygon', 1, range(nlay))
    g6.add_refinement_features(rfpoly, 'polygon', 1, range(nlay))
    gu.add_refinement_features(rfpoly, 'polygon', 1, range(nlay))

    rf1shp = os.path.join(gridgen_ws, 'rf1')
    xmin = 8 * delr
    xmax = 11 * delr
    ymin = 9 * delc
    ymax = 12 * delc
    rfpoly = [[[(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax),
                (xmin, ymin)]]]
    g.add_refinement_features(rfpoly, 'polygon', 2, range(nlay))
    g6.add_refinement_features(rfpoly, 'polygon', 2, range(nlay))
    gu.add_refinement_features(rfpoly, 'polygon', 2, range(nlay))

    rf2shp = os.path.join(gridgen_ws, 'rf2')
    xmin = 9 * delr
    xmax = 10 * delr
    ymin = 10 * delc
    ymax = 11 * delc
    rfpoly = [[[(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax),
                (xmin, ymin)]]]
    g.add_refinement_features(rfpoly, 'polygon', 3, range(nlay))
    g6.add_refinement_features(rfpoly, 'polygon', 3, range(nlay))
    gu.add_refinement_features(rfpoly, 'polygon', 3, range(nlay))

    # inactivate parts of mfusg layer 2 to test vertical-pass-through option
    xmin = 0 * delr
    xmax = 18 * delr
    ymin = 0 * delc
    ymax = 18 * delc
    adpoly2 = [[[(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax),
                (xmin, ymin)]]]
    gu.add_active_domain(adpoly2, layers = [1])
    adpoly1_3 = [[[(0., 0.), (Lx, 0.), (Lx, Ly), (0., Ly),
                (0., 0.)]]]    
    gu.add_active_domain(adpoly1_3, layers = [0,2])

    # if gridgen executable is available then do the main part of the test
    if run:

        # Use gridgen to build the grid
        g.build()
        g6.build()

        # test the different gridprops dictionaries, which contain all the
        # information needed to make the different discretization packages
        gridprops = g.get_gridprops_disv()
        gridprops = g.get_gridprops()
        #gridprops = g.get_gridprops_disu6()

        # test the gridgen point intersection
        points = [(4750., 5250.)]
        cells = g.intersect(points, 'point', 0)
        n = cells['nodenumber'][0]
        msg = ('gridgen point intersect did not identify the correct '
               'cell {} <> {}'.format(n, 308))
        assert n == 308, msg

        # test the gridgen line intersection
        line = [[[(Lx, Ly), (Lx, 0.)]]]
        cells = g.intersect(line, 'line', 0)
        nlist = [n for n in cells['nodenumber']]
        nlist2 = [19, 650, 39, 630, 59, 610, 79, 590, 99, 570, 119, 550, 139,
                  530, 159, 510, 194, 490, 265, 455, 384]
        msg = ('gridgen line intersect did not identify the correct '
               'cells {} <> {}'.format(nlist, nlist2))
        assert nlist == nlist2, msg

        # test getting a modflow-usg disu package
        mu = flopy.modflow.Modflow(version='mfusg', structured=False)
        disu = g.get_disu(mu)

        # test writing a modflow 6 disu package
        fname = os.path.join(cpth, 'mymf6model.disu')
        g6.to_disu6(fname)
        assert os.path.isfile(fname), \
            'MF6 disu file not created: {}'.format(fname)

        # test writing a modflow 6 disv package
        fname = os.path.join(cpth, 'mymf6model.disv')
        g6.to_disv6(fname)
        assert os.path.isfile(fname), \
            'MF6 disv file not created: {}'.format(fname)

        # test mfusg with vertical pass-through (True above at instantiation)
        gu.build()
        disu_vp = gu.get_disu(ms_u)
        #  -check that node 1 (layer 1) is connected to layer 3 but not layer 2:
        ja0 = disu_vp.ja[: disu_vp.iac[0]]
        msg = ("MFUSG node 1 (layer 1) is not connected to layer 3 but should "
               "be (with vertical pass through activated).")
        assert max(ja0) > sum(disu_vp.nodelay[:2]), msg
        #  -check that node 1 (layer 1) is not connected to any layer 2 nodes
        msg = ("MFUSG node 1 (layer 1) is connected to layer 2 but should not "
               "be (with vertical pass through activated).")
        assert len(ja0[(ja0 > disu_vp.nodelay[0]) & \
                       (ja0 <= sum(disu_vp.nodelay[:2]))]
                   ) == 0, msg
        #ms_u.disu.write_file()
        
        # test mfusg without vertical pass-through
        gu.vertical_pass_through = False
        gu.build()
        disu_vp = gu.get_disu(ms_u)
        #  -check that node 1 (layer 1) is connected to layer 1 only:
        ja0 = disu_vp.ja[: disu_vp.iac[0]]
        msg = ("MFUSG node 1 (layer 1) is connected to layer 2 or 3 but "
               "should not be (without vertical pass through activated).")
        assert max(ja0) <= disu_vp.nodelay[0], msg

    return


if __name__ == '__main__':
    test_gridgen()
