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

    gridgen_ws = cpth
    g = Gridgen(dis5, model_ws=gridgen_ws, exe_name=exe_name)

    rf0shp = os.path.join(gridgen_ws, 'rf0')
    xmin = 7 * delr
    xmax = 12 * delr
    ymin = 8 * delc
    ymax = 13 * delc
    rfpoly = [[[(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax),
                (xmin, ymin)]]]
    g.add_refinement_features(rfpoly, 'polygon', 1, range(nlay))

    rf1shp = os.path.join(gridgen_ws, 'rf1')
    xmin = 8 * delr
    xmax = 11 * delr
    ymin = 9 * delc
    ymax = 12 * delc
    rfpoly = [[[(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax),
                (xmin, ymin)]]]
    g.add_refinement_features(rfpoly, 'polygon', 2, range(nlay))

    rf2shp = os.path.join(gridgen_ws, 'rf2')
    xmin = 9 * delr
    xmax = 10 * delr
    ymin = 10 * delc
    ymax = 11 * delc
    rfpoly = [[[(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax),
                (xmin, ymin)]]]
    g.add_refinement_features(rfpoly, 'polygon', 3, range(nlay))

    # if gridgen executable is available then do the main part of the test
    if run:

        # Use gridgen to build the grid
        g.build()

        # test the different gridprops dictionaries, which contain all the
        # information needed to make the different discretiztion packages
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

    return


if __name__ == '__main__':
    test_gridgen()
