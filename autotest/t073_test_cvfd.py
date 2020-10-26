import numpy as np
import flopy
from flopy.utils.cvfdutil import to_cvfd, gridlist_to_disv_gridprops


def test_tocvfd1():
    vertdict = {}
    vertdict[0] = [(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)]
    vertdict[1] = [(100, 0), (120, 0), (120, 20), (100, 20), (100, 0)]
    verts, iverts = to_cvfd(vertdict)
    assert 6 in iverts[0]


def test_tocvfd2():
    vertdict = {}
    vertdict[0] = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
    vertdict[1] = [(1, 0), (3, 0), (3, 2), (1, 2), (1, 0)]
    verts, iverts = to_cvfd(vertdict)
    assert [1, 4, 5, 6, 2, 1] in iverts


def test_tocvfd3():
    # create the nested grid described in the modflow-usg documentation

    # outer grid
    nlay = 1
    nrow = ncol = 7
    delr = 100.0 * np.ones(ncol)
    delc = 100.0 * np.ones(nrow)
    tp = np.zeros((nrow, ncol))
    bt = -100.0 * np.ones((nlay, nrow, ncol))
    idomain = np.ones((nlay, nrow, ncol))
    idomain[:, 2:5, 2:5] = 0
    sg1 = flopy.discretization.StructuredGrid(
        delr=delr, delc=delc, top=tp, botm=bt, idomain=idomain
    )
    # inner grid
    nlay = 1
    nrow = ncol = 9
    delr = 100.0 / 3.0 * np.ones(ncol)
    delc = 100.0 / 3.0 * np.ones(nrow)
    tp = np.zeros((nrow, ncol))
    bt = -100 * np.ones((nlay, nrow, ncol))
    idomain = np.ones((nlay, nrow, ncol))
    sg2 = flopy.discretization.StructuredGrid(
        delr=delr,
        delc=delc,
        top=tp,
        botm=bt,
        xoff=200.0,
        yoff=200,
        idomain=idomain,
    )
    gridprops = gridlist_to_disv_gridprops([sg1, sg2])
    assert "ncpl" in gridprops
    assert "nvert" in gridprops
    assert "vertices" in gridprops
    assert "cell2d" in gridprops

    ncpl = gridprops["ncpl"]
    nvert = gridprops["nvert"]
    vertices = gridprops["vertices"]
    cell2d = gridprops["cell2d"]
    assert ncpl == 121
    assert nvert == 148
    assert len(vertices) == nvert
    assert len(cell2d) == 121

    # spot check information for cell 28 (zero based)
    answer = [28, 250.0, 150.0, 7, 38, 142, 143, 45, 46, 44, 38]
    for i, j in zip(cell2d[28], answer):
        assert i == j, "{} not equal {}".format(i, j)


if __name__ == "__main__":
    test_tocvfd1()
    test_tocvfd2()
    test_tocvfd3()
