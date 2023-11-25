import numpy as np

from flopy.modflow import Modflow, ModflowDis


def test_dis_sr():
    delr = 640
    delc = 640
    nrow = np.ceil(59040.0 / delc).astype(int)
    ncol = np.ceil(33128.0 / delr).astype(int)
    nlay = 3

    xul = 2746975.089
    yul = 1171446.45
    rotation = -39
    bg = Modflow(modelname="base")
    dis = ModflowDis(
        bg,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        lenuni=1,
        rotation=rotation,
        xul=xul,
        yul=yul,
        crs="epsg:2243",
    )

    # Use StructuredGrid instead
    x, y = bg.modelgrid.get_coords(0, delc * nrow)
    np.testing.assert_almost_equal(x, xul)
    np.testing.assert_almost_equal(y, yul)
    assert bg.modelgrid.epsg == 2243
    assert bg.modelgrid.angrot == rotation
