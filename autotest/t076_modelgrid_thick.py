"""
Model grid thick method tests

"""

import numpy as np
from flopy.discretization import StructuredGrid, VertexGrid, UnstructuredGrid


def test_structured_thick():
    nlay, nrow, ncol = 3, 2, 3
    delc = 1.0 * np.ones(nrow, dtype=float)
    delr = 1.0 * np.ones(ncol, dtype=float)
    top = 10.0 * np.ones((nrow, ncol), dtype=float)
    botm = np.zeros((nlay, nrow, ncol), dtype=float)
    botm[0, :, :] = 5.0
    botm[1, :, :] = 0.0
    botm[2, :, :] = -5.0
    grid = StructuredGrid(
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delc=delc,
        delr=delr,
        top=top,
        botm=botm,
    )
    thick = grid.thick
    assert np.allclose(thick, 5.0), "thicknesses != 5."

    sat_thick = grid.saturated_thick(grid.botm + 10.0)
    assert np.allclose(sat_thick, thick), "saturated thicknesses != 5."

    sat_thick = grid.saturated_thick(grid.botm + 5.0)
    assert np.allclose(sat_thick, thick), "saturated thicknesses != 5."

    sat_thick = grid.saturated_thick(grid.botm + 2.5)
    assert np.allclose(sat_thick, 2.5), "saturated thicknesses != 2.5"

    sat_thick = grid.saturated_thick(grid.botm)
    assert np.allclose(sat_thick, 0.0), "saturated thicknesses != 0."

    sat_thick = grid.saturated_thick(grid.botm - 100.0)
    assert np.allclose(sat_thick, 0.0), "saturated thicknesses != 0."

    return


def test_vertices_thick():
    nlay, ncpl = 3, 5
    vertices = [
        [0, 0.0, 3.0],
        [1, 1.0, 3.0],
        [2, 2.0, 3.0],
        [3, 0.0, 2.0],
        [4, 1.0, 2.0],
        [5, 2.0, 2.0],
        [6, 0.0, 1.0],
        [7, 1.0, 1.0],
        [8, 2.0, 1.0],
        [9, 0.0, 0.0],
        [10, 1.0, 0.0],
    ]
    iverts = [
        [0, 0, 1, 4, 3],
        [1, 1, 2, 5, 4],
        [2, 3, 4, 7, 6],
        [3, 4, 5, 8, 7],
        [4, 6, 7, 10, 9],
        [5, 0, 1, 4, 3],
        [6, 1, 2, 5, 4],
        [7, 3, 4, 7, 6],
        [8, 4, 5, 8, 7],
        [9, 6, 7, 10, 9],
        [10, 0, 1, 4, 3],
        [11, 1, 2, 5, 4],
        [12, 3, 4, 7, 6],
        [13, 4, 5, 8, 7],
        [14, 6, 7, 10, 9],
    ]
    top = np.ones(ncpl, dtype=float) * 10.0
    botm = np.zeros((nlay, ncpl), dtype=float)
    botm[0, :] = 5.0
    botm[1, :] = 0.0
    botm[2, :] = -5.0
    grid = VertexGrid(
        nlay=nlay,
        ncpl=ncpl,
        vertices=vertices,
        cell2d=iverts,
        top=top,
        botm=botm,
    )
    thick = grid.thick
    assert np.allclose(thick, 5.0), "thicknesses != 5."

    sat_thick = grid.saturated_thick(grid.botm + 10.0)
    assert np.allclose(sat_thick, thick), "saturated thicknesses != 5."

    sat_thick = grid.saturated_thick(grid.botm + 5.0)
    assert np.allclose(sat_thick, thick), "saturated thicknesses != 5."

    sat_thick = grid.saturated_thick(grid.botm + 2.5)
    assert np.allclose(sat_thick, 2.5), "saturated thicknesses != 2.5"

    sat_thick = grid.saturated_thick(grid.botm)
    assert np.allclose(sat_thick, 0.0), "saturated thicknesses != 0."

    sat_thick = grid.saturated_thick(grid.botm - 100.0)
    assert np.allclose(sat_thick, 0.0), "saturated thicknesses != 0."

    return


def test_unstructured_thick():
    nlay = 3
    ncpl = [5, 5, 5]
    vertices = [
        [0, 0.0, 3.0],
        [1, 1.0, 3.0],
        [2, 2.0, 3.0],
        [3, 0.0, 2.0],
        [4, 1.0, 2.0],
        [5, 2.0, 2.0],
        [6, 0.0, 1.0],
        [7, 1.0, 1.0],
        [8, 2.0, 1.0],
        [9, 0.0, 0.0],
        [10, 1.0, 0.0],
    ]
    iverts = [
        [0, 0, 1, 4, 3],
        [1, 1, 2, 5, 4],
        [2, 3, 4, 7, 6],
        [3, 4, 5, 8, 7],
        [4, 6, 7, 10, 9],
        [5, 0, 1, 4, 3],
        [6, 1, 2, 5, 4],
        [7, 3, 4, 7, 6],
        [8, 4, 5, 8, 7],
        [9, 6, 7, 10, 9],
        [10, 0, 1, 4, 3],
        [11, 1, 2, 5, 4],
        [12, 3, 4, 7, 6],
        [13, 4, 5, 8, 7],
        [14, 6, 7, 10, 9],
    ]
    xcenters = [
        0.5,
        1.5,
        0.5,
        1.5,
        0.5,
    ]
    ycenters = [
        2.5,
        2.5,
        1.5,
        1.5,
        0.5,
    ]
    top = np.ones((nlay, 5), dtype=float)
    top[0, :] = 10.0
    top[1, :] = 5.0
    top[2, :] = 0.0
    botm = np.zeros((nlay, 5), dtype=float)
    botm[0, :] = 5.0
    botm[1, :] = 0.0
    botm[2, :] = -5.0

    grid = UnstructuredGrid(
        vertices=vertices,
        iverts=iverts,
        xcenters=xcenters,
        ycenters=ycenters,
        ncpl=ncpl,
        top=top.flatten(),
        botm=botm.flatten(),
    )

    thick = grid.thick
    assert np.allclose(thick, 5.0), "thicknesses != 5."

    sat_thick = grid.saturated_thick(grid.botm + 10.0)
    assert np.allclose(sat_thick, thick), "saturated thicknesses != 5."

    sat_thick = grid.saturated_thick(grid.botm + 5.0)
    assert np.allclose(sat_thick, thick), "saturated thicknesses != 5."

    sat_thick = grid.saturated_thick(grid.botm + 2.5)
    assert np.allclose(sat_thick, 2.5), "saturated thicknesses != 2.5"

    sat_thick = grid.saturated_thick(grid.botm)
    assert np.allclose(sat_thick, 0.0), "saturated thicknesses != 0."

    sat_thick = grid.saturated_thick(grid.botm - 100.0)
    assert np.allclose(sat_thick, 0.0), "saturated thicknesses != 0."

    return


def test_ncb_thick():
    nlay = 3
    nrow = ncol = 15
    laycbd = np.array([1, 2, 0], dtype=int)
    ncb = np.count_nonzero(laycbd)
    dx = dy = 150
    delc = np.array(
        [
            dy,
        ]
        * nrow
    )
    delr = np.array(
        [
            dx,
        ]
        * ncol
    )
    top = np.ones((15, 15))
    botm = np.ones((nlay + ncb, nrow, ncol))
    elevations = np.array([-10, -20, -40, -50, -70])[:, np.newaxis]
    botm *= elevations[:, None]

    modelgrid = StructuredGrid(
        delc=delc,
        delr=delr,
        top=top,
        botm=botm,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        laycbd=laycbd,
    )

    thick = modelgrid.thick

    if not thick.shape[0] == nlay + ncb:
        raise AssertionError("grid thick attribute returns incorrect shape")

    thick = modelgrid.remove_confining_beds(modelgrid.thick)
    if not thick.shape == modelgrid.shape:
        raise AssertionError("quasi3d confining beds not properly removed")

    sat_thick = modelgrid.saturated_thick(modelgrid.thick)
    if not sat_thick.shape == modelgrid.shape:
        raise AssertionError("saturated_thickness confining beds not removed")

    if sat_thick[1, 0, 0] != 20:
        raise AssertionError(
            "saturated_thickness is not properly indexing confining beds"
        )


if __name__ == "__main__":
    test_unstructured_thick()
    test_vertices_thick()
    test_structured_thick()
    test_ncb_thick()
