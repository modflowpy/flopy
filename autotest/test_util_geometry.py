import numpy as np
import pytest

from autotest.test_grid_cases import GridCases
from flopy.utils.geometry import is_clockwise, point_in_polygon


def test_does_isclockwise_work():
    # Create some points
    verts = []
    verts.append([0, 20.0000, 30.0000])
    verts.append([1, 18.9394, 25.9806])
    verts.append([2, 21.9192, 25.3013])
    verts.append([3, 22.2834, 27.5068])

    # List the points above in counter-clockwise order
    iv = [0, 0, 1, 2, 3]

    # Organize the previous info into lists of x an y data
    xv, yv = [], []
    xyverts = []
    for v in iv[1:]:
        tiv, txv, tyv = verts[v]
        xv.append(txv)
        yv.append(tyv)

    # is_clockwise() should fail and return false
    rslt = is_clockwise(xv, yv)

    assert bool(rslt) is False, "is_clockwise() failed"


def debug_plot(grid, cell, xpts, ypts, mask):
    import matplotlib.pyplot as plt

    grid.plot()
    plt.plot(*list(zip(*cell)))
    plt.scatter(xpts[mask], ypts[mask], c="red")
    # plt.show()


def test_point_in_polygon_interior():
    grid = GridCases().structured_small()
    cell = grid.verts[grid.iverts[0]].tolist()
    xpts = grid.xcellcenters
    ypts = grid.ycellcenters
    mask = point_in_polygon(xpts, ypts, cell)
    assert mask.sum() == 1
    assert mask[0, 0]
    debug_plot(grid, cell, xpts, ypts, mask)


def test_point_in_polygon_vertices():
    grid = GridCases().structured_small()
    cell = grid.verts[grid.iverts[0]].tolist()
    xpts, ypts = list(zip(*cell))
    xpts = np.array([xpts])
    ypts = np.array([ypts])
    mask = point_in_polygon(xpts, ypts, cell)
    assert mask.sum() == 1  # only bottom left (due to axis-alignment)
    debug_plot(grid, cell, xpts, ypts, mask)

    # move points inside boundary
    xpts[0, 2] = xpts[0, 2] - 0.001
    ypts[0, 2] = ypts[0, 2] + 0.001
    xpts[0, 1] = xpts[0, 1] - 0.001
    ypts[0, 1] = ypts[0, 1] - 0.001
    xpts[0, 0] = xpts[0, 0] + 0.001
    ypts[0, 0] = ypts[0, 0] - 0.001
    mask = point_in_polygon(xpts, ypts, cell)
    assert mask.sum() == 4  # expect all points
    debug_plot(grid, cell, xpts, ypts, mask)


def test_point_in_polygon_faces():
    grid = GridCases().structured_small()
    cell = grid.verts[grid.iverts[0]].tolist()
    xpts_v, ypts_v = list(zip(*cell))
    xpts_v = np.array([xpts_v])
    ypts_v = np.array([ypts_v])
    xpts = np.array(
        [[xpts_v[0, 0], xpts_v[0, 2], np.mean(xpts_v), np.mean(xpts_v)]]
    )
    ypts = np.array(
        [[np.mean(ypts_v), np.mean(ypts_v), ypts_v[0, 0], ypts_v[0, 2]]]
    )
    mask = point_in_polygon(xpts, ypts, cell)
    assert mask.sum() == 2  # only inner faces
    debug_plot(grid, cell, xpts, ypts, mask)
