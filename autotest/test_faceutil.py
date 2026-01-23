import numpy as np
import pytest

from flopy.discretization import StructuredGrid, UnstructuredGrid, VertexGrid
from flopy.utils.faceutil import get_shared_face_3d


def test_get_shared_face_3d_structured_horizontal():
    nlay, nrow, ncol = 1, 2, 2
    delr = np.array([1.0, 1.0])
    delc = np.array([1.0, 1.0])
    top = np.array([[10.0, 10.0], [10.0, 10.0]])
    botm = np.array([[[0.0, 0.0], [0.0, 0.0]]])

    grid = StructuredGrid(delc=delc, delr=delr, top=top, botm=botm)

    # Test horizontally adjacent cells (0, 0, 0) and (0, 0, 1)
    cellid1 = (0, 0, 0)
    cellid2 = (0, 0, 1)

    face = get_shared_face_3d(grid, cellid1, cellid2)

    assert face is not None
    assert len(face) == 4  # Vertical face should have 4 vertices

    # Check that face is vertical (two z-coordinates: top and bottom)
    z_coords = sorted({v[2] for v in face})
    assert len(z_coords) == 2
    assert z_coords[0] == 0.0  # bottom
    assert z_coords[1] == 10.0  # top

    # Check x-coordinate is at the shared boundary (x=1.0)
    x_coords = {v[0] for v in face}
    assert 1.0 in x_coords


def test_get_shared_face_3d_structured_vertical():
    nlay, nrow, ncol = 2, 2, 2
    delr = np.array([1.0, 1.0])
    delc = np.array([1.0, 1.0])
    top = np.array([[10.0, 10.0], [10.0, 10.0]])
    botm = np.array([[[5.0, 5.0], [5.0, 5.0]], [[0.0, 0.0], [0.0, 0.0]]])

    grid = StructuredGrid(delc=delc, delr=delr, top=top, botm=botm)

    # Test vertically adjacent cells (0, 0, 0) and (1, 0, 0)
    cellid1 = (0, 0, 0)
    cellid2 = (1, 0, 0)

    face = get_shared_face_3d(grid, cellid1, cellid2)

    assert face is not None
    assert len(face) == 2  # Horizontal face should have 2 vertices (the edge)

    # Check that face is horizontal (all z-coordinates the same at interface)
    z_coords = {v[2] for v in face}
    assert len(z_coords) == 1
    assert next(iter(z_coords)) == 5.0  # Interface between layers


def test_get_shared_face_3d_vertex():
    nlay = 2
    ncpl = 2

    # Two square cells side by side
    vertices = [
        [0, 0.0, 0.0],
        [1, 1.0, 0.0],
        [2, 2.0, 0.0],
        [3, 0.0, 1.0],
        [4, 1.0, 1.0],
        [5, 2.0, 1.0],
    ]

    cell2d = [
        [0, 0.5, 0.5, 4, 0, 1, 4, 3],
        [1, 1.5, 0.5, 4, 1, 2, 5, 4],
    ]

    top = np.array([10.0, 10.0])
    botm = np.array([[5.0, 5.0], [0.0, 0.0]])

    grid = VertexGrid(vertices=vertices, cell2d=cell2d, top=top, botm=botm, nlay=nlay)

    # Test horizontally adjacent cells (0, 0) and (0, 1)
    cellid1 = (0, 0)
    cellid2 = (0, 1)

    face = get_shared_face_3d(grid, cellid1, cellid2)

    assert face is not None
    assert len(face) == 4  # Vertical face

    # Test vertically adjacent cells (0, 0) and (1, 0)
    cellid1 = (0, 0)
    cellid2 = (1, 0)

    face = get_shared_face_3d(grid, cellid1, cellid2)

    assert face is not None
    assert len(face) == 2  # Horizontal face

    # Check all z-coordinates are the same (interface elevation)
    z_coords = {v[2] for v in face}
    assert len(z_coords) == 1
    assert next(iter(z_coords)) == 5.0


def test_get_shared_face_3d_unstructured():
    # Two cells stacked vertically
    vertices = [
        # Bottom layer vertices (z=0)
        [0, 0.0, 0.0, 0.0],
        [1, 1.0, 0.0, 0.0],
        [2, 1.0, 1.0, 0.0],
        [3, 0.0, 1.0, 0.0],
        # Middle layer vertices (z=5)
        [4, 0.0, 0.0, 5.0],
        [5, 1.0, 0.0, 5.0],
        [6, 1.0, 1.0, 5.0],
        [7, 0.0, 1.0, 5.0],
        # Top layer vertices (z=10)
        [8, 0.0, 0.0, 10.0],
        [9, 1.0, 0.0, 10.0],
        [10, 1.0, 1.0, 10.0],
        [11, 0.0, 1.0, 10.0],
    ]

    iverts = [
        [0, 1, 2, 3, 4, 5, 6, 7],  # Bottom cell
        [4, 5, 6, 7, 8, 9, 10, 11],  # Top cell
    ]

    xc = np.array([0.5, 0.5])
    yc = np.array([0.5, 0.5])
    top = np.array([10.0, 10.0])
    botm = np.array([0.0, 5.0])

    grid = UnstructuredGrid(
        vertices=vertices,
        iverts=iverts,
        xcenters=xc,
        ycenters=yc,
        top=top,
        botm=botm,
        ncpl=[2],
    )

    # Test shared face between two vertically stacked cells
    cellid1 = (0,)
    cellid2 = (1,)

    face = get_shared_face_3d(grid, cellid1, cellid2)

    assert face is not None
    assert len(face) == 4  # Should have 4 shared vertices at the interface

    # Check all shared vertices are at z=5.0 (the interface)
    z_coords = {v[2] for v in face}
    assert len(z_coords) == 1
    assert next(iter(z_coords)) == 5.0


def test_get_shared_face_3d_error_cases():
    nlay, nrow, ncol = 1, 2, 2
    delr = np.array([1.0, 1.0])
    delc = np.array([1.0, 1.0])
    top = np.array([[10.0, 10.0], [10.0, 10.0]])
    botm = np.array([[[0.0, 0.0], [0.0, 0.0]]])

    grid = StructuredGrid(delc=delc, delr=delr, top=top, botm=botm)

    # Test with same cellid
    with pytest.raises(ValueError, match="cellid1 and cellid2 must be different"):
        get_shared_face_3d(grid, (0, 0, 0), (0, 0, 0))

    # Test with non-adjacent cells (should return None)
    cellid1 = (0, 0, 0)
    cellid2 = (0, 1, 1)  # Diagonally opposite, not adjacent

    face = get_shared_face_3d(grid, cellid1, cellid2)
    assert face is None
