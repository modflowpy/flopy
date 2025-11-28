"""
Tests for GeospatialIndex class.

Tests the KD-tree based spatial indexing for efficient geometric queries
on FloPy grids (VertexGrid, UnstructuredGrid).

Note: StructuredGrid has its own optimized spatial methods and is not
supported by GeospatialIndex.
"""

import numpy as np
import pytest
from scipy.spatial import Delaunay

from flopy.discretization import StructuredGrid, UnstructuredGrid, VertexGrid
from flopy.utils.geospatial_index import GeospatialIndex

# ============================================================================
# Shared Geometry Helpers
# ============================================================================


def _create_minimal_geometry():
    """Create shared 2-cell geometry used by multiple fixtures."""
    vertices = [
        [0, 0.0, 1.0],
        [1, 1.0, 1.0],
        [2, 2.0, 1.0],
        [3, 0.0, 0.0],
        [4, 1.0, 0.0],
        [5, 2.0, 0.0],
    ]
    iverts = [[0, 1, 4, 3], [1, 2, 5, 4]]
    xcenters = [0.5, 1.5]
    ycenters = [0.5, 0.5]
    return vertices, iverts, xcenters, ycenters


def _create_triangular_geometry(seed=42, n_points=30):
    """Create shared Delaunay triangulation geometry."""
    np.random.seed(seed)
    x_verts = np.random.uniform(0, 100, n_points)
    y_verts = np.random.uniform(0, 100, n_points)
    points = np.column_stack([x_verts, y_verts])

    tri = Delaunay(points)
    vertices = [[i, x_verts[i], y_verts[i]] for i in range(len(x_verts))]
    iverts = tri.simplices.tolist()
    xcenters = np.mean(points[tri.simplices], axis=1)[:, 0]
    ycenters = np.mean(points[tri.simplices], axis=1)[:, 1]

    return vertices, iverts, xcenters, ycenters


def _create_rectangular_vertex_grid(nrow, ncol, cell_size, angrot=0.0):
    """Factory for creating rectangular vertex grids."""
    vertices = []
    vid = 0
    for i in range(nrow + 1):
        for j in range(ncol + 1):
            x = j * cell_size
            y = (nrow - i) * cell_size
            vertices.append([vid, x, y])
            vid += 1

    cell2d = []
    cellid = 0
    for i in range(nrow):
        for j in range(ncol):
            xc = (j + 0.5) * cell_size
            yc = (nrow - i - 0.5) * cell_size
            v0 = i * (ncol + 1) + j
            v1 = v0 + 1
            v2 = v1 + (ncol + 1)
            v3 = v0 + (ncol + 1)
            cell2d.append([cellid, xc, yc, 4, v0, v1, v2, v3])
            cellid += 1

    top = np.ones(nrow * ncol) * 10.0
    botm = np.zeros(nrow * ncol)

    return VertexGrid(
        vertices=vertices, cell2d=cell2d, top=top, botm=botm, nlay=1, angrot=angrot
    )


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def minimal_vertex_grid():
    """Create a minimal 2-cell vertex grid."""
    vertices, iverts, xcenters, ycenters = _create_minimal_geometry()
    cell2d = [
        [0, xcenters[0], ycenters[0], 4] + iverts[0],
        [1, xcenters[1], ycenters[1], 4] + iverts[1],
    ]
    return VertexGrid(vertices=vertices, cell2d=cell2d, nlay=1)


@pytest.fixture
def minimal_unstructured_grid():
    """Create a minimal 2-cell unstructured grid."""
    vertices, iverts, xcenters, ycenters = _create_minimal_geometry()
    return UnstructuredGrid(
        vertices=vertices, iverts=iverts, xcenters=xcenters, ycenters=ycenters
    )


@pytest.fixture
def triangular_vertex_grid():
    """Create a triangular vertex grid using Delaunay triangulation."""
    vertices, iverts, xcenters, ycenters = _create_triangular_geometry()
    cell2d = [[i, xcenters[i], ycenters[i], 3] + iverts[i] for i in range(len(iverts))]
    ncells = len(cell2d)
    return VertexGrid(
        vertices=vertices,
        cell2d=cell2d,
        top=np.ones(ncells) * 10.0,
        botm=np.zeros(ncells),
    )


@pytest.fixture
def triangular_unstructured_grid():
    """Create a triangular unstructured grid using Delaunay triangulation."""
    vertices, iverts, xcenters, ycenters = _create_triangular_geometry()
    ncells = len(iverts)
    return UnstructuredGrid(
        vertices=vertices,
        iverts=iverts,
        xcenters=xcenters,
        ycenters=ycenters,
        top=np.ones(ncells) * 10.0,
        botm=np.zeros(ncells),
    )


@pytest.fixture
def simple_vertex_grid():
    """Create a 10x10 rectangular vertex grid (100 cells, 10x10 each)."""
    return _create_rectangular_vertex_grid(nrow=10, ncol=10, cell_size=10.0)


@pytest.fixture
def rotated_vertex_grid():
    """Create a rotated 5x5 rectangular vertex grid."""
    return _create_rectangular_vertex_grid(nrow=5, ncol=5, cell_size=20.0, angrot=45.0)


@pytest.fixture
def simple_structured_grid():
    """Create a simple 10x10 structured grid for testing rejection."""
    nrow, ncol = 10, 10
    delr = np.ones(ncol) * 10.0
    delc = np.ones(nrow) * 10.0
    top = np.ones((nrow, ncol)) * 10.0
    botm = np.zeros((1, nrow, ncol))
    return StructuredGrid(delr=delr, delc=delc, top=top, botm=botm)


@pytest.fixture
def layered_unstructured_grid():
    """Create a 3-layer unstructured grid for 3D testing.

    Grid has 2 cells per layer, 3 layers total = 6 cells.
    Cell IDs: 0-1 (layer 0), 2-3 (layer 1), 4-5 (layer 2).
    Z elevations: layer 0 (10-7), layer 1 (7-4), layer 2 (4-1).
    """
    vertices, base_iverts, base_xc, base_yc = _create_minimal_geometry()

    nlay = 3
    ncpl = 2

    # Repeat geometry for each layer
    iverts = base_iverts * nlay
    xcenters = list(base_xc) * nlay
    ycenters = list(base_yc) * nlay

    top = np.array([10.0, 10.0, 7.0, 7.0, 4.0, 4.0])
    botm = np.array([7.0, 7.0, 4.0, 4.0, 1.0, 1.0])

    return UnstructuredGrid(
        vertices=vertices,
        iverts=iverts,
        xcenters=xcenters,
        ycenters=ycenters,
        top=top,
        botm=botm,
        ncpl=np.array([ncpl] * nlay),
    )


# ============================================================================
# Test Index Building
# ============================================================================


def test_rejects_structured_grid(simple_structured_grid):
    """Test that GeospatialIndex rejects StructuredGrid."""
    with pytest.raises(ValueError, match="only supports vertex and unstructured"):
        GeospatialIndex(simple_structured_grid)


@pytest.mark.parametrize(
    "grid_fixture,grid_type,expected_cells,expected_points_per_cell",
    [
        ("minimal_vertex_grid", "vertex", 2, 5),
        ("minimal_unstructured_grid", "unstructured", 2, 5),
        ("simple_vertex_grid", "vertex", 100, 5),
    ],
)
def test_index_build(
    grid_fixture, grid_type, expected_cells, expected_points_per_cell, request
):
    """Test building index for different grid types."""
    grid = request.getfixturevalue(grid_fixture)
    index = GeospatialIndex(grid)

    assert index.grid.grid_type == grid_type
    assert index.grid is grid
    assert index.tree is not None
    assert index.points.shape[0] == expected_cells * expected_points_per_cell
    assert len(np.unique(index.point_to_cell)) == expected_cells


def test_repr(simple_vertex_grid):
    """Test string representation of index."""
    index = GeospatialIndex(simple_vertex_grid)
    repr_str = repr(index)

    assert "GeospatialIndex" in repr_str
    assert "vertex grid" in repr_str
    assert "100 cells" in repr_str
    assert "500 indexed points" in repr_str


# ============================================================================
# Test Single Point Queries
# ============================================================================


def test_query_point_basic(simple_vertex_grid):
    """Test single point queries at cell centers."""
    index = GeospatialIndex(simple_vertex_grid)

    # Test corners and middle of 10x10 grid
    test_cases = [
        (5.0, 95.0, 0),  # top-left cell
        (55.0, 95.0, 5),  # top row, middle
        (5.0, 45.0, 50),  # middle row, left
        (95.0, 5.0, 99),  # bottom-right cell
    ]

    for x, y, expected in test_cases:
        assert index.query_point(x, y) == expected


def test_query_point_outside(simple_vertex_grid):
    """Test query for points outside grid returns nan."""
    index = GeospatialIndex(simple_vertex_grid)

    outside_points = [
        (200.0, 200.0),  # far outside
        (-0.1, 50.0),  # just left of grid
        (50.0, 100.1),  # just above grid
    ]

    for x, y in outside_points:
        assert np.isnan(index.query_point(x, y))


def test_query_point_on_boundary(simple_vertex_grid):
    """Test query for points on cell boundaries."""
    index = GeospatialIndex(simple_vertex_grid)

    # Point on vertical boundary between cells 0 and 1
    assert index.query_point(10.0, 95.0) in [0, 1]

    # Point on horizontal boundary between cells 0 and 10
    assert index.query_point(5.0, 90.0) in [0, 10]


def test_query_point_near_corner(simple_vertex_grid):
    """Test query for points near grid corners."""
    index = GeospatialIndex(simple_vertex_grid)

    assert index.query_point(0.5, 99.5) == 0  # top-left
    assert index.query_point(99.5, 0.5) == 99  # bottom-right


# ============================================================================
# Test Vectorized Queries
# ============================================================================


def test_query_points_basic(simple_vertex_grid):
    """Test vectorized point queries."""
    index = GeospatialIndex(simple_vertex_grid)

    x = np.array([5.0, 55.0, 95.0])
    y = np.array([95.0, 95.0, 5.0])
    cellids = index.query_points(x, y)

    assert isinstance(cellids, np.ndarray)
    assert len(cellids) == 3
    np.testing.assert_array_equal(cellids, [0, 5, 99])


def test_query_points_mixed_inside_outside(simple_vertex_grid):
    """Test vectorized query with mix of inside and outside points."""
    index = GeospatialIndex(simple_vertex_grid)

    x = np.array([5.0, 150.0, 55.0, -10.0])
    y = np.array([95.0, 50.0, 95.0, 50.0])
    cellids = index.query_points(x, y)

    assert len(cellids) == 4
    assert cellids[0] == 0
    assert np.isnan(cellids[1])
    assert cellids[2] == 5
    assert np.isnan(cellids[3])


def test_query_points_single(simple_vertex_grid):
    """Test vectorized query with single point."""
    index = GeospatialIndex(simple_vertex_grid)

    cellids = index.query_points(np.array([5.0]), np.array([95.0]))
    assert len(cellids) == 1
    assert cellids[0] == 0


def test_query_points_mismatched_lengths(simple_vertex_grid):
    """Test that mismatched x/y arrays raise error."""
    index = GeospatialIndex(simple_vertex_grid)

    with pytest.raises(ValueError, match="x and y must have the same length"):
        index.query_points(np.array([5.0, 55.0]), np.array([95.0]))


# ============================================================================
# Test Rotated Grid and Different k Values
# ============================================================================


def test_query_rotated_grid(rotated_vertex_grid):
    """Test point query on rotated vertex grid."""
    index = GeospatialIndex(rotated_vertex_grid)

    # Query at cell centroids
    for cellid in [0, 24]:  # first and last cell
        xc = rotated_vertex_grid.xcellcenters[cellid]
        yc = rotated_vertex_grid.ycellcenters[cellid]
        assert index.query_point(xc, yc) == cellid


@pytest.mark.parametrize("k", [1, 5, 10, 20])
def test_different_k_values(simple_vertex_grid, k):
    """Test that different k values still find correct cell."""
    index = GeospatialIndex(simple_vertex_grid)

    # Single point
    assert index.query_point(5.0, 95.0, k=k) == 0

    # Vectorized
    cellids = index.query_points(
        np.array([5.0, 55.0, 95.0]),
        np.array([95.0, 95.0, 5.0]),
        k=k,
    )
    np.testing.assert_array_equal(cellids, [0, 5, 99])


# ============================================================================
# Test Complete Grid Coverage
# ============================================================================


@pytest.mark.parametrize(
    "grid_fixture",
    [
        "simple_vertex_grid",
        "triangular_unstructured_grid",
    ],
)
def test_complete_coverage(grid_fixture, request):
    """Test that index can find all cells when querying at centroids."""
    grid = request.getfixturevalue(grid_fixture)
    index = GeospatialIndex(grid)

    ncells = len(grid.xcellcenters)
    found_cells = set()

    for cellid in range(ncells):
        xc = grid.xcellcenters[cellid]
        yc = grid.ycellcenters[cellid]
        found_cellid = index.query_point(xc, yc)
        assert not np.isnan(found_cellid)
        found_cells.add(found_cellid)

    assert len(found_cells) == ncells


# ============================================================================
# Test Tie-Breaking (Lowest Cell ID)
# ============================================================================


@pytest.mark.parametrize(
    "grid_fixture",
    [
        "minimal_vertex_grid",
        "minimal_unstructured_grid",
    ],
)
def test_tiebreaker_lowest_id(grid_fixture, request):
    """Test that boundary points return lowest cell ID."""
    grid = request.getfixturevalue(grid_fixture)
    index = GeospatialIndex(grid)

    # Point on shared boundary between cells 0 and 1 (x=1.0)
    assert index.query_point(1.0, 0.5) == 0


# ============================================================================
# Test 3D Layered UnstructuredGrid
# ============================================================================


def test_3d_query(layered_unstructured_grid):
    """Test 3D point queries on layered unstructured grid."""
    index = GeospatialIndex(layered_unstructured_grid)

    # Test all 6 cells across 3 layers
    test_cases = [
        (0.5, 0.5, 8.5, 0),  # layer 0, left
        (1.5, 0.5, 8.5, 1),  # layer 0, right
        (0.5, 0.5, 5.5, 2),  # layer 1, left
        (1.5, 0.5, 5.5, 3),  # layer 1, right
        (0.5, 0.5, 2.5, 4),  # layer 2, left
        (1.5, 0.5, 2.5, 5),  # layer 2, right
    ]

    for x, y, z, expected in test_cases:
        assert index.query_point(x, y, z=z) == expected


def test_3d_layer_boundary_tiebreaker(layered_unstructured_grid):
    """Test tie-breaking at layer boundaries returns lowest cell ID."""
    index = GeospatialIndex(layered_unstructured_grid)

    # z=7.0: boundary between layer 0 (cell 0) and layer 1 (cell 2)
    assert index.query_point(0.5, 0.5, z=7.0) == 0

    # z=4.0: boundary between layer 1 (cell 2) and layer 2 (cell 4)
    assert index.query_point(0.5, 0.5, z=4.0) == 2


def test_3d_xy_boundary_tiebreaker(layered_unstructured_grid):
    """Test tie-breaking at x/y boundaries in 3D grid."""
    index = GeospatialIndex(layered_unstructured_grid)

    # x=1.0 boundary in each layer
    assert index.query_point(1.0, 0.5, z=8.5) == 0  # layer 0
    assert index.query_point(1.0, 0.5, z=5.5) == 2  # layer 1
    assert index.query_point(1.0, 0.5, z=2.5) == 4  # layer 2


def test_3d_1to1_mapping(layered_unstructured_grid):
    """Test 1:1 mapping for 3D vectorized queries."""
    index = GeospatialIndex(layered_unstructured_grid)

    x = np.array([0.5, 1.5, 0.5, 5.0])
    y = np.array([0.5, 0.5, 0.5, 5.0])
    z = np.array([8.5, 5.5, 2.5, 8.5])

    cellids = index.query_points(x, y, z=z)

    assert len(cellids) == 4
    assert cellids[0] == 0
    assert cellids[1] == 3
    assert cellids[2] == 4
    assert np.isnan(cellids[3])


# ============================================================================
# Test StructuredGrid Native Methods
# ============================================================================


def test_structured_boundary_tiebreaker(simple_structured_grid):
    """Test StructuredGrid uses lowest row/col for boundary points."""
    grid = simple_structured_grid

    # Boundary at x=10 -> col 0
    _, col = grid.intersect(10.0, 95.0)
    assert col == 0

    # Boundary at y=90 -> row 0
    row, _ = grid.intersect(5.0, 90.0)
    assert row == 0


# ============================================================================
# Test Return Type Consistency
# ============================================================================


@pytest.mark.parametrize(
    "grid_fixture",
    [
        "minimal_vertex_grid",
        "minimal_unstructured_grid",
    ],
)
def test_return_type_scalar(grid_fixture, request):
    """Test scalar return types: int for inside, nan for outside."""
    grid = request.getfixturevalue(grid_fixture)
    index = GeospatialIndex(grid)

    # Inside -> int
    result = index.query_point(0.5, 0.5)
    assert isinstance(result, (int, np.integer))

    result = grid.intersect(0.5, 0.5)
    assert isinstance(result, (int, np.integer))

    # Outside -> nan
    result = index.query_point(10.0, 10.0)
    assert np.isnan(result)

    result = grid.intersect(10.0, 10.0, forgive=True)
    assert np.isnan(result)


@pytest.mark.parametrize(
    "grid_fixture",
    [
        "minimal_vertex_grid",
        "minimal_unstructured_grid",
    ],
)
def test_return_type_array(grid_fixture, request):
    """Test array return types: int64 for all inside, float64 when nan present."""
    grid = request.getfixturevalue(grid_fixture)
    index = GeospatialIndex(grid)

    x_inside = np.array([0.5, 1.5])
    y_inside = np.array([0.5, 0.5])
    x_mixed = np.array([0.5, 10.0])
    y_mixed = np.array([0.5, 10.0])

    # All inside -> integer type
    result = index.query_points(x_inside, y_inside)
    assert np.issubdtype(result.dtype, np.integer)

    result = grid.intersect(x_inside, y_inside)
    assert np.issubdtype(result.dtype, np.integer)

    # Mixed -> float64
    result = index.query_points(x_mixed, y_mixed)
    assert result.dtype == np.float64
    assert not np.isnan(result[0])
    assert np.isnan(result[1])

    result = grid.intersect(x_mixed, y_mixed, forgive=True)
    assert result.dtype == np.float64


def test_return_type_structured(simple_structured_grid):
    """Test StructuredGrid return types are consistent."""
    grid = simple_structured_grid

    # Scalar inside -> int
    row, col = grid.intersect(5.0, 95.0)
    assert isinstance(row, (int, np.integer))
    assert isinstance(col, (int, np.integer))

    # Scalar outside -> nan
    row, col = grid.intersect(150.0, 50.0, forgive=True)
    assert np.isnan(row) and np.isnan(col)

    # Scalar with z inside -> int
    lay, row, col = grid.intersect(5.0, 95.0, z=5.0)
    assert all(isinstance(v, (int, np.integer)) for v in [lay, row, col])

    # Scalar with z outside -> nan
    lay, row, col = grid.intersect(150.0, 50.0, z=5.0, forgive=True)
    assert all(np.isnan(v) for v in [lay, row, col])

    # Array all inside -> any integer dtype
    rows, cols = grid.intersect(
        np.array([5.0, 55.0]),
        np.array([95.0, 95.0])
    )
    assert np.issubdtype(rows.dtype, np.integer)
    assert np.issubdtype(cols.dtype, np.integer)

    # Array with outside -> must be float64 (due to NaNs)
    rows, cols = grid.intersect(
        np.array([5.0, 150.0]),
        np.array([95.0, 50.0]),
        forgive=True
    )
    assert rows.dtype == np.float64
    assert cols.dtype == np.float64


# ============================================================================
# Test Edge Cases
# ============================================================================


def test_thin_sliver_cell():
    """Test GeospatialIndex finds points in very thin "sliver" cells.

    Tests the centroid+vertices KD-tree approach for cells where the
    centroid might be far from the actual cell location.
    """
    np.random.seed(42)

    # Create base random points with thin sliver vertices
    n_points = 15
    x_verts = np.random.uniform(0, 100, n_points).tolist()
    y_verts = np.random.uniform(0, 100, n_points).tolist()

    for i in range(4):
        x_verts.append(50.0 + i * 0.05)  # Very thin: 0.15 units wide
        y_verts.append(i * 33.33)  # Tall: 100 units high

    points = np.column_stack([x_verts, y_verts])
    tri = Delaunay(points)

    vertices = [[i, x_verts[i], y_verts[i]] for i in range(len(x_verts))]
    cell2d = []
    for i, simplex in enumerate(tri.simplices):
        cell_x = np.mean([x_verts[j] for j in simplex])
        cell_y = np.mean([y_verts[j] for j in simplex])
        cell2d.append([i, cell_x, cell_y, len(simplex)] + list(simplex))

    ncells = len(cell2d)
    grid = VertexGrid(
        vertices=vertices,
        cell2d=cell2d,
        top=np.ones(ncells) * 10.0,
        botm=np.zeros(ncells),
    )

    index = GeospatialIndex(grid)

    # Test points in sliver region
    test_points = [(50.025, 50.0), (50.075, 25.0), (50.025, 75.0)]
    found_count = 0

    for x, y in test_points:
        result = index.query_point(x, y, k=20)
        if not np.isnan(result):
            xv, yv, _ = grid.xyzvertices
            verts = np.column_stack([xv[result], yv[result]])
            from matplotlib.path import Path

            if Path(verts).contains_point((x, y), radius=1e-9):
                found_count += 1

    assert found_count > 0, f"Should find points in sliver cells, found {found_count}/3"
