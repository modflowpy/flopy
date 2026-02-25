import matplotlib
import numpy as np
import pytest
from flaky import flaky
from matplotlib import pyplot as plt

from flopy.discretization import StructuredGrid, UnstructuredGrid, VertexGrid
from flopy.mf6.utils import MfGrdFile

pytestmark = pytest.mark.mf6


@pytest.fixture(scope="module")
def mfgrd_test_path(example_data_path):
    return example_data_path / "mfgrd_test"


def test_mfgrddis_MfGrdFile(mfgrd_test_path):
    grb = MfGrdFile(mfgrd_test_path / "nwtp3.dis.grb", verbose=True)
    nodes = grb.nodes
    ia = grb.ia
    shape = ia.shape[0]
    assert shape == nodes + 1, f"ia size ({shape}) not equal to {nodes + 1}"

    nnz = ia[-1]
    ja = grb.ja
    shape = ja.shape[0]
    assert shape == nnz, f"ja size ({shape}) not equal to {nnz}"

    modelgrid = grb.modelgrid
    assert isinstance(modelgrid, StructuredGrid)


def test_mfgrddis_modelgrid(mfgrd_test_path):
    fn = mfgrd_test_path / "nwtp3.dis.grb"
    modelgrid = StructuredGrid.from_binary_grid_file(fn, verbose=True)
    assert isinstance(modelgrid, StructuredGrid), "invalid grid type"

    lc = modelgrid.plot()
    assert isinstance(lc, matplotlib.collections.LineCollection), (
        f"could not plot grid object created from {fn}"
    )
    plt.close()

    extents = modelgrid.extent
    errmsg = f"extents {extents} of {fn} does not equal (0.0, 8000.0, 0.0, 8000.0)"
    assert extents == (0.0, 8000.0, 0.0, 8000.0), errmsg

    ncpl = modelgrid.ncol * modelgrid.nrow
    assert modelgrid.ncpl == ncpl, f"ncpl ({modelgrid.ncpl}) does not equal {ncpl}"

    nvert = modelgrid.nvert
    iverts = modelgrid.iverts
    maxvertex = max(max(sublist[1:]) for sublist in iverts)
    assert maxvertex + 1 == nvert, f"nvert ({maxvertex + 1}) does not equal {nvert}"
    verts = modelgrid.verts
    assert nvert == verts.shape[0], (
        f"number of vertex (x, y) pairs ({verts.shape[0]}) does not equal {nvert}"
    )


def test_mfgrddisv_MfGrdFile(mfgrd_test_path):
    fn = mfgrd_test_path / "flow.disv.grb"
    grb = MfGrdFile(fn, verbose=True)

    nodes = grb.nodes
    ia = grb.ia
    shape = ia.shape[0]
    assert shape == nodes + 1, f"ia size ({shape}) not equal to {nodes + 1}"

    nnz = ia[-1]
    ja = grb.ja
    shape = ja.shape[0]
    assert shape == nnz, f"ja size ({shape}) not equal to {nnz}"

    mg = grb.modelgrid
    assert isinstance(mg, VertexGrid), f"invalid grid type ({type(mg)})"


@flaky
def test_mfgrddisv_modelgrid(mfgrd_test_path):
    fn = mfgrd_test_path / "flow.disv.grb"
    mg = VertexGrid.from_binary_grid_file(fn, verbose=True)
    assert isinstance(mg, VertexGrid), f"invalid grid type ({type(mg)})"

    ncpl = 218
    assert mg.ncpl == ncpl, f"ncpl ({mg.ncpl}) does not equal {ncpl}"

    lc = mg.plot()
    assert isinstance(lc, matplotlib.collections.LineCollection), (
        f"could not plot grid object created from {fn}"
    )
    plt.close("all")

    extents = mg.extent
    extents0 = (0.0, 700.0, 0.0, 700.0)
    errmsg = f"extents {extents} of {fn} does not equal {extents0}"
    assert extents == extents0, errmsg

    nvert = mg.nvert
    iverts = mg.iverts
    maxvertex = max(max(sublist[1:]) for sublist in iverts)
    assert maxvertex + 1 == nvert, f"nvert ({maxvertex + 1}) does not equal {nvert}"
    verts = mg.verts
    assert nvert == verts.shape[0], (
        f"number of vertex (x, y) pairs ({verts.shape[0]}) does not equal {nvert}"
    )

    cellxy = np.column_stack(mg.xyzcellcenters[:2])
    errmsg = f"shape of flow.disv centroids {cellxy.shape} not equal to (218, 2)."
    assert cellxy.shape == (218, 2), errmsg


def test_mfgrddisu_MfGrdFile(mfgrd_test_path):
    fn = mfgrd_test_path / "keating.disu.grb"
    grb = MfGrdFile(fn, verbose=True)

    nodes = grb.nodes
    ia = grb.ia
    shape = ia.shape[0]
    assert shape == nodes + 1, f"ia size ({shape}) not equal to {nodes + 1}"

    nnz = ia[-1]
    ja = grb.ja
    shape = ja.shape[0]
    assert shape == nnz, f"ja size ({shape}) not equal to {nnz}"

    mg = grb.modelgrid
    assert isinstance(mg, UnstructuredGrid), f"invalid grid type ({type(mg)})"


def test_mfgrddisu_modelgrid_fail(mfgrd_test_path):
    fn = mfgrd_test_path / "flow.disu.grb"
    with pytest.raises(TypeError):
        mg = UnstructuredGrid.from_binary_grid_file(fn, verbose=True)


def test_mfgrddisu_modelgrid(mfgrd_test_path):
    fn = mfgrd_test_path / "keating.disu.grb"
    mg = UnstructuredGrid.from_binary_grid_file(fn, verbose=True)
    assert isinstance(mg, UnstructuredGrid), f"invalid grid type ({type(mg)})"

    lc = mg.plot()
    assert isinstance(lc, matplotlib.collections.LineCollection), (
        f"could not plot grid object created from {fn}"
    )
    plt.close("all")

    extents = mg.extent
    extents0 = (0.0, 10000.0, 0.0, 1.0)
    errmsg = f"extents {extents} of {fn} does not equal {extents0}"
    assert extents == extents0, errmsg

    nvert = mg.nvert
    iverts = mg.iverts
    maxvertex = max(max(sublist[1:]) for sublist in iverts)
    assert maxvertex + 1 == nvert, f"nvert ({maxvertex + 1}) does not equal {nvert}"
    verts = mg.verts
    assert nvert == verts.shape[0], (
        f"number of vertex (x, y) pairs ({verts.shape[0]}) does not equal {nvert}"
    )


def test_build_structured_connectivity_simple():
    """Test build_structured_connectivity with simple grids."""
    from flopy.mf6.utils.binarygrid_util import build_structured_connectivity

    # Test 1x1x1 grid (single cell, only diagonal connection)
    ia, ja, nja = build_structured_connectivity(1, 1, 1)
    assert ia.shape == (2,), f"ia shape {ia.shape} != (2,)"
    assert ja.shape == (1,), f"ja shape {ja.shape} != (1,)"
    assert nja == 1, f"nja {nja} != 1"
    assert ia[0] == 0 and ia[1] == 1, f"ia {ia} incorrect"
    assert ja[0] == 0, f"ja {ja} != [0]"

    # Test 1x1x2 grid (2 cells in x, 1 connection between them + 2 diagonals = 3)
    ia, ja, nja = build_structured_connectivity(1, 1, 2)
    assert ia.shape == (3,), f"ia shape {ia.shape} != (3,)"
    assert nja == 3, f"nja {nja} != 3"
    # Cell 0: diagonal (0) + right neighbor (1) = 2 connections
    # Cell 1: diagonal (1) = 1 connection
    np.testing.assert_array_equal(ia, [0, 2, 3])
    np.testing.assert_array_equal(ja, [0, 1, 1])

    # Test 1x2x1 grid (2 cells in y, 1 connection between them + 2 diagonals = 3)
    ia, ja, nja = build_structured_connectivity(1, 2, 1)
    assert ia.shape == (3,), f"ia shape {ia.shape} != (3,)"
    assert nja == 3, f"nja {nja} != 3"
    # Cell 0: diagonal (0) + front neighbor (1) = 2 connections
    # Cell 1: diagonal (1) = 1 connection
    np.testing.assert_array_equal(ia, [0, 2, 3])
    np.testing.assert_array_equal(ja, [0, 1, 1])

    # Test 2x1x1 grid (2 layers, 1 connection between them + 2 diagonals = 3)
    ia, ja, nja = build_structured_connectivity(2, 1, 1)
    assert ia.shape == (3,), f"ia shape {ia.shape} != (3,)"
    assert nja == 3, f"nja {nja} != 3"
    # Cell 0 (layer 0): diagonal (0) + lower neighbor (1) = 2 connections
    # Cell 1 (layer 1): diagonal (1) = 1 connection
    np.testing.assert_array_equal(ia, [0, 2, 3])
    np.testing.assert_array_equal(ja, [0, 1, 1])


def test_build_structured_connectivity_2x2x2():
    """Test build_structured_connectivity with a 2x2x2 grid."""
    from flopy.mf6.utils.binarygrid_util import build_structured_connectivity

    nlay, nrow, ncol = 2, 2, 2
    ncells = nlay * nrow * ncol  # 8 cells
    ia, ja, nja = build_structured_connectivity(nlay, nrow, ncol)

    # Verify dimensions
    assert ia.shape == (ncells + 1,), f"ia shape {ia.shape} != {(ncells + 1,)}"
    assert ja.shape == (nja,), f"ja shape {ja.shape} != {(nja,)}"
    assert ia[-1] == nja, f"ia[-1] {ia[-1]} != nja {nja}"

    # Verify IA is monotonically increasing
    assert np.all(np.diff(ia) >= 0), "IA array not monotonically increasing"

    # Verify all JA entries are valid cell indices
    assert np.all(ja >= 0), "JA contains negative indices"
    assert np.all(ja < ncells), f"JA contains indices >= {ncells}"

    # Verify each cell has at least a diagonal connection
    for i in range(ncells):
        nconn = ia[i + 1] - ia[i]
        assert nconn >= 1, f"Cell {i} has {nconn} connections (should be >= 1)"

    # Verify diagonal entries
    for node in range(ncells):
        conn_start = ia[node]
        # First connection should be diagonal (self)
        assert ja[conn_start] == node, (
            f"Cell {node} first connection {ja[conn_start]} != {node}"
        )


def test_build_structured_connectivity_with_idomain():
    """Test build_structured_connectivity with inactive cells."""
    from flopy.mf6.utils.binarygrid_util import build_structured_connectivity

    nlay, nrow, ncol = 1, 3, 3  # 9 cells
    ncells = nlay * nrow * ncol

    # Make center cell inactive
    idomain = np.ones((nlay, nrow, ncol), dtype=np.int32)
    idomain[0, 1, 1] = 0  # Cell 4 (center) is inactive

    ia, ja, nja = build_structured_connectivity(nlay, nrow, ncol, idomain=idomain)

    assert ia.shape == (ncells + 1,)
    assert ja.shape == (nja,)

    # Inactive cell (node 4) should have no connections
    node_4_start = ia[4]
    node_4_end = ia[5]
    assert node_4_end == node_4_start, (
        f"Inactive cell 4 has {node_4_end - node_4_start} connections (should be 0)"
    )

    # Cells around the inactive cell should not connect to it
    for node in range(ncells):
        if node == 4:
            continue  # Skip inactive cell
        conn_start = ia[node]
        conn_end = ia[node + 1]
        connections = ja[conn_start:conn_end]
        assert 4 not in connections, (
            f"Cell {node} incorrectly connects to inactive cell 4"
        )


def test_build_structured_connectivity_known_values():
    """Test build_structured_connectivity against known values."""
    from flopy.mf6.utils.binarygrid_util import build_structured_connectivity

    # Simple 1x2x2 grid
    # Cell layout:
    #   2 3
    #   0 1
    # Expected connections:
    # Cell 0: 0 (diag), 1 (right), 2 (front) -> ia[0]=0, ia[1]=3
    # Cell 1: 1 (diag), 3 (front) -> ia[1]=3, ia[2]=5
    # Cell 2: 2 (diag), 3 (right) -> ia[2]=5, ia[3]=7
    # Cell 3: 3 (diag) -> ia[3]=7, ia[4]=8

    ia, ja, nja = build_structured_connectivity(1, 2, 2)
    assert nja == 8, f"nja {nja} != 8"
    np.testing.assert_array_equal(ia, [0, 3, 5, 7, 8])
    np.testing.assert_array_equal(ja, [0, 1, 2, 1, 3, 2, 3, 3])


def test_write_grb_instance_method(tmp_path, mfgrd_test_path):
    """Test MfGrdFile.write() instance method."""
    from flopy.mf6.utils.binarygrid_util import MfGrdFile

    # Read an existing GRB file
    original_file = mfgrd_test_path / "nwtp3.dis.grb"
    grb_orig = MfGrdFile(original_file, verbose=False)

    # Write using instance method
    output_file = tmp_path / "test_instance.dis.grb"
    grb_orig.write(output_file, verbose=False)

    # Read it back
    grb_new = MfGrdFile(output_file, verbose=False)

    # Verify all properties match
    assert grb_new.grid_type == grb_orig.grid_type
    assert grb_new.nodes == grb_orig.nodes
    assert grb_new.nlay == grb_orig.nlay
    assert grb_new.nrow == grb_orig.nrow
    assert grb_new.ncol == grb_orig.ncol
    assert grb_new.nja == grb_orig.nja

    np.testing.assert_allclose(grb_new.xorigin, grb_orig.xorigin)
    np.testing.assert_allclose(grb_new.yorigin, grb_orig.yorigin)
    np.testing.assert_allclose(grb_new.angrot, grb_orig.angrot)

    np.testing.assert_allclose(grb_new.delr, grb_orig.delr)
    np.testing.assert_allclose(grb_new.delc, grb_orig.delc)
    np.testing.assert_allclose(grb_new.top, grb_orig.top)
    np.testing.assert_allclose(grb_new.bot, grb_orig.bot)

    np.testing.assert_array_equal(grb_new.ia, grb_orig.ia)
    np.testing.assert_array_equal(grb_new.ja, grb_orig.ja)
    np.testing.assert_array_equal(grb_new.idomain, grb_orig.idomain)


def test_write_grb_instance_method_precision_conversion(tmp_path, mfgrd_test_path):
    """Test MfGrdFile.write() with precision conversion."""
    from flopy.mf6.utils.binarygrid_util import MfGrdFile

    # Read an existing GRB file (presumably double precision)
    original_file = mfgrd_test_path / "nwtp3.dis.grb"
    grb = MfGrdFile(original_file, verbose=False)

    # Write as single precision
    single_file = tmp_path / "test_single.grb"
    grb.write(single_file, precision="single", verbose=False)

    # Write as double precision
    double_file = tmp_path / "test_double.grb"
    grb.write(double_file, precision="double", verbose=False)

    # Read them back
    grb_single = MfGrdFile(single_file, verbose=False)
    grb_double = MfGrdFile(double_file, verbose=False)

    # Verify both work and have same basic properties
    assert grb_single.nodes == grb.nodes
    assert grb_double.nodes == grb.nodes

    # Single precision file should be smaller
    assert single_file.stat().st_size < double_file.stat().st_size
