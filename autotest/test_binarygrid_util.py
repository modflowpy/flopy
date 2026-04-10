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


def test_write_grb_instance_method(tmp_path, mfgrd_test_path):
    original_file = mfgrd_test_path / "nwtp3.dis.grb"
    grb_orig = MfGrdFile(original_file, verbose=False)

    output_file = tmp_path / "test_instance.dis.grb"
    grb_orig.export(output_file, verbose=False)

    grb_new = MfGrdFile(output_file, verbose=False)

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
    original_file = mfgrd_test_path / "nwtp3.dis.grb"
    grb = MfGrdFile(original_file, verbose=False)

    single_file = tmp_path / "test_single.grb"
    grb.export(single_file, precision="single", verbose=False)

    double_file = tmp_path / "test_double.grb"
    grb.export(double_file, precision="double", verbose=False)

    grb_single = MfGrdFile(single_file, verbose=False)
    grb_double = MfGrdFile(double_file, verbose=False)

    assert grb_single.nodes == grb.nodes
    assert grb_double.nodes == grb.nodes
    assert single_file.stat().st_size < double_file.stat().st_size


def test_write_grb_disv_roundtrip(tmp_path, mfgrd_test_path):
    """Test MfGrdFile.export() for DISV grid with roundtrip validation."""
    from flopy.mf6.utils.binarygrid_util import MfGrdFile

    # Read original DISV grb file
    original_file = mfgrd_test_path / "flow.disv.grb"
    grb_orig = MfGrdFile(original_file, verbose=False)

    # Write using instance method
    output_file = tmp_path / "test_disv.grb"
    grb_orig.export(output_file, verbose=False)

    # Read it back
    grb_new = MfGrdFile(output_file, verbose=False)

    # Verify grid type and dimensions
    assert grb_new.grid_type == "DISV"
    assert grb_new.grid_type == grb_orig.grid_type
    assert grb_new.nodes == grb_orig.nodes
    assert grb_new.nlay == grb_orig.nlay
    assert grb_new.ncpl == grb_orig.ncpl
    assert grb_new.nja == grb_orig.nja

    # Verify coordinates
    np.testing.assert_allclose(grb_new.xorigin, grb_orig.xorigin)
    np.testing.assert_allclose(grb_new.yorigin, grb_orig.yorigin)
    np.testing.assert_allclose(grb_new.angrot, grb_orig.angrot)

    # Verify elevation arrays
    np.testing.assert_allclose(grb_new.top, grb_orig.top)
    np.testing.assert_allclose(grb_new.bot, grb_orig.bot)

    # Verify cell connectivity
    np.testing.assert_array_equal(grb_new.ia, grb_orig.ia)
    np.testing.assert_array_equal(grb_new.ja, grb_orig.ja)

    # Verify DISV-specific data
    assert grb_new._datadict["NVERT"] == grb_orig._datadict["NVERT"]
    assert grb_new._datadict["NJAVERT"] == grb_orig._datadict["NJAVERT"]
    np.testing.assert_allclose(
        grb_new._datadict["VERTICES"], grb_orig._datadict["VERTICES"]
    )
    np.testing.assert_allclose(grb_new._datadict["CELLX"], grb_orig._datadict["CELLX"])
    np.testing.assert_allclose(grb_new._datadict["CELLY"], grb_orig._datadict["CELLY"])
    np.testing.assert_array_equal(
        grb_new._datadict["IAVERT"], grb_orig._datadict["IAVERT"]
    )
    np.testing.assert_array_equal(
        grb_new._datadict["JAVERT"], grb_orig._datadict["JAVERT"]
    )
    np.testing.assert_array_equal(grb_new.idomain, grb_orig.idomain)
    np.testing.assert_array_equal(
        grb_new._datadict["ICELLTYPE"], grb_orig._datadict["ICELLTYPE"]
    )


def test_write_grb_disv_precision_conversion(tmp_path, mfgrd_test_path):
    """Test MfGrdFile.export() for DISV grid with precision conversion."""
    from flopy.mf6.utils.binarygrid_util import MfGrdFile

    # Read original DISV grb file
    original_file = mfgrd_test_path / "flow.disv.grb"
    grb = MfGrdFile(original_file, verbose=False)

    # Write in single and double precision
    single_file = tmp_path / "test_disv_single.grb"
    grb.export(single_file, precision="single", verbose=False)

    double_file = tmp_path / "test_disv_double.grb"
    grb.export(double_file, precision="double", verbose=False)

    # Read them back
    grb_single = MfGrdFile(single_file, verbose=False)
    grb_double = MfGrdFile(double_file, verbose=False)

    # Verify dimensions are preserved
    assert grb_single.nodes == grb.nodes
    assert grb_double.nodes == grb.nodes
    assert grb_single.grid_type == "DISV"
    assert grb_double.grid_type == "DISV"

    # Single precision file should be smaller
    assert single_file.stat().st_size < double_file.stat().st_size

    # Verify data values are preserved (with appropriate tolerances)
    # Single precision has ~7 decimal digits of precision
    np.testing.assert_allclose(grb_single.top, grb.top, rtol=1e-6)
    np.testing.assert_allclose(grb_single.bot, grb.bot, rtol=1e-6)
    np.testing.assert_allclose(
        grb_single._datadict["VERTICES"], grb._datadict["VERTICES"], rtol=1e-6
    )
    np.testing.assert_allclose(
        grb_single._datadict["CELLX"], grb._datadict["CELLX"], rtol=1e-6
    )
    np.testing.assert_allclose(
        grb_single._datadict["CELLY"], grb._datadict["CELLY"], rtol=1e-6
    )

    # Double precision should match exactly (same precision as original)
    np.testing.assert_allclose(grb_double.top, grb.top, rtol=1e-12)
    np.testing.assert_allclose(grb_double.bot, grb.bot, rtol=1e-12)
    np.testing.assert_allclose(
        grb_double._datadict["VERTICES"], grb._datadict["VERTICES"], rtol=1e-12
    )


def test_write_grb_disu_roundtrip(tmp_path, mfgrd_test_path):
    """Test MfGrdFile.export() for DISU grid with roundtrip validation."""
    from flopy.mf6.utils.binarygrid_util import MfGrdFile

    # Read original DISU grb file
    original_file = mfgrd_test_path / "flow.disu.grb"
    grb_orig = MfGrdFile(original_file, verbose=False)

    # Write using instance method
    output_file = tmp_path / "test_disu.grb"
    grb_orig.export(output_file, verbose=False)

    # Read it back
    grb_new = MfGrdFile(output_file, verbose=False)

    # Verify grid type and dimensions
    assert grb_new.grid_type == "DISU"
    assert grb_new.grid_type == grb_orig.grid_type
    assert grb_new.nodes == grb_orig.nodes
    assert grb_new.nja == grb_orig.nja

    # Verify coordinates
    np.testing.assert_allclose(grb_new.xorigin, grb_orig.xorigin)
    np.testing.assert_allclose(grb_new.yorigin, grb_orig.yorigin)
    np.testing.assert_allclose(grb_new.angrot, grb_orig.angrot)

    # Verify elevation arrays (note: DISU uses TOP/BOT not TOP/BOTM)
    np.testing.assert_allclose(grb_new.top, grb_orig.top)
    np.testing.assert_allclose(grb_new._datadict["BOT"], grb_orig._datadict["BOT"])

    # Verify cell connectivity
    np.testing.assert_array_equal(grb_new.ia, grb_orig.ia)
    np.testing.assert_array_equal(grb_new.ja, grb_orig.ja)

    # Verify DISU-specific data
    np.testing.assert_array_equal(
        grb_new._datadict["ICELLTYPE"], grb_orig._datadict["ICELLTYPE"]
    )

    # IDOMAIN is optional in DISU - check if present
    if "IDOMAIN" in grb_orig._datadict:
        assert "IDOMAIN" in grb_new._datadict
        np.testing.assert_array_equal(grb_new.idomain, grb_orig.idomain)


def test_write_grb_disu_precision_conversion(tmp_path, mfgrd_test_path):
    """Test MfGrdFile.export() for DISU grid with precision conversion."""
    from flopy.mf6.utils.binarygrid_util import MfGrdFile

    # Read original DISU grb file
    original_file = mfgrd_test_path / "flow.disu.grb"
    grb = MfGrdFile(original_file, verbose=False)

    # Write in single and double precision
    single_file = tmp_path / "test_disu_single.grb"
    grb.export(single_file, precision="single", verbose=False)

    double_file = tmp_path / "test_disu_double.grb"
    grb.export(double_file, precision="double", verbose=False)

    # Read them back
    grb_single = MfGrdFile(single_file, verbose=False)
    grb_double = MfGrdFile(double_file, verbose=False)

    # Verify dimensions are preserved
    assert grb_single.nodes == grb.nodes
    assert grb_double.nodes == grb.nodes
    assert grb_single.grid_type == "DISU"
    assert grb_double.grid_type == "DISU"

    # Single precision file should be smaller
    assert single_file.stat().st_size < double_file.stat().st_size

    # Verify data values are preserved (with appropriate tolerances)
    # Single precision has ~7 decimal digits of precision
    np.testing.assert_allclose(grb_single.top, grb.top, rtol=1e-6)
    np.testing.assert_allclose(grb_single.bot, grb.bot, rtol=1e-6)

    # Double precision should match exactly (same precision as original)
    np.testing.assert_allclose(grb_double.top, grb.top, rtol=1e-12)
    np.testing.assert_allclose(grb_double.bot, grb.bot, rtol=1e-12)
