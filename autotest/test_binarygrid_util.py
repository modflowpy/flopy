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
    grb = MfGrdFile(str(mfgrd_test_path / "nwtp3.dis.grb"), verbose=True)
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
    fn = str(mfgrd_test_path / "nwtp3.dis.grb")
    modelgrid = StructuredGrid.from_binary_grid_file(fn, verbose=True)
    assert isinstance(modelgrid, StructuredGrid), "invalid grid type"

    lc = modelgrid.plot()
    assert isinstance(
        lc, matplotlib.collections.LineCollection
    ), f"could not plot grid object created from {fn}"
    plt.close()

    extents = modelgrid.extent
    errmsg = (
        f"extents {extents} of {fn} does not equal (0.0, 8000.0, 0.0, 8000.0)"
    )
    assert extents == (0.0, 8000.0, 0.0, 8000.0), errmsg

    ncpl = modelgrid.ncol * modelgrid.nrow
    assert (
        modelgrid.ncpl == ncpl
    ), f"ncpl ({modelgrid.ncpl}) does not equal {ncpl}"

    nvert = modelgrid.nvert
    iverts = modelgrid.iverts
    maxvertex = max([max(sublist[1:]) for sublist in iverts])
    assert (
        maxvertex + 1 == nvert
    ), f"nvert ({maxvertex + 1}) does not equal {nvert}"
    verts = modelgrid.verts
    assert nvert == verts.shape[0], (
        f"number of vertex (x, y) pairs ({verts.shape[0]}) "
        f"does not equal {nvert}"
    )


def test_mfgrddisv_MfGrdFile(mfgrd_test_path):
    fn = str(mfgrd_test_path / "flow.disv.grb")
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
    assert isinstance(
        lc, matplotlib.collections.LineCollection
    ), f"could not plot grid object created from {fn}"
    plt.close("all")

    extents = mg.extent
    extents0 = (0.0, 700.0, 0.0, 700.0)
    errmsg = f"extents {extents} of {fn} does not equal {extents0}"
    assert extents == extents0, errmsg

    nvert = mg.nvert
    iverts = mg.iverts
    maxvertex = max([max(sublist[1:]) for sublist in iverts])
    assert (
        maxvertex + 1 == nvert
    ), f"nvert ({maxvertex + 1}) does not equal {nvert}"
    verts = mg.verts
    assert nvert == verts.shape[0], (
        f"number of vertex (x, y) pairs ({verts.shape[0]}) "
        f"does not equal {nvert}"
    )

    cellxy = np.column_stack((mg.xyzcellcenters[:2]))
    errmsg = (
        f"shape of flow.disv centroids {cellxy.shape} not equal to (218, 2)."
    )
    assert cellxy.shape == (218, 2), errmsg


def test_mfgrddisu_MfGrdFile(mfgrd_test_path):
    fn = str(mfgrd_test_path / "keating.disu.grb")
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
    fn = str(mfgrd_test_path / "flow.disu.grb")
    with pytest.raises(TypeError):
        mg = UnstructuredGrid.from_binary_grid_file(fn, verbose=True)


def test_mfgrddisu_modelgrid(mfgrd_test_path):
    fn = str(mfgrd_test_path / "keating.disu.grb")
    mg = UnstructuredGrid.from_binary_grid_file(fn, verbose=True)
    assert isinstance(mg, UnstructuredGrid), f"invalid grid type ({type(mg)})"

    lc = mg.plot()
    assert isinstance(
        lc, matplotlib.collections.LineCollection
    ), f"could not plot grid object created from {fn}"
    plt.close("all")

    extents = mg.extent
    extents0 = (0.0, 10000.0, 0.0, 1.0)
    errmsg = f"extents {extents} of {fn} does not equal {extents0}"
    assert extents == extents0, errmsg

    nvert = mg.nvert
    iverts = mg.iverts
    maxvertex = max([max(sublist[1:]) for sublist in iverts])
    assert (
        maxvertex + 1 == nvert
    ), f"nvert ({maxvertex + 1}) does not equal {nvert}"
    verts = mg.verts
    assert nvert == verts.shape[0], (
        f"number of vertex (x, y) pairs ({verts.shape[0]}) "
        f"does not equal {nvert}"
    )
