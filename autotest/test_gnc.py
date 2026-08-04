"""
Tests for ghost node correction (GNC) data computed from a grid.

Cases:
  - synthetic : a hand built grid with known contributing cells and factors.
  - gridgen   : the computed data must reproduce gridgen's qtg.gnc.dat.
"""

import numpy as np
import pytest
from modflow_devtools.markers import requires_exe, requires_pkg

import flopy
from flopy.discretization import UnstructuredGrid, VertexGrid
from flopy.utils.gnc import (
    _check_gnc,
    get_gnc,
    get_gnc_dtype,
    get_gridprops_gnc6,
    get_numalphaj,
)
from flopy.utils.gridgen import Gridgen


def synthetic_grid(connectivity=True):
    """A coarse cell with four fine cells north and two fine cells east

    Cell 0 is a size 4 cell centered on the origin.  Cells 1 and 2 are size 2
    cells on its east face and cells 3 to 6 are size 1 cells on its north
    face, so the north face has a two level refinement jump.
    """
    centers = [
        (0.0, 0.0, 4.0),
        (3.0, 1.0, 2.0),
        (3.0, -1.0, 2.0),
        (-1.5, 3.0, 1.0),
        (-0.5, 3.0, 1.0),
        (0.5, 3.0, 1.0),
        (1.5, 3.0, 1.0),
    ]
    vertices, iverts = [], []
    for x, y, size in centers:
        half = size / 2.0
        iv = []
        for vx, vy in [
            (x - half, y - half),
            (x + half, y - half),
            (x + half, y + half),
            (x - half, y + half),
        ]:
            iv.append(len(vertices))
            vertices.append([len(vertices), vx, vy])
        iverts.append(iv)

    # every cell is connected to cell 0 only
    conn = {}
    if connectivity:
        conn["iac"] = np.array([7, 2, 2, 2, 2, 2, 2])
        conn["ja"] = np.array([0, 1, 2, 3, 4, 5, 6, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0])
    grid = UnstructuredGrid(
        vertices=vertices,
        iverts=iverts,
        xcenters=np.array([c[0] for c in centers]),
        ycenters=np.array([c[1] for c in centers]),
        ncpl=np.array([len(centers)]),
        **conn,
    )
    level = np.array([0, 1, 1, 2, 2, 2, 2])
    return grid, level


def gnc_key(gnc):
    """Sorted view of ghost node records for order independent comparison"""
    numalphaj = get_numalphaj(gnc)
    nodes = np.sort(np.column_stack([gnc[f"j{i}"] for i in range(numalphaj)]), axis=1)
    alpha = np.sort(
        np.column_stack([gnc[f"alpha{i}"] for i in range(numalphaj)]), axis=1
    )
    key = np.column_stack([gnc["n"], gnc["m"], nodes, alpha])
    return key[np.lexsort(key.T[::-1])]


def build_gridgen(ws, nlay=3, level=3, layers=None, **kwargs):
    """Build a gridgen grid with a refined block in the middle"""
    from shapely.geometry import Polygon

    botm = [1.0 - k * (1.0 / nlay) for k in range(1, nlay + 1)]
    sim = flopy.mf6.MFSimulation(sim_name="base", sim_ws=ws)
    gwf = flopy.mf6.ModflowGwf(sim, modelname="base")
    flopy.mf6.ModflowGwfdis(
        gwf, nlay=nlay, nrow=10, ncol=10, delr=1.0, delc=1.0, top=1.0, botm=botm
    )
    g = Gridgen(gwf.modelgrid, model_ws=ws, **kwargs)
    g.add_refinement_features(
        [Polygon([(4, 4), (6, 4), (6, 6), (4, 6)])],
        "polygon",
        level,
        range(nlay) if layers is None else layers,
    )
    g.build()
    return g


def test_get_gnc_dtype():
    dtype = get_gnc_dtype(3)
    assert dtype.names == (
        "n",
        "m",
        "j0",
        "j1",
        "j2",
        "alpha0",
        "alpha1",
        "alpha2",
    )
    assert get_numalphaj(np.recarray((0,), dtype=dtype)) == 3


@pytest.mark.parametrize("use_level", [True, False])
def test_get_gnc_synthetic(use_level):
    grid, level = synthetic_grid()
    gnc = get_gnc(grid, level=level if use_level else None)

    # cell 0 has four contributing cells on its north face
    assert get_numalphaj(gnc) == 4
    assert len(gnc) == 3

    # connections without a contributing cell on the offset side are dropped
    assert sorted(zip(gnc["n"], gnc["m"])) == [(0, 1), (0, 5), (0, 6)]

    records = {(rec["n"], rec["m"]): rec for rec in gnc}

    # cell 1 is offset 1.0 from cell 0 and the north cells are 3.0 away, so
    # the factors total 1/3 and are shared by four contributing cells
    rec = records[(0, 1)]
    assert [rec[f"j{i}"] for i in range(4)] == [3, 4, 5, 6]
    assert np.allclose([rec[f"alpha{i}"] for i in range(4)], 1.0 / 12.0)

    # cell 6 is offset 1.5 and the east cells are 3.0 away, so the factors
    # total 0.5 shared by two contributing cells, the first of which is
    # repeated three times to fill numalphaj
    rec = records[(0, 6)]
    assert [rec[f"j{i}"] for i in range(4)] == [1, 1, 1, 2]
    assert np.allclose(
        [rec[f"alpha{i}"] for i in range(4)], [1 / 12, 1 / 12, 1 / 12, 0.25]
    )


def factor_totals(gnc):
    """Total contributing factor of each cell, keyed by ghost node"""
    numalphaj = get_numalphaj(gnc)
    totals = {}
    for rec in gnc:
        entry = totals.setdefault((rec["n"], rec["m"]), {})
        for i in range(numalphaj):
            node = rec[f"j{i}"]
            entry[node] = entry.get(node, 0.0) + rec[f"alpha{i}"]
    return totals


def test_get_gnc_padding_preserves_factors():
    grid, level = synthetic_grid()
    unpadded = get_gnc(grid, level=level)
    padded = get_gnc(grid, level=level, numalphaj=6)
    assert get_numalphaj(unpadded) == 4
    assert get_numalphaj(padded) == 6

    # padding repeats a contributing cell and splits its factor, so the total
    # factor of every cell must be unchanged
    expected, actual = factor_totals(unpadded), factor_totals(padded)
    assert expected.keys() == actual.keys()
    for ghost, cells in expected.items():
        assert cells.keys() == actual[ghost].keys()
        for node, alpha in cells.items():
            assert np.allclose(alpha, actual[ghost][node])
        assert sum(cells.values()) < 1.0


def test_get_gnc_numalphaj_too_small():
    grid, level = synthetic_grid()
    with pytest.raises(ValueError, match="more than numalphaj"):
        get_gnc(grid, level=level, numalphaj=2)


def test_get_gnc_supplied_connectivity():
    grid, level = synthetic_grid()
    bare, _ = synthetic_grid(connectivity=False)
    assert bare.iac is None

    # the synthetic cells do not share vertices, so connectivity cannot be
    # built from shared edges and has to be supplied
    assert len(get_gnc(bare, level=level)) == 0

    gnc = get_gnc(bare, level=level, iac=grid.iac, ja=grid.ja)
    assert np.array_equal(gnc_key(gnc), gnc_key(get_gnc(grid, level=level)))


def test_check_gnc():
    dtype = get_gnc_dtype(2)
    gnc = np.recarray((1,), dtype=dtype)
    gnc[0] = (0, 1, 2, 2, 0.6, 0.6)
    with pytest.raises(ValueError, match="must be less than one"):
        _check_gnc(gnc)

    gnc[0] = (0, 1, 2, 2, 0.1, 0.1)
    _check_gnc(gnc)

    # cell 0 is connected to cell 2 but not to cell 1
    ia = np.array([0, 2, 3, 5])
    ja = np.array([0, 2, 1, 2, 0])
    with pytest.raises(ValueError, match="is not connected to cell"):
        _check_gnc(gnc, ia=ia, ja=ja)


@pytest.mark.parametrize("n,m", [(5, 1), (1, 5), (-3, 1), (1, -3)])
def test_check_gnc_node_out_of_range(n, m):
    """A node outside the grid is reported rather than indexed"""
    gnc = np.recarray((1,), dtype=get_gnc_dtype(1))
    gnc[0] = (n, m, 0, 0.1)
    # a negative node would otherwise wrap and check the wrong cell
    with pytest.raises(ValueError, match="which is not a cell of a grid"):
        _check_gnc(gnc, ia=np.array([0, 2, 3]), ja=np.array([0, 1, 1, 0]))


def test_get_gridprops_gnc6_requires_ncpl():
    grid, level = synthetic_grid()
    gnc = get_gnc(grid, level=level)
    with pytest.raises(ValueError, match="ncpl is required"):
        get_gridprops_gnc6(gnc, dis_type="disv")
    with pytest.raises(ValueError, match="Unknown dis_type"):
        get_gridprops_gnc6(gnc, dis_type="dis")

    gridprops = get_gridprops_gnc6(gnc, dis_type="disu")
    assert gridprops["numalphaj"] == 4
    assert gridprops["numgnc"] == len(gnc)
    assert gridprops["gncdata"][0][0] == (gnc["n"][0],)


@requires_exe("gridgen")
@requires_pkg("shapely", "geopandas")
@pytest.mark.parametrize(
    "nlay,level,layers,smoothing",
    [
        (3, 3, None, 1),
        (1, 2, None, 1),
        (1, 4, None, 1),
        (3, 3, [0], 1),
        (1, 3, None, 2),
    ],
)
def test_get_gnc_matches_gridgen(function_tmpdir, nlay, level, layers, smoothing):
    g = build_gridgen(
        function_tmpdir,
        nlay=nlay,
        level=level,
        layers=layers,
        smoothing_level_horizontal=smoothing,
        smoothing_level_vertical=smoothing,
    )
    # gridgen writes the ghost node data whenever it exports a grid
    expected = np.atleast_1d(
        np.genfromtxt(function_tmpdir / "qtg.gnc.dat", dtype=get_gnc_dtype(2))
    )
    for name in ("n", "m", "j0", "j1"):
        expected[name] -= 1
    assert len(expected) > 0

    grid = UnstructuredGrid(**g.get_gridprops_unstructuredgrid())
    gnc = get_gnc(grid, numalphaj=2)

    assert len(gnc) == len(expected)
    # gridgen writes the factors with six significant digits
    assert np.allclose(gnc_key(gnc), gnc_key(expected), atol=2.0e-6)


def square_cells(centers):
    """Build vertices and iverts for a list of (x, y, size) squares"""
    vertices, iverts = [], []
    for x, y, size in centers:
        half = size / 2.0
        iv = []
        for vx, vy in [
            (x - half, y - half),
            (x + half, y - half),
            (x + half, y + half),
            (x - half, y + half),
        ]:
            iv.append(len(vertices))
            vertices.append([len(vertices), vx, vy])
        iverts.append(iv)
    return vertices, iverts


def test_get_gnc_ihc():
    """A connection marked vertical is not corrected and does not contribute"""
    grid, level = synthetic_grid()

    # cells 0 and 1 are connected at positions 1 and 8 of ja
    ihc = np.ones(len(grid.ja), dtype=int)
    ihc[[1, 8]] = 0
    gnc = get_gnc(grid, level=level, ihc=ihc)

    # the ghost node on the 0 to 1 connection is gone, and cell 1 is no longer
    # available as a contributing cell, which leaves cell 2 on its own
    assert sorted((rec["n"], rec["m"]) for rec in gnc) == [(0, 5), (0, 6)]
    assert get_numalphaj(gnc) == 1
    assert np.allclose(sorted(gnc["alpha0"]), [1.0 / 6.0, 0.5])


def test_get_gnc_aligned_finer_cell():
    """A finer cell centered on the face of a coarse cell needs no ghost node"""
    centers = [(0.0, 0.0, 4.0), (3.0, 0.0, 2.0)]
    vertices, iverts = square_cells(centers)
    grid = UnstructuredGrid(
        vertices=vertices,
        iverts=iverts,
        xcenters=np.array([c[0] for c in centers]),
        ycenters=np.array([c[1] for c in centers]),
        ncpl=np.array([len(centers)]),
        iac=np.array([2, 2]),
        ja=np.array([0, 1, 1, 0]),
    )
    assert len(get_gnc(grid, level=np.array([0, 1]))) == 0


def test_get_gnc_connectivity_needs_constant_ncpl():
    """Shared edges cannot give the layer layout when ncpl varies by layer"""
    centers = [(0.5, 0.5, 1.0), (1.5, 0.5, 1.0), (2.5, 0.5, 1.0), (0.5, 1.5, 1.0)]
    vertices, iverts = square_cells(centers)
    grid = UnstructuredGrid(
        vertices=vertices,
        iverts=iverts,
        xcenters=np.array([c[0] for c in centers]),
        ycenters=np.array([c[1] for c in centers]),
        ncpl=np.array([3, 1]),
    )
    assert grid.iac is None
    with pytest.raises(ValueError, match="different"):
        get_gnc(grid)
