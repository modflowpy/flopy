"""
Tests for ghost node correction (GNC) data computed from a grid.

Cases:
  - synthetic : a hand built grid with known contributing cells and factors.
  - gridgen   : the computed data must reproduce gridgen's qtg.gnc.dat.
"""

import io

import numpy as np
import pytest
from modflow_devtools.markers import requires_exe, requires_pkg

import flopy
from flopy.discretization import UnstructuredGrid, VertexGrid
from flopy.utils.gnc import (
    _check_gnc,
    get_gnc,
    get_gnc_dtype,
    get_gridprops_gnc5,
    get_gridprops_gnc6,
    get_numalphaj,
)
from flopy.utils.gridgen import Gridgen, get_ia_from_iac


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


@requires_exe("gridgen")
@requires_pkg("shapely", "geopandas")
def test_get_gnc_inputs_agree(function_tmpdir):
    g = build_gridgen(function_tmpdir)
    expected = gnc_key(g.get_gnc())

    grid = UnstructuredGrid(**g.get_gridprops_unstructuredgrid())
    iac = g.get_iac()
    ia, ja = get_ia_from_iac(iac), g.get_ja(iac.sum())

    # refinement level of every cell, where level 0 is the base grid cell
    area = g.get_area()
    level = np.round(np.log2(np.sqrt(area.max() / area))).astype(int)
    assert level.max() > 0

    ncpl = g.get_gridprops_disv()["ncpl"]
    vertex_grid = VertexGrid(**g.get_gridprops_vertexgrid())

    for tag, gnc in [
        ("level", get_gnc(grid, level=level, numalphaj=2)),
        ("level per layer", get_gnc(grid, level=level[:ncpl], numalphaj=2)),
        ("vertex grid", get_gnc(vertex_grid, ia=ia, ja=ja, numalphaj=2)),
        ("iac", get_gnc(grid, iac=iac, ja=ja, numalphaj=2)),
        # connectivity built from the grid, either the iac and ja an
        # unstructured grid carries or the cells that share an edge
        ("unstructured grid only", get_gnc(grid, numalphaj=2)),
        ("vertex grid only", get_gnc(vertex_grid, numalphaj=2)),
    ]:
        assert np.allclose(gnc_key(gnc), expected, atol=2.0e-6), tag


@requires_exe("gridgen")
@requires_pkg("shapely", "geopandas")
@requires_exe("gridgen")
@requires_pkg("shapely", "geopandas")
def test_get_gridprops_gnc_matches_gridgen(function_tmpdir):
    g = build_gridgen(function_tmpdir, nlay=1)
    iac = g.get_iac()
    ia, ja = get_ia_from_iac(iac), g.get_ja(iac.sum())
    ncpl = g.get_gridprops_disv()["ncpl"]

    grid = UnstructuredGrid(**g.get_gridprops_unstructuredgrid())
    gnc = get_gnc(grid, numalphaj=2)

    gridprops = get_gridprops_gnc6(gnc, dis_type="disv", ncpl=ncpl, ia=ia, ja=ja)
    expected = g.get_gridprops_gnc6(dis_type="disv")
    assert gridprops["numgnc"] == expected["numgnc"]
    assert gridprops["numalphaj"] == expected["numalphaj"]

    gridprops = get_gridprops_gnc5(gnc, ia=ia, ja=ja)
    expected = g.get_gridprops_gnc5()
    assert gridprops["numgnc"] == expected["numgnc"]
    assert gridprops["iflalphan"] == 0
    assert gridprops["gncdata"].dtype == expected["gncdata"].dtype


@pytest.mark.slow
@requires_exe("mf6", "gridgen")
@requires_pkg("shapely", "geopandas")
def test_mf6disv_gnc_padding(function_tmpdir):
    """Repeating a contributing cell must not change the solution"""
    g = build_gridgen(function_tmpdir, nlay=1)
    disv_gridprops = g.get_gridprops_disv()
    iac = g.get_iac()
    ia, ja = get_ia_from_iac(iac), g.get_ja(iac.sum())
    grid = UnstructuredGrid(**g.get_gridprops_unstructuredgrid())

    chdspd = []
    for x, y, head in [(0, 10, 1.0), (10, 0, 0.0)]:
        ic = g.intersect([(x, y)], "point", 0)["nodenumber"][0]
        chdspd.append([(0, ic), head])

    def run(numalphaj):
        gnc = get_gnc(grid, numalphaj=numalphaj)
        assert get_numalphaj(gnc) == numalphaj
        gridprops = get_gridprops_gnc6(
            gnc, dis_type="disv", ncpl=disv_gridprops["ncpl"], ia=ia, ja=ja
        )
        ws = function_tmpdir / f"j{numalphaj}"
        sim = flopy.mf6.MFSimulation(sim_name="m", sim_ws=ws, exe_name="mf6")
        flopy.mf6.ModflowTdis(sim)
        flopy.mf6.ModflowIms(
            sim,
            linear_acceleration="bicgstab",
            inner_dvclose=1e-11,
            outer_dvclose=1e-11,
        )
        gwf = flopy.mf6.ModflowGwf(sim, modelname="m")
        flopy.mf6.ModflowGwfdisv(gwf, **disv_gridprops)
        flopy.mf6.ModflowGwfic(gwf)
        flopy.mf6.ModflowGwfnpf(gwf)
        flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chdspd)
        flopy.mf6.ModflowGwfoc(
            gwf, head_filerecord="m.hds", saverecord=[("HEAD", "ALL")]
        )
        flopy.mf6.ModflowGwfgnc(gwf, **gridprops)
        sim.write_simulation()
        success, buff = sim.run_simulation(silent=True)
        assert success, "\n".join(buff[-25:])
        return gwf.output.head().get_data().flatten()

    assert np.allclose(run(2), run(4), atol=1e-8)


@pytest.mark.slow
@requires_exe("mfusg", "gridgen")
@requires_pkg("shapely", "geopandas")
def test_mfusg_gnc_padding(function_tmpdir):
    """Repeating a contributing cell must not change the solution"""
    g = build_gridgen(function_tmpdir, nlay=1)
    disu_gridprops = g.get_gridprops_disu5()
    iac = g.get_iac()
    ia, ja = get_ia_from_iac(iac), g.get_ja(iac.sum())
    grid = UnstructuredGrid(**g.get_gridprops_unstructuredgrid())

    chdspd = []
    for x, y, head in [(0, 10, 1.0), (10, 0, 0.0)]:
        ic = g.intersect([(x, y)], "point", 0)["nodenumber"][0]
        chdspd.append([ic, head, head])

    def run(numalphaj):
        gridprops = get_gridprops_gnc5(get_gnc(grid, numalphaj=numalphaj), ia=ia, ja=ja)
        ws = function_tmpdir / f"j{numalphaj}"
        m = flopy.mfusg.MfUsg(
            modelname="m", model_ws=ws, exe_name="mfusg", structured=False
        )
        flopy.mfusg.MfUsgDisU(m, **disu_gridprops)
        flopy.mfusg.MfUsgBas(m)
        flopy.mfusg.MfUsgLpf(m)
        flopy.modflow.ModflowChd(m, stress_period_data=chdspd)
        flopy.mfusg.MfUsgSms(m, options="COMPLEX")
        flopy.modflow.ModflowOc(m, stress_period_data={(0, 0): ["save head"]})
        flopy.mfusg.MfUsgGnc(m, **gridprops)
        m.write_input()
        success, buff = m.run_model(silent=True)
        assert success, "\n".join(buff[-25:])
        return np.concatenate(flopy.utils.HeadUFile(ws / "m.hds").get_data())

    assert np.allclose(run(2), run(4), atol=1e-8)

    assert np.allclose(run(2), run(4), atol=1e-8)


def test_mfusg_gnc_file_fields_stay_separated(function_tmpdir):
    """A value that fills its format width must not run into the next field"""
    model = flopy.mfusg.MfUsg(modelname="m", model_ws=function_tmpdir, structured=False)
    dtype = flopy.mfusg.MfUsgGnc.get_default_dtype(2, 0)
    gncdata = np.zeros(2, dtype=dtype)
    gncdata[0] = (23, 33, 22, 22, 0.125, 0.166667)
    # ten digit node numbers fill the %10d field width
    gncdata[1] = (1234567889, 1234567889, 1234567889, 1234567889, 0.125, 0.125)
    flopy.mfusg.MfUsgGnc(model, numgnc=2, numalphaj=2, gncdata=gncdata)
    model.write_input()

    # the list is read with URWORD, so every record must have one token per field
    records = (function_tmpdir / "m.gnc").open().readlines()[2:]
    for line in records:
        if line.strip():
            assert len(line.split()) == 6

    # the contributing factors must not be truncated
    assert np.allclose(float(records[0].split()[5]), 0.166667, atol=1e-6)


def test_fmt_string_separates_free_format_fields():
    """A free format list is read with URWORD, so its fields must be separated

    A value that fills its format width runs into the next value when the
    field formats are concatenated, which made a record unreadable.  Ten digit
    node numbers fill the %10d field width.
    """
    from flopy.mfusg.cln_dtypes import MfUsgClnDtypes
    from flopy.mfusg.mfusg import fmt_string

    dtypes = {
        "gnc": flopy.mfusg.MfUsgGnc.get_default_dtype(2, 0),
        "cln node": MfUsgClnDtypes.get_clnnode_dtype(),
    }
    for name, dtype in dtypes.items():
        record = np.zeros(1, dtype=dtype)
        for field in dtype.names:
            if np.issubdtype(dtype[field], np.integer):
                record[0][field] = 1234567889

        buff = io.StringIO()
        np.savetxt(buff, record, fmt=fmt_string(record, free=True), delimiter="")
        assert len(buff.getvalue().split()) == len(dtype.names), name

        # a fixed format list is read by position, so it stays unseparated
        buff = io.StringIO()
        np.savetxt(buff, record, fmt=fmt_string(record, free=False), delimiter="")
        assert len(buff.getvalue().split()) < len(dtype.names), name
