import numpy as np
import pytest

from flopy.discretization import StructuredGrid
from flopy.utils.cvfdutil import get_disv_gridprops, gridlist_to_verts
from flopy.utils.lgrutil import Lgr, LgrToDisv, SimpleRegularGrid


def test_lgr_connections():
    nlayp = 5
    nrowp = 5
    ncolp = 5
    delrp = 100.0
    delcp = 100.0
    topp = 100.0
    botmp = [-100, -200, -300, -400, -500]
    idomainp = np.ones((nlayp, nrowp, ncolp), dtype=int)
    idomainp[0:2, 1:4, 1:4] = 0
    ncpp = 3
    ncppl = [1, 1, 0, 0, 0]

    lgr = Lgr(
        nlayp,
        nrowp,
        ncolp,
        delrp,
        delcp,
        topp,
        botmp,
        idomainp,
        ncpp=ncpp,
        ncppl=ncppl,
        xllp=100.0,
        yllp=100.0,
    )

    # child shape
    assert lgr.get_shape() == (2, 9, 9), "child shape is not (2, 9, 9)"

    # child delr/delc
    delr, delc = lgr.get_delr_delc()
    assert np.allclose(delr, delrp / ncpp), "child delr not correct"
    assert np.allclose(delc, delcp / ncpp), "child delc not correct"

    # child idomain
    idomain = lgr.get_idomain()
    assert idomain.min() == idomain.max() == 1
    assert idomain.shape == (2, 9, 9)

    # replicated parent array
    ap = np.arange(nrowp * ncolp).reshape((nrowp, ncolp))
    ac = lgr.get_replicated_parent_array(ap)
    assert ac[0, 0] == 6
    assert ac[-1, -1] == 18

    # child top/bottom
    topc, botmc = lgr.get_top_botm()
    assert topc.shape == (9, 9)
    assert botmc.shape == (2, 9, 9)
    assert topc.min() == topc.max() == 100.0
    errmsg = f"{botmc[:, 0, 0]} /= {np.array(botmp[:2])}"
    assert np.allclose(botmc[:, 0, 0], np.array(botmp[:2])), errmsg

    # exchange data
    exchange_data = lgr.get_exchange_data(angldegx=True, cdist=True)

    ans1 = [
        (0, 1, 0),
        (0, 0, 0),
        1,
        50.0,
        16.666666666666668,
        33.333333333333336,
        0.0,
        354.33819375782156,
    ]
    errmsg = f"{ans1} /= {exchange_data[0]}"
    assert exchange_data[0] == ans1, errmsg

    ans2 = [(2, 3, 3), (1, 8, 8), 0, 50.0, 50, 1111.1111111111113, 180.0, 100.0]
    errmsg = f"{ans2} /= {exchange_data[-1]}"
    assert exchange_data[-1] == ans2, errmsg

    errmsg = "exchanges should be 71 horizontal plus 81 vertical"
    assert len(exchange_data) == 72 + 81, errmsg

    # list of parent cells connected to a child cell
    assert lgr.get_parent_connections(0, 0, 0) == [((0, 1, 0), -1), ((0, 0, 1), 2)]
    assert lgr.get_parent_connections(1, 8, 8) == [
        ((1, 3, 4), 1),
        ((1, 4, 3), -2),
        ((2, 3, 3), -3),
    ]


def test_lgr_variable_rc_spacing():
    # Define parent grid information
    xoffp = 0.0
    yoffp = 0.0
    nlayp = 1
    nrowp = 5
    ncolp = 5
    dx = 100.0
    dy = 100.0
    dz = 100.0
    delrp = dx * np.array([1.0, 0.75, 0.5, 0.75, 1.0], dtype=float)
    delcp = dy * np.array([1.0, 0.75, 0.5, 0.75, 1.0], dtype=float)
    topp = dz * np.ones((nrowp, ncolp), dtype=float)
    botmp = np.empty((nlayp, nrowp, ncolp), dtype=float)
    for k in range(nlayp):
        botmp[k] = -(k + 1) * dz

    # Define relation of child to parent
    idomainp = np.ones((nlayp, nrowp, ncolp), dtype=int)
    idomainp[:, 1:4, 1:4] = 0
    ncpp = 3
    ncppl = nlayp * [1]

    lgr = Lgr(
        nlayp,
        nrowp,
        ncolp,
        delrp,
        delcp,
        topp,
        botmp,
        idomainp,
        ncpp=ncpp,
        ncppl=ncppl,
        xllp=xoffp,
        yllp=yoffp,
    )

    # check to make sure child delr and delc are calculated correctly for
    # the case where the parent grid has variable row and column spacings
    answer = [
        25.0,
        25.0,
        25.0,
        50.0 / 3.0,
        50.0 / 3.0,
        50.0 / 3.0,
        25.0,
        25.0,
        25.0,
    ]
    assert np.allclose(lgr.delr, answer), f"{lgr.delr} /= {answer}"
    assert np.allclose(lgr.delc, answer), f"{lgr.delc} /= {answer}"


def test_lgr_hanging_vertices():
    # Define parent grid information
    xoffp = 0.0
    yoffp = 0.0
    nlayp = 3
    nrowp = 3
    ncolp = 3

    dx = 100.0
    dy = 100.0
    dz = 10.0
    delrp = dx * np.ones(ncolp)
    delcp = dy * np.ones(nrowp)
    topp = dz * np.ones((nrowp, ncolp), dtype=float)
    botmp = np.empty((nlayp, nrowp, ncolp), dtype=float)
    for k in range(nlayp):
        botmp[k] = -(k + 1) * dz
    idomainp = np.ones((nlayp, nrowp, ncolp), dtype=int)
    idomainp[:, nrowp // 2, ncolp // 2] = 0
    ncpp = 3
    ncppl = nlayp * [1]
    lgr = Lgr(
        nlayp,
        nrowp,
        ncolp,
        delrp,
        delcp,
        topp,
        botmp,
        idomainp,
        ncpp=ncpp,
        ncppl=ncppl,
        xllp=xoffp,
        yllp=yoffp,
    )

    # check to make sure gridprops is accessible from lgr
    gridprops = lgr.to_disv_gridprops()
    assert "ncpl" in gridprops
    assert "nvert" in gridprops
    assert "vertices" in gridprops
    assert "nlay" in gridprops
    assert "top" in gridprops
    assert "botm" in gridprops
    assert gridprops["ncpl"] == 17
    assert gridprops["nvert"] == 28
    assert gridprops["nlay"] == 3

    # test the lgr to disv class
    lgrtodisv = LgrToDisv(lgr)

    # test guts of LgrToDisv to make sure hanging vertices added correctly
    assert lgrtodisv.right_face_hanging[1, 0] == [0, 4, 8, 12]
    assert lgrtodisv.left_face_hanging[1, 2] == [3, 7, 11, 15]
    assert lgrtodisv.back_face_hanging[2, 1] == [12, 13, 14, 15]
    assert lgrtodisv.front_face_hanging[0, 1] == [0, 1, 2, 3]

    # These cells have hanging vertices at parent-child boundaries
    assert lgrtodisv.iverts[1] == [1, 2, 6, 17, 16, 5]  # top-center parent cell
    assert lgrtodisv.iverts[3] == [4, 5, 18, 22, 9, 8]  # middle-left parent cell
    assert lgrtodisv.iverts[4] == [6, 7, 11, 10, 25, 21]  # middle-right parent cell
    assert lgrtodisv.iverts[6] == [9, 26, 27, 10, 14, 13]  # bottom-left parent cell

    assert np.allclose(gridprops["top"], dz * np.ones((17,)))

    assert gridprops["botm"].shape == (3, 17)
    b = np.empty((3, 17))
    b[0] = -dz
    b[1] = -2 * dz
    b[2] = -3 * dz
    assert np.allclose(gridprops["botm"], b)


def test_lgr_from_parent_grid():
    # Create a parent grid with center cells marked for refinement
    nlay, nrow, ncol = 1, 7, 7
    delr = delc = 100.0 * np.ones(7)
    top = np.zeros((nrow, ncol))
    botm = -100.0 * np.ones((nlay, nrow, ncol))
    idomain = np.ones((nlay, nrow, ncol), dtype=int)
    idomain[:, 2:5, 2:5] = 0  # Mark center 3x3 cells for refinement

    parent_grid = StructuredGrid(
        delr=delr, delc=delc, top=top, botm=botm, idomain=idomain
    )

    # Create Lgr using the classmethod (no warning - uses new API)
    lgr_from_classmethod = Lgr.from_parent_grid(parent_grid, idomain, ncpp=3, ncppl=1)

    # Create Lgr using the traditional constructor with deprecated parameter
    with pytest.warns(DeprecationWarning, match="idomainp.*deprecated"):
        lgr_traditional = Lgr(
            nlayp=nlay,
            nrowp=nrow,
            ncolp=ncol,
            delrp=delr,
            delcp=delc,
            topp=top,
            botmp=botm,
            idomainp=idomain,
            ncpp=3,
            ncppl=1,
            xllp=0.0,
            yllp=0.0,
        )

    # Verify both methods produce the same results
    assert lgr_from_classmethod.get_shape() == lgr_traditional.get_shape()
    assert np.allclose(lgr_from_classmethod.delr, lgr_traditional.delr)
    assert np.allclose(lgr_from_classmethod.delc, lgr_traditional.delc)
    assert np.allclose(lgr_from_classmethod.top, lgr_traditional.top)
    assert np.allclose(lgr_from_classmethod.botm, lgr_traditional.botm)

    # Verify child grid has expected dimensions (3x3 parent cells refined to 9x9)
    assert lgr_from_classmethod.get_shape() == (1, 9, 9)

    # Verify gridprops can be generated
    gridprops = lgr_from_classmethod.to_disv_gridprops()
    assert "ncpl" in gridprops
    assert "nvert" in gridprops
    assert "vertices" in gridprops
    assert "cell2d" in gridprops
    assert "nlay" in gridprops
    assert "top" in gridprops
    assert "botm" in gridprops

    # Expected: 40 parent cells + 81 child cells = 121 total cells
    assert gridprops["ncpl"] == 121


def test_lgr_deprecation_warnings():
    nlay, nrow, ncol = 1, 7, 7
    delr = delc = 100.0 * np.ones(7)
    top = np.zeros((nrow, ncol))
    botm = -100.0 * np.ones((nlay, nrow, ncol))
    refine_mask = np.ones((nlay, nrow, ncol))
    refine_mask[:, 2:5, 2:5] = 0

    # Test deprecated idomainp parameter
    with pytest.warns(DeprecationWarning, match="idomainp.*deprecated.*refine_mask"):
        lgr = Lgr(
            nlayp=nlay,
            nrowp=nrow,
            ncolp=ncol,
            delrp=delr,
            delcp=delc,
            topp=top,
            botmp=botm,
            idomainp=refine_mask,
            ncpp=3,
        )

    # Test deprecated idomain attribute access
    with pytest.warns(
        DeprecationWarning, match="idomain.*attribute.*deprecated.*refine_mask"
    ):
        _ = lgr.idomain

    # Verify new API works without warnings
    import warnings

    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        lgr_new = Lgr(
            nlayp=nlay,
            nrowp=nrow,
            ncolp=ncol,
            delrp=delr,
            delcp=delc,
            topp=top,
            botmp=botm,
            refine_mask=refine_mask,
            ncpp=3,
        )
        # Access via new attribute should not warn
        _ = lgr_new.refine_mask

    # Check no deprecation warnings were raised
    deprecation_warnings = [
        w for w in warning_list if issubclass(w.category, DeprecationWarning)
    ]
    assert len(deprecation_warnings) == 0, (
        "New API should not raise deprecation warnings"
    )


def test_lgr_nested_refinement_grandchild():
    # Create parent grid (9x9)
    nlay, nrow, ncol = 1, 9, 9
    delr = delc = 100.0 * np.ones(9)
    top = np.zeros((nrow, ncol))
    botm = -100.0 * np.ones((nlay, nrow, ncol))

    parent_grid = StructuredGrid(
        delr=delr, delc=delc, top=top, botm=botm, idomain=np.ones((nlay, nrow, ncol))
    )

    # First refinement: refine center 3x3 cells of parent
    parent_refine_mask = np.ones((nlay, nrow, ncol))
    parent_refine_mask[:, 3:6, 3:6] = 0

    lgr_child = Lgr.from_parent_grid(parent_grid, parent_refine_mask, ncpp=3)
    assert lgr_child.get_shape() == (1, 9, 9)

    # Get child grid (SimpleRegularGrid now inherits from StructuredGrid)
    child_grid = lgr_child.child
    assert isinstance(child_grid, StructuredGrid)
    assert (child_grid.nlay, child_grid.nrow, child_grid.ncol) == (1, 9, 9)

    # Second refinement: refine center 3x3 cells of child
    child_refine_mask = np.ones((child_grid.nlay, child_grid.nrow, child_grid.ncol))
    child_refine_mask[:, 3:6, 3:6] = 0

    lgr_grandchild = Lgr.from_parent_grid(child_grid, child_refine_mask, ncpp=3)
    assert lgr_grandchild.get_shape() == (1, 9, 9)

    # Verify grandchild has finer resolution than child
    assert np.allclose(lgr_grandchild.delr, lgr_child.delr / 3)
    assert np.allclose(lgr_grandchild.delc, lgr_child.delc / 3)

    # Verify we can generate gridprops for both
    child_gridprops = lgr_child.to_disv_gridprops()
    grandchild_gridprops = lgr_grandchild.to_disv_gridprops()
    assert "ncpl" in child_gridprops
    assert "ncpl" in grandchild_gridprops


def test_lgr_multiple_child_regions():
    # Create parent grid (15x15) with two separate areas to refine
    nlay, nrow, ncol = 1, 15, 15
    delr = delc = 100.0 * np.ones(15)
    top = np.zeros((nrow, ncol))
    botm = -100.0 * np.ones((nlay, nrow, ncol))

    parent_grid = StructuredGrid(
        delr=delr, delc=delc, top=top, botm=botm, idomain=np.ones((nlay, nrow, ncol))
    )

    # Region 1: top-left (2:5, 2:5)
    refine_mask_1 = np.ones((nlay, nrow, ncol))
    refine_mask_1[:, 2:5, 2:5] = 0
    lgr1 = Lgr.from_parent_grid(parent_grid, refine_mask_1, ncpp=3)

    # Region 2: bottom-right (10:13, 10:13)
    refine_mask_2 = np.ones((nlay, nrow, ncol))
    refine_mask_2[:, 10:13, 10:13] = 0
    lgr2 = Lgr.from_parent_grid(parent_grid, refine_mask_2, ncpp=3)

    # Verify both children have correct shape (3x3 parent cells -> 9x9 child)
    assert lgr1.get_shape() == (1, 9, 9)
    assert lgr2.get_shape() == (1, 9, 9)

    # Verify children are at different locations
    assert lgr1.xll == 200.0  # Starts at column 2
    assert lgr1.yll == 1000.0  # Starts at row 2 (from top)
    assert lgr2.xll == 1000.0  # Starts at column 10
    assert lgr2.yll == 200.0  # Starts at row 10

    # Verify both have same resolution (refinement of parent)
    assert np.allclose(lgr1.delr, lgr2.delr)
    assert np.allclose(lgr1.delc, lgr2.delc)

    # Verify we can generate gridprops for both
    gridprops1 = lgr1.to_disv_gridprops()
    gridprops2 = lgr2.to_disv_gridprops()
    assert gridprops1["ncpl"] == gridprops2["ncpl"]


def test_simple_regular_grid_deprecation():
    nlay, nrow, ncol = 1, 5, 5
    delr = delc = 100.0 * np.ones(5)
    top = np.zeros((nrow, ncol))
    botm = -100.0 * np.ones((nlay, nrow, ncol))
    idomain = np.ones((nlay, nrow, ncol), dtype=int)
    xorigin = 0.0
    yorigin = 0.0

    # Test that SimpleRegularGrid instantiation raises deprecation warning
    with pytest.warns(
        DeprecationWarning, match="SimpleRegularGrid is deprecated.*StructuredGrid"
    ):
        grid = SimpleRegularGrid(
            nlay, nrow, ncol, delr, delc, top, botm, idomain, xorigin, yorigin
        )

    # Verify it's an instance of StructuredGrid
    assert isinstance(grid, StructuredGrid)
    assert isinstance(grid, SimpleRegularGrid)

    # Test that modelgrid property raises deprecation warning
    with pytest.warns(
        DeprecationWarning, match="modelgrid.*deprecated.*use the instance directly"
    ):
        mg = grid.modelgrid

    # Verify modelgrid returns self
    assert mg is grid

    # Test that get_gridprops_dis6 raises deprecation warning
    with pytest.warns(DeprecationWarning, match="get_gridprops_dis6.*deprecated"):
        gridprops = grid.get_gridprops_dis6()

    # Verify gridprops contains expected keys
    assert "nlay" in gridprops
    assert "nrow" in gridprops
    assert "ncol" in gridprops
    assert gridprops["nlay"] == nlay
    assert gridprops["nrow"] == nrow
    assert gridprops["ncol"] == ncol

    # Verify backward compatibility attributes
    assert grid.xorigin == xorigin
    assert grid.yorigin == yorigin


def test_lgr_matches_legacy_gridlist_to_disv_gridprops():
    """
    Lgr.to_disv_gridprops() output should match
    legacy gridlist_to_disv_gridprops() output.
    """
    # TODO: this will need to be removed soon, and the underlying
    # implementation inlined, when we remove this function.
    from flopy.utils.cvfdutil import gridlist_to_disv_gridprops

    # 7x7 parent grid with 3x3 refinement in center
    nlay = 1
    nrow_p, ncol_p = 7, 7
    delr_p = delc_p = 100.0 * np.ones(7)
    top_p = np.zeros((nrow_p, ncol_p))
    botm_p = -100.0 * np.ones((nlay, nrow_p, ncol_p))

    # legacy approach
    idomain_p = np.ones((nlay, nrow_p, ncol_p), dtype=int)
    idomain_p[:, 2:5, 2:5] = 0  # Center 3x3 inactive
    sg_parent_legacy = StructuredGrid(
        delc=delc_p,
        delr=delr_p,
        top=top_p,
        botm=botm_p,
        idomain=idomain_p,
        xoff=0.0,
        yoff=0.0,
    )
    nrow_c, ncol_c = 9, 9
    delr_c = delc_c = 100.0 / 3.0 * np.ones(9)
    top_c = np.zeros((nrow_c, ncol_c))
    botm_c = -100.0 * np.ones((nlay, nrow_c, ncol_c))
    idomain_c = np.ones((nlay, nrow_c, ncol_c), dtype=int)
    sg_child = StructuredGrid(
        delc=delc_c,
        delr=delr_c,
        top=top_c,
        botm=botm_c,
        idomain=idomain_c,
        xoff=200.0,
        yoff=200.0,
    )
    gridprops_legacy = gridlist_to_disv_gridprops([sg_parent_legacy, sg_child])

    # lgr approach
    parent_grid = StructuredGrid(
        delc=delc_p,
        delr=delr_p,
        top=top_p,
        botm=botm_p,
        xoff=0.0,
        yoff=0.0,
    )
    refine_mask = np.ones((nlay, nrow_p, ncol_p))
    refine_mask[:, 2:5, 2:5] = 0
    lgr = Lgr.from_parent_grid(parent_grid, refine_mask, ncpp=3)
    gridprops_lgr = lgr.to_disv_gridprops()

    # Note: legacy gridlist_to_disv_gridprops doesn't return nlay, top, botm
    # Only check common keys
    assert gridprops_legacy["ncpl"] == gridprops_lgr["ncpl"]
    assert gridprops_legacy["nvert"] == gridprops_lgr["nvert"]

    # Verify vertex positions match (may be in different order)
    verts_legacy_set = {
        (round(v[1], 9), round(v[2], 9)) for v in gridprops_legacy["vertices"]
    }
    verts_lgr_set = {
        (round(v[1], 9), round(v[2], 9)) for v in gridprops_lgr["vertices"]
    }
    assert verts_legacy_set == verts_lgr_set

    for icpl in range(gridprops_legacy["ncpl"]):
        nverts_legacy = gridprops_legacy["cell2d"][icpl][3]
        nverts_lgr = gridprops_lgr["cell2d"][icpl][3]
        assert nverts_legacy >= 3
        assert nverts_lgr >= 3

    for icpl in range(gridprops_legacy["ncpl"]):
        xc_legacy, yc_legacy = gridprops_legacy["cell2d"][icpl][1:3]
        xc_lgr, yc_lgr = gridprops_lgr["cell2d"][icpl][1:3]
        assert np.allclose([xc_legacy, yc_legacy], [xc_lgr, yc_lgr])

    # Note: top and botm are only in Lgr gridprops,
    # not in legacy gridlist_to_disv_gridprops.
    # Just verify they exist in the Lgr version
    assert "nlay" in gridprops_lgr
    assert "top" in gridprops_lgr
    assert "botm" in gridprops_lgr
    assert gridprops_lgr["nlay"] == nlay
    assert gridprops_lgr["top"].shape == (gridprops_lgr["ncpl"],)
    assert gridprops_lgr["botm"].shape == (nlay, gridprops_lgr["ncpl"])


def test_hanging_vertex_algorithm_comparison():
    """
    Compare the two hanging vertex placement algorithms in detail:
    1. to_cvfd() - reactive geometric algorithm (legacy)
    2. Lgr.find_hanging_vertices() - proactive structured algorithm (new)

    This test creates a simple parent-child grid and analyzes the differences
    in vertex placement, connectivity, and grid structure. Intended mainly to
    be run manually to inspect the printed output.
    """
    # Create a simple 3x3 parent grid with 3x refinement in center cell
    nlay = 1
    nrow_p = 3
    ncol_p = 3
    delr_p = 100.0
    delc_p = 100.0
    top_p = np.zeros((nrow_p, ncol_p))
    botm_p = np.array([np.full((nrow_p, ncol_p), -10.0)])

    # Setup for both approaches
    # Legacy approach: use idomain to mark refinement region as inactive
    idomain_legacy = np.ones((nlay, nrow_p, ncol_p), dtype=int)
    idomain_legacy[:, 1, 1] = 0  # Center cell inactive

    sg_parent_legacy = StructuredGrid(
        delr=np.full(ncol_p, delr_p),
        delc=np.full(nrow_p, delc_p),
        top=top_p,
        botm=botm_p,
        idomain=idomain_legacy,
        xoff=0.0,
        yoff=0.0,
    )

    # Child grid at center position
    ncpp = 3
    top_c = np.zeros((ncpp, ncpp))
    botm_c = np.array([np.full((ncpp, ncpp), -10.0)])
    idomain_child = np.ones((nlay, ncpp, ncpp), dtype=int)
    sg_child = StructuredGrid(
        delr=np.full(ncpp, delr_p / ncpp),
        delc=np.full(ncpp, delc_p / ncpp),
        top=top_c,
        botm=botm_c,
        idomain=idomain_child,
        xoff=100.0,
        yoff=100.0,
    )

    # New approach: use refine_mask
    sg_parent_new = StructuredGrid(
        delr=np.full(ncol_p, delr_p),
        delc=np.full(nrow_p, delc_p),
        top=top_p.copy(),
        botm=botm_p.copy(),
        xoff=0.0,
        yoff=0.0,
    )
    refine_mask = np.ones((nlay, nrow_p, ncol_p))
    refine_mask[:, 1, 1] = 0

    # Generate grids with both algorithms
    # Legacy: to_cvfd() algorithm
    verts_legacy, iverts_legacy = gridlist_to_verts([sg_parent_legacy, sg_child])
    gridprops_legacy = get_disv_gridprops(verts_legacy, iverts_legacy)

    # New: Lgr.find_hanging_vertices() algorithm
    lgr = Lgr.from_parent_grid(sg_parent_new, refine_mask, ncpp=ncpp)
    gridprops_new = lgr.to_disv_gridprops()

    # Compare basic metrics
    print("\n" + "=" * 80)
    print("HANGING VERTEX ALGORITHM COMPARISON")
    print("=" * 80)
    print("\nGrid Configuration:")
    print(f"  Parent: {nrow_p}x{ncol_p} cells, {delr_p}x{delc_p} spacing")
    print(f"  Child: {ncpp}x{ncpp} refinement at center cell (1,1)")
    print(
        f"  Total cells: {nrow_p * ncol_p - 1} "
        f"parent + {ncpp * ncpp} "
        f"child = {nrow_p * ncol_p - 1 + ncpp * ncpp}"
    )

    print(
        f"\n{'Metric':<30} "
        f"{'Legacy (to_cvfd)':<20} "
        f"{'New (Lgr)':<20} "
        f"{'Difference':<20}"
    )
    print("-" * 90)
    print(
        f"{'Total vertices':<30} "
        f"{gridprops_legacy['nvert']:<20} "
        f"{gridprops_new['nvert']:<20} "
        f"{gridprops_new['nvert'] - gridprops_legacy['nvert']:<20}"
    )
    print(
        f"{'Total cells':<30} "
        f"{gridprops_legacy['ncpl']:<20} "
        f"{gridprops_new['ncpl']:<20} "
        f"{gridprops_new['ncpl'] - gridprops_legacy['ncpl']:<20}"
    )

    # Analyze vertex differences
    verts_legacy_dict = {
        i: (v[1], v[2]) for i, v in enumerate(gridprops_legacy["vertices"])
    }
    verts_new_dict = {i: (v[1], v[2]) for i, v in enumerate(gridprops_new["vertices"])}

    # Find unique vertices in each approach
    verts_legacy_set = {
        (round(x, 6), round(y, 6)) for x, y in verts_legacy_dict.values()
    }
    verts_new_set = {(round(x, 6), round(y, 6)) for x, y in verts_new_dict.values()}

    only_in_legacy = verts_legacy_set - verts_new_set
    only_in_new = verts_new_set - verts_legacy_set
    common = verts_legacy_set & verts_new_set

    print(f"\n{'Vertex Analysis':<30} {'Count':<20}")
    print("-" * 50)
    print(f"{'Common vertices':<30} {len(common):<20}")
    print(f"{'Only in legacy':<30} {len(only_in_legacy):<20}")
    print(f"{'Only in new':<30} {len(only_in_new):<20}")

    if only_in_legacy:
        print("\nVertices only in legacy (to_cvfd):")
        for x, y in sorted(only_in_legacy):
            print(f"  ({x:8.2f}, {y:8.2f})")

    if only_in_new:
        print("\nVertices only in new (Lgr):")
        for x, y in sorted(only_in_new):
            print(f"  ({x:8.2f}, {y:8.2f})")

    # Analyze cell vertex counts
    legacy_cell_nverts = [cell[3] for cell in gridprops_legacy["cell2d"]]
    new_cell_nverts = [cell[3] for cell in gridprops_new["cell2d"]]

    print(f"\n{'Cell Vertex Counts':<30} {'Legacy':<20} {'New':<20}")
    print("-" * 70)
    print(
        f"{'Min vertices per cell':<30} "
        f"{min(legacy_cell_nverts):<20} "
        f"{min(new_cell_nverts):<20}"
    )
    print(
        f"{'Max vertices per cell':<30} "
        f"{max(legacy_cell_nverts):<20} "
        f"{max(new_cell_nverts):<20}"
    )
    print(
        f"{'Avg vertices per cell':<30} "
        f"{np.mean(legacy_cell_nverts):<20.2f} "
        f"{np.mean(new_cell_nverts):<20.2f}"
    )

    # Find cells with different vertex counts
    print("\nCells with different vertex counts:")
    diff_count = 0
    for icell in range(min(len(legacy_cell_nverts), len(new_cell_nverts))):
        if legacy_cell_nverts[icell] != new_cell_nverts[icell]:
            print(
                f"  Cell {icell}: legacy={legacy_cell_nverts[icell]}, "
                f"new={new_cell_nverts[icell]}"
            )
            diff_count += 1
    if diff_count == 0:
        print("  None - all cells have same vertex counts")

    # Identify hanging vertices by analyzing parent-child boundary cells
    print(f"\n{'Hanging Vertex Analysis':<50}")
    print("-" * 50)

    # For each parent cell at the boundary, check vertex counts
    print("\nParent boundary cells (adjacent to refinement):")
    boundary_parents = [
        (0, "top-left"),
        (1, "top-center"),
        (2, "top-right"),
        (3, "middle-left"),
        (5, "middle-right"),
        (6, "bottom-left"),
        (7, "bottom-center"),
        (8, "bottom-right"),
    ]

    for icell, label in boundary_parents:
        if icell < len(legacy_cell_nverts):
            legacy_nv = legacy_cell_nverts[icell]
            new_nv = new_cell_nverts[icell]
            status = "DIFFERENT" if legacy_nv != new_nv else "same"
            print(
                f"  Cell {icell} ({label:15s}): "
                f"legacy={legacy_nv}, new={new_nv} [{status}]"
            )

    # Detailed vertex analysis - check for duplicates
    print(f"\n{'Duplicate Vertex Analysis':<50}")
    print("-" * 50)

    # Count occurrences of each coordinate in new approach
    new_coord_counts = {}
    for i, (x, y) in verts_new_dict.items():
        coord = (round(x, 6), round(y, 6))
        if coord not in new_coord_counts:
            new_coord_counts[coord] = []
        new_coord_counts[coord].append(i)

    duplicates_new = {
        coord: indices
        for coord, indices in new_coord_counts.items()
        if len(indices) > 1
    }
    if duplicates_new:
        print(f"Found {len(duplicates_new)} duplicate coordinates in new approach:")
        for coord, indices in sorted(duplicates_new.items()):
            print(f"  {coord}: vertex indices {indices}")
    else:
        print("No duplicate coordinates in new approach")

    # Count occurrences in legacy approach
    legacy_coord_counts = {}
    for i, (x, y) in verts_legacy_dict.items():
        coord = (round(x, 6), round(y, 6))
        if coord not in legacy_coord_counts:
            legacy_coord_counts[coord] = []
        legacy_coord_counts[coord].append(i)

    duplicates_legacy = {
        coord: indices
        for coord, indices in legacy_coord_counts.items()
        if len(indices) > 1
    }
    if duplicates_legacy:
        print(
            f"Found {len(duplicates_legacy)} duplicate coordinates in legacy approach:"
        )
        for coord, indices in sorted(duplicates_legacy.items()):
            print(f"  {coord}: vertex indices {indices}")
    else:
        print("No duplicate coordinates in legacy approach")

    assert (
        gridprops_legacy["nvert"] == gridprops_new["nvert"]
        and len(only_in_legacy) == 0
        and len(only_in_new) == 0
    )
