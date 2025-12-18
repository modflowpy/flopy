import numpy as np
import pytest

from flopy.discretization import StructuredGrid
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
    assert gridprops["nvert"] == 32
    assert gridprops["nlay"] == 3

    # test the lgr to disv class
    lgrtodisv = LgrToDisv(lgr)

    # test guts of LgrToDisv to make sure hanging vertices added correctly
    assert lgrtodisv.right_face_hanging[1, 0] == [0, 4, 8, 12]
    assert lgrtodisv.left_face_hanging[1, 2] == [3, 7, 11, 15]
    assert lgrtodisv.back_face_hanging[2, 1] == [12, 13, 14, 15]
    assert lgrtodisv.front_face_hanging[0, 1] == [0, 1, 2, 3]

    assert lgrtodisv.iverts[1] == [1, 2, 6, 18, 17, 5]
    assert lgrtodisv.iverts[3] == [4, 5, 20, 24, 9, 8]
    assert lgrtodisv.iverts[4] == [6, 7, 11, 10, 27, 23]
    assert lgrtodisv.iverts[6] == [9, 29, 30, 10, 14, 13]

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
