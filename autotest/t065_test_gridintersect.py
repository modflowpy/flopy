import sys
sys.path.insert(1, "..")
import flopy.discretization as fgrid
import flopy.plot as fplot
import matplotlib.pyplot as plt
import numpy as np
from descartes import PolygonPatch
from flopy.utils.triangle import Triangle
try:
    from shapely.geometry import (LineString, MultiLineString, MultiPoint,
                                  MultiPolygon, Point, Polygon)
except Exception as e:
    print("Shapely not installed, tests cannot be run.")
from flopy.utils.gridintersect import GridIntersect


def get_tri_grid(angrot=0., xyoffset=0., triangle_exe=None):
    if not triangle_exe:
        cell2d = [[0, 16.666666666666668, 13.333333333333334, 3, 4, 2, 7],
                  [1, 3.3333333333333335, 6.666666666666667, 3, 4, 0, 5],
                  [2, 6.666666666666667, 16.666666666666668, 3, 1, 8, 4],
                  [3, 3.3333333333333335, 13.333333333333334, 3, 5, 1, 4],
                  [4, 6.666666666666667, 3.3333333333333335, 3, 6, 0, 4],
                  [5, 13.333333333333334, 3.3333333333333335, 3, 4, 3, 6],
                  [6, 16.666666666666668, 6.666666666666667, 3, 7, 3, 4],
                  [7, 13.333333333333334, 16.666666666666668, 3, 8, 2, 4]]
        vertices = [[0, 0.0, 0.0],
                    [1, 0.0, 20.0],
                    [2, 20.0, 20.0],
                    [3, 20.0, 0.0],
                    [4, 10.0, 10.0],
                    [5, 0.0, 10.0],
                    [6, 10.0, 0.0],
                    [7, 20.0, 10.0],
                    [8, 10.0, 20.0]]
    else:
        maximum_area = 50.
        x0, x1, y0, y1 = (0.0, 20.0, 0.0, 20.0)
        domainpoly = [(x0, y0), (x0, y1), (x1, y1), (x1, y0)]
        tri = Triangle(maximum_area=maximum_area, angle=45, model_ws=".",
                       exe_name=triangle_exe)
        tri.add_polygon(domainpoly)
        tri.build(verbose=False)
        cell2d = tri.get_cell2d()
        vertices = tri.get_vertices()
    tgr = fgrid.VertexGrid(vertices, cell2d,
                           botm=np.atleast_2d(np.zeros(len(cell2d))),
                           top=np.ones(len(cell2d)), xoff=xyoffset,
                           yoff=xyoffset, angrot=angrot)
    return tgr


def get_rect_grid(angrot=0., xyoffset=0.):
    delc = 10 * np.ones(2, dtype=np.float)
    delr = 10 * np.ones(2, dtype=np.float)
    sgr = fgrid.StructuredGrid(
        delc, delr, top=None, botm=None, xoff=xyoffset, yoff=xyoffset,
        angrot=angrot)
    return sgr


def plot_structured_grid(sgr):
    _, ax = plt.subplots(1, 1, figsize=(8, 8))
    sgr.plot(ax=ax)
    return ax


def plot_vertex_grid(tgr):
    _, ax = plt.subplots(1, 1, figsize=(8, 8))
    pmv = fplot.PlotMapView(modelgrid=tgr)
    pmv.plot_grid(ax=ax)
    return ax


def plot_ix_polygon_result(rec, ax):
    for i, ishp in enumerate(rec.ixshapes):
        ppi = PolygonPatch(ishp, facecolor="C{}".format(i % 10))
        ax.add_patch(ppi)


def plot_ix_linestring_result(rec, ax):
    for i, ishp in enumerate(rec.ixshapes):
        if ishp.type == "MultiLineString":
            for part in ishp:
                ax.plot(part.xy[0], part.xy[1], ls="-",
                        c="C{}".format(i % 10))
        else:
            ax.plot(ishp.xy[0], ishp.xy[1], ls="-",
                    c="C{}".format(i % 10))


def plot_ix_point_result(rec, ax):
    x = [ip.x for ip in rec.ixshapes]
    y = [ip.y for ip in rec.ixshapes]
    ax.scatter(x, y)


# %% test point structured


def test_rect_grid_point_outside():
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="structured")
    result = ix.intersect_point(Point(25., 25.))
    assert len(result) == 0
    return result


def test_rect_grid_point_on_outer_boundary():
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="structured")
    result = ix.intersect_point(Point(20., 10.))
    assert len(result) == 1
    assert np.all(result.cellids[0] == (0, 1))
    return result


def test_rect_grid_point_on_inner_boundary():
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="structured")
    result = ix.intersect_point(Point(10., 10.))
    assert len(result) == 1
    assert np.all(result.cellids[0] == (0, 0))
    return result


def test_rect_grid_multipoint_in_one_cell():
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="structured")
    result = ix.intersect_point(MultiPoint([Point(1., 1.), Point(2., 2.)]))
    assert len(result) == 1
    assert result.cellids[0] == (1, 0)
    return result


def test_rect_grid_multipoint_in_multiple_cells():
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="structured")
    result = ix.intersect_point(MultiPoint([Point(1., 1.), Point(12., 12.)]))
    assert len(result) == 2
    assert result.cellids[0] == (1, 0)
    assert result.cellids[1] == (0, 1)
    return result


# %% test point shapely


def test_rect_grid_point_outside_shapely(rtree=True):
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_rect_grid()
    ix = GridIntersect(gr, method='vertex', rtree=rtree)
    result = ix.intersect_point(Point(25., 25.))
    assert len(result) == 0
    return result


def test_rect_grid_point_on_outer_boundary_shapely(rtree=True):
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_rect_grid()
    ix = GridIntersect(gr, method='vertex', rtree=rtree)
    result = ix.intersect_point(Point(20., 10.))
    assert len(result) == 1
    assert np.all(result.cellids[0] == (0, 1))
    return result


def test_rect_grid_point_on_inner_boundary_shapely(rtree=True):
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_rect_grid()
    ix = GridIntersect(gr, method='vertex', rtree=rtree)
    result = ix.intersect_point(Point(10., 10.))
    assert len(result) == 1
    assert np.all(result.cellids[0] == (0, 0))
    return result


def test_rect_grid_multipoint_in_one_cell_shapely(rtree=True):
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_rect_grid()
    ix = GridIntersect(gr, method='vertex', rtree=rtree)
    result = ix.intersect_point(MultiPoint([Point(1., 1.), Point(2., 2.)]))
    assert len(result) == 1
    assert result.cellids[0] == (1, 0)
    return result


def test_rect_grid_multipoint_in_multiple_cells_shapely(rtree=True):
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_rect_grid()
    ix = GridIntersect(gr, method='vertex', rtree=rtree)
    result = ix.intersect_point(MultiPoint([Point(1., 1.), Point(12., 12.)]))
    assert len(result) == 2
    assert result.cellids[0] == (0, 1)
    assert result.cellids[1] == (1, 0)
    return result


def test_tri_grid_point_outside(rtree=True):
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_tri_grid()
    if gr == -1:
        return
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect_point(Point(25., 25.))
    assert len(result) == 0
    return result


def test_tri_grid_point_on_outer_boundary(rtree=True):
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_tri_grid()
    if gr == -1:
        return
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect_point(Point(20., 10.))
    assert len(result) == 1
    assert np.all(result.cellids[0] == 0)
    return result


def test_tri_grid_point_on_inner_boundary(rtree=True):
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_tri_grid()
    if gr == -1:
        return
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect_point(Point(10., 10.))
    assert len(result) == 1
    assert np.all(result.cellids[0] == 0)
    return result


def test_tri_grid_multipoint_in_one_cell(rtree=True):
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_tri_grid()
    if gr == -1:
        return
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect_point(MultiPoint([Point(1., 1.), Point(2., 2.)]))
    assert len(result) == 1
    assert result.cellids[0] == 1
    return result


def test_tri_grid_multipoint_in_multiple_cells(rtree=True):
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_tri_grid()
    if gr == -1:
        return
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect_point(MultiPoint([Point(1., 1.), Point(12., 12.)]))
    assert len(result) == 2
    assert result.cellids[0] == 0
    assert result.cellids[1] == 1
    return result


# %% test linestring structured


def test_rect_grid_linestring_outside():
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="structured")
    result = ix.intersect_linestring(LineString([(25., 25.), (21., 5.)]))
    assert len(result) == 0
    return result


def test_rect_grid_linestring_in_2cells():
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="structured")
    result = ix.intersect_linestring(LineString([(5., 5.), (15., 5.)]))
    assert len(result) == 2
    assert result.lengths.sum() == 10.
    assert result.cellids[0] == (1, 0)
    assert result.cellids[1] == (1, 1)
    return result


def test_rect_grid_linestring_on_outer_boundary():
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="structured")
    result = ix.intersect_linestring(LineString([(15., 20.), (5., 20.)]))
    assert len(result) == 2
    assert result.lengths.sum() == 10.
    assert result.cellids[1] == (0, 0)
    assert result.cellids[0] == (0, 1)
    return result


def test_rect_grid_linestring_on_inner_boundary():
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="structured")
    result = ix.intersect_linestring(LineString([(5., 10.), (15., 10.)]))
    assert len(result) == 2
    assert result.lengths.sum() == 10.
    assert result.cellids[0] == (0, 0)
    assert result.cellids[1] == (0, 1)
    return result


def test_rect_grid_multilinestring_in_one_cell():
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="structured")
    result = ix.intersect_linestring(MultiLineString(
        [LineString([(1., 1), (9., 1.)]), LineString([(1., 9.), (9., 9.)])]))
    assert len(result) == 1
    assert result.lengths == 16.
    assert result.cellids[0] == (1, 0)
    return result


def test_rect_grid_linestring_in_and_out_of_cell():
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="structured")
    result = ix.intersect_linestring(
        LineString([(5., 9), (15., 5.), (5., 1.)]))
    assert len(result) == 2
    assert result.cellids[0] == (1, 0)
    assert result.cellids[1] == (1, 1)
    assert np.allclose(result.lengths.sum(), 21.540659228538015)
    return result


def test_rect_grid_linestring_in_and_out_of_cell2():
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="structured")
    result = ix.intersect_linestring(LineString(
        [(5, 15), (5., 9), (15., 5.), (5., 1.)]))
    assert len(result) == 3
    # assert result.cellids[0] == (1, 0)
    # assert result.cellids[1] == (1, 1)
    # assert np.allclose(result.lengths.sum(), 21.540659228538015)
    return result


# %% test linestring shapely


def test_rect_grid_linestring_outside_shapely(rtree=True):
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_rect_grid()
    ix = GridIntersect(gr, method='vertex', rtree=rtree)
    result = ix.intersect_linestring(LineString([(25., 25.), (21., 5.)]))
    assert len(result) == 0
    return result


def test_rect_grid_linestring_in_2cells_shapely(rtree=True):
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_rect_grid()
    ix = GridIntersect(gr, method='vertex', rtree=rtree)
    result = ix.intersect_linestring(LineString([(5., 5.), (15., 5.)]))
    assert len(result) == 2
    assert result.lengths.sum() == 10.
    assert result.cellids[0] == (1, 0)
    assert result.cellids[1] == (1, 1)
    return result


def test_rect_grid_linestring_on_outer_boundary_shapely(rtree=True):
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_rect_grid()
    ix = GridIntersect(gr, method='vertex', rtree=rtree)
    result = ix.intersect_linestring(LineString([(15., 20.), (5., 20.)]))
    assert len(result) == 2
    assert result.lengths.sum() == 10.
    assert result.cellids[0] == (0, 0)
    assert result.cellids[1] == (0, 1)
    return result


def test_rect_grid_linestring_on_inner_boundary_shapely(rtree=True):
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_rect_grid()
    ix = GridIntersect(gr, method='vertex', rtree=rtree)
    result = ix.intersect_linestring(LineString([(5., 10.), (15., 10.)]))
    assert len(result) == 2
    assert result.lengths.sum() == 10.
    assert result.cellids[0] == (0, 0)
    assert result.cellids[1] == (0, 1)
    return result


def test_rect_grid_multilinestring_in_one_cell_shapely(rtree=True):
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_rect_grid()
    ix = GridIntersect(gr, method='vertex', rtree=rtree)
    result = ix.intersect_linestring(MultiLineString(
        [LineString([(1., 1), (9., 1.)]), LineString([(1., 9.), (9., 9.)])]))
    assert len(result) == 1
    assert result.lengths == 16.
    assert result.cellids[0] == (1, 0)
    return result


def test_rect_grid_linestring_in_and_out_of_cell_shapely(rtree=True):
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_rect_grid()
    ix = GridIntersect(gr, method='vertex', rtree=rtree)
    result = ix.intersect_linestring(
        LineString([(5., 9), (15., 5.), (5., 1.)]))
    assert len(result) == 2
    assert result.cellids[0] == (1, 0)
    assert result.cellids[1] == (1, 1)
    assert np.allclose(result.lengths.sum(), 21.540659228538015)
    return result


def test_tri_grid_linestring_outside(rtree=True):
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_tri_grid()
    if gr == -1:
        return
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect_linestring(LineString([(25., 25.), (21., 5.)]))
    assert len(result) == 0
    return result


def test_tri_grid_linestring_in_2cells(rtree=True):
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_tri_grid()
    if gr == -1:
        return
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect_linestring(LineString([(5., 5.), (5., 15.)]))
    assert len(result) == 2
    assert result.lengths.sum() == 10.
    assert result.cellids[0] == 1
    assert result.cellids[1] == 3
    return result


def test_tri_grid_linestring_on_outer_boundary(rtree=True):
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_tri_grid()
    if gr == -1:
        return
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect_linestring(LineString([(15., 20.), (5., 20.)]))
    assert len(result) == 2
    assert result.lengths.sum() == 10.
    assert result.cellids[0] == 2
    assert result.cellids[1] == 7
    return result


def test_tri_grid_linestring_on_inner_boundary(rtree=True):
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_tri_grid()
    if gr == -1:
        return
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect_linestring(LineString([(5., 10.), (15., 10.)]))
    assert len(result) == 2
    assert result.lengths.sum() == 10.
    assert result.cellids[0] == 0
    assert result.cellids[1] == 1
    return result


def test_tri_grid_multilinestring_in_one_cell(rtree=True):
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_tri_grid()
    if gr == -1:
        return
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect_linestring(MultiLineString(
        [LineString([(1., 1), (9., 1.)]), LineString([(2., 2.), (9., 2.)])]))
    assert len(result) == 1
    assert result.lengths == 15.
    assert result.cellids[0] == 4
    return result


# %% test polygon structured


def test_rect_grid_polygon_outside():
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="structured")
    result = ix.intersect_polygon(
        Polygon([(21., 11.), (23., 17.), (25., 11.)]))
    assert len(result) == 0
    return result


def test_rect_grid_polygon_in_2cells():
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="structured")
    result = ix.intersect_polygon(
        Polygon([(2.5, 5.0), (7.5, 5.0), (7.5, 15.), (2.5, 15.)]))
    assert len(result) == 2
    assert result.areas.sum() == 50.
    return result


def test_rect_grid_polygon_on_outer_boundary():
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="structured")
    result = ix.intersect_polygon(
        Polygon([(20., 5.0), (25., 5.0), (25., 15.), (20., 15.)]))
    assert len(result) == 0
    return result


def test_rect_grid_polygon_on_inner_boundary():
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="structured")
    result = ix.intersect_polygon(
        Polygon([(5., 10.0), (15., 10.0), (15., 5.), (5., 5.)]))
    assert len(result) == 2
    assert result.areas.sum() == 50.
    return result


def test_rect_grid_multipolygon_in_one_cell():
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="structured")
    p1 = Polygon([(1., 1.), (8., 1.), (8., 3.), (1., 3.)])
    p2 = Polygon([(1., 9.), (8., 9.), (8., 7.), (1., 7.)])
    p = MultiPolygon([p1, p2])
    result = ix.intersect_polygon(p)
    assert len(result) == 1
    assert result.areas.sum() == 28.
    return result


def test_rect_grid_multipolygon_in_multiple_cells():
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="structured")
    p1 = Polygon([(1., 1.), (19., 1.), (19., 3.), (1., 3.)])
    p2 = Polygon([(1., 9.), (19., 9.), (19., 7.), (1., 7.)])
    p = MultiPolygon([p1, p2])
    result = ix.intersect_polygon(p)
    assert len(result) == 2
    assert result.areas.sum() == 72.
    return result


def test_rect_grid_polygon_with_hole():
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="structured")
    p = Polygon([(5., 5.), (5., 15.), (25., 15.), (25., -5.),
                 (5., -5.)], holes=[[(9., -1), (9, 11), (21, 11), (21, -1)]])
    result = ix.intersect_polygon(p)
    assert len(result) == 3
    assert result.areas.sum() == 104.
    return result


# %% test polygon shapely


def test_rect_grid_polygon_outside_shapely(rtree=True):
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_rect_grid()
    ix = GridIntersect(gr, method='vertex', rtree=rtree)
    result = ix.intersect_polygon(
        Polygon([(21., 11.), (23., 17.), (25., 11.)]))
    assert len(result) == 0
    return result


def test_rect_grid_polygon_in_2cells_shapely(rtree=True):
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_rect_grid()
    ix = GridIntersect(gr, method='vertex', rtree=rtree)
    result = ix.intersect_polygon(
        Polygon([(2.5, 5.0), (7.5, 5.0), (7.5, 15.), (2.5, 15.)]))
    assert len(result) == 2
    assert result.areas.sum() == 50.
    return result


def test_rect_grid_polygon_on_outer_boundary_shapely(rtree=True):
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_rect_grid()
    ix = GridIntersect(gr, method='vertex', rtree=rtree)
    result = ix.intersect_polygon(
        Polygon([(20., 5.0), (25., 5.0), (25., 15.), (20., 15.)]))
    assert len(result) == 0
    return result


def test_rect_grid_polygon_on_inner_boundary_shapely(rtree=True):
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_rect_grid()
    ix = GridIntersect(gr, method='vertex', rtree=rtree)
    result = ix.intersect_polygon(
        Polygon([(5., 10.0), (15., 10.0), (15., 5.), (5., 5.)]))
    assert len(result) == 2
    assert result.areas.sum() == 50.
    return result


def test_rect_grid_multipolygon_in_one_cell_shapely(rtree=True):
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_rect_grid()
    ix = GridIntersect(gr, method='vertex', rtree=rtree)
    p1 = Polygon([(1., 1.), (8., 1.), (8., 3.), (1., 3.)])
    p2 = Polygon([(1., 9.), (8., 9.), (8., 7.), (1., 7.)])
    p = MultiPolygon([p1, p2])
    result = ix.intersect_polygon(p)
    assert len(result) == 1
    assert result.areas.sum() == 28.
    return result


def test_rect_grid_multipolygon_in_multiple_cells_shapely(rtree=True):
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_rect_grid()
    ix = GridIntersect(gr, method='vertex', rtree=rtree)
    p1 = Polygon([(1., 1.), (19., 1.), (19., 3.), (1., 3.)])
    p2 = Polygon([(1., 9.), (19., 9.), (19., 7.), (1., 7.)])
    p = MultiPolygon([p1, p2])
    result = ix.intersect_polygon(p)
    assert len(result) == 2
    assert result.areas.sum() == 72.
    return result


def test_rect_grid_polygon_with_hole_shapely(rtree=True):
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_rect_grid()
    ix = GridIntersect(gr, method='vertex', rtree=rtree)
    p = Polygon([(5., 5.), (5., 15.), (25., 15.), (25., -5.),
                 (5., -5.)], holes=[[(9., -1), (9, 11), (21, 11), (21, -1)]])
    result = ix.intersect_polygon(p)
    assert len(result) == 3
    assert result.areas.sum() == 104.
    return result


def test_tri_grid_polygon_outside(rtree=True):
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_tri_grid()
    if gr == -1:
        return
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect_polygon(
        Polygon([(21., 11.), (23., 17.), (25., 11.)]))
    assert len(result) == 0
    return result


def test_tri_grid_polygon_in_2cells(rtree=True):
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_tri_grid()
    if gr == -1:
        return
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect_polygon(
        Polygon([(2.5, 5.0), (5.0, 5.0), (5.0, 15.), (2.5, 15.)]))
    assert len(result) == 2
    assert result.areas.sum() == 25.
    return result


def test_tri_grid_polygon_on_outer_boundary(rtree=True):
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_tri_grid()
    if gr == -1:
        return
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect_polygon(
        Polygon([(20., 5.0), (25., 5.0), (25., 15.), (20., 15.)]))
    assert len(result) == 0
    return result


def test_tri_grid_polygon_on_inner_boundary(rtree=True):
    gr = get_tri_grid()
    if gr == -1:
        return
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect_polygon(
        Polygon([(5., 10.0), (15., 10.0), (15., 5.), (5., 5.)]))
    assert len(result) == 4
    assert result.areas.sum() == 50.
    return result


def test_tri_grid_multipolygon_in_one_cell(rtree=True):
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_tri_grid()
    if gr == -1:
        return
    ix = GridIntersect(gr, rtree=rtree)
    p1 = Polygon([(1., 1.), (8., 1.), (8., 3.), (3., 3.)])
    p2 = Polygon([(5., 5.), (8., 5.), (8., 8.)])
    p = MultiPolygon([p1, p2])
    result = ix.intersect_polygon(p)
    assert len(result) == 1
    assert result.areas.sum() == 16.5
    return result


def test_tri_grid_multipolygon_in_multiple_cells(rtree=True):
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_tri_grid()
    if gr == -1:
        return
    ix = GridIntersect(gr, rtree=rtree)
    p1 = Polygon([(1., 1.), (19., 1.), (19., 3.), (1., 3.)])
    p2 = Polygon([(1., 9.), (19., 9.), (19., 7.), (1., 7.)])
    p = MultiPolygon([p1, p2])
    result = ix.intersect_polygon(p)
    assert len(result) == 4
    assert result.areas.sum() == 72.
    return result


def test_tri_grid_polygon_with_hole(rtree=True):
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    gr = get_tri_grid()
    if gr == -1:
        return
    ix = GridIntersect(gr, rtree=rtree)
    p = Polygon([(5., 5.), (5., 15.), (25., 15.), (25., -5.),
                 (5., -5.)], holes=[[(9., -1), (9, 11), (21, 11), (21, -1)]])
    result = ix.intersect_polygon(p)
    assert len(result) == 6
    assert result.areas.sum() == 104.
    return result


# %% test rotated offset grids


def test_point_offset_rot_structured_grid():
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    sgr = get_rect_grid(angrot=45., xyoffset=10.)
    p = Point(10., 10 + np.sqrt(200.))
    ix = GridIntersect(sgr, method="structured")
    result = ix.intersect_point(p)
    # assert len(result) == 1.
    return result


def test_linestring_offset_rot_structured_grid():
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    sgr = get_rect_grid(angrot=45., xyoffset=10.)
    ls = LineString([(5, 10. + np.sqrt(200.)), (15, 10. + np.sqrt(200.))])
    ix = GridIntersect(sgr, method="structured")
    result = ix.intersect_linestring(ls)
    # assert len(result) == 2.
    return result


def test_polygon_offset_rot_structured_grid():
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    sgr = get_rect_grid(angrot=45., xyoffset=10.)
    p = Polygon([(5, 10. + np.sqrt(200.)),
                 (15, 10. + np.sqrt(200.)),
                 (15, 10. + 1.5 * np.sqrt(200.)),
                 (5, 10. + 1.5 * np.sqrt(200.))])
    ix = GridIntersect(sgr, method="structured")
    result = ix.intersect_polygon(p)
    # assert len(result) == 3.
    return result


def test_point_offset_rot_structured_grid_shapely(rtree=True):
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    sgr = get_rect_grid(angrot=45., xyoffset=10.)
    p = Point(10., 10 + np.sqrt(200.))
    ix = GridIntersect(sgr, method="vertex", rtree=rtree)
    result = ix.intersect_point(p)
    # assert len(result) == 1.
    return result


def test_linestring_offset_rot_structured_grid_shapely(rtree=True):
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    sgr = get_rect_grid(angrot=45., xyoffset=10.)
    ls = LineString([(5, 10. + np.sqrt(200.)), (15, 10. + np.sqrt(200.))])
    ix = GridIntersect(sgr, method="vertex", rtree=rtree)
    result = ix.intersect_linestring(ls)
    # assert len(result) == 2.
    return result


def test_polygon_offset_rot_structured_grid_shapely(rtree=True):
    # avoid test fail when shapely not available
    try:
        import shapely
    except:
        return
    sgr = get_rect_grid(angrot=45., xyoffset=10.)
    p = Polygon([(5, 10. + np.sqrt(200.)),
                 (15, 10. + np.sqrt(200.)),
                 (15, 10. + 1.5 * np.sqrt(200.)),
                 (5, 10. + 1.5 * np.sqrt(200.))])
    ix = GridIntersect(sgr, method="vertex", rtree=rtree)
    result = ix.intersect_polygon(p)
    # assert len(result) == 3.
    return result


# %% test non strtree shapely intersect

def test_all_intersections_shapely_no_strtree():
    """avoid adding separate tests for rtree=False"""
    # Points
    # regular grid
    test_rect_grid_point_on_inner_boundary_shapely(rtree=False)
    test_rect_grid_point_on_outer_boundary_shapely(rtree=False)
    test_rect_grid_point_outside_shapely(rtree=False)
    test_rect_grid_multipoint_in_one_cell_shapely(rtree=False)
    test_rect_grid_multipoint_in_multiple_cells_shapely(rtree=False)
    # vertex grid
    test_tri_grid_point_on_inner_boundary(rtree=False)
    test_tri_grid_point_on_outer_boundary(rtree=False)
    test_tri_grid_point_outside(rtree=False)
    test_tri_grid_multipoint_in_multiple_cells(rtree=False)
    test_tri_grid_multipoint_in_one_cell(rtree=False)

    # LineStrings
    # regular grid
    test_rect_grid_linestring_on_inner_boundary_shapely(rtree=False)
    test_rect_grid_linestring_on_outer_boundary_shapely(rtree=False)
    test_rect_grid_linestring_outside_shapely(rtree=False)
    test_rect_grid_linestring_in_2cells_shapely(rtree=False)
    test_rect_grid_linestring_in_and_out_of_cell_shapely(rtree=False)
    test_rect_grid_multilinestring_in_one_cell_shapely(rtree=False)
    # vertex grid
    test_tri_grid_linestring_on_inner_boundary(rtree=False)
    test_tri_grid_linestring_on_outer_boundary(rtree=False)
    test_tri_grid_linestring_outside(rtree=False)
    test_tri_grid_linestring_in_2cells(rtree=False)
    test_tri_grid_multilinestring_in_one_cell(rtree=False)

    # Polygons
    # regular grid
    test_rect_grid_polygon_on_inner_boundary_shapely(rtree=False)
    test_rect_grid_polygon_on_outer_boundary_shapely(rtree=False)
    test_rect_grid_polygon_outside_shapely(rtree=False)
    test_rect_grid_polygon_in_2cells_shapely(rtree=False)
    test_rect_grid_polygon_with_hole_shapely(rtree=False)
    test_rect_grid_multipolygon_in_one_cell_shapely(rtree=False)
    test_rect_grid_multipolygon_in_multiple_cells_shapely(rtree=False)
    # vertex grid
    test_tri_grid_polygon_on_inner_boundary(rtree=False)
    test_tri_grid_polygon_on_outer_boundary(rtree=False)
    test_tri_grid_polygon_outside(rtree=False)
    test_tri_grid_polygon_in_2cells(rtree=False)
    test_tri_grid_polygon_with_hole(rtree=False)
    test_tri_grid_multipolygon_in_multiple_cells(rtree=False)
    test_tri_grid_multipolygon_in_one_cell(rtree=False)

    # offset and rotated grids
    test_point_offset_rot_structured_grid_shapely(rtree=False)
    test_linestring_offset_rot_structured_grid_shapely(rtree=False)
    ix = test_polygon_offset_rot_structured_grid_shapely(rtree=False)

    return ix


# %% test rasters


def test_rasters():
    from flopy.utils import Raster
    import os
    import flopy as fp

    ws = os.path.join("..", "examples", "data", "options")
    raster_name = "dem.img"

    try:
        rio = Raster.load(os.path.join(ws, "dem", raster_name))
    except:
        return

    ml = fp.modflow.Modflow.load("sagehen.nam", version="mfnwt",
                                 model_ws=os.path.join(ws, 'sagehen'))
    xoff = 214110
    yoff = 4366620
    ml.modelgrid.set_coord_info(xoff, yoff)

    # test sampling points and polygons
    val = rio.sample_point(xoff + 2000, yoff + 2000, band=1)
    print(val - 2336.3965)
    if abs(val - 2336.3965) > 1e-4:
        raise AssertionError

    x0, x1, y0, y1 = rio.bounds

    x0 += 1000
    y0 += 1000
    x1 -= 1000
    y1 -= 1000
    shape = np.array([(x0, y0), (x0, y1), (x1, y1), (x1, y0), (x0, y0)])

    data = rio.sample_polygon(shape, band=rio.bands[0])
    if data.size != 267050:
        raise AssertionError
    if abs(np.min(data) - 1942.1735) > 1e-4:
        raise AssertionError
    if (np.max(data) - 2608.557) > 1e-4:
        raise AssertionError

    rio.crop(shape)
    data = rio.get_array(band=rio.bands[0], masked=True)
    if data.size != 267050:
        raise AssertionError
    if abs(np.min(data) - 1942.1735) > 1e-4:
        raise AssertionError
    if (np.max(data) - 2608.557) > 1e-4:
        raise AssertionError

    data = rio.resample_to_grid(ml.modelgrid.xcellcenters,
                                ml.modelgrid.ycellcenters,
                                band=rio.bands[0],
                                method="nearest")
    if data.size != 5913:
        raise AssertionError
    if abs(np.min(data) - 1942.1735) > 1e-4:
        raise AssertionError
    if abs(np.max(data) - 2605.6204) > 1e-4:
        raise AssertionError

    del rio
