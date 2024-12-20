import matplotlib.pyplot as plt
import numpy as np
import pytest
from modflow_devtools.markers import requires_pkg
from modflow_devtools.misc import has_pkg

import flopy.discretization as fgrid
from flopy.utils.gridintersect import GridIntersect
from flopy.utils.triangle import Triangle

# TODO: remove all structured tests in v3.10.0, see TODO's in the tests

if has_pkg("shapely", strict=True):
    from shapely.geometry import (
        LineString,
        MultiLineString,
        MultiPoint,
        MultiPolygon,
        Point,
        Polygon,
    )

rtree_toggle = pytest.mark.parametrize("rtree", [True, False])


def get_tri_grid(angrot=0.0, xyoffset=0.0, triangle_exe=None):
    if not triangle_exe:
        cell2d = [
            [0, 16.666666666666668, 13.333333333333334, 3, 4, 2, 7],
            [1, 3.3333333333333335, 6.666666666666667, 3, 4, 0, 5],
            [2, 6.666666666666667, 16.666666666666668, 3, 1, 8, 4],
            [3, 3.3333333333333335, 13.333333333333334, 3, 5, 1, 4],
            [4, 6.666666666666667, 3.3333333333333335, 3, 6, 0, 4],
            [5, 13.333333333333334, 3.3333333333333335, 3, 4, 3, 6],
            [6, 16.666666666666668, 6.666666666666667, 3, 7, 3, 4],
            [7, 13.333333333333334, 16.666666666666668, 3, 8, 2, 4],
        ]
        vertices = [
            [0, 0.0, 0.0],
            [1, 0.0, 20.0],
            [2, 20.0, 20.0],
            [3, 20.0, 0.0],
            [4, 10.0, 10.0],
            [5, 0.0, 10.0],
            [6, 10.0, 0.0],
            [7, 20.0, 10.0],
            [8, 10.0, 20.0],
        ]
    else:
        maximum_area = 50.0
        x0, x1, y0, y1 = (0.0, 20.0, 0.0, 20.0)
        domainpoly = [(x0, y0), (x0, y1), (x1, y1), (x1, y0)]
        tri = Triangle(
            maximum_area=maximum_area,
            angle=45,
            model_ws=".",
            exe_name=triangle_exe,
        )
        tri.add_polygon(domainpoly)
        tri.build(verbose=False)
        cell2d = tri.get_cell2d()
        vertices = tri.get_vertices()
    tgr = fgrid.VertexGrid(
        vertices,
        cell2d,
        botm=np.atleast_2d(np.zeros(len(cell2d))),
        top=np.ones(len(cell2d)),
        xoff=xyoffset,
        yoff=xyoffset,
        angrot=angrot,
    )
    return tgr


def get_rect_grid(angrot=0.0, xyoffset=0.0, top=None, botm=None):
    delc = 10 * np.ones(2, dtype=float)
    delr = 10 * np.ones(2, dtype=float)
    sgr = fgrid.StructuredGrid(
        delc,
        delr,
        top=top,
        botm=botm,
        xoff=xyoffset,
        yoff=xyoffset,
        angrot=angrot,
    )
    return sgr


def get_rect_vertex_grid(angrot=0.0, xyoffset=0.0):
    cell2d = [
        [0, 5.0, 5.0, 4, 0, 1, 4, 3],
        [1, 15.0, 5.0, 4, 1, 2, 5, 4],
        [2, 5.0, 15.0, 4, 3, 4, 7, 6],
        [3, 15.0, 15.0, 4, 4, 5, 8, 7],
    ]
    vertices = [
        [0, 0.0, 0.0],
        [1, 10.0, 0.0],
        [2, 20.0, 0.0],
        [3, 0.0, 10.0],
        [4, 10.0, 10.0],
        [5, 20.0, 10.0],
        [6, 0.0, 20.0],
        [7, 10.0, 20.0],
        [8, 20.0, 20.0],
    ]
    tgr = fgrid.VertexGrid(
        vertices,
        cell2d,
        botm=np.atleast_2d(np.zeros(len(cell2d))),
        top=np.ones(len(cell2d)),
        xoff=xyoffset,
        yoff=xyoffset,
        angrot=angrot,
    )
    return tgr


# %% test point structured


@requires_pkg("shapely")
def test_rect_grid_3d_point_outside():
    botm = np.concatenate([np.ones(4), np.zeros(4)]).reshape((2, 2, 2))
    gr = get_rect_grid(top=np.ones(4).reshape((2, 2)), botm=botm)
    ix = GridIntersect(gr, method="vertex")
    result = ix.intersect(Point(25.0, 25.0, 0.5))
    assert len(result) == 0


# TODO: fix 3D point tests to work when above or below grid
# @requires_pkg("shapely")
# def test_rect_grid_3d_point_inside():
#     botm = np.concatenate(
#         [
#             np.ones(4),
#             0.5 * np.ones(4),
#             np.zeros(4),
#         ]
#     ).reshape((3, 2, 2))
#     gr = get_rect_grid(top=np.ones(4).reshape((2, 2)), botm=botm)
#     ix = GridIntersect(gr, method="vertex")
#     result = ix.intersect(Point(2.0, 2.0, 0.2))
#     assert result.cellids[0] == (1, 0)


# @requires_pkg("shapely")
# def test_rect_grid_3d_point_above():
#     botm = np.concatenate([np.ones(4), np.zeros(4)]).reshape((2, 2, 2))
#     gr = get_rect_grid(top=np.ones(4).reshape((2, 2)), botm=botm)
#     ix = GridIntersect(gr, method="vertex")
#     result = ix.intersect(Point(2.0, 2.0, 2.0))
#     assert len(result) == 0


@requires_pkg("shapely")
def test_rect_grid_point_outside():
    # TODO: remove in 3.10.0
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="structured")
    # use GeoSpatialUtil to convert to shapely geometry
    result = ix.intersect((25.0, 25.0), shapetype="point")
    assert len(result) == 0


@requires_pkg("shapely")
def test_rect_grid_point_on_outer_boundary():
    # TODO: remove in 3.10.0
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="structured")
    result = ix.intersect(Point(20.0, 10.0))
    assert len(result) == 1
    assert np.all(result.cellids[0] == (0, 1))


@requires_pkg("shapely")
def test_rect_grid_point_on_inner_boundary():
    # TODO: remove in 3.10.0
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="structured")
    result = ix.intersect(Point(10.0, 10.0))
    assert len(result) == 1
    assert np.all(result.cellids[0] == (0, 0))


@requires_pkg("shapely")
def test_rect_grid_multipoint_in_one_cell():
    # TODO: remove in 3.10.0
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="structured")
    result = ix.intersect(MultiPoint([Point(1.0, 1.0), Point(2.0, 2.0)]))
    assert len(result) == 1
    assert result.cellids[0] == (1, 0)


@requires_pkg("shapely")
def test_rect_grid_multipoint_in_multiple_cells():
    # TODO: remove in 3.10.0
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="structured")
    result = ix.intersect(MultiPoint([Point(1.0, 1.0), Point(12.0, 12.0)]))
    assert len(result) == 2
    assert result.cellids[0] == (1, 0)
    assert result.cellids[1] == (0, 1)


# %% test point shapely


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_point_outside_shapely(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="vertex", rtree=rtree)
    # use GeoSpatialUtil to convert to shapely geometry
    result = ix.intersect((25.0, 25.0), shapetype="point")
    assert len(result) == 0


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_point_on_outer_boundary_shapely(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="vertex", rtree=rtree)
    result = ix.intersect(Point(20.0, 10.0))
    assert len(result) == 1
    assert np.all(result.cellids[0] == (0, 1))


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_point_on_inner_boundary_shapely(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="vertex", rtree=rtree)
    result = ix.intersect(Point(10.0, 10.0))
    assert len(result) == 1
    assert np.all(result.cellids[0] == (0, 0))


@requires_pkg("shapely")
@rtree_toggle
def test_rect_vertex_grid_point_in_one_cell_shapely(rtree):
    gr = get_rect_vertex_grid()
    ix = GridIntersect(gr, method="vertex", rtree=rtree)
    result = ix.intersect(Point(4.0, 4.0))
    assert len(result) == 1
    assert result.cellids[0] == 0
    result = ix.intersect(Point(4.0, 6.0))
    assert len(result) == 1
    assert result.cellids[0] == 0
    result = ix.intersect(Point(6.0, 6.0))
    assert len(result) == 1
    assert result.cellids[0] == 0
    result = ix.intersect(Point(6.0, 4.0))
    assert len(result) == 1
    assert result.cellids[0] == 0


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_multipoint_in_one_cell_shapely(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="vertex", rtree=rtree)
    result = ix.intersect(MultiPoint([Point(1.0, 1.0), Point(2.0, 2.0)]))
    assert len(result) == 1
    assert result.cellids[0] == (1, 0)


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_multipoint_in_multiple_cells_shapely(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="vertex", rtree=rtree)
    result = ix.intersect(MultiPoint([Point(1.0, 1.0), Point(12.0, 12.0)]))
    assert len(result) == 2
    assert result.cellids[0] == (0, 1)
    assert result.cellids[1] == (1, 0)


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_point_outside(rtree):
    gr = get_tri_grid()
    if gr == -1:
        return
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect(Point(25.0, 25.0))
    assert len(result) == 0


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_point_on_outer_boundary(rtree):
    gr = get_tri_grid()
    if gr == -1:
        return
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect(Point(20.0, 10.0))
    assert len(result) == 1
    assert np.all(result.cellids[0] == 0)


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_point_on_inner_boundary(rtree):
    gr = get_tri_grid()
    if gr == -1:
        return
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect(Point(10.0, 10.0))
    assert len(result) == 1
    assert np.all(result.cellids[0] == 0)


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_multipoint_in_one_cell(rtree):
    gr = get_tri_grid()
    if gr == -1:
        return
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect(MultiPoint([Point(1.0, 1.0), Point(2.0, 2.0)]))
    assert len(result) == 1
    assert result.cellids[0] == 1


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_multipoint_in_multiple_cells(rtree):
    gr = get_tri_grid()
    if gr == -1:
        return
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect(MultiPoint([Point(1.0, 1.0), Point(12.0, 12.0)]))
    assert len(result) == 2
    assert result.cellids[0] == 0
    assert result.cellids[1] == 1


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_point_on_all_vertices_return_all_ix(rtree):
    # TODO: remove in 3.10.0
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="structured", rtree=rtree)
    n_intersections = [1, 2, 1, 2, 4, 2, 1, 2, 1]
    for v, n in zip(gr.verts, n_intersections):
        r = ix.intersect(Point(*v), return_all_intersections=True)
        assert len(r) == n


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_point_on_all_vertices_return_all_ix_shapely(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="vertex", rtree=rtree)
    n_intersections = [1, 2, 1, 2, 4, 2, 1, 2, 1]
    for v, n in zip(gr.verts, n_intersections):
        r = ix.intersect(Point(*v), return_all_intersections=True)
        assert len(r) == n


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_points_on_all_vertices_return_all_ix_shapely(rtree):
    gr = get_tri_grid()
    ix = GridIntersect(gr, method="vertex", rtree=rtree)
    n_intersections = [2, 2, 2, 2, 8, 2, 2, 2, 2]
    for v, n in zip(gr.verts, n_intersections):
        r = ix.intersect(Point(*v), return_all_intersections=True)
        assert len(r) == n


# %% test linestring structured


@requires_pkg("shapely")
def test_rect_grid_linestring_outside():
    # TODO: remove in 3.10.0
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="structured")
    result = ix.intersect(LineString([(25.0, 25.0), (21.0, 5.0)]))
    assert len(result) == 0


@requires_pkg("shapely")
def test_rect_grid_linestring_in_2cells():
    # TODO: remove in 3.10.0
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="structured")
    result = ix.intersect(LineString([(5.0, 5.0), (15.0, 5.0)]))
    assert len(result) == 2
    assert result.lengths.sum() == 10.0
    assert result.cellids[0] == (1, 0)
    assert result.cellids[1] == (1, 1)


@requires_pkg("shapely")
def test_rect_grid_linestring_on_outer_boundary():
    # TODO: remove in 3.10.0
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="structured")
    result = ix.intersect(LineString([(15.0, 20.0), (5.0, 20.0)]))
    assert len(result) == 2
    assert result.lengths.sum() == 10.0
    assert result.cellids[1] == (0, 0)
    assert result.cellids[0] == (0, 1)


@requires_pkg("shapely")
def test_rect_grid_linestring_on_inner_boundary():
    # TODO: remove in 3.10.0
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="structured")
    result = ix.intersect(LineString([(5.0, 10.0), (15.0, 10.0)]))
    assert len(result) == 2
    assert result.lengths.sum() == 10.0
    assert result.cellids[0] == (0, 0)
    assert result.cellids[1] == (0, 1)


@requires_pkg("shapely")
def test_rect_grid_multilinestring_in_one_cell():
    # TODO: remove in 3.10.0
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="structured")
    result = ix.intersect(
        MultiLineString(
            [LineString([(1.0, 1), (9.0, 1.0)]), LineString([(1.0, 9.0), (9.0, 9.0)])]
        )
    )
    assert len(result) == 1
    assert result.lengths == 16.0
    assert result.cellids[0] == (1, 0)


@requires_pkg("shapely")
def test_rect_grid_multilinestring_in_multiple_cells():
    # TODO: remove in 3.10.0
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="structured")
    result = ix.intersect(
        MultiLineString(
            [
                LineString([(20.0, 0.0), (7.5, 12.0), (2.5, 7.0), (0.0, 4.5)]),
                LineString([(5.0, 19.0), (2.5, 7.0)]),
            ]
        )
    )
    assert len(result) == 3
    assert np.allclose(sum(result.lengths), 40.19197584109293)


@requires_pkg("shapely")
def test_rect_grid_linestring_in_and_out_of_cell():
    # TODO: remove in 3.10.0
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="structured")
    result = ix.intersect(LineString([(5.0, 9), (15.0, 5.0), (5.0, 1.0)]))
    assert len(result) == 2
    assert result.cellids[0] == (1, 0)
    assert result.cellids[1] == (1, 1)
    assert np.allclose(result.lengths.sum(), 21.540659228538015)


@requires_pkg("shapely")
def test_rect_grid_linestring_in_and_out_of_cell2():
    # TODO: remove in 3.10.0
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="structured")
    result = ix.intersect(LineString([(5, 15), (5.0, 9), (15.0, 5.0), (5.0, 1.0)]))
    assert len(result) == 3


@requires_pkg("shapely")
def test_rect_grid_linestrings_on_boundaries_return_all_ix():
    # TODO: remove in 3.10.0
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="structured")
    x, y = ix._rect_grid_to_geoms_cellids()[0][0].exterior.xy
    n_intersections = [1, 2, 2, 1]
    for i in range(4):
        ls = LineString([(x[i], y[i]), (x[i + 1], y[i + 1])])
        r = ix.intersect(ls, return_all_intersections=True)
        assert len(r) == n_intersections[i]


@requires_pkg("shapely")
def test_rect_grid_linestring_starting_on_vertex():
    # TODO: remove in 3.10.0
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="structured")
    result = ix.intersect(LineString([(10.0, 10.0), (15.0, 5.0)]))
    assert len(result) == 1
    assert np.allclose(result.lengths.sum(), np.sqrt(50))
    assert result.cellids[0] == (1, 1)


# %% test linestring shapely


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_linestring_outside_shapely(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="vertex", rtree=rtree)
    result = ix.intersect(LineString([(25.0, 25.0), (21.0, 5.0)]))
    assert len(result) == 0


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_linestring_in_2cells_shapely(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="vertex", rtree=rtree)
    result = ix.intersect(LineString([(5.0, 5.0), (15.0, 5.0)]))
    assert len(result) == 2
    assert result.lengths.sum() == 10.0
    assert result.cellids[0] == (1, 0)
    assert result.cellids[1] == (1, 1)


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_linestring_on_outer_boundary_shapely(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="vertex", rtree=rtree)
    result = ix.intersect(LineString([(15.0, 20.0), (5.0, 20.0)]))
    assert len(result) == 2
    assert result.lengths.sum() == 10.0
    assert result.cellids[0] == (0, 0)
    assert result.cellids[1] == (0, 1)


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_linestring_on_inner_boundary_shapely(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="vertex", rtree=rtree)
    result = ix.intersect(LineString([(5.0, 10.0), (15.0, 10.0)]))
    assert len(result) == 2
    assert result.lengths.sum() == 10.0
    assert result.cellids[0] == (0, 0)
    assert result.cellids[1] == (0, 1)


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_multilinestring_in_one_cell_shapely(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="vertex", rtree=rtree)
    result = ix.intersect(
        MultiLineString(
            [LineString([(1.0, 1), (9.0, 1.0)]), LineString([(1.0, 9.0), (9.0, 9.0)])]
        )
    )
    assert len(result) == 1
    assert result.lengths == 16.0
    assert result.cellids[0] == (1, 0)


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_multilinestring_in_multiple_cells_shapely(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="vertex", rtree=rtree)
    result = ix.intersect(
        MultiLineString(
            [
                LineString([(20.0, 0.0), (7.5, 12.0), (2.5, 7.0), (0.0, 4.5)]),
                LineString([(5.0, 19.0), (2.5, 7.0)]),
            ]
        )
    )
    assert len(result) == 3
    assert np.allclose(sum(result.lengths), 40.19197584109293)


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_linestring_in_and_out_of_cell_shapely(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="vertex", rtree=rtree)
    result = ix.intersect(LineString([(5.0, 9), (15.0, 5.0), (5.0, 1.0)]))
    assert len(result) == 2
    assert result.cellids[0] == (1, 0)
    assert result.cellids[1] == (1, 1)
    assert np.allclose(result.lengths.sum(), 21.540659228538015)


@requires_pkg("shapely")
def test_rect_grid_linestring_in_and_out_of_cell2_shapely():
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="vertex")
    result = ix.intersect(LineString([(5, 15), (5.0, 9), (15.0, 5.0), (5.0, 1.0)]))
    assert len(result) == 3


@requires_pkg("shapely")
def test_rect_grid_linestring_starting_on_vertex_shapely():
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="vertex")
    result = ix.intersect(LineString([(10.0, 10.0), (15.0, 5.0)]))
    assert len(result) == 1
    assert np.allclose(result.lengths.sum(), np.sqrt(50))
    assert result.cellids[0] == (1, 1)


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_linestrings_on_boundaries_return_all_ix_shapely(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="vertex", rtree=rtree)
    x, y = ix._rect_grid_to_geoms_cellids()[0][0].exterior.xy
    n_intersections = [1, 2, 2, 1]
    for i in range(4):
        ls = LineString([(x[i], y[i]), (x[i + 1], y[i + 1])])
        r = ix.intersect(ls, return_all_intersections=True)
        assert len(r) == n_intersections[i]


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_linestring_cell_boundary_shapely(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="vertex", rtree=rtree)
    ls = LineString(ix._rect_grid_to_geoms_cellids()[0][0].exterior.coords)
    r = ix.intersect(ls, return_all_intersections=False)
    assert len(r) == 1


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_linestring_cell_boundary_return_all_ix_shapely(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="vertex", rtree=rtree)
    ls = LineString(ix._rect_grid_to_geoms_cellids()[0][0].exterior.coords)
    r = ix.intersect(ls, return_all_intersections=True)
    assert len(r) == 3


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_linestring_outside(rtree):
    gr = get_tri_grid()
    if gr == -1:
        return
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect(LineString([(25.0, 25.0), (21.0, 5.0)]))
    assert len(result) == 0


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_linestring_in_2cells(rtree):
    gr = get_tri_grid()
    if gr == -1:
        return
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect(LineString([(5.0, 5.0), (5.0, 15.0)]))
    assert len(result) == 2
    assert result.lengths.sum() == 10.0
    assert result.cellids[0] == 1
    assert result.cellids[1] == 3


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_linestring_on_outer_boundary(rtree):
    gr = get_tri_grid()
    if gr == -1:
        return
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect(LineString([(15.0, 20.0), (5.0, 20.0)]))
    assert len(result) == 2
    assert result.lengths.sum() == 10.0
    assert result.cellids[0] == 2
    assert result.cellids[1] == 7


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_linestring_on_inner_boundary(rtree):
    gr = get_tri_grid()
    if gr == -1:
        return
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect(LineString([(5.0, 10.0), (15.0, 10.0)]))
    assert len(result) == 2
    assert result.lengths.sum() == 10.0
    assert result.cellids[0] == 0
    assert result.cellids[1] == 1


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_multilinestring_in_one_cell(rtree):
    gr = get_tri_grid()
    if gr == -1:
        return
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect(
        MultiLineString(
            [LineString([(1.0, 1), (9.0, 1.0)]), LineString([(2.0, 2.0), (9.0, 2.0)])]
        )
    )
    assert len(result) == 1
    assert result.lengths == 15.0
    assert result.cellids[0] == 4


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_multilinestring_in_multiple_cells(rtree):
    gr = get_tri_grid()
    if gr == -1:
        return
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect(
        MultiLineString(
            [
                LineString([(20.0, 0.0), (7.5, 12.0), (2.5, 7.0), (0.0, 4.5)]),
                LineString([(5.0, 19.0), (2.5, 7.0)]),
            ]
        )
    )
    assert len(result) == 5
    assert np.allclose(sum(result.lengths), 40.19197584109293)


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_linestrings_on_boundaries_return_all_ix(rtree):
    tgr = get_tri_grid()
    ix = GridIntersect(tgr, method="vertex", rtree=rtree)
    x, y = ix._vtx_grid_to_geoms_cellids()[0][0].exterior.xy
    n_intersections = [2, 1, 2]
    for i in range(len(x) - 1):
        ls = LineString([(x[i], y[i]), (x[i + 1], y[i + 1])])
        r = ix.intersect(ls, return_all_intersections=True)
        assert len(r) == n_intersections[i]


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_linestring_cell_boundary_shapely(rtree):
    tgr = get_tri_grid()
    ix = GridIntersect(tgr, method="vertex", rtree=rtree)
    ls = LineString(ix._vtx_grid_to_geoms_cellids()[0][0].exterior.coords)
    r = ix.intersect(ls, return_all_intersections=False)
    assert len(r) == 1


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_linestring_cell_boundary_return_all_ix_shapely(rtree):
    tgr = get_tri_grid()
    ix = GridIntersect(tgr, method="vertex", rtree=rtree)
    ls = LineString(ix._vtx_grid_to_geoms_cellids()[0][0].exterior.coords)
    r = ix.intersect(ls, return_all_intersections=True)
    assert len(r) == 3


@requires_pkg("shapely")
def test_rect_vertex_grid_linestring_geomcollection():
    gr = get_rect_vertex_grid()
    ix = GridIntersect(gr, method="vertex")
    ls = LineString(
        [
            (20.0, 0.0),
            (5.0, 5.0),
            (15.0, 7.5),
            (10.0, 10.0),
            (5.0, 15.0),
            (10.0, 19.0),
            (10.0, 20.0),
        ]
    )
    result = ix.intersect(ls)
    assert len(result) == 3
    assert np.allclose(result.lengths.sum(), ls.length)


# %% test polygon structured


@requires_pkg("shapely")
def test_rect_grid_polygon_outside():
    # TODO: remove in 3.10.0
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="structured")
    result = ix.intersect(Polygon([(21.0, 11.0), (23.0, 17.0), (25.0, 11.0)]))
    assert len(result) == 0


@requires_pkg("shapely")
def test_rect_grid_polygon_in_2cells():
    # TODO: remove in 3.10.0
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="structured")
    result = ix.intersect(Polygon([(2.5, 5.0), (7.5, 5.0), (7.5, 15.0), (2.5, 15.0)]))
    assert len(result) == 2
    assert result.areas.sum() == 50.0


@requires_pkg("shapely")
def test_rect_grid_polygon_on_outer_boundary():
    # TODO: remove in 3.10.0
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="structured")
    result = ix.intersect(
        Polygon([(20.0, 5.0), (25.0, 5.0), (25.0, 15.0), (20.0, 15.0)])
    )
    assert len(result) == 0


@requires_pkg("shapely")
def test_rect_grid_polygon_running_along_boundary():
    # TODO: remove in 3.10.0
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="structured")
    result = ix.intersect(
        Polygon(
            [(5.0, 5.0), (5.0, 10.0), (9.0, 10.0), (9.0, 15.0), (1.0, 15.0), (1.0, 5.0)]
        )
    )


@requires_pkg("shapely")
def test_rect_grid_polygon_on_inner_boundary():
    # TODO: remove in 3.10.0
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="structured")
    result = ix.intersect(Polygon([(5.0, 10.0), (15.0, 10.0), (15.0, 5.0), (5.0, 5.0)]))
    assert len(result) == 2
    assert result.areas.sum() == 50.0

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    gr.plot(ax=ax)
    ix.plot_polygon(result, ax=ax)
    # plt.show()


@requires_pkg("shapely")
def test_rect_grid_polygon_multiple_polygons():
    # TODO: remove in 3.10.0
    gr = get_rect_grid()
    p = Polygon(
        [
            (0, 0),
            (0, 10),
            (4, 10),
            (4, 0),
            (6, 0),
            (6, 10),
            (9, 10),
            (9, -1),
            (0, -1),
            (0, 0),
        ]
    )

    ix = GridIntersect(gr, method="structured")
    result = ix.intersect(p)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    gr.plot(ax=ax)
    ix.plot_polygon(result, ax=ax)
    # plt.show()


@requires_pkg("shapely")
def test_rect_grid_multiple_disjoint_polygons_on_inner_boundaries():
    # TODO: remove in 3.10.0
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="structured")
    p1 = Polygon([(5.0, 10.0), (15.0, 10.0), (15.0, 5.0), (5.0, 5.0)])
    p2 = Polygon([(5.0, 17.5), (15.0, 17.5), (15.0, 12.5), (5.0, 12.5)])
    result = ix.intersect(MultiPolygon([p1, p2]))

    assert len(result) == 4
    assert result.areas.sum() == 100.0

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    gr.plot(ax=ax)
    ix.plot_polygon(result, ax=ax)
    # plt.show()


@requires_pkg("shapely")
@pytest.mark.parametrize("transform", [True, False])
def test_rect_grid_polygon_reintersects_cell(transform):
    # TODO: remove in 3.10.0
    gr = get_rect_grid()
    if transform:
        gr.set_coord_info(xoff=1, yoff=1, angrot=10.5)

    ix = GridIntersect(gr, method="structured")
    p1 = Polygon(
        [
            (x, y + 3)
            for x, y in [
                (2.5, 2.5),
                (2.5, 10.0),
                (10.0, 10.0),
                (10.0, 2.5),
                (7.5, 2.5),
                (7.5, 7.5),
                (5.0, 7.5),
                (5.0, 2.5),
            ]
        ]
    )
    p2 = Polygon([(1, 1), (1, 2), (2, 2), (2, 1)])
    result = ix.intersect(MultiPolygon([p1, p2]))

    assert len(result) == 4 if transform else 2
    assert np.isclose(result.areas.sum(), 44.65733 if transform else 44.75)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    gr.plot(ax=ax)
    ix.plot_polygon(result, ax=ax)
    # plt.show()


@requires_pkg("shapely")
def test_rect_grid_multipolygon_in_one_cell():
    # TODO: remove in 3.10.0
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="structured")
    p1 = Polygon([(1.0, 1.0), (8.0, 1.0), (8.0, 3.0), (1.0, 3.0)])
    p2 = Polygon([(1.0, 9.0), (8.0, 9.0), (8.0, 7.0), (1.0, 7.0)])
    p = MultiPolygon([p1, p2])
    result = ix.intersect(p)
    assert len(result) == 1
    assert result.areas.sum() == 28.0


@requires_pkg("shapely")
def test_rect_grid_multipolygon_in_multiple_cells():
    # TODO: remove in 3.10.0
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="structured")
    p1 = Polygon([(1.0, 1.0), (19.0, 1.0), (19.0, 3.0), (1.0, 3.0)])
    p2 = Polygon([(1.0, 9.0), (19.0, 9.0), (19.0, 7.0), (1.0, 7.0)])
    p = MultiPolygon([p1, p2])
    result = ix.intersect(p)
    assert len(result) == 2
    assert result.areas.sum() == 72.0


@requires_pkg("shapely")
def test_rect_grid_polygon_with_hole():
    # TODO: remove in 3.10.0
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="structured")
    p = Polygon(
        [(5.0, 5.0), (5.0, 15.0), (25.0, 15.0), (25.0, -5.0), (5.0, -5.0)],
        holes=[[(9.0, -1), (9, 11), (21, 11), (21, -1)]],
    )
    result = ix.intersect(p)
    assert len(result) == 3
    assert result.areas.sum() == 104.0


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_polygon_contains_centroid(rtree):
    # TODO: remove in 3.10.0
    gr = get_rect_grid()
    ix = GridIntersect(gr, rtree=rtree)
    p = Polygon(
        [(6.0, 5.0), (4.0, 16.0), (25.0, 14.0), (25.0, -5.0), (6.0, -5.0)],
        holes=[[(9.0, -1), (9, 11), (21, 11), (21, -1)]],
    )
    result = ix.intersect(p, contains_centroid=True)
    assert len(result) == 1


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_polygon_min_area(rtree):
    # TODO: remove in 3.10.0
    gr = get_rect_grid()
    ix = GridIntersect(gr, rtree=rtree)
    p = Polygon(
        [(5.0, 5.0), (5.0, 15.0), (25.0, 15.0), (25.0, -5.0), (5.0, -5.0)],
        holes=[[(9.0, -1), (9, 11), (21, 11), (21, -1)]],
    )
    result = ix.intersect(p, min_area_fraction=0.4)
    assert len(result) == 2


@requires_pkg("shapely")
def test_rect_grid_polygon_centroid_and_min_area():
    # TODO: remove in 3.10.0
    gr = get_rect_grid()
    ix = GridIntersect(gr)
    p = Polygon(
        [(5.0, 5.0), (5.0, 15.0), (25.0, 14.0), (25.0, -5.0), (5.0, -5.0)],
        holes=[[(9.0, -1), (9, 11), (21, 11), (21, -1)]],
    )
    result = ix.intersect(p, min_area_fraction=0.35, contains_centroid=True)
    assert len(result) == 1


# %% test polygon shapely


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_polygon_outside_shapely(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="vertex", rtree=rtree)
    result = ix.intersect(Polygon([(21.0, 11.0), (23.0, 17.0), (25.0, 11.0)]))
    assert len(result) == 0


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_polygon_in_2cells_shapely(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="vertex", rtree=rtree)
    result = ix.intersect(Polygon([(2.5, 5.0), (7.5, 5.0), (7.5, 15.0), (2.5, 15.0)]))
    assert len(result) == 2
    assert result.areas.sum() == 50.0


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_polygon_on_outer_boundary_shapely(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="vertex", rtree=rtree)
    result = ix.intersect(
        Polygon([(20.0, 5.0), (25.0, 5.0), (25.0, 15.0), (20.0, 15.0)])
    )
    assert len(result) == 0


@requires_pkg("shapely")
def test_rect_grid_polygon_running_along_boundary_shapely():
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="vertex")
    result = ix.intersect(
        Polygon(
            [(5.0, 5.0), (5.0, 10.0), (9.0, 10.0), (9.0, 15.0), (1.0, 15.0), (1.0, 5.0)]
        )
    )


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_polygon_on_inner_boundary_shapely(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="vertex", rtree=rtree)
    result = ix.intersect(Polygon([(5.0, 10.0), (15.0, 10.0), (15.0, 5.0), (5.0, 5.0)]))
    assert len(result) == 2
    assert result.areas.sum() == 50.0


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_multipolygon_in_one_cell_shapely(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="vertex", rtree=rtree)
    p1 = Polygon([(1.0, 1.0), (8.0, 1.0), (8.0, 3.0), (1.0, 3.0)])
    p2 = Polygon([(1.0, 9.0), (8.0, 9.0), (8.0, 7.0), (1.0, 7.0)])
    p = MultiPolygon([p1, p2])
    result = ix.intersect(p)
    assert len(result) == 1
    assert result.areas.sum() == 28.0


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_multipolygon_in_multiple_cells_shapely(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="vertex", rtree=rtree)
    p1 = Polygon([(1.0, 1.0), (19.0, 1.0), (19.0, 3.0), (1.0, 3.0)])
    p2 = Polygon([(1.0, 9.0), (19.0, 9.0), (19.0, 7.0), (1.0, 7.0)])
    p = MultiPolygon([p1, p2])
    result = ix.intersect(p)
    assert len(result) == 2
    assert result.areas.sum() == 72.0


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_polygon_with_hole_shapely(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="vertex", rtree=rtree)
    p = Polygon(
        [(5.0, 5.0), (5.0, 15.0), (25.0, 15.0), (25.0, -5.0), (5.0, -5.0)],
        holes=[[(9.0, -1), (9, 11), (21, 11), (21, -1)]],
    )
    result = ix.intersect(p)
    assert len(result) == 3
    assert result.areas.sum() == 104.0


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_polygon_in_edge_in_cell(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, method="vertex", rtree=rtree)
    p = Polygon(
        [(0.0, 5.0), (3.0, 0.0), (7.0, 0.0), (10.0, 5.0), (10.0, -1.0), (0.0, -1.0)]
    )
    result = ix.intersect(p)
    assert len(result) == 1
    assert result.areas.sum() == 15.0


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_polygon_outside(rtree):
    gr = get_tri_grid()
    if gr == -1:
        return
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect(Polygon([(21.0, 11.0), (23.0, 17.0), (25.0, 11.0)]))
    assert len(result) == 0


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_polygon_in_2cells(rtree):
    gr = get_tri_grid()
    if gr == -1:
        return
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect(Polygon([(2.5, 5.0), (5.0, 5.0), (5.0, 15.0), (2.5, 15.0)]))
    assert len(result) == 2
    assert result.areas.sum() == 25.0


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_polygon_on_outer_boundary(rtree):
    gr = get_tri_grid()
    if gr == -1:
        return
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect(
        Polygon([(20.0, 5.0), (25.0, 5.0), (25.0, 15.0), (20.0, 15.0)])
    )
    assert len(result) == 0


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_polygon_on_inner_boundary(rtree):
    gr = get_tri_grid()
    if gr == -1:
        return
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect(Polygon([(5.0, 10.0), (15.0, 10.0), (15.0, 5.0), (5.0, 5.0)]))
    assert len(result) == 4
    assert result.areas.sum() == 50.0


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_multipolygon_in_one_cell(rtree):
    gr = get_tri_grid()
    if gr == -1:
        return
    ix = GridIntersect(gr, rtree=rtree)
    p1 = Polygon([(1.0, 1.0), (8.0, 1.0), (8.0, 3.0), (3.0, 3.0)])
    p2 = Polygon([(5.0, 5.0), (8.0, 5.0), (8.0, 8.0)])
    p = MultiPolygon([p1, p2])
    result = ix.intersect(p)
    assert len(result) == 1
    assert result.areas.sum() == 16.5


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_multipolygon_in_multiple_cells(rtree):
    gr = get_tri_grid()
    if gr == -1:
        return
    ix = GridIntersect(gr, rtree=rtree)
    p1 = Polygon([(1.0, 1.0), (19.0, 1.0), (19.0, 3.0), (1.0, 3.0)])
    p2 = Polygon([(1.0, 9.0), (19.0, 9.0), (19.0, 7.0), (1.0, 7.0)])
    p = MultiPolygon([p1, p2])
    result = ix.intersect(p)
    assert len(result) == 4
    assert result.areas.sum() == 72.0


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_polygon_with_hole(rtree):
    gr = get_tri_grid()
    if gr == -1:
        return
    ix = GridIntersect(gr, rtree=rtree)
    p = Polygon(
        [(5.0, 5.0), (5.0, 15.0), (25.0, 15.0), (25.0, -5.0), (5.0, -5.0)],
        holes=[[(9.0, -1), (9, 11), (21, 11), (21, -1)]],
    )
    result = ix.intersect(p)
    assert len(result) == 6
    assert result.areas.sum() == 104.0


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_polygon_min_area(rtree):
    gr = get_tri_grid()
    if gr == -1:
        return
    ix = GridIntersect(gr, rtree=rtree)
    p = Polygon(
        [(5.0, 5.0), (5.0, 15.0), (25.0, 15.0), (25.0, -5.0), (5.0, -5.0)],
        holes=[[(9.0, -1), (9, 11), (21, 11), (21, -1)]],
    )
    result = ix.intersect(p, min_area_fraction=0.5)
    assert len(result) == 2


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_polygon_contains_centroid(rtree):
    gr = get_tri_grid()
    if gr == -1:
        return
    ix = GridIntersect(gr, rtree=rtree)
    p = Polygon(
        [(5.0, 5.0), (6.0, 14.0), (25.0, 15.0), (25.0, -5.0), (5.0, -5.0)],
        holes=[[(9.0, -1), (9, 11), (21, 11), (21, -1)]],
    )
    result = ix.intersect(p, contains_centroid=True)
    assert len(result) == 2


# %% test rotated offset grids


@requires_pkg("shapely")
def test_point_offset_rot_structured_grid():
    # TODO: remove in 3.10.0
    sgr = get_rect_grid(angrot=45.0, xyoffset=10.0)
    p = Point(10.0, 10 + np.sqrt(200.0))
    ix = GridIntersect(sgr, method="structured")
    result = ix.intersect(p)
    assert len(result) == 1
    # check empty result when using local model coords
    ix = GridIntersect(sgr, method="structured", local=True)
    result = ix.intersect(p)
    assert len(result) == 0


@requires_pkg("shapely")
def test_linestring_offset_rot_structured_grid():
    # TODO: remove in 3.10.0
    sgr = get_rect_grid(angrot=45.0, xyoffset=10.0)
    ls = LineString([(5, 25), (15, 25)])
    ix = GridIntersect(sgr, method="structured")
    result = ix.intersect(ls)
    assert len(result) == 3
    # check empty result when using local model coords
    ix = GridIntersect(sgr, method="structured", local=True)
    result = ix.intersect(ls)
    assert len(result) == 0


@requires_pkg("shapely")
def test_polygon_offset_rot_structured_grid():
    # TODO: remove in 3.10.0
    sgr = get_rect_grid(angrot=45.0, xyoffset=10.0)
    p = Polygon(
        [
            (5, 10.0 + np.sqrt(200.0)),
            (15, 10.0 + np.sqrt(200.0)),
            (15, 10.0 + 1.5 * np.sqrt(200.0)),
            (5, 10.0 + 1.5 * np.sqrt(200.0)),
        ]
    )
    ix = GridIntersect(sgr, method="structured")
    result = ix.intersect(p)
    assert len(result) == 3
    # check empty result when using local model coords
    ix = GridIntersect(sgr, method="structured", local=True)
    result = ix.intersect(p)
    assert len(result) == 0


@requires_pkg("shapely")
@rtree_toggle
def test_point_offset_rot_structured_grid_shapely(rtree):
    sgr = get_rect_grid(angrot=45.0, xyoffset=10.0)
    p = Point(10.0, 10 + np.sqrt(200.0))
    ix = GridIntersect(sgr, method="vertex", rtree=rtree)
    result = ix.intersect(p)
    assert len(result) == 1
    # check empty result when using local model coords
    ix = GridIntersect(sgr, method="vertex", rtree=rtree, local=True)
    result = ix.intersect(p)
    assert len(result) == 0


@requires_pkg("shapely")
@rtree_toggle
def test_linestring_offset_rot_structured_grid_shapely(rtree):
    sgr = get_rect_grid(angrot=45.0, xyoffset=10.0)
    ls = LineString([(5, 10.0 + np.sqrt(200.0)), (15, 10.0 + np.sqrt(200.0))])
    ix = GridIntersect(sgr, method="vertex", rtree=rtree)
    result = ix.intersect(ls)
    assert len(result) == 2
    # check empty result when using local model coords
    ix = GridIntersect(sgr, method="vertex", rtree=rtree, local=True)
    result = ix.intersect(ls)
    assert len(result) == 0


@requires_pkg("shapely")
@rtree_toggle
def test_polygon_offset_rot_structured_grid_shapely(rtree):
    sgr = get_rect_grid(angrot=45.0, xyoffset=10.0)
    p = Polygon(
        [
            (5, 10.0 + np.sqrt(200.0)),
            (15, 10.0 + np.sqrt(200.0)),
            (15, 10.0 + 1.5 * np.sqrt(200.0)),
            (5, 10.0 + 1.5 * np.sqrt(200.0)),
        ]
    )
    ix = GridIntersect(sgr, method="vertex", rtree=rtree)
    result = ix.intersect(p)
    assert len(result) == 3
    # check empty result when using local model coords
    ix = GridIntersect(sgr, method="vertex", rtree=rtree, local=True)
    result = ix.intersect(p)
    assert len(result) == 0


@requires_pkg("shapely")
@rtree_toggle
def test_point_offset_rot_vertex_grid_shapely(rtree):
    sgr = get_rect_vertex_grid(angrot=45.0, xyoffset=10.0)
    p = Point(10.0, 10 + np.sqrt(200.0))
    ix = GridIntersect(sgr, method="vertex", rtree=rtree)
    result = ix.intersect(p)
    assert len(result) == 1
    # check empty result when using local model coords
    ix = GridIntersect(sgr, method="vertex", rtree=rtree, local=True)
    result = ix.intersect(p)
    assert len(result) == 0


@requires_pkg("shapely")
@rtree_toggle
def test_linestring_offset_rot_vertex_grid_shapely(rtree):
    sgr = get_rect_vertex_grid(angrot=45.0, xyoffset=10.0)
    ls = LineString([(5, 10.0 + np.sqrt(200.0)), (15, 10.0 + np.sqrt(200.0))])
    ix = GridIntersect(sgr, method="vertex", rtree=rtree)
    result = ix.intersect(ls)
    assert len(result) == 2
    # check empty result when using local model coords
    ix = GridIntersect(sgr, method="vertex", rtree=rtree, local=True)
    result = ix.intersect(ls)
    assert len(result) == 0


@requires_pkg("shapely")
@rtree_toggle
def test_polygon_offset_rot_vertex_grid_shapely(rtree):
    sgr = get_rect_vertex_grid(angrot=45.0, xyoffset=10.0)
    p = Polygon(
        [
            (5, 10.0 + np.sqrt(200.0)),
            (15, 10.0 + np.sqrt(200.0)),
            (15, 10.0 + 1.5 * np.sqrt(200.0)),
            (5, 10.0 + 1.5 * np.sqrt(200.0)),
        ]
    )
    ix = GridIntersect(sgr, method="vertex", rtree=rtree)
    result = ix.intersect(p)
    assert len(result) == 3
    # check empty result when using local model coords
    ix = GridIntersect(sgr, method="vertex", rtree=rtree, local=True)
    result = ix.intersect(p)
    assert len(result) == 0
