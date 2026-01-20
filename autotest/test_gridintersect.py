# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from modflow_devtools.markers import requires_pkg
from modflow_devtools.misc import has_pkg

import flopy.discretization as fgrid
from flopy.utils.gridintersect import GridIntersect
from flopy.utils.triangle import Triangle

if has_pkg("shapely", strict=True):
    from shapely import linestrings, points, polygons
    from shapely.geometry import (
        LineString,
        MultiLineString,
        MultiPoint,
        MultiPolygon,
        Point,
        Polygon,
    )

rtree_toggle = pytest.mark.parametrize("rtree", [True, False])
df_toggle = False  # set False to silence warnings, remove when dataframe is default


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


# %% test point structured shapely


@requires_pkg("shapely")
def test_rect_grid_3d_point_outside():
    botm = np.concatenate([np.ones(4), np.zeros(4)]).reshape((2, 2, 2))
    gr = get_rect_grid(top=2 * np.ones(4).reshape((2, 2)), botm=botm)
    ix = GridIntersect(gr)
    result = ix.intersect(
        Point(25.0, 25.0, 0.5), handle_z=False, geo_dataframe=df_toggle
    )
    assert len(result) == 0
    result = ix.intersect(
        Point(25.0, 25.0, 0.5), handle_z=True, geo_dataframe=df_toggle
    )
    assert len(result) == 0


@requires_pkg("shapely")
def test_rect_grid_3d_point_inside():
    botm = np.concatenate(
        [
            np.ones(4),
            0.5 * np.ones(4),
            np.zeros(4),
        ]
    ).reshape((3, 2, 2))
    gr = get_rect_grid(top=2 * np.ones(4).reshape((2, 2)), botm=botm)
    ix = GridIntersect(gr)
    result = ix.intersect(Point(2.0, 2.0, 0.2), handle_z=False, geo_dataframe=df_toggle)
    assert result["cellids"][0] == (1, 0)
    result = ix.intersect(Point(2.0, 2.0, 0.2), handle_z=True, geo_dataframe=df_toggle)
    assert result["cellids"][0] == (1, 0)
    assert result["layer"][0] == 2


@requires_pkg("shapely")
def test_rect_grid_3d_point_above():
    botm = np.concatenate([np.ones(4), np.zeros(4)]).reshape((2, 2, 2))
    gr = get_rect_grid(top=2 * np.ones(4).reshape((2, 2)), botm=botm)
    ix = GridIntersect(gr)
    result = ix.intersect(
        Point(2.0, 2.0, 10.0), handle_z=False, geo_dataframe=df_toggle
    )
    assert len(result) == 1
    assert result.cellids[0] == (1, 0)
    result = ix.intersect(Point(2.0, 2.0, 10.0), handle_z=True, geo_dataframe=df_toggle)
    assert len(result) == 1
    assert np.ma.is_masked(result["layer"][0])


# %% test point shapely


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_point_outside(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, rtree=rtree)
    # use GeoSpatialUtil to convert to shapely geometry
    result = ix.intersect((25.0, 25.0), shapetype="point", geo_dataframe=df_toggle)
    assert len(result) == 0


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_point_on_outer_boundary(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect(Point(20.0, 10.0), geo_dataframe=df_toggle)
    assert len(result) == 1
    assert np.all(result.cellids[0] == (0, 1))


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_point_on_inner_boundary(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect(Point(10.0, 10.0), geo_dataframe=df_toggle)
    assert len(result) == 1
    assert np.all(result.cellids[0] == (0, 0))


@requires_pkg("shapely")
@rtree_toggle
def test_rect_vertex_grid_point_in_one_cell(rtree):
    gr = get_rect_vertex_grid()
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect(Point(4.0, 4.0), geo_dataframe=df_toggle)
    assert len(result) == 1
    assert result.cellids[0] == 0
    result = ix.intersect(Point(4.0, 6.0), geo_dataframe=df_toggle)
    assert len(result) == 1
    assert result.cellids[0] == 0
    result = ix.intersect(Point(6.0, 6.0), geo_dataframe=df_toggle)
    assert len(result) == 1
    assert result.cellids[0] == 0
    result = ix.intersect(Point(6.0, 4.0), geo_dataframe=df_toggle)
    assert len(result) == 1
    assert result.cellids[0] == 0


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_multipoint_in_one_cell(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect(
        MultiPoint([Point(1.0, 1.0), Point(2.0, 2.0)]), geo_dataframe=df_toggle
    )
    assert len(result) == 1
    assert result.cellids[0] == (1, 0)


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_multipoint_in_multiple_cells(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect(
        MultiPoint([Point(1.0, 1.0), Point(12.0, 12.0)]), geo_dataframe=df_toggle
    )
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
    result = ix.intersect(Point(25.0, 25.0), geo_dataframe=df_toggle)
    assert len(result) == 0


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_point_on_outer_boundary(rtree):
    gr = get_tri_grid()
    if gr == -1:
        return
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect(Point(20.0, 10.0), geo_dataframe=df_toggle)
    assert len(result) == 1
    assert np.all(result.cellids[0] == 0)


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_point_on_inner_boundary(rtree):
    gr = get_tri_grid()
    if gr == -1:
        return
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect(Point(10.0, 10.0), geo_dataframe=df_toggle)
    assert len(result) == 1
    assert np.all(result.cellids[0] == 0)


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_multipoint_in_one_cell(rtree):
    gr = get_tri_grid()
    if gr == -1:
        return
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect(
        MultiPoint([Point(1.0, 1.0), Point(2.0, 2.0)]), geo_dataframe=df_toggle
    )
    assert len(result) == 1
    assert result.cellids[0] == 1


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_multipoint_in_multiple_cells(rtree):
    gr = get_tri_grid()
    if gr == -1:
        return
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect(
        MultiPoint([Point(1.0, 1.0), Point(12.0, 12.0)]), geo_dataframe=df_toggle
    )
    assert len(result) == 2
    assert result.cellids[0] == 0
    assert result.cellids[1] == 1


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_point_on_all_vertices_return_all_ix(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, rtree=rtree)
    n_intersections = [1, 2, 1, 2, 4, 2, 1, 2, 1]
    for v, n in zip(gr.verts, n_intersections):
        r = ix.intersect(
            Point(*v), return_all_intersections=True, geo_dataframe=df_toggle
        )
        assert len(r) == n


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_points_on_all_vertices_return_all_ix(rtree):
    gr = get_tri_grid()
    ix = GridIntersect(gr, rtree=rtree)
    n_intersections = [2, 2, 2, 2, 8, 2, 2, 2, 2]
    for v, n in zip(gr.verts, n_intersections):
        r = ix.intersect(
            Point(*v), return_all_intersections=True, geo_dataframe=df_toggle
        )
        assert len(r) == n


# %% test linestring shapely


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_linestring_outside(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect(
        LineString([(25.0, 25.0), (21.0, 5.0)]), geo_dataframe=df_toggle
    )
    assert len(result) == 0


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_linestring_in_2cells(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect(
        LineString([(5.0, 5.0), (15.0, 5.0)]), geo_dataframe=df_toggle
    )
    assert len(result) == 2
    assert result.lengths.sum() == 10.0
    assert result.cellids[0] == (1, 0)
    assert result.cellids[1] == (1, 1)


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_linestring_on_outer_boundary(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect(
        LineString([(15.0, 20.0), (5.0, 20.0)]), geo_dataframe=df_toggle
    )
    assert len(result) == 2
    assert result.lengths.sum() == 10.0
    assert result.cellids[0] == (0, 0)
    assert result.cellids[1] == (0, 1)


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_linestring_on_inner_boundary(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect(
        LineString([(5.0, 10.0), (15.0, 10.0)]), geo_dataframe=df_toggle
    )
    assert len(result) == 2
    assert result.lengths.sum() == 10.0
    assert result.cellids[0] == (0, 0)
    assert result.cellids[1] == (0, 1)


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_multilinestring_in_one_cell(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect(
        MultiLineString(
            [LineString([(1.0, 1), (9.0, 1.0)]), LineString([(1.0, 9.0), (9.0, 9.0)])]
        ),
        geo_dataframe=df_toggle,
    )
    assert len(result) == 1
    assert result.lengths == 16.0
    assert result.cellids[0] == (1, 0)


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_multilinestring_in_multiple_cells(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect(
        MultiLineString(
            [
                LineString([(20.0, 0.0), (7.5, 12.0), (2.5, 7.0), (0.0, 4.5)]),
                LineString([(5.0, 19.0), (2.5, 7.0)]),
            ]
        ),
        geo_dataframe=df_toggle,
    )
    assert len(result) == 3
    assert np.allclose(sum(result.lengths), 40.19197584109293)


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_linestring_in_and_out_of_cell(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect(
        LineString([(5.0, 9), (15.0, 5.0), (5.0, 1.0)]), geo_dataframe=df_toggle
    )
    assert len(result) == 2
    assert result.cellids[0] == (1, 0)
    assert result.cellids[1] == (1, 1)
    assert np.allclose(result.lengths.sum(), 21.540659228538015)


@requires_pkg("shapely")
def test_rect_grid_linestring_in_and_out_of_cell2():
    gr = get_rect_grid()
    ix = GridIntersect(gr)
    result = ix.intersect(
        LineString([(5, 15), (5.0, 9), (15.0, 5.0), (5.0, 1.0)]),
        geo_dataframe=df_toggle,
    )
    assert len(result) == 3


@requires_pkg("shapely")
def test_rect_grid_linestring_starting_on_vertex():
    gr = get_rect_grid()
    ix = GridIntersect(gr)
    result = ix.intersect(
        LineString([(10.0, 10.0), (15.0, 5.0)]), geo_dataframe=df_toggle
    )
    assert len(result) == 1
    assert np.allclose(result.lengths.sum(), np.sqrt(50))
    assert result.cellids[0] == (1, 1)


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_linestrings_on_boundaries_return_all_ix(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, rtree=rtree)
    x, y = ix._rect_grid_to_geoms_cellids()[0][0].exterior.xy
    n_intersections = [1, 2, 2, 1]
    for i in range(4):
        ls = LineString([(x[i], y[i]), (x[i + 1], y[i + 1])])
        r = ix.intersect(ls, return_all_intersections=True, geo_dataframe=df_toggle)
        assert len(r) == n_intersections[i]


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_linestring_cell_boundary(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, rtree=rtree)
    ls = LineString(ix._rect_grid_to_geoms_cellids()[0][0].exterior.coords)
    r = ix.intersect(ls, return_all_intersections=False, geo_dataframe=df_toggle)
    assert len(r) == 1


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_linestring_cell_boundary_return_all_ix(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, rtree=rtree)
    ls = LineString(ix._rect_grid_to_geoms_cellids()[0][0].exterior.coords)
    r = ix.intersect(ls, return_all_intersections=True, geo_dataframe=df_toggle)
    assert len(r) == 3


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_linestring_outside(rtree):
    gr = get_tri_grid()
    if gr == -1:
        return
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect(
        LineString([(25.0, 25.0), (21.0, 5.0)]), geo_dataframe=df_toggle
    )
    assert len(result) == 0


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_linestring_in_2cells(rtree):
    gr = get_tri_grid()
    if gr == -1:
        return
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect(
        LineString([(5.0, 5.0), (5.0, 15.0)]), geo_dataframe=df_toggle
    )
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
    result = ix.intersect(
        LineString([(15.0, 20.0), (5.0, 20.0)]), geo_dataframe=df_toggle
    )
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
    result = ix.intersect(
        LineString([(5.0, 10.0), (15.0, 10.0)]), geo_dataframe=df_toggle
    )
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
        ),
        geo_dataframe=df_toggle,
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
        ),
        geo_dataframe=df_toggle,
    )
    assert len(result) == 5
    assert np.allclose(sum(result.lengths), 40.19197584109293)


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_linestrings_on_boundaries_return_all_ix(rtree):
    tgr = get_tri_grid()
    ix = GridIntersect(tgr, rtree=rtree)
    x, y = ix._vtx_grid_to_geoms_cellids()[0][0].exterior.xy
    n_intersections = [2, 1, 2]
    for i in range(len(x) - 1):
        ls = LineString([(x[i], y[i]), (x[i + 1], y[i + 1])])
        r = ix.intersect(ls, return_all_intersections=True, geo_dataframe=df_toggle)
        assert len(r) == n_intersections[i]


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_linestring_cell_boundary(rtree):
    tgr = get_tri_grid()
    ix = GridIntersect(tgr, rtree=rtree)
    ls = LineString(ix._vtx_grid_to_geoms_cellids()[0][0].exterior.coords)
    r = ix.intersect(ls, return_all_intersections=False, geo_dataframe=df_toggle)
    assert len(r) == 1


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_linestring_cell_boundary_return_all_ix(rtree):
    tgr = get_tri_grid()
    ix = GridIntersect(tgr, rtree=rtree)
    ls = LineString(ix._vtx_grid_to_geoms_cellids()[0][0].exterior.coords)
    r = ix.intersect(ls, return_all_intersections=True, geo_dataframe=df_toggle)
    assert len(r) == 3


@requires_pkg("shapely")
def test_rect_vertex_grid_linestring_geomcollection():
    gr = get_rect_vertex_grid()
    ix = GridIntersect(gr)
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
    result = ix.intersect(ls, geo_dataframe=df_toggle)
    assert len(result) == 3
    assert np.allclose(result.lengths.sum(), ls.length)


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_polygon_contains_centroid(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, rtree=rtree)
    p = Polygon(
        [(6.0, 5.0), (4.0, 16.0), (25.0, 14.0), (25.0, -5.0), (6.0, -5.0)],
        holes=[[(9.0, -1), (9, 11), (21, 11), (21, -1)]],
    )
    result = ix.intersect(p, contains_centroid=True, geo_dataframe=df_toggle)
    assert len(result) == 1


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_polygon_min_area(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, rtree=rtree)
    p = Polygon(
        [(5.0, 5.0), (5.0, 15.0), (25.0, 15.0), (25.0, -5.0), (5.0, -5.0)],
        holes=[[(9.0, -1), (9, 11), (21, 11), (21, -1)]],
    )
    result = ix.intersect(p, min_area_fraction=0.4, geo_dataframe=df_toggle)
    assert len(result) == 2


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_polygon_centroid_and_min_area(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, rtree=rtree)
    p = Polygon(
        [(5.0, 5.0), (5.0, 15.0), (25.0, 14.0), (25.0, -5.0), (5.0, -5.0)],
        holes=[[(9.0, -1), (9, 11), (21, 11), (21, -1)]],
    )
    result = ix.intersect(
        p, min_area_fraction=0.35, contains_centroid=True, geo_dataframe=df_toggle
    )
    assert len(result) == 1


# %% test polygon shapely


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_polygon_outside(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect(
        Polygon([(21.0, 11.0), (23.0, 17.0), (25.0, 11.0)]), geo_dataframe=df_toggle
    )
    assert len(result) == 0


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_polygon_in_2cells(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect(
        Polygon([(2.5, 5.0), (7.5, 5.0), (7.5, 15.0), (2.5, 15.0)]),
        geo_dataframe=df_toggle,
    )
    assert len(result) == 2
    assert result.areas.sum() == 50.0


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_polygon_on_outer_boundary(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect(
        Polygon([(20.0, 5.0), (25.0, 5.0), (25.0, 15.0), (20.0, 15.0)]),
        geo_dataframe=df_toggle,
    )
    assert len(result) == 0


@requires_pkg("shapely")
def test_rect_grid_polygon_running_along_boundary():
    gr = get_rect_grid()
    ix = GridIntersect(gr)
    result = ix.intersect(
        Polygon(
            [(5.0, 5.0), (5.0, 10.0), (9.0, 10.0), (9.0, 15.0), (1.0, 15.0), (1.0, 5.0)]
        ),
        geo_dataframe=df_toggle,
    )


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_polygon_on_inner_boundary(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect(
        Polygon([(5.0, 10.0), (15.0, 10.0), (15.0, 5.0), (5.0, 5.0)]),
        geo_dataframe=df_toggle,
    )
    assert len(result) == 2
    assert result.areas.sum() == 50.0


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_multipolygon_in_one_cell(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, rtree=rtree)
    p1 = Polygon([(1.0, 1.0), (8.0, 1.0), (8.0, 3.0), (1.0, 3.0)])
    p2 = Polygon([(1.0, 9.0), (8.0, 9.0), (8.0, 7.0), (1.0, 7.0)])
    p = MultiPolygon([p1, p2])
    result = ix.intersect(p, geo_dataframe=df_toggle)
    assert len(result) == 1
    assert result.areas.sum() == 28.0


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_multipolygon_in_multiple_cells(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, rtree=rtree)
    p1 = Polygon([(1.0, 1.0), (19.0, 1.0), (19.0, 3.0), (1.0, 3.0)])
    p2 = Polygon([(1.0, 9.0), (19.0, 9.0), (19.0, 7.0), (1.0, 7.0)])
    p = MultiPolygon([p1, p2])
    result = ix.intersect(p, geo_dataframe=df_toggle)
    assert len(result) == 2
    assert result.areas.sum() == 72.0


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_polygon_with_hole(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, rtree=rtree)
    p = Polygon(
        [(5.0, 5.0), (5.0, 15.0), (25.0, 15.0), (25.0, -5.0), (5.0, -5.0)],
        holes=[[(9.0, -1), (9, 11), (21, 11), (21, -1)]],
    )
    result = ix.intersect(p, geo_dataframe=df_toggle)
    assert len(result) == 3
    assert result.areas.sum() == 104.0


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_polygon_in_edge_in_cell(rtree):
    gr = get_rect_grid()
    ix = GridIntersect(gr, rtree=rtree)
    p = Polygon(
        [(0.0, 5.0), (3.0, 0.0), (7.0, 0.0), (10.0, 5.0), (10.0, -1.0), (0.0, -1.0)]
    )
    result = ix.intersect(p, geo_dataframe=df_toggle)
    assert len(result) == 1
    assert result.areas.sum() == 15.0


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_polygon_outside(rtree):
    gr = get_tri_grid()
    if gr == -1:
        return
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect(
        Polygon([(21.0, 11.0), (23.0, 17.0), (25.0, 11.0)]), geo_dataframe=df_toggle
    )
    assert len(result) == 0


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_polygon_in_2cells(rtree):
    gr = get_tri_grid()
    if gr == -1:
        return
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect(
        Polygon([(2.5, 5.0), (5.0, 5.0), (5.0, 15.0), (2.5, 15.0)]),
        geo_dataframe=df_toggle,
    )
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
        Polygon([(20.0, 5.0), (25.0, 5.0), (25.0, 15.0), (20.0, 15.0)]),
        geo_dataframe=df_toggle,
    )
    assert len(result) == 0


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_polygon_on_inner_boundary(rtree):
    gr = get_tri_grid()
    if gr == -1:
        return
    ix = GridIntersect(gr, rtree=rtree)
    result = ix.intersect(
        Polygon([(5.0, 10.0), (15.0, 10.0), (15.0, 5.0), (5.0, 5.0)]),
        geo_dataframe=df_toggle,
    )
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
    result = ix.intersect(p, geo_dataframe=df_toggle)
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
    result = ix.intersect(p, geo_dataframe=df_toggle)
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
    result = ix.intersect(p, geo_dataframe=df_toggle)
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
    result = ix.intersect(p, min_area_fraction=0.5, geo_dataframe=df_toggle)
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
    result = ix.intersect(p, contains_centroid=True, geo_dataframe=df_toggle)
    assert len(result) == 2


# %% test rotated offset grids


@requires_pkg("shapely")
@rtree_toggle
def test_point_offset_rot_structured_grid(rtree):
    sgr = get_rect_grid(angrot=45.0, xyoffset=10.0)
    p = Point(10.0, 10 + np.sqrt(200.0))
    ix = GridIntersect(sgr, rtree=rtree)
    result = ix.intersect(p, geo_dataframe=df_toggle)
    assert len(result) == 1
    # check empty result when using local model coords
    ix = GridIntersect(sgr, rtree=rtree, local=True)
    result = ix.intersect(p, geo_dataframe=df_toggle)
    assert len(result) == 0


@requires_pkg("shapely")
@rtree_toggle
def test_linestring_offset_rot_structured_grid(rtree):
    sgr = get_rect_grid(angrot=45.0, xyoffset=10.0)
    ls = LineString([(5, 10.0 + np.sqrt(200.0)), (15, 10.0 + np.sqrt(200.0))])
    ix = GridIntersect(sgr, rtree=rtree)
    result = ix.intersect(ls, geo_dataframe=df_toggle)
    assert len(result) == 2
    # check empty result when using local model coords
    ix = GridIntersect(sgr, rtree=rtree, local=True)
    result = ix.intersect(ls, geo_dataframe=df_toggle)
    assert len(result) == 0


@requires_pkg("shapely")
@rtree_toggle
def test_polygon_offset_rot_structured_grid(rtree):
    sgr = get_rect_grid(angrot=45.0, xyoffset=10.0)
    p = Polygon(
        [
            (5, 10.0 + np.sqrt(200.0)),
            (15, 10.0 + np.sqrt(200.0)),
            (15, 10.0 + 1.5 * np.sqrt(200.0)),
            (5, 10.0 + 1.5 * np.sqrt(200.0)),
        ]
    )
    ix = GridIntersect(sgr, rtree=rtree)
    result = ix.intersect(p, geo_dataframe=df_toggle)
    assert len(result) == 3
    # check empty result when using local model coords
    ix = GridIntersect(sgr, rtree=rtree, local=True)
    result = ix.intersect(p, geo_dataframe=df_toggle)
    assert len(result) == 0


@requires_pkg("shapely")
@rtree_toggle
def test_point_offset_rot_vertex_grid(rtree):
    sgr = get_rect_vertex_grid(angrot=45.0, xyoffset=10.0)
    p = Point(10.0, 10 + np.sqrt(200.0))
    ix = GridIntersect(sgr, rtree=rtree)
    result = ix.intersect(p, geo_dataframe=df_toggle)
    assert len(result) == 1
    # check empty result when using local model coords
    ix = GridIntersect(sgr, rtree=rtree, local=True)
    result = ix.intersect(p, geo_dataframe=df_toggle)
    assert len(result) == 0


@requires_pkg("shapely")
@rtree_toggle
def test_linestring_offset_rot_vertex_grid(rtree):
    sgr = get_rect_vertex_grid(angrot=45.0, xyoffset=10.0)
    ls = LineString([(5, 10.0 + np.sqrt(200.0)), (15, 10.0 + np.sqrt(200.0))])
    ix = GridIntersect(sgr, rtree=rtree)
    result = ix.intersect(ls, geo_dataframe=df_toggle)
    assert len(result) == 2
    # check empty result when using local model coords
    ix = GridIntersect(sgr, rtree=rtree, local=True)
    result = ix.intersect(ls, geo_dataframe=df_toggle)
    assert len(result) == 0


@requires_pkg("shapely")
@rtree_toggle
def test_polygon_offset_rot_vertex_grid(rtree):
    sgr = get_rect_vertex_grid(angrot=45.0, xyoffset=10.0)
    p = Polygon(
        [
            (5, 10.0 + np.sqrt(200.0)),
            (15, 10.0 + np.sqrt(200.0)),
            (15, 10.0 + 1.5 * np.sqrt(200.0)),
            (5, 10.0 + 1.5 * np.sqrt(200.0)),
        ]
    )
    ix = GridIntersect(sgr, rtree=rtree)
    result = ix.intersect(p, geo_dataframe=df_toggle)
    assert len(result) == 3
    # check empty result when using local model coords
    ix = GridIntersect(sgr, rtree=rtree, local=True)
    result = ix.intersect(p, geo_dataframe=df_toggle)
    assert len(result) == 0


# %% array-based inputs - structured grid points


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_single_point_array_inside(rtree):
    """Single point in array inside, returns single intersection."""
    gr = get_rect_grid()
    ix = GridIntersect(gr, rtree=rtree)
    pts = points([1.0], [1.0])
    result = ix.intersects(pts, dataframe=df_toggle)
    assert len(result) == 1
    assert result.cellids[0] == (1, 0)


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_single_point_array_outside(rtree):
    """Single point in array outside, returns empty result."""
    gr = get_rect_grid()
    ix = GridIntersect(gr, rtree=rtree)
    pts = points([25.0], [25.0])
    result = ix.intersects(pts, dataframe=df_toggle)
    assert len(result) == 0


@requires_pkg("shapely")
def test_rect_grid_single_point_array_outside_points_to_cellids():
    """Single point in array outside in points_to_cellids, returns single nan result."""
    gr = get_rect_grid()
    ix = GridIntersect(gr)
    pts = points([25.0], [25.0])
    result = ix.points_to_cellids(pts)
    assert len(result) == 1
    assert pd.isna(result.cellids[0])


@requires_pkg("shapely")
def test_rect_grid_single_point_array_on_boundary_points_to_cellids():
    """Single point in array on boundary, returns single result."""
    gr = get_rect_grid()
    ix = GridIntersect(gr)
    pts = points([10.0], [20.0])
    result = ix.points_to_cellids(pts)
    assert len(result) == 1
    assert result.cellids[0] == (0, 0)


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_single_point_array_on_boundary(rtree):
    """Single point in array on boundary, returns multiple results."""
    gr = get_rect_grid()
    ix = GridIntersect(gr, rtree=rtree)
    pts = points([10.0], [20.0])
    result = ix.intersects(pts, dataframe=df_toggle)
    assert len(result) == 2
    assert result.cellids[0] == (0, 0)
    assert result.cellids[1] == (0, 1)


@requires_pkg("shapely")
def test_rect_grid_multiple_points_array_in_one_cell():
    """Multiple points in array in one cell, returns results per point."""
    gr = get_rect_grid()
    ix = GridIntersect(gr)
    pts = points([1.0, 5.0], [2.0, 5.0])
    result = ix.intersects(pts, dataframe=df_toggle)
    assert len(result) == 2
    assert result.cellids[0] == (1, 0)
    assert result.cellids[1] == (1, 0)


@requires_pkg("shapely")
def test_rect_grid_multiple_points_array_in_multiple_cells():
    """Multiple points in array in multiple cells, returns results per point."""
    gr = get_rect_grid()
    ix = GridIntersect(gr)
    pts = points([1.0, 15.0], [2.0, 15.0])
    result = ix.intersects(pts, dataframe=df_toggle)
    assert len(result) == 2
    assert result.cellids[0] == (1, 0)
    assert result.cellids[1] == (0, 1)


@requires_pkg("shapely")
def test_rect_grid_multiple_points_array_inside_and_outside():
    """Multiple points in array inside and outside, returns one result."""
    gr = get_rect_grid()
    ix = GridIntersect(gr)
    pts = points([1.0, 25.0], [2.0, 25.0])
    result = ix.intersects(pts, dataframe=df_toggle)
    assert len(result) == 1
    assert result.cellids[0] == (1, 0)


@requires_pkg("shapely")
def test_rect_grid_multiple_points_array_inside_and_outside_points_to_cellids():
    """Multiple points in array inside/outside, returns one result and one nan."""
    gr = get_rect_grid()
    ix = GridIntersect(gr)
    pts = points([1.0, 25.0], [2.0, 25.0])
    result = ix.points_to_cellids(pts)
    assert len(result) == 2
    assert result.cellids[0] == (1, 0)
    assert pd.isna(result.cellids[1])


@requires_pkg("shapely")
def test_rect_grid_multiple_points_array_with_z_points_to_cellids():
    gr = get_rect_grid(
        top=np.ones(4).reshape((2, 2)), botm=np.zeros(4).reshape((1, 2, 2))
    )
    ix = GridIntersect(gr)
    pts = points([1.0, 25.0], [2.0, 25.0], [10.0, 0.5])
    result = ix.points_to_cellids(pts, handle_z=False)
    assert result.cellids[0] == (1, 0)
    assert pd.isna(result.cellids[1])
    result = ix.points_to_cellids(pts, handle_z=True)
    assert pd.isna(result.layer[0])
    assert pd.isna(result.cellids[1])


# %% array-based input - structured grid linestrings


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_single_linestring_array_in_one_cell(rtree):
    """Single linestring in array in 1 cell, returns single intersection."""
    gr = get_rect_grid()
    ix = GridIntersect(gr, rtree=rtree)
    ls = linestrings([[(5.0, 5.0), (7.5, 5.0)]])
    result = ix.intersects(ls, dataframe=df_toggle)
    assert len(result) == 1
    assert result.cellids[0] == (1, 0)


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_single_linestring_array_in_two_cells(rtree):
    """Single linestring in array in 2 cells, returns multiple intersections."""
    gr = get_rect_grid()
    ix = GridIntersect(gr, rtree=rtree)
    ls = linestrings([[(5.0, 5.0), (15.0, 5.0)]])
    result = ix.intersects(ls, dataframe=df_toggle)
    assert len(result) == 2
    assert result.cellids[0] == (1, 0)
    assert result.cellids[1] == (1, 1)


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_single_linestring_array_outside(rtree):
    """Single linestring in array outside, returns empty result."""
    gr = get_rect_grid()
    ix = GridIntersect(gr, rtree=rtree)
    ls = linestrings([[(25.0, 5.0), (35.0, 5.0)]])
    result = ix.intersects(ls, dataframe=df_toggle)
    assert len(result) == 0


@requires_pkg("shapely")
def test_rect_grid_multiple_linestring_array_in_multiple_cells():
    """Multiple linestrings in array; returns results per linestring,
    multiple cellids per linestring."""
    gr = get_rect_grid()
    ix = GridIntersect(gr)
    ls = linestrings([[(5.0, 5.0), (15.0, 5.0)], [(5.0, 15.0), (15.0, 15.0)]])
    result = ix.intersects(ls, dataframe=df_toggle)
    assert len(result) == 4
    assert result.cellids[0] == (1, 0)
    assert result.cellids[1] == (1, 1)
    assert result.cellids[2] == (0, 0)
    assert result.cellids[3] == (0, 1)


@requires_pkg("shapely")
def test_rect_grid_multiple_linestring_array_inside_outside():
    """Multiple linestrings in array inside/outside; returns results per linestring,
    multiple cellids per linestring."""
    gr = get_rect_grid()
    ix = GridIntersect(gr)
    ls = linestrings([[(5.0, 5.0), (15.0, 5.0)], [(25.0, 15.0), (35.0, 15.0)]])
    result = ix.intersects(ls, dataframe=df_toggle)
    assert len(result) == 2
    assert (result.shp_id == 0).all()
    assert result.cellids[0] == (1, 0)
    assert result.cellids[1] == (1, 1)


# %% array-based input - structured grid polygons


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_single_polygon_array_in_one_cell(rtree):
    """Single polygon in array inside, returns single intersection."""
    gr = get_rect_grid()
    ix = GridIntersect(gr, rtree=rtree)
    p = polygons([[(2.5, 5.0), (7.5, 5.0), (7.5, 7.5), (2.5, 7.5)]])
    result = ix.intersects(p, dataframe=df_toggle)
    assert len(result) == 1
    assert result.cellids[0] == (1, 0)


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_single_polygon_array_in_two_cells(rtree):
    """Single polygon in array in 2 cells, returns multiple intersections."""
    gr = get_rect_grid()
    ix = GridIntersect(gr, rtree=rtree)
    p = polygons([[(2.5, 5.0), (15, 5.0), (15, 7.5), (2.5, 7.5)]])
    result = ix.intersects(p, dataframe=df_toggle)
    assert len(result) == 2
    assert result.cellids[0] == (1, 0)
    assert result.cellids[1] == (1, 1)


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_single_polygon_array_outside(rtree):
    """Single polygon in array outside, returns empty result."""
    gr = get_rect_grid()
    ix = GridIntersect(gr, rtree=rtree)
    p = polygons([[(25, 5.0), (75, 5.0), (75, 7.5), (25, 7.5)]])
    result = ix.intersects(p, dataframe=df_toggle)
    assert len(result) == 0


@requires_pkg("shapely")
def test_rect_grid_multiple_polygon_array_single_result_per_polygon():
    """Multiple polygons in array; returns results per polygon,
    single result per polygon."""
    gr = get_rect_grid()
    ix = GridIntersect(gr)
    p = polygons(
        [
            [(2.5, 5.0), (7.5, 5.0), (7.5, 7.5), (2.5, 7.5)],
            [(2.5, 15.0), (7.5, 15.0), (7.5, 17.5), (2.5, 17.5)],
        ]
    )
    result = ix.intersects(p, dataframe=df_toggle)
    assert len(result) == 2
    assert result.cellids[0] == (1, 0)
    assert result.cellids[1] == (0, 0)


@requires_pkg("shapely")
def test_rect_grid_multiple_polygon_array_multiple_results_per_polygon():
    """Multiple polygons in array; returns results per polygon,
    multiple results per polygon."""
    gr = get_rect_grid()
    ix = GridIntersect(gr)
    p = polygons(
        [
            [(2.5, 5.0), (17.5, 5.0), (17.5, 7.5), (2.5, 7.5)],
            [(2.5, 15.0), (17.5, 15.0), (17.5, 17.5), (2.5, 17.5)],
        ]
    )
    result = ix.intersects(p, dataframe=df_toggle)
    assert len(result) == 4
    assert result.cellids[0] == (1, 0)
    assert result.cellids[1] == (1, 1)
    assert result.cellids[2] == (0, 0)
    assert result.cellids[3] == (0, 1)


@requires_pkg("shapely")
def test_rect_grid_multiple_polygon_array_inside_outside():
    """Multiple polygons in array inside/outside; returns results per polygon,
    multiple cellids per polygon."""
    gr = get_rect_grid()
    ix = GridIntersect(gr)
    p = polygons(
        [
            [(2.5, 5.0), (7.5, 5.0), (7.5, 7.5), (2.5, 7.5)],
            [(25, 15.0), (75, 15.0), (75, 17.5), (25, 17.5)],
        ]
    )
    result = ix.intersects(p, dataframe=df_toggle)
    assert len(result) == 1
    assert result.cellids[0] == (1, 0)


# %% array-based input - structured grid intersection method


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_intersect_single_point_array(rtree):
    """Single point in array ok."""
    gr = get_rect_grid()
    ix = GridIntersect(gr, rtree=rtree)
    pts = points([10], [20])
    result = ix.intersect(pts, geo_dataframe=df_toggle)
    assert result.cellids[0] == (0, 0)


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_intersect_multiple_points_array(rtree):
    """Multiple points in array raises error."""
    gr = get_rect_grid()
    ix = GridIntersect(gr, rtree=rtree)
    pts = points([10, 1], [20, 5])
    with pytest.raises(
        ValueError, match="intersect\(\) only accepts arrays containing one"
    ):
        ix.intersect(pts, geo_dataframe=df_toggle)


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_intersect_single_linestring_array(rtree):
    """Single linestring in array ok."""
    gr = get_rect_grid()
    ix = GridIntersect(gr, rtree=rtree)
    ls = linestrings([[(5.0, 5.0), (15.0, 5.0)]])
    result = ix.intersect(ls, geo_dataframe=df_toggle)
    assert len(result) == 2
    assert result.cellids[0] == (1, 0)
    assert result.cellids[1] == (1, 1)
    assert (result.lengths == 5.0).all()


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_intersect_multiple_linestring_array(rtree):
    """Multiple linestrings in array raises error."""
    gr = get_rect_grid()
    ix = GridIntersect(gr, rtree=rtree)
    ls = linestrings(
        [
            [(5.0, 5.0), (15.0, 5.0)],
            [(5.0, 15.0), (15.0, 15.0)],
        ]
    )
    with pytest.raises(
        ValueError, match="intersect\(\) only accepts arrays containing one"
    ):
        ix.intersect(ls, geo_dataframe=df_toggle)


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_intersect_single_polygon_array(rtree):
    """Single polygon in array ok."""
    gr = get_rect_grid()
    ix = GridIntersect(gr, rtree=rtree)
    p = polygons(
        [
            [(2.5, 5.0), (17.5, 5.0), (17.5, 7.5), (2.5, 7.5)],
        ]
    )
    result = ix.intersect(p, geo_dataframe=df_toggle)
    assert len(result) == 2
    assert result.cellids[0] == (1, 0)
    assert result.cellids[1] == (1, 1)
    assert (result.areas == 18.75).all()


@requires_pkg("shapely")
@rtree_toggle
def test_rect_grid_intersect_multiple_polygon_array(rtree):
    """Multiple polygons in array input raises error."""
    gr = get_rect_grid()
    ix = GridIntersect(gr, rtree=rtree)
    p = polygons(
        [
            [(2.5, 5.0), (17.5, 5.0), (17.5, 7.5), (2.5, 7.5)],
            [(2.5, 15.0), (17.5, 15.0), (17.5, 17.5), (2.5, 17.5)],
        ]
    )
    with pytest.raises(
        ValueError, match="intersect\(\) only accepts arrays containing one"
    ):
        ix.intersect(p, geo_dataframe=df_toggle)


# %% vertex grid points


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_single_point_array_inside(rtree):
    """Single point in array inside returns single intersection."""
    gr = get_tri_grid()
    ix = GridIntersect(gr, rtree=rtree)
    pts = points([9.0], [1.0])
    result = ix.intersects(pts, dataframe=df_toggle)
    assert len(result) == 1
    assert result.cellids[0] == 4


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_single_point_array_outside(rtree):
    """Single point in array outside, returns empty result."""
    gr = get_tri_grid()
    ix = GridIntersect(gr, rtree=rtree)
    pts = points([25.0], [25.0])
    result = ix.intersects(pts, dataframe=df_toggle)
    assert len(result) == 0


@requires_pkg("shapely")
def test_tri_grid_single_point_array_outside_points_to_cellids():
    """Single point in array outside + return_all, returns single nan result."""
    gr = get_tri_grid()
    ix = GridIntersect(gr)
    pts = points([25.0], [25.0])
    result = ix.points_to_cellids(pts)
    assert len(result) == 1
    assert pd.isna(result.cellids[0])


@requires_pkg("shapely")
def test_tri_grid_single_point_array_on_boundary_points_to_cellids():
    """Single point in array on boundary, returns single intersection."""
    gr = get_tri_grid()
    ix = GridIntersect(gr)
    pts = points([9.0], [1.0])
    result = ix.points_to_cellids(pts)
    assert len(result) == 1
    assert result.cellids[0] == 4


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_single_point_array_on_boundary(rtree):
    """Single point in array on boundary, returns multiple intersections."""
    gr = get_tri_grid()
    ix = GridIntersect(gr, rtree=rtree)
    pts = points([5.0], [5.0])
    result = ix.intersects(pts, dataframe=df_toggle)
    assert len(result) == 2
    assert result.cellids[0] == 1
    assert result.cellids[1] == 4


@requires_pkg("shapely")
def test_tri_grid_multiple_points_array_in_one_cell():
    """Multiple points in array in one cell, returns results per point."""
    gr = get_tri_grid()
    ix = GridIntersect(gr)
    pts = points([9.0, 9.0], [1.0, 8.0])
    result = ix.intersects(pts, dataframe=df_toggle)
    assert len(result) == 2
    assert (result.cellids == 4).all()


@requires_pkg("shapely")
def test_tri_grid_multiple_points_array_in_multiple_cells():
    """Multiple points in array in multiple cells, returns results per point."""
    gr = get_tri_grid()
    ix = GridIntersect(gr)
    pts = points([15.0, 9.0], [3.0, 3.0])
    result = ix.intersects(pts, dataframe=df_toggle)
    assert len(result) == 2
    assert result.cellids[0] == 5
    assert result.cellids[1] == 4


@requires_pkg("shapely")
def test_tri_grid_multiple_points_array_inside_and_outside_points_to_cellids():
    """Multiple points in array inside and outside, returns one result and one nan."""
    gr = get_tri_grid()
    ix = GridIntersect(gr)
    pts = points([5.0, 25.0], [3.0, 25.0])
    result = ix.points_to_cellids(pts)
    assert len(result) == 2
    assert result.cellids[0] == 4
    assert pd.isna(result.cellids[1])


@requires_pkg("shapely")
def test_tri_grid_multiple_points_array_with_z_points_to_cellids():
    gr = get_rect_grid(
        top=np.ones(4).reshape((2, 2)), botm=np.zeros(4).reshape((1, 2, 2))
    )
    ix = GridIntersect(gr)
    pts = points([1.0, 25.0], [2.0, 25.0], [0.5, 10.0])
    result = ix.points_to_cellids(pts, handle_z=True)
    assert result.layer[0] == 0.0
    assert pd.isna(result.cellids[1])


@requires_pkg("shapely")
def test_tri_grid_multiple_points_array_inside_and_outside():
    """Multiple points in array inside and outside, returns 1 result."""
    gr = get_tri_grid()
    ix = GridIntersect(gr)
    pts = points([5.0, 25.0], [3.0, 25.0])
    result = ix.intersects(pts, dataframe=df_toggle)
    assert len(result) == 1
    assert result.cellids[0] == 4


# %% vertex grid linestrings


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_single_linestring_array_in_one_cell(rtree):
    """Single linestring in array in 1 cell, returns single intersection."""
    gr = get_tri_grid()
    ix = GridIntersect(gr, rtree=rtree)
    ls = linestrings([[(2.0, 1.0), (7.5, 1.0)]])
    result = ix.intersects(ls, dataframe=df_toggle)
    assert len(result) == 1
    assert result.cellids[0] == 4


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_single_linestring_array_in_two_cells(rtree):
    """Single linestring in array in 2 cells, returns multiple intersections."""
    gr = get_tri_grid()
    ix = GridIntersect(gr, rtree=rtree)
    ls = linestrings([[(2.0, 1.0), (15.0, 1.0)]])
    result = ix.intersects(ls, dataframe=df_toggle)
    assert len(result) == 2
    assert result.cellids[0] == 4
    assert result.cellids[1] == 5


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_single_linestring_array_outside(rtree):
    """Single linestring in array outside, returns empty result."""
    gr = get_tri_grid()
    ix = GridIntersect(gr, rtree=rtree)
    ls = linestrings([[(25.0, 5.0), (35.0, 5.0)]])
    result = ix.intersects(ls, dataframe=df_toggle)
    assert len(result) == 0


@requires_pkg("shapely")
def test_tri_grid_multiple_linestring_array_in_multiple_cells():
    """Multiple linestrings in array, returns results per linestring,
    multiple cellids per linestring."""
    gr = get_tri_grid()
    ix = GridIntersect(gr)
    ls = linestrings([[(2.0, 1.0), (15.0, 1.0)], [(2.0, 19.0), (15.0, 19.0)]])
    result = ix.intersects(ls, dataframe=df_toggle)
    assert len(result) == 4
    assert result.cellids[0] == 4
    assert result.cellids[1] == 5
    assert result.cellids[2] == 2
    assert result.cellids[3] == 7


@requires_pkg("shapely")
def test_tri_grid_multiple_linestring_array_inside_outside():
    """Multiple linestrings in array inside/outside, returns multiple cellids per
    linestring."""
    gr = get_tri_grid()
    ix = GridIntersect(gr)
    ls = linestrings([[(2.0, 1.0), (15.0, 1.0)], [(25.0, 15.0), (35.0, 15.0)]])
    result = ix.intersects(ls, dataframe=df_toggle)
    assert len(result) == 2
    assert (result.shp_id == 0).all()
    assert result.cellids[0] == 4
    assert result.cellids[1] == 5


# %% vertex grid polygons


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_single_polygon_array_in_one_cell(rtree):
    """Single polygon in array inside, returns single intersection."""
    gr = get_tri_grid()
    ix = GridIntersect(gr, rtree=rtree)
    p = polygons([[(2.0, 1.0), (9.0, 1.0), (9.0, 7.0), (2.0, 1.0)]])
    result = ix.intersects(p, dataframe=df_toggle)
    assert len(result) == 1
    assert result.cellids[0] == 4


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_single_polygon_array_in_two_cells(rtree):
    """Single polygon in array in 2 cells, returns multiple intersections."""
    gr = get_tri_grid()
    ix = GridIntersect(gr, rtree=rtree)
    p = polygons([[(5.0, 1.0), (15.0, 1.0), (15.0, 2.0), (5.0, 2.0)]])
    result = ix.intersects(p, dataframe=df_toggle)
    assert len(result) == 2
    assert result.cellids[0] == 4
    assert result.cellids[1] == 5


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_single_polygon_array_outside(rtree):
    """Single polygon in array outside, returns empty result."""
    gr = get_tri_grid()
    ix = GridIntersect(gr, rtree=rtree)
    p = polygons([[(25, 5.0), (75, 5.0), (75, 7.5), (25, 7.5)]])
    result = ix.intersects(p, dataframe=df_toggle)
    assert len(result) == 0


@requires_pkg("shapely")
def test_tri_grid_multiple_polygon_array_single_result_per_polygon():
    """Multiple polygons in array, returns results per polygon,
    single result per polygon."""
    gr = get_tri_grid()
    ix = GridIntersect(gr)
    p = polygons(
        [
            [(2.0, 1.0), (9.0, 1.0), (9.0, 7.0), (2.0, 1.0)],
            [(2.0, 19.0), (9.0, 19.0), (9.0, 17.0), (2.0, 19.0)],
        ]
    )
    result = ix.intersects(p, dataframe=df_toggle)
    assert len(result) == 2
    assert result.cellids[0] == 4
    assert result.cellids[1] == 2


@requires_pkg("shapely")
def test_tri_grid_multiple_polygon_array_multiple_results_per_polygon():
    """Multiple polygons in array, returns results per polygon,
    multiple results per polygon."""
    gr = get_tri_grid()
    ix = GridIntersect(gr)
    p = polygons(
        [
            [(5.0, 1.0), (15.0, 1.0), (15.0, 2.0), (5.0, 2.0)],
            [(5.0, 19.0), (15.0, 19.0), (15.0, 18.0), (5.0, 18.0)],
        ]
    )
    result = ix.intersects(p, dataframe=df_toggle)
    assert len(result) == 4
    assert result.cellids[0] == 4
    assert result.cellids[1] == 5
    assert result.cellids[2] == 2
    assert result.cellids[3] == 7


@requires_pkg("shapely")
def test_tri_grid_multiple_polygon_array_inside_outside():
    """Multiple polygons in array inside/outside, returns results per polygon,
    multiple cellids per polygon."""
    gr = get_tri_grid()
    ix = GridIntersect(gr)
    p = polygons(
        [
            [(2.0, 1.0), (9.0, 1.0), (9.0, 7.0), (2.0, 1.0)],
            [(25, 15.0), (75, 15.0), (75, 17.5), (25, 17.5)],
        ]
    )
    result = ix.intersects(p, dataframe=df_toggle)
    assert len(result) == 1
    assert result.cellids[0] == 4


# %% vertex grid intersection method


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_intersect_single_point_array(rtree):
    """Single point in array ok."""
    gr = get_tri_grid()
    ix = GridIntersect(gr, rtree=rtree)
    pts = points([10], [20])
    result = ix.intersect(pts, return_all_intersections=True, geo_dataframe=df_toggle)
    assert len(result.cellids) == 2
    assert result.cellids[0] == 2
    assert result.cellids[1] == 7
    result = ix.intersect(pts, return_all_intersections=False, geo_dataframe=df_toggle)
    assert len(result.cellids) == 1
    assert result.cellids[0] == 2


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_intersect_multiple_points_array(rtree):
    """Multiple points in array raises error."""
    gr = get_tri_grid()
    ix = GridIntersect(gr, rtree=rtree)
    pts = points([10, 5], [20, 5])
    with pytest.raises(
        ValueError, match="intersect\(\) only accepts arrays containing one"
    ):
        ix.intersect(pts, geo_dataframe=df_toggle)


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_intersect_single_linestring_array(rtree):
    """Single linestring in array ok."""
    gr = get_tri_grid()
    ix = GridIntersect(gr, rtree=rtree)
    ls = linestrings([[(5.0, 5.0), (15.0, 5.0)]])
    result = ix.intersect(ls, geo_dataframe=df_toggle)
    assert len(result) == 2
    assert result.cellids[0] == 4
    assert result.cellids[1] == 5
    assert (result.lengths == 5.0).all()


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_intersect_multiple_linestring_array(rtree):
    """Multiple linestrings in array raises error."""
    gr = get_tri_grid()
    ix = GridIntersect(gr, rtree=rtree)
    ls = linestrings(
        [
            [(5.0, 5.0), (15.0, 5.0)],
            [(5.0, 15.0), (15.0, 15.0)],
        ]
    )
    with pytest.raises(
        ValueError, match="intersect\(\) only accepts arrays containing one"
    ):
        ix.intersect(ls, geo_dataframe=df_toggle)


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_intersect_single_polygon_array(rtree):
    """Single polygon in array ok."""
    gr = get_tri_grid()
    ix = GridIntersect(gr, rtree=rtree)
    p = polygons(
        [
            [(2.5, 5.0), (17.5, 5.0), (17.5, 7.5), (2.5, 7.5)],
        ]
    )
    result = ix.intersect(p, geo_dataframe=df_toggle)
    assert len(result) == 4
    assert result.cellids[0] == 1
    assert result.cellids[1] == 4
    assert result.cellids[2] == 5
    assert result.cellids[3] == 6
    assert (result.areas == 9.375).all()


@requires_pkg("shapely")
@rtree_toggle
def test_tri_grid_intersect_multiple_polygon_array(rtree):
    """Multi-array input raises error."""
    gr = get_tri_grid()
    ix = GridIntersect(gr, rtree=rtree)
    p = polygons(
        [
            [(2.5, 5.0), (17.5, 5.0), (17.5, 7.5), (2.5, 7.5)],
            [(2.5, 15.0), (17.5, 15.0), (17.5, 17.5), (2.5, 17.5)],
        ]
    )
    with pytest.raises(
        ValueError, match="intersect\(\) only accepts arrays containing one"
    ):
        ix.intersect(p, geo_dataframe=df_toggle)


def test_rtree_false_raises_in_points_to_cellids():
    """rtree=False raises error in points_to_cellids."""
    gr = get_rect_grid()
    ix = GridIntersect(gr, rtree=False)
    pts = points([1.0], [1.0])
    with pytest.raises(
        ValueError,
        match="points_to_cellids\(\) requires rtree=True when",
    ):
        ix.points_to_cellids(pts)


def test_rtree_false_raises_with_arrays_in_intersects():
    """rtree=False raises error in points_to_cellids."""
    gr = get_rect_grid()
    ix = GridIntersect(gr, rtree=False)
    pts = points([1.0, 10.0], [1.0, 10.0])
    with pytest.raises(
        ValueError,
        match="points_to_cellids\(\) requires rtree=True when initializing",
    ):
        ix.points_to_cellids(pts)
