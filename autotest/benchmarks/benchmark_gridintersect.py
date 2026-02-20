"""
Benchmarks for spatial intersection operations including:

1. GridIntersect class (flopy.utils.gridintersect.GridIntersect):
   - Initialization (with/without STR-tree spatial index)
   - Point, LineString, and Polygon intersections
   - Spatial query performance

2. Grid.intersect() method (direct coordinate-based intersection):
   - Single point lookup
   - Batch point operations
   - 3D coordinate intersection
"""

import numpy as np
import pytest

from autotest.test_grid_cases import GridCases
from flopy.utils.geometry import LineString, Point, Polygon
from flopy.utils.gridintersect import GridIntersect

STRUCTURED_GRIDS = {
    "small": GridCases.structured_small(),
    "medium": GridCases.structured_medium(),
    "large": GridCases.structured_large(),
}


# GridIntersect class benchmarks


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.parametrize("grid", STRUCTURED_GRIDS.values(), ids=STRUCTURED_GRIDS.keys())
@pytest.mark.parametrize("rtree", [True, False], ids=["rtree", "no_rtree"])
def test_init(benchmark, grid, rtree):
    benchmark(lambda: GridIntersect(grid, rtree=rtree))


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.parametrize("grid", STRUCTURED_GRIDS.values(), ids=STRUCTURED_GRIDS.keys())
@pytest.mark.parametrize("rtree", [True, False], ids=["rtree", "no_rtree"])
def test_intersect_point(benchmark, grid, rtree):
    gi = GridIntersect(grid, rtree=rtree)
    xmin, xmax, ymin, ymax = grid.extent
    point = Point((xmin + xmax) / 2, (ymin + ymax) / 2)
    benchmark(lambda: gi.intersect(point, "point"))


def make_line(grid, line_type) -> LineString:
    xmin, xmax, ymin, ymax = grid.extent
    if line_type == "diagonal":
        return LineString([(xmin, ymin), (xmax, ymax)])
    elif line_type == "horizontal":
        y_mid = (ymin + ymax) / 2
        return LineString([(xmin, y_mid), (xmax, y_mid)])
    elif line_type == "complex":
        x = np.linspace(xmin, xmax, 20)
        y_mid = (ymin + ymax) / 2
        y_range = (ymax - ymin) * 0.2
        y = y_mid + y_range * np.sin(x / (xmax - xmin) * 10)
        coords = list(zip(x, y))
        return LineString(coords)


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.parametrize("grid", STRUCTURED_GRIDS.values(), ids=STRUCTURED_GRIDS.keys())
@pytest.mark.parametrize("line", ["diagonal", "horizontal", "complex"])
@pytest.mark.parametrize("rtree", [True, False], ids=["rtree", "no_rtree"])
def test_intersect_linestring_diagonal(benchmark, grid, line, rtree):
    gi = GridIntersect(grid, rtree=rtree)
    line = make_line(grid, line)
    benchmark(lambda: gi.intersect(line, "linestring"))


def make_poly(grid, poly_type) -> Polygon:
    xmin, xmax, ymin, ymax = grid.extent
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    x_range = xmax - xmin
    y_range = ymax - ymin

    if poly_type == "small":
        # 10% of grid extent centered
        dx = x_range * 0.05
        dy = y_range * 0.05
        coords = [
            (x_center - dx, y_center - dy),
            (x_center + dx, y_center - dy),
            (x_center + dx, y_center + dy),
            (x_center - dx, y_center + dy),
        ]
    elif poly_type == "medium":
        # 50% of grid extent centered
        dx = x_range * 0.25
        dy = y_range * 0.25
        coords = [
            (x_center - dx, y_center - dy),
            (x_center + dx, y_center - dy),
            (x_center + dx, y_center + dy),
            (x_center - dx, y_center + dy),
        ]
    elif poly_type == "large":
        # 90% of grid extent centered
        dx = x_range * 0.45
        dy = y_range * 0.45
        coords = [
            (x_center - dx, y_center - dy),
            (x_center + dx, y_center - dy),
            (x_center + dx, y_center + dy),
            (x_center - dx, y_center + dy),
        ]
    elif poly_type == "irregular":
        # Irregular star-like polygon
        n_points = 20
        theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        r_base = min(x_range, y_range) * 0.3
        r_var = r_base * 0.33
        r = r_base + r_var * np.sin(5 * theta)
        x = x_center + r * np.cos(theta)
        y = y_center + r * np.sin(theta)
        coords = list(zip(x, y))

    return Polygon(coords)


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.parametrize("grid", STRUCTURED_GRIDS.values(), ids=STRUCTURED_GRIDS.keys())
@pytest.mark.parametrize("poly", ["small", "medium", "large", "irregular"])
@pytest.mark.parametrize("rtree", [True, False], ids=["rtree", "no_rtree"])
def test_intersect_polygon(benchmark, grid, poly, rtree):
    gi = GridIntersect(grid, rtree=rtree)
    polygon = make_poly(grid, poly)
    benchmark(lambda: gi.intersect(polygon, "polygon"))


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.parametrize("grid", STRUCTURED_GRIDS.values(), ids=STRUCTURED_GRIDS.keys())
@pytest.mark.parametrize("poly", ["small", "medium", "large", "irregular"])
@pytest.mark.parametrize("rtree", [True, False], ids=["rtree", "no_rtree"])
def test_query_grid_polygon(benchmark, grid, poly, rtree):
    gi = GridIntersect(grid, rtree=rtree)
    polygon = make_poly(grid, poly)
    benchmark(lambda: gi.query_grid(polygon))


# Grid.intersect() method benchmarks (coordinate-based intersection)


@pytest.mark.benchmark
@pytest.mark.parametrize("grid", STRUCTURED_GRIDS.values(), ids=STRUCTURED_GRIDS.keys())
def test_grid_intersect_single_point(benchmark, grid):
    xmin, xmax, ymin, ymax = grid.extent
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    benchmark(lambda: grid.intersect(x_center, y_center))


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.parametrize("grid", STRUCTURED_GRIDS.values(), ids=STRUCTURED_GRIDS.keys())
def test_grid_intersect_batch_points(benchmark, grid):
    xmin, xmax, ymin, ymax = grid.extent
    x = np.linspace(xmin + 0.1 * (xmax - xmin), xmax - 0.1 * (xmax - xmin), 100)
    y = np.linspace(ymin + 0.1 * (ymax - ymin), ymax - 0.1 * (ymax - ymin), 100)
    xx, yy = np.meshgrid(x, y)
    benchmark(lambda: grid.intersect(xx.ravel(), yy.ravel()))


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.parametrize("grid", STRUCTURED_GRIDS.values(), ids=STRUCTURED_GRIDS.keys())
def test_grid_intersect_3d(benchmark, grid):
    xmin, xmax, ymin, ymax = grid.extent
    x = np.linspace(xmin + 0.1 * (xmax - xmin), xmax - 0.1 * (xmax - xmin), 50)
    y = np.linspace(ymin + 0.1 * (ymax - ymin), ymax - 0.1 * (ymax - ymin), 50)
    xx, yy = np.meshgrid(x, y)
    zz = np.ones_like(xx) * 5.0
    benchmark(lambda: grid.intersect(xx.ravel(), yy.ravel(), zz.ravel()))
