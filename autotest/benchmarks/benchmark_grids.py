"""
Benchmarks for flopy.discretization grid operations including:
- cellid <-> node number conversions
- grid intersection operations
- grid geometry properties
"""

import numpy as np
import pytest

from autotest.test_grid_cases import GridCases

STRUCTURED_GRIDS = {
    "small": GridCases.structured_small(),
    "medium": GridCases.structured_medium(),
    "large": GridCases.structured_large(),
}


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.parametrize("grid", STRUCTURED_GRIDS.values(), ids=STRUCTURED_GRIDS.keys())
def test_structured_grid_get_lrc(benchmark, grid):
    nodes = list(range(grid.nnodes))
    benchmark(lambda: grid.get_lrc(nodes))


@pytest.mark.benchmark
@pytest.mark.parametrize("grid", STRUCTURED_GRIDS.values(), ids=STRUCTURED_GRIDS.keys())
def test_structured_grid_get_node(benchmark, grid):
    cellids = [
        (lay, row, col)
        for lay in range(grid.nlay)
        for row in range(0, grid.nrow)
        for col in range(0, grid.ncol)
    ]
    benchmark(lambda: grid.get_node(cellids))


@pytest.mark.benchmark
@pytest.mark.parametrize("grid", STRUCTURED_GRIDS.values(), ids=STRUCTURED_GRIDS.keys())
def test_structured_grid_intersect_linestring(benchmark, grid):
    # Create x, y coordinates along a diagonal line across the grid
    x = np.linspace(0, grid.ncol, 100)
    y = np.linspace(0, grid.nrow, 100)
    benchmark(lambda: grid.intersect(x, y, forgive=True))


@pytest.mark.benchmark
@pytest.mark.parametrize("grid", STRUCTURED_GRIDS.values(), ids=STRUCTURED_GRIDS.keys())
def test_structured_grid_intersect_point(benchmark, grid):
    # Use grid center point
    x = grid.ncol / 2.0
    y = grid.nrow / 2.0
    benchmark(lambda: grid.intersect(x, y))


@pytest.mark.benchmark
@pytest.mark.parametrize("grid", STRUCTURED_GRIDS.values(), ids=STRUCTURED_GRIDS.keys())
def test_structured_grid_extent(benchmark, grid):
    benchmark(lambda: grid.extent)


@pytest.mark.benchmark
@pytest.mark.parametrize("grid", STRUCTURED_GRIDS.values(), ids=STRUCTURED_GRIDS.keys())
def test_structured_grid_xyzcellcenters(benchmark, grid):
    benchmark(lambda: grid.xyzcellcenters)
