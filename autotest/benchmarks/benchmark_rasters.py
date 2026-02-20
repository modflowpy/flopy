import numpy as np
import pytest
from modflow_devtools.misc import has_pkg

pytest.importorskip("rasterio")
pytest.importorskip("affine")

from autotest.conftest import get_example_data_path
from flopy.discretization.structuredgrid import StructuredGrid
from flopy.utils.geometry import Polygon
from flopy.utils.rasters import Raster


@pytest.fixture(scope="module")
def raster_path(example_data_path):
    return example_data_path / "options" / "dem" / "dem.img"


@pytest.fixture(scope="module")
def raster(raster_path):
    return Raster.load(raster_path)


def origin_and_extent(raster) -> tuple[float, float, float, float]:
    x0, x1, y0, y1 = raster.bounds
    # central region
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    extent_x, extent_y = (x1 - x0) * 0.6, (y1 - y0) * 0.6
    grid_x0, grid_y0 = cx - extent_x / 2, cy - extent_y / 2
    return grid_x0, grid_y0, extent_x, extent_y


def make_grid(raster, nrow, ncol) -> StructuredGrid:
    grid_x0, grid_y0, extent_x, extent_y = origin_and_extent(raster)
    delr = np.full(ncol, extent_x / ncol)
    delc = np.full(nrow, extent_y / nrow)
    return StructuredGrid(
        delc=delc, delr=delr, xoff=grid_x0, yoff=grid_y0, crs=raster.crs
    )


@pytest.mark.benchmark
def test_raster_load(benchmark, raster_path):
    benchmark(lambda: Raster.load(raster_path))


RASTER_PATH = get_example_data_path() / "options" / "dem" / "dem.img"
GRIDS = [
    make_grid(Raster.load(RASTER_PATH), 10, 10),
    make_grid(Raster.load(RASTER_PATH), 50, 50),
    make_grid(Raster.load(RASTER_PATH), 200, 200),
]


@pytest.mark.slow
@pytest.mark.benchmark
@pytest.mark.parametrize("grid", GRIDS, ids=["small", "medium", "large"])
@pytest.mark.parametrize(
    "method", ["linear", "nearest", "cubic", "mean", "median", "min", "max"]
)
def test_raster_resample(benchmark, raster, grid, method):
    benchmark(lambda: raster.resample_to_grid(grid, band=1, method=method))


@pytest.mark.slow
@pytest.mark.benchmark
@pytest.mark.skipif(not has_pkg("pyproj"), reason="requires pyproj")
def test_raster_to_crs_transform(benchmark, raster):
    benchmark(lambda: raster.to_crs(epsg=4326))


def small_poly(raster):
    # small central polygon, ~20% of extent
    x0, x1, y0, y1 = raster.bounds
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    dx, dy = (x1 - x0) * 0.1, (y1 - y0) * 0.1
    return Polygon(
        [(cx - dx, cy - dy), (cx + dx, cy - dy), (cx + dx, cy + dy), (cx - dx, cy + dy)]
    )


def medium_poly(raster):
    # ~50% of extent
    x0, x1, y0, y1 = raster.bounds
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    dx, dy = (x1 - x0) * 0.25, (y1 - y0) * 0.25
    return Polygon(
        [(cx - dx, cy - dy), (cx + dx, cy - dy), (cx + dx, cy + dy), (cx - dx, cy + dy)]
    )


def large_poly(raster):
    # 80% of extent
    x0, x1, y0, y1 = raster.bounds
    dx, dy = (x1 - x0) * 0.1, (y1 - y0) * 0.1
    return Polygon(
        [(x0 + dx, y0 + dy), (x1 - dx, y0 + dy), (x1 - dx, y1 - dy), (x0 + dx, y1 - dy)]
    )


POLYGONS = [
    small_poly(Raster.load(RASTER_PATH)),
    medium_poly(Raster.load(RASTER_PATH)),
    large_poly(Raster.load(RASTER_PATH)),
]


@pytest.mark.benchmark
@pytest.mark.parametrize("poly", POLYGONS, ids=["small", "medium", "large"])
def test_raster_crop(benchmark, raster, poly):
    benchmark(lambda: raster.crop(poly))


@pytest.mark.benchmark
@pytest.mark.parametrize("poly", POLYGONS, ids=["small", "medium", "large"])
def test_raster_sample(benchmark, raster, poly):
    benchmark(lambda: raster.sample_polygon(poly, band=1))


@pytest.mark.benchmark
def test_raster_sample_point(benchmark, raster):
    x0, x1, y0, y1 = raster.bounds
    x, y = (x0 + x1) / 2, (y0 + y1) / 2  # center point
    benchmark(lambda: raster.sample_point(x, y, band=1))


@pytest.mark.benchmark
@pytest.mark.parametrize("masked", [True, False], ids=["masked", "unmasked"])
def test_raster_get_array_masked(benchmark, raster, masked):
    benchmark(lambda: raster.get_array(band=1, masked=masked))


@pytest.mark.benchmark
def test_raster_write(benchmark, raster, function_tmpdir):
    output_path = function_tmpdir / "output_raster.tif"
    benchmark(lambda: raster.write(str(output_path)))
