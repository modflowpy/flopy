import os

import numpy as np
import pytest
from modflow_devtools.markers import requires_pkg

import flopy
from flopy.modflow import Modflow
from flopy.utils import Raster

# %% test rasters


@requires_pkg("rasterstats", "scipy", "shapely")
def test_rasters(example_data_path):
    ws = example_data_path / "options"
    raster_name = "dem.img"

    rio = Raster.load(ws / "dem" / raster_name)

    ml = Modflow.load(
        "sagehen.nam", version="mfnwt", model_ws=os.path.join(ws, "sagehen")
    )
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

    data = rio.resample_to_grid(ml.modelgrid, band=rio.bands[0], method="nearest")
    if data.size != 5913:
        raise AssertionError
    if abs(np.min(data) - 1942.1735) > 1e-4:
        raise AssertionError
    if abs(np.max(data) - 2605.6204) > 1e-4:
        raise AssertionError

    del rio


# %% test raster sampling methods


@pytest.mark.slow
@requires_pkg("rasterstats")
def test_raster_sampling_methods(example_data_path):
    ws = example_data_path / "options"
    raster_name = "dem.img"

    rio = Raster.load(ws / "dem" / raster_name)

    ml = Modflow.load("sagehen.nam", version="mfnwt", model_ws=ws / "sagehen")
    xoff = 214110
    yoff = 4366620
    ml.modelgrid.set_coord_info(xoff, yoff)

    x0, x1, y0, y1 = rio.bounds

    x0 += 3000
    y0 += 3000
    x1 -= 3000
    y1 -= 3000
    shape = np.array([(x0, y0), (x0, y1), (x1, y1), (x1, y0), (x0, y0)])

    rio.crop(shape)

    methods = {
        "min": 2088.52343,
        "max": 2103.54882,
        "mean": 2097.05054,
        "median": 2097.36254,
        "mode": 2088.52343,
        "nearest": 2097.81079,
        "linear": 2097.81079,
        "cubic": 2097.81079,
    }

    for method, value in methods.items():
        data = rio.resample_to_grid(ml.modelgrid, band=rio.bands[0], method=method)

        print(data[30, 37])
        if np.abs(data[30, 37] - value) > 1e-05:
            raise AssertionError(f"{method} resampling returning incorrect values")


@requires_pkg("rasterio")
def test_raster_reprojection(example_data_path):
    ws = example_data_path / "options" / "dem"
    raster_name = "dem.img"

    wgs_epsg = 4326
    wgs_xmin = -120.32116799649168
    wgs_ymax = 39.46620605907534

    raster = Raster.load(ws / raster_name)

    print(raster.crs.to_epsg())
    wgs_raster = raster.to_crs(crs=f"EPSG:{wgs_epsg}")

    if not wgs_raster.crs.to_epsg() == wgs_epsg:
        raise AssertionError(f"Raster not converted to EPSG {wgs_epsg}")

    transform = wgs_raster._meta["transform"]
    if not np.isclose(transform.c, wgs_xmin) and not np.isclose(transform.f, wgs_ymax):
        raise AssertionError(f"Raster not reprojected to EPSG {wgs_epsg}")

    raster.to_crs(epsg=wgs_epsg, inplace=True)
    transform2 = raster._meta["transform"]
    for ix, val in enumerate(transform):
        if not np.isclose(val, transform2[ix]):
            raise AssertionError("In place reprojection not working")


@requires_pkg("rasterio")
def test_create_raster_from_array_modelgrid(example_data_path):
    ws = example_data_path / "options" / "dem"
    raster_name = "dem.img"

    raster = Raster.load(ws / raster_name)

    xsize = 200
    ysize = 100
    xmin, xmax, ymin, ymax = raster.bounds

    nbands = 5
    nlay = 1
    nrow = int(np.floor((ymax - ymin) / ysize))
    ncol = int(np.floor((xmax - xmin) / xsize))

    delc = np.full((nrow,), ysize)
    delr = np.full((ncol,), xsize)

    grid = flopy.discretization.StructuredGrid(
        delc=delc,
        delr=delr,
        top=np.ones((nrow, ncol)),
        botm=np.zeros((nlay, nrow, ncol)),
        idomain=np.ones((nlay, nrow, ncol), dtype=int),
        xoff=xmin,
        yoff=ymin,
        crs=raster.crs,
    )

    array = np.random.random((grid.ncpl * nbands,)) * 100
    robj = Raster.raster_from_array(array, grid)

    if nbands != len(robj.bands):
        raise AssertionError("Number of raster bands is incorrect")

    array = array.reshape((nbands, nrow, ncol))
    for band in robj.bands:
        ra = robj.get_array(band)
        np.testing.assert_allclose(
            array[band - 1],
            ra,
            err_msg="Array not properly reshaped or converted to raster",
        )


@requires_pkg("rasterio", "affine")
def test_create_raster_from_array_transform(example_data_path):
    import affine

    ws = example_data_path / "options" / "dem"
    raster_name = "dem.img"

    raster = Raster.load(ws / raster_name)

    transform = raster._meta["transform"]
    array = raster.get_array(band=raster.bands[0])

    array = np.expand_dims(array, axis=0)
    # same location but shrink raster by factor 2
    new_transform = affine.Affine(
        transform.a / 2, 0, transform.c, 0, transform.e / 2, transform.f
    )

    robj = Raster.raster_from_array(array, crs=raster.crs, transform=new_transform)

    rxmin, rxmax, rymin, rymax = robj.bounds
    xmin, xmax, ymin, ymax = raster.bounds

    if (
        not ((xmax - xmin) / (rxmax - rxmin)) == 2
        or not ((ymax - ymin) / (rymax - rymin)) == 2
    ):
        raise AssertionError("Transform based raster not working properly")
