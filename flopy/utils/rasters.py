import warnings
from os import PathLike
from typing import Union

import numpy as np

from .geometry import Polygon
from .utl_import import import_optional_dependency

warnings.simplefilter("always", DeprecationWarning)


class Raster:
    """
    The Raster object is used for cropping, sampling raster values,
    and re-sampling raster values to grids, and provides methods to
    plot rasters and histograms of raster digital numbers for visualization
    and analysis purposes.

    Parameters
    ----------
    array : np.ndarray
        a three dimensional array of raster values with dimensions
        defined by (raster band, nrow, ncol)
    bands : tuple
        a tuple of raster bands
    crs : int, string, rasterio.crs.CRS object
        either a epsg code, a proj4 string, or a CRS object
    transform : affine.Affine object
        affine object, which is used to define geometry
    nodataval : float
        raster no data value
    rio_ds : DatasetReader object
        rasterIO dataset Reader object

    Notes
    -----


    Examples
    --------
    >>> from flopy.utils import Raster
    >>>
    >>> rio = Raster.load("myraster.tif")

    """

    FLOAT32 = (float, np.float32)
    FLOAT64 = (np.float64,)
    INT8 = (np.int8, np.uint8)
    INT16 = (np.int16, np.uint16)
    INT32 = (int, np.int32, np.uint32, np.uint, np.uintc, np.uint32)
    INT64 = (np.int64, np.uint64)

    def __init__(
        self,
        array,
        bands,
        crs,
        transform,
        nodataval,
        driver="GTiff",
        rio_ds=None,
    ):
        from .geometry import point_in_polygon

        rasterio = import_optional_dependency("rasterio")
        from rasterio.crs import CRS

        self._affine = import_optional_dependency("affine")

        self._point_in_polygon = point_in_polygon
        self._array = array
        self._bands = bands

        meta = {"driver": driver, "nodata": nodataval}

        # create metadata dictionary
        if array.dtype in Raster.FLOAT32:
            dtype = "float32"
        elif array.dtype in Raster.FLOAT64:
            dtype = "float64"
        elif array.dtype in Raster.INT8:
            dtype = "int8"
        elif array.dtype in Raster.INT16:
            dtype = "int16"
        elif array.dtype in Raster.INT32:
            dtype = "int32"
        elif array.dtype in Raster.INT64:
            dtype = "int64"
        else:
            raise TypeError("dtype cannot be determined from Raster")

        meta["dtype"] = dtype

        if isinstance(crs, CRS):
            pass
        elif crs is not None:
            crs = CRS.from_user_input(crs)

        meta["crs"] = crs

        count, height, width = array.shape
        meta["count"] = count
        meta["height"] = height
        meta["width"] = width

        if not isinstance(transform, self._affine.Affine):
            raise TypeError("Transform must be defined by an Affine object")

        meta["transform"] = transform

        self._meta = meta
        self._dataset = None
        self.__arr_dict = {self._bands[b]: arr for b, arr in enumerate(self._array)}

        self.__xcenters = None
        self.__ycenters = None

        if isinstance(rio_ds, rasterio.io.DatasetReader):
            self._dataset = rio_ds

    @property
    def crs(self):
        """
        Returns a rasterio CRS object
        """
        return self._meta["crs"]

    @property
    def bounds(self):
        """
        Returns a tuple of xmin, xmax, ymin, ymax boundaries
        """
        height = self._meta["height"]
        width = self._meta["width"]
        transform = self._meta["transform"]
        xmin = transform[2]
        ymax = transform[5]
        xmax, ymin = transform * (width, height)

        return xmin, xmax, ymin, ymax

    @property
    def transform(self):
        """
        Returns the affine transform for the raster
        """
        return self._meta["transform"]

    @property
    def bands(self):
        """
        Returns a tuple of raster bands
        """
        if self._dataset is None:
            return tuple(self._bands)
        else:
            return self._dataset.indexes

    @property
    def nodatavals(self):
        """
        Returns a Tuple of values used to define no data
        """
        if self._dataset is None:
            if isinstance(self._meta["nodata"], list):
                nodata = tuple(self._meta["nodata"])
            elif isinstance(self._meta["nodata"], tuple):
                nodata = self._meta["nodata"]
            else:
                nodata = (self._meta["nodata"],)
            return nodata
        else:
            return self._dataset.nodatavals

    @property
    def xcenters(self):
        """
        Returns a np.ndarray of raster x cell centers
        """
        if self.__xcenters is None:
            self.__xycenters()
        return self.__xcenters

    @property
    def ycenters(self):
        """
        Returns a np.ndarray of raster y cell centers
        """
        if self.__ycenters is None:
            self.__xycenters()
        return self.__ycenters

    def to_crs(self, crs=None, epsg=None, inplace=False):
        """
        Method for re-projecting rasters from an existing CRS to a
        new CRS

        Parameters
        ----------
        crs : CRS user input of many different kinds
        epsg : int
            epsg code input that defines the coordinate system
        inplace : bool
            Boolean flag to indicate if the operation takes place "in place"
            which reprojects the raster within the current object or the
            default (False) to_crs() returns a reprojected Raster object

        Returns
        -------
            Raster or None: returns a reprojected raster object if
                inplace=False, otherwise the reprojected information
                overwrites the current Raster object

        """
        from rasterio.crs import CRS

        if self.crs is None:
            raise ValueError(
                "Cannot transform naive geometries.  "
                "Please set a crs on the object first."
            )
        if crs is not None:
            dst_crs = CRS.from_user_input(crs)
        elif epsg is not None:
            dst_crs = CRS.from_epsg(epsg)
        else:
            raise ValueError("Must pass either crs or epsg.")

        # skip if the input CRS and output CRS are the exact same
        if self.crs.to_epsg() == dst_crs.to_epsg():
            return self

        return self.__transform(dst_crs=dst_crs, inplace=inplace)

    def __transform(self, dst_crs, inplace):
        """

        Parameters
        ----------
        dst_crs : rasterio.CRS object
        inplace : bool

        Returns
        -------
            Raster or None: returns a reprojected raster object if
                inplace=False, otherwise the reprojected information
                overwrites the current Raster object

        """
        import rasterio
        from rasterio.io import MemoryFile
        from rasterio.warp import Resampling, calculate_default_transform, reproject

        height = self._meta["height"]
        width = self._meta["width"]
        xmin, xmax, ymin, ymax = self.bounds

        transform, width, height = calculate_default_transform(
            self.crs, dst_crs, width, height, xmin, ymin, xmax, ymax
        )

        kwargs = {
            "transform": transform,
            "width": width,
            "height": height,
            "crs": dst_crs,
            "nodata": self.nodatavals[0],
            "driver": self._meta["driver"],
            "count": self._meta["count"],
            "dtype": self._meta["dtype"],
        }

        with MemoryFile() as memfile:
            with memfile.open(**kwargs) as dst:
                for band in self.bands:
                    reproject(
                        source=self.get_array(band),
                        destination=rasterio.band(dst, band),
                        src_transform=self.transform,
                        src_crs=self.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest,
                    )
            with memfile.open() as dataset:
                array = dataset.read()
                bands = dataset.indexes
                meta = dataset.meta

                if inplace:
                    for ix, band in enumerate(bands):
                        self.__arr_dict[band] = array[ix]

                    self.__xcenters = None
                    self.__ycenters = None
                    self._meta.update(dict(kwargs.items()))
                    self._dataset = None

                else:
                    return Raster(
                        array,
                        bands,
                        meta["crs"],
                        meta["transform"],
                        meta["nodata"],
                        meta["driver"],
                    )

    def __xycenters(self):
        """
        Method to create np.arrays of the xy-cell centers
        in the raster object
        """
        arr = None
        for _, arr in self.__arr_dict.items():
            break

        if arr is None:
            raise AssertionError("No array data was found")

        ylen, xlen = arr.shape

        # assume that transform is an unrotated plane
        # if transform indicates a rotated plane additional
        # processing will need to be added in this portion of the code
        xd = abs(self._meta["transform"][0])
        yd = abs(self._meta["transform"][4])
        x0, x1, y0, y1 = self.bounds

        # adjust bounds to centroids
        x0 += xd / 2.0
        x1 -= xd / 2.0
        y0 += yd / 2.0
        y1 -= yd / 2.0

        x = np.linspace(x0, x1, xlen)
        y = np.linspace(y1, y0, ylen)
        self.__xcenters, self.__ycenters = np.meshgrid(x, y)

    def sample_point(self, *point, band=1):
        """
        Method to get nearest raster value at a user provided
        point

        Parameters
        ----------
        *point : point geometry representation
            accepted data types:
            x, y values : ex. sample_point(1, 3, band=1)
            tuple of x, y: ex sample_point((1, 3), band=1)
            shapely.geometry.Point
            geojson.Point
            flopy.geometry.Point

        band : int
            raster band to re-sample

        Returns
        -------
            value : float
        """
        from .geospatial_utils import GeoSpatialUtil

        if isinstance(point[0], (tuple, list, np.ndarray)):
            point = point[0]

        geom = GeoSpatialUtil(point, shapetype="Point")

        x, y = geom.points

        # 1: get grid.
        rxc = self.xcenters
        ryc = self.ycenters

        # 2: apply distance equation
        xt = (rxc - x) ** 2
        yt = (ryc - y) ** 2
        dist = np.sqrt(xt + yt)

        # 3: find indices of minimum distance
        md = np.asarray(dist == np.nanmin(dist)).nonzero()

        # 4: sample the array and average if necessary
        vals = []
        arr = self.get_array(band)
        for ix, i in enumerate(md[0]):
            j = md[1][ix]
            vals.append(arr[i, j])

        value = np.nanmean(vals)

        return value

    def sample_polygon(self, polygon, band, invert=False, **kwargs):
        """
        Method to get an unordered list of raster values that are located
        within a arbitrary polygon

        Parameters
        ----------
        polygon : list, geojson, shapely.geometry, shapefile.Shape
            sample_polygon method accepts any of these geometries:

            a list of (x, y) points, ex. [(x1, y1), ...]
            geojson Polygon object
            shapely Polygon object
            shapefile Polygon shape
            flopy Polygon shape

        band : int
            raster band to re-sample

        invert : bool
            Default value is False. If invert is True then the
            area inside the shapes will be masked out

        Returns
        -------
            np.ndarray of unordered raster values

        """
        if band not in self.bands:
            err = (
                "Band number is not recognized, use self.bands for a list "
                "of raster bands"
            )
            raise AssertionError(err)

        if self._dataset is not None:
            arr_dict = self._sample_rio_dataset(polygon, invert)[0]

            for b, arr in arr_dict.items():
                for val in self.nodatavals:
                    t = arr[arr != val]
                    arr_dict[b] = t

        else:
            mask = self._intersection(polygon, invert, **kwargs)

            arr_dict = {}
            for b, arr in self.__arr_dict.items():
                t = arr[mask]
                arr_dict[b] = t

        return arr_dict[band]

    def resample_to_grid(
        self,
        modelgrid,
        band,
        method="nearest",
        extrapolate_edges=False,
    ):
        """
        Method to resample the raster data to a
        user supplied grid of x, y coordinates.

        x, y coordinate arrays should correspond
        to grid vertices

        Parameters
        ----------
        modelgrid : flopy.Grid object
            model grid to sample data from
        band : int
            raster band to re-sample
        method : str
            resampling methods

            ``linear`` for bi-linear interpolation

            ``nearest`` for nearest neighbor

            ``cubic`` for bi-cubic interpolation

            ``mean`` for mean sampling

            ``median`` for median sampling

            ``min`` for minimum sampling

            ``max`` for maximum sampling

            `'mode'` for majority sampling

        extrapolate_edges : bool
            boolean flag indicating if areas without data should be filled
            using the ``nearest`` interpolation method. This option
            has no effect when using the ``nearest`` interpolation method.

        Returns
        -------
            np.array
        """
        import_optional_dependency("scipy")
        rasterstats = import_optional_dependency("rasterstats")
        from scipy.interpolate import griddata

        xmin, xmax, ymin, ymax = modelgrid.extent
        rxmin, rxmax, rymin, rymax = self.bounds
        if any([rxmax < xmin, rxmin > xmax, rymax < ymin, rymin > ymax]):
            raise AssertionError(
                "Raster and model grid do not intersect. Check that the grid "
                "and raster are in the same coordinate reference system"
            )

        method = method.lower()
        if method in {"linear", "nearest", "cubic"}:
            xc = modelgrid.xcellcenters
            yc = modelgrid.ycellcenters

            data_shape = xc.shape
            xc = xc.flatten()
            yc = yc.flatten()
            # step 1: create grid from raster bounds
            rxc = self.xcenters
            ryc = self.ycenters

            # step 2: flatten grid
            rxc = rxc.flatten()
            ryc = ryc.flatten()

            # step 3: get array
            if method == "cubic":
                arr = self.get_array(band, masked=False)
            else:
                arr = self.get_array(band, masked=True)
            arr = arr.flatten()

            # step 3: use griddata interpolation to snap to grid
            rescale = method in {"linear", "nearest"}
            data = griddata((rxc, ryc), arr, (xc, yc), method=method, rescale=rescale)

        elif method in {"median", "mean", "min", "max", "mode"}:
            # these methods are slow and could use speed ups
            data_shape = modelgrid.xcellcenters.shape

            if method == "mode":
                method = "majority"
            xv, yv = modelgrid.cross_section_vertices
            polygons = [list(zip(x, yv[ix])) for ix, x in enumerate(xv)]
            polygons = [Polygon(p) for p in polygons]
            rstr = self.get_array(band, masked=False)
            affine = self._meta["transform"]
            nodata = self.nodatavals[0]
            zs = rasterstats.zonal_stats(
                polygons, rstr, affine=affine, stats=[method], nodata=nodata
            )
            data = np.array(
                [z[method] if z[method] is not None else np.nan for z in zs]
            )

        else:
            raise TypeError(f"{method} method not supported")

        if extrapolate_edges:
            xc = modelgrid.xcellcenters
            yc = modelgrid.ycellcenters

            xc = xc.flatten()
            yc = yc.flatten()

            # step 1: create grid from raster bounds
            rxc = self.xcenters
            ryc = self.ycenters

            # step 2: flatten grid
            rxc = rxc.flatten()
            ryc = ryc.flatten()

            arr = self.get_array(band, masked=True).flatten()

            # filter out nan values from the original dataset
            if np.isnan(np.sum(arr)):
                idx = np.isfinite(arr)
                rxc = rxc[idx]
                ryc = ryc[idx]
                arr = arr[idx]

            extrapolate = griddata(
                (rxc, ryc), arr, (xc, yc), method="nearest", rescale=True
            )
            data = np.where(np.isnan(data), extrapolate, data)

        # step 4: return grid to user in shape provided
        data.shape = data_shape

        # step 5: re-apply nodata values
        data[np.isnan(data)] = self.nodatavals[0]

        return data

    def crop(self, polygon, invert=False):
        """
        Method to crop a new raster object
        from the current raster object

        Parameters
        ----------
        polygon : list, geojson, shapely.geometry, shapefile.Shape
            crop method accepts any of these geometries:

            a list of (x, y) points, ex. [(x1, y1), ...]
            geojson Polygon object
            shapely Polygon object
            shapefile Polygon shape
            flopy Polygon shape

        invert : bool
            Default value is False. If invert is True then the
            area inside the shapes will be masked out

        """
        if self._dataset is not None:
            arr_dict, rstr_crp_meta = self._sample_rio_dataset(polygon, invert)
            self.__arr_dict = arr_dict
            self._meta = rstr_crp_meta
            self._dataset = None
            self.__xcenters = None
            self.__ycenters = None

        else:
            mask = self._intersection(polygon, invert)

            xc = self.xcenters
            yc = self.ycenters
            # step 4: find bounding box
            xba = np.copy(xc)
            yba = np.copy(yc)
            xba[~mask] = np.nan
            yba[~mask] = np.nan

            xmin = np.nanmin(xba)
            xmax = np.nanmax(xba)
            ymin = np.nanmin(yba)
            ymax = np.nanmax(yba)

            bbox = [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]

            # step 5: use bounding box to crop array
            xind = []
            yind = []
            for pt in bbox:
                xt = (pt[0] - xc) ** 2
                yt = (pt[1] - yc) ** 2
                hypot = np.sqrt(xt + yt)
                ind = np.asarray(hypot == np.min(hypot)).nonzero()
                yind.append(ind[0][0])
                xind.append(ind[1][0])

            xmii = np.min(xind)
            xmai = np.max(xind)
            ymii = np.min(yind)
            ymai = np.max(yind)

            crp_mask = mask[ymii : ymai + 1, xmii : xmai + 1]
            nodata = self._meta["nodata"]
            if not isinstance(nodata, float) and not isinstance(nodata, int):
                try:
                    nodata = nodata[0]
                except (IndexError, TypeError):
                    nodata = -1.0e38
                    self._meta["nodata"] = nodata

            arr_dict = {}
            for band, arr in self.__arr_dict.items():
                t = arr[ymii : ymai + 1, xmii : xmai + 1]
                t[~crp_mask] = nodata
                arr_dict[band] = t

            self.__arr_dict = arr_dict

            # adjust xmin, ymax back to appropriate grid locations
            xd = abs(self._meta["transform"][0])
            yd = abs(self._meta["transform"][4])
            xmin -= xd / 2.0
            ymax += yd / 2.0

            # step 6: update metadata including a new Affine
            self._meta["height"] = crp_mask.shape[0]
            self._meta["width"] = crp_mask.shape[1]
            transform = self._meta["transform"]
            self._meta["transform"] = self._affine.Affine(
                transform[0], transform[1], xmin, transform[3], transform[4], ymax
            )
            self.__xcenters = None
            self.__ycenters = None

    def _sample_rio_dataset(self, polygon, invert):
        """
        Internal method to sample a rasterIO dataset using
        rasterIO built ins

        Parameters
        ----------
        polygon : list, geojson, shapely.geometry, shapefile.Shape
            _sample_rio_dataset method accepts any of these geometries:

            a list of (x, y) points, ex. [(x1, y1), ...]
            geojson Polygon object
            shapely Polygon object
            shapefile Polygon shape
            flopy Polygon shape

        invert : bool
            Default value is False. If invert is True then the
            area inside the shapes will be masked out

        Returns
        -------
            tuple : (arr_dict, raster_crp_meta)

        """
        import_optional_dependency("rasterio")
        from rasterio.mask import mask

        from .geospatial_utils import GeoSpatialUtil

        if isinstance(polygon, (list, tuple, np.ndarray)):
            polygon = [polygon]

        geom = GeoSpatialUtil(polygon, shapetype="Polygon")
        shapes = [geom]

        rstr_crp, rstr_crp_affine = mask(
            self._dataset, shapes, crop=True, invert=invert
        )

        rstr_crp_meta = self._dataset.meta.copy()
        rstr_crp_meta.update(
            {
                "driver": "GTiff",
                "height": rstr_crp.shape[1],
                "width": rstr_crp.shape[2],
                "transform": rstr_crp_affine,
            }
        )

        arr_dict = {self.bands[b]: arr for b, arr in enumerate(rstr_crp)}

        return arr_dict, rstr_crp_meta

    def _intersection(self, polygon, invert, **kwargs):
        """
        Internal method to create an intersection mask, used for cropping
        arrays and sampling arrays.

        Parameters
        ----------
        polygon : list, geojson, shapely.geometry, shapefile.Shape
            _intersection method accepts any of these geometries:

            a list of (x, y) points, ex. [(x1, y1), ...]
            geojson Polygon object
            shapely Polygon object
            shapefile Polygon shape
            flopy Polygon shape

        invert : bool
            Default value is False. If invert is True then the
            area inside the shapes will be masked out

        Returns
        -------
            mask : np.ndarray (dtype = bool)

        """
        # the convert kwarg is to speed up the resample_to_grid method
        #  which already provides the proper datatype to _intersect()
        convert = kwargs.pop("convert", True)
        if convert:
            from .geospatial_utils import GeoSpatialUtil

            if isinstance(polygon, (list, tuple, np.ndarray)):
                polygon = [polygon]

            geom = GeoSpatialUtil(polygon, shapetype="Polygon")

            polygon = list(geom.points[0])

        # step 2: create a grid of centoids
        xc = self.xcenters
        yc = self.ycenters

        # step 3: do intersection
        mask = self._point_in_polygon(xc, yc, polygon)
        if invert:
            mask = np.invert(mask)

        return mask

    def get_array(self, band, masked=True):
        """
        Method to get a numpy array corresponding to the
        provided raster band. Nodata vals are set to
        np.nan

        Parameters
        ----------
        band : int
            band number from the raster
        masked : bool
            determines if nodatavals will be returned as np.nan to
            the user

        Returns
        -------
            np.ndarray

        """
        if band not in self.bands:
            raise ValueError("Band {} not a valid value")

        if self._dataset is None:
            array = np.copy(self.__arr_dict[band])
        else:
            array = self._dataset.read(band)

        if masked:
            for v in self.nodatavals:
                array = array.astype(float)
                array[array == v] = np.nan

        return array

    def write(self, name):
        """
        Method to write raster data to a .tif
        file

        Parameters
        ----------
        name : str
            output raster .tif file name

        """
        rasterio = import_optional_dependency("rasterio")

        if not name.endswith(".tif"):
            name += ".tif"

        with rasterio.open(name, "w", **self._meta) as foo:
            for band, arr in self.__arr_dict.items():
                foo.write(arr, band)

    @staticmethod
    def load(raster: Union[str, PathLike]):
        """
        Static method to load a raster file
        into the raster object

        Parameters
        ----------
        raster : str or PathLike
            The path to the raster file

        Returns
        -------
            A Raster object

        """
        rasterio = import_optional_dependency("rasterio")

        dataset = rasterio.open(raster)
        array = dataset.read()
        bands = dataset.indexes
        meta = dataset.meta

        return Raster(
            array,
            bands,
            meta["crs"],
            meta["transform"],
            meta["nodata"],
            meta["driver"],
        )

    @staticmethod
    def raster_from_array(
        array,
        modelgrid=None,
        nodataval=1e-10,
        crs=None,
        transform=None,
    ):
        """
        Method to create a raster from an array. When using a modelgrid to
        define the transform, delc and delr must be uniform in each dimension.
        Otherwise, the user can define their own transform using the affine
        package.

        Parameters
        ----------
        array : np.ndarray
            array of (n-bands, nrows, ncols) for the raster
        modelgrid : flopy.discretization.StructuredGrid
            StructuredGrid object (optional), but transform must be defined
            if a StructuredGrid is not supplied
        nodataval : (int, float)
            Null value
        crs : coordinate reference system input of many types
        transform : affine.Affine
            optional affine transform that defines the spatial parameters
            of the raster. This must be supplied if a modelgrid is not
            used to define the transform

        Returns
        -------
            Raster object
        """
        from affine import Affine

        if not isinstance(array, np.ndarray):
            array = np.array(array)

        if modelgrid is not None:
            if crs is None:
                if modelgrid.crs is None:
                    raise ValueError(
                        "Cannot create a raster from a grid without a "
                        "coordinate reference system, please provide a crs "
                        "using crs="
                    )
                crs = modelgrid.crs

            if modelgrid.grid_type != "structured":
                raise TypeError(f"{type(modelgrid)} discretizations are not supported")

            if not np.all(modelgrid.delc == modelgrid.delc[0]):
                raise AssertionError("DELC must have a uniform spacing")

            if not np.all(modelgrid.delr == modelgrid.delr[0]):
                raise AssertionError("DELR must have a uniform spacing")

            yul = modelgrid.yvertices[0, 0]
            xul = modelgrid.xvertices[0, 0]
            angrot = modelgrid.angrot
            transform = Affine(modelgrid.delr[0], 0, xul, 0, -modelgrid.delc[0], yul)

            if angrot != 0:
                transform *= Affine.rotation(angrot)

            if array.size % modelgrid.ncpl != 0:
                raise AssertionError(
                    f"Array size {array.size} is not a multiple of the "
                    f"number of cells per layer in the model grid "
                    f"{modelgrid.ncpl}"
                )

            array = array.reshape((-1, modelgrid.nrow, modelgrid.ncol))

        if transform is not None:
            if crs is None:
                raise ValueError(
                    "Cannot create a raster without a coordinate reference "
                    "system, please use crs= to provide a coordinate reference"
                )

            bands, height, width = array.shape

            return Raster(
                array,
                bands=list(range(1, bands + 1)),
                crs=crs,
                transform=transform,
                nodataval=nodataval,
            )

    def plot(self, ax=None, contour=False, **kwargs):
        """
        Method to plot raster layers or contours.

        Parameters
        ----------
        ax : matplotlib.pyplot.axes
            optional matplotlib axes for plotting
        contour : bool
            flag to indicate creation of contour plot

        **kwargs :
            matplotlib keyword arguments
            see matplotlib documentation for valid
            arguments for plot and contour.

        Returns
        -------
            ax : matplotlib.pyplot.axes

        """
        import_optional_dependency("rasterio")
        from rasterio.plot import show

        if self._dataset is not None:
            ax = show(self._dataset, ax=ax, contour=contour, **kwargs)

        else:
            d0 = len(self.__arr_dict)
            d1, d2 = None, None
            for _, arr in self.__arr_dict.items():
                d1, d2 = arr.shape

            if d1 is None:
                raise AssertionError("No plottable arrays found")

            data = np.zeros((d0, d1, d2), dtype=float)
            i = 0
            for _, arr in sorted(self.__arr_dict.items()):
                data[i, :, :] = arr
                i += 1

            data = np.ma.masked_where(data == self.nodatavals, data)
            ax = show(
                data,
                ax=ax,
                contour=contour,
                transform=self._meta["transform"],
                **kwargs,
            )

        return ax

    def histogram(self, ax=None, **kwargs):
        """
        Method to plot a histogram of digital numbers

        Parameters
        ----------
        ax : matplotlib.pyplot.axes
            optional matplotlib axes for plotting

        **kwargs :
            matplotlib keyword arguments
            see matplotlib documentation for valid
            arguments for histogram

        Returns
        -------
            ax : matplotlib.pyplot.axes

        """
        import_optional_dependency("rasterio")
        from rasterio.plot import show_hist

        if "alpha" not in kwargs:
            kwargs["alpha"] = 0.3

        if self._dataset is not None:
            ax = show_hist(self._dataset, ax=ax, **kwargs)

        else:
            d0 = len(self.__arr_dict)
            d1, d2 = None, None
            for _, arr in self.__arr_dict.items():
                d1, d2 = arr.shape

            if d1 is None:
                raise AssertionError("No plottable arrays found")

            data = np.zeros((d0, d1, d2), dtype=float)
            i = 0
            for _, arr in sorted(self.__arr_dict.items()):
                data[i, :, :] = arr
                i += 1

            data = np.ma.masked_where(data == self.nodatavals, data)
            ax = show_hist(data, ax=ax, **kwargs)

        return ax
