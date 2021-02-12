import numpy as np

try:
    import rasterio
except ImportError:
    rasterio = None

try:
    import affine
except ImportError:
    affine = None

try:
    import scipy
except ImportError:
    scipy = None


class Raster(object):
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

    FLOAT32 = (float, np.float32, np.float_)
    FLOAT64 = (np.float64,)
    INT8 = (np.int8,)
    INT16 = (np.int16,)
    INT32 = (int, np.int32, np.int_)
    INT64 = (np.int64,)

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
        if rasterio is None:
            msg = (
                "Raster(): error "
                + 'importing rasterio - try "pip install rasterio"'
            )
            raise ImportError(msg)
        else:
            from rasterio.crs import CRS

        if affine is None:
            msg = (
                "Raster(): error "
                + 'importing affine - try "pip install affine"'
            )
            raise ImportError(msg)

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
        elif isinstance(crs, int):
            crs = CRS.from_epsg(crs)
        elif isinstance(crs, str):
            crs = CRS.from_string(crs)
        else:
            TypeError("crs type not understood, provide an epsg or proj4")

        meta["crs"] = crs

        count, height, width = array.shape
        meta["count"] = count
        meta["height"] = height
        meta["width"] = width

        if not isinstance(transform, affine.Affine):
            raise TypeError("Transform must be defined by an Affine object")

        meta["transform"] = transform

        self._meta = meta
        self._dataset = None
        self.__arr_dict = {
            self._bands[b]: arr for b, arr in enumerate(self._array)
        }

        self.__xcenters = None
        self.__ycenters = None

        if isinstance(rio_ds, rasterio.io.DatasetReader):
            self._dataset = rio_ds

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
        md = np.where(dist == np.nanmin(dist))

        # 4: sample the array and average if necessary
        vals = []
        arr = self.get_array(band)
        for ix, i in enumerate(md[0]):
            j = md[1][ix]
            vals.append(arr[i, j])

        value = np.nanmean(vals)

        return value

    def sample_polygon(self, polygon, band, invert=False):
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
            mask = self._intersection(polygon, invert)

            arr_dict = {}
            for b, arr in self.__arr_dict.items():
                t = arr[mask]
                arr_dict[b] = t

        return arr_dict[band]

    def resample_to_grid(self, xc, yc, band, method="nearest"):
        """
        Method to resample the raster data to a
        user supplied grid of x, y coordinates.

        x, y coordinate arrays should correspond
        to grid vertices

        Parameters
        ----------
        xc : np.ndarray or list
            an array of x-cell centers
        yc : np.ndarray or list
            an array of y-cell centers
        band : int
            raster band to re-sample
        method : str
            scipy interpolation method options

            "linear" for bi-linear interpolation
            "nearest" for nearest neighbor
            "cubic" for bi-cubic interpolation

        Returns
        -------
            np.array
        """
        if scipy is None:
            print(
                "Raster().resample_to_grid(): error "
                + 'importing scipy - try "pip install scipy"'
            )
        else:
            from scipy.interpolate import griddata

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
        data = griddata((rxc, ryc), arr, (xc, yc), method=method)

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
            # crop from user supplied points using numpy
            if rasterio is None:
                msg = (
                    "Raster().crop(): error "
                    + 'importing rasterio try "pip install rasterio"'
                )
                raise ImportError(msg)
            else:
                from rasterio.mask import mask

            if affine is None:
                msg = (
                    "Raster(),crop(): error "
                    + 'importing affine - try "pip install affine"'
                )
                raise ImportError(msg)
            else:
                from affine import Affine

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
                ind = np.where(hypot == np.min(hypot))
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
            self._meta["transform"] = Affine(
                transform[0],
                transform[1],
                xmin,
                transform[3],
                transform[4],
                ymax,
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
        if rasterio is None:
            msg = (
                "Raster()._sample_rio_dataset(): error "
                + 'importing rasterio try "pip install rasterio"'
            )
            raise ImportError(msg)
        else:
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

    def _intersection(self, polygon, invert):
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
        from .geospatial_utils import GeoSpatialUtil

        if isinstance(polygon, (list, tuple, np.ndarray)):
            polygon = [polygon]

        geom = GeoSpatialUtil(polygon, shapetype="Polygon")

        polygon = geom.points[0]

        # step 2: create a grid of centoids
        xc = self.xcenters
        yc = self.ycenters

        # step 3: do intersection
        mask = self._point_in_polygon(xc, yc, polygon)
        if invert:
            mask = np.invert(mask)

        return mask

    @staticmethod
    def _point_in_polygon(xc, yc, polygon):
        """
        Use the ray casting algorithm to determine if a point
        is within a polygon. Enables very fast
        intersection calculations!

        Parameters
        ----------
        xc : np.ndarray
            array of xpoints
        yc : np.ndarray
            array of ypoints
        polygon : iterable (list)
            polygon vertices [(x0, y0),....(xn, yn)]
            note: polygon can be open or closed

        Returns
        -------
        mask: np.array
            True value means point is in polygon!

        """
        x0, y0 = polygon[0]
        xt, yt = polygon[-1]

        # close polygon if it isn't already
        if (x0, y0) != (xt, yt):
            polygon.append((x0, y0))

        ray_count = np.zeros(xc.shape, dtype=int)
        num = len(polygon)
        j = num - 1
        for i in range(num):

            tmp = polygon[i][0] + (polygon[j][0] - polygon[i][0]) * (
                yc - polygon[i][1]
            ) / (polygon[j][1] - polygon[i][1])

            comp = np.where(
                ((polygon[i][1] > yc) ^ (polygon[j][1] > yc)) & (xc < tmp)
            )

            j = i
            if len(comp[0]) > 0:
                ray_count[comp[0], comp[1]] += 1

        mask = np.ones(xc.shape, dtype=bool)
        mask[ray_count % 2 == 0] = False

        return mask

    def get_array(self, band, masked=True):
        """
        Method to get a numpy array corresponding to the
        provided raster band. Nodata vals are set to
        np.NaN

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
        if rasterio is None:
            msg = (
                "Raster().write(): error "
                + 'importing rasterio - try "pip install rasterio"'
            )
            raise ImportError(msg)

        if not name.endswith(".tif"):
            name += ".tif"

        with rasterio.open(name, "w", **self._meta) as foo:
            for band, arr in self.__arr_dict.items():
                foo.write(arr, band)

    @staticmethod
    def load(raster):
        """
        Static method to load a raster file
        into the raster object

        Parameters
        ----------
        raster : str

        Returns
        -------
            Raster object

        """
        if rasterio is None:
            msg = (
                "Raster().load(): error "
                + 'importing rasterio - try "pip install rasterio"'
            )
            raise ImportError(msg)

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
        if rasterio is None:
            msg = (
                "Raster().plot(): error "
                + 'importing rasterio - try "pip install rasterio"'
            )
            raise ImportError(msg)
        else:
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
                **kwargs
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
        if rasterio is None:
            msg = (
                "Raster().histogram(): error "
                + 'importing rasterio - try "pip install rasterio"'
            )
            raise ImportError(msg)
        else:
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
