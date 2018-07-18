import numpy as np
from pandas import DataFrame
from ..utils.datautil import PyListUtil
from .modelgrid import ModelGrid, GridType, PointType, LocationType, \
                       CachedData, CachedDataType


class StructuredModelGrid(ModelGrid):
    """
    def get_cell_vertices(i, j, point_type)
        returns vertices for a single cell or sequence of i, j locations.
    get_row_array : ()
        returns a numpy ndarray sized to a model row
    get_column_array : ()
        returns a numpy ndarray sized to a model column
    get_layer_array : ()
        returns a numpy ndarray sized to a model layer
    num_rows
        returns the number of model rows
    num_columns
        returns the number of model columns

    """
    def __init__(self, delc, delr, top, botm, idomain, sr=None,
                 simulation_time=None, model_name='', steady=False):
        super(StructuredModelGrid, self).__init__(GridType.structured, sr,
                                                  simulation_time,
                                                  model_name, steady)
        self._delc = delc
        self._delr = delr
        self._top = top
        self._botm = botm
        self._idomain = idomain
        self._nlay = len(botm)
        self._nrow = len(delc)
        self._ncol = len(delr)

    ####################
    # Properties
    ####################
    @property
    def nlay(self):
        return self._nlay

    @property
    def nrow(self):
        return self._nrow

    @property
    def ncol(self):
        return self._ncol

    @property
    def delc(self):
        return self._delc

    @delc.setter
    def delc(self, delc):
        self._delc = delc
        self._ncol = len(delc)
        self._require_cache_updates()

    @property
    def delr(self):
        return self._delr

    @delr.setter
    def delr(self, delr):
        self._delr = delr
        self._nrow = len(delr)
        self._require_cache_updates()

    @property
    def top(self):
        return self._top

    @top.setter
    def top(self, top):
        self._top = top
        self._require_cache_updates()

    @property
    def botm(self):
        return self._botm

    @property
    def top_botm(self):
        new_top = np.expand_dims(self._top, 0)
        return np.concatenate((new_top, self._botm), axis=0)

    @botm.setter
    def botm(self, botm):
        self._botm = botm
        self._botm = len(botm)
        self._require_cache_updates()

    @property
    def idomain(self):
        return self._idomain

    @idomain.setter
    def idomain(self, idomain):
        self._idomain = idomain
        self._require_cache_updates()

    @property
    def bounds(self):
        """Return bounding box in shapely order."""
        xmin, xmax, ymin, ymax = self.get_extent(point_type=
                                                 PointType.spatialxyz)
        return xmin, ymin, xmax, ymax

    ####################
    # Methods
    ####################
    def get_tabular_data(self, data, location_type=LocationType.spatialxyz):
        # initialize counters
        current_row = 0
        current_col = 0
        current_cellid = 0
        current_layer = 0
        current_layer_cellid = 0
        cells_per_layer = self._nrow * self._ncol
        # initialize "pandas" dictionary
        if location_type == LocationType.modelxyz or \
                location_type == LocationType.spatialxyz:
            data_d = {'X': [], 'Y': [], 'data': []}
        elif location_type == LocationType.cellid:
            data_d = {'cellid' : [], 'data': []}
        elif location_type == LocationType.layer_cellid:
            data_d = {'layer': [], 'layer_cellid': [], 'data': []}
        elif location_type == LocationType.lrc:
            data_d = {'layer': [], 'row': [], 'column': [], 'data': []}

        # loop through the data and build out pandas dictionary
        for item in PyListUtil.next_item(data):
            # build location data
            if location_type == LocationType.modelxyz or \
                    location_type == LocationType.spatialxyz:
                if location_type == LocationType.modelxyz:
                    point_type = PointType.modelxyz
                else:
                    point_type = PointType.spatialxyz
                cell_center = self.get_cellcenter(current_row,
                                                  current_col,
                                                  point_type)
                data_d['X'].append(cell_center[0])
                data_d['Y'].append(cell_center[1])
            elif location_type == LocationType.cellid:
                data_d['cellid'].append(current_cellid)
            elif location_type == LocationType.layer_cellid:
                data_d['layer'].append(current_layer)
                data_d['layer_cellid'].append(current_layer_cellid)
            elif location_type == LocationType.lrc:
                data_d['layer'].append(current_layer)
                data_d['row'].append(current_row)
                data_d['column'].append(current_col)
            # build data
            data_d['data'].append(item[0])
            # update counters
            if current_col == self._ncol - 1:
                current_col = 0
                current_row += 1
            else:
                current_col += 1
            current_cellid += 1
            if current_layer_cellid == cells_per_layer - 1:
                current_layer_cellid = 0
                current_layer += 1
                current_row = 0
                current_col = 0
                if current_layer == self._nlay:
                    break
            else:
                current_layer_cellid += 1
        return DataFrame(data=data_d)

    def get_edge_array(self):
        """
        Return two numpy one-dimensional float arrays. One array has the cell
        edge x coordinates for every column in the grid in model space -
        not offset or rotated.  Array is of size (ncol + 1). The other array
        has the cell edge y coordinates.
        """
        if CachedDataType.edge_array.value not in self._cache_dict or \
                self._cache_dict[CachedDataType.edge_array.value].out_of_date:
            xedge = np.concatenate(([0.], np.add.accumulate(self._delr)))
            length_y = np.add.reduce(self.delc)
            yedge = np.concatenate(([length_y], length_y -
                                    np.add.accumulate(self.delc)))
            self._cache_dict[CachedDataType.edge_array.value] = \
                CachedData([xedge, yedge])
        return self._cache_dict[CachedDataType.edge_array.value].data

    def get_xygrid(self, point_type=PointType.spatialxyz):
        xgrid, ygrid = np.meshgrid(self.xedge, self.yedge)
        if point_type == PointType.spatialxyz:
            return self.sr.transform(xgrid, ygrid)
        else:
            return xgrid, ygrid

    def get_cell_vertices(self, i, j, point_type=PointType.spatialxyz):
        """Get vertices for a single cell or sequence of i, j locations."""
        pts = []
        xgrid, ygrid = self.xedgegrid(point_type), self.yedgegrid(point_type)
        pts.append([xgrid[i, j], ygrid[i, j]])
        pts.append([xgrid[i + 1, j], ygrid[i + 1, j]])
        pts.append([xgrid[i + 1, j + 1], ygrid[i + 1, j + 1]])
        pts.append([xgrid[i, j + 1], ygrid[i, j + 1]])
        pts.append([xgrid[i, j], ygrid[i, j]])
        if np.isscalar(i):
            return pts
        else:
            vrts = np.array(pts).transpose([2, 0, 1])
            return [v.tolist() for v in vrts]

    def xyvertices(self, point_type=PointType.spatialxyz):
        cache_index = (CachedDataType.xyvertices.value, point_type.value)
        if cache_index not in self._cache_dict or \
                self._cache_dict[cache_index].out_of_date:
            jj, ii = np.meshgrid(range(self._ncol), range(self._nrow))
            jj, ii = jj.ravel(), ii.ravel()
            self._cache_dict[cache_index] = \
                CachedData(self.get_cell_vertices(ii, jj))
        return self._cache_dict[cache_index].data

    def get_cellcenter(self, row, col, point_type=PointType.spatialxyz):
        cell_centers = self.get_cellcenters(point_type)
        return cell_centers[0][row], cell_centers[1][col]

    def get_cellcenters(self, point_type=PointType.spatialxyz):
        """
        Return a list of two numpy one-dimensional float array one with
        the cell center x coordinate and the other with the cell center y
        coordinate for every row in the grid in model space -
        not offset of rotated, with the cell center y coordinate.
        """
        cache_index = (CachedDataType.cell_centers.value, point_type.value)
        if cache_index not in self._cache_dict or \
                self._cache_dict[cache_index].out_of_date:
            # get x centers
            x = np.add.accumulate(self._delr) - 0.5 * self.delr
            # get y centers
            Ly = np.add.reduce(self._delc)
            y = Ly - (np.add.accumulate(self._delc) - 0.5 *
                      self._delc)
            x_mesh, y_mesh = np.meshgrid(x, y)
            # get z centers
            z = np.empty((self._nlay, self._nrow, self._ncol))
            z[0, :, :] = (self._top[:, :] + self._botm[0, :, :]) / 2.
            for l in range(1, self._nlay):
                z[l, :, :] = (self._botm[l - 1, :, :] +
                              self._botm[l, :, :]) / 2.
            if point_type == PointType.spatialxyz:
                # transform x and y
                x_mesh, y_mesh = self.sr.transform(x_mesh, y_mesh)
            # store in cache
            self._cache_dict[cache_index] = CachedData([x_mesh, y_mesh, z])
        return self._cache_dict[cache_index].data

    def get_grid_lines(self, point_type=PointType.spatialxyz):
        """
            Get the grid lines as a list

        """
        xmin = self.xedge[0]
        xmax = self.xedge[-1]
        ymin = self.yedge[-1]
        ymax = self.yedge[0]
        lines = []
        # Vertical lines
        for j in range(self.ncol + 1):
            x0 = self.xedge[j]
            x1 = x0
            y0 = ymin
            y1 = ymax
            if point_type == PointType.spatialxyz:
                x0, y0 = self.sr.transform(x0, y0)
                x1, y1 = self.sr.transform(x1, y1)
            lines.append([(x0, y0), (x1, y1)])

        # horizontal lines
        for i in range(self.nrow + 1):
            x0 = xmin
            x1 = xmax
            y0 = self.yedge[i]
            y1 = y0
            if point_type == PointType.spatialxyz:
                x0, y0 = self.sr.transform(x0, y0)
                x1, y1 = self.sr.transform(x1, y1)
            lines.append([(x0, y0), (x1, y1)])
        return lines

    def get_extent(self, point_type=PointType.spatialxyz):
        """
        Get the extent of the rotated and offset grid

        Return (xmin, xmax, ymin, ymax)

        """
        x0 = self.xedge[0]
        x1 = self.xedge[-1]
        y0 = self.yedge[0]
        y1 = self.yedge[-1]

        if point_type == PointType.spatialxyz:
            # upper left point
            x0r, y0r = self.sr.transform(x0, y0)

            # upper right point
            x1r, y1r = self.sr.transform(x1, y0)

            # lower right point
            x2r, y2r = self.sr.transform(x1, y1)

            # lower left point
            x3r, y3r = self.sr.transform(x0, y1)

            xmin = min(x0r, x1r, x2r, x3r)
            xmax = max(x0r, x1r, x2r, x3r)
            ymin = min(y0r, y1r, y2r, y3r)
            ymax = max(y0r, y1r, y2r, y3r)

            return (xmin, xmax, ymin, ymax)
        else:
            return (x0, x1, y0, y1)

    def get_all_model_cells(self):
        model_cells = []
        for layer in range(0, self._nlay):
            for row in range(0, self._nrow):
                for column in range(0, self._ncol):
                    model_cells.append((layer + 1, row + 1, column + 1))
        return model_cells

    def write_gridSpec(self, filename):
        """ write a PEST-style grid specification file
        """
        f = open(filename, 'w')
        f.write(
            "{0:10d} {1:10d}\n".format(self.delc.shape[0], self.delr.shape[0]))
        f.write("{0:15.6E} {1:15.6E} {2:15.6E}\n".format(
            self.sr.xul * self.sr.length_multiplier,
            self.sr.yul * self.sr.length_multiplier,
            self.sr.rotation))

        for r in self.delr:
            f.write("{0:15.6E} ".format(r))
        f.write('\n')
        for c in self.delc:
            f.write("{0:15.6E} ".format(c))
        f.write('\n')
        return

    def plot_array(self, a, ax=None, **kwargs):
        """
        Create a QuadMesh plot of the specified array using pcolormesh

        Parameters
        ----------
        a : np.ndarray

        Returns
        -------
        quadmesh : matplotlib.collections.QuadMesh

        """
        from flopy.plot.plotutil import PlotUtilities

        ax = PlotUtilities._plot_array_helper(a, axes=ax, modelgrid=self,
                                              **kwargs)
        return ax

    def export_contours(self, filename, contours,
                        fieldname='level', epsg=None, prj=None,
                        **kwargs):
        """Convert matplotlib contour plot object to shapefile.

        Parameters
        ----------
        filename : str
            path of output shapefile
        contours : matplotlib.contour.QuadContourSet or list of them
            (object returned by matplotlib.pyplot.contour)
        epsg : int
            EPSG code. See https://www.epsg-registry.org/ or spatialreference.org
        prj : str
            Existing projection file to be used with new shapefile.
        **kwargs : key-word arguments to flopy.export.shapefile_utils.recarray2shp

        Returns
        -------
        df : dataframe of shapefile contents
        """
        from flopy.utils.geometry import LineString
        from flopy.export.shapefile_utils import recarray2shp

        if not isinstance(contours, list):
            contours = [contours]

        if epsg is None:
            epsg = self._sr._epsg
        if prj is None:
            prj = self._sr.proj4_str

        geoms = []
        level = []
        for ctr in contours:
            levels = ctr.levels
            for i, c in enumerate(ctr.collections):
                paths = c.get_paths()
                geoms += [LineString(p.vertices) for p in paths]
                level += list(np.ones(len(paths)) * levels[i])

        # convert the dictionary to a recarray
        ra = np.array(level,
                      dtype=[(fieldname, float)]).view(np.recarray)

        recarray2shp(ra, geoms, filename, epsg, prj, **kwargs)

    def export_array_contours(self, filename, a,
                              fieldname='level',
                              interval=None,
                              levels=None,
                              maxlevels=1000,
                              epsg=None,
                              prj=None,
                              **kwargs):
        """Contour an array using matplotlib; write shapefile of contours.

        Parameters
        ----------
        filename : str
            Path of output file with '.shp' extention.
        a : 2D numpy array
            Array to contour
        epsg : int
            EPSG code. See https://www.epsg-registry.org/ or spatialreference.org
        prj : str
            Existing projection file to be used with new shapefile.
        **kwargs : key-word arguments to flopy.export.shapefile_utils.recarray2shp
        """
        import matplotlib.pyplot as plt

        if epsg is None:
            epsg = self._sr._epsg
        if prj is None:
            prj = self._sr.proj4_str

        if interval is not None:
            min = np.nanmin(a)
            max = np.nanmax(a)
            nlevels = np.round(np.abs(max - min) / interval, 2)
            msg = '{:.0f} levels at interval of {} > maxlevels={}'.format(
                nlevels,
                interval,
                maxlevels)
            assert nlevels < maxlevels, msg
            levels = np.arange(min, max, interval)
        fig, ax = plt.subplots()
        ctr = self.contour_array(ax, a, levels=levels)
        self.export_contours(filename, ctr, fieldname, epsg, prj, **kwargs)
        plt.close()

    def contour_array(self, ax, a, **kwargs):
        """
        Create a QuadMesh plot of the specified array using pcolormesh

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            ax to add the contours

        a : np.ndarray
            array to contour

        Returns
        -------
        contour_set : ContourSet

        """
        from flopy.plot import PlotMapView

        kwargs['ax'] = ax
        map = PlotMapView(modelgrid=self)
        contour_set = map.contour_array(a=a, **kwargs)

        return contour_set

    # move to export folder
    def load_shapefile_data(self, shapefile):
        # this will only support loading shapefiles with the same
        # spatial reference as the model grid
        raise NotImplementedError('must define get_model_dim_arrays in child '
                                  'class to use this base class')

    def get_row_array(self):
        return np.arange(1, self._nrow + 1, 1, np.int)

    def get_column_array(self):
        return np.arange(1, self._ncol + 1, 1, np.int)

    def get_layer_array(self):
        return np.arange(1, self._nlay + 1, 1, np.int)

    def get_model_dim_arrays(self):
        return [np.arange(1, self._nlay + 1, 1, np.int),
                np.arange(1, self._nrow + 1, 1, np.int),
                np.arange(1, self._ncol + 1, 1, np.int)]

    def num_cells_per_layer(self):
        return self._nrow * self._ncol

    def num_cells(self, active_only=False):
        if active_only:
            raise NotImplementedError('this feature is not yet implemented')
        else:
            return self._nrow * self._ncol * self._nlay

    def get_model_dim(self):
        if self.grid_type() == GridType.structured:
            return [self._nlay, self._nrow, self._ncol]
        elif self.grid_type() == GridType.layered_vertex:
            return [self._nlay, self.num_cells_per_layer()]
        elif self.grid_type() == GridType.unlayered_vertex:
            return [self.num_cells()]

    def get_model_dim_names(self):
        if self.grid_type() == GridType.structured:
            return ['layer', 'row', 'column']
        elif self.grid_type() == GridType.layered_vertex:
            return ['layer', 'layer_cell_num']
        elif self.grid_type() == GridType.unlayered_vertex:
            return ['node']

    def get_horizontal_cross_section_dim_names(self):
        return ['row', 'column']

    def get_horizontal_cross_section_dim_arrays(self):
        return [np.arange(1, self._nrow + 1, 1, np.int),
                np.arange(1, self._ncol + 1, 1, np.int)]

    def export_array(self, filename, a, nodata=-9999,
                     fieldname='value',
                     **kwargs):
        """Write a numpy array to Arc Ascii grid
        or shapefile with the model reference.

        Parameters
        ----------
        filename : str
            Path of output file. Export format is determined by
            file extention.
            '.asc'  Arc Ascii grid
            '.tif'  GeoTIFF (requries rasterio package)
            '.shp'  Shapefile
        a : 2D numpy.ndarray
            Array to export
        nodata : scalar
            Value to assign to np.nan entries (default -9999)
        fieldname : str
            Attribute field name for array values (shapefile export only).
            (default 'values')
        kwargs:
            keyword arguments to np.savetxt (ascii)
            rasterio.open (GeoTIFF)
            or flopy.export.shapefile_utils.write_grid_shapefile2

        Notes
        -----
        Rotated grids will be either be unrotated prior to export,
        using scipy.ndimage.rotate (Arc Ascii format) or rotation will be
        included in their transform property (GeoTiff format). In either case
        the pixels will be displayed in the (unrotated) projected geographic coordinate system,
        so the pixels will no longer align exactly with the model grid
        (as displayed from a shapefile, for example). A key difference between
        Arc Ascii and GeoTiff (besides disk usage) is that the
        unrotated Arc Ascii will have a different grid size, whereas the GeoTiff
        will have the same number of rows and pixels as the original.
        """

        if filename.lower().endswith(".asc"):
            if len(np.unique(self.delr)) != len(np.unique(self.delc)) != 1 \
                    or self.delr[0] != self.delc[0]:
                raise ValueError('Arc ascii arrays require a uniform grid.')

            xll, yll = self.sr.xll, self.sr.yll
            cellsize = self.delr[0] * self.sr.length_multiplier
            fmt = kwargs.get('fmt', '%.18e')
            a = a.copy()
            a[np.isnan(a)] = nodata
            if self.sr.rotation != 0:
                try:
                    from scipy.ndimage import rotate
                    a = rotate(a, self.sr.rotation, cval=nodata)
                    height_rot, width_rot = a.shape
                    xmin, ymin, xmax, ymax = self.bounds
                    dx = (xmax - xmin) / width_rot
                    dy = (ymax - ymin) / height_rot
                    cellsize = np.max((dx, dy))
                    # cellsize = np.cos(np.radians(self.rotation)) * cellsize
                    xll, yll = xmin, ymin
                except ImportError:
                    print('scipy package required to export rotated grid.')
                    pass

            filename = '.'.join(
                filename.split('.')[:-1]) + '.asc'  # enforce .asc ending
            nrow, ncol = a.shape
            a[np.isnan(a)] = nodata
            txt = 'ncols  {:d}\n'.format(ncol)
            txt += 'nrows  {:d}\n'.format(nrow)
            txt += 'xllcorner  {:f}\n'.format(xll)
            txt += 'yllcorner  {:f}\n'.format(yll)
            txt += 'cellsize  {}\n'.format(cellsize)
            # ensure that nodata fmt consistent w values
            txt += 'NODATA_value  {}\n'.format(fmt) % (nodata)
            with open(filename, 'w') as output:
                output.write(txt)
            with open(filename, 'ab') as output:
                np.savetxt(output, a, **kwargs)
            print('wrote {}'.format(filename))

        elif filename.lower().endswith(".tif"):
            if len(np.unique(self.delr)) != len(np.unique(self.delc)) != 1 \
                    or self.delr[0] != self.delc[0]:
                raise ValueError('GeoTIFF export require a uniform grid.')
            try:
                import rasterio
                from rasterio import Affine
            except:
                print('GeoTIFF export requires the rasterio package.')
                return
            dxdy = self.delc[0] * self.sr.length_multiplier
            trans = Affine.translation(self.sr.xul, self.sr.yul) * \
                    Affine.rotation(self.sr.rotation) * \
                    Affine.scale(dxdy, -dxdy)

            # third dimension is the number of bands
            a = a.copy()
            if len(a.shape) == 2:
                a = np.reshape(a, (1, a.shape[0], a.shape[1]))
            if a.dtype.name == 'int64':
                a = a.astype('int32')
                dtype = rasterio.int32
            elif a.dtype.name == 'int32':
                dtype = rasterio.int32
            elif a.dtype.name == 'float64':
                dtype = rasterio.float64
            elif a.dtype.name == 'float32':
                dtype = rasterio.float32
            else:
                msg = 'ERROR: invalid dtype "{}"'.format(a.dtype.name)
                raise TypeError(msg)

            meta = {'count': a.shape[0],
                    'width': a.shape[2],
                    'height': a.shape[1],
                    'nodata': nodata,
                    'dtype': dtype,
                    'driver': 'GTiff',
                    'crs': self.sr.proj4_str,
                    'transform': trans
                    }
            meta.update(kwargs)
            with rasterio.open(filename, 'w', **meta) as dst:
                dst.write(a)
            print('wrote {}'.format(filename))

        elif filename.lower().endswith(".shp"):
            from ..export.shapefile_utils import write_grid_shapefile2
            epsg = kwargs.get('epsg', None)
            prj = kwargs.get('prj', None)
            if epsg is None and prj is None:
                epsg = self.sr.epsg
            write_grid_shapefile2(filename, self, array_dict={fieldname: a},
                                  nan_val=nodata,
                                  epsg=epsg, prj=prj)

    def write_shapefile(self, filename='grid.shp', epsg=None, prj=None):
        """Write a shapefile of the grid with just the row and column attributes"""
        from ..export.shapefile_utils import write_grid_shapefile2
        if epsg is None and prj is None:
            epsg = self.sr.epsg
        write_grid_shapefile2(filename, self, array_dict={}, nan_val=-1.0e9,
                              epsg=epsg, prj=prj)

    def interpolate(self, a, xi, method='nearest'):
        """
        Use the griddata method to interpolate values from an array onto the
        points defined in xi.  For any values outside of the grid, use
        'nearest' to find a value for them.

        Parameters
        ----------
        a : numpy.ndarray
            array to interpolate from.  It must be of size nrow, ncol
        xi : numpy.ndarray
            array containing x and y point coordinates of size (npts, 2). xi
            also works with broadcasting so that if a is a 2d array, then
            xi can be passed in as (xgrid, ygrid).
        method : {'linear', 'nearest', 'cubic'}
            method to use for interpolation (default is 'nearest')

        Returns
        -------
        b : numpy.ndarray
            array of size (npts)

        """
        try:
            from scipy.interpolate import griddata
        except:
            print('scipy not installed\ntry pip install scipy')
            return None

        # Create a 2d array of points for the grid centers
        points = np.empty((self.ncol * self.nrow, 2))
        points[:, 0] = self.get_cellcenters[0].flatten()
        points[:, 1] = self.get_cellcenters[1].flatten()

        # Use the griddata function to interpolate to the xi points
        b = griddata(points, a.flatten(), xi, method=method, fill_value=np.nan)

        # if method is linear or cubic, then replace nan's with a value
        # interpolated using nearest
        if method != 'nearest':
            bn = griddata(points, a.flatten(), xi, method='nearest')
            idx = np.isnan(b)
            b[idx] = bn[idx]

        return b

    def get_2d_vertex_connectivity(self):
        """
        Create the cell 2d vertices array and the iverts index array.  These
        are the same form as the ones used to instantiate an unstructured
        spatial reference.

        Returns
        -------

        verts : ndarray
            array of x and y coordinates for the grid vertices

        iverts : list
            a list with a list of vertex indices for each cell in clockwise
            order starting with the upper left corner

        """
        x, y = self.get_xygrid()
        x = x.flatten()
        y = y.flatten()
        nrowvert = self.nrow + 1
        ncolvert = self.ncol + 1
        npoints = nrowvert * ncolvert
        verts = np.empty((npoints, 2), dtype=np.float)
        verts[:, 0] = x
        verts[:, 1] = y
        iverts = []
        for i in range(self.nrow):
            for j in range(self.ncol):
                iv1 = i * ncolvert + j  # upper left point number
                iv2 = iv1 + 1
                iv4 = (i + 1) * ncolvert + j
                iv3 = iv4 + 1
                iverts.append([iv1, iv2, iv3, iv4])
        return verts, iverts

    def get_3d_shared_vertex_connectivity(self):

        # get the x and y points for the grid
        x, y = self.get_xygrid()
        x = x.flatten()
        y = y.flatten()

        # set the size of the vertex grid
        nrowvert = self.nrow + 1
        ncolvert = self.ncol + 1
        nlayvert = self._nlay + 1
        nrvncv = nrowvert * ncolvert
        npoints = nrvncv * nlayvert

        # create and fill a 3d points array for the grid
        verts = np.empty((npoints, 3), dtype=np.float)
        verts[:, 0] = np.tile(x, nlayvert)
        verts[:, 1] = np.tile(y, nlayvert)
        istart = 0
        istop = nrvncv
        top_botm = self.top_botm
        for k in range(self._nlay + 1):
            verts[istart:istop, 2] = self.interpolate(top_botm[k],
                                                      verts[istart:istop, :2],
                                                      method='linear')
            istart = istop
            istop = istart + nrvncv

        # create the list of points comprising each cell. points must be
        # listed a specific way according to vtk requirements.
        iverts = []
        for k in range(self._nlay):
            koffset = k * nrvncv
            for i in range(self.nrow):
                for j in range(self.ncol):
                    if self._idomain is not None:
                        if self._idomain[k, i, j] == 0:
                            continue
                    iv1 = i * ncolvert + j + koffset
                    iv2 = iv1 + 1
                    iv4 = (i + 1) * ncolvert + j + koffset
                    iv3 = iv4 + 1
                    iverts.append([iv4 + nrvncv, iv3 + nrvncv,
                                   iv1 + nrvncv, iv2 + nrvncv,
                                   iv4, iv3, iv1, iv2])

        return verts, iverts

    def get_rc(self, x, y):
        return self.get_ij(x, y)

    def get_ij(self, x, y):
        """Return the row and column of a point or sequence of points
        in real-world coordinates.

        Parameters
        ----------
        x : scalar or sequence of x coordinates
        y : scalar or sequence of y coordinates

        Returns
        -------
        i : row or sequence of rows (zero-based)
        j : column or sequence of columns (zero-based)
        """
        if np.isscalar(x):
            c = (np.abs(self.get_cellcenters[0][0] - x)).argmin()
            r = (np.abs(self.get_cellcenters[1][:, 0] - y)).argmin()
        else:
            xcp = np.array([self.get_cellcenters[0][0]] * (len(x)))
            ycp = np.array([self.get_cellcenters[1][:, 0]] * (len(x)))
            c = (np.abs(xcp.transpose() - x)).argmin(axis=0)
            r = (np.abs(ycp.transpose() - y)).argmin(axis=0)
        return r, c

    def get_3d_vertex_connectivity(self):
        if self.idomain is None:
            ncells = self._nlay * self.nrow * self.ncol
            ibound = np.ones((self._nlay, self.nrow, self.ncol), dtype=np.int)
        else:
            ncells = (self.idomain != 0).sum()
            ibound = self.idomain
        npoints = ncells * 8
        verts = np.empty((npoints, 3), dtype=np.float)
        iverts = []
        ipoint = 0
        top_botm = self.top_botm
        for k in range(self._nlay):
            for i in range(self.nrow):
                for j in range(self.ncol):
                    if ibound[k, i, j] == 0:
                        continue

                    ivert = []
                    pts = self.get_cell_vertices(i, j)
                    pt0, pt1, pt2, pt3, pt0 = pts

                    z = top_botm[k + 1, i, j]

                    verts[ipoint, 0:2] = np.array(pt1)
                    verts[ipoint, 2] = z
                    ivert.append(ipoint)
                    ipoint += 1

                    verts[ipoint, 0:2] = np.array(pt2)
                    verts[ipoint, 2] = z
                    ivert.append(ipoint)
                    ipoint += 1

                    verts[ipoint, 0:2] = np.array(pt0)
                    verts[ipoint, 2] = z
                    ivert.append(ipoint)
                    ipoint += 1

                    verts[ipoint, 0:2] = np.array(pt3)
                    verts[ipoint, 2] = z
                    ivert.append(ipoint)
                    ipoint += 1

                    z = top_botm[k, i, j]

                    verts[ipoint, 0:2] = np.array(pt1)
                    verts[ipoint, 2] = z
                    ivert.append(ipoint)
                    ipoint += 1

                    verts[ipoint, 0:2] = np.array(pt2)
                    verts[ipoint, 2] = z
                    ivert.append(ipoint)
                    ipoint += 1

                    verts[ipoint, 0:2] = np.array(pt0)
                    verts[ipoint, 2] = z
                    ivert.append(ipoint)
                    ipoint += 1

                    verts[ipoint, 0:2] = np.array(pt3)
                    verts[ipoint, 2] = z
                    ivert.append(ipoint)
                    ipoint += 1

                    iverts.append(ivert)

        return verts, iverts