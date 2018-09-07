import numpy as np
from .modelgrid import ModelGrid, CachedData, CachedDataType


class StructuredModelGrid(ModelGrid):
    """
    cell_vertices(i, j, point_type)
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
    def __init__(self, delc, delr, top=None, botm=None, idomain=None,
                 simulation_time=None, lenuni=2, sr=None, origin_loc='ul',
                 origin_x=None, origin_y=None, rotation=0.0):
        super(StructuredModelGrid, self).__init__('structured', top, botm,
                                                  idomain, sr, simulation_time,
                                                  lenuni, origin_loc, origin_x,
                                                  origin_y, rotation)
        self.__delc = delc
        self.__delr = delr
        self.__nrow = len(delc)
        self.__ncol = len(delr)
        if botm is not None:
            self.__nlay = len(botm)
        else:
            self.__nlay = None

    ####################
    # Properties
    ####################
    @property
    def extent(self):
        return (min(self.xedges), max(self.xedges),
                min(self.yedges), max(self.yedges))

    @property
    def xll(self):
        if self._origin_loc == 'll':
            xll = self._origin_x if self._origin_x is not None else 0.
        elif self._origin_loc == 'ul':
            # calculate coords for lower left corner
            xll = self._origin_x - (np.sin(self._rotation) * self.yedges[0] *
                                    self.length_multiplier)
        else:
            raise Exception('Invalid origin location "{}".'.format(
                self._origin_loc))
        return xll

    @property
    def yll(self):
        if self._origin_loc == 'll':
            yll = self._origin_y if self._origin_y is not None else 0.
        elif self._origin_loc == 'ul':
            # calculate coords for lower left corner
            yll = self._origin_y - (np.cos(self._rotation) * self.yedges[0] *
                                    self.length_multiplier)
        else:
            raise Exception('Invalid origin location "{}".'.format(
                self._origin_loc))
        return yll

    @property
    def xul(self):
        if self._origin_loc == 'll':
            # calculate coords for upper left corner
            xul = self._origin_x + (np.sin(self._rotation) * self.yedges[0] *
                                    self.length_multiplier)
        elif self._origin_loc == 'ul':
            # calculate coords for lower left corner
            xul = self._origin_x if self._origin_x is not None else 0.
        else:
            raise Exception('Invalid origin location "{}".'.format(
                self._origin_loc))

        return xul

    @property
    def yul(self):
        if self._origin_loc == 'll':
            # calculate coords for upper left corner
            yul = self._origin_y + (np.cos(self._rotation) * self.yedges[0] *
                                    self.length_multiplier)
        elif self._origin_loc == 'ul':
            # calculate coords for lower left corner
            yul = self._origin_y if self._origin_y is not None else 0.
        else:
            raise Exception('Invalid origin location "{}".'.format(
                self._origin_loc))
        return yul

    @classmethod
    def from_gridspec(cls, gridspec_file, lenuni=0):
        f = open(gridspec_file, 'r')
        raw = f.readline().strip().split()
        nrow = int(raw[0])
        ncol = int(raw[1])
        raw = f.readline().strip().split()
        xul, yul, rot = float(raw[0]), float(raw[1]), float(raw[2])
        delr = []
        j = 0
        while j < ncol:
            raw = f.readline().strip().split()
            for r in raw:
                if '*' in r:
                    rraw = r.split('*')
                    for n in range(int(rraw[0])):
                        delr.append(float(rraw[1]))
                        j += 1
                else:
                    delr.append(float(r))
                    j += 1
        delc = []
        i = 0
        while i < nrow:
            raw = f.readline().strip().split()
            for r in raw:
                if '*' in r:
                    rraw = r.split('*')
                    for n in range(int(rraw[0])):
                        delc.append(float(rraw[1]))
                        i += 1
                else:
                    delc.append(float(r))
                    i += 1
        f.close()
        return cls(np.array(delc), np.array(delr), lenuni=lenuni, origin_x=xul,
                   origin_y=yul, rotation=rot)

    @property
    def nlay(self):
        return self.__nlay

    @property
    def nrow(self):
        return self.__nrow

    @property
    def ncol(self):
        return self.__ncol

    @property
    def delc(self):
        return self.__delc

    @property
    def delr(self):
        return self.__delr

    @property
    def bounds(self):
        """Return bounding box in shapely order."""
        xmin, xmax, ymin, ymax = self.extent
        return xmin, ymin, xmax, ymax

    @property
    def grid_lines(self):
        """
        Returns a the grid line vertices as a list
        """
        xmin, xmax, ymin, ymax = self.extent
        xedge = self.xedges
        yedge = self.yedges
        lines = []

        for j in range(self.ncol + 1):
            x0 = xedge[j]
            lines.append([(x0, ymin), (x0, ymax)])

        for i in range(self.nrow + 1):
            y0 = yedge[i]
            lines.append([(xmin, y0), (xmax, y0)])

        return lines

    @property
    def xedges(self):
        """
        Return two numpy one-dimensional float arrays. One array has the cell
        edge y coordinates for every column in the grid in model space -
        """
        return self.xyedges[0]

    @property
    def yedges(self):
        """
        Return two numpy one-dimensional float arrays. One array has the cell
        edge x coordinates for every column in the grid in model space -
        """
        return self.xyedges[1]

    @property
    def xyzgrid(self):
        """
        """
        cache_index = (CachedDataType.edge_grid.value,
                       self._use_ref_coordinates)
        if cache_index not in self._cache_dict or \
                self._cache_dict[cache_index].out_of_date:
            xedge = np.concatenate(([0.], np.add.accumulate(self.__delr)))
            length_y = np.add.reduce(self.__delc)
            yedge = np.concatenate(([length_y], length_y -
                                    np.add.accumulate(self.delc)))
            xgrid, ygrid = np.meshgrid(xedge, yedge)
            zgrid, zcenter = self._zcoords()
            if self._use_ref_coordinates:
                # transform x and y
                xgrid, ygrid = self.transform(xgrid, ygrid)
            self._cache_dict[cache_index] = \
                CachedData([xgrid, ygrid, zgrid])
        return self._cache_dict[cache_index].data

    @property
    def xyedges(self):
        cache_index = (CachedDataType.edge_array.value,
                       self._use_ref_coordinates)
        if cache_index not in self._cache_dict or \
                self._cache_dict[cache_index].out_of_date:
            xedge = np.concatenate(([0.], np.add.accumulate(self.__delr)))
            length_y = np.add.reduce(self.__delc)
            yedge = np.concatenate(([length_y], length_y -
                                    np.add.accumulate(self.delc)))
            if self._use_ref_coordinates:
                # transform x and y
                xedge, yedge = self.transform(xedge, yedge)
            self._cache_dict[cache_index] = \
                CachedData([xedge, yedge])
        return self._cache_dict[cache_index].data

    #    @property
#    def xygrid(self):
#        return np.meshgrid(self.xedges, self.yedges)
        #if self.sr is not None:
        #    return self.sr.transform(xgrid, ygrid)
        #else:
        #    return xgrid, ygrid

    @property
    def xyvertices(self):
        cache_index = (CachedDataType.xyvertices.value,
                       self._use_ref_coordinates)
        if cache_index not in self._cache_dict or \
                self._cache_dict[cache_index].out_of_date:
            jj, ii = np.meshgrid(range(self.__ncol), range(self.__nrow))
            jj, ii = jj.ravel(), ii.ravel()
            self._cache_dict[cache_index] = \
                CachedData(self.cell_vertices(ii, jj))
        return self._cache_dict[cache_index].data

    @property
    def cellcenters(self):
        """
        Return a list of two numpy one-dimensional float array one with
        the cell center x coordinate and the other with the cell center y
        coordinate for every row in the grid in model space -
        not offset of rotated, with the cell center y coordinate.
        """
        cache_index = (CachedDataType.cell_centers.value,
                       self._use_ref_coordinates)
        if cache_index not in self._cache_dict or \
                self._cache_dict[cache_index].out_of_date:
            # get x centers
            x = np.add.accumulate(self.__delr) - 0.5 * self.delr
            # get y centers
            Ly = np.add.reduce(self.__delc)
            y = Ly - (np.add.accumulate(self.__delc) - 0.5 *
                      self.__delc)
            x_mesh, y_mesh = np.meshgrid(x, y)
            if self.__nlay is not None:
                # get z centers
                z = np.empty((self.__nlay, self.__nrow, self.__ncol))
                z[0, :, :] = (self._top[:, :] + self._botm[0, :, :]) / 2.
                for l in range(1, self.__nlay):
                    z[l, :, :] = (self._botm[l - 1, :, :] +
                                  self._botm[l, :, :]) / 2.
            else:
                z = None
            if self._use_ref_coordinates:
                # transform x and y
                x_mesh, y_mesh = self.transform(x_mesh, y_mesh)
            # store in cache
            self._cache_dict[cache_index] = CachedData([x_mesh, y_mesh, z])
        return self._cache_dict[cache_index].data

    @property
    def gridlines(self):
        """
            Get the grid lines as a list

        """
        # get edges initially in model coordinates
        use_ref_coords = self.use_ref_coords
        self.use_ref_coords = False
        xyedges = self.xyedges
        self.use_ref_coords = use_ref_coords

        xmin = xyedges[0][0]
        xmax = xyedges[0][-1]
        ymin = xyedges[1][-1]
        ymax = xyedges[1][0]
        lines = []
        # Vertical lines
        for j in range(self.ncol + 1):
            x0 = xyedges[0][j]
            x1 = x0
            y0 = ymin
            y1 = ymax
            lines.append([(x0, y0), (x1, y1)])

        # horizontal lines
        for i in range(self.nrow + 1):
            x0 = xmin
            x1 = xmax
            y0 = xyedges[1][i]
            y1 = y0
            lines.append([(x0, y0), (x1, y1)])

        if self._use_ref_coordinates:
            lines_trans = []
            for ln in lines:
                lines_trans.append([self.transform(*ln[0]),
                                    self.transform(*ln[1])])
            return lines_trans
        return lines

    def cell_vertices(self, i, j):
        """Get vertices for a single cell or sequence of i, j locations."""
        pts = []
        xgrid, ygrid = self.xgrid, self.ygrid
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

    def cellcenter(self, row, col):
        cell_centers = self.cellcenters
        return cell_centers[0][row], cell_centers[1][col]

    def get_extent(self):
        """
        Get the extent of the rotated and offset grid

        Return (xmin, xmax, ymin, ymax)

        """
        x0 = self.xedges[0]
        x1 = self.xedges[-1]
        y0 = self.yedges[0]
        y1 = self.yedges[-1]

        if self._use_ref_coordinates:
            # upper left point
            x0r, y0r = self.transform(x0, y0)

            # upper right point
            x1r, y1r = self.transform(x1, y0)

            # lower right point
            x2r, y2r = self.transform(x1, y1)

            # lower left point
            x3r, y3r = self.transform(x0, y1)

            xmin = min(x0r, x1r, x2r, x3r)
            xmax = max(x0r, x1r, x2r, x3r)
            ymin = min(y0r, y1r, y2r, y3r)
            ymax = max(y0r, y1r, y2r, y3r)

            return (xmin, xmax, ymin, ymax)
        else:
            return (x0, x1, y0, y1)

    def get_all_model_cells(self):
        model_cells = []
        for layer in range(0, self.__nlay):
            for row in range(0, self.__nrow):
                for column in range(0, self.__ncol):
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

    def plot(self, **kwargs):
        """
        Plot the grid lines.

        Parameters
        ----------
        kwargs : ax, colors.  The remaining kwargs are passed into the
            the LineCollection constructor.

        Returns
        -------
        lc : matplotlib.collections.LineCollection
p
        """
        from flopy.plot import PlotMapView

        mm = PlotMapView(modelgrid=self)
        return mm.plot_grid(**kwargs)

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
        points[:, 0] = self.cellcenters[0].flatten()
        points[:, 1] = self.cellcenters[1].flatten()

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
        x, y = self.xygrid
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
        x, y = self.xygrid
        x = x.flatten()
        y = y.flatten()

        # set the size of the vertex grid
        nrowvert = self.nrow + 1
        ncolvert = self.ncol + 1
        nlayvert = self.__nlay + 1
        nrvncv = nrowvert * ncolvert
        npoints = nrvncv * nlayvert

        # create and fill a 3d points array for the grid
        verts = np.empty((npoints, 3), dtype=np.float)
        verts[:, 0] = np.tile(x, nlayvert)
        verts[:, 1] = np.tile(y, nlayvert)
        istart = 0
        istop = nrvncv
        top_botm = self.top_botm
        for k in range(self.__nlay + 1):
            verts[istart:istop, 2] = self.interpolate(top_botm[k],
                                                      verts[istart:istop, :2],
                                                      method='linear')
            istart = istop
            istop = istart + nrvncv

        # create the list of points comprising each cell. points must be
        # listed a specific way according to vtk requirements.
        iverts = []
        for k in range(self.__nlay):
            koffset = k * nrvncv
            for i in range(self.nrow):
                for j in range(self.ncol):
                    if self.__idomain is not None:
                        if self.__idomain[k, i, j] == 0:
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
            c = (np.abs(self.cellcenters[0][0] - x)).argmin()
            r = (np.abs(self.cellcenters[1][:, 0] - y)).argmin()
        else:
            xcp = np.array([self.cellcenters[0][0]] * (len(x)))
            ycp = np.array([self.cellcenters[1][:, 0]] * (len(x)))
            c = (np.abs(xcp.transpose() - x)).argmin(axis=0)
            r = (np.abs(ycp.transpose() - y)).argmin(axis=0)
        return r, c

    def get_3d_vertex_connectivity(self):
        if self.idomain is None:
            ncells = self.__nlay * self.nrow * self.ncol
            ibound = np.ones((self.__nlay, self.nrow, self.ncol), dtype=np.int)
        else:
            ncells = (self.idomain != 0).sum()
            ibound = self.idomain
        npoints = ncells * 8
        verts = np.empty((npoints, 3), dtype=np.float)
        iverts = []
        ipoint = 0
        top_botm = self.top_botm
        for k in range(self.__nlay):
            for i in range(self.nrow):
                for j in range(self.ncol):
                    if ibound[k, i, j] == 0:
                        continue

                    ivert = []
                    pts = self.cell_vertices(i, j)
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