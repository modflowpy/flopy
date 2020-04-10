import copy
import numpy as np
from .grid import Grid, CachedData

def array_at_verts_basic2d(a):
    """
    Computes values at cell vertices on 2d array using neighbor averaging.

    Parameters
    ----------
    a : ndarray
        Array values at cell centers, could be a slice in any orientation.

    Returns
    -------
    averts : ndarray
        Array values at cell vertices, shape (a.shape[0]+1, a.shape[1]+1).
    """
    assert a.ndim == 2
    shape_verts2d = (a.shape[0]+1, a.shape[1]+1)

    # create a 3D array of size (nrow+1, ncol+1, 4)
    averts3d = np.full(shape_verts2d + (4,), np.nan)
    averts3d[:-1, :-1, 0] = a
    averts3d[:-1, 1:, 1] = a
    averts3d[1:, :-1, 2] = a
    averts3d[1:, 1:, 3] = a

    # calculate the mean over the last axis, ignoring NaNs
    averts = np.nanmean(averts3d, axis=2)

    return averts

class StructuredGrid(Grid):
    """
    class for a structured model grid

    Parameters
    ----------
    delc
        delc array
    delr
        delr array

    Properties
    ----------
    nlay
        returns the number of model layers
    nrow
        returns the number of model rows
    ncol
        returns the number of model columns
    delc
        returns the delc array
    delr
        returns the delr array
    xyedges
        returns x-location points for the edges of the model grid and
        y-location points for the edges of the model grid

    Methods
    ----------
    get_cell_vertices(i, j)
        returns vertices for a single cell at row, column i, j.
    """
    def __init__(self, delc=None, delr=None, top=None, botm=None, idomain=None,
                 lenuni=None, epsg=None, proj4=None, prj=None, xoff=0.0,
                 yoff=0.0, angrot=0.0, nlay=None, nrow=None, ncol=None,
                 laycbd=None):
        super(StructuredGrid, self).__init__('structured', top, botm, idomain,
                                             lenuni, epsg, proj4, prj, xoff,
                                             yoff, angrot)
        if delc is not None:
            self.__nrow = len(delc)
            self.__delc = delc.astype(float)
        else:
            self.__nrow = nrow
            self.__delc = delc
        if delr is not None:
            self.__ncol = len(delr)
            self.__delr = delr.astype(float)
        else:
            self.__ncol = ncol
            self.__delr = delr
        if top is not None:
            assert self.__nrow * self.__ncol == len(np.ravel(top))
        if botm is not None:
            assert self.__nrow * self.__ncol == len(np.ravel(botm[0]))
            if nlay is not None:
                self.__nlay = nlay
            else:
                if laycbd is not None:
                    self.__nlay = len(botm) - np.sum(laycbd>0)
                else:
                    self.__nlay = len(botm)
        else:
            self.__nlay = nlay
        if laycbd is not None:
            self.__laycbd = laycbd
        else:
            self.__laycbd = np.zeros(self.__nlay, dtype=int)

    ####################
    # Properties
    ####################
    @property
    def is_valid(self):
        if self.__delc is not None and self.__delr is not None:
            return True
        return False

    @property
    def is_complete(self):
        if self.__delc is not None and self.__delr is not None and \
                super(StructuredGrid, self).is_complete:
            return True
        return False

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
    def nnodes(self):
        return self.__nlay * self.__nrow * self.__ncol

    @property
    def shape(self):
        return self.__nlay, self.__nrow, self.__ncol

    @property
    def extent(self):
        self._copy_cache = False
        xyzgrid = self.xyzvertices
        self._copy_cache = True
        return (np.min(xyzgrid[0]), np.max(xyzgrid[0]),
                np.min(xyzgrid[1]), np.max(xyzgrid[1]))

    @property
    def delc(self):
        return copy.deepcopy(self.__delc)

    @property
    def delr(self):
        return copy.deepcopy(self.__delr)

    @property
    def delz(self):
        cache_index = 'delz'
        if cache_index not in self._cache_dict or \
                self._cache_dict[cache_index].out_of_date:
            delz = self.top_botm[:-1, :, :] - self.top_botm[1:, :, :]
            self._cache_dict[cache_index] = CachedData(delz)
        if self._copy_cache:
            return self._cache_dict[cache_index].data
        else:
            return self._cache_dict[cache_index].data_nocopy

    @property
    def top_botm_withnan(self):
        """
        Same as top_botm array but with NaN where idomain==0 both above and
        below a cell.
        """
        cache_index = 'top_botm_withnan'
        if cache_index not in self._cache_dict or \
                self._cache_dict[cache_index].out_of_date:
            is_inactive_above = np.full(self.top_botm.shape, True)
            is_inactive_above[:-1, :, :] = self._idomain==0
            is_inactive_below = np.full(self.top_botm.shape, True)
            is_inactive_below[1:, :, :] = self._idomain==0
            where_to_nan = np.logical_and(is_inactive_above, is_inactive_below)
            top_botm_withnan = np.where(where_to_nan, np.nan, self.top_botm)
            self._cache_dict[cache_index] = CachedData(top_botm_withnan)
        if self._copy_cache:
            return self._cache_dict[cache_index].data
        else:
            return self._cache_dict[cache_index].data_nocopy

    @property
    def xyzvertices(self):
        """
        Method to get all grid vertices in a layer

        Returns:
            []
            2D array
        """
        cache_index = 'xyzgrid'
        if cache_index not in self._cache_dict or \
                self._cache_dict[cache_index].out_of_date:
            xedge = np.concatenate(([0.], np.add.accumulate(self.__delr)))
            length_y = np.add.reduce(self.__delc)
            yedge = np.concatenate(([length_y], length_y -
                                    np.add.accumulate(self.delc)))
            xgrid, ygrid = np.meshgrid(xedge, yedge)
            zgrid, zcenter = self._zcoords()
            if self._has_ref_coordinates:
                # transform x and y
                pass
            xgrid, ygrid = self.get_coords(xgrid, ygrid)
            if zgrid is not None:
                self._cache_dict[cache_index] = \
                    CachedData([xgrid, ygrid, zgrid])
            else:
                self._cache_dict[cache_index] = \
                    CachedData([xgrid, ygrid])

        if self._copy_cache:
            return self._cache_dict[cache_index].data
        else:
            return self._cache_dict[cache_index].data_nocopy

    @property
    def xyedges(self):
        """
        Return a list of two 1D numpy arrays: one with the cell edge x
        coordinate (size = ncol+1) and the other with the cell edge y
        coordinate (size = nrow+1) in model space - not offset or rotated.
        """
        cache_index = 'xyedges'
        if cache_index not in self._cache_dict or \
                self._cache_dict[cache_index].out_of_date:
            xedge = np.concatenate(([0.], np.add.accumulate(self.__delr)))
            length_y = np.add.reduce(self.__delc)
            yedge = np.concatenate(([length_y], length_y -
                                    np.add.accumulate(self.delc)))
            self._cache_dict[cache_index] = \
                CachedData([xedge, yedge])
        if self._copy_cache:
            return self._cache_dict[cache_index].data
        else:
            return self._cache_dict[cache_index].data_nocopy

    @property
    def zedges(self):
        """
        Return zedges for (column, row)==(0, 0).
        """
        cache_index = 'zedges'
        if cache_index not in self._cache_dict or \
                self._cache_dict[cache_index].out_of_date:
            zedge = np.concatenate((np.array([self.top[0, 0]]),
                                    self.botm[:, 0, 0]))
            self._cache_dict[cache_index] = \
                CachedData(zedge)
        if self._copy_cache:
            return self._cache_dict[cache_index].data
        else:
            return self._cache_dict[cache_index].data_nocopy

    @property
    def zverts_smooth(self):
        """
        Get a unique z of cell vertices for smooth (instead of stepwise) layer
        elevations using bilinear interpolation.

        Returns
        -------
        zverts : ndarray, shape (nlay+1, nrow+1, ncol+1)
            z of cell vertices. NaN values are assigned in accordance with
            inactive cells defined by idomain.
        """
        cache_index = 'zverts_smooth'
        if cache_index not in self._cache_dict or \
                self._cache_dict[cache_index].out_of_date:
            zverts_smooth = self._zverts_smooth()
            self._cache_dict[cache_index] = CachedData(zverts_smooth)
        if self._copy_cache:
            return self._cache_dict[cache_index].data
        else:
            return self._cache_dict[cache_index].data_nocopy

    def _zverts_smooth(self):
        """
        For internal use only. The user should call zverts_smooth.
        """
        # initialize the result array
        shape_verts = (self.nlay+1, self.nrow+1, self.ncol+1)
        zverts_basic = np.empty(shape_verts, dtype='float64')

        # assign NaN to top_botm where idomain==0 both above and below
        if self._idomain is not None:
            _top_botm = self.top_botm_withnan

        # perform basic interpolation (this will be useful in all cases)
        # loop through layers
        for k in range(self.nlay+1):
            zvertsk = array_at_verts_basic2d(_top_botm[k, : , :])
            zverts_basic[k, : , :] = zvertsk

        if self.is_regular():
            # if the grid is regular, basic interpolation is the correct one
            zverts = zverts_basic
        else:
            # cell centers
            xcenters, ycenters = self.get_local_coords(self.xcellcenters,
                                                       self.ycellcenters)
            # flip y direction because RegularGridInterpolator requires
            # increasing input coordinates
            ycenters = np.flip(ycenters, axis=0)
            _top_botm = np.flip(_top_botm, axis=1)
            xycenters = (ycenters[:, 0], xcenters[0, :])

            # vertices
            xverts, yverts = self.get_local_coords(self.xvertices,
                                                     self.yvertices)
            xyverts = np.ndarray((xverts.size, 2))
            xyverts[:, 0] = yverts.ravel()
            xyverts[:, 1] = xverts.ravel()

            # interpolate
            import scipy.interpolate as interp
            shape_verts2d = (self.nrow+1, self.ncol+1)
            zverts = np.empty(shape_verts, dtype='float64')
            # loop through layers
            for k in range(self.nlay+1):
                # interpolate layer elevation
                zcenters_k = _top_botm[k, : , :]
                interp_func = interp.RegularGridInterpolator(xycenters,
                    zcenters_k, bounds_error=False, fill_value=np.nan)
                zverts_k = interp_func(xyverts)
                zverts_k = zverts_k.reshape(shape_verts2d)
                zverts[k, : , :] = zverts_k

            # use basic interpolation for remaining NaNs at boundaries
            where_nan = np.isnan(zverts)
            zverts[where_nan] = zverts_basic[where_nan]

        return zverts

    @property
    def xyzcellcenters(self):
        """
        Return a list of two numpy one-dimensional float array one with
        the cell center x coordinate and the other with the cell center y
        coordinate for every row in the grid in model space -
        not offset of rotated, with the cell center y coordinate.
        """
        cache_index = 'cellcenters'
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
                ibs = np.arange(self.__nlay)
                quasi3d = [cbd !=0 for cbd in self.__laycbd]
                if np.any(quasi3d):
                    ibs[1:] = ibs[1:] + np.cumsum(quasi3d)[:self.__nlay - 1]
                for l, ib in enumerate(ibs[1:], 1):
                    z[l, :, :] = (self._botm[ib - 1, :, :] +
                                  self._botm[ib, :, :]) / 2.
            else:
                z = None
            if self._has_ref_coordinates:
                # transform x and y
                x_mesh, y_mesh = self.get_coords(x_mesh, y_mesh)
            # store in cache
            self._cache_dict[cache_index] = CachedData([x_mesh, y_mesh, z])
        if self._copy_cache:
            return self._cache_dict[cache_index].data
        else:
            return self._cache_dict[cache_index].data_nocopy

    @property
    def grid_lines(self):
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

        if self._has_ref_coordinates:
            lines_trans = []
            for ln in lines:
                lines_trans.append([self.get_coords(*ln[0]),
                                    self.get_coords(*ln[1])])
            return lines_trans
        return lines

    ###############
    ### Methods ###
    ###############
    def intersect(self, x, y, local=False, forgive=False):
        """
        Get the row and column of a point with coordinates x and y

        When the point is on the edge of two cells, the cell with the lowest
        row or column is returned.

        Parameters
        ----------
        x : float
            The x-coordinate of the requested point
        y : float
            The y-coordinate of the requested point
        local: bool (optional)
            If True, x and y are in local coordinates (defaults to False)
        forgive: bool (optional)
            Forgive x,y arguments that fall outside the model grid and
            return NaNs instead (defaults to False - will throw exception)

        Returns
        -------
        row : int
            The row number
        col : int
            The column number

        """
        # transform x and y to local coordinates
        x, y = super(StructuredGrid, self).intersect(x, y, local, forgive)

        # get the cell edges in local coordinates
        xe, ye = self.xyedges

        xcomp = x > xe
        if np.all(xcomp) or not np.any(xcomp):
            if forgive:
                col = np.nan
            else:
                raise Exception(
                    'x, y point given is outside of the model area')
        else:
            col = np.where(xcomp)[0][-1]

        ycomp = y < ye
        if np.all(ycomp) or not np.any(ycomp):
            if forgive:
                row = np.nan
            else:
                raise Exception(
                    'x, y point given is outside of the model area')
        else:
            row = np.where(ycomp)[0][-1]
        if np.any(np.isnan([row, col])):
            row = col = np.nan
        return row, col

    def _cell_vert_list(self, i, j):
        """Get vertices for a single cell or sequence of i, j locations."""
        self._copy_cache = False
        pts = []
        xgrid, ygrid = self.xvertices, self.yvertices
        pts.append([xgrid[i, j], ygrid[i, j]])
        pts.append([xgrid[i + 1, j], ygrid[i + 1, j]])
        pts.append([xgrid[i + 1, j + 1], ygrid[i + 1, j + 1]])
        pts.append([xgrid[i, j + 1], ygrid[i, j + 1]])
        pts.append([xgrid[i, j], ygrid[i, j]])
        self._copy_cache = True
        if np.isscalar(i):
            return pts
        else:
            vrts = np.array(pts).transpose([2, 0, 1])
            return [v.tolist() for v in vrts]

    def get_cell_vertices(self, i, j):
        """
        Method to get a set of cell vertices for a single cell
            used in the Shapefile export utilities
        :param i: (int) cell row number
        :param j: (int) cell column number
        :return: list of x,y cell vertices
        """
        self._copy_cache = False
        cell_verts = [(self.xvertices[i, j], self.yvertices[i, j]),
                      (self.xvertices[i, j+1], self.yvertices[i, j+1]),
                      (self.xvertices[i+1, j+1], self.yvertices[i+1, j+1]),
                      (self.xvertices[i+1, j], self.yvertices[i+1, j]),]
        self._copy_cache = True
        return cell_verts

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

        """
        from ..plot import PlotMapView

        mm = PlotMapView(modelgrid=self)
        return mm.plot_grid(**kwargs)

    # Importing
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
        grd = cls(np.array(delc), np.array(delr), lenuni=lenuni)
        xll = grd._xul_to_xll(xul)
        yll = grd._yul_to_yll(yul)
        cls.set_coord_info(xoff=xll, yoff=yll, angrot=rot)
        return cls

    # Exporting
    def write_shapefile(self, filename='grid.shp', epsg=None, prj=None):
        """
        Write a shapefile of the grid with just the row and column attributes.
        """
        from ..export.shapefile_utils import write_grid_shapefile
        if epsg is None and prj is None:
            epsg = self.epsg
        write_grid_shapefile(filename, self, array_dict={}, nan_val=-1.0e9,
                             epsg=epsg, prj=prj)

    def is_regular(self):
        """
        Test whether the grid spacing is regular or not (including in the
        vertical direction).
        """
        # Relative tolerance to use in test
        rel_tol = 1.e-5

        # Regularity test in x direction
        rel_diff_x = (self.delr - self.delr[0]) / self.delr[0]
        is_regular_x = np.count_nonzero(rel_diff_x > rel_tol) == 0

        # Regularity test in y direction
        rel_diff_y = (self.delc - self.delc[0]) / self.delc[0]
        is_regular_y = np.count_nonzero(rel_diff_y > rel_tol) == 0

        # Regularity test in z direction
        rel_diff_thick0 = (self.delz[0, :, :] - self.delz[0, 0, 0]) \
            / self.delz[0, 0, 0]
        failed = np.abs(rel_diff_thick0) > rel_tol
        is_regular_z = np.count_nonzero(failed) == 0
        for k in range(1, self.nlay):
            rel_diff_zk = (self.delz[k, :, :] - self.delz[0, :, :]) \
                / self.delz[0, :, :]
            failed = np.abs(rel_diff_zk) > rel_tol
            is_regular_z = is_regular_z and np.count_nonzero(failed) == 0

        return is_regular_x and is_regular_y and is_regular_z

    def is_rectilinear(self):
        """
        Test whether the grid is rectilinear (it is always so in the x and
        y directions, but not necessarily in the z direction).
        """
        # Relative tolerance to use in test
        rel_tol = 1.e-5

        # Rectilinearity test in z direction
        is_rect_z = True
        for k in range(self.nlay):
            rel_diff_zk = (self.delz[k, :, :] - self.delz[k, 0, 0]) \
                / self.delz[k, 0, 0]
            failed = np.abs(rel_diff_zk) > rel_tol
            is_rect_z = is_rect_z and np.count_nonzero(failed) == 0

        return is_rect_z

    def array_at_verts_basic(self, a):
        """
        Computes values at cell vertices using neighbor averaging.

        Parameters
        ----------
        a : ndarray
            Array values at cell centers.

        Returns
        -------
        averts : ndarray
            Array values at cell vertices, shape
            (a.shape[0]+1, a.shape[1]+1, a.shape[2]+1). NaN values are assigned
            in accordance with inactive cells defined by idomain.
        """
        assert a.ndim == 3
        shape_verts = (a.shape[0]+1, a.shape[1]+1, a.shape[2]+1)

        # set to NaN where idomain==0
        a[self._idomain==0] = np.nan

        # create a 4D array of size (nlay+1, nrow+1, ncol+1, 8)
        averts4d = np.full(shape_verts + (8,), np.nan)
        averts4d[:-1, :-1, :-1, 0] = a
        averts4d[:-1, :-1, 1:, 1] = a
        averts4d[:-1, 1:, :-1, 2] = a
        averts4d[:-1, 1:, 1:, 3] = a
        averts4d[1:, :-1, :-1, 4] = a
        averts4d[1:, :-1, 1:, 5] = a
        averts4d[1:, 1:, :-1, 6] = a
        averts4d[1:, 1:, 1:, 7] = a

        # calculate the mean over the last axis, ignoring NaNs
        averts = np.nanmean(averts4d, axis=3)

        return averts

    def array_at_verts(self, a):
        """
        Computes values at cell vertices using trilinear interpolation.

        Parameters
        ----------
        a : ndarray
            Array values. Allowed shapes are: (nlay, nrow, ncol),
            (nlay, nrow, ncol+1), (nlay, nrow+1, ncol) and
            (nlay+1, nrow, ncol).
            * When the shape is (nlay, nrow, ncol), the input values are
            considered at cell centers.
            * When the shape is extended in one direction, the input values are
            considered at the center of cell faces in this direction.

        Returns
        -------
        averts : ndarray
            Array values interpolated at cell vertices, shape
            (nlay+1, nrow+1, ncol+1). NaN values are assigned in accordance
            with inactive cells defined by idomain.
        """
        # define shapes
        shape_ext_x = (self.nlay, self.nrow, self.ncol+1)
        shape_ext_y = (self.nlay, self.nrow+1, self.ncol)
        shape_ext_z = (self.nlay+1, self.nrow, self.ncol)
        shape_verts = (self.nlay+1, self.nrow+1, self.ncol+1)

        # perform basic interpolation (this will be useful in all cases)
        if a.shape == self.shape:
            averts_basic = self.array_at_verts_basic(a)
        elif a.shape == shape_ext_x:
            averts_basic = np.empty(shape_verts, dtype=a.dtype)
            for j in range(self.ncol+1):
                averts_basic[:, :, j] = array_at_verts_basic2d(a[:, :, j])
        elif a.shape == shape_ext_y:
            averts_basic = np.empty(shape_verts, dtype=a.dtype)
            for i in range(self.nrow+1):
                averts_basic[:, i, :] = array_at_verts_basic2d(a[:, i, :])
        elif a.shape == shape_ext_z:
            averts_basic = np.empty(shape_verts, dtype=a.dtype)
            for k in range(self.nlay+1):
                averts_basic[k, :, :] = array_at_verts_basic2d(a[k, :, :])

        if self.is_regular():
            # if the grid is regular, basic interpolation is the correct one
            averts = averts_basic
        else:
            # get input coordinates
            xcenters, ycenters = self.get_local_coords(self.xcellcenters,
                                                       self.ycellcenters)
            zcenters = self.zcellcenters
            if a.shape == self.shape:
                xinput = xcenters * np.ones(self.shape)
                yinput = ycenters * np.ones(self.shape)
                zinput = zcenters
                # set array to NaN where inactive
                if self._idomain is not None:
                    a = np.where(self._idomain == 0, np.nan, a)
            elif a.shape == shape_ext_x:
                xinput = np.reshape(self.xyedges[0], (1, 1, self.ncol+1))
                xinput = xinput * np.ones(shape_ext_x)
                yinput = np.reshape(self.ycellcenters[:, 0], (1, self.nrow, 1))
                yinput = yinput * np.ones(shape_ext_x)
                zinput = self.array_at_faces(zcenters, 'x', withnan=False)
            elif a.shape == shape_ext_y:
                xinput = np.reshape(self.xcellcenters[0, :], (1, 1, self.ncol))
                xinput = xinput * np.ones(shape_ext_y)
                yinput = np.reshape(self.xyedges[1], (1, self.nrow+1, 1))
                yinput = yinput * np.ones(shape_ext_y)
                zinput = self.array_at_faces(zcenters, 'y', withnan=False)
            elif a.shape == shape_ext_z:
                xinput = xcenters * np.ones(shape_ext_z)
                yinput = ycenters * np.ones(shape_ext_z)
                zinput = self.top_botm
            else:
                raise ValueError('Incompatible array shape')

            # flip y and z directions because RegularGridInterpolator requires
            # increasing input coordinates
            xinput = np.flip(xinput, axis=[0, 1])
            yinput = np.flip(yinput, axis=[0, 1])
            zinput = np.flip(zinput, axis=[0, 1])
            _a = np.flip(a, axis=[0, 1])

            # get output coordinates (i.e. vertices)
            xoutput, youtput = self.get_local_coords(self.xvertices,
                                                     self.yvertices)
            xoutput = xoutput * np.ones(shape_verts)
            youtput = youtput * np.ones(shape_verts)
            zoutput = self.zverts_smooth
            xyzoutput = np.ndarray((zoutput.size, 3))
            xyzoutput[:, 0] = zoutput.ravel()
            xyzoutput[:, 1] = youtput.ravel()
            xyzoutput[:, 2] = xoutput.ravel()

            # interpolate
            import scipy.interpolate as interp
            if self.is_rectilinear():
                xyzinput = (zinput[:, 0, 0], yinput[0, :, 0], xinput[0, 0, :])
                interp_func = interp.RegularGridInterpolator(xyzinput, _a,
                    bounds_error=False, fill_value=np.nan)
            else:
                # format inputs, excluding NaN
                valid_input = np.logical_not(np.isnan(_a))
                xyzinput = np.ndarray((np.count_nonzero(valid_input), 3))
                xyzinput[:, 0] = zinput[valid_input]
                xyzinput[:, 1] = yinput[valid_input]
                xyzinput[:, 2] = xinput[valid_input]
                _a = _a[valid_input]
                interp_func = interp.LinearNDInterpolator(xyzinput, _a,
                                                          fill_value=np.nan)
            averts = interp_func(xyzoutput)
            averts = averts.reshape(shape_verts)

            # use basic interpolation for remaining NaNs at boundaries
            where_nan = np.isnan(averts)
            averts[where_nan] = averts_basic[where_nan]

            # assign NaN where idomain==0 at all 8 neighbors (these should be
            # the same locations as in averts_basic)
            averts[np.isnan(averts_basic)] = np.nan

        return averts

    def array_at_faces(self, a, direction, withnan=True):
        """
        Computes values at the center of cell faces using linear interpolation.

        Parameters
        ----------
        a : ndarray
            Values at cell centers, shape (nlay, row, ncol).
        direction : str, possible values are 'x', 'y' and 'z'
            Direction in which values will be interpolated at cell faces.
        withnan : bool
            If True (default), the result value will be set to NaN where the
            cell face sits between inactive cells. If False, not.

        Returns
        -------
        afaces : ndarray
            Array values interpolated at cell vertices, shape as input extended
            by 1 along the specified direction.

        """
        assert a.shape == self.shape

        # get the dimension that corresponds to the direction
        dir_to_dim = {'x': 2, 'y': 1, 'z': 0}
        dim = dir_to_dim[direction]

        # extended array with ghost cells on both sides having zero values
        ghost_shape = list(a.shape)
        ghost_shape[dim] += 2
        a_ghost = np.zeros(ghost_shape, dtype=a.dtype)

        # extended delta with ghost cells on both sides having zero values
        delta_ghost = np.zeros(ghost_shape, dtype=a.dtype)

        # inactive bool array
        if withnan and self._idomain is not None:
            inactive = self._idomain == 0

        if dim == 0:
            # fill array with ghost cells
            a_ghost[1:-1, :, :] = a
            a_ghost[0, :, :] = a[0, :, :]
            a_ghost[-1, :, :] = a[-1, :, :]

            # calculate weights
            delta_ghost[1:-1, :, :] = self.delz
            weight2 = delta_ghost[:-1, :, :] / (delta_ghost[:-1, :, :] + \
                                                delta_ghost[1:, :, :])
            weight1 = 1. - weight2

            # interpolate
            afaces = a_ghost[:-1, :, :]*weight1 + a_ghost[1:, :, :]*weight2

            # assign NaN where idomain==0 on both sides
            if withnan and self._idomain is not None:
                inactive_faces = np.full(afaces.shape, True)
                inactive_faces[:-1, :, :] = np.logical_and(
                    inactive_faces[:-1, :, :], inactive)
                inactive_faces[1:, :, :] = np.logical_and(
                    inactive_faces[1:, :, :], inactive)
                afaces[inactive_faces] = np.nan

        elif dim == 1:
            # fill array with ghost cells
            a_ghost[:, 1:-1, :] = a
            a_ghost[:, 0, :] = a[:, 0, :]
            a_ghost[:, -1, :] = a[:, -1, :]

            # calculate weights
            delc = np.reshape(self.delc, (1, self.nrow, 1))
            delc_3D = delc * np.ones(self.shape)
            delta_ghost[:, 1:-1, :] = delc_3D
            weight2 = delta_ghost[:, :-1, :] / (delta_ghost[:, :-1, :] + \
                                                delta_ghost[:, 1:, :])
            weight1 = 1. - weight2

            # interpolate
            afaces = a_ghost[:, :-1, :]*weight1 + a_ghost[:, 1:, :]*weight2

            # assign NaN where idomain==0 on both sides
            if withnan and self._idomain is not None:
                inactive_faces = np.full(afaces.shape, True)
                inactive_faces[:, :-1, :] = np.logical_and(
                    inactive_faces[:, :-1, :], inactive)
                inactive_faces[:, 1:, :] = np.logical_and(
                    inactive_faces[:, 1:, :], inactive)
                afaces[inactive_faces] = np.nan

        elif dim == 2:
            # fill array with ghost cells
            a_ghost[:, :, 1:-1] = a
            a_ghost[:, :, 0] = a[:, :, 0]
            a_ghost[:, :, -1] = a[:, :, -1]

            # calculate weights
            delr = np.reshape(self.delr, (1, 1, self.ncol))
            delr_3D = delr * np.ones(self.shape)
            delta_ghost[:, :, 1:-1] = delr_3D
            weight2 = delta_ghost[:, :, :-1] / (delta_ghost[:, :, :-1] + \
                                                delta_ghost[:, :, 1:])
            weight1 = 1. - weight2

            # interpolate
            afaces = a_ghost[:, :, :-1]*weight1 + a_ghost[:, :, 1:]*weight2

            # assign NaN where idomain==0 on both sides
            if withnan and self._idomain is not None:
                inactive_faces = np.full(afaces.shape, True)
                inactive_faces[:, :, :-1] = np.logical_and(
                    inactive_faces[:, :, :-1], inactive)
                inactive_faces[:, :, 1:] = np.logical_and(
                    inactive_faces[:, :, 1:], inactive)
                afaces[inactive_faces] = np.nan

        return afaces

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    delc = np.ones((10,)) * 1
    delr = np.ones((20,)) * 1

    top = np.ones((10, 20)) * 2000
    botm = np.ones((1, 10, 20)) * 1100

    t = StructuredGrid(delc, delr, top, botm, xoff=0, yoff=0,
                       angrot=45)

    #plt.scatter(np.ravel(t.xcenters), np.ravel(t.ycenters), c="b")
    #t.plot_grid_lines()
    #plt.show()
    #plt.close()

    #delc = np.ones(10,) * 2
    #t.delc = delc

    #plt.scatter(np.ravel(t.xcenters), np.ravel(t.ycenters), c="b")
    #t.plot_grid_lines()
    #plt.show()

    t.use_ref_coords = False
    x = t.xvertices
    y = t.yvertices
    xc = t.xcellcenters
    yc = t.ycellcenters
    #extent = t.extent
    grid = t.grid_lines

    t.use_ref_coords = True
    sr_x = t.xvertices
    sr_y = t.yvertices
    sr_xc = t.xcellcenters
    sr_yc = t.ycellcenters
    #sr_extent = t.extent
    sr_grid = t.grid_lines
    print(sr_grid)
    #t.plot_grid_lines()
    #plt.show()
