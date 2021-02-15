import copy
import numpy as np
from .grid import Grid, CachedData

try:
    from numpy.lib import NumpyVersion

    numpy115 = NumpyVersion(np.__version__) >= "1.15.0"
except ImportError:
    numpy115 = False

if not numpy115:

    def flip_numpy115(m, axis=None):
        """Provide same behavior for np.flip since numpy 1.15.0."""
        import numpy.core.numeric as _nx
        from numpy.core.numeric import asarray

        if not hasattr(m, "ndim"):
            m = asarray(m)
        if axis is None:
            indexer = (np.s_[::-1],) * m.ndim
        else:
            axis = _nx.normalize_axis_tuple(axis, m.ndim)
            indexer = [np.s_[:]] * m.ndim
            for ax in axis:
                indexer[ax] = np.s_[::-1]
            indexer = tuple(indexer)
        return m[indexer]

    np.flip = flip_numpy115


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
    shape_verts2d = (a.shape[0] + 1, a.shape[1] + 1)

    # create a 3D array of size (nrow+1, ncol+1, 4)
    averts3d = np.full(shape_verts2d + (4,), np.nan)
    averts3d[:-1, :-1, 0] = a
    averts3d[:-1, 1:, 1] = a
    averts3d[1:, :-1, 2] = a
    averts3d[1:, 1:, 3] = a

    # calculate the mean over the last axis, ignoring NaNs
    averts = np.nanmean(averts3d, axis=2)

    return averts


def array_at_faces_1d(a, delta):
    """
    Interpolate array at cell faces of a 1d grid using linear interpolation.

    Parameters
    ----------
    a : 1d ndarray
        Values at cell centers.
    delta : 1d ndarray
        Grid steps.

    Returns
    -------
    afaces : 1d ndarray
        Array values interpolated at cell faces, shape as input extended by 1.

    """
    # extended array with ghost cells on both sides having zero values
    ghost_shape = list(a.shape)
    ghost_shape[0] += 2
    a_ghost = np.zeros(ghost_shape, dtype=a.dtype)

    # extended delta with ghost cells on both sides having zero values
    delta_ghost = np.zeros(ghost_shape, dtype=a.dtype)

    # fill array with ghost cells
    a_ghost[1:-1] = a
    a_ghost[0] = a[0]
    a_ghost[-1] = a[-1]

    # calculate weights
    delta_ghost[1:-1] = delta
    weight2 = delta_ghost[:-1] / (delta_ghost[:-1] + delta_ghost[1:])
    weight1 = 1.0 - weight2

    # interpolate
    afaces = a_ghost[:-1] * weight1 + a_ghost[1:] * weight2

    return afaces


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

    def __init__(
        self,
        delc=None,
        delr=None,
        top=None,
        botm=None,
        idomain=None,
        lenuni=None,
        epsg=None,
        proj4=None,
        prj=None,
        xoff=0.0,
        yoff=0.0,
        angrot=0.0,
        nlay=None,
        nrow=None,
        ncol=None,
        laycbd=None,
    ):
        super(StructuredGrid, self).__init__(
            "structured",
            top,
            botm,
            idomain,
            lenuni,
            epsg,
            proj4,
            prj,
            xoff,
            yoff,
            angrot,
        )
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
                    self.__nlay = len(botm) - np.sum(laycbd > 0)
                else:
                    self.__nlay = len(botm)
        else:
            self.__nlay = nlay
        if laycbd is not None:
            self.__laycbd = laycbd
        else:
            self.__laycbd = np.zeros(self.__nlay or (), dtype=int)

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
        if (
            self.__delc is not None
            and self.__delr is not None
            and super(StructuredGrid, self).is_complete
        ):
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
        return (
            np.min(xyzgrid[0]),
            np.max(xyzgrid[0]),
            np.min(xyzgrid[1]),
            np.max(xyzgrid[1]),
        )

    @property
    def delc(self):
        return copy.deepcopy(self.__delc)

    @property
    def delr(self):
        return copy.deepcopy(self.__delr)

    @property
    def delz(self):
        cache_index = "delz"
        if (
            cache_index not in self._cache_dict
            or self._cache_dict[cache_index].out_of_date
        ):
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
        cache_index = "top_botm_withnan"
        if (
            cache_index not in self._cache_dict
            or self._cache_dict[cache_index].out_of_date
        ):
            is_inactive_above = np.full(self.top_botm.shape, True)
            is_inactive_above[:-1, :, :] = self._idomain == 0
            is_inactive_below = np.full(self.top_botm.shape, True)
            is_inactive_below[1:, :, :] = self._idomain == 0
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
        cache_index = "xyzgrid"
        if (
            cache_index not in self._cache_dict
            or self._cache_dict[cache_index].out_of_date
        ):
            xedge = np.concatenate(([0.0], np.add.accumulate(self.__delr)))
            length_y = np.add.reduce(self.__delc)
            yedge = np.concatenate(
                ([length_y], length_y - np.add.accumulate(self.delc))
            )
            xgrid, ygrid = np.meshgrid(xedge, yedge)
            zgrid, zcenter = self._zcoords()
            if self._has_ref_coordinates:
                # transform x and y
                pass
            xgrid, ygrid = self.get_coords(xgrid, ygrid)
            if zgrid is not None:
                self._cache_dict[cache_index] = CachedData(
                    [xgrid, ygrid, zgrid]
                )
            else:
                self._cache_dict[cache_index] = CachedData([xgrid, ygrid])

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
        cache_index = "xyedges"
        if (
            cache_index not in self._cache_dict
            or self._cache_dict[cache_index].out_of_date
        ):
            xedge = np.concatenate(([0.0], np.add.accumulate(self.__delr)))
            length_y = np.add.reduce(self.__delc)
            yedge = np.concatenate(
                ([length_y], length_y - np.add.accumulate(self.delc))
            )
            self._cache_dict[cache_index] = CachedData([xedge, yedge])
        if self._copy_cache:
            return self._cache_dict[cache_index].data
        else:
            return self._cache_dict[cache_index].data_nocopy

    @property
    def zedges(self):
        """
        Return zedges for (column, row)==(0, 0).
        """
        cache_index = "zedges"
        if (
            cache_index not in self._cache_dict
            or self._cache_dict[cache_index].out_of_date
        ):
            zedges = np.concatenate(
                (np.array([self.top[0, 0]]), self.botm[:, 0, 0])
            )
            self._cache_dict[cache_index] = CachedData(zedges)
        if self._copy_cache:
            return self._cache_dict[cache_index].data
        else:
            return self._cache_dict[cache_index].data_nocopy

    @property
    def zverts_smooth(self):
        """
        Get a unique z of cell vertices using bilinear interpolation of top and
        bottom elevation layers.

        Returns
        -------
        zverts : ndarray, shape (nlay+1, nrow+1, ncol+1)
            z of cell vertices. NaN values are assigned in accordance with
            inactive cells defined by idomain.
        """
        cache_index = "zverts_smooth"
        if (
            cache_index not in self._cache_dict
            or self._cache_dict[cache_index].out_of_date
        ):
            zverts_smooth = self.array_at_verts(self.top_botm)
            self._cache_dict[cache_index] = CachedData(zverts_smooth)
        if self._copy_cache:
            return self._cache_dict[cache_index].data
        else:
            return self._cache_dict[cache_index].data_nocopy

    @property
    def xycenters(self):
        """
        Return a list of two numpy one-dimensional float arrays for center x
        and y coordinates in model space - not offset or rotated.
        """
        cache_index = "xycenters"
        if (
            cache_index not in self._cache_dict
            or self._cache_dict[cache_index].out_of_date
        ):
            # get x centers
            x = np.add.accumulate(self.__delr) - 0.5 * self.delr
            # get y centers
            Ly = np.add.reduce(self.__delc)
            y = Ly - (np.add.accumulate(self.__delc) - 0.5 * self.__delc)
            # store in cache
            self._cache_dict[cache_index] = CachedData([x, y])
        if self._copy_cache:
            return self._cache_dict[cache_index].data
        else:
            return self._cache_dict[cache_index].data_nocopy

    @property
    def xyzcellcenters(self):
        """
        Return a list of three numpy float arrays: two two-dimensional arrays
        for center x and y coordinates, and one three-dimensional array for
        center z coordinates. Coordinates are given in real-world coordinates.
        """
        cache_index = "cellcenters"
        if (
            cache_index not in self._cache_dict
            or self._cache_dict[cache_index].out_of_date
        ):
            # get x centers
            x = np.add.accumulate(self.__delr) - 0.5 * self.delr
            # get y centers
            Ly = np.add.reduce(self.__delc)
            y = Ly - (np.add.accumulate(self.__delc) - 0.5 * self.__delc)
            x_mesh, y_mesh = np.meshgrid(x, y)
            if self.__nlay is not None:
                # get z centers
                z = np.empty((self.__nlay, self.__nrow, self.__ncol))
                z[0, :, :] = (self._top[:, :] + self._botm[0, :, :]) / 2.0
                ibs = np.arange(self.__nlay)
                quasi3d = [cbd != 0 for cbd in self.__laycbd]
                if np.any(quasi3d):
                    ibs[1:] = ibs[1:] + np.cumsum(quasi3d)[: self.__nlay - 1]
                for l, ib in enumerate(ibs[1:], 1):
                    z[l, :, :] = (
                        self._botm[ib - 1, :, :] + self._botm[ib, :, :]
                    ) / 2.0
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
                lines_trans.append(
                    [self.get_coords(*ln[0]), self.get_coords(*ln[1])]
                )
            return lines_trans
        return lines

    @property
    def is_regular_x(self):
        """
        Test whether the grid spacing is regular in the x direction.
        """
        cache_index = "is_regular_x"
        if (
            cache_index not in self._cache_dict
            or self._cache_dict[cache_index].out_of_date
        ):
            # relative tolerance to use in test
            rel_tol = 1.0e-5

            # regularity test in x direction
            rel_diff_x = (self.__delr - self.__delr[0]) / self.__delr[0]
            is_regular_x = np.count_nonzero(np.abs(rel_diff_x) > rel_tol) == 0

            self._cache_dict[cache_index] = CachedData(is_regular_x)
        if self._copy_cache:
            return self._cache_dict[cache_index].data
        else:
            return self._cache_dict[cache_index].data_nocopy

    @property
    def is_regular_y(self):
        """
        Test whether the grid spacing is regular in the y direction.
        """
        cache_index = "is_regular_y"
        if (
            cache_index not in self._cache_dict
            or self._cache_dict[cache_index].out_of_date
        ):
            # relative tolerance to use in test
            rel_tol = 1.0e-5

            # regularity test in y direction
            rel_diff_y = (self.__delc - self.__delc[0]) / self.__delc[0]
            is_regular_y = np.count_nonzero(np.abs(rel_diff_y) > rel_tol) == 0

            self._cache_dict[cache_index] = CachedData(is_regular_y)
        if self._copy_cache:
            return self._cache_dict[cache_index].data
        else:
            return self._cache_dict[cache_index].data_nocopy

    @property
    def is_regular_z(self):
        """
        Test if the grid spacing is regular in z direction.
        """
        cache_index = "is_regular_z"
        if (
            cache_index not in self._cache_dict
            or self._cache_dict[cache_index].out_of_date
        ):
            # relative tolerance to use in test
            rel_tol = 1.0e-5

            # regularity test in z direction
            rel_diff_thick0 = (
                self.delz[0, :, :] - self.delz[0, 0, 0]
            ) / self.delz[0, 0, 0]
            failed = np.abs(rel_diff_thick0) > rel_tol
            is_regular_z = np.count_nonzero(failed) == 0
            for k in range(1, self.nlay):
                rel_diff_zk = (
                    self.delz[k, :, :] - self.delz[0, :, :]
                ) / self.delz[0, :, :]
                failed = np.abs(rel_diff_zk) > rel_tol
                is_regular_z = is_regular_z and np.count_nonzero(failed) == 0

            self._cache_dict[cache_index] = CachedData(is_regular_z)
        if self._copy_cache:
            return self._cache_dict[cache_index].data
        else:
            return self._cache_dict[cache_index].data_nocopy

    @property
    def is_regular_xy(self):
        """
        Test if the grid spacing is regular and equal in x and y directions.
        """
        cache_index = "is_regular_xy"
        if (
            cache_index not in self._cache_dict
            or self._cache_dict[cache_index].out_of_date
        ):
            # relative tolerance to use in test
            rel_tol = 1.0e-5

            # test if the first delta is equal in x and z
            rel_diff_0 = (self.__delc[0] - self.__delr[0]) / self.__delr[0]
            first_equal = np.abs(rel_diff_0) <= rel_tol

            # combine with regularity tests in x and z directions
            is_regular_xy = (
                first_equal and self.is_regular_x and self.is_regular_y
            )

            self._cache_dict[cache_index] = CachedData(is_regular_xy)
        if self._copy_cache:
            return self._cache_dict[cache_index].data
        else:
            return self._cache_dict[cache_index].data_nocopy

    @property
    def is_regular_xz(self):
        """
        Test if the grid spacing is regular and equal in x and z directions.
        """
        cache_index = "is_regular_xz"
        if (
            cache_index not in self._cache_dict
            or self._cache_dict[cache_index].out_of_date
        ):
            # relative tolerance to use in test
            rel_tol = 1.0e-5

            # test if the first delta is equal in x and z
            rel_diff_0 = (self.delz[0, 0, 0] - self.__delr[0]) / self.__delr[0]
            first_equal = np.abs(rel_diff_0) <= rel_tol

            # combine with regularity tests in x and z directions
            is_regular_xz = (
                first_equal and self.is_regular_x and self.is_regular_z
            )

            self._cache_dict[cache_index] = CachedData(is_regular_xz)
        if self._copy_cache:
            return self._cache_dict[cache_index].data
        else:
            return self._cache_dict[cache_index].data_nocopy

    @property
    def is_regular_yz(self):
        """
        Test if the grid spacing is regular and equal in y and z directions.
        """
        cache_index = "is_regular_yz"
        if (
            cache_index not in self._cache_dict
            or self._cache_dict[cache_index].out_of_date
        ):
            # relative tolerance to use in test
            rel_tol = 1.0e-5

            # test if the first delta is equal in y and z
            rel_diff_0 = (self.delz[0, 0, 0] - self.__delc[0]) / self.__delc[0]
            first_equal = np.abs(rel_diff_0) <= rel_tol

            # combine with regularity tests in x and y directions
            is_regular_yz = (
                first_equal and self.is_regular_y and self.is_regular_z
            )

            self._cache_dict[cache_index] = CachedData(is_regular_yz)
        if self._copy_cache:
            return self._cache_dict[cache_index].data
        else:
            return self._cache_dict[cache_index].data_nocopy

    @property
    def is_regular(self):
        """
        Test if the grid spacing is regular and equal in x, y and z directions.
        """
        cache_index = "is_regular"
        if (
            cache_index not in self._cache_dict
            or self._cache_dict[cache_index].out_of_date
        ):
            # relative tolerance to use in test
            rel_tol = 1.0e-5

            # test if the first delta is equal in x and z
            rel_diff_0 = (self.delz[0, 0, 0] - self.__delr[0]) / self.__delr[0]
            first_equal = np.abs(rel_diff_0) <= rel_tol

            # combine with regularity tests in x, y and z directions
            is_regular = (
                first_equal and self.is_regular_z and self.is_regular_xy
            )

            self._cache_dict[cache_index] = CachedData(is_regular)
        if self._copy_cache:
            return self._cache_dict[cache_index].data
        else:
            return self._cache_dict[cache_index].data_nocopy

    @property
    def is_rectilinear(self):
        """
        Test whether the grid is rectilinear (it is always so in the x and
        y directions, but not necessarily in the z direction).
        """
        cache_index = "is_rectilinear"
        if (
            cache_index not in self._cache_dict
            or self._cache_dict[cache_index].out_of_date
        ):
            # relative tolerance to use in test
            rel_tol = 1.0e-5

            # rectilinearity test in z direction
            is_rect_z = True
            for k in range(self.nlay):
                rel_diff_zk = (
                    self.delz[k, :, :] - self.delz[k, 0, 0]
                ) / self.delz[k, 0, 0]
                failed = np.abs(rel_diff_zk) > rel_tol
                is_rect_z = is_rect_z and np.count_nonzero(failed) == 0

            self._cache_dict[cache_index] = CachedData(is_rect_z)
        if self._copy_cache:
            return self._cache_dict[cache_index].data
        else:
            return self._cache_dict[cache_index].data_nocopy

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
                    "x, y point given is outside of the model area"
                )
        else:
            col = np.where(xcomp)[0][-1]

        ycomp = y < ye
        if np.all(ycomp) or not np.any(ycomp):
            if forgive:
                row = np.nan
            else:
                raise Exception(
                    "x, y point given is outside of the model area"
                )
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
        Returns
        ------- list of x,y cell vertices
        """
        self._copy_cache = False
        cell_verts = [
            (self.xvertices[i, j], self.yvertices[i, j]),
            (self.xvertices[i, j + 1], self.yvertices[i, j + 1]),
            (self.xvertices[i + 1, j + 1], self.yvertices[i + 1, j + 1]),
            (self.xvertices[i + 1, j], self.yvertices[i + 1, j]),
        ]
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
        f = open(gridspec_file, "r")
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
                if "*" in r:
                    rraw = r.split("*")
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
                if "*" in r:
                    rraw = r.split("*")
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
        shape_verts = (a.shape[0] + 1, a.shape[1] + 1, a.shape[2] + 1)

        # set to NaN where idomain==0
        a[self._idomain == 0] = np.nan

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
        Interpolate array values at cell vertices.

        Parameters
        ----------
        a : ndarray
            Array values. Allowed shapes are: (nlay, nrow, ncol),
            (nlay, nrow, ncol+1), (nlay, nrow+1, ncol) and
            (nlay+1, nrow, ncol).
            * When the shape is (nlay, nrow, ncol), input values are
            considered at cell centers, and output values are computed by
            trilinear interpolation.
            * When the shape is extended in one direction, input values are
            considered at the center of cell faces in this direction, and
            output values are computed by bilinear interpolation in planes
            defined by these cell faces.

        Returns
        -------
        averts : ndarray
            Array values interpolated at cell vertices, shape
            (nlay+1, nrow+1, ncol+1).

        Notes
        -----
            * Output values are smooth (continuous) even if top elevations or
            bottom elevations are not constant across layers (i.e., in this
            case, vertices of neighboring cells are implicitly merged).
            * NaN values are assigned in accordance with inactive cells defined
            by idomain.
        """
        import scipy.interpolate as interp

        # define shapes
        shape_ext_x = (self.nlay, self.nrow, self.ncol + 1)
        shape_ext_y = (self.nlay, self.nrow + 1, self.ncol)
        shape_ext_z = (self.nlay + 1, self.nrow, self.ncol)
        shape_verts = (self.nlay + 1, self.nrow + 1, self.ncol + 1)

        # get inactive cells
        if self._idomain is not None:
            inactive = self._idomain == 0

        # get local x and y cell center coordinates (1d arrays)
        xcenters, ycenters = self.xycenters

        # get z center coordinates: make the grid rectilinear if it is not,
        # in order to always use RegularGridInterpolator; in most cases this
        # will give better results than with the non-structured interpolator
        # LinearNDInterpolator (in addition, it will run faster)
        zcenters = self.zcellcenters
        if self._idomain is not None:
            zcenters = np.where(inactive, np.nan, zcenters)
        if (
            not self.is_rectilinear
            or np.count_nonzero(np.isnan(zcenters)) != 0
        ):
            zedges = np.nanmean(self.top_botm_withnan, axis=(1, 2))
        else:
            zedges = self.top_botm_withnan[:, 0, 0]
        zcenters = 0.5 * (zedges[1:] + zedges[:-1])

        # test grid regularity in z
        rel_tol = 1.0e-5
        delz = np.diff(zedges)
        rel_diff = (delz - delz[0]) / delz[0]
        _is_regular_z = np.count_nonzero(np.abs(rel_diff) > rel_tol) == 0

        # test equality of first grid spacing in x and z, and in y and z
        first_equal_xz = np.abs(self.__delr[0] - delz[0]) / delz[0] <= rel_tol
        first_equal_yz = np.abs(self.__delc[0] - delz[0]) / delz[0] <= rel_tol

        # get output coordinates (i.e. vertices)
        xedges, yedges = self.xyedges
        xedges = xedges.reshape((1, 1, self.ncol + 1))
        xoutput = xedges * np.ones(shape_verts)
        yedges = yedges.reshape((1, self.nrow + 1, 1))
        youtput = yedges * np.ones(shape_verts)
        zoutput = zedges.reshape((self.nlay + 1, 1, 1))
        zoutput = zoutput * np.ones(shape_verts)

        # indicator of whether basic interpolation is used or not
        basic = False

        if a.shape == self.shape:
            # set array to NaN where inactive
            if self._idomain is not None:
                inactive = self._idomain == 0
                a = np.where(inactive, np.nan, a)

            # perform basic interpolation (this will be useful in all cases)
            averts_basic = self.array_at_verts_basic(a)

            if self.is_regular_xy and _is_regular_z and first_equal_xz:
                # in this case, basic interpolation is the correct one
                averts = averts_basic
                basic = True

            else:
                if self.nlay == 1:
                    # in this case we need a 2d interpolation in the x, y plane
                    # flip y coordinates because RegularGridInterpolator
                    # requires increasing input coordinates
                    xyinput = (np.flip(ycenters), xcenters)
                    a = np.squeeze(np.flip(a, axis=[1]))
                    # interpolate
                    interp_func = interp.RegularGridInterpolator(
                        xyinput, a, bounds_error=False, fill_value=np.nan
                    )
                    xyoutput = np.empty((youtput[0, :, :].size, 2))
                    xyoutput[:, 0] = youtput[0, :, :].ravel()
                    xyoutput[:, 1] = xoutput[0, :, :].ravel()
                    averts2d = interp_func(xyoutput)
                    averts2d = averts2d.reshape(
                        (1, self.nrow + 1, self.ncol + 1)
                    )
                    averts = averts2d * np.ones(shape_verts)
                elif self.nrow == 1:
                    # in this case we need a 2d interpolation in the x, z plane
                    # flip z coordinates because RegularGridInterpolator
                    # requires increasing input coordinates
                    xzinput = (np.flip(zcenters), xcenters)
                    a = np.squeeze(np.flip(a, axis=[0]))
                    # interpolate
                    interp_func = interp.RegularGridInterpolator(
                        xzinput, a, bounds_error=False, fill_value=np.nan
                    )
                    xzoutput = np.empty((zoutput[:, 0, :].size, 2))
                    xzoutput[:, 0] = zoutput[:, 0, :].ravel()
                    xzoutput[:, 1] = xoutput[:, 0, :].ravel()
                    averts2d = interp_func(xzoutput)
                    averts2d = averts2d.reshape(
                        (self.nlay + 1, 1, self.ncol + 1)
                    )
                    averts = averts2d * np.ones(shape_verts)
                elif self.ncol == 1:
                    # in this case we need a 2d interpolation in the y, z plane
                    # flip y and z coordinates because RegularGridInterpolator
                    # requires increasing input coordinates
                    yzinput = (np.flip(zcenters), np.flip(ycenters))
                    a = np.squeeze(np.flip(a, axis=[0, 1]))
                    # interpolate
                    interp_func = interp.RegularGridInterpolator(
                        yzinput, a, bounds_error=False, fill_value=np.nan
                    )
                    yzoutput = np.empty((zoutput[:, :, 0].size, 2))
                    yzoutput[:, 0] = zoutput[:, :, 0].ravel()
                    yzoutput[:, 1] = youtput[:, :, 0].ravel()
                    averts2d = interp_func(yzoutput)
                    averts2d = averts2d.reshape(
                        (self.nlay + 1, self.nrow + 1, 1)
                    )
                    averts = averts2d * np.ones(shape_verts)
                else:
                    # 3d interpolation
                    # flip y and z coordinates because RegularGridInterpolator
                    # requires increasing input coordinates
                    xyzinput = (np.flip(zcenters), np.flip(ycenters), xcenters)
                    a = np.flip(a, axis=[0, 1])
                    # interpolate
                    interp_func = interp.RegularGridInterpolator(
                        xyzinput, a, bounds_error=False, fill_value=np.nan
                    )
                    xyzoutput = np.empty((zoutput.size, 3))
                    xyzoutput[:, 0] = zoutput.ravel()
                    xyzoutput[:, 1] = youtput.ravel()
                    xyzoutput[:, 2] = xoutput.ravel()
                    averts = interp_func(xyzoutput)
                    averts = averts.reshape(shape_verts)

        elif a.shape == shape_ext_x:
            # set array to NaN where inactive on both side
            if self._idomain is not None:
                inactive_ext_x = np.full(shape_ext_x, True)
                inactive_ext_x[:, :, :-1] = inactive
                inactive_ext_x[:, :, 1:] = np.logical_and(
                    inactive_ext_x[:, :, 1:], inactive
                )
                a = np.where(inactive_ext_x, np.nan, a)

            averts = np.empty(shape_verts, dtype=a.dtype)
            averts_basic = np.empty(shape_verts, dtype=a.dtype)
            for j in range(self.ncol + 1):
                # perform basic interpolation (will be useful in all cases)
                averts_basic[:, :, j] = array_at_verts_basic2d(a[:, :, j])

                if self.is_regular_y and _is_regular_z and first_equal_yz:
                    # in this case, basic interpolation is the correct one
                    averts2d = averts_basic[:, :, j]
                    basic = True

                else:
                    if self.nlay == 1:
                        # in this case we need a 1d interpolation along y
                        averts1d = array_at_faces_1d(a[0, :, j], self.__delc)
                        averts2d = averts1d.reshape((1, self.nrow + 1))
                        averts2d = averts2d * np.ones((2, self.nrow + 1))
                    elif self.nrow == 1:
                        # in this case we need a 1d interpolation along z
                        delz1d = np.abs(np.diff(self.zverts_smooth[:, 0, j]))
                        averts1d = array_at_faces_1d(a[:, 0, j], delz1d)
                        averts2d = averts1d.reshape((self.nlay + 1, 1))
                        averts2d = averts2d * np.ones((self.nlay + 1, 2))
                    else:
                        # 2d interpolation
                        # flip y and z coordinates because
                        # RegularGridInterpolator requires increasing input
                        # coordinates
                        yzinput = (np.flip(zcenters), np.flip(ycenters))
                        a2d = np.flip(a[:, :, j], axis=[0, 1])
                        interp_func = interp.RegularGridInterpolator(
                            yzinput, a2d, bounds_error=False, fill_value=np.nan
                        )
                        yzoutput = np.empty((zoutput[:, :, j].size, 2))
                        yzoutput[:, 0] = zoutput[:, :, j].ravel()
                        yzoutput[:, 1] = youtput[:, :, j].ravel()
                        averts2d = interp_func(yzoutput)
                        averts2d = averts2d.reshape(zoutput[:, :, j].shape)

                averts[:, :, j] = averts2d

        elif a.shape == shape_ext_y:
            # set array to NaN where inactive on both side
            if self._idomain is not None:
                inactive_ext_y = np.full(shape_ext_y, True)
                inactive_ext_y[:, :-1, :] = inactive
                inactive_ext_y[:, 1:, :] = np.logical_and(
                    inactive_ext_y[:, 1:, :], inactive
                )
                a = np.where(inactive_ext_y, np.nan, a)

            averts = np.empty(shape_verts, dtype=a.dtype)
            averts_basic = np.empty(shape_verts, dtype=a.dtype)
            for i in range(self.nrow + 1):
                # perform basic interpolation (will be useful in all cases)
                averts_basic[:, i, :] = array_at_verts_basic2d(a[:, i, :])

                if self.is_regular_x and _is_regular_z and first_equal_xz:
                    # in this case, basic interpolation is the correct one
                    averts2d = averts_basic[:, i, :]
                    basic = True

                else:
                    if self.nlay == 1:
                        # in this case we need a 1d interpolation along x
                        averts1d = array_at_faces_1d(a[0, i, :], self.__delr)
                        averts2d = averts1d.reshape((1, self.ncol + 1))
                        averts2d = averts2d * np.ones((2, self.ncol + 1))
                    elif self.ncol == 1:
                        # in this case we need a 1d interpolation along z
                        delz1d = np.abs(np.diff(self.zverts_smooth[:, i, 0]))
                        averts1d = array_at_faces_1d(a[:, i, 0], delz1d)
                        averts2d = averts1d.reshape((self.nlay + 1, 1))
                        averts2d = averts2d * np.ones((self.nlay + 1, 2))
                    else:
                        # 2d interpolation
                        # flip z coordinates because RegularGridInterpolator
                        # requires increasing input coordinates
                        xzinput = (np.flip(zcenters), xcenters)
                        a2d = np.flip(a[:, i, :], axis=[0])
                        interp_func = interp.RegularGridInterpolator(
                            xzinput, a2d, bounds_error=False, fill_value=np.nan
                        )
                        xzoutput = np.empty((zoutput[:, i, :].size, 2))
                        xzoutput[:, 0] = zoutput[:, i, :].ravel()
                        xzoutput[:, 1] = xoutput[:, i, :].ravel()
                        averts2d = interp_func(xzoutput)
                        averts2d = averts2d.reshape(zoutput[:, i, :].shape)

                averts[:, i, :] = averts2d

        elif a.shape == shape_ext_z:
            # set array to NaN where inactive on both side
            if self._idomain is not None:
                inactive_ext_z = np.full(shape_ext_z, True)
                inactive_ext_z[:-1, :, :] = inactive
                inactive_ext_z[1:, :, :] = np.logical_and(
                    inactive_ext_z[1:, :, :], inactive
                )
                a = np.where(inactive_ext_z, np.nan, a)

            averts = np.empty(shape_verts, dtype=a.dtype)
            averts_basic = np.empty(shape_verts, dtype=a.dtype)
            for k in range(self.nlay + 1):
                # perform basic interpolation (will be useful in all cases)
                averts_basic[k, :, :] = array_at_verts_basic2d(a[k, :, :])

                if self.is_regular_xy:
                    # in this case, basic interpolation is the correct one
                    averts2d = averts_basic[k, :, :]
                    basic = True

                else:
                    if self.nrow == 1:
                        # in this case we need a 1d interpolation along x
                        averts1d = array_at_faces_1d(a[k, 0, :], self.__delr)
                        averts2d = averts1d.reshape((1, self.ncol + 1))
                        averts2d = averts2d * np.ones((2, self.ncol + 1))
                    elif self.ncol == 1:
                        # in this case we need a 1d interpolation along y
                        averts1d = array_at_faces_1d(a[k, :, 0], self.__delc)
                        averts2d = averts1d.reshape((self.nrow + 1, 1))
                        averts2d = averts2d * np.ones((self.nrow + 1, 2))
                    else:
                        # 2d interpolation
                        # flip y coordinates because RegularGridInterpolator
                        # requires increasing input coordinates
                        xyinput = (np.flip(ycenters), xcenters)
                        a2d = np.flip(a[k, :, :], axis=[0])
                        interp_func = interp.RegularGridInterpolator(
                            xyinput, a2d, bounds_error=False, fill_value=np.nan
                        )
                        xyoutput = np.empty((youtput[k, :, :].size, 2))
                        xyoutput[:, 0] = youtput[k, :, :].ravel()
                        xyoutput[:, 1] = xoutput[k, :, :].ravel()
                        averts2d = interp_func(xyoutput)
                        averts2d = averts2d.reshape(youtput[k, :, :].shape)

                averts[k, :, :] = averts2d

        if not basic:
            # use basic interpolation for remaining NaNs at boundaries
            where_nan = np.isnan(averts)
            averts[where_nan] = averts_basic[where_nan]

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
        # get the dimension that corresponds to the direction
        dir_to_dim = {"x": 2, "y": 1, "z": 0}
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
            weight2 = delta_ghost[:-1, :, :] / (
                delta_ghost[:-1, :, :] + delta_ghost[1:, :, :]
            )
            weight1 = 1.0 - weight2

            # interpolate
            afaces = a_ghost[:-1, :, :] * weight1 + a_ghost[1:, :, :] * weight2

            # assign NaN where idomain==0 on both sides
            if withnan and self._idomain is not None:
                inactive_faces = np.full(afaces.shape, True)
                inactive_faces[:-1, :, :] = np.logical_and(
                    inactive_faces[:-1, :, :], inactive
                )
                inactive_faces[1:, :, :] = np.logical_and(
                    inactive_faces[1:, :, :], inactive
                )
                afaces[inactive_faces] = np.nan

        elif dim == 1:
            # fill array with ghost cells
            a_ghost[:, 1:-1, :] = a
            a_ghost[:, 0, :] = a[:, 0, :]
            a_ghost[:, -1, :] = a[:, -1, :]

            # calculate weights
            delc = np.reshape(self.delc, (1, self.nrow, 1))
            delc_3D = delc * np.ones(a.shape)
            delta_ghost[:, 1:-1, :] = delc_3D
            weight2 = delta_ghost[:, :-1, :] / (
                delta_ghost[:, :-1, :] + delta_ghost[:, 1:, :]
            )
            weight1 = 1.0 - weight2

            # interpolate
            afaces = a_ghost[:, :-1, :] * weight1 + a_ghost[:, 1:, :] * weight2

            # assign NaN where idomain==0 on both sides
            if withnan and self._idomain is not None:
                inactive_faces = np.full(afaces.shape, True)
                inactive_faces[:, :-1, :] = np.logical_and(
                    inactive_faces[:, :-1, :], inactive
                )
                inactive_faces[:, 1:, :] = np.logical_and(
                    inactive_faces[:, 1:, :], inactive
                )
                afaces[inactive_faces] = np.nan

        elif dim == 2:
            # fill array with ghost cells
            a_ghost[:, :, 1:-1] = a
            a_ghost[:, :, 0] = a[:, :, 0]
            a_ghost[:, :, -1] = a[:, :, -1]

            # calculate weights
            delr = np.reshape(self.delr, (1, 1, self.ncol))
            delr_3D = delr * np.ones(a.shape)
            delta_ghost[:, :, 1:-1] = delr_3D
            weight2 = delta_ghost[:, :, :-1] / (
                delta_ghost[:, :, :-1] + delta_ghost[:, :, 1:]
            )
            weight1 = 1.0 - weight2

            # interpolate
            afaces = a_ghost[:, :, :-1] * weight1 + a_ghost[:, :, 1:] * weight2

            # assign NaN where idomain==0 on both sides
            if withnan and self._idomain is not None:
                inactive_faces = np.full(afaces.shape, True)
                inactive_faces[:, :, :-1] = np.logical_and(
                    inactive_faces[:, :, :-1], inactive
                )
                inactive_faces[:, :, 1:] = np.logical_and(
                    inactive_faces[:, :, 1:], inactive
                )
                afaces[inactive_faces] = np.nan

        return afaces

    def get_number_plottable_layers(self, a):
        """
        Calculate and return the number of 2d plottable arrays that can be
        obtained from the array passed (a)

        Parameters
        ----------
        a : ndarray
            array to check for plottable layers

        Returns
        -------
        nplottable : int
            number of plottable layers

        """
        nplottable = 0
        required_shape = self.get_plottable_layer_shape()
        if a.shape == required_shape:
            nplottable = 1
        else:
            nplottable = a.size / self.nrow / self.ncol
            nplottable = int(nplottable)
        return nplottable

    def get_plottable_layer_array(self, a, layer):
        # ensure plotarray is 2d and correct shape
        required_shape = self.get_plottable_layer_shape()
        if a.ndim == 3:
            plotarray = a[layer, :, :]
        elif a.ndim == 2:
            plotarray = a
        elif a.ndim == 1:
            plotarray = a
            if plotarray.shape[0] == self.nrow * self.ncol:
                plotarray = plotarray.reshape(required_shape)
            elif plotarray.shape[0] == self.nnodes:
                plotarray = plotarray.reshape(self.shape)
                plotarray = plotarray[layer, :, :]
        else:
            raise Exception("Array to plot must be of dimension 1, 2, or 3")
        msg = "{} /= {}".format(plotarray.shape, required_shape)
        assert plotarray.shape == required_shape, msg
        return plotarray


if __name__ == "__main__":
    delc = np.ones((10,)) * 1
    delr = np.ones((20,)) * 1

    top = np.ones((10, 20)) * 2000
    botm = np.ones((1, 10, 20)) * 1100

    t = StructuredGrid(delc, delr, top, botm, xoff=0, yoff=0, angrot=45)

    t.use_ref_coords = False
    x = t.xvertices
    y = t.yvertices
    xc = t.xcellcenters
    yc = t.ycellcenters
    grid = t.grid_lines

    t.use_ref_coords = True
    sr_x = t.xvertices
    sr_y = t.yvertices
    sr_xc = t.xcellcenters
    sr_yc = t.ycellcenters
    sr_grid = t.grid_lines
    print(sr_grid)
