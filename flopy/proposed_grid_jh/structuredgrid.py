import numpy as np
from flopy.grid.grid import Grid, CachedData


class StructuredGrid(Grid):
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
                 lenuni=2, epsg=None, proj4=None,
                 xoff=None, yoff=None, angrot=0.0):
        super(StructuredGrid, self).__init__('structured', top, botm,
                                             idomain, epsg, proj4,
                                             lenuni, xoff, yoff, angrot)
        self.__delc = delc
        self.__delr = delr
        self.__nrow = len(delc)
        self.__ncol = len(delr)
        if top is not None:
            assert self.__nrow * self.__ncol == len(np.ravel(top))
        if botm is not None:
            assert self.__nrow * self.__ncol == len(np.ravel(botm[0]))
            self.__nlay = len(botm)
        else:
            self.__nlay = None

    ####################
    # Properties
    ####################
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
    def extent(self):
        xyzgrid = self.xyzgrid
        return (np.min(xyzgrid[0]), np.max(xyzgrid[0]),
                np.min(xyzgrid[1]), np.max(xyzgrid[1]))

    @property
    def xgridlength(self):
        return abs(self.xedges[0] - self.xedges[-1])

    @property
    def ygridlength(self):
        return abs(self.yedges[0] - self.yedges[-1])

    @property
    def delc(self):
        return self.__delc

    @property
    def delr(self):
        return self.__delr

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
                xgrid, ygrid = self.transform(xgrid, ygrid)
            self._cache_dict[cache_index] = \
                CachedData([xgrid, ygrid, zgrid])
        return self._cache_dict[cache_index].data

    @property
    def xyedges(self):
        cache_index = 'xyedges'
        if cache_index not in self._cache_dict or \
                self._cache_dict[cache_index].out_of_date:
            xedge = np.concatenate(([0.], np.add.accumulate(self.__delr)))
            length_y = np.add.reduce(self.__delc)
            yedge = np.concatenate(([length_y], length_y -
                                    np.add.accumulate(self.delc)))
            self._cache_dict[cache_index] = \
                CachedData([xedge, yedge])
        return self._cache_dict[cache_index].data

    @property
    def xyvertices(self):
        cache_index = 'xyvertices'
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
                for l in range(1, self.__nlay):
                    z[l, :, :] = (self._botm[l - 1, :, :] +
                                  self._botm[l, :, :]) / 2.
            else:
                z = None
            if self._has_ref_coordinates:
                # transform x and y
                x_mesh, y_mesh = self.transform(x_mesh, y_mesh)
            # store in cache
            self._cache_dict[cache_index] = CachedData([x_mesh, y_mesh, z])
        return self._cache_dict[cache_index].data

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

    def get_all_model_cells(self):
        model_cells = []
        for layer in range(0, self.__nlay):
            for row in range(0, self.__nrow):
                for column in range(0, self.__ncol):
                    model_cells.append((layer + 1, row + 1, column + 1))
        return model_cells


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from flopy.proposed_grid_srp.reference import SpatialReference
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
    x = t.xgrid
    y = t.ygrid
    xc = t.xcenters
    yc = t.ycenters
    #extent = t.extent
    grid = t.grid_lines

    #print('break')

    t.use_ref_coords = True
    sr_x = t.xgrid
    sr_y = t.ygrid
    sr_xc = t.xcenters
    sr_yc = t.ycenters
    #sr_extent = t.extent
    sr_grid = t.grid_lines
    print(sr_grid)
    #t.plot_grid_lines()
    #plt.show()
    #print('break')