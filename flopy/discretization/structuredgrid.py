import numpy as np
from .grid import Grid, CachedData


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
    def __init__(self, delc, delr, top=None, botm=None, idomain=None,
                 lenuni=None, epsg=None, proj4=None, prj=None, xoff=0.0,
                 yoff=0.0, angrot=0.0, length_multiplier=1.):
        super(StructuredGrid, self).__init__('structured', top, botm, idomain,
                                             lenuni, epsg, proj4, prj, xoff,
                                             yoff, angrot, length_multiplier)
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
        return self.__delc * self._length_multiplier

    @property
    def delr(self):
        return self.__delr * self._length_multiplier

    @property
    def xyzvertices(self):
        """
        """
        cache_index = 'xyzgrid'
        if cache_index not in self._cache_dict or \
                self._cache_dict[cache_index].out_of_date:
            xedge = np.concatenate(([0.], np.add.accumulate(self.delr)))
            length_y = np.add.reduce(self.delc)
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
        cache_index = 'xyedges'
        if cache_index not in self._cache_dict or \
                self._cache_dict[cache_index].out_of_date:
            xedge = np.concatenate(([0.], np.add.accumulate(self.delr)))
            length_y = np.add.reduce(self.delc)
            yedge = np.concatenate(([length_y], length_y -
                                    np.add.accumulate(self.delc)))
            self._cache_dict[cache_index] = \
                CachedData([xedge, yedge])
        if self._copy_cache:
            return self._cache_dict[cache_index].data
        else:
            return self._cache_dict[cache_index].data_nocopy

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
            x = np.add.accumulate(self.delr) - 0.5 * self.delr
            # get y centers
            Ly = np.add.reduce(self.delc)
            y = Ly - (np.add.accumulate(self.delc) - 0.5 *
                      self.delc)
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
    def intersect(self, x, y, local=False):
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


        Returns
        -------
        row : int
            The row number
        col : int
            The column number

        """
        # transform x and y to local coordinates
        x, y = super(StructuredGrid, self).intersect(x, y, local)

        # get the cell edges in local coordinates
        xe, ye = self.xyedges

        xcomp = x > xe
        if np.all(xcomp) or not np.any(xcomp):
            raise Exception('x, y point given is outside of the model area')
        col = np.where(xcomp)[0][-1]

        ycomp = y < ye
        if np.all(ycomp) or not np.any(ycomp):
            raise Exception('x, y point given is outside of the model area')
        row = np.where(ycomp)[0][-1]

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
        """Write a shapefile of the grid with just the row and column attributes"""
        from ..export.shapefile_utils import write_grid_shapefile2
        if epsg is None and prj is None:
            epsg = self.epsg
        write_grid_shapefile2(filename, self, array_dict={}, nan_val=-1.0e9,
                              epsg=epsg, prj=prj)


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

    #print('break')

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
    #print('break')