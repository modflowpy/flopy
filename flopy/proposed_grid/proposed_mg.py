import numpy as np
from proposed_sr import SpatialReference
from flopy.mf6.data.mfdataarray import MFArray


class StructuredModelGrid(object):
    """
    Method for structured model grid creation
    """
    def __init__(self, delc, delr, top=None, botm=None, idomain=None,
                 sr=None, xoffset=None, yoffset=None, rotation=0.,
                 origin_location="ll", length_multiplier=1.,):

        self.__delc = delc
        self.__delr = delr
        self.__top = top
        self.__botm = botm
        self.__idomain = idomain
        self.__sr = sr

        if self.__sr is None:
            self.__sr = SpatialReference(xoffset=xoffset,
                                         yoffset=yoffset,
                                         rotation=rotation,
                                         origin_location=origin_location,
                                         length_multiplier=length_multiplier,
                                         model_grid=self)

        self.__xcenters = None
        self.__ycenters = None
        self.__xgrid = None
        self.__ygrid = None
        self.__xedges = None
        self.__yedges = None


    def __clear_cache(self):
        self.__xcenters = None
        self.__ycenters = None
        self.__xgrid = None
        self.__ygrid = None
        self.__xedges = None
        self.__yedges = None
        if self.__sr is not None:
            self.__sr.model_grid = self

    def __setattr__(self, key, value):
        clear_cache = True
        if key == "delc":
            if isinstance(value, list) or isinstance(value, np.ndarray):
                super(StructuredModelGrid, self). \
                    __setattr__("_StructuredModelGrid__delc", np.array(value))

            elif isinstance(value, MFArray):
                super(StructuredModelGrid, self). \
                    __setattr__("_StructuredModelGrid__delc", value.array)

        elif key == "delr":
            if isinstance(value, list) or isinstance(value, np.ndarray):
                super(StructuredModelGrid, self). \
                    __setattr__("_StructuredModelGrid__delr", np.array(value))

            elif isinstance(value, MFArray):
                super(StructuredModelGrid, self). \
                    __setattr__("_StructuredModelGrid__delr", value.array)

        elif key == "top":
            if isinstance(value, list) or isinstance(value, np.ndarray):
                super(StructuredModelGrid, self). \
                    __setattr__("_StructuredModelGrid__top", np.array(value))

            elif isinstance(value, MFArray):
                super(StructuredModelGrid, self). \
                    __setattr__("_StructuredModelGrid__top", value.array)

            elif value is None:
                super(StructuredModelGrid, self). \
                    __setattr__("_StructuredModelGrid__top", value)

        elif key == "botm":
            if isinstance(value, list) or isinstance(value, np.ndarray):
                super(StructuredModelGrid, self). \
                    __setattr__("_StructuredModelGrid__botm", np.array(value))

            elif isinstance(value, MFArray):
                super(StructuredModelGrid, self). \
                    __setattr__("_StructuredModelGrid__botm", value.array)

            elif value is None:
                super(StructuredModelGrid, self). \
                    __setattr__("_StructuredModelGrid__botm", value)

        elif key == "idomain":
            if isinstance(value, list) or isinstance(value, np.ndarray):
                super(StructuredModelGrid, self). \
                    __setattr__("_StructuredModelGrid__idomain", np.array(value))

            elif isinstance(value, MFArray):
                super(StructuredModelGrid, self). \
                    __setattr__("_StructuredModelGrid__idomain", value.array)

            elif value is None:
                super(StructuredModelGrid, self). \
                    __setattr__("_StructuredModelGrid__idomain", value)

        elif key == "sr":
            clear_cache = False
            if isinstance(value, SpatialReference):
                super(StructuredModelGrid, self). \
                    __setattr__("_StructuredModelGrid__sr", value)

        else:
            clear_cache = False
            if isinstance(value, MFArray):
                super(StructuredModelGrid, self). \
                    __setattr__(key, value.array)

            else:
                super(StructuredModelGrid, self).__setattr__(key, value)

        if clear_cache:
            self.__clear_cache()

    @property
    def grid_type(self):
        return "structured"

    @property
    def extent(self):
        return (min(self.xedges), max(self.xedges),
                min(self.yedges), max(self.yedges))

    @property
    def nrow(self):
        return len(self.__delc)

    @property
    def ncol(self):
        return len(self.__delr)

    @property
    def nlay(self):
        if self.__botm is not None:
            return len(self.__botm) + 1

    @property
    def delc(self):
        return self.__delc

    @property
    def delr(self):
        return self.__delr

    @property
    def top(self):
        return self.__top

    @property
    def botm(self):
        return self.__botm

    @property
    def idomain(self):
        return self.__idomain

    @property
    def sr(self):
        # trap for the odd cases!
        if self.__sr._SpatialReference__model_grid is None:
            self.__sr.model_grid = self
        return self.__sr

    @property
    def xcenters(self):
        if self.__xcenters is None:
            self.__xcenters = self.__get_grid_centers(dimension="x")
        return self.__xcenters

    @property
    def ycenters(self):
        if self.__ycenters is None:
            self.__ycenters = self.__get_grid_centers(dimension="y")
        return self.__ycenters

    @property
    def xgrid(self):
        if self.__xgrid is None:
            self.__xgrid = self.__get_grid(dimension="x")
        return self.__xgrid

    @property
    def ygrid(self):
        if self.__ygrid is None:
            self.__ygrid = self.__get_grid(dimension="y")
        return self.__ygrid

    @property
    def xedges(self):
        if self.__xedges is None:
            self.__xedges = self.__get_xedges()
        return self.__xedges

    @property
    def yedges(self):
        if self.__yedges is None:
            self.__yedges = self.__get_yedges()
        return self.__yedges

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

    def plot_grid_lines(self, **kwargs):
        """
        Get a LineCollection of the model grid in
        model coordinates

        Parameters
            **kwargs: matplotlib.pyplot keyword arguments

        Returns
            matplotlib.collections.LineCollection
        """
        from flopy.plot.plotbase import PlotMapView

        map = PlotMapView(modelgrid=self)
        lc = map.plot_grid(**kwargs)
        return lc

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

        ax = PlotUtilities._plot_array_helper(a, sr=self, axes=ax, **kwargs)
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
        map = PlotMapView(sr=self)
        contour_set = map.contour_array(a=a, **kwargs)

        return contour_set

    def __get_xedges(self):
        """
        Return a numpy one-dimensional float array that has the cell edge x
        coordinates for every column in the grid in model space - not offset
        or rotated.  Array is of size (ncol + 1)

        """
        xedge = np.concatenate(([0.], np.add.accumulate(self.delr)))
        return xedge

    def __get_yedges(self):
        """
        Return a numpy one-dimensional float array that has the cell edge y
        coordinates for every row in the grid in model space - not offset or
        rotated. Array is of size (nrow + 1)

        """
        length_y = np.add.reduce(self.delc)
        yedge = np.concatenate(([length_y], length_y -
                                np.add.accumulate(self.delc)))
        return yedge

    def __get_grid(self, dimension="x"):

        if dimension.lower() == "x":
            grid = np.tile(self.xedges, (self.nrow + 1, 1))

        elif dimension.lower() == "y":
            grid = np.tile(self.yedges, (self.ncol + 1, 1))
            grid = grid.T

        elif dimension.lower() == "z":
            raise NotImplementedError()

        else:
            raise AssertionError("dimension must be either {'x', 'y', or 'z'}")

        return grid

    def __get_grid_centers(self, dimension="x"):

        if dimension.lower() == "x":
            grid = np.add.accumulate(self.delr) - 0.5 * self.delr
            grid = np.tile(grid, (self.nrow, 1))

        elif dimension.lower() == "y":
            Ly = np.add.reduce(self.delc)
            grid = Ly - (np.add.accumulate(self.delc) - 0.5 *
                         self.delc)
            grid = np.tile(grid, (self.ncol, 1))
            grid = grid.T

        elif dimension.lower() == "z":
            raise NotImplementedError()

        else:
            raise AssertionError("dimension must be either {'x','y', or 'z'}")

        return grid


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    delc = np.ones((10,)) * 1
    delr = np.ones((10,)) * 1

    top = np.ones((10, 20)) * 2000
    botm = np.ones((1, 10, 20)) * 1100

    t = StructuredModelGrid(delc, delr, top, botm, xoffset=0, yoffset=0,
                            rotation=45, origin_location="ul",
                            length_multiplier=1.)

    sr = SpatialReference(10, 20, rotation=35)

    t.sr = sr

    plt.scatter(np.ravel(t.sr.xcenters), np.ravel(t.sr.ycenters), c="b")
    t.sr.plot_grid_lines()
    plt.show()
    plt.close()

    delc = np.ones(10,) * 2
    t.delc = delc

    plt.scatter(np.ravel(t.sr.xcenters), np.ravel(t.sr.ycenters), c="b")
    t.sr.plot_grid_lines()
    plt.show()

    x = t.xgrid
    y = t.ygrid
    xc = t.xcenters
    yc = t.ycenters
    extent = t.extent
    grid = t.grid_lines


    print('break')

    sr_x = t.sr.xgrid
    sr_y = t.sr.ygrid
    sr_xc = t.sr.xcenters
    sr_yc = t.sr.ycenters
    sr_extent = t.sr.extent
    sr_grid = t.sr.grid_lines

    t.sr.plot_grid_lines()
    plt.show()
    print('break')
