import numpy as np


class SpatialReference(object):
    """
    Transformation class
    """

    def __init__(self, xoffset=None, yoffset=None, rotation=0.,
                 origin_location="ll", length_multiplier=1.,
                 model_grid=None):

        if xoffset is None:
            xoffset = 0.
        if yoffset is None:
            yoffset = 0

        self.__model_grid = model_grid
        self.__xoffset = xoffset
        self.__yoffset = yoffset
        self.__rotation = rotation
        self.__length_multiplier = length_multiplier
        self.__origin_location = origin_location.lower()

        if self.__model_grid is not None:
            self.__model_grid.sr = self

        self.__xcenters = None
        self.__ycenters = None
        self.__xgrid = None
        self.__ygrid = None
        self.__xedges = None
        self.__yedges = None

    def __clear_cache(self):
        """
        Clears cached arrays
        """
        self.__xcenters = None
        self.__ycenters = None
        self.__xgrid = None
        self.__ygrid = None
        self.__xedge = None
        self.__yedge = None

    def __setattr__(self, key, value):
        clear_cache = True
        if key == "xoffset":
            super(SpatialReference, self). \
                __setattr__("_SpatialReference__xoffset", float(value))

        elif key == "yoffset":
            super(SpatialReference, self). \
                __setattr__("_SpatialReference__yoffset", float(value))

        elif key == "length_multiplier":
            super(SpatialReference, self). \
                __setattr__("_SpatialReference__length_multiplier", float(value))

        elif key == "origin_location":
            if value.lower() not in ("ul", "ll"):
                err = "Offset location must be specified as {ul or ll}"
                raise ValueError(err)

            elif value.lower() == "ul" and self.__model_grid is None:
                err = "Spatial reference orgin_location must be ll" \
                      " if a model grid is not attached"
                raise ValueError(err)

            super(SpatialReference, self). \
                __setattr__("_SpatialReference__origin_location", str(value.lower()))

        elif key == "rotation":
            super(SpatialReference, self). \
                __setattr__("_SpatialReference__rotation", float(value))

        elif key in ("model_grid"):
            super(SpatialReference, self). \
                __setattr__("_SpatialReference__model_grid", value)
            self.__model_grid.sr = self

        else:
            clear_cache = False
            super(SpatialReference, self).__setattr__(key, value)

        if clear_cache:
            self.__clear_cache()

    @property
    def xoffset(self):
        return self.__xoffset

    @property
    def yoffset(self):
        return self.__yoffset

    @property
    def rotation(self):
        return self.__rotation

    @property
    def origin_location(self):
        return self.__origin_location

    @property
    def length_multiplier(self):
        return self.__length_multiplier

    @property
    def xcenters(self):
        if self.__xcenters is None:
            self.__set_centers()
        return self.__xcenters

    @property
    def ycenters(self):
        if self.__ycenters is None:
            self.__set_centers()
        return self.__ycenters

    @property
    def xgrid(self):
        if self.__xgrid is None:
            self.__set_grid()
        return self.__xgrid

    @property
    def ygrid(self):
        if self.__ygrid is None:
            self.__set_grid()
        return self.__ygrid

    @property
    def extent(self):
        if self.__model_grid is None:
            return

        x0, x1, y0, y1 = self.__model_grid.extent
        x0r, y0r = self.transform(x0, y0)
        x1r, y1r = self.transform(x1, y0)
        x2r, y2r = self.transform(x1, y1)
        x3r, y3r = self.transform(x0, y1)

        xmin = min(x0r, x1r, x2r, x3r)
        xmax = max(x0r, x1r, x2r, x3r)
        ymin = min(y0r, y1r, y2r, y3r)
        ymax = max(y0r, y1r, y2r, y3r)

        return xmin, xmax, ymin, ymax

    @property
    def grid_lines(self):
        """
        Returns a list of rotated, translated set of line vertices
        """
        if self.__model_grid is None:
            return

        grid_lines = self.__model_grid.grid_lines
        lines = []
        for verts in grid_lines:
            x0, y0 = verts[0]
            x1, y1 = verts[1]

            x0r, y0r = self.transform(x0, y0)
            x1r, y1r = self.transform(x1, y1)
            lines.append([(x0r, y0r), (x1r, y1r)])

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

    def __set_grid(self):
        if self.__model_grid is None:
            return

        self.__xgrid, self.__ygrid = self.transform(self.__model_grid.xgrid,
                                                    self.__model_grid.ygrid)

    def __set_centers(self):
        if self.__model_grid is None:
            return

        self.__xcenters, self.__ycenters = self.transform(self.__model_grid.xcenters,
                                                          self.__model_grid.ycenters)

    def set_spatialreference(self, xoffset=0, yoffset=0, rotation=0.0,
                             origin_location="ll", length_multiplier=1.):
        """

        :param xoffset:
        :param yoffset:
        :param rotation:
        :param length_multiplier:
        :return:
        """
        self.xoffset = xoffset
        self.yoffset = yoffset
        self.rotation = rotation
        self.origin_location = origin_location
        self.length_multiplier = length_multiplier

    @staticmethod
    def rotate(x, y, theta, xorigin=0., yorigin=0.):
        """
        Given x and y array-like values calculate the rotation about an
        arbitrary origin and then return the rotated coordinates.  theta is in
        degrees.

        """
        # jwhite changed on Oct 11 2016 - rotation is now positive CCW
        # theta = -theta * np.pi / 180.
        theta = theta * np.pi / 180.

        xrot = xorigin + np.cos(theta) * (x - xorigin) - np.sin(theta) * \
               (y - yorigin)
        yrot = yorigin + np.sin(theta) * (x - xorigin) + np.cos(theta) * \
               (y - yorigin)
        return xrot, yrot

    def transform(self, x, y, inverse=False):
        """
        Given x and y array-like values, apply rotation, scale and offset,
        to convert them from model coordinates to real-world coordinates.
        """
        # todo: Not sure that there is a way to deal with upper-left corner coords. without model grid?

        if self.origin_location == "ul":
            yul = self.__model_grid.extent[-1]
            theta = self.rotation * np.pi / 180
            xoffset = self.xoffset + (np.sin(theta) * yul *
                                    self.length_multiplier)
            yoffset = self.yoffset - (np.cos(theta) * yul *
                                      self.length_multiplier)


        else:
            yoffset = self.yoffset
            xoffset = self.xoffset

        if isinstance(x, list):
            x = np.array(x)
            y = np.array(y)
        if not np.isscalar(x):
            x, y = x.copy(), y.copy()

        if not inverse:
            x *= self.length_multiplier
            y *= self.length_multiplier
            x += xoffset
            y += yoffset
            x, y = SpatialReference.rotate(x, y, theta=self.rotation,
                                           xorigin=xoffset, yorigin=yoffset)
        else:
            x, y = SpatialReference.rotate(x, y, -self.rotation,
                                           xorigin=xoffset, yorigin=yoffset)
            x -= xoffset
            y -= yoffset
            x /= self.length_multiplier
            y /= self.length_multiplier

        return x, y


if __name__ == "__main__":
    t = SpatialReference(xoffset=1, yoffset=10,
                         rotation=0, length_multiplier=3.28)

    print('break')