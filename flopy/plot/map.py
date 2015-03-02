import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

bc_color_dict = {'default': 'black', 'WEL': 'red', 'DRN': 'yellow',
                 'RIV': 'green', 'GHB': 'cyan', 'CHD': 'navy'}


def rotate(x, y, theta, xorigin=0., yorigin=0.):
    """
    Given x and y array-like values calculate the rotation about an
    arbitrary origin and then return the rotated coordinates.  theta is in
    radians.

    """
    xrot = xorigin + np.cos(theta) * (x - xorigin) - np.sin(theta) * \
                                                     (y - yorigin)
    yrot = yorigin + np.sin(theta) * (x - xorigin) + np.cos(theta) * \
                                                     (y - yorigin)
    return xrot, yrot


class ModelMap(object):
    """
    Class to create a map of the model.

    Parameters
    ----------
    ax : matplotlib.pyplot axis
        The plot axis.  If not provided it, plt.gca() will be used.
    dis : flopy discretization object
    layer : int
        Layer to plot.  Default is 0.
    xul : float
        x coordinate for upper left corner
    yul : float
        y coordinate for upper left corner
    rotation : float
        Angle of grid rotation around the upper left corner.  A positive value
        indicates clockwise rotation.  Angles are in degrees.
    extent : tuple of floats
        (xmin, xmax, ymin, ymax) will be used to specify axes limits.  If None
        then these will be calculated based on grid, coordinates, and rotation
    """
    def __init__(self, ax=None, ml=None, dis=None, layer=0, xul=None, yul=None,
                 rotation=0., extent=None):
        self.ml = ml
        self.layer = layer
        if dis is None:
            if ml is None:
                raise Exception('Cannot find discretization package')
            else:
                self.dis = ml.get_package('DIS')
        else:
            self.dis = dis

        if ax is None:
            self.ax = plt.gca()
        else:
            self.ax = ax

        # Set origin and rotation
        if xul is None:
            self.xul = 0.
        else:
            self.xul = xul
        if yul is None:
            self.yul = np.add.reduce(self.dis.delc.array)
        else:
            self.yul = yul
        self.rotation = -rotation * np.pi / 180.

        # Create edge arrays and meshgrid for pcolormesh
        self.xedge = self.get_xedge_array()
        self.yedge = self.get_yedge_array()
        self.xgrid, self.ygrid = np.meshgrid(self.xedge, self.yedge)
        self.xgrid, self.ygrid = rotate(self.xgrid, self.ygrid, self.rotation,
                                        0, self.yedge[0])
        self.xgrid += self.xul
        self.ygrid += self.yul - self.yedge[0]

        # Create model extent
        if extent is None:
            self.extent = self.get_extent()
        else:
            self.extent = extent

        # Set axis limits
        self.ax.set_xlim(self.extent[0], self.extent[1])
        self.ax.set_ylim(self.extent[2], self.extent[3])

        return

    def plot_array(self, a, **kwargs):
        """
        Plot the array

        """
        if a.ndim == 3:
            plotarray = a[self.layer, :, :]
        elif a.ndim == 2:
            plotarray = a
        else:
            raise Exception('Array must be of dimension 2 or 3')
        quadmesh = plt.pcolormesh(self.xgrid, self.ygrid, plotarray, **kwargs)
        return quadmesh

    def plot_ibound(self, ibound=None, color_noflow='black', color_ch='blue'):
        """
        Make a plot of ibound.  If not specified, then pull ibound from the
        self.ml

        """
        if ibound is None:
            bas = self.ml.get_package('BAS6')
            ibound = bas.ibound
        plotarray = np.zeros(ibound.shape, dtype=np.int)
        idx1 = (ibound == 0)
        idx2 = (ibound < 0)
        plotarray[idx1] = 1
        plotarray[idx2] = 2
        plotarray = np.ma.masked_equal(plotarray, 0)
        cmap = matplotlib.colors.ListedColormap(['0', color_noflow, color_ch])
        bounds=[0, 1, 2, 3]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        quadmesh = self.plot_array(plotarray, cmap=cmap, norm=norm)
        return quadmesh

    def plot_grid(self, **kwargs):
        """
        Plot the grid lines.

        Parameters
        ----------
            kwargs : ax, colors.  The remaining kwargs are passed into the
                the LineCollection constructor.

        Returns
        -------
            lc : LineCollection

        """
        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax

        if 'colors' not in kwargs:
            kwargs['colors'] = '0.5'

        lc = self.get_grid_line_collection(**kwargs)
        ax.add_collection(lc)
        return lc

    def plot_bc(self, ftype=None, package=None, kper=0, color=None, **kwargs):
        """
        Plot a boundary locations for a flopy model

        """
        # Find package to plot
        if package is not None:
            p = package
        elif self.ml is not None:
            if ftype is None:
                raise Exception('ftype not specified')
            p = self.ml.get_package(ftype)
        else:
            raise Exception('Cannot find package to plot')

        # Get the list data
        try:
            mflist = p.stress_period_data[kper]
        except:
            raise Exception('Not a list-style boundary package')

        # Plot the list locations
        plotarray = np.zeros(self.dis.botm.shape, dtype=np.int)
        idx = [mflist['k'], mflist['i'], mflist['j']]
        plotarray[idx] = 1
        plotarray = np.ma.masked_equal(plotarray, 0)
        if color is None:
            if ftype in bc_color_dict:
                c = bc_color_dict[ftype]
            else:
                c = bc_color_dict['default']
        else:
            c = color
        cmap = matplotlib.colors.ListedColormap(['0', c])
        bounds=[0, 1, 2]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        quadmesh = self.plot_array(plotarray, cmap=cmap, norm=norm, **kwargs)
        return quadmesh

    def get_grid_line_collection(self, **kwargs):
        """
        Get a LineCollection of the grid
        """
        from matplotlib.collections import LineCollection
        xmin = self.xedge[0]
        xmax = self.xedge[-1]
        ymin = self.yedge[-1]
        ymax = self.yedge[0]
        linecol = []
        # Vertical lines
        for j in xrange(self.dis.ncol + 1):
            x0 = self.xedge[j]
            x1 = x0
            y0 = ymin
            y1 = ymax
            x0r, y0r = rotate(x0, y0, self.rotation, 0, self.yedge[0])
            x0r += self.xul
            y0r += self.yul - self.yedge[0]
            x1r, y1r = rotate(x1, y1, self.rotation, 0, self.yedge[0])
            x1r += self.xul
            y1r += self.yul - self.yedge[0]
            linecol.append(((x0r, y0r), (x1r, y1r)))

        #horizontal lines
        for i in xrange(self.dis.nrow + 1):
            x0 = xmin
            x1 = xmax
            y0 = self.yedge[i]
            y1 = y0
            x0r, y0r = rotate(x0, y0, self.rotation, 0, self.yedge[0])
            x0r += self.xul
            y0r += self.yul - self.yedge[0]
            x1r, y1r = rotate(x1, y1, self.rotation, 0, self.yedge[0])
            x1r += self.xul
            y1r += self.yul - self.yedge[0]
            linecol.append(((x0r, y0r), (x1r, y1r)))

        lc = LineCollection(linecol, **kwargs)
        return lc

    def get_xedge_array(self):
        """
        Return a numpy one-dimensional float array that has the cell edge x
        coordinates for every cell in the grid.  Array is of size (ncol + 1)

        """
        xedge = np.concatenate(([0.], np.add.accumulate(self.dis.delr.array)))
        return xedge

    def get_yedge_array(self):
        """
        Return a numpy one-dimensional float array that has the cell edge y
        coordinates for every cell in the grid.  Array is of size (nrow + 1)

        """
        length_y = np.add.reduce(self.dis.delc.array)
        yedge = np.concatenate(([length_y], length_y -
                             np.add.accumulate(self.dis.delc.array)))
        return yedge

    def get_extent(self):
        """
        Get the extent of the rotated and offset grid

        Return (xmin, xmax, ymin, ymax)

        """
        x0 = self.xedge[0]
        x1 = self.xedge[-1]
        y0 = self.yedge[0]
        y1 = self.yedge[-1]

        # upper left point
        x0r, y0r = rotate(x0, y0, self.rotation, 0, self.yedge[0])
        x0r += self.xul
        y0r += self.yul - self.yedge[0]

        # upper right point
        x1r, y1r = rotate(x1, y0, self.rotation, 0, self.yedge[0])
        x1r += self.xul
        y1r += self.yul - self.yedge[0]

        # lower right point
        x2r, y2r = rotate(x1, y1, self.rotation, 0, self.yedge[0])
        x2r += self.xul
        y2r += self.yul - self.yedge[0]

        # lower left point
        x3r, y3r = rotate(x0, y1, self.rotation, 0, self.yedge[0])
        x3r += self.xul
        y3r += self.yul - self.yedge[0]

        xmin = min(x0r, x1r, x2r, x3r)
        xmax = max(x0r, x1r, x2r, x3r)
        ymin = min(y0r, y1r, y2r, y3r)
        ymax = max(y0r, y1r, y2r, y3r)

        return (xmin, xmax, ymin, ymax)


