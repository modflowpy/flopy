import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from . import plotutil
from .plotutil import bc_color_dict, rotate

from flopy.utils import util_2d,util_3d,transient_2d

class ModelMap(object):
    """
    Class to create a map of the model.

    Parameters
    ----------
    ax : matplotlib.pyplot axis
        The plot axis.  If not provided it, plt.gca() will be used.
    model : flopy.modflow object
        flopy model object. (Default is None)
    dis : flopy.modflow.ModflowDis object
        flopy discretization object. (Default is None)
    layer : int
        Layer to plot.  Default is 0.  Must be between 0 and nlay - 1.
    xul : float
        x coordinate for upper left corner
    yul : float
        y coordinate for upper left corner.  The default is the sum of the
        delc array.
    rotation : float
        Angle of grid rotation around the upper left corner.  A positive value
        indicates clockwise rotation.  Angles are in degrees.
    extent : tuple of floats
        (xmin, xmax, ymin, ymax) will be used to specify axes limits.  If None
        then these will be calculated based on grid, coordinates, and rotation.

    """
    def __init__(self, ax=None, model=None, dis=None, layer=0, xul=None,
                 yul=None, rotation=0., extent=None):
        self.ml = model
        self.layer = layer
        if dis is None:
            if model is None:
                raise Exception('Cannot find discretization package')
            else:
                self.dis = model.get_package('DIS')
        else:
            self.dis = dis

        if self.layer < 0 or self.layer > self.dis.nlay - 1:
            s = 'Not a valid layer: {}.  Must be between 0 and {}.'.format(
                self.layer, self.dis.nlay - 1)
            raise Exception(s)

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

        # Create x and y center arrays and meshgrid of centers
        self.xcenter = self.get_xcenter_array()
        self.ycenter = self.get_ycenter_array()
        self.xcentergrid, self.ycentergrid = np.meshgrid(self.xcenter,
                                                         self.ycenter)
        self.xcentergrid, self.ycentergrid = rotate(self.xcentergrid,
                                                    self.ycentergrid,
                                                    self.rotation,
                                                    0, self.yedge[0])
        self.xcentergrid += self.xul
        self.ycentergrid += self.yul - self.yedge[0]

        # Create model extent
        if extent is None:
            self.extent = self.get_extent()
        else:
            self.extent = extent

        # Set axis limits
        self.ax.set_xlim(self.extent[0], self.extent[1])
        self.ax.set_ylim(self.extent[2], self.extent[3])

        return

    def write_grid_shapefile(self, filename, package_names=None,array_dict=None):
        """
        Write a shapefile for the model grid.  If package_names is not none,
        then search through the requested packages looking for arrays that can
        be added to the shapefile as attributes

        Parameters
        ----------
        filename : string
            name of the shapefile to write
        package_names : (optional) list of package names (e.g. ["dis","lpf"])
            packages to scrap arrays out of for adding to shapefile
        array_dict : (optional) dict of {name:2D array} pairs
           additional 2D arrays to add as attributes to the grid shapefile


        Returns
        -------
        None

        """

        try:
            import shapefile
        except Exception as e:
            raise Exception("ModelMap.write_grid_shapefile(): error " +
                            "importing shapefile - need to install pyshp")

        wr = shapefile.Writer(shapeType=shapefile.POLYGON)
        wr.field("row", "N", 10, 0)
        wr.field("column", "N", 10, 0)

        arrays = []
        if array_dict is not None:
            for name,array in array_dict.items():
                assert array.shape == (self.ml.nrow,self.ml.ncol)
                wr.field(name,"N",20,12)
                arrays.append(array)

        if package_names is not None:
            if not isinstance(package_names, list):
                package_names = [package_names]
            for pname in package_names:
                pak = self.ml.get_package(pname)
                if pak is not None:
                    attrs = dir(pak)
                    for attr in attrs:
                        a = pak.__getattribute__(attr)
                        if isinstance(a, util_2d) and a.shape == (self.ml.nrow,
                                                                  self.ml.ncol):
                            name = a.name.lower()
                            wr.field(name, 'N', 20, 12)
                            arrays.append(a.array)
                        elif isinstance(a, util_3d):
                            for i,u2d in enumerate(a):
                                name = u2d.name.lower().replace(' ','_')
                                if "_layer" in name:
                                    name = name.replace("_layer", '')
                                else:
                                    name += '_{0:d}'.format(i+1)
                                wr.field(name, 'N', 20, 12)
                                arrays.append(u2d.array)
                        elif isinstance(a,transient_2d):
                            kpers = list(a.transient_2ds.keys())
                            kpers.sort()
                            for kper in kpers:
                                u2d = a.transient_2ds[kper]
                                name = u2d.name.lower() + "_{0:d}".format(kper+1)
                                wr.field(name, 'N', 20, 12)
                                arrays.append(u2d.array)

        for i in range(self.ml.nrow):
            for j in range(self.ml.ncol):
                pts = []
                pts.append([self.xgrid[i, j], self.ygrid[i, j]])
                pts.append([self.xgrid[i, j], self.ygrid[i+1, j]])
                pts.append([self.xgrid[i, j+1], self.ygrid[i+1, j]])
                pts.append([self.xgrid[i, j+1], self.ygrid[i, j]])
                pts.append([self.xgrid[i, j], self.ygrid[i, j]])
                wr.poly(parts=[pts])
                rec = [i+1, j+1]
                for array in arrays:
                    rec.append(array[i, j])
                wr.record(*rec)
        wr.save(filename)


    def plot_array(self, a, masked_values=None, **kwargs):
        """
        Plot an array.  If the array is three-dimensional, then the method
        will plot the layer tied to this class (self.layer).

        Parameters
        ----------
        a : numpy.ndarray
            Array to plot.
        masked_values : iterable of floats, ints
            Values to mask.
        **kwargs : dictionary
            keyword arguments passed to matplotlib.pyplot.pcolormesh

        Returns
        -------
        quadmesh : matplotlib.collections.QuadMesh

        """
        if a.ndim == 3:
            plotarray = a[self.layer, :, :]
        elif a.ndim == 2:
            plotarray = a
        else:
            raise Exception('Array must be of dimension 2 or 3')
        if masked_values is not None:
            for mval in masked_values:
                plotarray = np.ma.masked_equal(plotarray, mval)
        quadmesh = self.ax.pcolormesh(self.xgrid, self.ygrid, plotarray,
                                      **kwargs)
        return quadmesh

    def contour_array(self, a, masked_values=None, **kwargs):
        """
        Contour an array.  If the array is three-dimensional, then the method
        will contour the layer tied to this class (self.layer).

        Parameters
        ----------
        a : numpy.ndarray
            Array to plot.
        masked_values : iterable of floats, ints
            Values to mask.
        **kwargs : dictionary
            keyword arguments passed to matplotlib.pyplot.pcolormesh

        Returns
        -------
        contour_set : matplotlib.pyplot.contour

        """
        if a.ndim == 3:
            plotarray = a[self.layer, :, :]
        elif a.ndim == 2:
            plotarray = a
        else:
            raise Exception('Array must be of dimension 2 or 3')
        if masked_values is not None:
            for mval in masked_values:
                plotarray = np.ma.masked_equal(plotarray, mval)
        contour_set = self.ax.contour(self.xcentergrid, self.ycentergrid,
                                      plotarray, **kwargs)
        return contour_set

    def plot_ibound(self, ibound=None, color_noflow='black', color_ch='blue',
                    **kwargs):
        """
        Make a plot of ibound.  If not specified, then pull ibound from the
        self.ml

        Parameters
        ----------
        ibound : numpy.ndarray
            ibound array to plot.  (Default is ibound in 'BAS6' package.)
        color_noflow : string
            (Default is 'black')
        color_ch : string
            Color for constant heads (Default is 'blue'.)

        Returns
        -------
        quadmesh : matplotlib.collections.QuadMesh

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
        quadmesh = self.plot_array(plotarray, cmap=cmap, norm=norm, **kwargs)
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
        lc : matplotlib.collections.LineCollection

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
        Plot boundary conditions locations for a specific boundary
        type from a flopy model

        Parameters
        ----------
        ftype : string
            Package name string ('WEL', 'GHB', etc.). (Default is None)
        package : flopy.modflow.Modflow package class instance
            flopy package class instance. (Default is None)
        kper : int
            Stress period to plot
        color : string
            matplotlib color string. (Default is None)
        **kwargs : dictionary
            keyword arguments passed to matplotlib.collections.PatchCollection

        Returns
        -------
        quadmesh : matplotlib.collections.QuadMesh

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

        # Return if mflist is None
        if mflist is None:
            return None

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

    def plot_shapefile(self, shp, **kwargs):
        """
        Plot a shapefile.  The shapefile must be in the same coordinates as
        the rotated and offset grid.

        Parameters
        ----------
        shp : string
            Name of the shapefile to plot

        kwargs : dictionary
            Keyword arguments passed to plotutil.plot_shapefile()

        """
        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax
        patch_collection = plotutil.plot_shapefile(shp, ax, **kwargs)
        return patch_collection

    def plot_discharge(self, frf, fff, flf=None, head=None, istep=1, jstep=1,
                       **kwargs):
        """
        Use quiver to plot vectors.

        Parameters
        ----------
        frf : numpy.ndarray
            MODFLOW's 'flow right face'
        fff : numpy.ndarray
            MODFLOW's 'flow front face'
        flf : numpy.ndarray
            MODFLOW's 'flow lower face' (Default is None.)
        head : numpy.ndarray
            MODFLOW's head array.  If not provided, then will assume confined
            conditions in order to calculated saturated thickness.
        istep : int
            row frequency to plot. (Default is 1.)
        jstep : int
            column frequency to plot. (Default is 1.)
        kwargs : dictionary
            Keyword arguments passed to plt.quiver()

        Returns
        -------
        quiver : matplotlib.pyplot.quiver
            Vectors of specific discharge.

        """

        # Calculate specific discharge
        delr = self.dis.delr.array
        delc = self.dis.delc.array
        top = self.dis.top.array
        botm = self.dis.botm.array
        nlay, nrow, ncol = botm.shape
        laytyp = None
        hnoflo = 999.
        hdry = 999.
        if self.ml is not None:
            lpf = self.ml.get_package('LPF')
            if lpf is not None:
                laytyp = lpf.laytyp.array
                hdry = lpf.hdry
            bas = self.ml.get_package('BAS6')
            if bas is not None:
                hnoflo = bas.hnoflo

        # If no access to head or laytyp, then calculate confined saturated
        # thickness by setting laytyp to zeros
        if head is None or laytyp is None:
            head = np.zeros(botm.shape, np.float32)
            laytyp = np.zeros((nlay), dtype=np.int)
        sat_thk = plotutil.saturated_thickness(head, top, botm, laytyp,
                                               [hnoflo, hdry])

        # Calculate specific discharge
        qx, qy, qz = plotutil.centered_specific_discharge(frf, fff, flf, delr,
                                                          delc, sat_thk)

        # Select correct slice and step
        x = self.xcentergrid[::istep, ::jstep]
        y = self.ycentergrid[::istep, ::jstep]
        u = qx[self.layer, :, :]
        v = qy[self.layer, :, :]
        u = u[::istep, ::jstep]
        v = v[::istep, ::jstep]

        # Rotate and plot
        urot, vrot = rotate(u, v, self.rotation)
        quiver = self.ax.quiver(x, y, urot, vrot, **kwargs)

        return quiver

    def plot_pathline(self, pl, **kwargs):
        """
        Plot the MODPATH pathlines.

        Parameters
        ----------
        pl : list of rec arrays or a single rec array
            rec array or list of rec arrays is data returned from
            modpathfile PathlineFile get_data() or get_alldata()
            methods. Data in rec array is 'x', 'y', 'z', 'time',
            'k', and 'particleid'.

        kwargs : layer, ax, colors.  The remaining kwargs are passed
            into the LineCollection constructor. If layer='all',
            pathlines are output for all layers

        Returns
        -------
        lc : matplotlib.collections.LineCollection

        """
        from matplotlib.collections import LineCollection
        #make sure pathlines is a list
        if isinstance(pl, np.ndarray):
            pl = [pl]
        
        if 'layer' in kwargs:
            kon = kwargs.pop('layer')
            if isinstance(kon, str):
                if kon.lower() == 'all':
                    kon = -1
                else:
                    kon = self.layer
        else:
            kon = self.layer

        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax

        if 'colors' not in kwargs:
            kwargs['colors'] = '0.5'

        dtype = np.dtype([("x", np.float32), ("y", np.float32)])

        linecol = []
        for p in pl:
            vlc = []
            #rotate data
            x0r, y0r = rotate(p['x'], p['y'], self.rotation, 0., self.yedge[0])
            x0r += self.xul
            y0r += self.yul - self.yedge[0]
            #select based on layer
            if kon < 0:
                isel = p['k'] >= 0
            else:
                isel = p['k'] == kon
            for idx, lv in enumerate(isel):
                if lv:
                    vlc.append((x0r[idx], y0r[idx]))
                else:
                    if len(vlc) > 0:
                        linecol.append(vlc)
                        vlc = []
                #--finalize pathline
                if len(vlc) > 0:
                    linecol.append(vlc)

        lc = LineCollection(linecol, **kwargs)            
        ax.add_collection(lc)
        return lc


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
        for j in range(self.dis.ncol + 1):
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
        for i in range(self.dis.nrow + 1):
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

    def get_xcenter_array(self):
        """
        Return a numpy one-dimensional float array that has the cell center x
        coordinate for every column in the grid.

        """
        x = np.add.accumulate(self.dis.delr.array) - 0.5 * self.dis.delr.array
        return x

    def get_ycenter_array(self):
        """
        Return a numpy one-dimensional float array that has the cell center x
        coordinate for every row in the grid.

        """
        Ly = np.add.reduce(self.dis.delc.array)
        y = Ly - (np.add.accumulate(self.dis.delc.array) - 0.5 *
                   self.dis.delc.array)
        return y

    def get_xedge_array(self):
        """
        Return a numpy one-dimensional float array that has the cell edge x
        coordinates for every column in the grid.  Array is of size (ncol + 1)

        """
        xedge = np.concatenate(([0.], np.add.accumulate(self.dis.delr.array)))
        return xedge

    def get_yedge_array(self):
        """
        Return a numpy one-dimensional float array that has the cell edge y
        coordinates for every row in the grid.  Array is of size (nrow + 1)

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


