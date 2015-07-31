import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from . import plotutil
from .plotutil import bc_color_dict

from flopy.utils import util_2d, util_3d, transient_2d

class ModelMap(object):
    """
    Class to create a map of the model.

    Parameters
    ----------
    sr : flopy.utils.reference.SpatialReference
        The spatial reference class (Default is None)
    ax : matplotlib.pyplot axis
        The plot axis.  If not provided it, plt.gca() will be used.
        If there is not a current axis then a new one will be created.
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

    Notes
    -----
    ModelMap must know the position and rotation of the grid in order to make
    the plot.  This information is contained in the SpatialReference class
    (sr), which can be passed.  If sr is None, then it looks for sr in dis.
    If dis is None, then it looks for sr in model.dis.  If all of these
    arguments are none, then it uses xul, yul, and rotation.  If none of these
    arguments are provided, then it puts the lower-left-hand corner of the
    grid at (0, 0).

    """
    def __init__(self, sr=None, ax=None, model=None, dis=None, layer=0,
                 extent=None, xul=None, yul=None, rotation=None):
        self.model = model
        self.layer = layer
        self.dis = dis
        self.sr = None
        if sr is not None:
            self.sr = copy.deepcopy(sr)
        elif dis is not None:
            #print("warning: the dis arg to model map is deprecated")
            self.sr = copy.deepcopy(dis.sr)
        elif model is not None:
            #print("warning: the model arg to model map is deprecated")
            self.sr = copy.deepcopy(model.dis.sr)
    
        # model map override spatial reference settings
        if xul is not None:
            self.sr.xul = xul
        if yul is not None:
            self.sr.yul = yul
        if rotation is not None:
            self.sr.rotation = rotation
        

        if ax is None:
            try:
                self.ax = plt.gca()
                self.ax.set_aspect('equal')
            except:
                self.ax = plt.subplot(1, 1, 1, aspect='equal', axisbg="white")
        else:
            self.ax = ax
        if extent is not None:
            self._extent = extent
        else:
            self._extent = None
        
        # why is this non-default color scale used??
        #  This should be passed as a kwarg by the user to the indivudual plotting method.
        #self.cmap = plotutil.viridis

        return

    @property
    def extent(self):
        if self._extent is None:
            self._extent = self.sr.get_extent()
        return self._extent

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
        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax
        quadmesh = ax.pcolormesh(self.sr.xgrid, self.sr.ygrid, plotarray,
                                 **kwargs)
        ax.set_xlim(self.extent[0], self.extent[1])
        ax.set_ylim(self.extent[2], self.extent[3])
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
        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax
        if 'colors' in kwargs.keys():
            if 'cmap' in kwargs.keys():
                cmap = kwargs.pop('cmap')
            cmap = None
        contour_set = ax.contour(self.sr.xcentergrid, self.sr.ycentergrid,
                                      plotarray, **kwargs)
        ax.set_xlim(self.extent[0], self.extent[1])
        ax.set_ylim(self.extent[2], self.extent[3])

        return contour_set

    def plot_inactive(self, ibound=None, color_noflow='black', **kwargs):
        """
        Make a plot of inactive cells.  If not specified, then pull ibound from the
        self.ml

        Parameters
        ----------
        ibound : numpy.ndarray
            ibound array to plot.  (Default is ibound in 'BAS6' package.)
        color_noflow : string
            (Default is 'black')

        Returns
        -------
        quadmesh : matplotlib.collections.QuadMesh

        """
        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax

        if ibound is None:
            bas = self.model.get_package('BAS6')
            ibound = bas.ibound

        plotarray = np.zeros(ibound.shape, dtype=np.int)
        idx1 = (ibound == 0)
        plotarray[idx1] = 1
        plotarray = np.ma.masked_equal(plotarray, 0)
        cmap = matplotlib.colors.ListedColormap(['0', color_noflow])
        bounds = [0, 1, 2]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        quadmesh = self.plot_array(plotarray, cmap=cmap, norm=norm, **kwargs)
        return quadmesh

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
        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax

        if ibound is None:
            bas = self.model.get_package('BAS6')
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
        ax.set_xlim(self.extent[0], self.extent[1])
        ax.set_ylim(self.extent[2], self.extent[3])

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
        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax

        # Find package to plot
        if package is not None:
            p = package
        elif self.model is not None:
            if ftype is None:
                raise Exception('ftype not specified')
            p = self.model.get_package(ftype)
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
        nlay = self.model.nlay
        # Plot the list locations
        plotarray = np.zeros((nlay, self.sr.nrow, self.sr.ncol), dtype=np.int)
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
        bounds = [0, 1, 2]
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

    def plot_discharge(self, frf, fff, dis=None, flf=None, head=None, istep=1, jstep=1,
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
        # make sure dis is defined
        if dis is None:
            if self.model is not None:
                dis = self.model.dis
            else:
                print("ModelMap.plot_quiver() error: self.dis is None and dis arg is None ")
                return
        delr = dis.delr.array
        delc = dis.delc.array
        top = dis.top.array
        botm = dis.botm.array
        nlay, nrow, ncol = botm.shape
        laytyp = None
        hnoflo = 999.
        hdry = 999.
        if self.model is not None:
            lpf = self.model.get_package('LPF')
            if lpf is not None:
                laytyp = lpf.laytyp.array
                hdry = lpf.hdry
            bas = self.model.get_package('BAS6')
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
        x = self.sr.xcentergrid[::istep, ::jstep]
        y = self.sr.ycentergrid[::istep, ::jstep]
        u = qx[self.layer, :, :]
        v = qy[self.layer, :, :]
        u = u[::istep, ::jstep]
        v = v[::istep, ::jstep]

        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax

        # Rotate and plot
        urot, vrot = self.sr.rotate(u, v, self.sr.rotation)
        quiver = ax.quiver(x, y, urot, vrot, **kwargs)

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

        linecol = []
        for p in pl:
            vlc = []
            #rotate data
            x0r, y0r = self.sr.rotate(p['x'], p['y'], self.sr.rotation, 0., self.sr.yedge[0])
            x0r += self.sr.xul
            y0r += self.sr.yul - self.sr.yedge[0]
            #build polyline array
            arr = np.vstack((x0r, y0r)).T
            #select based on layer
            if kon >= 0:
                arr = np.ma.masked_where((p['k'] != kon), arr)
            #append line to linecol if there is some unmasked segment
            if not arr.mask.all():
                linecol.append(arr)
        #create line collection
        lc = None
        if len(linecol) > 0:
            lc = LineCollection(linecol, **kwargs)
            ax.add_collection(lc)
        return lc


    def plot_endpoint(self, ep, **kwargs):
        """
        Plot the MODPATH endpoints.

        Parameters
        ----------
        ep : rec array
            rec array is data returned from modpathfile EndpointFile
            get_data() or get_alldata() methods. Data in rec array
            is 'x', 'y', 'z', 'time', 'k', and 'particleid'.

        kwargs : layer, ax, c, s, colorbar, colorbar_label, shrink. The
            remaining kwargs are passed into the matplotlib scatter
            method. If layer='all', endpoints are output for all layers.
            If colorbar is True a colorbar will be added to the plot.
            If colorbar_label is passed in and colorbar is True then
            colorbar_label will be passed to the colorbar set_label()
            method. If shrink is passed in and colorbar is True then
            the colorbar size will be set using shrink.

        Returns
        -------
        sp : matplotlib.pyplot.scatter

        """

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

        #scatter kwargs that users may redefine
        if 'c' not in kwargs:
            c = ep['time']
        else:
            c = np.empty((ep.shape[0]), dtype="S30")
            c.fill(kwargs.pop('c'))

        if 's' not in kwargs:
            s = 50.
        else:
            s = float(kwargs.pop('s'))**2.

        #colorbar kwargs
        createcb = False
        if 'colorbar' in kwargs:
            createcb = kwargs.pop('colorbar')

        colorbar_label = 'Endpoint Time'
        if 'colorbar_label' in kwargs:
            colorbar_label = kwargs.pop('colorbar_label')

        shrink = 1.
        if 'shrink' in kwargs:
            shrink = float(kwargs.pop('shrink'))

        #rotate data
        x0r, y0r = self.sr.rotate(ep['x'], ep['y'], self.sr.rotation, 0., self.sr.yedge[0])
        x0r += self.sr.xul
        y0r += self.sr.yul - self.sr.yedge[0]
        #build array to plot
        arr = np.vstack((x0r, y0r)).T
        #select based on layer
        if kon >= 0:
            c = np.ma.masked_where((ep['k'] != kon), c)
        #plot the end point data
        sp = plt.scatter(arr[:, 0], arr[:, 1], c=c, s=s, **kwargs)
        #add a colorbar for endpoint times
        if createcb:
            cb = plt.colorbar(sp, shrink=shrink)
            cb.set_label(colorbar_label)
        return sp


    def get_grid_line_collection(self, **kwargs):
        """
        Get a LineCollection of the grid

        """
        from matplotlib.collections import LineCollection
        xmin = self.sr.xedge[0]
        xmax = self.sr.xedge[-1]
        ymin = self.sr.yedge[-1]
        ymax = self.sr.yedge[0]
        linecol = []
        # Vertical lines
        for j in range(self.sr.ncol + 1):
            x0 = self.sr.xedge[j]
            x1 = x0
            y0 = ymin
            y1 = ymax
            x0r, y0r = self.sr.rotate(x0, y0, self.sr.rotation, 0, self.sr.yedge[0])
            x0r += self.sr.xul
            y0r += self.sr.yul - self.sr.yedge[0]
            x1r, y1r = self.sr.rotate(x1, y1, self.sr.rotation, 0, self.sr.yedge[0])
            x1r += self.sr.xul
            y1r += self.sr.yul - self.sr.yedge[0]
            linecol.append(((x0r, y0r), (x1r, y1r)))

        #horizontal lines
        for i in range(self.sr.nrow + 1):
            x0 = xmin
            x1 = xmax
            y0 = self.sr.yedge[i]
            y1 = y0
            x0r, y0r = self.sr.rotate(x0, y0, self.sr.rotation, 0, self.sr.yedge[0])
            x0r += self.sr.xul
            y0r += self.sr.yul - self.sr.yedge[0]
            x1r, y1r = self.sr.rotate(x1, y1, self.sr.rotation, 0, self.sr.yedge[0])
            x1r += self.sr.xul
            y1r += self.sr.yul - self.sr.yedge[0]
            linecol.append(((x0r, y0r), (x1r, y1r)))

        lc = LineCollection(linecol, **kwargs)
        return lc
