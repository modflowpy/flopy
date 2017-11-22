import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from ..plot import plotutil
from ..plot.plotutil import bc_color_dict
from ..utils.decorators import deprecated

class StructuredModelMap(object):
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
            self.sr = copy.deepcopy(dis.sr)
        elif model is not None:
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

        # self.sr.xgrid
        # self.sr.ygrid

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

    @deprecated('<plot_inactive> replaced by <plot_idomain>')
    def plot_inactive(self, ibound=None, color_noflow='black', color_vpt='red', **kwargs):
        """
        Deprecated method! use <plot_idomain>
        Make a plot of inactive cells.  If not specified, then pull ibound from the
        self.ml

        Parameters
        ----------
        ibound : numpy.ndarray
            ibound array to plot.  (Default is ibound in 'BAS6' package.)
        color_noflow : string
            (Default is 'black')
        color_vpt : string
            Color for vertical pass through cells

        Returns
        -------
        quadmesh : matplotlib.collections.QuadMesh

        """
        return self.plot_idomain(idomain=ibound, color_noflow=color_noflow, color_vpt=color_vpt,
                                 **kwargs)

    @deprecated('IBOUND array does not exist in MODFLOW6,'+\
                ' consider <plot_idomain> and <CHD.plot>')
    def plot_ibound(self, ibound=None, kper=0, color_noflow='black', color_ch='blue',
                    color_vpt='red', **kwargs):
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
        color_vpt: string
            Color for vertical pass through cells from the idomain array

        Returns
        -------
        quadmesh : matplotlib.collections.QuadMesh

        """
        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax


        if ibound is None:
            try:
            # todo: create a self.model.get_package('CHD8')
                idomain = self.model.dis.idomain
            except:
                try:
                    idomain = self.dis.idomain
                except:
                    raise Exception
            try:
                chd = self.model.get_package('CHD').stress_period_data[kper]
            except:
                raise Exception

        # todo: update ibound to reflect current
            plotarray = np.zeros(ibound.shape, dtype=np.int)
            idx1 = (idomain == 0)
            idx2 = (chd != 0)
            idx3 = (idomain == -1)

            plotarray[idx1] = 1
            plotarray[idx2] = 2
            plotarray[idx3] = 3
        else:
            raise NotImplementedError('ibound passed as an array is not yet implemented')

        plotarray = np.ma.masked_equal(plotarray, 0)
        cmap = matplotlib.colors.ListedColormap(['0', color_noflow, color_ch,
                                                 color_vpt])
        bounds = [0, 1, 2, 3, 4]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        quadmesh = self.plot_array(plotarray, cmap=cmap, norm=norm,
                                   masked_values=[0.], **kwargs)
        return quadmesh

    def plot_idomain(self, idomain=None, color_noflow='black', color_vpt='red', **kwargs):
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
            color_vpt: string
                Color for vertical pass through cells from the idomain array

            Returns
            -------
            quadmesh : matplotlib.collections.QuadMesh

        """
        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax
        if idomain is None:
            try:
                idomain = self.model.dis.idomain
            except:
                try:
                    idomain = self.dis.idomain
                except:
                    raise AssertionError('Discretation information not supplied to ModelMap()')

        plotarray = np.zeros(idomain.shape, dtype=np.int)
        idx1 = (idomain == 0)
        idx2 = (idomain == -1)
        plotarray[idx1] = 1
        plotarray[idx2] = 2
        plotarray = np.ma.masked_equal(plotarray, 0)
        cmap = matplotlib.colors.ListedColormap(['1', color_noflow, color_vpt])
        bounds = [0, 1, 2, 3]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        quadmesh = self.plot_array(plotarray, cmap=cmap, norm=norm,
                                   masked_values=[0.], **kwargs)
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

    def plot_bc(self, ftype=None, package=None, kper=0, color=None, plotAll=False,
                **kwargs):
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
        plotAll : bool
            Boolean used to specify that boundary condition locations for all
            layers will be plotted on the current ModelMap layer.
            (Default is False)
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
            # todo: restore functionality with packages, remove test case
            mflist = p.data[kper]
            # mflist = p.stress_period_data[kper]
        except Exception as e:
            raise Exception('Not a list-style boundary package:'+str(e))

        # Return if MfList is None
        if mflist is None:
            return None

        nlay = self.sr.nlay
        # Plot the list locations
        plotarray = np.zeros((nlay, self.sr.nrow, self.sr.ncol), dtype=np.int)
        if plotAll:
            # todo: check if mflist data is zero based, if so remove <-1>
            idx = [mflist['row'] - 1, mflist['column'] - 1]
            plotarray[:, idx] = 1
            pa = np.zeros((self.sr.nrow, self.sr.ncol), dtype=np.int)
            pa[idx] = 1
            for k in range(nlay):
                plotarray[k, :, :] = pa.copy()
        else:
            # todo: check if mflist data is zero based, if so remove <-1>
            idx = [mflist['layer'] - 1, mflist['row'] - 1, mflist['column'] - 1]

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

    def plot_discharge(self, ja, dis=None, head=None, istep=1, jstep=1,
                       normalize=False, **kwargs):
        """
        Use quiver to plot vectors.

        Parameters
        ----------
        ja : numpy.ndarray:
            flow ja face array from modflow cell by cell flow
        head : numpy.ndarray
            MODFLOW's head array.  If not provided, then will assume confined
            conditions in order to calculated saturated thickness.
        istep : int
            row frequency to plot. (Default is 1.)
        jstep : int
            column frequency to plot. (Default is 1.)
        normalize : bool
            boolean flag used to determine if discharge vectors should
            be normalized using the magnitude of the specific discharge in each
            cell. (default is False)
        kwargs : dictionary
            Keyword arguments passed to plt.quiver()

        Returns
        -------
        quiver : matplotlib.pyplot.quiver
            Vectors of specific discharge.

        """

        # by default the center of the arrow is plotted in the center of a cell

        if 'pivot' in kwargs:
            pivot = kwargs.pop('pivot')
        else:
            pivot = 'middle'

        if dis is None:
            try:
                dis = self.dis
            except:
                try:
                    dis = self.model.dis
                except:
                    raise AssertionError("ModelMap.plot_quiver() error: self.dis is None and dis arg is None ")

        # todo: update to reflect new ibound setup
        if len(head.shape) == 3:
            head = head[0]

        ja = np.array(ja)
        if len(ja.shape) == 4:
            ja = ja[0][0][0]

        delr = np.tile(dis.delr, (dis.ncol, 1))
        delc = np.tile(dis.delc, (dis.nrow, 1))
        top = np.copy(dis.top)
        botm = np.copy(dis.botm)
        nlay, nrow, ncol = botm.shape
        ncpl = nrow * ncol
        laytyp = None
        hnoflo = 999.
        hdry = 999.

        if head is None:
            head = np.zeros(botm.shape, np.float32)

        head.shape = (nlay, ncpl)
        top.shape = (ncpl)
        botm.shape = (nlay, ncpl)
        delr.shape = (ncpl)
        delc.shape = (ncpl)

        sat_thk = plotutil.saturated_thickness(head, top, botm,
                                               [hnoflo, hdry])

        frf, fff, flf = plotutil.vectorize_flow(ja, dis)
        # flf = None
        # Calculate specific discharge
        qx, qy, qz = plotutil.specific_discharge(frf, fff, flf, delr,
                                                 delc, sat_thk)

        qx.shape = (nlay, nrow, ncol)
        qy.shape = (nlay, nrow, ncol)
        # Select correct slice
        u = qx[self.layer, :, :]
        v = qy[self.layer, :, :]
        # apply step
        x = self.sr.xcentergrid[::istep, ::jstep]
        y = self.sr.ycentergrid[::istep, ::jstep]
        u = u[::istep, ::jstep]
        v = v[::istep, ::jstep]
        # normalize
        if normalize:
            vmag = np.sqrt(u**2. + v**2.)
            idx = vmag > 0.
            u[idx] /= vmag[idx]
            v[idx] /= vmag[idx]

        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax

        # mask discharge in inactive cells
        idomain = dis.idomain
        idx = (idomain[self.layer, ::istep, ::jstep] == 0)
        idx[idomain[self.layer, ::istep, ::jstep] == -1] = 1

        u[idx] = np.nan
        v[idx] = np.nan

        # Rotate and plot
        urot, vrot = self.sr.rotate(u, v, self.sr.rotation)
        quiver = ax.quiver(x, y, urot, vrot, pivot=pivot, **kwargs)

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
        try:
            raise NotImplementedError('method not yet implemented')
        except:
            pass

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

        try:
            raise NotImplementedError('method not yet implemented')
        except:
            pass

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

        lc = LineCollection(self.sr.get_grid_lines(), **kwargs)
        return lc


    def get_patch_collection(self, vdict, data, **kwargs):
        """
        Method to create matplotlib.patches objects from verticies

        """
        from matplotlib.patches import Polygon
        from matplotlib.collections import PatchCollection

        patches = []
        for key in vdict:
            patch = np.array([[line[0], line[1]] for line in vdict[key]])

            # sort vertices by angle. Set center to 0 and use arctan2
            patches.append(Polygon(patch, True))
        p =  PatchCollection(patches, **kwargs)
        p.set_array(data)
        return p


class VertexModelMap(object):
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
    """
    def __init___(self, sr=None, ax=None, model=None, dis=None, layer=0,
            extent=None, **kwargs):
        self.model = model
        self.dis = dis
        self.layer = layer
        self.sr = None
        if sr is not None:
            self.sr = copy.deepcopy(sr)
        elif dis is not None:
            self.sr = copy.deepcopy(dis.sr)
        elif model is not None:
            self.sr = copy.deepcopy(model.dis.sr)

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

        if a.ndim == 2:
            plotarray = a[self.layer, :]
        elif a.ndim == 1:
            plotarray = a
        else:
            raise Exception('Array must be of dimension 1 or 2')

        mask = [False]*plotarray.size
        if masked_values is not None:
            for mval in masked_values:
                plotarray = np.ma.masked_equal(plotarray, mval)
                mask = np.ma.getmask(plotarray)

        if type(mask) is np.bool_:
            mask = [False] * plotarray.size

        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax

        vertexdict = self.sr.xydict

        p = self.get_patch_collection(vertexdict, plotarray, mask, **kwargs)
        patch_collection = ax.add_collection(p)

        ax.set_xlim(self.extent[0], self.extent[1])
        ax.set_ylim(self.extent[2], self.extent[3])

        return patch_collection

    def plot_grid(self, **kwargs):
        """
        Plot the grid lines.

        Parameters
        ----------
        kwargs : ax, colors.  The remaining kwargs are passed into the
            the LineCollection constructor.

        Returns
        -------
        pc : matplotlib.collections.PatchCollection

        """
        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax

        if 'edgecolor' not in kwargs:
            kwargs['edgecolor'] = '0.5'

        if 'facecolor' not in kwargs:
            kwargs['facecolor'] = 'none'

        vertexdict = self.sr.xydict
        pc = self.get_patch_collection(vertexdict, grid=True, **kwargs)

        ax.add_collection(pc)
        ax.set_xlim(self.extent[0], self.extent[1])
        ax.set_ylim(self.extent[2], self.extent[3])

        return pc

    @deprecated('<plot_inactive> replaced by <plot_idomain>')
    def plot_inactive(self, ibound=None, color_noflow='black', color_vpt='red', **kwargs):
        """
        Deprecated method! use <plot_idomain>
        Make a plot of inactive cells.  If not specified, then pull ibound from the
        self.ml

        Parameters
        ----------
        ibound : numpy.ndarray
            ibound array to plot.  (Default is ibound in 'BAS6' package.)
        color_noflow : string
            (Default is 'black')
        color_vpt : string
            Color for vertical pass through cells

        Returns
        -------
        quadmesh : matplotlib.collections.QuadMesh

        """
        return self.plot_idomain(idomain=ibound, color_noflow=color_noflow, **kwargs)

    @deprecated('IBOUND array does not exist in MODFLOW6,'+\
                ' consider <plot_idomain> and <CHD.plot>')
    def plot_ibound(self, ibound=None, kper=0, color_noflow='black', color_ch='blue',
                    color_vpt='red', **kwargs):
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
        color_vpt: string
            Color for vertical pass through cells from the idomain array

        Returns
        -------
        quadmesh : matplotlib.collections.QuadMesh

        """
        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax


        if ibound is None:
            try:
            # todo: create a self.model.get_package('CHD8')
                idomain = self.model.dis.idomain
            except:
                try:
                    idomain = self.dis.idomain
                except:
                    raise Exception
            try:
                chd = self.model.get_package('CHD').stress_period_data[kper]
            except:
                raise Exception

        # todo: update ibound to reflect current
            plotarray = np.zeros(ibound.shape, dtype=np.int)
            idx1 = (idomain == 0)
            idx2 = (chd != 0)
            idx3 = (idomain == -1)

            plotarray[idx1] = 1
            plotarray[idx2] = 2
            plotarray[idx3] = 3
        else:
            raise NotImplementedError('ibound passed as an array is not yet implemented')

        plotarray = np.ma.masked_equal(plotarray, 0)
        #todo: check cmap, bounds, and norm. Change color scheme as appropriate
        cmap = matplotlib.colors.ListedColormap(['0', color_noflow, color_ch,
                                                 color_vpt])
        bounds = [0, 1, 2, 3, 4]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        quadmesh = self.plot_array(plotarray, cmap=cmap, norm=norm,
                                   masked_values=[0.], **kwargs)
        return quadmesh

    def plot_idomain(self, idomain=None, color_noflow='black', color_vpt='red', **kwargs):
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
            color_vpt: string
                Color for vertical pass through cells from the idomain array

            Returns
            -------
            quadmesh : matplotlib.collections.QuadMesh

        """
        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax
        if idomain is None:
            try:
                idomain = self.model.dis.idomain
            except:
                try:
                    idomain = self.dis.idomain
                except:
                    raise AssertionError('Discretation information not supplied to ModelMap()')

        plotarray = np.zeros(idomain.shape, dtype=np.int)
        idx1 = (idomain == 0)
        idx2 = (idomain == -1)
        plotarray[idx1] = 1
        plotarray[idx2] = 2
        plotarray = np.ma.masked_equal(plotarray, 0)
        cmap = matplotlib.colors.ListedColormap(['1',  color_noflow, color_vpt])
        bounds = [0, 1, 2, 3]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        quadmesh = self.plot_array(plotarray, cmap=cmap, norm=norm,
                                   masked_values=[0.], **kwargs)
        return quadmesh

    def plot_bc(self, ftype=None, package=None, kper=0, color=None, plotAll=False,
                **kwargs):

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
        plotAll : bool
            Boolean used to specify that boundary condition locations for all
            layers will be plotted on the current ModelMap layer.
            (Default is False)
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
            # todo: remove test case from try statement, update to flopy code
            mflist = p.data[kper]
            # mflist = p.stress_period_data[kper]
        except Exception as e:
            raise Exception('Not a list-style boundary package:' + str(e))

        # Return if MfList is None
        if mflist is None:
            return None

        nlay = self.sr.nlay
        # Plot the list locations
        plotarray = np.zeros((nlay, self.sr.ncpl), dtype=np.int)
        if plotAll:
            # todo: check if raw data is zero based or 1 based remove <-1> if appropriate
            idx = [mflist['ncpl'] - 1]
            # plotarray[:, idx] = 1
            pa = np.zeros((self.sr.ncpl), dtype=np.int)
            pa[idx] = 1
            for k in range(nlay):
                plotarray[k, :] = pa.copy()
        else:
            # todo: check if raw data is zero based or 1 based remove <-1> if appropriate
            idx = [mflist['layer'] - 1, mflist['ncpl'] - 1]

            plotarray[idx] = 1

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
        patch_collection = self.plot_array(plotarray, cmap=cmap, norm=norm, masked_values=[0], **kwargs)
        return patch_collection

    def contour_array(self, a, masked_values=None, **kwargs):
        """
        Contour an array.  If the array is two-dimensional, then the method
        will contour the layer tied to this class (self.layer).
        Uses scipy.interpolate.griddata to create a mesh for plotting irregularly
        spaced data common with vertex grid

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
        from scipy.interpolate import griddata

        if a.ndim == 2:
            plotarray = a[self.layer, :]
        elif a.ndim == 1:
            plotarray = a
        else:
            raise Exception('Array must be of dimension 1 or 2')

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

        x = self.sr.xcenter_array
        y = self.sr.ycenter_array

        xi = np.linspace(np.min(x), np.max(x), 1000)
        yi = np.linspace(np.min(y), np.max(y), 1000)

        zi = griddata((x, y), plotarray, (xi[None, :], yi[:, None]), method='cubic')

        contour_set = ax.contour(xi, yi, zi, **kwargs)
        # contour_set = ax.contourf(xi, yi, zi, **kwargs)

        ax.set_xlim(self.extent[0], self.extent[1])
        ax.set_ylim(self.extent[2], self.extent[3])

        return contour_set

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
        # make sure pathlines is a list
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
            # build polyline array
            arr = np.vstack((p['x'], p['y'])).T
            # select based on layer
            if kon >= 0:
                arr = np.ma.masked_where((p['k'] != kon), arr)
            # append line to linecol if there is some unmasked segment
            if not arr.mask.all():
                linecol.append(arr)
        # create line collection
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

        # scatter kwargs that users may redefine
        if 'c' not in kwargs:
            c = ep['time']
        else:
            c = np.empty((ep.shape[0]), dtype="S30")
            c.fill(kwargs.pop('c'))

        if 's' not in kwargs:
            s = 50.
        else:
            s = float(kwargs.pop('s')) ** 2.

        # colorbar kwargs
        createcb = False
        if 'colorbar' in kwargs:
            createcb = kwargs.pop('colorbar')

        colorbar_label = 'Endpoint Time'
        if 'colorbar_label' in kwargs:
            colorbar_label = kwargs.pop('colorbar_label')

        shrink = 1.
        if 'shrink' in kwargs:
            shrink = float(kwargs.pop('shrink'))

        # build array to plot
        arr = np.vstack((ep['x'], ep['y'])).T
        # select based on layer
        if kon >= 0:
            c = np.ma.masked_where((ep['k'] != kon), c)
        # plot the end point data
        sp = plt.scatter(arr[:, 0], arr[:, 1], c=c, s=s, **kwargs)
        # add a colorbar for endpoint times
        if createcb:
            cb = plt.colorbar(sp, shrink=shrink)
            cb.set_label(colorbar_label)
        return sp

    def plot_discharge(self, ja, dis=None, head=None, istep=1,
                       normalize=False, **kwargs):
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
            frequency to plot. (Default is 1.)
        normalize : bool
            boolean flag used to determine if discharge vectors should
            be normalized using the magnitude of the specific discharge in each
            cell. (default is False)
        kwargs : dictionary
            Keyword arguments passed to plt.quiver()

        Returns
        -------
        quiver : matplotlib.pyplot.quiver
            Vectors of specific discharge.

        """

        # remove 'pivot' keyword argument
        # by default the center of the arrow is plotted in the center of a cell
        if 'pivot' in kwargs:
            pivot = kwargs.pop('pivot')
        else:
            pivot = 'middle'

        # Calculate specific discharge
        # make sure dis is defined
        if dis is None:
            try:
                dis = self.dis
            except:
                try:
                    dis = self.model.dis
                except:
                    raise AssertionError("ModelMap.plot_quiver() error: self.dis is None and dis arg is None ")

        ja = np.array(ja)
        top = np.copy(dis.top)
        botm = np.copy(dis.botm)
        nlay, ncpl = botm.shape
        delr = np.tile([np.max(i) - np.min(i) for i in dis.yvert], (nlay, 1))
        delc = np.tile([np.max(i) - np.min(i) for i in dis.xvert], (nlay, 1))
        hnoflo = 999.
        hdry = 999.

        if len(head.shape) == 3:
            head = head[:,0]

        if len(ja.shape) == 4:
            ja = ja[0][0][0]

        if head is None:
            head = np.zeros(botm.shape, np.float32)

        sat_thk = plotutil.saturated_thickness(head, top, botm,
                                               [hnoflo, hdry])

        frf, fff, flf = plotutil.vectorize_flow(ja, dis)

        frf.shape = (nlay, ncpl)
        fff.shape = (nlay, ncpl)
        flf.shape = (nlay, ncpl)
        # Calculate specific discharge
        qx, qy, qz = plotutil.specific_discharge(frf, fff, flf, delr,
                                                 delc, sat_thk)

        # Select correct slice
        u = qx[self.layer, :]
        v = qy[self.layer, :]
        # apply step
        x = self.dis.sr.xcenter_array[::istep]
        y = self.dis.sr.ycenter_array[::istep]
        u = u[::istep]
        v = v[::istep]
        # normalize
        if normalize:
            vmag = np.sqrt(u ** 2. + v ** 2.)
            idx = vmag > 0.
            u[idx] /= vmag[idx]
            v[idx] /= vmag[idx]

        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax

        # mask discharge in inactive cells
        idomain = dis.idomain
        idx = (idomain[self.layer, ::istep] == 0)
        idx[idomain[self.layer, ::istep] == -1] = 1

        u[idx] = np.nan
        v[idx] = np.nan

        # Rotate and plot
        urot, vrot = self.dis.sr.rotate(u, v, self.sr.rotation)
        quiver = ax.quiver(x, y, urot, vrot, scale=1, units='xy', pivot=pivot, **kwargs)

        return quiver

    def get_patch_collection(self, vertexdict, plotarray=None, mask=None, grid=False, **kwargs):
        """
        Method to create matplotlib.patches objects from verticies

        """
        from matplotlib.patches import Polygon
        from matplotlib.collections import PatchCollection

        if not grid:

            assert plotarray is not None, 'plotarray must be provided to get_patch_collection'
            assert mask is not None, 'mask must be provided to get_patch_collection'

            patches = []
            a = np.array([])
            for idx, key in enumerate(vertexdict):
                if not mask[idx]:
                    patch = np.array([[line[0], line[1]] for line in vertexdict[idx]])
                    patches.append(Polygon(patch, True))
                    a = np.append(a, plotarray[idx])

            p = PatchCollection(patches, **kwargs)
            p.set_array(a)

        else:
            # method to plot gridlines using the patch collection
            patches = []
            for key in vertexdict:
                patch = np.array([[line[0], line[1]] for line in vertexdict[key]])
                patches.append(Polygon(patch, True))

            p = PatchCollection(patches, **kwargs)

        return p


class ModelMap(object):
    """
        Dynamically determines inheritance class to create a map of the model.

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

        distype: string
            Discretation type to determine inheritance to the ModelMap plotting class
            if structured, ModelMap inherits from StructuredModelMap, if vertex ModelMap
            inherits from VertexModelMap
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
    def __new__(cls, sr=None, ax=None, model=None, dis=None, layer=0,
                extent=None, xul=None, yul=None, rotation=None, distype="structured"):
        #todo: programatically check the dis class type
        if distype == 'structured':
            new = StructuredModelMap.__new__(StructuredModelMap)
            new.__init__(sr=sr, ax=ax, model=model, dis=dis, layer=layer,
                         extent=extent, xul=xul, yul=yul, rotation=rotation)

        elif distype == 'vertex':
            new = VertexModelMap.__new__(VertexModelMap)
            new.__init___(sr=sr, ax=ax, model=model, dis=dis, layer=layer,
                          extent=extent)

        elif distype == 'unstructured':
            raise NotImplementedError('Unstructured grid not yet supported')

        else:
            raise TypeError('Discretation type {} not supported'.format(distype))

        return new

