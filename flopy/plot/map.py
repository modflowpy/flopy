import copy
import sys
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors
except:
    plt = None
from . import plotutil
from .plotutil import bc_color_dict
from ..utils import SpatialReference, SpatialReferenceUnstructured


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
                 extent=None, xul=None, yul=None, xll=None, yll=None,
                 rotation=0., length_multiplier=1.):
        if plt is None:
            s = 'Could not import matplotlib.  Must install matplotlib ' + \
                ' in order to use ModelMap method'
            raise Exception(s)

        self.model = model
        self.layer = layer
        self.dis = dis
        self.sr = None
        if sr is not None:
            self.sr = copy.deepcopy(sr)
        elif dis is not None:
            # print("warning: the dis arg to model map is deprecated")
            self.sr = copy.deepcopy(dis.parent.sr)
        elif model is not None:
            # print("warning: the model arg to model map is deprecated")
            self.sr = copy.deepcopy(model.sr)
        else:
            self.sr = SpatialReference(xll=xll, yll=yll, xul=xul, yul=yul,
                                       rotation=rotation,
                                       length_multiplier=length_multiplier)

        # model map override spatial reference settings
        if any(elem is not None for elem in (xul, yul, xll, yll)) or \
                rotation != 0 or length_multiplier != 1.:
            self.sr.length_multiplier = length_multiplier
            self.sr.set_spatialreference(xul, yul, xll, yll, rotation)

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
        elif a.ndim == 1:
            plotarray = a
        else:
            raise Exception('Array must be of dimension 1, 2 or 3')
        if masked_values is not None:
            for mval in masked_values:
                plotarray = np.ma.masked_equal(plotarray, mval)
        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax

        # quadmesh = ax.pcolormesh(self.sr.xgrid, self.sr.ygrid, plotarray,
        #                          **kwargs)
        quadmesh = self.sr.plot_array(plotarray, ax=ax)

        # set max and min
        if 'vmin' in kwargs:
            vmin = kwargs.pop('vmin')
        else:
            vmin = None
        if 'vmax' in kwargs:
            vmax = kwargs.pop('vmax')
        else:
            vmax = None
        quadmesh.set_clim(vmin=vmin, vmax=vmax)

        # send rest of kwargs to quadmesh
        quadmesh.set(**kwargs)

        # add collection to axis
        ax.add_collection(quadmesh)

        # set limits
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
        elif a.ndim == 1:
            plotarray = a
        else:
            raise Exception('Array must be of dimension 1, 2 or 3')
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
        # contour_set = ax.contour(self.sr.xcentergrid, self.sr.ycentergrid,
        #                         plotarray, **kwargs)
        contour_set = self.sr.contour_array(ax, plotarray, **kwargs)
        ax.set_xlim(self.extent[0], self.extent[1])
        ax.set_ylim(self.extent[2], self.extent[3])

        return contour_set

    def plot_inactive(self, ibound=None, color_noflow='black', **kwargs):
        """
        Make a plot of inactive cells.  If not specified, then pull ibound
        from the self.ml

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
            ibound = bas.ibound.array

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
            ibound = bas.ibound.array
        plotarray = np.zeros(ibound.shape, dtype=np.int)
        idx1 = (ibound == 0)
        idx2 = (ibound < 0)
        plotarray[idx1] = 1
        plotarray[idx2] = 2
        plotarray = np.ma.masked_equal(plotarray, 0)
        cmap = matplotlib.colors.ListedColormap(['0', color_noflow, color_ch])
        bounds = [0, 1, 2, 3]
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

        lc = self.sr.get_grid_line_collection(**kwargs)
        ax.add_collection(lc)
        ax.set_xlim(self.extent[0], self.extent[1])
        ax.set_ylim(self.extent[2], self.extent[3])

        return lc

    def plot_bc(self, ftype=None, package=None, kper=0, color=None,
                plotAll=False, **kwargs):
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
            ftype = p.name[0]
        elif self.model is not None:
            if ftype is None:
                raise Exception('ftype not specified')
            ftype = ftype.upper()
            p = self.model.get_package(ftype)
        else:
            raise Exception('Cannot find package to plot')

        # Get the list data
        try:
            mflist = p.stress_period_data[kper]
        except Exception as e:
            raise Exception('Not a list-style boundary package:' + str(e))

        # Return if MfList is None
        if mflist is None:
            return None
        nlay = self.model.nlay
        # Plot the list locations
        plotarray = np.zeros((nlay, self.sr.nrow, self.sr.ncol), dtype=np.int)
        if plotAll:
            idx = (mflist['i'], mflist['j'])
            # plotarray[:, idx] = 1
            pa = np.zeros((self.sr.nrow, self.sr.ncol), dtype=np.int)
            pa[idx] = 1
            for k in range(nlay):
                plotarray[k, :, :] = pa.copy()
        else:
            idx = (mflist['k'], mflist['i'], mflist['j'])
            plotarray[idx] = 1

        # mask the plot array
        plotarray = np.ma.masked_equal(plotarray, 0)

        # set the colormap
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

        # create normalized quadmesh
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

    def plot_cvfd(self, verts, iverts, **kwargs):
        """
        Plot a cvfd grid.  The vertices must be in the same coordinates as
        the rotated and offset grid.

        Parameters
        ----------
        verts : ndarray
            2d array of x and y points.
        iverts : list of lists
            should be of len(ncells) with a list of vertex number for each cell

        kwargs : dictionary
            Keyword arguments passed to plotutil.plot_cvfd()

        """
        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax
        patch_collection = plotutil.plot_cvfd(verts, iverts, ax, self.layer,
                                              **kwargs)
        return patch_collection

    def contour_array_cvfd(self, vertc, a, masked_values=None, **kwargs):
        """
        Contour an array.  If the array is three-dimensional, then the method
        will contour the layer tied to this class (self.layer).

        Parameters
        ----------
        vertc : np.ndarray
            Array with of size (nc, 2) with centroid location of cvfd
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
        if 'ncpl' in kwargs:
            nlay = self.layer + 1
            ncpl = kwargs.pop('ncpl')
            if isinstance(ncpl, int):
                i = int(ncpl)
                ncpl = np.ones((nlay), dtype=np.int) * i
            elif isinstance(ncpl, list) or isinstance(ncpl, tuple):
                ncpl = np.array(ncpl)
            i0 = 0
            i1 = 0
            for k in range(nlay):
                i0 = i1
                i1 = i0 + ncpl[k]
            # retain vertc in selected layer
            vertc = vertc[i0:i1, :]
        else:
            i0 = 0
            i1 = vertc.shape[0]

        plotarray = a[i0:i1]

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
        contour_set = ax.tricontour(vertc[:, 0], vertc[:, 1],
                                    plotarray, **kwargs)

        return contour_set

    def plot_discharge(self, frf, fff, dis=None, flf=None, head=None, istep=1,
                       jstep=1, normalize=False, **kwargs):
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
            if self.model is not None:
                dis = self.model.dis
            else:
                print('ModelMap.plot_quiver() error: self.dis is None and dis '
                      'arg is None.')
                return
        ib = self.model.bas6.ibound.array
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
            vmag = np.sqrt(u ** 2. + v ** 2.)
            idx = vmag > 0.
            u[idx] /= vmag[idx]
            v[idx] /= vmag[idx]

        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax

        # mask discharge in inactive cells
        idx = (ib[self.layer, ::istep, ::jstep] == 0)
        u[idx] = np.nan
        v[idx] = np.nan

        # Rotate and plot
        urot, vrot = self.sr.rotate(u, v, self.sr.rotation)
        quiver = ax.quiver(x, y, urot, vrot, pivot=pivot, **kwargs)

        return quiver

    def plot_pathline(self, pl, travel_time=None, **kwargs):
        """
        Plot the MODPATH pathlines.

        Parameters
        ----------
        pl : list of rec arrays or a single rec array
            rec array or list of rec arrays is data returned from
            modpathfile PathlineFile get_data() or get_alldata()
            methods. Data in rec array is 'x', 'y', 'z', 'time',
            'k', and 'particleid'.
        travel_time: float or str
            travel_time is a travel time selection for the displayed
            pathlines. If a float is passed then pathlines with times
            less than or equal to the passed time are plotted. If a
            string is passed a variety logical constraints can be added
            in front of a time value to select pathlines for a select
            period of time. Valid logical constraints are <=, <, >=, and
            >. For example, to select all pathlines less than 10000 days
            travel_time='< 10000' would be passed to plot_pathline.
            (default is None)
        kwargs : layer, ax, colors.  The remaining kwargs are passed
            into the LineCollection constructor. If layer='all',
            pathlines are output for all layers

        Returns
        -------
        lc : matplotlib.collections.LineCollection

        """
        from matplotlib.collections import LineCollection

        # make sure pathlines is a list
        if not isinstance(pl, list):
            pl = [pl]

        if 'layer' in kwargs:
            kon = kwargs.pop('layer')
            if sys.version_info[0] > 2:
                if isinstance(kon, bytes):
                    kon = kon.decode()
            if isinstance(kon, str):
                if kon.lower() == 'all':
                    kon = -1
                else:
                    kon = self.layer
        else:
            kon = self.layer

        if 'marker' in kwargs:
            marker = kwargs.pop('marker')
        else:
            marker = None

        if 'markersize' in kwargs:
            markersize = kwargs.pop('markersize')
        elif 'ms' in kwargs:
            markersize = kwargs.pop('ms')
        else:
            markersize = None

        if 'markercolor' in kwargs:
            markercolor = kwargs.pop('markercolor')
        else:
            markercolor = None

        if 'markerevery' in kwargs:
            markerevery = kwargs.pop('markerevery')
        else:
            markerevery = 1

        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax

        if 'colors' not in kwargs:
            kwargs['colors'] = '0.5'

        linecol = []
        markers = []
        for p in pl:
            if travel_time is None:
                tp = p.copy()
            else:
                if isinstance(travel_time, str):
                    if '<=' in travel_time:
                        time = float(travel_time.replace('<=', ''))
                        idx = (p['time'] <= time)
                    elif '<' in travel_time:
                        time = float(travel_time.replace('<', ''))
                        idx = (p['time'] < time)
                    elif '>=' in travel_time:
                        time = float(travel_time.replace('>=', ''))
                        idx = (p['time'] >= time)
                    elif '<' in travel_time:
                        time = float(travel_time.replace('>', ''))
                        idx = (p['time'] > time)
                    else:
                        try:
                            time = float(travel_time)
                            idx = (p['time'] <= time)
                        except:
                            errmsg = 'flopy.map.plot_pathline travel_time ' + \
                                     'variable cannot be parsed. ' + \
                                     'Acceptable logical variables are , ' + \
                                     '<=, <, >=, and >. ' + \
                                     'You passed {}'.format(travel_time)
                            raise Exception(errmsg)
                else:
                    time = float(travel_time)
                    idx = (p['time'] <= time)
                tp = p[idx]

            vlc = []
            # rotate data
            if isinstance(self.sr, SpatialReferenceUnstructured):
                x0r, y0r = self.sr.rotate(tp['x'], tp['y'], self.sr.rotation,
                                          0., 0.)
                x0r += self.sr.xul
                y0r += self.sr.yul
            elif isinstance(self.sr, SpatialReference):
                x0r, y0r = self.sr.rotate(tp['x'], tp['y'], self.sr.rotation,
                                          0., self.sr.yedge[0])
                x0r += self.sr.xul
                y0r += self.sr.yul - self.sr.yedge[0]

            # build polyline array
            arr = np.vstack((x0r, y0r)).T
            # select based on layer
            if kon >= 0:
                kk = p['k'].copy().reshape(p.shape[0], 1)
                kk = np.repeat(kk, 2, axis=1)
                arr = np.ma.masked_where((kk != kon), arr)
            else:
                arr = np.ma.asarray(arr)
            # append line to linecol if there is some unmasked segment
            if not arr.mask.all():
                linecol.append(arr)
                if marker is not None:
                    for xy in arr[::markerevery]:
                        if not xy.mask:
                            markers.append(xy)
        # create line collection
        lc = None
        if len(linecol) > 0:
            lc = LineCollection(linecol, **kwargs)
            ax.add_collection(lc)
            if marker is not None:
                markers = np.array(markers)
                ax.plot(markers[:, 0], markers[:, 1], lw=0, marker=marker,
                        color=markercolor, ms=markersize)
        return lc

    def plot_endpoint(self, ep, direction='ending',
                      selection=None, selection_direction=None, **kwargs):
        """
        Plot the MODPATH endpoints.

        Parameters
        ----------
        ep : rec array
            A numpy recarray with the endpoint particle data from the
            MODPATH 6 endpoint file
        direction : str
            String defining if starting or ending particle locations should be
            considered. (default is 'ending')
        selection : tuple
            tuple that defines the zero-base layer, row, column location
            (l, r, c) to use to make a selection of particle endpoints.
            The selection could be a well location to determine capture zone
            for the well. If selection is None, all particle endpoints for
            the user-sepcified direction will be plotted. (default is None)
        selection_direction : str
            String defining is a selection should be made on starting or
            ending particle locations. If selection is not None and
            selection_direction is None, the selection direction will be set
            to the opposite of direction. (default is None)

        kwargs : ax, c, s or size, colorbar, colorbar_label, shrink. The
            remaining kwargs are passed into the matplotlib scatter
            method. If colorbar is True a colorbar will be added to the plot.
            If colorbar_label is passed in and colorbar is True then
            colorbar_label will be passed to the colorbar set_label()
            method. If shrink is passed in and colorbar is True then
            the colorbar size will be set using shrink.

        Returns
        -------
        sp : matplotlib.pyplot.scatter

        """
        if direction.lower() == 'ending':
            direction = 'ending'
        elif direction.lower() == 'starting':
            direction = 'starting'
        else:
            errmsg = 'flopy.map.plot_endpoint direction must be "ending" ' + \
                     'or "starting".'
            raise Exception(errmsg)

        if direction == 'starting':
            xp, yp = 'x0', 'y0'
        elif direction == 'ending':
            xp, yp = 'x', 'y'

        if selection_direction is not None:
            if selection_direction.lower() != 'starting' and \
                    selection_direction.lower() != 'ending':
                errmsg = 'flopy.map.plot_endpoint selection_direction ' + \
                         'must be "ending" or "starting".'
                raise Exception(errmsg)
        else:
            if direction.lower() == 'starting':
                selection_direction = 'ending'
            elif direction.lower() == 'ending':
                selection_direction = 'starting'

        # selection of endpoints
        if selection is not None:
            if isinstance(selection, int):
                selection = tuple((selection,))
            try:
                if len(selection) == 1:
                    node = selection[0]
                    if selection_direction.lower() == 'starting':
                        nsel = 'node0'
                    else:
                        nsel = 'node'
                    # make selection
                    idx = (ep[nsel] == node)
                    tep = ep[idx]
                elif len(selection) == 3:
                    k, i, j = selection[0], selection[1], selection[2]
                    if selection_direction.lower() == 'starting':
                        ksel, isel, jsel = 'k0', 'i0', 'j0'
                    else:
                        ksel, isel, jsel = 'k', 'i', 'j'
                    # make selection
                    idx = (ep[ksel] == k) & (ep[isel] == i) & (ep[jsel] == j)
                    tep = ep[idx]
                else:
                    errmsg = 'flopy.map.plot_endpoint selection must be ' + \
                             'a zero-based layer, row, column tuple ' + \
                             '(l, r, c) or node number (MODPATH 7) of ' + \
                             'the location to evaluate (i.e., well location).'
                    raise Exception(errmsg)
            except:
                errmsg = 'flopy.map.plot_endpoint selection must be a ' + \
                         'zero-based layer, row, column tuple (l, r, c) ' + \
                         'or node number (MODPATH 7) of the location ' + \
                         'to evaluate (i.e., well location).'
                raise Exception(errmsg)
        # all endpoints
        else:
            tep = ep.copy()

        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax

        # scatter kwargs that users may redefine
        if 'c' not in kwargs:
            c = tep['time'] - tep['time0']
        else:
            c = np.empty((tep.shape[0]), dtype="S30")
            c.fill(kwargs.pop('c'))

        s = 50
        if 's' in kwargs:
            s = float(kwargs.pop('s')) ** 2.
        elif 'size' in kwargs:
            s = float(kwargs.pop('size')) ** 2.

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

        # rotate data
        if isinstance(self.sr, SpatialReferenceUnstructured):
            x0r, y0r = self.sr.rotate(tep[xp], tep[yp], self.sr.rotation,
                                      0., 0.)
            x0r += self.sr.xul
            y0r += self.sr.yul
        elif isinstance(self.sr, SpatialReference):
            x0r, y0r = self.sr.rotate(tep[xp], tep[yp], self.sr.rotation,
                                      0., self.sr.yedge[0])
            x0r += self.sr.xul
            y0r += self.sr.yul - self.sr.yedge[0]

        # build array to plot
        arr = np.vstack((x0r, y0r)).T

        # plot the end point data
        sp = ax.scatter(arr[:, 0], arr[:, 1], c=c, s=s, **kwargs)

        # add a colorbar for travel times
        if createcb:
            cb = plt.colorbar(sp, ax=ax, shrink=shrink)
            cb.set_label(colorbar_label)
        return sp

    def plot_timeseries(self, ts, travel_time=None, **kwargs):
        """
        Plot the MODPATH timeseries.

        Parameters
        ----------
        ts : list of rec arrays or a single rec array
            rec array or list of rec arrays is data returned from
            modpathfile TimeseriesFile get_data() or get_alldata()
            methods. Data in rec array is 'x', 'y', 'z', 'time',
            'k', and 'particleid'.
        travel_time: float or str
            travel_time is a travel time selection for the displayed
            pathlines. If a float is passed then pathlines with times
            less than or equal to the passed time are plotted. If a
            string is passed a variety logical constraints can be added
            in front of a time value to select pathlines for a select
            period of time. Valid logical constraints are <=, <, >=, and
            >. For example, to select all pathlines less than 10000 days
            travel_time='< 10000' would be passed to plot_pathline.
            (default is None)
        kwargs : layer, ax, colors.  The remaining kwargs are passed
            into the LineCollection constructor. If layer='all',
            pathlines are output for all layers

        Returns
        -------
        lo : list of Line2D objects

        """
        from matplotlib.collections import LineCollection
        # make sure timeseries is a list
        if not isinstance(ts, list):
            ts = [ts]

        if 'layer' in kwargs:
            kon = kwargs.pop('layer')
            if sys.version_info[0] > 2:
                if isinstance(kon, bytes):
                    kon = kon.decode()
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

        if 'color' not in kwargs:
            kwargs['color'] = 'red'

        linecol = []
        for t in ts:
            if travel_time is None:
                tp = t.copy()
            else:
                if isinstance(travel_time, str):
                    if '<=' in travel_time:
                        time = float(travel_time.replace('<=', ''))
                        idx = (ts['time'] <= time)
                    elif '<' in travel_time:
                        time = float(travel_time.replace('<', ''))
                        idx = (ts['time'] < time)
                    elif '>=' in travel_time:
                        time = float(travel_time.replace('>=', ''))
                        idx = (ts['time'] >= time)
                    elif '<' in travel_time:
                        time = float(travel_time.replace('>', ''))
                        idx = (ts['time'] > time)
                    else:
                        try:
                            time = float(travel_time)
                            idx = (ts['time'] <= time)
                        except:
                            errmsg = 'flopy.map.plot_pathline travel_time ' + \
                                     'variable cannot be parsed. ' + \
                                     'Acceptable logical variables are , ' + \
                                     '<=, <, >=, and >. ' + \
                                     'You passed {}'.format(travel_time)
                            raise Exception(errmsg)
                else:
                    time = float(travel_time)
                    idx = (ts['time'] <= time)
                tp = ts[idx]

            # rotate data
            if isinstance(self.sr, SpatialReferenceUnstructured):
                x0r, y0r = self.sr.rotate(tp['x'], tp['y'], self.sr.rotation,
                                          0., 0.)
                x0r += self.sr.xul
                y0r += self.sr.yul
            elif isinstance(self.sr, SpatialReference):
                x0r, y0r = self.sr.rotate(tp['x'], tp['y'], self.sr.rotation,
                                          0., self.sr.yedge[0])
                x0r += self.sr.xul
                y0r += self.sr.yul - self.sr.yedge[0]

            # build polyline array
            arr = np.vstack((x0r, y0r)).T
            # select based on layer
            if kon >= 0:
                kk = t['k'].copy().reshape(t.shape[0], 1)
                kk = np.repeat(kk, 2, axis=1)
                arr = np.ma.masked_where((kk != kon), arr)
            else:
                arr = np.ma.asarray(arr)
            # append line to linecol if there is some unmasked segment
            if not arr.mask.all():
                linecol.append(arr)

        # plot timeseries data
        lo = []
        for lc in linecol:
            if not lc.mask.all():
                lo += ax.plot(lc[:, 0], lc[:, 1], **kwargs)

        return lo
