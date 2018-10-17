import sys
import numpy as np
from ..plot.map import StructuredMapView
from ..plot.vmap import VertexMapView
from ..plot.umap import UnstructuredMapView
from ..plot.crosssection import StructuredCrossSection
from ..plot.vcrosssection import VertexCrossSection
from ..utils.reference import SpatialReferenceUnstructured
from ..plot import plotutil
from ..utils import geometry

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors
except ImportError:
    plt = None


class PlotMapView(object):
    """
    Class to create a map of the model. Delegates plotting
    functionality based on model grid type.

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
    def __init__(self, modelgrid=None, model=None, dis=None,  ax=None,
                 layer=0, extent=None, xul=None, yul=None, xll=None, yll=None,
                 rotation=0.):

        if plt is None:
            s = 'Could not import matplotlib.  Must install matplotlib ' + \
                ' in order to use ModelMap method'
            raise ImportError(s)

        # todo: remove SpatialReferenceUnstructured when an UnstructuredModelGrid
        # todo: has been made
        #if isinstance(modelgrid, SpatialReferenceUnstructured):
        #    tmp = "unstructured"
        #else:

        try:
            tmp = modelgrid.grid_type
            if not isinstance(tmp, str):
                tmp = "structured"
        except:
            tmp = "structured"

        if tmp == "structured":
            self.__cls = StructuredMapView(ax=ax, model=model, dis=dis,
                                           modelgrid=modelgrid, layer=layer,
                                           extent=extent, xul=xul,
                                           yul=yul, xll=xll, yll=yll, rotation=rotation)

        elif tmp == "unstructured":
            self.__cls = UnstructuredMapView(modelgrid=modelgrid, model=model,
                                             extent=extent)

        else:
            self.__cls = VertexMapView(ax=ax, model=model, dis=dis,
                                       modelgrid=modelgrid, layer=layer,
                                       extent=extent, xul=xul, yul=yul, xll=xll,
                                       yll=yll, rotation=rotation)

        self.model = self.__cls.model
        self.layer = self.__cls.layer
        self.dis = self.__cls.dis
        self.mg = self.__cls.mg
        self.ax = self.__cls.ax
        # self._extent = self.__cls._extent

    @property
    def extent(self):
        return self.__cls.extent

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
        return self.__cls.plot_array(a=a, masked_values=masked_values, **kwargs)

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
        return self.__cls.contour_array(a=a, masked_values=masked_values,
                                        **kwargs)

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
        if ibound is None:
            try:
                bas = self.model.get_package('BAS6')
                ibound = bas.ibound.array
            except AttributeError:
                ibound = self.mg.idomain

        plotarray = np.zeros(ibound.shape, dtype=np.int)
        idx1 = (ibound == 0)
        plotarray[idx1] = 1
        plotarray = np.ma.masked_equal(plotarray, 0)
        cmap = matplotlib.colors.ListedColormap(['0', color_noflow])
        bounds = [0, 1, 2]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        quadmesh = self.__cls.plot_array(plotarray, cmap=cmap, norm=norm, **kwargs)
        return quadmesh

    def plot_ibound(self, ibound=None, color_noflow='black', color_ch='blue',
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

        Returns
        -------
        quadmesh : matplotlib.collections.QuadMesh

        """
        import matplotlib.colors

        if ibound is None:
            try:
                bas = self.model.get_package('BAS6')
                ibound = bas.ibound.array
            except:
                ibound = self.mg.idomain
                color_ch = color_vpt

        plotarray = np.zeros(ibound.shape, dtype=np.int)
        idx1 = (ibound == 0)
        idx2 = (ibound < 0)
        plotarray[idx1] = 1
        plotarray[idx2] = 2
        plotarray = np.ma.masked_equal(plotarray, 0)
        cmap = matplotlib.colors.ListedColormap(['0', color_noflow, color_ch])
        bounds = [0, 1, 2, 3]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        quadmesh = self.__cls.plot_array(plotarray, cmap=cmap, norm=norm, **kwargs)
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
        from matplotlib.collections import LineCollection

        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.__cls.ax

        if 'colors' not in kwargs:
            kwargs['colors'] = '0.5'

        if isinstance(self.mg, SpatialReferenceUnstructured):
            lc = self.mg.get_grid_line_collection(**kwargs)
        else:
            lc = LineCollection(self.__cls.mg.grid_lines, **kwargs)

        ax.add_collection(lc)
        ax.set_xlim(self.__cls.extent[0], self.__cls.extent[1])
        ax.set_ylim(self.__cls.extent[2], self.__cls.extent[3])

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
        return self.__cls.plot_bc(ftype=ftype, package=package, kper=kper,
                                  color=color, plotAll=plotAll, **kwargs)

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
        Contour a cvfd array.  If the array is three-dimensional, then the method
        will contour the layer tied to this class (self.layer). The vertices
        must be in the same coordinates as the rotated and offset grid.

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
                ncpl = np.ones((nlay,), dtype=np.int) * i
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
                kwargs.pop('cmap')

        contour_set = ax.tricontour(vertc[:, 0], vertc[:, 1],
                                    plotarray, **kwargs)

        return contour_set

    def plot_discharge(self, frf=None, fff=None, fja=None, dis=None,
                       flf=None, head=None, istep=1, jstep=1,
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
        if self.mg.grid_type == "vertex":
            return self.__cls.plot_discharge(fja=fja, dis=dis, head=head, istep=istep,
                                             normalize=normalize, **kwargs)

        elif self.mg.grid_type == "unstructured":
            return self.__cls.plot_discharge()

        else:
            return self.__cls.plot_discharge(frf=frf, fff=fff, dis=dis, flf=flf, head=head,
                                             istep=istep, jstep=jstep, normalize=normalize,
                                             **kwargs)

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

            # transform data!
            if isinstance(self.mg, SpatialReferenceUnstructured):
                x0r, y0r = geometry.transform(tp['x'], tp['y'],
                                              self.mg.xul,
                                              self.mg.yul,
                                              self.mg.rotation)
            else:
                x0r, y0r = geometry.transform(tp['x'], tp['y'],
                                              self.mg.xoffset,
                                              self.mg.yoffset,
                                              self.mg.angrot_radians)
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
                        idx = (t['time'] <= time)
                    elif '<' in travel_time:
                        time = float(travel_time.replace('<', ''))
                        idx = (t['time'] < time)
                    elif '>=' in travel_time:
                        time = float(travel_time.replace('>=', ''))
                        idx = (t['time'] >= time)
                    elif '<' in travel_time:
                        time = float(travel_time.replace('>', ''))
                        idx = (t['time'] > time)
                    else:
                        try:
                            time = float(travel_time)
                            idx = (t['time'] <= time)
                        except:
                            errmsg = 'flopy.map.plot_pathline travel_time ' + \
                                     'variable cannot be parsed. ' + \
                                     'Acceptable logical variables are , ' + \
                                     '<=, <, >=, and >. ' + \
                                     'You passed {}'.format(travel_time)
                            raise Exception(errmsg)
                else:
                    time = float(travel_time)
                    idx = (t['time'] <= time)
                tp = ts[idx]

            if isinstance(self.mg, SpatialReferenceUnstructured):
                x0r, y0r = geometry.transform(tp['x'], tp['y'],
                                              self.mg.xul,
                                              self.mg.yul,
                                              self.mg.rotation)
            else:
                x0r, y0r = geometry.transform(tp['x'], tp['y'],
                                              self.mg.xoffset,
                                              self.mg.yoffset,
                                              self.mg.angrot_radians)

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
        # return self.__cls.plot_timeseries(ts=ts, travel_time=travel_time,
        #                                   **kwargs)

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
        direction = direction.lower()
        if direction == 'starting':
            xp, yp = 'x0', 'y0'

        elif direction == 'ending':
            xp, yp = 'x', 'y'

        else:
            errmsg = 'flopy.map.plot_endpoint direction must be "ending" ' + \
                     'or "starting".'
            raise Exception(errmsg)

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

        # transform data!
        if isinstance(self.mg, SpatialReferenceUnstructured):
            x0r, y0r = geometry.transform(tep[xp], tep[yp],
                                          self.mg.xul,
                                          self.mg.yul,
                                          self.mg.rotation)
        else:
            x0r, y0r = geometry.transform(tep[xp], tep[yp],
                                          self.mg.xoffset,
                                          self.mg.yoffset,
                                          self.mg.angrot_radians)
        # build array to plot
        arr = np.vstack((x0r, y0r)).T

        # plot the end point data
        sp = ax.scatter(arr[:, 0], arr[:, 1], c=c, s=s, **kwargs)

        # add a colorbar for travel times
        if createcb:
            cb = plt.colorbar(sp, ax=ax, shrink=shrink)
            cb.set_label(colorbar_label)
        return sp


class PlotCrossSection(object):
    """
    Class to create a cross section of the model.

    Parameters
    ----------
    ax : matplotlib.pyplot axis
        The plot axis.  If not provided it, plt.gca() will be used.
    model : flopy.modflow object
        flopy model object. (Default is None)
    dis : flopy.modflow.ModflowDis object
        flopy discretization object. (Default is None)
    line : dict
        Dictionary with either "row", "column", or "line" key. If key
        is "row" or "column" key value should be the zero-based row or
        column index for cross-section. If key is "line" value should
        be an array of (x, y) tuples with vertices of cross-section.
        Vertices should be in map coordinates consistent with xul,
        yul, and rotation.
    xul : float
        x coordinate for upper left corner
    yul : float
        y coordinate for upper left corner.  The default is the sum of the
        delc array.
    rotation : float
        Angle of grid rotation around the upper left corner.  A positive value
        indicates clockwise rotation.  Angles are in degrees. Default is None
    extent : tuple of floats
        (xmin, xmax, ymin, ymax) will be used to specify axes limits.  If None
        then these will be calculated based on grid, coordinates, and rotation.

    """

    def __init__(self, ax=None, model=None, dis=None, modelgrid=None,
                 line=None, xul=None, yul=None, xll=None, yll=None,
                 rotation=0., extent=None, length_multiplier=1.):
        if plt is None:
            s = 'Could not import matplotlib.  Must install matplotlib ' + \
                ' in order to use ModelMap method'
            raise ImportError(s)

       # update this after unstructured grid is finished!
        try:
            tmp = modelgrid.grid_type
            if not isinstance(tmp, str):
                tmp = "structured"
        except:
            tmp = "structured"

        if tmp == "structured":
            self.__cls = StructuredCrossSection(ax=ax, model=model, dis=dis,
                                                modelgrid=modelgrid,
                                                line=line, xul=xul, yul=yul,
                                                xll=xll, yll=yll,
                                                rotation=rotation, extent=extent,
                                                length_multiplier=length_multiplier)

        elif tmp == "unstructured":
            raise NotImplementedError("Unstructured xc not yet implemented")

        else:
            self.__cls = VertexCrossSection(ax=ax, model=model, dis=dis,
                                            modelgrid=modelgrid,
                                            line=line, xul=xul, yul=yul,
                                            xll=xll, yll=yll, rotation=rotation,
                                            extent=extent, length_multiplier=length_multiplier)

        self.model = self.__cls.model
        self.dis = self.__cls.dis
        self.mg = self.__cls.mg
        self.ax = self.__cls.ax
        self.direction = self.__cls.direction
        self.pts = self.__cls.pts
        self.xpts = self.__cls.xpts
        self.d = self.__cls.d
        self.ncb = self.__cls.ncb
        self.laycbd = self.__cls.laycbd
        self.active = self.__cls.active
        self.elev = self.__cls.elev
        self.layer0 = self.__cls.layer0
        self.layer1 = self.__cls.layer1
        self.zpts = self.__cls.zpts
        self.xcentergrid = self.__cls.xcentergrid
        self.zcentergrid = self.__cls.zcentergrid
        self.extent = self.__cls.extent

    def plot_array(self, a, masked_values=None, head=None, **kwargs):
        """
        Plot a three-dimensional array as a patch collection.

        Parameters
        ----------
        a : numpy.ndarray
            Three-dimensional array to plot.
        masked_values : iterable of floats, ints
            Values to mask.
        head : numpy.ndarray
            Three-dimensional array to set top of patches to the minimum
            of the top of a layer or the head value. Used to create
            patches that conform to water-level elevations.
        **kwargs : dictionary
            keyword arguments passed to matplotlib.collections.PatchCollection

        Returns
        -------
        patches : matplotlib.collections.PatchCollection

        """
        return self.__cls.plot_array(a=a, masked_values=masked_values,
                                     head=head, **kwargs)

    def plot_surface(self, a, masked_values=None, **kwargs):
        """
        Plot a two- or three-dimensional array as line(s).

        Parameters
        ----------
        a : numpy.ndarray
            Two- or three-dimensional array to plot.
        masked_values : iterable of floats, ints
            Values to mask.
        **kwargs : dictionary
            keyword arguments passed to matplotlib.pyplot.plot

        Returns
        -------
        plot : list containing matplotlib.plot objects

        """
        return self.__cls.plot_surface(a=a, masked_values=masked_values,
                                       **kwargs)

    def plot_fill_between(self, a, colors=('blue', 'red'),
                          masked_values=None, head=None, **kwargs):
        """
        Plot a three-dimensional array as lines.

        Parameters
        ----------
        a : numpy.ndarray
            Three-dimensional array to plot.
        colors: list
            matplotlib fill colors, two required
        masked_values : iterable of floats, ints
            Values to mask.
        head : numpy.ndarray
            Three-dimensional array to set top of patches to the minimum
            of the top of a layer or the head value. Used to create
            patches that conform to water-level elevations.
        **kwargs : dictionary
            keyword arguments passed to matplotlib.pyplot.plot

        Returns
        -------
        plot : list containing matplotlib.fillbetween objects

        """
        return self.__cls.plot_fill_between(a=a, colors=colors,
                                            masked_values=masked_values,
                                            head=head, **kwargs)

    def contour_array(self, a, masked_values=None, head=None, **kwargs):
        """
        Contour a three-dimensional array.

        Parameters
        ----------
        a : numpy.ndarray
            Three-dimensional array to plot.
        masked_values : iterable of floats, ints
            Values to mask.
        head : numpy.ndarray
            Three-dimensional array to set top of patches to the minimum
            of the top of a layer or the head value. Used to create
            patches that conform to water-level elevations.
        **kwargs : dictionary
            keyword arguments passed to matplotlib.pyplot.contour

        Returns
        -------
        contour_set : matplotlib.pyplot.contour

        """
        return self.__cls.contour_array(a=a, masked_values=masked_values,
                                        head=head, **kwargs)

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
        return self.__cls.plot_inactive(ibound=ibound, color_noflow=color_noflow,
                                        **kwargs)

    def plot_ibound(self, ibound=None, color_noflow='black', color_ch='blue',
                    head=None, **kwargs):
        """
        Make a plot of ibound.  If not specified, then pull ibound from the
        self.model

        Parameters
        ----------
        ibound : numpy.ndarray
            ibound array to plot.  (Default is ibound in 'BAS6' package.)
        color_noflow : string
            (Default is 'black')
        color_ch : string
            Color for constant heads (Default is 'blue'.)
        head : numpy.ndarray
            Three-dimensional array to set top of patches to the minimum
            of the top of a layer or the head value. Used to create
            patches that conform to water-level elevations.
        **kwargs : dictionary
            keyword arguments passed to matplotlib.collections.PatchCollection

        Returns
        -------
        patches : matplotlib.collections.PatchCollection

        """
        return self.__cls.plot_ibound(ibound=ibound, color_noflow=color_noflow,
                                      color_ch=color_ch, head=head, **kwargs)

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
        return self.__cls.plot_grid(**kwargs)

    def plot_bc(self, ftype=None, package=None, kper=0, color=None,
                head=None, **kwargs):
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
        head : numpy.ndarray
            Three-dimensional array to set top of patches to the minimum
            of the top of a layer or the head value. Used to create
            patches that conform to water-level elevations.
        **kwargs : dictionary
            keyword arguments passed to matplotlib.collections.PatchCollection

        Returns
        -------
        patches : matplotlib.collections.PatchCollection

        """
        return self.__cls.plot_bc(ftype=ftype, package=package, kper=kper,
                                  color=color, head=head, **kwargs)

    def plot_discharge(self, frf=None, fff=None, flf=None, fja=None,
                       head=None, kstep=1, hstep=1, normalize=False,
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
        kstep : int
            layer frequency to plot. (Default is 1.)
        hstep : int
            horizontal frequency to plot. (Default is 1.)
        normalize : bool
            boolean flag used to determine if discharge vectors should
            be normalized using the magnitude of the specific discharge in each
            cell. (default is False)
        kwargs : dictionary
            Keyword arguments passed to plt.quiver()

        Returns
        -------
        quiver : matplotlib.pyplot.quiver
            Vectors

        """
        if self.mg.grid_type == "vertex":
            return self.__cls.plot_discharge(fja=fja, head=head, kstep=kstep,
                                             hstep=hstep, normalize=normalize, **kwargs)

        elif self.mg.grid_type == "unstructured":
            raise self.__cls.plot_discharge()

        else:
            return self.__cls.plot_discharge(frf=frf, fff=fff, flf=flf, head=head,
                                             kstep=kstep, hstep=hstep, normalize=normalize,
                                             **kwargs)

    def get_grid_patch_collection(self, zpts, plotarray, **kwargs):
        """
        Get a PatchCollection of plotarray in unmasked cells

        Parameters
        ----------
        zpts : numpy.ndarray
            array of z elevations that correspond to the x, y, and horizontal
            distance along the cross-section (self.xpts). Constructed using
            plotutil.cell_value_points().
        plotarray : numpy.ndarray
            Three-dimensional array to attach to the Patch Collection.
        **kwargs : dictionary
            keyword arguments passed to matplotlib.collections.PatchCollection

        Returns
        -------
        patches : matplotlib.collections.PatchCollection

        """
        if self.mg.grid_type == "structured":
            return self.__cls.get_grid_patch_collection(zpts=zpts, plotarray=plotarray,
                                                        **kwargs)
        elif self.mg.grid_type == "unstructured":
            raise NotImplementedError()

        else:
            return self.__cls.get_grid_patch_collection(projpts=zpts, plotarray=plotarray,
                                                        **kwargs)

    def get_grid_line_collection(self, **kwargs):
        """
        Get a LineCollection of the grid

        Parameters
        ----------
        **kwargs : dictionary
            keyword arguments passed to matplotlib.collections.LineCollection

        Returns
        -------
        linecollection : matplotlib.collections.LineCollection
        """
        return self.__cls.get_grid_line_collection(**kwargs)

