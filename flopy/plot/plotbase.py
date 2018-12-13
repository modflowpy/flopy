import sys
import numpy as np
from ..plot.map import _StructuredMapView
from ..plot.vmap import _VertexMapView
from ..plot.umap import _UnstructuredMapView
from ..plot.crosssection import _StructuredCrossSection
from ..plot.vcrosssection import _VertexCrossSection
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
    modelgrid : flopy.discretiztion.Grid
        The modelgrid class can be StructuredGrid, VertexGrid,
        or UnstructuredGrid (Default is None)
    ax : matplotlib.pyplot axis
        The plot axis.  If not provided it, plt.gca() will be used.
        If there is not a current axis then a new one will be created.
    model : flopy.modflow object
        flopy model object. (Default is None)
    layer : int
        Layer to plot.  Default is 0.  Must be between 0 and nlay - 1.
    extent : tuple of floats
        (xmin, xmax, ymin, ymax) will be used to specify axes limits.  If None
        then these will be calculated based on grid, coordinates, and rotation.

    Notes
    -----


    """
    def __init__(self, model=None, modelgrid=None, ax=None,
                 layer=0, extent=None):

        if plt is None:
            s = 'Could not import matplotlib.  Must install matplotlib ' + \
                ' in order to use ModelMap method'
            raise ImportError(s)


        if modelgrid is None and model is not None:
            modelgrid = model.modelgrid

        tmp = modelgrid.grid_type

        if tmp == "structured":
            self.__cls = _StructuredMapView(ax=ax, model=model,
                                            modelgrid=modelgrid, layer=layer,
                                            extent=extent)

        elif tmp == "unstructured":
            self.__cls = _UnstructuredMapView(ax=ax, modelgrid=modelgrid,
                                              model=model,
                                              layer=layer, extent=extent)

        elif tmp == "vertex":
            self.__cls = _VertexMapView(ax=ax, model=model,
                                        modelgrid=modelgrid, layer=layer,
                                        extent=extent)

        else:
            raise TypeError("Unrecognized grid type {}".format(tmp))

        self.model = self.__cls.model
        self.layer = self.__cls.layer
        self.mg = self.__cls.mg
        self.ax = self.__cls.ax

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
            if self.mg.idomain is None:
                raise AssertionError("Ibound/Idomain array must be provided")

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
            ibound array to plot.  (Default is ibound in the modelgrid)
        color_noflow : string
            (Default is 'black')
        color_ch : string
            Color for constant heads (Default is 'blue'.)
        color_vpt: string
            Color for vertical pass through cells (Default is 'red')

        Returns
        -------
        quadmesh : matplotlib.collections.QuadMesh

        """
        import matplotlib.colors

        if ibound is None:
            if self.model is not None:
                if self.model.version == "mf6":
                    color_ch = color_vpt

            if self.mg.idomain is None:
                raise AssertionError("Ibound/Idomain array must be provided")

            ibound = self.mg.idomain

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
        shp : string or pyshp shapefile object
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

    def plot_specific_discharge(self, spdis, istep=1,
                                jstep=1, normalize=False, **kwargs):
        """
        Method to plot specific discharge from discharge vectors
        provided by the cell by cell flow output file. In MODFLOW-6
        this option is controled in the NPF options block. This method
        uses matplotlib quiver to create a matplotlib plot of the output.

        Parameters:
        ----------
            qx: (np.ndarray)
            qy: (np.ndarray)
            qz: (np.ndarray)
            istep: (int) row frequency to plot. (Default is 1.)
            jstep: (int) column frequency to plot. (Default is 1.)
            kwargs: matplotlib.pyplot keyword arguments for the
                plt.quiver method.

        Returns:
            quiver: (matplotlib.pyplot.quiver) quiver plot of discharge vectors
        """
        if 'pivot' in kwargs:
            pivot = kwargs.pop('pivot')
        else:
            pivot = 'middle'

        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax

        if isinstance(spdis, list):
            print("Warning: Selecting the final stress period from Specific"
                  " Discharge list")
            spdis = spdis[-1]

        if self.mg.grid_type == "structured":
            ncpl = self.mg.nrow * self.mg.ncol

        else:
            ncpl = self.mg.ncpl

        nlay = self.mg.nlay

        qx = np.zeros((nlay * ncpl))
        qy = np.zeros((nlay * ncpl))

        idx = np.array(spdis['node']) - 1
        qx[idx] = spdis['qx']
        qy[idx] = spdis["qy"]

        if self.mg.grid_type == "structured":
            qx.shape = (self.mg.nlay, self.mg.nrow, self.mg.ncol)
            qy.shape = (self.mg.nlay, self.mg.nrow, self.mg.ncol)
            x = self.mg.xcellcenters[::istep, ::jstep]
            y = self.mg.ycellcenters[::istep, ::jstep]
            u = qx[:, ::istep, ::jstep]
            v = qy[:, ::istep, ::jstep]
        else:
            qx.shape = (self.mg.nlay, self.mg.ncpl)
            qy.shape = (self.mg.nlay, self.mg.ncpl)
            x = self.mg.xcellcenters[::istep]
            y = self.mg.ycellcenters[::istep]
            u = qx[:, ::istep]
            v = qy[:, ::istep]

        # normalize
        if normalize:
            vmag = np.sqrt(u ** 2. + v ** 2.)
            idx = vmag > 0.
            u[idx] /= vmag[idx]
            v[idx] /= vmag[idx]

        u[u == 0] = np.nan
        v[v == 0] = np.nan

        u = u[self.layer, :]
        v = v[self.layer, :]
        # Rotate and plot, offsets must be zero since
        # these are vectors not locations
        urot, vrot = geometry.rotate(u, v, 0., 0.,
                                     self.mg.angrot_radians)
        quiver = ax.quiver(x, y, urot, vrot, pivot=pivot, **kwargs)
        return quiver

    def plot_discharge(self, frf=None, fff=None,
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
        dis : flopy.modflow.ModflowDis package
            Depricated parameter
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
        if self.mg.grid_type != "structured":
            err_msg = "Use plot_specific_discharge for " \
                      "{} grids".format(self.mg.grid_type)
            raise NotImplementedError(err_msg)

        else:
            if self.mg.top is None:
                err = "StructuredModelGrid must have top and " \
                      "botm defined to use plot_discharge()"
                raise AssertionError(err)

            ib = np.ones((self.mg.nlay, self.mg.nrow, self.mg.ncol))
            if self.mg.idomain is not None:
                ib = self.mg.idomain

            delr = self.mg.delr
            delc = self.mg.delc
            top = np.copy(self.mg.top)
            botm = np.copy(self.mg.botm)
            nlay, nrow, ncol = botm.shape
            laytyp = None
            hnoflo = 999.
            hdry = 999.

            if self.model is not None:
                if self.model.laytyp is not None:
                    laytyp = self.model.laytyp

                if self.model.hnoflo is not None:
                    hnoflo = self.model.hnoflo

                if self.model.hdry is not None:
                    hdry = self.model.hdry

            # If no access to head or laytyp, then calculate confined saturated
            # thickness by setting laytyp to zeros
            if head is None or laytyp is None:
                head = np.zeros(botm.shape, np.float32)
                laytyp = np.zeros((nlay,), dtype=np.int)

            # calculate the saturated thickness
            sat_thk = plotutil.PlotUtilities. \
                saturated_thickness(head, top, botm, laytyp,
                                    [hnoflo, hdry])

            # Calculate specific discharge
            qx, qy, qz = plotutil.PlotUtilities. \
                centered_specific_discharge(frf, fff, flf, delr,
                                            delc, sat_thk)
            ib = ib.ravel()
            qx = qx.ravel()
            qy = qy.ravel()

            temp = []
            for ix, val in enumerate(ib):
                if val != 0:
                    temp.append((ix + 1, qx[ix], qy[ix]))

            spdis = np.recarray((len(temp),), dtype=[('node', np.int),
                                                     ("qx", np.float),
                                                     ("qy", np.float)])
            for ix, tup in enumerate(temp):
                spdis[ix] = tup

            self.plot_specific_discharge(spdis, istep=istep, jstep=jstep,
                                         normalize=normalize, **kwargs)

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
        ep = ep.copy()
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


class DeprecatedMapView(PlotMapView):
    """
    Deprecation handler for the PlotMapView class
    """
    def __init__(self, model=None, modelgrid=None, ax=None,
                 layer=0, extent=None):
        super(DeprecatedMapView, self).__init__(model=model,
                                                modelgrid=modelgrid,
                                                ax=ax,
                                                layer=layer,
                                                extent=extent)

    def plot_discharge(self, frf, fff, dis=None,
                       flf=None, head=None, istep=1, jstep=1,
                       normalize=False, **kwargs):
        if dis is not None:
            self.__cls.mg = plotutil._depreciated_dis_handler(modelgrid=self.mg,
                                                              dis=dis)

        super(DeprecatedMapView, self).plot_discharge(frf=frf, fff=fff,
                                                      flf=flf, head=head,
                                                      istep=1, jstep=1,
                                                      normalize=normalize,
                                                      **kwargs)


class PlotCrossSection(object):
    """
    Class to create a cross section of the model.

    Parameters
    ----------
    ax : matplotlib.pyplot axis
        The plot axis.  If not provided it, plt.gca() will be used.
    model : flopy.modflow object
        flopy model object. (Default is None)
    modelgrid : flopy.discretization.Grid object
        can be a StructuredGrid, VertexGrid, or UnstructuredGrid object
    line : dict
        Dictionary with either "row", "column", or "line" key. If key
        is "row" or "column" key value should be the zero-based row or
        column index for cross-section. If key is "line" value should
        be an array of (x, y) tuples with vertices of cross-section.
        Vertices should be in map coordinates consistent with xul,
        yul, and rotation.
    extent : tuple of floats
        (xmin, xmax, ymin, ymax) will be used to specify axes limits.  If None
        then these will be calculated based on grid, coordinates, and rotation.

    """

    def __init__(self, model=None, modelgrid=None, ax=None,
                 line=None, extent=None):
        if plt is None:
            s = 'Could not import matplotlib.  Must install matplotlib ' + \
                ' in order to use ModelMap method'
            raise ImportError(s)

        if modelgrid is None and model is not None:
            modelgrid = model.modelgrid

        # update this after unstructured grid is finished!
        tmp = modelgrid.grid_type

        if tmp == "structured":
            self.__cls = _StructuredCrossSection(ax=ax, model=model,
                                                 modelgrid=modelgrid,
                                                 line=line, extent=extent)

        elif tmp == "unstructured":
            raise NotImplementedError("Unstructured xc not yet implemented")

        elif tmp == "vertex":
            self.__cls = _VertexCrossSection(ax=ax, model=model,
                                             modelgrid=modelgrid,
                                             line=line, extent=extent)

        else:
            raise ValueError("Unknown modelgrid type {}".format(tmp))

        self.model = self.__cls.model
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
        if ibound is None:
            if self.mg.idomain is None:
                raise AssertionError("An idomain array must be provided")
            else:
                ibound = self.mg.idomain

        plotarray = np.zeros(ibound.shape, dtype=np.int)
        idx1 = (ibound == 0)
        plotarray[idx1] = 1
        plotarray = np.ma.masked_equal(plotarray, 0)
        cmap = matplotlib.colors.ListedColormap(['0', color_noflow])
        bounds = [0, 1, 2]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        patches = self.plot_array(plotarray, cmap=cmap, norm=norm, **kwargs)

        return patches

    def plot_ibound(self, ibound=None, color_noflow='black', color_ch='blue',
                    color_vpt="red", head=None, **kwargs):
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
        if ibound is None:
            if self.model is not None:
                if self.model.version == "mf6":
                    color_ch = color_vpt

            if self.mg.idomain is None:
                raise AssertionError("Ibound/Idomain array must be provided")

            ibound = self.mg.idomain

        plotarray = np.zeros(ibound.shape, dtype=np.int)
        idx1 = (ibound == 0)
        idx2 = (ibound < 0)
        plotarray[idx1] = 1
        plotarray[idx2] = 2
        plotarray = np.ma.masked_equal(plotarray, 0)
        cmap = matplotlib.colors.ListedColormap(['none', color_noflow,
                                                 color_ch])
        bounds = [0, 1, 2, 3]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        # mask active cells
        patches = self.plot_array(plotarray, masked_values=[0], head=head,
                                  cmap=cmap, norm=norm, **kwargs)
        return patches

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

        col = self.get_grid_line_collection(**kwargs)
        if col is not None:
            ax.add_collection(col)
            ax.set_xlim(self.extent[0], self.extent[1])
            ax.set_ylim(self.extent[2], self.extent[3])

        return col

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
            Three-dimensional array (structured grid) or
            Two-dimensional array (vertex grid)
            to set top of patches to the minimum of the top of a\
            layer or the head value. Used to create
            patches that conform to water-level elevations.
        **kwargs : dictionary
            keyword arguments passed to matplotlib.collections.PatchCollection

        Returns
        -------
        patches : matplotlib.collections.PatchCollection

        """
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
            arr_dict = p.stress_period_data.to_array(kper)
        except:
            raise Exception('Not a list-style boundary package')

        if not arr_dict:
            return None

        for key in arr_dict:
            fluxes = arr_dict[key]
            break

        # Plot the list locations
        if self.mg.grid_type == "vertex":
            plotarray = np.zeros((self.mg.nlay, self.mg.ncpl), dtype=np.int)
        else:
            plotarray = np.zeros((self.mg.nlay, self.mg.nrow, self.mg.ncol), dtype=np.int)

        plotarray[fluxes != 0] = 1

        plotarray = np.ma.masked_equal(plotarray, 0)
        if color is None:
            if ftype in plotutil.bc_color_dict:
                c = plotutil.bc_color_dict[ftype]
            else:
                c = plotutil.bc_color_dict['default']
        else:
            c = color
        cmap = matplotlib.colors.ListedColormap(['none', c])
        bounds = [0, 1, 2]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        patches = self.plot_array(plotarray, masked_values=[0],
                                  head=head, cmap=cmap, norm=norm, **kwargs)

        return patches

    def plot_specific_discharge(self, spdis, head=None, kstep=1,
                                hstep=1, normalize=False, **kwargs):
        """
        Use quiver to plot vectors.

        Parameters
        ----------
        spdis : np.recarray
            numpy recarray of specific discharge information. This
            can be grabbed directly from the CBC file if SAVE_SPECIFIC_DISCHARGE
            is used in the MF6 NPF file.
        head : numpy.ndarray
            MODFLOW's head array.  If not provided, then the quivers will be plotted
            in the cell center.
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
        if 'pivot' in kwargs:
            pivot = kwargs.pop('pivot')
        else:
            pivot = 'middle'

        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax

        if isinstance(spdis, list):
            print("Warning: Selecting the final stress period from Specific"
                  " Discharge list")
            spdis = spdis[-1]

        if self.mg.grid_type == "structured":
            ncpl = self.mg.nrow * self.mg.ncol

        else:
            ncpl = self.mg.ncpl

        nlay = self.mg.nlay

        qx = np.zeros((nlay * ncpl))
        qz = np.zeros((nlay * ncpl))
        ib = np.zeros((nlay * ncpl), dtype=bool)

        idx = np.array(spdis['node']) - 1

        # check that vertex grid cross sections are not arbitrary
        # within a tolerance!
        if self.mg.grid_type != 'structured':
            pts = self.pts
            xuniform = [True if abs(pts.T[0, 0] - i) < 1
                        else False for i in pts.T[0]]
            yuniform = [True if abs(pts.T[1, 0] - i) < 1
                        else False for i in pts.T[1]]
            if not np.all(xuniform):
                if not np.all(yuniform):
                    err_msg = "plot_specific_discharge does not " \
                              "support aribtrary cross sections"
                    raise AssertionError(err_msg)

        if self.direction == 'x':
            qx[idx] = spdis['qx']
        elif self.direction == 'y':
            qx[idx] = spdis['qy']
        else:
            err_msg = 'plot_specific_discharge does not ' \
                      'support arbitrary cross-sections'
            raise AssertionError(err_msg)

        qz[idx] = spdis["qz"]
        ib[idx] = True

        if self.mg.grid_type == "structured":
            qx.shape = (self.mg.nlay, self.mg.nrow, self.mg.ncol)
            qz.shape = (self.mg.nlay, self.mg.nrow, self.mg.ncol)
            ib.shape = (self.mg.nlay, self.mg.nrow, self.mg.ncol)

            if isinstance(head, np.ndarray):
                zcentergrid = self.__cls.set_zcentergrid(head)
            else:
                zcentergrid = self.zcentergrid

            if nlay == 1:
                x = []
                z = []
                for k in range(nlay):
                    for i in range(self.xcentergrid.shape[1]):
                        x.append(self.xcentergrid[k, i])
                        z.append(0.5 * (zcentergrid[k, i] + zcentergrid[k + 1, i]))
                x = np.array(x).reshape((1, self.xcentergrid.shape[1]))
                z = np.array(z).reshape((1, self.xcentergrid.shape[1]))
            else:
                x = self.xcentergrid
                z = zcentergrid

            u = []
            v = []
            ibx = []
            xedge, yedge = self.mg.xyedges
            for k in range(self.mg.nlay):
                u.append(plotutil.cell_value_points(self.xpts, xedge,
                                                    yedge, qx[k, :, :]))
                v.append(plotutil.cell_value_points(self.xpts, xedge,
                                                    yedge, qz[k, :, :]))
                ibx.append(plotutil.cell_value_points(self.xpts, xedge,
                                                      yedge, ib[k, :, :]))
            u = np.array(u)
            v = np.array(v)
            ibx = np.array(ibx)
            x = x[::kstep, ::hstep]
            z = z[::kstep, ::hstep]
            u = u[::kstep, ::hstep]
            v = v[::kstep, ::hstep]
            ib = ibx[::kstep, ::hstep]

            # upts and vpts has a value for the left and right
            # sides of a cell. Sample every other value for quiver
            u = u[:, ::2]
            v = v[:, ::2]
            ib = ib[:, ::2]

        else:
            # kstep implementation for vertex grid
            projpts = {key: value for key, value in self.__cls.projpts.items()
                       if (key // ncpl) % kstep == 0}

            # set x and z centers
            if isinstance(head, np.ndarray):
                # pipe kstep to set_zcentergrid to assure consistent array size
                zcenters = self.__cls.set_zcentergrid(np.ravel(head), kstep=kstep)
            else:
                zcenters = [np.mean(np.array(v).T[1]) for i, v
                            in sorted(projpts.items())]

            u = np.array([qx[cell] for cell in sorted(projpts)])

            if self.direction == "x":
                x = np.array([np.mean(np.array(v).T[0]) for i, v
                                     in sorted(projpts.items())])
            else:
                x = np.array([np.mean(np.array(v).T[1]) for i, v
                                     in sorted(projpts.items())])

            z = np.ravel(zcenters)
            v = np.array([qz[cell] for cell
                          in sorted(projpts)])
            ib = np.array([ib[cell] for cell
                           in sorted(projpts)])

            x = x[::hstep]
            z = z[::hstep]
            u = u[::hstep]
            v = v[::hstep]
            ib = ib[::hstep]

        if normalize:
            vmag = np.sqrt(u ** 2. + v ** 2.)
            idx = vmag > 0.
            u[idx] /= vmag[idx]
            v[idx] /= vmag[idx]

        # mask with an ibound array
        u[~ib] = np.nan
        v[~ib] = np.nan

        quiver = ax.quiver(x, z, u, v, pivot=pivot, **kwargs)

        return quiver

    def plot_discharge(self, frf, fff, flf=None,
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
        if self.mg.grid_type != "structured":
            err_msg = "Use plot_specific_discharge for " \
                      "{} grids".format(self.mg.grid_type)
            raise NotImplementedError(err_msg)

        else:
            ib = np.ones((self.mg.nlay, self.mg.nrow, self.mg.ncol))
            if self.mg.idomain is not None:
                ib = self.mg.idomain

            delr = self.mg.delr
            delc = self.mg.delc
            top = self.mg.top
            botm = self.mg.botm
            nlay, nrow, ncol = botm.shape
            laytyp = None
            hnoflo = 999.
            hdry = 999.

            if self.model is not None:
                if self.model.laytyp is not None:
                    laytyp = self.model.laytyp

                if self.model.hnoflo is not None:
                    hnoflo = self.model.hnoflo

                if self.model.hdry is not None:
                    hdry = self.model.hdry

            # If no access to head or laytyp, then calculate confined saturated
            # thickness by setting laytyp to zeros
            if head is None or laytyp is None:
                head = np.zeros(botm.shape, np.float32)
                laytyp = np.zeros((nlay), dtype=np.int)
                head[0, :, :] = top
                if nlay > 1:
                    head[1:, :, :] = botm[:-1, :, :]

            sat_thk = plotutil.PlotUtilities. \
                saturated_thickness(head, top, botm,
                                    laytyp, [hnoflo, hdry])

            # Calculate specific discharge
            qx, qy, qz = plotutil.PlotUtilities. \
                centered_specific_discharge(frf, fff, flf,
                                            delr, delc, sat_thk)

            if qz is None:
                qz = np.zeros((qx.shape), dtype=np.float)

            ib = ib.ravel()
            qx = qx.ravel()
            qy = qy.ravel()
            qz = qz.ravel()

            temp = []
            for ix, val in enumerate(ib):
                if val != 0:
                    temp.append((ix + 1, qx[ix], -qy[ix], qz[ix]))

            spdis = np.recarray((len(temp),), dtype=[('node', np.int),
                                                     ("qx", np.float),
                                                     ("qy", np.float),
                                                     ("qz", np.float)])
            for ix, tup in enumerate(temp):
                spdis[ix] = tup

            self.plot_specific_discharge(spdis, head=head, kstep=kstep,
                                         hstep=hstep, normalize=normalize,
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


class DeprecatedCrossSection(PlotCrossSection):
    """
    Deprecation handler for PlotCrossSection
    """
    def __init__(self, ax=None, model=None, modelgrid=None,
                 line=None, extent=None):
        super(DeprecatedCrossSection, self).__init__(ax=ax, model=model,
                                                     modelgrid=modelgrid,
                                                     line=line,
                                                     extent=extent)

    def plot_discharge(self, frf, fff, flf=None,
                       head=None, kstep=1, hstep=1, normalize=False,
                       **kwargs):
        super(DeprecatedCrossSection, self).plot_discharge(frf=frf,
                                                           fff=fff,
                                                           flf=flf,
                                                           head=head,
                                                           kstep=kstep,
                                                           hstep=hstep,
                                                           normalize=normalize,
                                                           **kwargs)