import copy
import numpy as np
from ..utils import geometry

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon
except ImportError:
    plt = None

from . import plotutil
from .map import MapView

import warnings
warnings.simplefilter('always', PendingDeprecationWarning)


class VertexMapView(MapView):
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
    def __init__(self, sr=None, ax=None, model=None, dis=None, modelgrid=None,
                 layer=0, extent=None, xul=None, yul=None, xll=None, yll=None,
                 rotation=0., length_multiplier=1.):
        super(VertexMapView, self).__init__(sr, ax, model, dis, modelgrid,
                                            layer, extent, xul, yul, xll,
                                            yll, rotation, length_multiplier)

    @property
    def extent(self):
        if self._extent is None:
            self._extent = self.mg.extent
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
            keyword arguments passed to matplotlib.pyplot.patchcollection

        Returns
        -------
        quadmesh : matplotlib.collections.QuadMesh

        """
        if not isinstance(a, np.ndarray):
            a = np.array(a)

        if a.ndim == 2:
            plotarray = a[self.layer, :]
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

        xgrid = np.array(self.mg.xvertices)
        ygrid = np.array(self.mg.yvertices)

        patches = [Polygon(list(zip(xgrid[i], ygrid[i])), closed=True)
                   for i in range(xgrid.shape[0])]

        p = PatchCollection(patches)
        p.set_array(plotarray)

        if 'vmin' in kwargs:
            vmin = kwargs.pop('vmin')
        else:
            vmin = None

        if 'vmax' in kwargs:
            vmax = kwargs.pop('vmax')
        else:
            vmax = None

        p.set_clim(vmin=vmin, vmax=vmax)
        # send rest of kwargs to quadmesh
        p.set(**kwargs)

        ax.add_collection(p)
        ax.set_xlim(self.extent[0], self.extent[1])
        ax.set_ylim(self.extent[2], self.extent[3])
        return p

    def contour_array(self, a, masked_values=None, **kwargs):
        """
        Contour an array.  If the array is two-dimensional, then the method
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
        import matplotlib.tri as tri

        xcentergrid = np.array(self.mg.xcellcenters)
        ycentergrid = np.array(self.mg.ycellcenters)

        if not isinstance(a, np.ndarray):
            a = np.array(a)

        if a.ndim == 2:
            plotarray = a[self.layer, :]
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
                kwargs.pop('cmap')

        plot_triplot = False
        if 'plot_triplot' in kwargs:
            plot_triplot = kwargs.pop('plot_triplot')

        if 'extent' in kwargs:
            extent = kwargs.pop('extent')

            idx = (xcentergrid >= extent[0]) & (
                    xcentergrid <= extent[1]) & (
                          ycentergrid >= extent[2]) & (
                          ycentergrid <= extent[3])
            plotarray = plotarray[idx].flatten()
            xcentergrid = xcentergrid[idx].flatten()
            ycentergrid = ycentergrid[idx].flatten()

        triang = tri.Triangulation(xcentergrid, ycentergrid)

        mask = None
        try:
            amask = plotarray.mask
            mask = [False for i in range(triang.triangles.shape[0])]
            for ipos, (n0, n1, n2) in enumerate(triang.triangles):
                if amask[n0] or amask[n1] or amask[n2]:
                    mask[ipos] = True
            triang.set_mask(mask)
        except:
            pass

        contour_set = ax.tricontour(triang, plotarray, **kwargs)

        if plot_triplot:
            ax.triplot(triang, color="black", marker="o", lw=0.75)

        ax.set_xlim(self.extent[0], self.extent[1])
        ax.set_ylim(self.extent[2], self.extent[3])

        return contour_set

    def plot_inactive(self, idomain=None, color_noflow='black', **kwargs):
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
        quadmesh : matplotlib.pyplot.axes object

        """
        raise NotImplementedError("plot_inactive must be called "
                                  "from a PlotMapView instance")

    def plot_ibound(self, idomain=None, color_noflow='black', color_ch="blue",
                    color_vpt="red", **kwargs):
        """
        Make a plot of ibound.  If not specified, then pull ibound from the
        self.ml

        Parameters
        ----------
        ibound : numpy.ndarray
            ibound array to plot.  (Default is ibound in 'BAS6' package.)
        color_noflow : string
            (Default is 'black')
        color_vpt : string
            Color for vertical pass through cells (Default is 'red'.)

        Returns
        -------
        ax : matplotlib.pyplot.axes object

        """
        raise NotImplementedError("plot_ibound must be called"
                                 " from PlotMapView instance")

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
        err_msg = "plot_grid() must be called " \
                  "from a PlotMapView instance"
        raise NotImplementedError(err_msg)

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
        #find package to plot
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

        arr_dict = p.stress_period_data.to_array(kper)
        if not arr_dict:
            return None

        for key in arr_dict:
            fluxes = arr_dict[key]
            break

        nlay = self.mg.nlay

        # Plot the list locations
        plotarray = np.zeros((nlay, self.mg.ncpl), dtype=np.int)

        if plotAll:
            t = np.sum(fluxes, axis=0)
            pa = np.zeros((self.mg.ncpl,), dtype=np.int)
            pa[t != 0] = 1
            for k in range(nlay):
                plotarray[k, :] = pa.copy()
        else:
            plotarray[fluxes != 0] = 1

        # mask the plot array
        plotarray = np.ma.masked_equal(plotarray, 0)

        # set the colormap
        if color is None:
            if ftype in plotutil.bc_color_dict:
                c = plotutil.bc_color_dict[ftype]
            else:
                c = plotutil.bc_color_dict['default']
        else:
            c = color

        cmap = matplotlib.colors.ListedColormap(['0', c])
        bounds = [0, 1, 2]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

        ax = self.plot_array(plotarray, cmap=cmap, norm=norm, **kwargs)

        return ax

    def plot_shapefile(self, shp, **kwargs):
        return NotImplementedError()

    def plot_cvfd(self, verts, iverts, **kwargs):
        return NotImplementedError()

    def contour_array_cvfd(self, vertc, a, masked_values=None, **kwargs):
        return NotImplementedError()

    def plot_discharge(self, fja, dis=None, head=None, istep=1,
                       normalize=False, **kwargs):
        """
        Use quiver to plot vectors.

        Parameters
        ----------
        fja : numpy.ndarray
            MODFLOW's 'flow ja face'
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
        if 'pivot' in kwargs:
            pivot = kwargs.pop('pivot')
        else:
            pivot = 'middle'

        # todo: eventually remove dis reference!
        top = self.mg.top
        botm = self.mg.botm
        idomain = self.mg.idomain
        if self.mg.top is None or self.mg.botm is None:
            if dis is None:
                if self.model is not None:
                    dis = self.model.dis
                    idomain = self.model.dis.idomain.array
                    top = self.model.dis.top.array
                    botm = self.model.dis.botm.array

                else:
                    err_msg = "ModelMap.plot_quiver() error: DIS package not found"
                    raise AssertionError(err_msg)
            else:
                top = dis.top.array
                botm = dis.botm.array
                idomain = dis.idomain.array

        fja = np.array(fja)
        nlay = self.mg.nlay

        delr = np.tile([np.max(i) - np.min(i) for i in self.mg.yvertices], (nlay, 1))
        delc = np.tile([np.max(i) - np.min(i) for i in self.mg.xvertices], (nlay, 1))

        # no modflow6 equivalent?????
        hnoflo = 999.
        hdry = 999.

        if head is None:
            head = np.zeros(botm.shape)

        if len(head.shape) == 3:
            head.shape = (nlay, -1)

        # if isinstance(fja, list):
        #    fja = fja[0]

        if len(fja.shape) == 4:
            fja = fja[0][0][0]

        laytyp = np.zeros((nlay,))
        if self.model is not None:
            if self.model.sto is not None:
                laytyp = self.model.sto.iconvert.array

        sat_thk = plotutil.PlotUtilities.\
            saturated_thickness(head, top,
                                botm, laytyp,
                                mask_values=[hnoflo, hdry])

        frf, fff, flf = plotutil.UnstructuredPlotUtilities.\
            vectorize_flow(fja, model_grid=self.mg,
                           idomain=idomain)

        qx, qy, qz = plotutil.UnstructuredPlotUtilities.\
            specific_discharge(frf, fff, flf,
                               delr, delc, sat_thk)

        # Select the correct layer slice
        u = qx[self.layer, :]
        v = qy[self.layer, :]

        # apply step
        x = self.mg.xcellcenters[::istep]
        y = self.mg.ycellcenters[::istep]
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
        idx = (idomain[self.layer, ::istep] == 0)
        idx[idomain[self.layer, ::istep] == -1] = 1

        u[idx] = np.nan
        v[idx] = np.nan

        # Rotate and plot, offsets must be zero since
        # these are vectors not locations
        urot, vrot = geometry.rotate(u, v, 0., 0.,
                                     self.mg.angrot_radians)
        quiver = ax.quiver(x, y, urot, vrot, scale=1, units='xy', pivot=pivot, **kwargs)
        return quiver

    def plot_pathline(self, pl, travel_time=None, **kwargs):
        return NotImplementedError("MODPATH 7 support is not yet implemented")

    def plot_timeseries(self, ts, travel_time=None, **kwargs):
        return NotImplementedError("MODPATH 7 support is not yet implemented")

    def plot_endpoint(self, ep, direction="ending", selection=None,
                      selection_direction=None, **kwargs):
        return NotImplementedError("MODPATH 7 support is not yet implemented")
