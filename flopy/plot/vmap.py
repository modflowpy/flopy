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
from .map import _MapView

import warnings
warnings.simplefilter('always', PendingDeprecationWarning)


class _VertexMapView(_MapView):
    """
    Class to create a VertexGrid based map of the model.

    Parameters
    ----------
    model : flopy.modflow object
        flopy model object. (Default is None)
    modelgrid : flopy.discretization.VertexGrid
        Vertex model grid object
    ax : matplotlib.pyplot axis
        The plot axis.  If not provided it, plt.gca() will be used.
        If there is not a current axis then a new one will be created.
    layer : int
        Layer to plot.  Default is 0.  Must be between 0 and nlay - 1.
    extent : tuple of floats
        (xmin, xmax, ymin, ymax) will be used to specify axes limits.  If None
        then these will be calculated based on grid, coordinates, and rotation.

     Notes
    -----
        _VertexMapView should not be instantiated directly. PlotMapView uses
        this class for VertexGrid specific plotting routines.

    """
    def __init__(self, modelgrid=None, model=None, ax=None, layer=0,
                 extent=None):
        super(_VertexMapView, self).__init__(ax=ax, model=model,
                                             modelgrid=modelgrid, layer=layer,
                                             extent=extent)

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
        contour_set : matplotlib.tri.tricontour object

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

    def plot_inactive(self):
        raise NotImplementedError("Function must be called in PlotMapView")

    def plot_ibound(self):
        raise NotImplementedError("Function must be called in PlotMapView")

    def plot_grid(self):
        raise NotImplementedError("Function must be called in PlotMapView")

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

    def plot_shapefile(self):
        return NotImplementedError("Function must be called in PlotMapView")

    def plot_cvfd(self):
        return NotImplementedError("Function must be called in PlotMapView")

    def contour_array_cvfd(self,):
        return NotImplementedError("Function must be called in PlotMapView")

    def plot_specific_discharge(self):
        return NotImplementedError("Function must be called in PlotMapView")

    def plot_discharge(self):
        raise NotImplementedError("Function must be called in PlotMapView")

    def plot_pathline(self):
        return NotImplementedError("Function must be called in PlotMapView")

    def plot_timeseries(self):
        return NotImplementedError("Function must be called in PlotMapView")

    def plot_endpoint(self):
        return NotImplementedError("Function must be called in PlotMapView")
