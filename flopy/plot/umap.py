import copy, warnings
import sys
import numpy as np
from ..utils import geometry
from .map import _MapView

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors
except ImportError:
    plt = None

from . import plotutil
import warnings
warnings.simplefilter('always', PendingDeprecationWarning)


class _UnstructuredMapView(_MapView):
    """
    _UnstructutredMapView is a class that holds unique code for
    plotting unstructured discretization modflow models. this class
    is a work in progress, but currently supports plotting arrays,
    contouring arrays, plotting ibound arrays, plotting modpath results,
    and plotting inactive arrays. More support will be added in the
    future

    Parameters
    ----------
    model : flopy.modflow object
        flopy model object. (Default is None)
    modelgrid : flopy.modflow.StructuredGrid
        flopy StructuredGrid object
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
    _UnstructuredMapView should not be instantiated directly. PlotMapView uses
    this class for UnstructuredGrid specific plotting routines.

    """
    def __init__(self, model=None, modelgrid=None, ax=None,
                 layer=None, extent=None):
        super(_UnstructuredMapView, self).__init__(modelgrid=modelgrid,
                                                   ax=ax,
                                                   model=model,
                                                   layer=layer,
                                                   extent=extent)
        """
        self.mg = None
        self.layer = layer
        self.model = model

        if model is not None:
            self.mg = model.modelgrid

        elif modelgrid is not None:
            self.mg = modelgrid

        else:
            raise AssertionError("No model grid was found")

        self.ax = ax
        if ax is None:
            try:
                self.ax = plt.gca()
                self.ax.set_aspect('equal')
            except:
                self.ax = plt.subplot(1, 1, 1, aspect='equal', axisbg='white')

        self._extent = extent
        """

    @property
    def extent(self):
        if self._extent is None:
            self._extent = self.mg.extent
        return self._extent


    def plot_array(self, a, masked_values=None, **kwargs):
        """
        Plot an array.

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
        pc : matplotlib.collections.PatchCollection

        """
        if not isinstance(a, np.ndarray):
            a = np.array(a)

        if masked_values is not None:
            for mval in masked_values:
                a = np.ma.masked_equal(a, mval)

        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax

        pc = plotutil.plot_cvfd(self.mg._vertices, self.mg._iverts, a=a,
                                ax=ax)

        if 'vmin' in kwargs:
            vmin = kwargs.pop('vmin')
        else:
            vmin = None

        if 'vmax' in kwargs:
            vmax = kwargs.pop('vmax')
        else:
            vmax = None

        pc.set_clim(vmin=vmin, vmax=vmax)

        pc.set(**kwargs)

        # add collection to axis
        ax.add_collection(pc)

        # set limits
        ax.set_xlim(self.extent[0], self.extent[1])
        ax.set_ylim(self.extent[2], self.extent[3])
        return pc

    def contour_array(self, a, masked_values=None, **kwargs):
        """
        Contour an array using matplotlib.tri.tricontour methods

        Parameters
        ----------
        a : np.ndarray or list
            array of values to contour
        masked_values : list
            values to mask out of the contours, ex. noflow values
        kwargs : ax and matplotlib.tri.tricontour keyword arguments

        Returns
        -------
        contour_set : matplotlib.tri.tricontour object

        """

        if not isinstance(a, np.ndarray):
            a = np.array(a)

        if masked_values is not None:
            for mval in masked_values:
                a = np.ma.masked_equal(a, mval)

        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax

        if 'colors' in kwargs.keys():
            if 'cmap' in kwargs.keys():
                cmap = kwargs.pop('cmap')
            cmap = None

        contour_set = ax.tricontour(self.mg.xcellcenters, self.mg.ycellcenters,
                                    a, **kwargs)

        ax.set_xlim(self.extent[0], self.extent[1])
        ax.set_ylim(self.extent[2], self.extent[3])

        return contour_set

    def plot_ibound(self):
        raise NotImplementedError("Function must be called from PlotMapView")

    def plot_grid_lines(self):
        raise NotImplementedError("Function must be called in PlotMapView")

    def plot_inactive(self):
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
            unused parameter in unstructured grid
        **kwargs : dictionary
            keyword arguments passed to matplotlib.collections.PatchCollection

        Returns
        -------
        quadmesh : matplotlib.collections.QuadMesh

        """
        if plotAll:
            print("plotAll is not used for unstructued grid plotting")

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
            fluxes = np.array(arr_dict[key])
            break

        plotarray = np.zeros(fluxes.shape, dtype=np.int)

        # Plot the list locations
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

    def plot_discharge(self):
        raise NotImplementedError("Function must be called in PlotMapView")

    def plot_pathline(self):
        raise NotImplementedError("Function must be called in PlotMapView")

    def plot_timeseries(self):
        raise NotImplementedError("Function must be called in PlotMapView")

    def plot_endpoint(self):
        raise NotImplementedError("Function must be called in PlotMapView")
