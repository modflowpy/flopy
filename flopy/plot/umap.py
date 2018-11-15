import copy, warnings
import sys
import numpy as np
from ..utils import geometry

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors
except ImportError:
    plt = None

from . import plotutil
import warnings
warnings.simplefilter('always', PendingDeprecationWarning)


class UnstructuredMapView(object):
    """
    UnstructutredMapView is the class that holds unique code for
    plotting unstructured discretization modflow models. this class
    is a work in progress, but currently supports plotting arrays,
    contouring arrays, plotting ibound arrays, plotting modpath results,
    and plotting inactive arrays. More support will be added in the
    future

    Parameters:
    ----------
        :param object modelgrid:
            flopy.discretization.unstructuredgrid.UnstructuredGrid
            object
        :param object model: flopy.modflow.Modflow object
        :param object ax: matplotlib.axes object
        :param list/np.ndarray extent:  a list of xmin, xmax, ymin, ymax
            as boundaries for plotting
    """
    def __init__(self, modelgrid=None, model=None, ax=None,
                 extent=None, layer=None):

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

    def plot_ibound(self, idomain=None, color_noflow='black', color_ch='blue',
                    color_vpt="red", **kwargs):
        """
        Make a plot of ibound.  If not specified, then pull ibound from the
        self.ml

        Parameters
        ----------
        idomain : numpy.ndarray
            ibound array to plot.  (Default is ibound in 'BAS6' package.)
        color_noflow : string
            (Default is 'black')
        color_ch : string
            Color for constant heads (Default is 'blue'.)
        color_vpt: string
            Color for vertical pass through cells mf6 (Default is "red")
        Returns
        -------
        quadmesh : matplotlib.collections.QuadMesh
        """
        raise NotImplementedError("Function must be called from PlotMapView")

    def plot_grid_lines(self, **kwargs):
        raise NotImplementedError("Function must be called in PlotMapView")

    def plot_inactive(self):
        raise NotImplementedError("Function must be called in PlotMapView")

    def plot_bc(self):
        raise NotImplementedError()

    def plot_discharge(self):
        raise NotImplementedError()

    def plot_pathline(self):
        raise NotImplementedError("Function must be called in PlotMapView")

    def plot_timeseries(self):
        raise NotImplementedError("Function must be called in PlotMapView")

    def plot_endpoint(self):
        raise NotImplementedError("Function must be called in PlotMapView")
