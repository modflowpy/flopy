import numpy as np
from flopy.discretization import StructuredGrid
from flopy.discretization import UnstructuredGrid

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors
except ImportError:
    plt = None

from . import plotutil
import warnings
warnings.simplefilter('always', PendingDeprecationWarning)


class _MapView(object):
    """
    This class is a base for all three mapview types. No information
    specific to a single type of model grid ex.(Structured, Vertex, Unstructured)
    is contained in this class!

    This class should not be instantiated by the User.

    Parameters
    ----------
    modelgrid : fp.discretization.Grid object
        StructuredGrid, UnstructuredGrid, or VertexGrid
    ax : matplotlib.pyplot.axes object, optional
    model : fp.modflow.Modflow object
    layer : int
        model layer to plot
    extent : tuple of floats
        the plotting extent as (xmin, xmax, ymin, ymax)

    """
    def __init__(self, modelgrid=None, ax=None, model=None, layer=0,
                 extent=None):
        if plt is None:
            s = 'Could not import matplotlib.  Must install matplotlib ' + \
                ' in order to use ModelMap method'
            raise Exception(s)

        self.model = model
        self.layer = layer
        self.mg = None

        if model is not None:
            self.mg = model.modelgrid

        elif modelgrid is not None:
            self.mg = modelgrid

        else:
            err_msg = "A model grid instance must be provided to PlotMapView"
            raise AssertionError(err_msg)

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


class _StructuredMapView(_MapView):
    """
    Class to create a map of the model.

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
    StructuredMapView should not be instantiated directly. PlotMapView uses
    this class for StructuredGrid specific plotting routines.

    """

    def __init__(self, model=None, modelgrid=None, ax=None, layer=0,
                 extent=None):
        super(_StructuredMapView, self).__init__(ax=ax, model=model,
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
            keyword arguments passed to matplotlib.pyplot.pcolormesh

        Returns
        -------
        quadmesh : matplotlib.collections.QuadMesh

        """
        if not isinstance(a, np.ndarray):
            a = np.array(a)

        if a.ndim == 3:
            plotarray = a[self.layer, :, :]
        elif a.ndim == 2:
            plotarray = a
        elif a.ndim == 1:
            plotarray = a
        else:
            raise Exception('Array must be of dimension 1, 2, or 3')

        if masked_values is not None:
            for mval in masked_values:
                plotarray = np.ma.masked_equal(plotarray, mval)

        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax

        xgrid = self.mg.xvertices
        ygrid = self.mg.yvertices

        quadmesh = ax.pcolormesh(xgrid, ygrid, plotarray)

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
        try:
            import matplotlib.tri as tri
        except ImportError:
            tri = None

        xcentergrid = self.mg.xcellcenters
        ycentergrid = self.mg.ycellcenters

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
                kwargs.pop('cmap')

        plot_triplot = False
        if 'plot_triplot' in kwargs:
            plot_triplot = kwargs.pop('plot_triplot')

        if 'extent' in kwargs and tri is not None:
            extent = kwargs.pop('extent')

            idx = (xcentergrid >= extent[0]) & (
                   xcentergrid <= extent[1]) & (
                          ycentergrid >= extent[2]) & (
                          ycentergrid <= extent[3])
            a = a[idx].flatten()
            xc = xcentergrid[idx].flatten()
            yc = ycentergrid[idx].flatten()
            triang = tri.Triangulation(xc, yc)
            try:
                amask = a.mask
                mask = [False for i in range(triang.triangles.shape[0])]
                for ipos, (n0, n1, n2) in enumerate(triang.triangles):
                    if amask[n0] or amask[n1] or amask[n2]:
                        mask[ipos] = True
                triang.set_mask(mask)
            except:
                mask = None
            contour_set = ax.tricontour(triang, plotarray, **kwargs)
            if plot_triplot:
                ax.triplot(triang, color='black', marker='o', lw=0.75)
        else:

            contour_set = ax.contour(xcentergrid, ycentergrid,
                                     plotarray, **kwargs)

        ax.set_xlim(self.extent[0], self.extent[1])
        ax.set_ylim(self.extent[2], self.extent[3])

        return contour_set

    def plot_inactive(self):
        raise NotImplementedError("Function must be called in PlotMapView")

    def plot_ibound(self, ibound=None, color_noflow='black', color_ch='blue',
                    color_vpt="red", **kwargs):
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

        # use a general expression to get stress period data
        arr_dict = p.stress_period_data.to_array(kper)
        if not arr_dict:
            return None

        for key in arr_dict:
            fluxes = arr_dict[key]
            break

        nlay = self.mg.nlay

        # Plot the list locations
        plotarray = np.zeros((nlay, self.mg.nrow, self.mg.ncol), dtype=np.int)
        if plotAll:
            t = np.sum(fluxes, axis=0)
            pa = np.zeros((self.mg.nrow, self.mg.ncol), dtype=np.int)
            pa[t != 0] = 1
            for k in range(nlay):
                plotarray[k, :, :] = pa.copy()
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

        # create normalized quadmesh
        quadmesh = self.plot_array(plotarray, cmap=cmap, norm=norm, **kwargs)

        return quadmesh

    def plot_shapefile(self):
        raise NotImplementedError("Function must be called in PlotMapView")

    def plot_cvfd(self):
        raise NotImplementedError("Function must be called in PlotMapView")

    def contour_array_cvfd(self):
        raise NotImplementedError("Function must be called in PlotMapView")

    def plot_specific_discharge(self):
        raise NotImplementedError("Function must be called in PlotMapView")

    def plot_discharge(self):
        raise NotImplementedError("Function must be called in PlotMapView")

    def plot_pathline(self):
        raise NotImplementedError("Function must be called in PlotMapView")

    def plot_timeseries(self):
        return NotImplementedError("Function must be called from PlotMapView")

    def plot_endpoint(self):
        raise NotImplementedError("Function must be called from PlotMapView")


class ModelMap(object):
    """
    Pending Depreciation: ModelMap acts as a PlotMapView factory
    object. Please migrate to PlotMapView for plotting
    functionality and future code compatibility

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
    length_multiplier : float
        scaling factor for conversion from model units to another unit
        length base ex. ft to m.

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
                extent=None, xul=None, yul=None, xll=None, yll=None,
                rotation=None, length_multiplier=None):

        from ..utils.reference import SpatialReferenceUnstructured
        from ..plot.plotbase import DeprecatedMapView

        err_msg = "ModelMap will be replaced by " \
                  "PlotMapView(); Calling PlotMapView()"
        warnings.warn(err_msg, PendingDeprecationWarning)

        modelgrid = None
        if model is not None:
            if (xul, yul, xll, yll, rotation) != (None, None, None, None, None):
                modelgrid = plotutil._set_coord_info(model.modelgrid,
                                                     xul, yul, xll, yll,
                                                     rotation)
        elif sr is not None:
            if length_multiplier is not None:
                sr.length_multiplier = length_multiplier

            if (xul, yul, xll, yll, rotation) != (None, None, None, None, None):
                sr.set_spatialreference(xul, yul, xll, yll, rotation)

            if isinstance(sr, SpatialReferenceUnstructured):
                if dis is not None:
                    modelgrid = UnstructuredGrid(vertices=sr.verts,
                                                 iverts=sr.iverts,
                                                 xcenters=sr.xc,
                                                 ycenters=sr.yc,
                                                 top=dis.top.array,
                                                 botm=dis.botm.array,
                                                 ncpl=sr.ncpl)
                else:
                    modelgrid = UnstructuredGrid(vertices=sr.verts,
                                                 iverts=sr.iverts,
                                                 xcenters=sr.xc,
                                                 ycenters=sr.yc,
                                                 ncpl=sr.ncpl)

            elif dis is not None:
                modelgrid = StructuredGrid(delc=sr.delc, delr=sr.delr,
                                           top=dis.top.array, botm=dis.botm.array,
                                           xoff=sr.xll, yoff=sr.yll,
                                           angrot=sr.rotation)
            else:
                modelgrid = StructuredGrid(delc=sr.delc, delr=sr.delr,
                                           xoff=sr.xll, yoff=sr.yll,
                                           angrot=sr.rotation)

        else:
            pass

        return DeprecatedMapView(model=model, modelgrid=modelgrid, ax=ax,
                                 layer=layer, extent=extent)
