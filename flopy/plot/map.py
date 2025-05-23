import warnings

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.path import Path
from numpy.lib.recfunctions import stack_arrays

from ..utils import geometry
from . import plotutil
from .plotutil import to_mp7_endpoints, to_mp7_pathlines

warnings.simplefilter("always", PendingDeprecationWarning)


class PlotMapView:
    """
    Class to create a map of the model. Delegates plotting
    functionality based on model grid type.

    Parameters
    ----------
    modelgrid : flopy.discretization.Grid
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

    def __init__(self, model=None, modelgrid=None, ax=None, layer=0, extent=None):
        self.model = model
        self.layer = layer
        self.mg = None

        if modelgrid is not None:
            self.mg = modelgrid
        elif model is not None:
            self.mg = model.modelgrid
        else:
            err_msg = "A model grid instance must be provided to PlotMapView"
            raise AssertionError(err_msg)

        if ax is None:
            try:
                self.ax = plt.gca()
                self.ax.set_aspect("equal")
            except (AttributeError, ValueError):
                self.ax = plt.subplot(1, 1, 1, aspect="equal", axisbg="white")
        else:
            self.ax = ax

        if extent is not None:
            self._extent = extent
        else:
            self._extent = None

        if model is None:
            self._masked_values = [1e30, -1e30]
        else:
            self._masked_values = [model.hnoflo, model.hdry]

    @property
    def extent(self):
        if self._extent is None:
            self._extent = self.mg.extent
        return self._extent

    def _set_axes_limits(self, ax):
        """
        Internal method to set axes limits

        Parameters
        ----------
        ax : matplotlib.pyplot axis
            The plot axis

        Returns
        -------
        ax : matplotlib.pyplot axis object

        """
        if ax.get_autoscale_on():
            ax.set_xlim(self.extent[0], self.extent[1])
            ax.set_ylim(self.extent[2], self.extent[3])
        return ax

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
        quadmesh : matplotlib.collections.QuadMesh or
            matplotlib.collections.PatchCollection

        """

        if not isinstance(a, np.ndarray):
            a = np.array(a)

        a = a.astype(float)
        # Use the model grid to pass back an array of the correct shape
        plotarray = self.mg.get_plottable_layer_array(a, self.layer)

        # if masked_values are provided mask the plotting array
        if masked_values is not None:
            self._masked_values.extend(list(masked_values))
        for mval in self._masked_values:
            plotarray = np.ma.masked_values(plotarray, mval)

        # add NaN values to mask
        plotarray = np.ma.masked_where(np.isnan(plotarray), plotarray)

        ax = kwargs.pop("ax", self.ax)

        # use cached patch collection for plotting
        polygons = self.mg.map_polygons
        if isinstance(polygons, dict):
            polygons = polygons[self.layer]

        if len(polygons) == 0:
            return

        if not isinstance(polygons[0], Path):
            collection = ax.pcolormesh(self.mg.xvertices, self.mg.yvertices, plotarray)

        else:
            plotarray = plotarray.ravel()
            collection = PathCollection(polygons)
            collection.set_array(plotarray)

        # set max and min
        vmin = kwargs.pop("vmin", None)
        vmax = kwargs.pop("vmax", None)

        if "cmap" not in kwargs:
            kwargs["cmap"] = "viridis"

        # set matplotlib kwargs
        collection.set_clim(vmin=vmin, vmax=vmax)
        collection.set(**kwargs)
        ax.add_collection(collection)

        # set limits
        ax = self._set_axes_limits(ax)
        return collection

    def contour_array(self, a, masked_values=None, tri_mask=False, **kwargs):
        """
        Contour an array on the grid. By default the top layer
        is contoured. To select a different layer, specify the
        layer in the class constructor.

        For structured and vertex grids, the array may be 1D, 2D or 3D.
        For unstructured grids, the array must be 1D or 2D.

        Parameters
        ----------
        a : 1D, 2D or 3D array-like
            Array to plot.
        masked_values : iterable of floats, ints
            Values to mask.
        tri_mask : bool
            Boolean flag that masks triangulation and contouring
            by nearest grid neighbors. This flag is useful for contouring
            on unstructured model domains that have holes in the grid.
        **kwargs : dictionary
            keyword arguments passed to matplotlib.pyplot.pcolormesh

        Returns
        -------
        contour_set : matplotlib.pyplot.contour

        """
        import matplotlib.tri as tri

        # coerce array to ndarray of floats
        a = np.copy(a)
        if not isinstance(a, np.ndarray):
            a = np.array(a)
        a = a.astype(float)

        # Use the model grid to pass back an array of the correct shape
        plotarray = self.mg.get_plottable_layer_array(a, self.layer)

        # Get vertices for the selected layer
        xcentergrid = self.mg.get_xcellcenters_for_layer(self.layer)
        ycentergrid = self.mg.get_ycellcenters_for_layer(self.layer)

        ax = kwargs.pop("ax", self.ax)
        filled = kwargs.pop("filled", False)
        plot_triplot = kwargs.pop("plot_triplot", False)

        if "colors" in kwargs.keys():
            if "cmap" in kwargs.keys():
                kwargs.pop("cmap")

        if "extent" in kwargs:
            extent = kwargs.pop("extent")

            idx = (
                (xcentergrid >= extent[0])
                & (xcentergrid <= extent[1])
                & (ycentergrid >= extent[2])
                & (ycentergrid <= extent[3])
            )
            plotarray = plotarray[idx]
            xcentergrid = xcentergrid[idx]
            ycentergrid = ycentergrid[idx]

        # use standard contours for structured grid, otherwise tricontours
        if self.mg.grid_type == "structured":
            ismasked = None
            if masked_values is not None:
                self._masked_values.extend(list(masked_values))

            for mval in self._masked_values:
                if ismasked is None:
                    ismasked = np.isclose(plotarray, mval)
                else:
                    t = np.isclose(plotarray, mval)
                    ismasked += t

            if ismasked is not None:
                plotarray[ismasked] = np.nan

            contour_set = (
                ax.contourf(xcentergrid, ycentergrid, plotarray, **kwargs)
                if filled
                else ax.contour(xcentergrid, ycentergrid, plotarray, **kwargs)
            )
        else:
            # work around for tri-contour ignore vmin & vmax
            # necessary block for tri-contour NaN issue
            if "levels" not in kwargs:
                vmin = kwargs.pop("vmin", np.nanmin(plotarray))
                vmax = kwargs.pop("vmax", np.nanmax(plotarray))
                levels = np.linspace(vmin, vmax, 7)
                kwargs["levels"] = levels

            # workaround for tri-contour nan issue
            # use -2**31 to allow for 32 bit int arrays
            plotarray[np.isnan(plotarray)] = -(2**31)
            if masked_values is None:
                masked_values = [-(2**31)]
            else:
                masked_values = list(masked_values)
                if -(2**31) not in masked_values:
                    masked_values.append(-(2**31))

            ismasked = None
            if masked_values is not None:
                self._masked_values.extend(list(masked_values))

            for mval in self._masked_values:
                if ismasked is None:
                    ismasked = np.isclose(plotarray, mval)
                else:
                    t = np.isclose(plotarray, mval)
                    ismasked += t

            plotarray = plotarray.flatten()
            xcentergrid = xcentergrid.flatten()
            ycentergrid = ycentergrid.flatten()
            triang = tri.Triangulation(xcentergrid, ycentergrid)
            analyze = tri.TriAnalyzer(triang)
            mask = analyze.get_flat_tri_mask(rescale=False)

            # mask out holes, optional???
            if tri_mask:
                triangles = triang.triangles
                for i in range(2):
                    for ix, nodes in enumerate(triangles):
                        neighbors = self.mg.neighbors(nodes[i], as_nodes=True)
                        isin = np.isin(nodes[i + 1 :], neighbors)
                        if not np.all(isin):
                            mask[ix] = True

            if ismasked is not None:
                ismasked = ismasked.flatten()
                mask2 = np.any(
                    np.where(ismasked[triang.triangles], True, False), axis=1
                )
                mask[mask2] = True

            triang.set_mask(mask)

            contour_set = (
                ax.tricontourf(triang, plotarray.flatten(), **kwargs)
                if filled
                else ax.tricontour(triang, plotarray.flatten(), **kwargs)
            )

            if plot_triplot:
                ax.triplot(triang, color="black", marker="o", lw=0.75)

        ax = self._set_axes_limits(ax)

        return contour_set

    def plot_inactive(self, ibound=None, color_noflow="black", **kwargs):
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

        plotarray = np.zeros(ibound.shape, dtype=int)
        idx1 = ibound == 0
        plotarray[idx1] = 1
        plotarray = np.ma.masked_equal(plotarray, 0)
        cmap = matplotlib.colors.ListedColormap(["0", color_noflow])
        bounds = [0, 1, 2]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        quadmesh = self.plot_array(plotarray, cmap=cmap, norm=norm, **kwargs)
        return quadmesh

    def plot_ibound(
        self,
        ibound=None,
        color_noflow="black",
        color_ch="blue",
        color_vpt="red",
        **kwargs,
    ):
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

        if ibound is None:
            if self.model is not None:
                if self.model.version == "mf6":
                    color_ch = color_vpt

            if self.mg.idomain is None:
                raise AssertionError("Ibound/Idomain array must be provided")

            ibound = self.mg.idomain

        plotarray = np.zeros(ibound.shape, dtype=int)
        idx1 = ibound == 0
        idx2 = ibound < 0
        plotarray[idx1] = 1
        plotarray[idx2] = 2
        plotarray = np.ma.masked_equal(plotarray, 0)
        cmap = matplotlib.colors.ListedColormap(["0", color_noflow, color_ch])
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
        ax = kwargs.pop("ax", self.ax)
        colors = kwargs.pop("colors", "grey")
        colors = kwargs.pop("color", colors)
        colors = kwargs.pop("ec", colors)
        colors = kwargs.pop("edgecolor", colors)

        grid_lines = self.mg.grid_lines
        if isinstance(grid_lines, dict):
            grid_lines = grid_lines[self.layer]

        collection = LineCollection(grid_lines, colors=colors, **kwargs)

        ax.add_collection(collection)
        ax = self._set_axes_limits(ax)
        return collection

    def plot_bc(
        self,
        name=None,
        package=None,
        kper=0,
        color=None,
        plotAll=False,
        boundname=None,
        **kwargs,
    ):
        """
        Plot boundary conditions locations for a specific boundary
        type from a flopy model

        Parameters
        ----------
        name : string
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
        if "ftype" in kwargs and name is None:
            name = kwargs.pop("ftype")

        # Find package to plot
        if package is not None:
            p = package
            name = p.name[0]
        elif self.model is not None:
            if name is None:
                raise Exception("ftype not specified")
            name = name.upper()
            p = self.model.get_package(name)
        else:
            raise Exception("Cannot find package to plot")

        # trap for mf6 'cellid' vs mf2005 'k', 'i', 'j' convention
        if isinstance(p, list) or p.parent.version == "mf6":
            if not isinstance(p, list):
                p = [p]

            idx = np.array([])
            for pp in p:
                if pp.package_type in ("lak", "sfr", "maw", "uzf"):
                    t = plotutil.advanced_package_bc_helper(pp, self.mg, kper)
                else:
                    try:
                        mflist = pp.stress_period_data.array[kper]
                    except Exception as e:
                        raise Exception(f"Not a list-style boundary package: {e!s}")
                    if mflist is None:
                        return
                    if boundname is not None:
                        mflist = mflist[mflist["boundname"] == boundname]
                    t = np.array([list(i) for i in mflist["cellid"]], dtype=int).T

                if len(idx) == 0:
                    idx = np.copy(t)
                else:
                    idx = np.append(idx, t, axis=1)

        else:
            # modflow-2005 structured and unstructured grid
            if p.package_type in ("uzf", "lak"):
                idx = plotutil.advanced_package_bc_helper(p, self.mg, kper)
            else:
                try:
                    mflist = p.stress_period_data[kper]
                except Exception as e:
                    raise Exception(f"Not a list-style boundary package: {e!s}")
                if mflist is None:
                    return
                if len(self.mg.shape) == 3:
                    idx = [mflist["k"], mflist["i"], mflist["j"]]
                else:
                    idx = mflist["node"]

        nlay = self.mg.nlay

        plotarray = np.zeros(self.mg.shape, dtype=int)
        if plotAll and len(self.mg.shape) > 1:
            pa = np.zeros(self.mg.shape[1:], dtype=int)
            pa[tuple(idx[1:])] = 1
            for k in range(nlay):
                plotarray[k] = pa.copy()
        elif len(self.mg.shape) > 1:
            plotarray[tuple(idx)] = 1
        else:
            plotarray[idx] = 1

        # mask the plot array
        plotarray = np.ma.masked_equal(plotarray, 0)

        # set the colormap
        if color is None:
            # modflow 6 ftype fix, since multiple packages append _0, _1, etc:
            key = name[:3].upper()
            if key in plotutil.bc_color_dict:
                c = plotutil.bc_color_dict[key]
            else:
                c = plotutil.bc_color_dict["default"]
        else:
            c = color

        cmap = matplotlib.colors.ListedColormap(["0", c])
        bounds = [0, 1, 2]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

        # create normalized quadmesh or patch object depending on grid type
        quadmesh = self.plot_array(plotarray, cmap=cmap, norm=norm, **kwargs)

        return quadmesh

    def plot_shapefile(self, shp, **kwargs):
        """
        Plot a shapefile.  The shapefile must be in the same coordinates as
        the rotated and offset grid.

        Parameters
        ----------
        shp : str, os.PathLike or pyshp shapefile object
            Path of the shapefile to plot

        kwargs : dictionary
            Keyword arguments passed to plotutil.plot_shapefile()

        """
        return self.plot_shapes(shp, **kwargs)

    def plot_shapes(self, obj, **kwargs):
        """
        Plot shapes is a method that facilitates plotting a collection
        of geospatial objects

        Parameters
        ----------
        obj : collection object
            obj can accept the following types

            str : shapefile path
            PathLike : shapefile path
            shapefile.Reader object
            list of [shapefile.Shape, shapefile.Shape,]
            shapefile.Shapes object
            flopy.utils.geometry.Collection object
            list of [flopy.utils.geometry, ...] objects
            geojson.GeometryCollection object
            geojson.FeatureCollection object
            shapely.GeometryCollection object
            list of [[vertices], ...]
        kwargs : dictionary
            keyword arguments passed to plotutil.plot_shapefile()

        Returns
        -------
            matplotlib.Collection object
        """
        ax = kwargs.pop("ax", self.ax)
        patch_collection = plotutil.plot_shapefile(obj, ax, **kwargs)
        ax = self._set_axes_limits(ax)
        return patch_collection

    def plot_centers(
        self, a=None, s=None, masked_values=None, inactive=False, **kwargs
    ):
        """
        Method to plot cell centers on cross-section using matplotlib
        scatter. This method accepts an optional data array(s) for
        coloring and scaling the cell centers. Cell centers in inactive
        nodes are not plotted by default

        Parameters
        ----------
        a : None, np.ndarray
            optional numpy nd.array of size modelgrid.nnodes
        s : None, float, numpy array
            optional point size parameter
        masked_values : None, iterable
            optional list, tuple, or np array of array (a) values to mask
        inactive : bool
            boolean flag to include inactive cell centers in the plot.
            Default is False
        **kwargs :
            matplotlib ax.scatter() keyword arguments

        Returns
        -------
            matplotlib ax.scatter() object
        """
        ax = kwargs.pop("ax", self.ax)

        xcenters = self.mg.get_xcellcenters_for_layer(self.layer).ravel()
        ycenters = self.mg.get_ycellcenters_for_layer(self.layer).ravel()
        idomain = self.mg.get_plottable_layer_array(self.mg.idomain, self.layer).ravel()

        active_ixs = list(range(len(xcenters)))
        if not inactive:
            active_ixs = np.where(idomain != 0)[0]

        xcenters = xcenters[active_ixs]
        ycenters = ycenters[active_ixs]

        if a is not None:
            a = self.mg.get_plottable_layer_array(a).ravel()

            if masked_values is not None:
                self._masked_values.extend(list(masked_values))

            for mval in self._masked_values:
                a[a == mval] = np.nan

            a = a[active_ixs]

        if s is not None:
            if not isinstance(s, (int, float)):
                s = self.mg.get_plottable_layer_array(s).ravel()
                s = s[active_ixs]

        scat = ax.scatter(xcenters, ycenters, c=a, s=s, **kwargs)
        return scat

    def plot_vector(
        self,
        vx,
        vy,
        istep=1,
        jstep=1,
        normalize=False,
        masked_values=None,
        **kwargs,
    ):
        """
        Plot a vector.

        Parameters
        ----------
        vx : np.ndarray
            x component of the vector to be plotted (non-rotated)
            array shape must be (nlay, nrow, ncol) for a structured grid
            array shape must be (nlay, ncpl) for a unstructured grid
        vy : np.ndarray
            y component of the vector to be plotted (non-rotated)
            array shape must be (nlay, nrow, ncol) for a structured grid
            array shape must be (nlay, ncpl) for a unstructured grid
        istep : int
            row frequency to plot (default is 1)
        jstep : int
            column frequency to plot (default is 1)
        normalize : bool
            boolean flag used to determine if vectors should be normalized
            using the vector magnitude in each cell (default is False)
        masked_values : iterable of floats
            values to mask
        kwargs : matplotlib.pyplot keyword arguments for the
            plt.quiver method

        Returns
        -------
        quiver : matplotlib.pyplot.quiver
            result of the quiver function

        """
        pivot = kwargs.pop("pivot", "middle")
        ax = kwargs.pop("ax", self.ax)

        # get ibound array to mask inactive cells
        ib = np.ones((self.mg.nnodes,), dtype=int)
        if self.mg.idomain is not None:
            ib = self.mg.idomain.ravel()

        xcentergrid = self.mg.get_xcellcenters_for_layer(self.layer)
        ycentergrid = self.mg.get_ycellcenters_for_layer(self.layer)
        vx = self.mg.get_plottable_layer_array(vx, self.layer)
        vy = self.mg.get_plottable_layer_array(vy, self.layer)
        ib = self.mg.get_plottable_layer_array(ib, self.layer)

        try:
            x = xcentergrid[::istep, ::jstep]
            y = ycentergrid[::istep, ::jstep]
            u = vx[::istep, ::jstep]
            v = vy[::istep, ::jstep]
            ib = ib[::istep, ::jstep]
        except IndexError:
            x = xcentergrid[::jstep]
            y = ycentergrid[::jstep]
            u = vx[::jstep]
            v = vy[::jstep]
            ib = ib[::jstep]

        # if necessary, copy to avoid changing the passed values
        if masked_values is not None or normalize:
            u = np.copy(u)
            v = np.copy(v)

        # mask values
        if masked_values is not None:
            for mval in masked_values:
                to_mask = np.logical_or(u == mval, v == mval)
                u[to_mask] = np.nan
                v[to_mask] = np.nan

        # normalize
        if normalize:
            vmag = np.sqrt(u**2.0 + v**2.0)
            idx = vmag > 0.0
            u[idx] /= vmag[idx]
            v[idx] /= vmag[idx]

        u[ib == 0] = np.nan
        v[ib == 0] = np.nan

        # rotate and plot, offsets must be zero since
        # these are vectors not locations
        urot, vrot = geometry.rotate(u, v, 0.0, 0.0, self.mg.angrot_radians)
        quiver = ax.quiver(x, y, urot, vrot, pivot=pivot, **kwargs)
        ax = self._set_axes_limits(ax)
        return quiver

    def plot_pathline(self, pl, travel_time=None, **kwargs):
        """
        Plot particle pathlines. Compatible with MODFLOW 6 PRT particle track
        data format, or MODPATH 6 or 7 pathline data format.

        Parameters
        ----------
        pl : list of recarrays or dataframes, or a single recarray or dataframe
            Particle pathline data. If a list of recarrays or dataframes,
            each must contain the path of only a single particle. If just
            one recarray or dataframe, it should contain the paths of all
            particles. The flopy.utils.modpathfile.PathlineFile.get_data()
            or get_alldata() return value may be passed directly as this
            argument.

            For MODPATH 6 or 7 pathlines, columns must include 'x', 'y', 'z',
            'time', 'k', and 'particleid'. Additional columns are ignored.

            For MODFLOW 6 PRT pathlines, columns must include 'x', 'y', 'z',
            't', 'trelease', 'imdl', 'iprp', 'irpt', and 'ilay'. Additional
            columns are ignored. Note that MODFLOW 6 PRT does not assign to
            particles a unique ID, but infers particle identity from 'imdl',
            'iprp', 'irpt', and 'trelease' combos (i.e. via composite key).
        travel_time : float or str
            Travel time selection. If a float, then pathlines with total
            time less than or equal to the given value are plotted. If a
            string, the value must be a comparison operator, then a time
            value. Valid operators are <=, <, ==, >=, and >. For example,
            to filter pathlines with less than 10000 units of total time
            traveled, use '< 10000'. (Default is None.)
        kwargs : dict
            Explicitly supported kwargs are layer, ax, colors.
            Any remaining kwargs are passed into the LineCollection
            constructor. If layer='all', pathlines are shown for all layers.

        Returns
        -------
        lc : matplotlib.collections.LineCollection
            The pathlines added to the plot.
        """

        from matplotlib.collections import LineCollection

        # make sure pl is a list
        if not isinstance(pl, list):
            if not isinstance(pl, (np.ndarray, pd.DataFrame)):
                raise TypeError(
                    "Pathline data must be a list of recarrays or dataframes, "
                    f"or a single recarray or dataframe, got {type(pl)}"
                )
            pl = [pl]

        # convert prt to mp7 format
        pl = [
            to_mp7_pathlines(
                p.to_records(index=False) if isinstance(p, pd.DataFrame) else p
            )
            for p in pl
        ]

        # merge pathlines then split on particleid
        pls = stack_arrays(pl, asrecarray=True, usemask=False)
        pids = np.unique(pls["particleid"])
        pl = [pls[pls["particleid"] == pid] for pid in pids]

        # configure layer
        if "layer" in kwargs:
            kon = kwargs.pop("layer")
            if isinstance(kon, bytes):
                kon = kon.decode()
            if isinstance(kon, str):
                if kon.lower() == "all":
                    kon = -1
                else:
                    kon = self.layer
        else:
            kon = -1

        # configure plot settings
        marker = kwargs.pop("marker", None)
        markersize = kwargs.pop("markersize", None)
        markersize = kwargs.pop("ms", markersize)
        markercolor = kwargs.pop("markercolor", None)
        markerevery = kwargs.pop("markerevery", 1)
        ax = kwargs.pop("ax", self.ax)
        if "colors" not in kwargs:
            kwargs["colors"] = "0.5"

        # compose pathlines
        linecol = []
        markers = []
        for p in pl:
            # filter by travel time
            tp = plotutil.filter_modpath_by_travel_time(p, travel_time)

            # transform data!
            x0r, y0r = geometry.transform(
                tp["x"],
                tp["y"],
                self.mg.xoffset,
                self.mg.yoffset,
                self.mg.angrot_radians,
            )

            # build polyline array
            arr = np.vstack((x0r, y0r)).T

            # select based on layer
            if kon >= 0:
                kk = p["k"].copy().reshape(p.shape[0], 1)
                kk = np.repeat(kk, 2, axis=1)
                arr = np.ma.masked_where((kk != kon), arr)
            else:
                arr = np.ma.asarray(arr)

            # append pathline if there are any unmasked segments
            if not arr.mask.all():
                linecol.append(arr)
                if not arr.mask.all():
                    linecol.append(arr)
                    if marker is not None:
                        for xy in arr[::markerevery]:
                            if not np.all(xy.mask):
                                markers.append(xy)

        # create line collection
        lc = None
        if len(linecol) > 0:
            lc = LineCollection(linecol, **kwargs)
            ax.add_collection(lc)
            if marker is not None:
                markers = np.array(markers)
                ax.plot(
                    markers[:, 0],
                    markers[:, 1],
                    lw=0,
                    marker=marker,
                    color=markercolor,
                    ms=markersize,
                )

        # set axis limits
        ax = self._set_axes_limits(ax)
        return lc

    def plot_timeseries(self, ts, travel_time=None, **kwargs):
        """
        Plot MODPATH 6 or 7 timeseries. Incompatible with MODFLOW 6 PRT.

        Parameters
        ----------
        ts : list of recarrays or dataframes, or a single recarray or dataframe
            Particle timeseries data. If a list of recarrays or dataframes,
            each must contain the path of only a single particle. If just
            one recarray or dataframe, it should contain the paths of all
            particles. Timeseries data returned from TimeseriesFile.get_data()
            or get_alldata() can be passed directly as this argument. Data
            columns should be 'x', 'y', 'z', 'time', 'k', and 'particleid'
            at minimum. Additional columns are ignored. The 'particleid'
            column must be unique to each particle path.
        travel_time : float or str
            Travel time selection. If a float, then pathlines with total
            time less than or equal to the given value are plotted. If a
            string, the value must be a comparison operator, then a time
            value. Valid operators are <=, <, ==, >=, and >. For example,
            to filter pathlines with less than 10000 units of total time
            traveled, use '< 10000'. (Default is None.)
        kwargs : dict
            Explicitly supported kwargs are layer, ax, colors.
            Any remaining kwargs are passed into the LineCollection
            constructor. If layer='all', pathlines are shown for all layers.

        Returns
        -------
        lc : matplotlib.collections.LineCollection
            The pathlines added to the plot.
        """
        if "color" in kwargs:
            kwargs["markercolor"] = kwargs["color"]

        return self.plot_pathline(ts, travel_time=travel_time, **kwargs)

    def plot_endpoint(
        self,
        ep,
        direction="ending",
        selection=None,
        selection_direction=None,
        **kwargs,
    ):
        """
        Plot particle endpoints. Compatible with MODFLOW 6 PRT particle
        track data format, or MODPATH 6 or 7 endpoint data format.

        Parameters
        ----------
        ep : recarray or dataframe
            A numpy recarray with the endpoint particle data from the
            MODPATH endpoint file.

            For MODFLOW 6 PRT pathlines, columns must include 'x', 'y', 'z',
            't', 'trelease', 'imdl', 'iprp', 'irpt', and 'ilay'. Additional
            columns are ignored. Note that MODFLOW 6 PRT does not assign to
            particles a unique ID, but infers particle identity from 'imdl',
            'iprp', 'irpt', and 'trelease' combos (i.e. via composite key).
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
        sp : matplotlib.collections.PathCollection
            The PathCollection added to the plot.

        """

        # convert ep to recarray if needed
        if isinstance(ep, pd.DataFrame):
            ep = ep.to_records(index=False)

        # convert ep from prt to mp7 format if needed
        if "t" in ep.dtype.names:
            from .plotutil import to_mp7_endpoints

            ep = to_mp7_endpoints(ep)

        # parse selection options
        ax = kwargs.pop("ax", self.ax)
        tep, _, xp, yp = plotutil.parse_modpath_selection_options(
            ep, direction, selection, selection_direction
        )

        # marker size
        s = kwargs.pop("s", np.sqrt(50))
        s = float(kwargs.pop("size", s)) ** 2.0

        # colorbar kwargs
        createcb = kwargs.pop("colorbar", False)
        colorbar_label = kwargs.pop("colorbar_label", "Endpoint Time")
        shrink = float(kwargs.pop("shrink", 1.0))

        # transform data!
        x0r, y0r = geometry.transform(
            tep[xp], tep[yp], self.mg.xoffset, self.mg.yoffset, self.mg.angrot_radians
        )
        # build array to plot
        arr = np.vstack((x0r, y0r)).T

        # plot the end point data
        if "c" in kwargs or "color" in kwargs:
            if "c" in kwargs and "color" in kwargs:
                kwargs.pop("color")
            sp = ax.scatter(arr[:, 0], arr[:, 1], s=s, **kwargs)
        else:
            c = tep["time"] - tep["time0"]
            sp = ax.scatter(arr[:, 0], arr[:, 1], c=c, s=s, **kwargs)

        # add a colorbar for travel times
        if createcb:
            cb = plt.colorbar(sp, ax=ax, shrink=shrink)
            cb.set_label(colorbar_label)

        ax = self._set_axes_limits(ax)
        return sp
