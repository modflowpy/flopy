import numpy as np
from ..discretization import StructuredGrid, UnstructuredGrid
from ..utils import geometry

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon
except ImportError:
    plt = None

from . import plotutil
import warnings

warnings.simplefilter("always", PendingDeprecationWarning)


class PlotMapView(object):
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

    def __init__(
        self, model=None, modelgrid=None, ax=None, layer=0, extent=None
    ):

        if plt is None:
            s = (
                "Could not import matplotlib.  Must install matplotlib "
                + " in order to use ModelMap method"
            )
            raise ImportError(s)

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

        if self.mg.grid_type not in ("structured", "vertex", "unstructured"):
            err_msg = "Unrecognized modelgrid type {}"
            raise TypeError(err_msg.format(self.mg.grid_type))

        if ax is None:
            try:
                self.ax = plt.gca()
                self.ax.set_aspect("equal")
            except:
                self.ax = plt.subplot(1, 1, 1, aspect="equal", axisbg="white")
        else:
            self.ax = ax

        if extent is not None:
            self._extent = extent
        else:
            self._extent = None

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
        quadmesh : matplotlib.collections.QuadMesh or
            matplotlib.collections.PatchCollection

        """

        if self.mg.grid_type not in ("structured", "vertex", "unstructured"):
            raise TypeError(
                "Unrecognized grid type {}".format(self.mg.grid_type)
            )

        if not isinstance(a, np.ndarray):
            a = np.array(a)

        # Use the model grid to pass back an array of the correct shape
        plotarray = self.mg.get_plottable_layer_array(a, self.layer)

        # if masked_values are provided mask the plotting array
        if masked_values is not None:
            for mval in masked_values:
                plotarray = np.ma.masked_values(plotarray, mval)

        # add NaN values to mask
        plotarray = np.ma.masked_where(np.isnan(plotarray), plotarray)

        if "ax" in kwargs:
            ax = kwargs.pop("ax")
        else:
            ax = self.ax

        # Get vertices for the selected layer
        xgrid = self.mg.get_xvertices_for_layer(self.layer)
        ygrid = self.mg.get_yvertices_for_layer(self.layer)

        if self.mg.grid_type == "structured":
            quadmesh = ax.pcolormesh(xgrid, ygrid, plotarray)
        else:
            # use patch collection for vertex and unstructured
            patches = [
                Polygon(list(zip(xgrid[i], ygrid[i])), closed=True)
                for i in range(xgrid.shape[0])
            ]
            quadmesh = PatchCollection(patches)
            quadmesh.set_array(plotarray)

        # set max and min
        if "vmin" in kwargs:
            vmin = kwargs.pop("vmin")
        else:
            vmin = None

        if "vmax" in kwargs:
            vmax = kwargs.pop("vmax")
        else:
            vmax = None

        # limit the color range
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
            err_msg = "matplotlib must be installed to use contour_array()"
            raise ImportError(err_msg)

        a = np.copy(a)
        if not isinstance(a, np.ndarray):
            a = np.array(a)

        # Use the model grid to pass back an array of the correct shape
        plotarray = self.mg.get_plottable_layer_array(a, self.layer)

        # work around for tri-contour ignore vmin & vmax
        # necessary block for tri-contour NaN issue
        if "levels" not in kwargs:
            if "vmin" not in kwargs:
                vmin = np.nanmin(plotarray)
            else:
                vmin = kwargs.pop("vmin")
            if "vmax" not in kwargs:
                vmax = np.nanmax(plotarray)
            else:
                vmax = kwargs.pop("vmax")

            levels = np.linspace(vmin, vmax, 7)
            kwargs["levels"] = levels

        # workaround for tri-contour nan issue
        # use -2**31 to allow for 32 bit int arrays
        plotarray[np.isnan(plotarray)] = -(2 ** 31)
        if masked_values is None:
            masked_values = [-(2 ** 31)]
        else:
            masked_values = list(masked_values)
            if -(2 ** 31) not in masked_values:
                masked_values.append(-(2 ** 31))

        ismasked = None
        if masked_values is not None:
            for mval in masked_values:
                if ismasked is None:
                    ismasked = np.isclose(plotarray, mval)
                else:
                    t = np.isclose(plotarray, mval)
                    ismasked += t

        if "ax" in kwargs:
            ax = kwargs.pop("ax")
        else:
            ax = self.ax

        if "colors" in kwargs.keys():
            if "cmap" in kwargs.keys():
                kwargs.pop("cmap")

        plot_triplot = False
        if "plot_triplot" in kwargs:
            plot_triplot = kwargs.pop("plot_triplot")

        # Get vertices for the selected layer
        xcentergrid = self.mg.get_xcellcenters_for_layer(self.layer)
        ycentergrid = self.mg.get_ycellcenters_for_layer(self.layer)

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

        plotarray = plotarray.flatten()
        xcentergrid = xcentergrid.flatten()
        ycentergrid = ycentergrid.flatten()
        triang = tri.Triangulation(xcentergrid, ycentergrid)

        if ismasked is not None:
            ismasked = ismasked.flatten()
            mask = np.any(
                np.where(ismasked[triang.triangles], True, False), axis=1
            )
            triang.set_mask(mask)

        contour_set = ax.tricontour(triang, plotarray, **kwargs)

        if plot_triplot:
            ax.triplot(triang, color="black", marker="o", lw=0.75)

        ax.set_xlim(self.extent[0], self.extent[1])
        ax.set_ylim(self.extent[2], self.extent[3])

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
        if plt is None:
            err_msg = "matplotlib must be installed to use plot_inactive()"
            raise ImportError(err_msg)

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
        **kwargs
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
        if plt is None:
            err_msg = "matplotlib must be installed to use plot_ibound()"
            raise ImportError(err_msg)

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
        if plt is None:
            err_msg = "matplotlib must be installed to use plot_grid()"
            raise ImportError(err_msg)
        else:
            from matplotlib.collections import LineCollection

        if "ax" in kwargs:
            ax = kwargs.pop("ax")
        else:
            ax = self.ax

        if "colors" not in kwargs:
            kwargs["colors"] = "0.5"

        grid_lines = self.mg.grid_lines
        if isinstance(grid_lines, dict):
            # grid_lines are passed back as a dictionary with keys equal to
            # layers for an UnstructuredGrid
            grid_lines = grid_lines[self.layer]
        lc = LineCollection(grid_lines, **kwargs)

        ax.add_collection(lc)
        ax.set_xlim(self.extent[0], self.extent[1])
        ax.set_ylim(self.extent[2], self.extent[3])

        return lc

    def plot_bc(
        self,
        name=None,
        package=None,
        kper=0,
        color=None,
        plotAll=False,
        **kwargs
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
                        raise Exception(
                            "Not a list-style boundary package: " + str(e)
                        )
                    if mflist is None:
                        return

                    t = np.array(
                        [list(i) for i in mflist["cellid"]], dtype=int
                    ).T

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
                    raise Exception(
                        "Not a list-style boundary package: " + str(e)
                    )
                if mflist is None:
                    return
                if len(self.mg.shape) == 3:
                    idx = [mflist["k"], mflist["i"], mflist["j"]]
                else:
                    idx = mflist["node"]

        if plotAll and self.mg.grid_type == "unstructured":
            raise Exception("plotAll cannot be used with unstructured grid.")
        else:
            nlay = self.mg.nlay

        # Plot the list locations
        plotarray = np.zeros(self.mg.shape, dtype=int)
        if plotAll:
            pa = np.zeros(self.mg.shape[1:], dtype=int)
            pa[tuple(idx[1:])] = 1
            for k in range(nlay):
                plotarray[k] = pa.copy()
        else:
            plotarray[tuple(idx)] = 1

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
        shp : string or pyshp shapefile object
            Name of the shapefile to plot

        kwargs : dictionary
            Keyword arguments passed to plotutil.plot_shapefile()

        """
        if "ax" in kwargs:
            ax = kwargs.pop("ax")
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
        if "ax" in kwargs:
            ax = kwargs.pop("ax")
        else:
            ax = self.ax
        patch_collection = plotutil.plot_cvfd(
            verts, iverts, ax, self.layer, **kwargs
        )
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
        try:
            import matplotlib.tri as tri
        except ImportError:
            err_msg = "matplotlib must be updated to use contour_array()"
            raise ImportError(err_msg)

        if "ncpl" in kwargs:
            nlay = self.layer + 1
            ncpl = kwargs.pop("ncpl")
            if isinstance(ncpl, int):
                i = int(ncpl)
                ncpl = np.ones((nlay,), dtype=int) * i
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

        ismasked = None
        if masked_values is not None:
            for mval in masked_values:
                if ismasked is None:
                    ismasked = np.isclose(plotarray, mval)
                else:
                    t = np.isclose(plotarray, mval)
                    ismasked += t

        # add NaN values to mask
        if ismasked is None:
            ismasked = np.isnan(plotarray)
        else:
            ismasked += np.isnan(plotarray)

        if "ax" in kwargs:
            ax = kwargs.pop("ax")
        else:
            ax = self.ax

        if "colors" in kwargs.keys():
            if "cmap" in kwargs.keys():
                kwargs.pop("cmap")

        triang = tri.Triangulation(vertc[:, 0], vertc[:, 1])

        if ismasked is not None:
            ismasked = ismasked.flatten()
            mask = np.any(
                np.where(ismasked[triang.triangles], True, False), axis=1
            )
            triang.set_mask(mask)

        contour_set = ax.tricontour(triang, plotarray, **kwargs)

        return contour_set

    def plot_vector(
        self,
        vx,
        vy,
        istep=1,
        jstep=1,
        normalize=False,
        masked_values=None,
        **kwargs
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
        if "pivot" in kwargs:
            pivot = kwargs.pop("pivot")
        else:
            pivot = "middle"

        if "ax" in kwargs:
            ax = kwargs.pop("ax")
        else:
            ax = self.ax

        # get actual values to plot
        if self.mg.grid_type == "structured":
            x = self.mg.xcellcenters[::istep, ::jstep]
            y = self.mg.ycellcenters[::istep, ::jstep]
            u = vx[self.layer, ::istep, ::jstep]
            v = vy[self.layer, ::istep, ::jstep]
        else:
            x = self.mg.xcellcenters[::istep]
            y = self.mg.ycellcenters[::istep]
            u = vx[self.layer, ::istep]
            v = vy[self.layer, ::istep]

        # if necessary, copy to avoid changing the passed values
        if masked_values is not None or normalize:
            import copy

            u = copy.copy(u)
            v = copy.copy(v)

        # mask values
        if masked_values is not None:
            for mval in masked_values:
                to_mask = np.logical_or(u == mval, v == mval)
                u[to_mask] = np.nan
                v[to_mask] = np.nan

        # normalize
        if normalize:
            vmag = np.sqrt(u ** 2.0 + v ** 2.0)
            idx = vmag > 0.0
            u[idx] /= vmag[idx]
            v[idx] /= vmag[idx]

        # rotate and plot, offsets must be zero since
        # these are vectors not locations
        urot, vrot = geometry.rotate(u, v, 0.0, 0.0, self.mg.angrot_radians)

        # plot with quiver
        quiver = ax.quiver(x, y, urot, vrot, pivot=pivot, **kwargs)

        return quiver

    def plot_specific_discharge(
        self, spdis, istep=1, jstep=1, normalize=False, **kwargs
    ):
        """
        DEPRECATED. Use plot_vector() instead, which should follow after
        postprocessing.get_specific_discharge().

        Method to plot specific discharge from discharge vectors
        provided by the cell by cell flow output file. In MODFLOW-6
        this option is controled in the NPF options block. This method
        uses matplotlib quiver to create a matplotlib plot of the output.

        Parameters
        ----------
        spdis : np.recarray
            specific discharge recarray from cbc file
        istep : int
            row frequency to plot. (Default is 1.)
        jstep : int
            column frequency to plot. (Default is 1.)
        kwargs : matplotlib.pyplot keyword arguments for the
            plt.quiver method.

        Returns
        -------
        quiver : matplotlib.pyplot.quiver
            quiver plot of discharge vectors

        """
        warnings.warn(
            "plot_specific_discharge() has been deprecated. Use "
            "plot_vector() instead, which should follow after "
            "postprocessing.get_specific_discharge()",
            DeprecationWarning,
        )

        if "pivot" in kwargs:
            pivot = kwargs.pop("pivot")
        else:
            pivot = "middle"

        if "ax" in kwargs:
            ax = kwargs.pop("ax")
        else:
            ax = self.ax

        if isinstance(spdis, list):
            print(
                "Warning: Selecting the final stress period from Specific"
                " Discharge list"
            )
            spdis = spdis[-1]

        nodes = self.mg.nnodes

        qx = np.zeros(nodes)
        qy = np.zeros(nodes)

        idx = np.array(spdis["node"]) - 1
        qx[idx] = spdis["qx"]
        qy[idx] = spdis["qy"]

        qx = self.mg.get_plottable_layer_array(qx, self.layer)
        qy = self.mg.get_plottable_layer_array(qy, self.layer)

        # Get vertices for the selected layer
        xcentergrid = self.mg.get_xcellcenters_for_layer(self.layer)
        ycentergrid = self.mg.get_ycellcenters_for_layer(self.layer)

        if self.mg.grid_type == "structured":
            x = xcentergrid[::istep, ::jstep]
            y = ycentergrid[::istep, ::jstep]
            u = qx[::istep, ::jstep]
            v = qy[::istep, ::jstep]
        else:
            x = xcentergrid[::istep]
            y = ycentergrid[::istep]
            u = qx[::istep]
            v = qy[::istep]

        # normalize
        if normalize:
            vmag = np.sqrt(u ** 2.0 + v ** 2.0)
            idx = vmag > 0.0
            u[idx] /= vmag[idx]
            v[idx] /= vmag[idx]

        u[u == 0] = np.nan
        v[v == 0] = np.nan

        # Rotate and plot, offsets must be zero since
        # these are vectors not locations
        urot, vrot = geometry.rotate(u, v, 0.0, 0.0, self.mg.angrot_radians)
        quiver = ax.quiver(x, y, urot, vrot, pivot=pivot, **kwargs)
        return quiver

    def plot_discharge(
        self,
        frf=None,
        fff=None,
        flf=None,
        head=None,
        istep=1,
        jstep=1,
        normalize=False,
        **kwargs
    ):
        """
        DEPRECATED. Use plot_vector() instead, which should follow after
        postprocessing.get_specific_discharge().

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
        warnings.warn(
            "plot_discharge() has been deprecated. Use "
            "plot_vector() instead, which should follow after "
            "postprocessing.get_specific_discharge()",
            DeprecationWarning,
        )

        if self.mg.grid_type != "structured":
            err_msg = "Use plot_specific_discharge for " "{} grids".format(
                self.mg.grid_type
            )
            raise NotImplementedError(err_msg)

        else:
            if self.mg.top is None:
                err = (
                    "StructuredGrid must have top and "
                    "botm defined to use plot_discharge()"
                )
                raise AssertionError(err)

            ib = np.ones((self.mg.nlay, self.mg.nrow, self.mg.ncol))
            if self.mg.idomain is not None:
                ib = self.mg.idomain

            delr = self.mg.delr
            delc = self.mg.delc
            top = np.copy(self.mg.top)
            botm = np.copy(self.mg.botm)
            laytyp = None
            hnoflo = 999.0
            hdry = 999.0
            laycbd = None

            if self.model is not None:
                if self.model.laytyp is not None:
                    laytyp = self.model.laytyp

                if self.model.hnoflo is not None:
                    hnoflo = self.model.hnoflo

                if self.model.hdry is not None:
                    hdry = self.model.hdry

                if self.model.laycbd is not None:
                    laycbd = self.model.laycbd

            if laycbd is not None and 1 in laycbd:
                active = np.ones((botm.shape[0],), dtype=int)
                kon = 0
                for cbd in laycbd:
                    if cbd > 0:
                        kon += 1
                        active[kon] = 0
                botm = botm[active == 1]

            # If no access to head or laytyp, then calculate confined saturated
            # thickness by setting laytyp to zeros
            if head is None or laytyp is None:
                head = np.zeros(botm.shape, np.float32)
                laytyp = np.zeros((botm.shape[0],), dtype=int)

            # calculate the saturated thickness
            sat_thk = plotutil.PlotUtilities.saturated_thickness(
                head, top, botm, laytyp, [hnoflo, hdry]
            )

            # Calculate specific discharge
            qx, qy, qz = plotutil.PlotUtilities.centered_specific_discharge(
                frf, fff, flf, delr, delc, sat_thk
            )
            ib = ib.ravel()
            qx = qx.ravel()
            qy = qy.ravel()
            del qz

            temp = []
            for ix, val in enumerate(ib):
                if val != 0:
                    temp.append((ix + 1, qx[ix], qy[ix]))

            spdis = np.recarray(
                (len(temp),),
                dtype=[("node", int), ("qx", float), ("qy", float)],
            )
            for ix, tup in enumerate(temp):
                spdis[ix] = tup

            return self.plot_specific_discharge(
                spdis, istep=istep, jstep=jstep, normalize=normalize, **kwargs
            )

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
        travel_time : float or str
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
        if plt is None:
            err_msg = "matplotlib must be installed to use plot_pathline()"
            raise ImportError(err_msg)
        else:
            from matplotlib.collections import LineCollection

        # make sure pathlines is a list
        if not isinstance(pl, list):
            pl = [pl]

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
            kon = self.layer

        if "marker" in kwargs:
            marker = kwargs.pop("marker")
        else:
            marker = None

        if "markersize" in kwargs:
            markersize = kwargs.pop("markersize")
        elif "ms" in kwargs:
            markersize = kwargs.pop("ms")
        else:
            markersize = None

        if "markercolor" in kwargs:
            markercolor = kwargs.pop("markercolor")
        else:
            markercolor = None

        if "markerevery" in kwargs:
            markerevery = kwargs.pop("markerevery")
        else:
            markerevery = 1

        if "ax" in kwargs:
            ax = kwargs.pop("ax")
        else:
            ax = self.ax

        if "colors" not in kwargs:
            kwargs["colors"] = "0.5"

        linecol = []
        markers = []
        for p in pl:
            if travel_time is None:
                tp = p.copy()
            else:
                if isinstance(travel_time, str):
                    if "<=" in travel_time:
                        time = float(travel_time.replace("<=", ""))
                        idx = p["time"] <= time
                    elif "<" in travel_time:
                        time = float(travel_time.replace("<", ""))
                        idx = p["time"] < time
                    elif ">=" in travel_time:
                        time = float(travel_time.replace(">=", ""))
                        idx = p["time"] >= time
                    elif "<" in travel_time:
                        time = float(travel_time.replace(">", ""))
                        idx = p["time"] > time
                    else:
                        try:
                            time = float(travel_time)
                            idx = p["time"] <= time
                        except:
                            errmsg = (
                                "flopy.map.plot_pathline travel_time "
                                + "variable cannot be parsed. "
                                + "Acceptable logical variables are , "
                                + "<=, <, >=, and >. "
                                + "You passed {}".format(travel_time)
                            )
                            raise Exception(errmsg)
                else:
                    time = float(travel_time)
                    idx = p["time"] <= time
                tp = p[idx]

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
                ax.plot(
                    markers[:, 0],
                    markers[:, 1],
                    lw=0,
                    marker=marker,
                    color=markercolor,
                    ms=markersize,
                )
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
        travel_time : float or str
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
        if plt is None:
            err_msg = "matplotlib must be installed to use plot_timeseries()"
            raise ImportError(err_msg)

        # make sure timeseries is a list
        if not isinstance(ts, list):
            ts = [ts]

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
            kon = self.layer

        if "ax" in kwargs:
            ax = kwargs.pop("ax")

        else:
            ax = self.ax

        if "color" not in kwargs:
            kwargs["color"] = "red"

        linecol = []
        for t in ts:
            if travel_time is None:
                tp = t.copy()

            else:
                if isinstance(travel_time, str):
                    if "<=" in travel_time:
                        time = float(travel_time.replace("<=", ""))
                        idx = t["time"] <= time
                    elif "<" in travel_time:
                        time = float(travel_time.replace("<", ""))
                        idx = t["time"] < time
                    elif ">=" in travel_time:
                        time = float(travel_time.replace(">=", ""))
                        idx = t["time"] >= time
                    elif "<" in travel_time:
                        time = float(travel_time.replace(">", ""))
                        idx = t["time"] > time
                    else:
                        try:
                            time = float(travel_time)
                            idx = t["time"] <= time
                        except:
                            errmsg = (
                                "flopy.map.plot_pathline travel_time "
                                + "variable cannot be parsed. "
                                + "Acceptable logical variables are , "
                                + "<=, <, >=, and >. "
                                + "You passed {}".format(travel_time)
                            )
                            raise Exception(errmsg)
                else:
                    time = float(travel_time)
                    idx = t["time"] <= time
                tp = ts[idx]

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
                kk = t["k"].copy().reshape(t.shape[0], 1)
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

    def plot_endpoint(
        self,
        ep,
        direction="ending",
        selection=None,
        selection_direction=None,
        **kwargs
    ):
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
        if plt is None:
            err_msg = "matplotlib must be installed to use plot_endpoint()"
            raise ImportError(err_msg)

        ep = ep.copy()
        direction = direction.lower()
        if direction == "starting":
            xp, yp = "x0", "y0"

        elif direction == "ending":
            xp, yp = "x", "y"

        else:
            errmsg = (
                'flopy.map.plot_endpoint direction must be "ending" '
                + 'or "starting".'
            )
            raise Exception(errmsg)

        if selection_direction is not None:
            if (
                selection_direction.lower() != "starting"
                and selection_direction.lower() != "ending"
            ):
                errmsg = (
                    "flopy.map.plot_endpoint selection_direction "
                    + 'must be "ending" or "starting".'
                )
                raise Exception(errmsg)
        else:
            if direction.lower() == "starting":
                selection_direction = "ending"
            elif direction.lower() == "ending":
                selection_direction = "starting"

        # selection of endpoints
        if selection is not None:
            if isinstance(selection, int):
                selection = tuple((selection,))
            try:
                if len(selection) == 1:
                    node = selection[0]
                    if selection_direction.lower() == "starting":
                        nsel = "node0"
                    else:
                        nsel = "node"
                    # make selection
                    idx = ep[nsel] == node
                    tep = ep[idx]
                elif len(selection) == 3:
                    k, i, j = selection[0], selection[1], selection[2]
                    if selection_direction.lower() == "starting":
                        ksel, isel, jsel = "k0", "i0", "j0"
                    else:
                        ksel, isel, jsel = "k", "i", "j"
                    # make selection
                    idx = (ep[ksel] == k) & (ep[isel] == i) & (ep[jsel] == j)
                    tep = ep[idx]
                else:
                    errmsg = (
                        "flopy.map.plot_endpoint selection must be "
                        + "a zero-based layer, row, column tuple "
                        + "(l, r, c) or node number (MODPATH 7) of "
                        + "the location to evaluate (i.e., well location)."
                    )
                    raise Exception(errmsg)
            except:
                errmsg = (
                    "flopy.map.plot_endpoint selection must be a "
                    + "zero-based layer, row, column tuple (l, r, c) "
                    + "or node number (MODPATH 7) of the location "
                    + "to evaluate (i.e., well location)."
                )
                raise Exception(errmsg)
        # all endpoints
        else:
            tep = ep.copy()

        if "ax" in kwargs:
            ax = kwargs.pop("ax")
        else:
            ax = self.ax

        # scatter kwargs that users may redefine
        if "c" not in kwargs:
            c = tep["time"] - tep["time0"]
        else:
            c = np.empty((tep.shape[0]), dtype="S30")
            c.fill(kwargs.pop("c"))

        s = 50
        if "s" in kwargs:
            s = float(kwargs.pop("s")) ** 2.0
        elif "size" in kwargs:
            s = float(kwargs.pop("size")) ** 2.0

        # colorbar kwargs
        createcb = False
        if "colorbar" in kwargs:
            createcb = kwargs.pop("colorbar")

        colorbar_label = "Endpoint Time"
        if "colorbar_label" in kwargs:
            colorbar_label = kwargs.pop("colorbar_label")

        shrink = 1.0
        if "shrink" in kwargs:
            shrink = float(kwargs.pop("shrink"))

        # transform data!
        x0r, y0r = geometry.transform(
            tep[xp],
            tep[yp],
            self.mg.xoffset,
            self.mg.yoffset,
            self.mg.angrot_radians,
        )
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

    Parameters
    ----------
    model : flopy.modflow.Modflow object
    modelgrid : flopy.discretization.Grid object
    ax : matplotlib.pyplot.axes object
    layer : int
        model layer to plot, default is layer 1
    extent : tuple of floats
        (xmin, xmax, ymin, ymax) will be used to specify axes limits.  If None
        then these will be calculated based on grid, coordinates, and rotation.

    """

    def __init__(
        self, model=None, modelgrid=None, ax=None, layer=0, extent=None
    ):
        super(DeprecatedMapView, self).__init__(
            model=model, modelgrid=modelgrid, ax=ax, layer=layer, extent=extent
        )

    def plot_discharge(
        self,
        frf,
        fff,
        dis=None,
        flf=None,
        head=None,
        istep=1,
        jstep=1,
        normalize=False,
        **kwargs
    ):
        """
        Use quiver to plot vectors. Deprecated method that uses
        the old function call to pass the method to PlotMapView

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

        if dis is not None:
            self.mg = plotutil._depreciated_dis_handler(
                modelgrid=self.mg, dis=dis
            )

        super(DeprecatedMapView, self).plot_discharge(
            frf=frf,
            fff=fff,
            flf=flf,
            head=head,
            istep=1,
            jstep=1,
            normalize=normalize,
            **kwargs
        )


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

    def __new__(
        cls,
        sr=None,
        ax=None,
        model=None,
        dis=None,
        layer=0,
        extent=None,
        xul=None,
        yul=None,
        xll=None,
        yll=None,
        rotation=None,
        length_multiplier=None,
    ):

        from ..utils.reference import SpatialReferenceUnstructured

        # from ..plot.plotbase import DeprecatedMapView

        err_msg = (
            "ModelMap will be replaced by "
            "PlotMapView(); Calling PlotMapView()"
        )
        warnings.warn(err_msg, PendingDeprecationWarning)

        modelgrid = None
        if model is not None:
            if (xul, yul, xll, yll, rotation) != (
                None,
                None,
                None,
                None,
                None,
            ):
                modelgrid = plotutil._set_coord_info(
                    model.modelgrid, xul, yul, xll, yll, rotation
                )
        elif sr is not None:
            if length_multiplier is not None:
                sr.length_multiplier = length_multiplier

            if (xul, yul, xll, yll, rotation) != (
                None,
                None,
                None,
                None,
                None,
            ):
                sr.set_spatialreference(xul, yul, xll, yll, rotation)

            if isinstance(sr, SpatialReferenceUnstructured):
                if dis is not None:
                    modelgrid = UnstructuredGrid(
                        vertices=sr.verts,
                        iverts=sr.iverts,
                        xcenters=sr.xc,
                        ycenters=sr.yc,
                        top=dis.top.array,
                        botm=dis.botm.array,
                        ncpl=sr.ncpl,
                    )
                else:
                    modelgrid = UnstructuredGrid(
                        vertices=sr.verts,
                        iverts=sr.iverts,
                        xcenters=sr.xc,
                        ycenters=sr.yc,
                        ncpl=sr.ncpl,
                    )

            elif dis is not None:
                modelgrid = StructuredGrid(
                    delc=sr.delc,
                    delr=sr.delr,
                    top=dis.top.array,
                    botm=dis.botm.array,
                    xoff=sr.xll,
                    yoff=sr.yll,
                    angrot=sr.rotation,
                )
            else:
                modelgrid = StructuredGrid(
                    delc=sr.delc,
                    delr=sr.delr,
                    xoff=sr.xll,
                    yoff=sr.yll,
                    angrot=sr.rotation,
                )

        else:
            pass

        return DeprecatedMapView(
            model=model, modelgrid=modelgrid, ax=ax, layer=layer, extent=extent
        )
