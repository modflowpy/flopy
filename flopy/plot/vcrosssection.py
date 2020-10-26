import numpy as np

try:
    import matplotlib.pyplot as plt
except:
    plt = None
from flopy.plot import plotutil
from flopy.utils import geometry
from flopy.plot.crosssection import _CrossSection
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


class _VertexCrossSection(_CrossSection):
    """
    Class to create a cross section of the model from a vertex
    discretization.

    Class is not to be instantiated by the user!

    Parameters
    ----------
    ax : matplotlib.pyplot axis
        The plot axis.  If not provided it, plt.gca() will be used.
    model : flopy.modflow object
        flopy model object. (Default is None)
    modelgrid : flopy.discretization.VertexGrid
        Vertex model grid object
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
    geographic_coords : bool
        boolean flag to allow the user to plot cross section lines in
        geographic coordinates. If False (default), cross section is plotted
        as the distance along the cross section line.

    """

    def __init__(
        self,
        ax=None,
        model=None,
        modelgrid=None,
        line=None,
        extent=None,
        geographic_coords=False,
    ):
        super(_VertexCrossSection, self).__init__(
            ax=ax,
            model=model,
            modelgrid=modelgrid,
            geographic_coords=geographic_coords,
        )

        if line is None:
            err_msg = "line must be specified."
            raise Exception(err_msg)

        linekeys = [linekeys.lower() for linekeys in list(line.keys())]

        if len(linekeys) != 1:
            err_msg = (
                "Either row, column, or line must be specified "
                "in line dictionary.\nkeys specified: "
            )
            for k in linekeys:
                err_msg += "{} ".format(k)
            raise Exception(err_msg)

        elif "line" not in linekeys:
            err_msg = (
                "only line can be specified in line dictionary "
                "for vertex Discretization"
            )
            raise AssertionError(err_msg)

        onkey = linekeys[0]

        if ax is None:
            self.ax = plt.gca()
        else:
            self.ax = ax

        self.direction = "xy"
        # convert pts list to a numpy array
        verts = line[onkey]
        xp = []
        yp = []
        for [v1, v2] in verts:
            xp.append(v1)
            yp.append(v2)

        # unrotate and untransform modelgrid into modflow coordinates!
        xp, yp = geometry.transform(
            xp,
            yp,
            self.mg.xoffset,
            self.mg.yoffset,
            self.mg.angrot_radians,
            inverse=True,
        )

        self.xcellcenters, self.ycellcenters = geometry.transform(
            self.mg.xcellcenters,
            self.mg.ycellcenters,
            self.mg.xoffset,
            self.mg.yoffset,
            self.mg.angrot_radians,
            inverse=True,
        )

        try:
            self.xvertices, self.yvertices = geometry.transform(
                self.mg.xvertices,
                self.mg.yvertices,
                self.mg.xoffset,
                self.mg.yoffset,
                self.mg.angrot_radians,
                inverse=True,
            )
        except ValueError:
            # irregular shapes in vertex grid ie. squares and triangles
            (
                xverts,
                yverts,
            ) = plotutil.UnstructuredPlotUtilities.irregular_shape_patch(
                self.mg.xvertices, self.mg.yvertices
            )

            self.xvertices, self.yvertices = geometry.transform(
                xverts,
                yverts,
                self.mg.xoffset,
                self.mg.yoffset,
                self.mg.angrot_radians,
                inverse=True,
            )

        pts = [(xt, yt) for xt, yt in zip(xp, yp)]
        self.pts = np.array(pts)

        # get points along the line

        self.xypts = plotutil.UnstructuredPlotUtilities.line_intersect_grid(
            self.pts, self.xvertices, self.yvertices
        )

        if len(self.xypts) < 2:
            s = "cross-section cannot be created\n."
            s += "   less than 2 points intersect the model grid\n"
            s += "   {} points intersect the grid.".format(len(self.xypts))
            raise Exception(s)

        if self.geographic_coords:
            # transform back to geographic coordinates
            xypts = {}
            for nn, pt in self.xypts.items():
                xp = [t[0] for t in pt]
                yp = [t[1] for t in pt]
                xp, yp = geometry.transform(
                    xp,
                    yp,
                    self.mg.xoffset,
                    self.mg.yoffset,
                    self.mg.angrot_radians,
                )
                xypts[nn] = [(xt, yt) for xt, yt in zip(xp, yp)]

            self.xypts = xypts

        top = self.mg.top
        top.shape = (1, -1)
        botm = self.mg.botm
        nlay = len(botm)
        ncpl = self.mg.ncpl

        elev = list(top.copy())
        for k in range(nlay):
            elev.append(botm[k, :])

        self.elev = np.array(elev)

        self.idomain = self.mg.idomain
        if self.mg.idomain is None:
            self.idomain = np.ones((nlay, ncpl), dtype=int)

        # choose a projection direction based on maximum information
        xpts = []
        ypts = []
        for nn, verts in self.xypts.items():
            for v in verts:
                xpts.append(v[0])
                ypts.append(v[1])

        if np.max(xpts) - np.min(xpts) > np.max(ypts) - np.min(ypts):
            self.direction = "x"
        else:
            self.direction = "y"

        # make vertex array based on projection direction
        self.projpts = self.set_zpts(None)

        # Create cross-section extent
        if extent is None:
            self.extent = self.get_extent()
        else:
            self.extent = extent

        self.layer0 = None
        self.layer1 = None

        self.d = {
            i: (np.min(np.array(v).T[0]), np.max(np.array(v).T[0]))
            for i, v in sorted(self.projpts.items())
        }

        self.xpts = None
        self.active = None
        self.ncb = None
        self.laycbd = None
        self.zpts = None
        self.xcentergrid = None
        self.zcentergrid = None
        self.geographic_xcentergrid = None
        self.geographic_xpts = None

        # Set axis limits
        self.ax.set_xlim(self.extent[0], self.extent[1])
        self.ax.set_ylim(self.extent[2], self.extent[3])

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
        if "ax" in kwargs:
            ax = kwargs.pop("ax")
        else:
            ax = self.ax

        if not isinstance(a, np.ndarray):
            a = np.array(a)

        if a.ndim > 1:
            a = np.ravel(a)

        if masked_values is not None:
            for mval in masked_values:
                a = np.ma.masked_values(a, mval)

        if isinstance(head, np.ndarray):
            projpts = self.set_zpts(np.ravel(head))
        else:
            projpts = self.projpts

        pc = self.get_grid_patch_collection(projpts, a, **kwargs)
        if pc is not None:
            ax.add_collection(pc)
            ax.set_xlim(self.extent[0], self.extent[1])
            ax.set_ylim(self.extent[2], self.extent[3])

        return pc

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
        if "ax" in kwargs:
            ax = kwargs.pop("ax")
        else:
            ax = self.ax

        if "color" in kwargs:
            color = kwargs.pop("color")
        elif "c" in kwargs:
            color = kwargs.pop("c")
        else:
            color = "b"

        if not isinstance(a, np.ndarray):
            a = np.array(a)

        if a.ndim > 1:
            a = np.ravel(a)

        if a.size % self.mg.ncpl != 0:
            raise AssertionError("Array size must be a multiple of ncpl")

        if masked_values is not None:
            for mval in masked_values:
                a = np.ma.masked_values(a, mval)

        data = []
        lay_data = []
        d = []
        lay_d = []
        dim = self.mg.ncpl
        for cell, verts in sorted(self.projpts.items()):

            if cell >= a.size:
                continue
            elif np.isnan(a[cell]):
                continue
            elif a[cell] is np.ma.masked:
                continue

            if cell >= dim:
                data.append(lay_data)
                d.append(lay_d)
                dim += self.mg.ncpl
                lay_data = [(a[cell], a[cell])]
                lay_d = [self.d[cell]]
            else:
                lay_data.append((a[cell], a[cell]))
                lay_d.append(self.d[cell])

        if lay_data:
            data.append(lay_data)
            d.append(lay_d)

        data = np.array(data)
        d = np.array(d)

        plot = []
        for k in range(data.shape[0]):
            if ax is None:
                ax = plt.gca()
            for ix, _ in enumerate(data[k]):
                ax.plot(d[k, ix], data[k, ix], color=color, **kwargs)

            ax.set_xlim(self.extent[0], self.extent[1])
            ax.set_ylim(self.extent[2], self.extent[3])
            plot.append(ax)

        return plot

    def plot_fill_between(
        self,
        a,
        colors=("blue", "red"),
        masked_values=None,
        head=None,
        **kwargs
    ):
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
        if "ax" in kwargs:
            ax = kwargs.pop("ax")
        else:
            ax = self.ax

        if not isinstance(a, np.ndarray):
            a = np.array(a)

        a = np.ravel(a)

        if masked_values is not None:
            for mval in masked_values:
                a = np.ma.masked_values(a, mval)

        if isinstance(head, np.ndarray):
            projpts = self.set_zpts(head)
        else:
            projpts = self.projpts

        plot = []
        for cell, verts in sorted(projpts.items()):
            if cell >= a.size:
                continue
            elif np.isnan(a[cell]):
                continue
            elif a[cell] is np.ma.masked:
                continue

            x = list(set(np.array(verts.T[0])))
            y1 = np.max(np.array(verts.T[1]))
            y2 = np.min(np.array(verts.T[1]))
            v = a[cell]

            if v > y1:
                v = y1

            elif v < y2:
                v = y2

            v = [v] * len(x)

            x = np.array(x)
            plot.append(ax.fill_between(x, y1, v, color=colors[0], **kwargs))
            plot.append(ax.fill_between(x, v, y2, color=colors[1], **kwargs))

        return plot

    def contour_array(self, a, masked_values=None, head=None, **kwargs):
        """
        Contour a two-dimensional array.

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
        if plt is None:
            err_msg = (
                "matplotlib must be installed to " + "use contour_array()"
            )
            raise ImportError(err_msg)
        else:
            import matplotlib.tri as tri

        if not isinstance(a, np.ndarray):
            a = np.array(a)

        if a.ndim > 1:
            a = np.ravel(a)

        if "ax" in kwargs:
            ax = kwargs.pop("ax")
        else:
            ax = self.ax

        xcenters = [
            np.mean(np.array(v).T[0]) for i, v in sorted(self.projpts.items())
        ]

        plotarray = np.array([a[cell] for cell in sorted(self.projpts)])

        # work around for tri-contour ignore vmin & vmax
        # necessary for the tri-contour NaN issue fix
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

        if isinstance(head, np.ndarray):
            zcenters = self.set_zcentergrid(np.ravel(head))
        else:
            zcenters = [
                np.mean(np.array(v).T[1])
                for i, v in sorted(self.projpts.items())
            ]

        plot_triplot = False
        if "plot_triplot" in kwargs:
            plot_triplot = kwargs.pop("plot_triplot")

        if "extent" in kwargs:
            extent = kwargs.pop("extent")

            idx = (
                (xcenters >= extent[0])
                & (xcenters <= extent[1])
                & (zcenters >= extent[2])
                & (zcenters <= extent[3])
            )
            plotarray = plotarray[idx].flatten()
            xcenters = xcenters[idx].flatten()
            zcenters = zcenters[idx].flatten()

        triang = tri.Triangulation(xcenters, zcenters)

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

    def plot_inactive(self):
        raise NotImplementedError(
            "Function must be called in PlotCrossSection"
        )

    def plot_ibound(self):
        raise NotImplementedError(
            "Function must be called in PlotCrossSection"
        )

    def plot_grid(self):
        raise NotImplementedError(
            "Function must be called in PlotCrossSection"
        )

    def plot_bc(self):
        raise NotImplementedError(
            "Function must be called in PlotCrossSection"
        )

    def plot_specific_discharge(self):
        raise NotImplementedError(
            "Function must be called in PlotCrossSection"
        )

    def plot_discharge(self):
        raise NotImplementedError(
            "plot_specific_discharge must be " "used for VertexGrid models"
        )

    @classmethod
    def get_grid_patch_collection(cls, projpts, plotarray, **kwargs):
        """
        Get a PatchCollection of plotarray in unmasked cells

        Parameters
        ----------
        projpts : dict
            dictionary defined by node number which contains model patch vertices.
        plotarray : numpy.ndarray
            One-dimensional array to attach to the Patch Collection.
        **kwargs : dictionary
            keyword arguments passed to matplotlib.collections.PatchCollection

        Returns
        -------
        patches : matplotlib.collections.PatchCollection

        """
        if plt is None:
            err_msg = (
                "matplotlib must be installed to "
                + "use get_grid_patch_collection()"
            )
            raise ImportError(err_msg)
        else:
            from matplotlib.patches import Polygon
            from matplotlib.collections import PatchCollection

        if "vmin" in kwargs:
            vmin = kwargs.pop("vmin")
        else:
            vmin = None
        if "vmax" in kwargs:
            vmax = kwargs.pop("vmax")
        else:
            vmax = None

        rectcol = []
        data = []
        for cell, verts in sorted(projpts.items()):
            verts = plotutil.UnstructuredPlotUtilities.arctan2(np.array(verts))

            if np.isnan(plotarray[cell]):
                continue
            elif plotarray[cell] is np.ma.masked:
                continue

            rectcol.append(Polygon(verts, closed=True))
            data.append(plotarray[cell])

        if len(rectcol) > 0:
            patches = PatchCollection(rectcol, **kwargs)
            patches.set_array(np.array(data))
            patches.set_clim(vmin, vmax)

        else:
            patches = None

        return patches

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
        if plt is None:
            err_msg = (
                "matplotlib must be installed to "
                + "use get_grid_line_collection()"
            )
            raise ImportError(err_msg)
        else:
            from matplotlib.patches import Polygon
            from matplotlib.collections import PatchCollection

        color = "grey"
        if "ec" in kwargs:
            color = kwargs.pop("ec")
        if color in kwargs:
            color = kwargs.pop("color")

        rectcol = []
        for _, verts in sorted(self.projpts.items()):
            verts = plotutil.UnstructuredPlotUtilities.arctan2(np.array(verts))

            rectcol.append(Polygon(verts, closed=True))

        if len(rectcol) > 0:
            patches = PatchCollection(
                rectcol, edgecolor=color, facecolor="none", **kwargs
            )
        else:
            patches = None

        return patches

    def set_zpts(self, vs):
        """
        Get an array of projection vertices corrected for
         elevations based on minimum of cell elevation
        (self.elev) or passed vs numpy.ndarray

        Parameters
        ----------
        vs : numpy.ndarray
            Two-dimensional array to plot.

        Returns
        -------
        zpts : dict

        """
        # make vertex array based on projection direction
        if vs is not None:
            if not isinstance(vs, np.ndarray):
                vs = np.array(vs)

        if self.direction == "x":
            xyix = 0
        else:
            xyix = -1

        projpts = {}
        for k in range(1, self.mg.nlay + 1):
            top = self.elev[k - 1, :]
            botm = self.elev[k, :]
            adjnn = (k - 1) * self.mg.ncpl
            d0 = 0
            for nn, verts in sorted(
                self.xypts.items(), key=lambda q: q[-1][xyix][xyix]
            ):
                if vs is None:
                    t = top[nn]
                else:
                    t = vs[nn]
                    if top[nn] < vs[nn]:
                        t = top[nn]
                b = botm[nn]
                if self.geographic_coords:
                    if self.direction == "x":
                        projt = [(v[0], t) for v in verts]
                        projb = [(v[0], b) for v in verts]
                    else:
                        projt = [(v[1], t) for v in verts]
                        projb = [(v[1], b) for v in verts]
                else:
                    verts = np.array(verts).T
                    a2 = (np.max(verts[0]) - np.min(verts[0])) ** 2
                    b2 = (np.max(verts[1]) - np.min(verts[1])) ** 2
                    c = np.sqrt(a2 + b2)
                    d1 = d0 + c
                    projt = [(d0, t), (d1, t)]
                    projb = [(d0, b), (d1, b)]
                    d0 += c

                projpts[nn + adjnn] = projt + projb

        return projpts

    def set_zcentergrid(self, vs, kstep=1):
        """
        Get an array of z elevations at the center of a cell that is based
        on minimum of cell top elevation (self.elev) or passed vs numpy.ndarray

        Parameters
        ----------
        vs : numpy.ndarray
            Three-dimensional array to plot.

        Returns
        -------
        zcentergrid : numpy.ndarray

        """
        verts = self.set_zpts(vs)
        zcenters = [
            np.mean(np.array(v).T[1])
            for i, v in sorted(verts.items())
            if (i // self.mg.ncpl) % kstep == 0
        ]
        return zcenters

    def get_extent(self):
        """
        Get the extent of the rotated and offset grid

        Returns
        -------
        tuple : (xmin, xmax, ymin, ymax)
        """
        xpts = []
        for _, verts in self.projpts.items():
            for v in verts:
                xpts.append(v[0])

        xmin = np.min(xpts)
        xmax = np.max(xpts)

        ymin = np.min(self.elev)
        ymax = np.max(self.elev)

        return (xmin, xmax, ymin, ymax)
