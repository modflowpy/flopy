import numpy as np

try:
    import matplotlib.pyplot as plt
except:
    plt = None
from flopy.plot import plotutil
from flopy.utils import geometry
import warnings

warnings.simplefilter("always", PendingDeprecationWarning)


class _CrossSection(object):
    """
    Base class for CrossSection plotting. Handles the model grid
    transforms and searching for modelgrid and dis file information.

    This class must be general with absolutely no code specific to
    a single model grid type. The user should not directly instantiate this
    class

    Parameters
    ----------
    ax : matplotlib.pyplot.axes object
    model : flopy.mf6.Modflow or flopy.modflow.Modflow object
    modelgrid : flopy.discretization.grid object
    geographic_coords : bool
        boolean flag to allow the user to plot cross section lines in
        geographic coordinates. If False (default), cross section is plotted
        as the distance along the cross section line.

    """

    def __init__(
        self, ax=None, model=None, modelgrid=None, geographic_coords=False
    ):

        self.ax = ax
        self.geographic_coords = geographic_coords
        if plt is None:
            s = (
                "Could not import matplotlib.  Must install matplotlib "
                + " in order to use ModelCrossSection method"
            )
            raise ImportError(s)

        self.model = model

        if model is not None:
            self.mg = model.modelgrid

        elif modelgrid is not None:
            self.mg = modelgrid
            if self.mg is None:
                raise AssertionError("Cannot find model grid ")

        else:
            raise Exception("Cannot find model grid")

        if self.mg.top is None or self.mg.botm is None:
            raise AssertionError("modelgrid top and botm must be defined")


class _StructuredCrossSection(_CrossSection):
    """
    Class to create a cross section of the model using
    Structured discretization.

    Class is not to be instantiated by the user.

    Parameters
    ----------
    ax : matplotlib.pyplot axis
        The plot axis.  If not provided it, plt.gca() will be used.
    model : flopy.modflow object
        flopy model object. (Default is None)
    modelgrid : flopy.discretization.StructuredGrid
        Structured model grid object
    line : dict
        Dictionary with either "row", "column", or "line" key. If key
        is "row" or "column" key value should be the zero-based row or
        column index for cross-section. If key is "line" value should
        be an array of (x, y) tuples with vertices of cross-section.
        Vertices should be in map coordinates consistent with xul,
        yul, and rotation.
    extent : tuple of floats
        (xmin, xmax, ymin, ymax) will be used to specify axes limits.  If None
        then these will be calculated based on grid, coordinates, and rotation
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
        super(_StructuredCrossSection, self).__init__(
            ax=ax,
            model=model,
            modelgrid=modelgrid,
            geographic_coords=geographic_coords,
        )

        if line is None:
            s = "line must be specified."
            raise Exception(s)

        linekeys = [linekeys.lower() for linekeys in list(line.keys())]

        if len(linekeys) != 1:
            s = "only row, column, or line can be specified in line dictionary.\n"
            s += "keys specified: "
            for k in linekeys:
                s += "{} ".format(k)
            raise AssertionError(s)

        if ax is None:
            self.ax = plt.gca()
        else:
            self.ax = ax

        onkey = list(line.keys())[0]
        eps = 1.0e-4
        xedge, yedge = self.mg.xyedges
        self.__geographic_xpts = None

        # un-translate model grid into model coordinates
        self.xcellcenters, self.ycellcenters = geometry.transform(
            self.mg.xcellcenters,
            self.mg.ycellcenters,
            self.mg.xoffset,
            self.mg.yoffset,
            self.mg.angrot_radians,
            inverse=True,
        )

        if "row" in linekeys:
            self.direction = "x"
            ycenter = self.ycellcenters.T[0]
            pts = [
                (xedge[0] + eps, ycenter[int(line[onkey])] - eps),
                (xedge[-1] - eps, ycenter[int(line[onkey])] + eps),
            ]
        elif "column" in linekeys:
            self.direction = "y"
            xcenter = self.xcellcenters[0, :]
            pts = [
                (xcenter[int(line[onkey])] + eps, yedge[0] - eps),
                (xcenter[int(line[onkey])] - eps, yedge[-1] + eps),
            ]
        else:
            self.direction = "xy"
            verts = line[onkey]
            xp = []
            yp = []
            for [v1, v2] in verts:
                xp.append(v1)
                yp.append(v2)

            xp, yp = self.mg.get_local_coords(xp, yp)
            pts = [(xt, yt) for xt, yt in zip(xp, yp)]
            # for now set offset to zero, since we do not have
            # information on projection from the user

        # convert pts list to numpy array
        self.pts = np.array(pts)

        # get points along the line
        self.xpts = plotutil.line_intersect_grid(
            self.pts, self.mg.xyedges[0], self.mg.xyedges[1]
        )
        if len(self.xpts) < 2:
            s = "cross-section cannot be created\n."
            s += "   less than 2 points intersect the model grid\n"
            s += "   {} points intersect the grid.".format(len(self.xpts))
            raise Exception(s)

        # set horizontal distance
        d = []
        for v in self.xpts:
            d.append(v[2])
        self.d = np.array(d)

        self.idomain = self.mg.idomain
        if self.mg.idomain is None:
            self.idomain = np.ones(
                (self.mg.nlay, self.mg.nrow, self.mg.ncol), dtype=int
            )

        self.ncb = 0
        self.laycbd = []

        if self.model is not None:
            if self.model.laycbd is not None:
                self.laycbd = self.model.laycbd

        for l in self.laycbd:
            if l > 0:
                self.ncb += 1

        self.active = np.ones((self.mg.nlay + self.ncb), dtype=int)
        kon = 0

        if len(self.laycbd) > 0:
            for k in range(self.mg.nlay):
                if self.laycbd[k] > 0:
                    kon += 1
                    self.active[kon] = 0
                kon += 1

        top = self.mg.top
        botm = self.mg.botm
        elev = [top.copy()]
        for k in range(self.mg.nlay + self.ncb):
            elev.append(botm[k, :, :])

        self.elev = np.array(elev)
        self.layer0 = 0
        self.layer1 = self.mg.nlay + self.ncb + 1

        zpts = []
        for k in range(self.layer0, self.layer1):
            zpts.append(
                plotutil.cell_value_points(
                    self.xpts,
                    self.mg.xyedges[0],
                    self.mg.xyedges[1],
                    self.elev[k, :, :],
                )
            )
        self.zpts = np.array(zpts)

        xcentergrid, zcentergrid = self.get_centergrids(self.xpts, self.zpts)
        self.xcentergrid = xcentergrid
        self.zcentergrid = zcentergrid

        geo_xcentergrid, _ = self.get_centergrids(
            self.geographic_xpts, self.zpts
        )
        self.geographic_xcentergrid = geo_xcentergrid

        # Create cross-section extent
        if extent is None:
            self.extent = self.get_extent()
        else:
            self.extent = extent

        # Set axis limits
        self.ax.set_xlim(self.extent[0], self.extent[1])
        self.ax.set_ylim(self.extent[2], self.extent[3])

        return

    @property
    def geographic_xpts(self):
        """
        Method to retranslate model coordinates to geometric
        coordinates for plotting

        Returns:

        """
        if self.__geographic_xpts is None:
            xypts = self.xpts.T
            xypts = geometry.transform(
                xypts[0],
                xypts[1],
                self.mg.xoffset,
                self.mg.yoffset,
                self.mg.angrot_radians,
            )

            if self.direction == "xy":
                xdist = np.max(xypts[0]) - np.min(xypts[0])
                ydist = np.max(xypts[1]) - np.min(xypts[1])
                if xdist >= ydist:
                    xypts = np.append(xypts, np.array([xypts[0]]), axis=0)
                else:
                    xypts = np.append(xypts, np.array([xypts[1]]), axis=0)
            else:
                xypts = np.append(xypts, np.array([xypts[0]]), axis=0)

            self.__geographic_xpts = xypts.T

        return self.__geographic_xpts

    def get_centergrids(self, xpts, zpts):
        """
        Method to calculate the centergrid information for plotting

        Parameters
        ----------
        xpts : np.ndarray
            array of x, y, distance along the cross section
        zpts : np.ndarray
            array of elevation values along the cross section

        Returns
        -------
            tuple : (xcentergrid, zcentergrid)
        """
        xcentergrid = []
        zcentergrid = []
        nz = 0
        if self.mg.nlay == 1:
            for k in range(0, zpts.shape[0]):
                nz += 1
                nx = 0
                for i in range(0, xpts.shape[0], 2):
                    try:
                        xp = 0.5 * (xpts[i][2] + xpts[i + 1][2])
                        zp = zpts[k, i]
                        xcentergrid.append(xp)
                        zcentergrid.append(zp)
                        nx += 1
                    except:
                        break
        else:
            for k in range(0, zpts.shape[0] - 1):
                if not self.active[k]:
                    continue
                nz += 1
                nx = 0
                for i in range(0, xpts.shape[0], 2):
                    try:
                        xp = 0.5 * (xpts[i][2] + xpts[i + 1][2])
                        zp = 0.5 * (zpts[k, i] + zpts[k + 1, i + 1])
                        xcentergrid.append(xp)
                        zcentergrid.append(zp)
                        nx += 1
                    except:
                        break

        xcentergrid = np.array(xcentergrid).reshape((nz, nx))
        zcentergrid = np.array(zcentergrid).reshape((nz, nx))
        return xcentergrid, zcentergrid

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

        xedge, yedge = self.mg.xyedges
        vpts = []
        for k in range(self.mg.nlay):
            vpts.append(
                plotutil.cell_value_points(self.xpts, xedge, yedge, a[k, :, :])
            )
            if len(self.laycbd) > 0:
                if self.laycbd[k] > 0:
                    ta = np.empty((self.mg.nrow, self.mg.ncol), dtype=float)
                    ta[:, :] = -1e9
                    vpts.append(
                        plotutil.cell_value_points(self.xpts, xedge, yedge, ta)
                    )
        vpts = np.array(vpts)
        if masked_values is not None:
            for mval in masked_values:
                vpts = np.ma.masked_values(vpts, mval)

        if isinstance(head, np.ndarray):
            zpts = self.set_zpts(head)
        else:
            zpts = self.zpts

        if self.ncb > 0:
            vpts = np.ma.masked_values(vpts, -1e9)

        pc = self.get_grid_patch_collection(zpts, vpts, **kwargs)
        if pc != None:
            ax.add_collection(pc)
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

        plotarray = a

        vpts = []
        if len(plotarray.shape) == 2:
            nlay = 1
            plotarray = np.reshape(
                plotarray, (1, plotarray.shape[0], plotarray.shape[1])
            )
        elif len(plotarray.shape) == 3:
            nlay = plotarray.shape[0]
        else:
            raise Exception("plot_array array must be a 2D or 3D array")

        xedge, yedge = self.mg.xyedges
        for k in range(nlay):
            vpts.append(
                plotutil.cell_value_points(
                    self.xpts, xedge, yedge, plotarray[k, :, :]
                )
            )
        vpts = np.array(vpts)

        if masked_values is not None:
            for mval in masked_values:
                vpts = np.ma.masked_values(vpts, mval)

        plot = []
        # adust distance array for modelgrid offset
        if self.geographic_coords:
            d = self.geographic_xpts.T[-1]
        else:
            d = self.d
        for k in range(vpts.shape[0]):
            plot.append(ax.plot(d, vpts[k, :], **kwargs))

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
        colors : list
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

        plotarray = a

        vpts = []
        for k in range(self.mg.nlay):
            # print('k', k, self.laycbd[k])
            vpts.append(
                plotutil.cell_value_points(
                    self.xpts,
                    self.mg.xyedges[0],
                    self.mg.xyedges[1],
                    plotarray[k, :, :],
                )
            )
            if len(self.laycbd) > 0:
                if self.laycbd[k] > 0:
                    ta = np.empty((self.mg.nrow, self.mg.ncol), dtype=float)
                    ta[:, :] = self.mg.botm.array[k, :, :]
                    vpts.append(
                        plotutil.cell_value_points(
                            self.xpts,
                            self.mg.xyedges[0],
                            self.mg.xyedges[1],
                            ta,
                        )
                    )

        vpts = np.ma.array(vpts, mask=False)

        if isinstance(head, np.ndarray):
            zpts = self.set_zpts(head)
        else:
            zpts = self.zpts

        if masked_values is not None:
            for mval in masked_values:
                vpts = np.ma.masked_values(vpts, mval)
        if self.ncb > 0:
            vpts = np.ma.masked_values(vpts, -1e9)
        idxm = np.ma.getmask(vpts)

        plot = []
        # print(zpts.shape)
        for k in range(self.mg.nlay + self.ncb):
            if self.active[k] == 0:
                continue
            idxmk = idxm[k, :]
            v = vpts[k, :]
            y1 = zpts[k, :]
            y2 = zpts[k + 1, :]
            # make sure y1 is not below y2
            idx = y1 < y2
            y1[idx] = y2[idx]
            # make sure v is not below y2
            idx = v < y2
            v[idx] = y2[idx]
            # make sure v is not above y1
            idx = v > y1
            v[idx] = y1[idx]
            # set y2 to v
            y2 = v
            # mask cells
            y1[idxmk] = np.nan
            y2[idxmk] = np.nan
            # adjust distance array for modelgrid offset
            if self.geographic_coords:
                d = self.geographic_xpts.T[-1]
            else:
                d = self.d
            plot.append(
                ax.fill_between(d, y1=y1, y2=y2, color=colors[0], **kwargs)
            )
            y1 = y2
            y2 = self.zpts[k + 1, :]
            y2[idxmk] = np.nan
            plot.append(
                ax.fill_between(d, y1=y1, y2=y2, color=colors[1], **kwargs)
            )
        return plot

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
        plotarray = a

        vpts = []
        xedge, yedge = self.mg.xyedges
        for k in range(self.mg.nlay):
            vpts.append(
                plotutil.cell_value_points(
                    self.xpts, xedge, yedge, plotarray[k, :, :]
                )
            )
        vpts = np.array(vpts)
        vpts = vpts[:, ::2]
        if self.mg.nlay == 1:
            vpts = np.vstack((vpts, vpts))

        if masked_values is not None:
            for mval in masked_values:
                vpts = np.ma.masked_values(vpts, mval)

        if isinstance(head, np.ndarray):
            zcentergrid = self.set_zcentergrid(head)
        else:
            zcentergrid = self.zcentergrid

        if self.geographic_coords:
            xcentergrid = self.geographic_xcentergrid
        else:
            xcentergrid = self.xcentergrid
        contour_set = self.ax.contour(xcentergrid, zcentergrid, vpts, **kwargs)
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
            "Function must be called in PlotCrossSection"
        )

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
        if plt is None:
            err_msg = (
                "matplotlib must be installed to "
                + "use get_grid_patch_collection()"
            )
            raise ImportError(err_msg)
        else:
            from matplotlib.patches import Polygon
            from matplotlib.collections import PatchCollection
        rectcol = []

        if "vmin" in kwargs:
            vmin = kwargs.pop("vmin")
        else:
            vmin = None
        if "vmax" in kwargs:
            vmax = kwargs.pop("vmax")
        else:
            vmax = None

        colors = []
        if self.geographic_coords:
            xpts = self.geographic_xpts
        else:
            xpts = self.xpts
        for k in range(zpts.shape[0] - 1):
            for idx in range(0, len(xpts) - 1, 2):
                try:
                    ll = (xpts[idx][2], zpts[k + 1, idx])
                    try:
                        dx = xpts[idx + 2][2] - xpts[idx][2]
                    except:
                        dx = xpts[idx + 1][2] - xpts[idx][2]
                    dz = zpts[k, idx] - zpts[k + 1, idx]
                    pts = (
                        ll,
                        (ll[0], ll[1] + dz),
                        (ll[0] + dx, ll[1] + dz),
                        (ll[0] + dx, ll[1]),
                    )  # , ll)
                    if np.isnan(plotarray[k, idx]):
                        continue
                    if plotarray[k, idx] is np.ma.masked:
                        continue
                    rectcol.append(Polygon(pts, closed=True))
                    colors.append(plotarray[k, idx])
                except:
                    pass

        if len(rectcol) > 0:
            patches = PatchCollection(rectcol, **kwargs)
            patches.set_array(np.array(colors))
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
            from matplotlib.collections import LineCollection

        color = "grey"
        if "color" in kwargs:
            color = kwargs.pop("color")

        linecol = []
        if self.geographic_coords:
            xpts = self.geographic_xpts
        else:
            xpts = self.xpts
        for k in range(self.zpts.shape[0] - 1):
            for idx in range(0, len(xpts) - 1, 2):
                try:
                    ll = (xpts[idx][2], self.zpts[k + 1, idx])
                    try:
                        dx = xpts[idx + 2][2] - xpts[idx][2]
                    except (IndexError, ValueError):
                        dx = xpts[idx + 1][2] - xpts[idx][2]
                    dz = self.zpts[k, idx] - self.zpts[k + 1, idx]
                    # horizontal lines
                    linecol.append(((ll), (ll[0] + dx, ll[1])))
                    linecol.append(
                        ((ll[0], ll[1] + dz), (ll[0] + dx, ll[1] + dz))
                    )
                    # vertical lines
                    linecol.append(((ll), (ll[0], ll[1] + dz)))
                    linecol.append(
                        ((ll[0] + dx, ll[1]), (ll[0] + dx, ll[1] + dz))
                    )
                except (IndexError, AttributeError, ValueError):
                    pass

        linecollection = LineCollection(linecol, color=color, **kwargs)
        return linecollection

    def set_zpts(self, vs):
        """
        Get an array of z elevations based on minimum of cell elevation
        (self.elev) or passed vs numpy.ndarray

        Parameters
        ----------
        vs : numpy.ndarray
            Three-dimensional array to plot.

        Returns
        -------
        zpts : numpy.ndarray

        """
        zpts = []
        xedge, yedge = self.mg.xyedges
        for k in range(self.layer0, self.layer1):
            e = self.elev[k, :, :]
            if k < self.mg.nlay:
                v = vs[k, :, :]
                idx = v < e
                e[idx] = v[idx]
            zpts.append(plotutil.cell_value_points(self.xpts, xedge, yedge, e))
        return np.array(zpts)

    def set_zcentergrid(self, vs):
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
        vpts = []
        xedge, yedge = self.mg.xyedges
        for k in range(self.layer0, self.layer1):
            if k < self.mg.nlay:
                e = vs[k, :, :]
            else:
                e = self.elev[k, :, :]
            vpts.append(plotutil.cell_value_points(self.xpts, xedge, yedge, e))
        vpts = np.array(vpts)

        zcentergrid = []
        nz = 0
        if self.mg.nlay == 1:
            for k in range(0, self.zpts.shape[0]):
                nz += 1
                nx = 0
                for i in range(0, self.xpts.shape[0], 2):
                    nx += 1
                    vp = vpts[k, i]
                    zp = self.zpts[k, i]
                    if k == 0:
                        if vp < zp:
                            zp = vp
                    zcentergrid.append(zp)
        else:
            for k in range(0, self.zpts.shape[0] - 1):
                if not self.active[k] == 1:
                    continue
                nz += 1
                nx = 0
                for i in range(0, self.xpts.shape[0], 2):
                    nx += 1
                    vp = vpts[k, i]
                    ep = self.zpts[k, i]
                    if vp < ep:
                        ep = vp
                    zp = 0.5 * (ep + self.zpts[k + 1, i + 1])
                    zcentergrid.append(zp)
        return np.array(zcentergrid).reshape((nz, nx))

    def get_extent(self):
        """
        Get the extent of the rotated and offset grid

        Returns
        -------
        tuple : (xmin, xmax, ymin, ymax)

        """
        if self.geographic_coords:
            xpts = self.geographic_xpts
        else:
            xpts = self.xpts

        xmin = xpts[0][2]
        xmax = xpts[-1][2]

        ymin = self.zpts.min()
        ymax = self.zpts.max()

        return (xmin, xmax, ymin, ymax)


class ModelCrossSection(object):
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

    def __new__(
        cls,
        ax=None,
        model=None,
        dis=None,
        line=None,
        xul=None,
        yul=None,
        rotation=None,
        extent=None,
    ):

        from flopy.plot.plotbase import DeprecatedCrossSection
        from flopy.discretization import StructuredGrid

        err_msg = (
            "ModelCrossSection will be replaced by "
            + "PlotCrossSection(), Calling PlotCrossSection()"
        )
        warnings.warn(err_msg, PendingDeprecationWarning)

        modelgrid = None
        if model is not None:
            if (xul, yul, rotation) != (None, None, None):
                modelgrid = plotutil._set_coord_info(
                    model.modelgrid, xul, yul, None, None, rotation
                )

        elif dis is not None:
            modelgrid = StructuredGrid(
                delr=dis.delr.array,
                delc=dis.delc.array,
                top=dis.top.array,
                botm=dis.botm.array,
            )

        if (xul, yul, rotation) != (None, None, None):
            modelgrid = plotutil._set_coord_info(
                modelgrid, xul, yul, None, None, rotation
            )

        return DeprecatedCrossSection(
            ax=ax, model=model, modelgrid=modelgrid, line=line, extent=extent
        )
