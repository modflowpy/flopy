import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors
except:
    plt = None
from flopy.plot import plotutil
from flopy.utils import geometry
import warnings
import copy
warnings.simplefilter('always', PendingDeprecationWarning)


class CrossSection(object):
    """
    Base class for CrossSection plotting. Handles the model grid
    transforms and searching for modelgrid and dis file information.

    This class must be general with absolutely no code specific to
    a single model grid type as that would break the CrossSection plotting
    ability of one or more child classes.
    """
    def __init__(self, ax=None, model=None, modelgrid=None):

        self.ax = ax

        if plt is None:
            s = 'Could not import matplotlib.  Must install matplotlib ' + \
                ' in order to use ModelCrossSection method'
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

        if type(None) in (type(self.mg.top), type(self.mg.botm)):
            raise AssertionError("modelgrid top and botm must be defined")


class StructuredCrossSection(CrossSection):
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

    def __init__(self, ax=None, model=None, modelgrid=None,
                 line=None, extent=None):
        super(StructuredCrossSection, self).__init__(ax=ax, model=model,
                                                     modelgrid=modelgrid)

        if line is None:
            s = 'line must be specified.'
            raise Exception(s)

        linekeys = [linekeys.lower() for linekeys in list(line.keys())]

        if len(linekeys) != 1:
            s = 'only row, column, or line can be specified in line dictionary.\n'
            s += 'keys specified: '
            for k in linekeys:
                s += '{} '.format(k)
            raise AssertionError(s)

        if ax is None:
            self.ax = plt.gca()
        else:
            self.ax = ax

        onkey = list(line.keys())[0]
        eps = 1.e-4
        xedge, yedge = self.mg.xyedges

        # un-translate model grid into model coordinates
        self.xcellcenters, self.ycellcenters = \
            geometry.transform(self.mg.xcellcenters,
                               self.mg.ycellcenters,
                               self.mg.xoffset, self.mg.yoffset,
                               self.mg.angrot_radians, inverse=True)

        if 'row' in linekeys:
            self.direction = 'x'
            ycenter = self.ycellcenters.T[0]
            pts = [(xedge[0] + eps,
                    ycenter[int(line[onkey])] - eps),
                   (xedge[-1] - eps,
                    ycenter[int(line[onkey])] + eps)]
        elif 'column' in linekeys:
            self.direction = 'y'
            xcenter = self.xcellcenters[0, :]
            pts = [(xcenter[int(line[onkey])] + eps,
                    yedge[0] - eps),
                   (xcenter[int(line[onkey])] - eps,
                    yedge[-1] + eps)]
        else:
            self.direction = 'xy'
            verts = line[onkey]
            xp = []
            yp = []
            for [v1, v2] in verts:
                xp.append(v1)
                yp.append(v2)

            xp, yp = self.mg.get_local_coords(xp, yp)
            pts = [(xt, yt) for xt, yt in zip(xp, yp)]

        # convert pts list to numpy array
        self.pts = np.array(pts)

        # get points along the line
        self.xpts = plotutil.line_intersect_grid(self.pts, self.mg.xyedges[0],
                                                 self.mg.xyedges[1])
        if len(self.xpts) < 2:
            s = 'cross-section cannot be created\n.'
            s += '   less than 2 points intersect the model grid\n'
            s += '   {} points intersect the grid.'.format(len(self.xpts))
            raise Exception(s)

            # set horizontal distance
        d = []
        for v in self.xpts:
            d.append(v[2])
        self.d = np.array(d)

        self.idomain = self.mg.idomain
        if self.mg.idomain is None:
            self.idomain = np.ones((self.mg.nlay, self.mg.nrow,
                                    self.mg.ncol), dtype=int)

        self.ncb = 0
        self.laycbd = []

        if self.model is not None:
            if self.model.laycbd is not None:
                self.laycbd = self.model.laycbd

        for l in self.laycbd:
            if l > 0:
                self.ncb += 1
        self.active = np.ones((self.mg.nlay + self.ncb), dtype=np.int)
        kon = 0
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
            zpts.append(plotutil.cell_value_points(self.xpts, self.mg.xyedges[0],
                                                   self.mg.xyedges[1],
                                                   self.elev[k, :, :]))
        self.zpts = np.array(zpts)

        xcentergrid = []
        zcentergrid = []
        nz = 0
        if self.mg.nlay == 1:
            for k in range(0, self.zpts.shape[0]):
                nz += 1
                nx = 0
                for i in range(0, self.xpts.shape[0], 2):
                    try:
                        xp = 0.5 * (self.xpts[i][2] + self.xpts[i + 1][2])
                        zp = self.zpts[k, i]
                        xcentergrid.append(xp)
                        zcentergrid.append(zp)
                        nx += 1
                    except:
                        break
        else:
            for k in range(0, self.zpts.shape[0] - 1):
                nz += 1
                nx = 0
                for i in range(0, self.xpts.shape[0], 2):
                    try:
                        xp = 0.5 * (self.xpts[i][2] + self.xpts[i + 1][2])
                        zp = 0.5 * (self.zpts[k, i] + self.zpts[k + 1, i + 1])
                        xcentergrid.append(xp)
                        zcentergrid.append(zp)
                        nx += 1
                    except:
                        break
        self.xcentergrid = np.array(xcentergrid).reshape((nz, nx))
        self.zcentergrid = np.array(zcentergrid).reshape((nz, nx))

        # Create cross-section extent
        if extent is None:
            self.extent = self.get_extent()
        else:
            self.extent = extent

        # Set axis limits
        self.ax.set_xlim(self.extent[0], self.extent[1])
        self.ax.set_ylim(self.extent[2], self.extent[3])

        return

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
        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax

        xedge, yedge = self.mg.xyedges
        vpts = []
        for k in range(self.mg.nlay):
            vpts.append(plotutil.cell_value_points(self.xpts, xedge,
                                                   yedge, a[k, :, :]))
            if self.laycbd[k] > 0:
                ta = np.empty((self.mg.nrow, self.mg.ncol), dtype=np.float)
                ta[:, :] = -1e9
                vpts.append(plotutil.cell_value_points(self.xpts,
                                                       xedge, yedge, ta))
        vpts = np.array(vpts)
        if masked_values is not None:
            for mval in masked_values:
                vpts = np.ma.masked_equal(vpts, mval)

        if isinstance(head, np.ndarray):
            zpts = self.set_zpts(head)
        else:
            zpts = self.zpts

        if masked_values is not None:
            for mval in masked_values:
                vpts = np.ma.masked_equal(vpts, mval)

        if self.ncb > 0:
            vpts = np.ma.masked_equal(vpts, -1e9)

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
        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax

        plotarray = a

        vpts = []
        if len(plotarray.shape) == 2:
            nlay = 1
            plotarray = np.reshape(plotarray,
                                   (1, plotarray.shape[0], plotarray.shape[1]))
        elif len(plotarray.shape) == 3:
            nlay = plotarray.shape[0]
        else:
            raise Exception('plot_array array must be a 2D or 3D array')

        xedge, yedge = self.mg.xyedges
        for k in range(nlay):
            vpts.append(plotutil.cell_value_points(self.xpts, xedge,
                                                   yedge,
                                                   plotarray[k, :, :]))
        vpts = np.array(vpts)

        if masked_values is not None:
            for mval in masked_values:
                vpts = np.ma.masked_equal(vpts, mval)

        plot = []
        for k in range(vpts.shape[0]):
            plot.append(ax.plot(self.d, vpts[k, :], **kwargs))

        return plot

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
        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax

        plotarray = a

        vpts = []
        for k in range(self.mg.nlay):
            # print('k', k, self.laycbd[k])
            vpts.append(plotutil.cell_value_points(self.xpts, self.mg.xyedges[0],
                                                   self.mg.xyedges[1],
                                                   plotarray[k, :, :]))
            if self.laycbd[k] > 0:
                ta = np.empty((self.mg.nrow, self.mg.ncol), dtype=np.float)
                ta[:, :] = self.mg.botm.array[k, :, :]
                vpts.append(plotutil.cell_value_points(self.xpts,
                                                       self.mg.xyedges[0],
                                                       self.mg.xyedges[1], ta))

        vpts = np.ma.array(vpts, mask=False)

        if isinstance(head, np.ndarray):
            zpts = self.set_zpts(head)
        else:
            zpts = self.zpts

        if masked_values is not None:
            for mval in masked_values:
                vpts = np.ma.masked_equal(vpts, mval)
        if self.ncb > 0:
            vpts = np.ma.masked_equal(vpts, -1e9)
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
            plot.append(ax.fill_between(self.d, y1=y1, y2=y2,
                                        color=colors[0], **kwargs))
            y1 = y2
            y2 = self.zpts[k + 1, :]
            y2[idxmk] = np.nan
            plot.append(ax.fill_between(self.d, y1=y1, y2=y2,
                                        color=colors[1], **kwargs))
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
            vpts.append(plotutil.cell_value_points(self.xpts, xedge,
                                                   yedge,
                                                   plotarray[k, :, :]))
        vpts = np.array(vpts)
        vpts = vpts[:, ::2]
        if self.mg.nlay == 1:
            vpts = np.vstack((vpts, vpts))

        if masked_values is not None:
            for mval in masked_values:
                vpts = np.ma.masked_equal(vpts, mval)

        if isinstance(head, np.ndarray):
            zcentergrid = self.set_zcentergrid(head)
        else:
            zcentergrid = self.zcentergrid

        contour_set = self.ax.contour(self.xcentergrid, zcentergrid,
                                      vpts, **kwargs)
        return contour_set

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
        raise NotImplementedError("plot_inactive must be "
                                  "called from PlotCrossSection")

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
        color_vpt : str
            Color for vertical pass through cells (Default is 'red')
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
        raise NotImplementedError("plot_ibound must be "
                                  "called from PlotCrossSection")

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
        raise NotImplementedError("plot_grid must be "
                                  "called from PlotCrossSection")

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
            Three-dimensional array to set top of patches to the minimum
            of the top of a layer or the head value. Used to create
            patches that conform to water-level elevations.
        **kwargs : dictionary
            keyword arguments passed to matplotlib.collections.PatchCollection

        Returns
        -------
        patches : matplotlib.collections.PatchCollection

        """
        raise NotImplementedError("plot_bc must be "
                                  "called from PlotCrossSection")

    def plot_discharge(self, frf, fff, flf=None, head=None,
                       kstep=1, hstep=1, normalize=False,
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
        # remove 'pivot' keyword argument
        # by default the center of the arrow is plotted in the center of a cell
        if 'pivot' in kwargs:
            pivot = kwargs.pop('pivot')
        else:
            pivot = 'middle'

        # Calculate specific discharge
        ib = self.idomain

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

        sat_thk = plotutil.PlotUtilities.\
            saturated_thickness(head, top, botm,
                                laytyp, [hnoflo, hdry])

        # Calculate specific discharge
        qx, qy, qz = plotutil.PlotUtilities.\
            centered_specific_discharge(frf, fff, flf,
                                        delr, delc, sat_thk)

        if qz is None:
            qz = np.zeros((qx.shape), dtype=np.float)

        # Select correct specific discharge direction
        if self.direction == 'x':
            u = qx[:, :, :]
            u2 = -qy[:, :, :]
            v = qz[:, :, :]
        elif self.direction == 'y':
            u = -qy[:, :, :]
            u2 = -qx[:, :, :]
            v = qz[:, :, :]
        elif self.direction == 'xy':
            print('csplot_discharge does not support arbitrary cross-sections')
            return None

        if isinstance(head, np.ndarray):
            zcentergrid = self.set_zcentergrid(head)
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

        upts = []
        u2pts = []
        vpts = []
        ibpts = []
        xedge, yedge = self.mg.xyedges
        for k in range(self.mg.nlay):
            upts.append(plotutil.cell_value_points(self.xpts, xedge,
                                                   yedge, u[k, :, :]))
            u2pts.append(plotutil.cell_value_points(self.xpts, xedge,
                                                    yedge,
                                                    u2[k, :, :]))
            vpts.append(plotutil.cell_value_points(self.xpts, xedge,
                                                   yedge, v[k, :, :]))
            ibpts.append(plotutil.cell_value_points(self.xpts, xedge,
                                                    yedge,
                                                    ib[k, :, :]))
        # convert upts, u2pts, and vpts to numpy arrays
        upts = np.array(upts)
        u2pts = np.array(u2pts)
        vpts = np.array(vpts)
        ibpts = np.array(ibpts)

        # Select correct slice and apply step
        x = x[::kstep, ::hstep]
        z = z[::kstep, ::hstep]
        upts = upts[::kstep, ::hstep]
        u2pts = u2pts[::kstep, ::hstep]
        vpts = vpts[::kstep, ::hstep]
        ibpts = ibpts[::kstep, ::hstep]

        # normalize
        if normalize:
            if self.direction == 'xy':
                vmag = np.sqrt(upts ** 2. + u2pts ** 2. + vpts ** 2.)
            else:
                vmag = np.sqrt(upts ** 2. + vpts ** 2.)
            idx = vmag > 0.
            upts[idx] /= vmag[idx]
            u2pts[idx] /= vmag[idx]
            vpts[idx] /= vmag[idx]

        # upts and vpts has a value for the left and right
        # sides of a cell. Sample every other value for quiver
        upts = upts[:, ::2]
        vpts = vpts[:, ::2]
        ibpts = ibpts[:, ::2]

        # mask discharge in inactive cells
        idx = (ibpts == 0)
        upts[idx] = np.nan
        vpts[idx] = np.nan

        # plot the vectors
        quiver = self.ax.quiver(x, z, upts, vpts, pivot=pivot, **kwargs)

        return quiver

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
        from matplotlib.patches import Polygon
        from matplotlib.collections import PatchCollection
        rectcol = []

        if 'vmin' in kwargs:
            vmin = kwargs.pop('vmin')
        else:
            vmin = None
        if 'vmax' in kwargs:
            vmax = kwargs.pop('vmax')
        else:
            vmax = None

        colors = []
        for k in range(zpts.shape[0] - 1):
            for idx in range(0, len(self.xpts) - 1, 2):
                try:
                    ll = ((self.xpts[idx][2], zpts[k + 1, idx]))
                    try:
                        dx = self.xpts[idx + 2][2] - self.xpts[idx][2]
                    except:
                        dx = self.xpts[idx + 1][2] - self.xpts[idx][2]
                    dz = zpts[k, idx] - zpts[k + 1, idx]
                    pts = (ll,
                           (ll[0], ll[1] + dz), (ll[0] + dx, ll[1] + dz),
                           (ll[0] + dx, ll[1]))  # , ll)
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
        from matplotlib.collections import LineCollection

        color = "grey"
        if "color" in kwargs:
            color = kwargs.pop('color')

        linecol = []
        for k in range(self.zpts.shape[0] - 1):
            for idx in range(0, len(self.xpts) - 1, 2):
                try:
                    ll = ((self.xpts[idx][2], self.zpts[k + 1, idx]))
                    try:
                        dx = self.xpts[idx + 2][2] - self.xpts[idx][2]
                    except:
                        dx = self.xpts[idx + 1][2] - self.xpts[idx][2]
                    dz = self.zpts[k, idx] - self.zpts[k + 1, idx]
                    # horizontal lines
                    linecol.append(((ll), (ll[0] + dx, ll[1])))
                    linecol.append(
                        ((ll[0], ll[1] + dz), (ll[0] + dx, ll[1] + dz)))
                    # vertical lines
                    linecol.append(((ll), (ll[0], ll[1] + dz)))
                    linecol.append(
                        ((ll[0] + dx, ll[1]), (ll[0] + dx, ll[1] + dz)))
                except:
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
            zpts.append(plotutil.cell_value_points(self.xpts, xedge,
                                                   yedge, e))
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
            vpts.append(plotutil.cell_value_points(self.xpts, xedge,
                                                   yedge, e))
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

        Return (xmin, xmax, ymin, ymax)

        """

        xmin = self.xpts[0][2]
        xmax = self.xpts[-1][2]

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
    def __new__(cls, ax=None, model=None, dis=None, line=None,
                xul=None, yul=None, xll=None, yll=None,
                rotation=None, extent=None, length_multiplier=1.):

        from flopy.plot.plotbase import PlotCrossSection
        from flopy.discretization import StructuredGrid

        err_msg = "ModelCrossSection will be replaced by " +\
            "PlotCrossSection(), Calling PlotCrossSection()"
        warnings.warn(err_msg, PendingDeprecationWarning)

        modelgrid = None
        if model is not None:
            if (xul, yul, xll, yll, rotation) != (None, None, None, None, None):
                modelgrid = plotutil._set_coord_info(model.modelgrid,
                                                     xul, yul, xll, yll,
                                                     rotation)

        elif dis is not None:
            modelgrid = StructuredGrid(delr=dis.delr.array,
                                       delc=dis.delc.array,
                                       top=dis.top.array,
                                       botm=dis.botm.array)

        if (xul, yul, xll, yll, rotation) != (None, None, None, None, None):
            modelgrid = plotutil._set_coord_info(modelgrid,
                                          xul, yul, xll, yll,
                                          rotation)


        return PlotCrossSection(ax=ax, model=model,
                                modelgrid=modelgrid,
                                line=line, extent=extent)

