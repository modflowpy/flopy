import copy
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors
except:
    plt = None
from . import plotutil
from .plotutil import bc_color_dict


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

    def __init__(self, ax=None, model=None, dis=None, line=None,
                 xul=None, yul=None, rotation=None, extent=None):
        if plt is None:
            s = 'Could not import matplotlib.  Must install matplotlib ' + \
                ' in order to use ModelCrossSection method'
            raise Exception(s)

        self.model = model
        if dis is None:
            if model is None:
                raise Exception('Cannot find discretization package')
            else:
                self.dis = model.get_package('DIS')
                self.sr = copy.deepcopy(self.dis.parent.sr)
        else:
            self.dis = dis
            self.sr = copy.deepcopy(dis.parent.sr)
        if line == None:
            s = 'line must be specified.'
            raise Exception(s)

        linekeys = [linekeys.lower() for linekeys in list(line.keys())]

        if len(linekeys) < 1:
            s = 'only row, column, or line can be specified in line dictionary.\n'
            s += 'keys specified: '
            for k in linekeys:
                s += '{} '.format(k)
            raise Exception(s)

        if 'row' in linekeys and 'column' in linekeys:
            s = 'row and column cannot both be specified in line dictionary.'
            raise Exception(s)

        if 'row' in linekeys and 'line' in linekeys:
            s = 'row and line cannot both be specified in line dictionary.'
            raise Exception(s)

        if 'column' in linekeys and 'line' in linekeys:
            s = 'column and line cannot both be specified in line dictionary.'
            raise Exception(s)

        if ax is None:
            self.ax = plt.gca()
        else:
            self.ax = ax

        # Set origin and rotation
        if xul is not None:
            self.sr.xul = xul
        if yul is not None:
            self.sr.yul = yul
        if rotation is not None:
            self.sr.rotation = rotation

        onkey = list(line.keys())[0]
        eps = 1.e-4
        if 'row' in linekeys:
            self.direction = 'x'
            pts = [(self.sr.xedge[0] + eps,
                    self.sr.ycenter[int(line[onkey])] - eps),
                   (self.sr.xedge[-1] - eps,
                    self.sr.ycenter[int(line[onkey])] + eps)]
        elif 'column' in linekeys:
            self.direction = 'y'
            pts = [(self.sr.xcenter[int(line[onkey])] + eps,
                    self.sr.yedge[0] - eps),
                   (self.sr.xcenter[int(line[onkey])] - eps,
                    self.sr.yedge[-1] + eps)]
        else:
            self.direction = 'xy'
            verts = line[onkey]
            xp = []
            yp = []
            for [v1, v2] in verts:
                xp.append(v1)
                yp.append(v2)

            xp, yp = self.sr.transform(xp, yp, inverse=True)
            pts = [(xt, yt) for xt, yt in zip(xp, yp)]

        # convert pts list to numpy array
        self.pts = np.array(pts)

        # get points along the line
        self.xpts = plotutil.line_intersect_grid(self.pts, self.sr.xedge,
                                                 self.sr.yedge)
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

        self.ncb = 0
        self.laycbd = self.dis.laycbd.array
        for l in self.laycbd:
            if l > 0:
                self.ncb += 1
        self.active = np.ones((self.dis.nlay + self.ncb), dtype=np.int)
        kon = 0
        for k in range(self.dis.nlay):
            if self.laycbd[k] > 0:
                kon += 1
                self.active[kon] = 0
            kon += 1
        top = self.dis.top.array
        botm = self.dis.botm.array
        elev = [top.copy()]
        for k in range(self.dis.nlay + self.ncb):
            elev.append(botm[k, :, :])

        self.elev = np.array(elev)
        self.layer0 = 0
        self.layer1 = self.dis.nlay + self.ncb + 1

        zpts = []
        for k in range(self.layer0, self.layer1):
            zpts.append(plotutil.cell_value_points(self.xpts, self.sr.xedge,
                                                   self.sr.yedge,
                                                   self.elev[k, :, :]))
        self.zpts = np.array(zpts)

        xcentergrid = []
        zcentergrid = []
        nz = 0
        if self.dis.nlay == 1:
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

        vpts = []
        for k in range(self.dis.nlay):
            vpts.append(plotutil.cell_value_points(self.xpts, self.sr.xedge,
                                                   self.sr.yedge,
                                                   a[k, :, :]))
            if self.laycbd[k] > 0:
                ta = np.empty((self.dis.nrow, self.dis.ncol), dtype=np.float)
                ta[:, :] = -1e9
                vpts.append(plotutil.cell_value_points(self.xpts,
                                                       self.sr.xedge,
                                                       self.sr.yedge, ta))
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
        for k in range(nlay):
            vpts.append(plotutil.cell_value_points(self.xpts, self.sr.xedge,
                                                   self.sr.yedge,
                                                   plotarray[k, :, :]))
        vpts = np.array(vpts)

        if masked_values is not None:
            for mval in masked_values:
                vpts = np.ma.masked_equal(vpts, mval)

        plot = []
        for k in range(vpts.shape[0]):
            plot.append(ax.plot(self.d, vpts[k, :], **kwargs))
        return plot

    def plot_fill_between(self, a, colors=['blue', 'red'],
                          masked_values=None, head=None, **kwargs):
        """
        Plot a three-dimensional array as lines.

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
        for k in range(self.dis.nlay):
            # print('k', k, self.laycbd[k])
            vpts.append(plotutil.cell_value_points(self.xpts, self.sr.xedge,
                                                   self.sr.yedge,
                                                   plotarray[k, :, :]))
            if self.laycbd[k] > 0:
                ta = np.empty((self.dis.nrow, self.dis.ncol), dtype=np.float)
                ta[:, :] = self.dis.botm.array[k, :, :]
                vpts.append(plotutil.cell_value_points(self.xpts,
                                                       self.sr.xedge,
                                                       self.sr.yedge, ta))

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
        for k in range(self.dis.nlay + self.ncb):
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
        for k in range(self.dis.nlay):
            vpts.append(plotutil.cell_value_points(self.xpts, self.sr.xedge,
                                                   self.sr.yedge,
                                                   plotarray[k, :, :]))
        vpts = np.array(vpts)
        vpts = vpts[:, ::2]
        if self.dis.nlay == 1:
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

    def plot_ibound(self, ibound=None, color_noflow='black', color_ch='blue',
                    head=None, **kwargs):
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
        if ibound is None:
            bas = self.model.get_package('BAS6')
            ibound = bas.ibound.array
        plotarray = np.zeros(ibound.shape, dtype=np.int)
        idx1 = (ibound == 0)
        idx2 = (ibound < 0)
        plotarray[idx1] = 1
        plotarray[idx2] = 2
        plotarray = np.ma.masked_equal(plotarray, 0)
        cmap = matplotlib.colors.ListedColormap(['none', color_noflow,
                                                 color_ch])
        bounds = [0, 1, 2, 3]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        # mask active cells
        patches = self.plot_array(plotarray, masked_values=[0], head=head,
                                  cmap=cmap, norm=norm, **kwargs)
        return patches

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
        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax

        if 'color' not in kwargs:
            kwargs['color'] = '0.5'

        lc = self.get_grid_line_collection(**kwargs)
        ax.add_collection(lc)
        return lc

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

        # Get the list data
        try:
            mflist = p.stress_period_data[kper]
        except:
            raise Exception('Not a list-style boundary package')

        # Return if MfList is None
        if mflist is None:
            return None

        # Plot the list locations
        plotarray = np.zeros(self.dis.botm.shape, dtype=np.int)
        idx = (mflist['k'], mflist['i'], mflist['j'])
        plotarray[idx] = 1
        plotarray = np.ma.masked_equal(plotarray, 0)
        if color is None:
            if ftype in bc_color_dict:
                c = bc_color_dict[ftype]
            else:
                c = bc_color_dict['default']
        else:
            c = color
        cmap = matplotlib.colors.ListedColormap(['none', c])
        bounds = [0, 1, 2]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        patches = self.plot_array(plotarray, masked_values=[0],
                                  head=head, cmap=cmap, norm=norm, **kwargs)
        return patches

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
        ib = self.model.bas6.ibound.array
        delr = self.dis.delr.array
        delc = self.dis.delc.array
        top = self.dis.top.array
        botm = self.dis.botm.array
        nlay, nrow, ncol = botm.shape
        laytyp = None
        hnoflo = 999.
        hdry = 999.
        if self.model is not None:
            lpf = self.model.get_package('LPF')
            if lpf is not None:
                laytyp = lpf.laytyp.array
                hdry = lpf.hdry
            bas = self.model.get_package('BAS6')
            if bas is not None:
                hnoflo = bas.hnoflo

        # If no access to head or laytyp, then calculate confined saturated
        # thickness by setting laytyp to zeros
        if head is None or laytyp is None:
            head = np.zeros(botm.shape, np.float32)
            laytyp = np.zeros((nlay), dtype=np.int)
            head[0, :, :] = top
            if nlay > 1:
                head[1:, :, :] = botm[:-1, :, :]
        sat_thk = plotutil.saturated_thickness(head, top, botm, laytyp,
                                               [hnoflo, hdry])

        # Calculate specific discharge
        qx, qy, qz = plotutil.centered_specific_discharge(frf, fff, flf, delr,
                                                          delc, sat_thk)

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
        for k in range(self.dis.nlay):
            upts.append(plotutil.cell_value_points(self.xpts, self.sr.xedge,
                                                   self.sr.yedge, u[k, :, :]))
            u2pts.append(plotutil.cell_value_points(self.xpts, self.sr.xedge,
                                                    self.sr.yedge,
                                                    u2[k, :, :]))
            vpts.append(plotutil.cell_value_points(self.xpts, self.sr.xedge,
                                                   self.sr.yedge, v[k, :, :]))
            ibpts.append(plotutil.cell_value_points(self.xpts, self.sr.xedge,
                                                    self.sr.yedge,
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

        v = []

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

        linecollection = LineCollection(linecol, **kwargs)
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
        for k in range(self.layer0, self.layer1):
            e = self.elev[k, :, :]
            if k < self.dis.nlay:
                v = vs[k, :, :]
                idx = v < e
                e[idx] = v[idx]
            zpts.append(plotutil.cell_value_points(self.xpts, self.sr.xedge,
                                                   self.sr.yedge, e))
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
        for k in range(self.layer0, self.layer1):
            if k < self.dis.nlay:
                e = vs[k, :, :]
            else:
                e = self.elev[k, :, :]
            vpts.append(plotutil.cell_value_points(self.xpts, self.sr.xedge,
                                                   self.sr.yedge, e))
        vpts = np.array(vpts)

        zcentergrid = []
        nz = 0
        if self.dis.nlay == 1:
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
