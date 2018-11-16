import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors
except:
    plt = None
from flopy.plot import plotutil
from flopy.utils import geometry
from flopy.plot.crosssection import CrossSection
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class VertexCrossSection(CrossSection):
    """
    Class to create a cross section of the model from a VertexGrid

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

    """
    def __init__(self, ax=None, model=None, modelgrid=None,
                 line=None, extent=None):
        super(VertexCrossSection, self).__init__(ax=ax, model=model,
                                                 modelgrid=modelgrid)

        if line is None:
            err_msg = 'line must be specified.'
            raise Exception(err_msg)

        linekeys = [linekeys.lower() for linekeys in list(line.keys())]

        if len(linekeys) != 1:
            err_msg = 'Either row, column, or line must be specified ' \
                      'in line dictionary.\nkeys specified: '
            for k in linekeys:
                err_msg += '{} '.format(k)
            raise Exception(err_msg)

        elif "line" not in linekeys:
            err_msg = "only line can be specified in line dictionary " \
                      "for vertex Discretization"
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
        xp, yp = geometry.transform(xp, yp,
                                    self.mg.xoffset,
                                    self.mg.yoffset,
                                    self.mg.angrot_radians,
                                    inverse=True)

        self.xcellcenters, self.ycellcenters = \
            geometry.transform(self.mg.xcellcenters,
                               self.mg.ycellcenters,
                               self.mg.xoffset, self.mg.yoffset,
                               self.mg.angrot_radians, inverse=True)

        self.xvertices, self.yvertices = \
            geometry.transform(self.mg.xvertices,
                               self.mg.yvertices,
                               self.mg.xoffset, self.mg.yoffset,
                               self.mg.angrot_radians, inverse=True)

        pts = [(xt, yt) for xt, yt in zip(xp, yp)]
        self.pts = np.array(pts)

        # get points along the line

        self.xypts = plotutil.UnstructuredPlotUtilities.\
            line_intersect_grid(self.pts,
                                self.xvertices,
                                self.yvertices)

        if len(self.xypts) < 2:
            s = 'cross-section cannot be created\n.'
            s += '   less than 2 points intersect the model grid\n'
            s += '   {} points intersect the grid.'.format(len(self.xypts))
            raise Exception(s)

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
        self.projpts = {}
        for k in range(1, nlay + 1):
            top = self.elev[k - 1, :]
            botm = self.elev[k, :]
            adjnn = (k - 1) * ncpl
            for nn, verts in self.xypts.items():
                t = top[nn]
                b = botm[nn]
                if self.direction == "x":
                    projt = [(v[0], t) for v in verts]
                    projb = [(v[0], b) for v in verts]
                else:
                    projt = [(v[1], t) for v in verts]
                    projb = [(v[1], b) for v in verts]

                self.projpts[nn + adjnn] = projt + projb

                # Create cross-section extent
        if extent is None:
            self.extent = self.get_extent()
        else:
            self.extent = extent

        self.layer0 = None
        self.layer1 = None
        self.d = {i: (np.min(np.array(v).T[0]),
                      np.max(np.array(v).T[0])) for
                  i, v in sorted(self.projpts.items())}
        self.xpts = None
        self.active = None
        self.ncb = None
        self.laycbd = None
        self.zpts = None
        self.xcentergrid = None
        self.ycentergrid = None
        self.zcentergrid = None

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
        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax

        if not isinstance(a, np.ndarray):
            a = np.array(a)

        if a.ndim > 1:
            a = np.ravel(a)

        if masked_values is not None:
            for mval in masked_values:
                a = np.ma.masked_equal(a, mval)

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
        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax

        if 'color' in kwargs:
            color = kwargs.pop('color')
        elif 'c' in kwargs:
            color = kwargs.pop('c')
        else:
            color = 'b'

        if not isinstance(a, np.ndarray):
            a = np.array(a)

        if a.ndim > 1:
            a = np.ravel(a)

        if a.size % self.mg.ncpl != 0:
            raise AssertionError("Array size must be a multiple of ncpl")

        if masked_values is not None:
            for mval in masked_values:
                a = np.ma.masked_equal(a, mval)

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
            for ix, val in enumerate(data[k]):
                ax.plot(d[k, ix], data[k, ix], color=color, **kwargs)

            ax.set_xlim(self.extent[0], self.extent[1])
            ax.set_ylim(self.extent[2], self.extent[3])
            plot.append(ax)

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
        if "ax" in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax

        if not isinstance(a, np.ndarray):
            a = np.array(a)

        a = np.ravel(a)

        if masked_values is not None:
            for mval in masked_values:
                a = np.ma.masked_equal(a, mval)

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
        import matplotlib.tri as tri

        if not isinstance(a, np.ndarray):
            a = np.array(a)

        if a.ndim > 1:
            a = np.ravel(a)

        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax

        xcenters = [np.mean(np.array(v).T[0]) for i, v
                    in sorted(self.projpts.items())]

        plotarray = np.array([a[cell] for cell
                              in sorted(self.projpts)])

        if masked_values is not None:
            for mval in masked_values:
                plotarray = np.ma.masked_equal(plotarray, mval)

        if isinstance(head, np.ndarray):
            zcenters = self.set_zcentergrid(np.ravel(head))
        else:
            zcenters = [np.mean(np.array(v).T[1]) for i, v
                        in sorted(self.projpts.items())]

        plot_triplot = False
        if 'plot_triplot' in kwargs:
            plot_triplot = kwargs.pop('plot_triplot')

        if 'extent' in kwargs:
            extent = kwargs.pop('extent')

            idx = (xcenters >= extent[0]) & (
                    xcenters <= extent[1]) & (
                          zcenters >= extent[2]) & (
                          zcenters <= extent[3])
            plotarray = plotarray[idx].flatten()
            xcenters = xcenters[idx].flatten()
            zcenters = zcenters[idx].flatten()

        triang = tri.Triangulation(xcenters, zcenters)

        try:
            amask = plotarray.mask
            mask = [False for _ in range(triang.triangles.shape[0])]
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

    def plot_inactive(self, ibound=None, color_noflow='black', **kwargs):
        """
        Make a plot of inactive cells.  If not specified, then pull ibound
        from the self.ml

        Parameters
        ----------
        ibound : numpy.ndarray
            ibound array to plot.

        color_noflow : string
            (Default is 'black')

        Returns
        -------
        quadmesh : matplotlib.collections.PatchCollection

        """
        raise NotImplementedError("plot_inactive must be "
                                  "called from PlotCrossSection")


    def plot_ibound(self, ibound=None, color_noflow='black', color_ch='blue',
                    color_vpt='red', head=None, **kwargs):
        """
        Make a plot of ibound.  If not specified, then pull ibound from the
        self.model

        Parameters
        ----------
        ibound : numpy.ndarray
            ibound array to plot.
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

    def plot_discharge(self, fja=None, head=None,
                       kstep=1, hstep=1, normalize=False,
                       **kwargs):
        """
        Use quiver to plot vectors.

        Parameters
        ----------
        fja : numpy.ndarray
            MODFLOW's 'flow ja face'
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
        if 'pivot' in kwargs:
            pivot = kwargs.pop('pivot')
        else:
            pivot = 'middle'

        pts = self.pts

        # check that the cross section in not arbitrary within a tolerance...
        xuniform = [True if abs(pts.T[0, 0] - i) < 1
                    else False for i in pts.T[0]]
        yuniform = [True if abs(pts.T[1, 0] - i) < 1
                    else False for i in pts.T[1]]
        if not np.all(xuniform):
            if not np.all(yuniform):
                err_msg = "plot_discharge cannot plot " \
                          "aribtrary cross sections"
                raise AssertionError(err_msg)

        top = self.mg.top
        botm = self.mg.botm

        fja = np.array(fja)
        nlay = self.mg.nlay
        ncpl = self.mg.ncpl

        delr = np.tile([np.max(i) - np.min(i) for i in self.yvertices], (nlay, 1))
        delc = np.tile([np.max(i) - np.min(i) for i in self.xvertices], (nlay, 1))

        # no modflow6 equivalent???
        hnoflo = 999.
        hdry = 999.

        if head is None:
            head = np.zeros(botm.shape)

        if len(head.shape) == 3:
            head.shape = (nlay, -1)

        if len(fja.shape) == 4:
            fja = fja[0][0][0]

        # kstep implementation, check for bugs!
        projpts = {key: value for key, value in self.projpts.items()
                   if (key // ncpl) % kstep == 0}

        if isinstance(head, np.ndarray):
            # pipe kstep to set_zcentergrid to assure consistent array size
            zcenters = self.set_zcentergrid(np.ravel(head), kstep=kstep)
        else:
            zcenters = [np.mean(np.array(v).T[1]) for i, v
                        in sorted(projpts.items())]

        laytyp = np.zeros((nlay,))
        if self.model is not None:
            if self.model.laytyp is not None:
                laytyp = self.model.laytyp

        sat_thk = plotutil.PlotUtilities. \
            saturated_thickness(head, top,
                                botm, laytyp,
                                mask_values=[hnoflo, hdry])

        frf, fff, flf = plotutil.UnstructuredPlotUtilities. \
            vectorize_flow(fja, model_grid=self.mg,
                           idomain=self.idomain)

        qx, qy, qz = plotutil.UnstructuredPlotUtilities. \
            specific_discharge(frf, fff, flf,
                               delr, delc, sat_thk)

        # determine the projection direction
        if self.direction == "x":
            qx = np.ravel(qx)
            u = np.array([qx[cell] for cell
                          in sorted(projpts)])
            x = [np.mean(np.array(v).T[0]) for i, v
                 in sorted(projpts.items())]

        else:
            qy = np.ravel(qy)
            u = np.array([qy[cell] for cell
                          in sorted(projpts)])
            x = [np.mean(np.array(v).T[0]) for i, v
                 in sorted(projpts.items())]

        qz = np.ravel(qz)
        v = np.array([-qz[cell] for cell
                      in sorted(projpts)])
        y = np.ravel(zcenters)

        x = x[::hstep]
        y = y[::hstep]
        u = u[::hstep]
        v = v[::hstep]
        # normalize
        if normalize:
            vmag = np.sqrt(u ** 2. + v ** 2.)
            idx = vmag > 0.
            u[idx] /= vmag[idx]
            v[idx] /= vmag[idx]

        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax

        quiver = ax.quiver(x, y, u, v, scale=1, units='xy', pivot=pivot, **kwargs)

        return quiver

    def get_grid_patch_collection(self, projpts, plotarray, **kwargs):
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
        from matplotlib.patches import Polygon
        from matplotlib.collections import PatchCollection

        if 'vmin' in kwargs:
            vmin = kwargs.pop('vmin')
        else:
            vmin = None
        if 'vmax' in kwargs:
            vmax = kwargs.pop('vmax')
        else:
            vmax = None

        rectcol = []
        data = []
        for cell, verts in sorted(projpts.items()):
            verts = plotutil.UnstructuredPlotUtilities\
                .arctan2(np.array(verts))

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
        from matplotlib.patches import Polygon
        from matplotlib.collections import PatchCollection

        color = "grey"
        if 'ec' in kwargs:
            color = kwargs.pop('ec')
        if color in kwargs:
            color = kwargs.pop('color')

        rectcol = []
        for cell, verts in sorted(self.projpts.items()):
            verts = plotutil.UnstructuredPlotUtilities\
                .arctan2(np.array(verts))

            rectcol.append(Polygon(verts, closed=True))

        if len(rectcol) > 0:
            patches = PatchCollection(rectcol, edgecolor=color,
                                      facecolor='none', **kwargs)
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
        zpts : numpy.ndarray
        """
        # make vertex array based on projection direction
        if not isinstance(vs, np.ndarray):
            vs = np.array(vs)

        projpts = {}
        for k in range(1, self.mg.nlay + 1):
            top = self.elev[k - 1, :]
            botm = self.elev[k, :]
            adjnn = (k - 1) * self.mg.ncpl
            for nn, verts in sorted(self.xypts.items()):
                t = vs[nn]
                if top[nn] < vs[nn]:
                    t = top[nn]
                b = botm[nn]
                if self.direction == "x":
                    projt = [(v[0], t) for v in verts]
                    projb = [(v[0], b) for v in verts]
                else:
                    projt = [(v[1], t) for v in verts]
                    projb = [(v[1], b) for v in verts]

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
        zcenters =[np.mean(np.array(v).T[1]) for i, v
                   in sorted(verts.items())
                   if (i // self.mg.ncpl) % kstep == 0]
        return zcenters

    def get_extent(self):
        """
        Get the extent of the rotated and offset grid

        Return (xmin, xmax, ymin, ymax)
        """
        xpts = []
        if self.direction == "x":
            for nn, verts in self.xypts.items():
                for v in verts:
                    xpts.append(v[0])
        else:
            for nn, verts in self.xypts.items():
                for v in verts:
                    xpts.append(v[1])

        xmin = np.min(xpts)
        xmax = np.max(xpts)

        ymin = np.min(self.elev)
        ymax = np.max(self.elev)

        return (xmin, xmax, ymin, ymax)

