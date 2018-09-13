import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors
except:
    plt = None
from flopy.plot import plotutil
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class VertexCrossSection(object):
    """
    Class to create a cross section of the model from a VertexGrid

    Parameters
    ----------

    """
    def __init__(self, ax=None, model=None, dis=None, modelgrid=None,
                 line=None, xul=None, yul=None, xll=None, yll=None,
                 rotation=0., extent=None, length_multiplier=1.):
        if plt is None:
            s = 'Could not import matplotlib.  Must install matplotlib ' + \
                ' in order to use ModelCrossSection method'
            raise ImportError(s)

        self.model = model

        if model is not None:
            self.mg = model.modelgrid
            self.sr = model.modelgrid.sr
            self.dis = model.get_package("DIS")

        elif modelgrid is not None:
            self.mg = modelgrid
            self.sr = modelgrid.sr
            self.dis = dis
            if dis is None:
                raise AssertionError("Cannot find model discretization package")

        elif dis is not None:
            self.mg = dis.parent.modelgrid
            self.sr = dis.parent.modelgrid.sr
            self.dis = dis

        else:
            raise Exception("Cannot find model discretization package")

        # Set origin and rotation,
        if any(elem is not None for elem in (xul, yul, xll, yll)) or \
               rotation != 0 or length_multiplier != 1.:
            self.sr.length_multiplier = length_multiplier
            self.sr.set_spatialreference(delc=self.mg.delc,
                                         xul=xul, yul=yul,
                                         xll=xll, yll=yll,
                                         rotation=rotation)
            self.mg.sr = self.sr

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

        # todo: consider giving the user a flag for modelgrid based coords.
        xp, yp = self.sr.transform(xp, yp, inverse=True)
        pts = [(xt, yt) for xt, yt in zip(xp, yp)]
        self.pts = np.array(pts)

        # get points along the line

        self.xypts = plotutil.UnstructuredPlotUtilities.\
            line_intersect_grid(self.pts,
                                self.mg.xgrid,
                                self.mg.ygrid)

        if len(self.xypts) < 2:
            s = 'cross-section cannot be created\n.'
            s += '   less than 2 points intersect the model grid\n'
            s += '   {} points intersect the grid.'.format(len(self.xypts))
            raise Exception(s)

        top = self.dis.top.array
        top.shape = (1, -1)
        botm = self.dis.botm.array
        nlay = len(botm)
        ncpl = self.mg.ncpl
        elev = list(top.copy())
        for k in range(nlay):
            elev.append(botm[k, :])

        self.elev = np.array(elev)

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
            projpts =  self.set_zpts(head)
        else:
            projpts = self.projpts

        pc = self.get_patch_collection(projpts, a, **kwargs)
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
        elif 'c' in  kwargs:
            color = kwargs.pop('c')
        else:
            color = 'b'

        if not isinstance(a, np.ndarray):
            a = np.array(a)

        if a.ndim > 1:
            a = np.ravel(a)

        if a.size % self.mg.ncpl != 0:
            raise AssertionError("Array size must be a multiple of ncpl")

        nlay = a.size / self.mg.ncpl

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
            projpts =  self.set_zpts(head)
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

        if masked_values is not None:
            for mval in masked_values:
                a = np.ma.masked_equal(a, mval)

        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax

        plotarray = np.array([a[cell] for cell
                              in sorted(self.projpts)])

        xcenters = [np.mean(np.array(v).T[0]) for i, v
                    in sorted(self.projpts.items())]

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
            mask = [False for i in range(triang.triangles.shape[0])]
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
            ibound array to plot.  (Default is ibound in 'BAS6' package.)

        color_noflow : string
            (Default is 'black')

        Returns
        -------
        quadmesh : matplotlib.collections.PatchCollection

        """
        if ibound is None:
            if self.mg.idomain is not None:
                ibound = self.mg.idomain

            else:
                raise AssertionError("idomain must be provided to use plot_inactive")

        plotarray = np.zeros(ibound.shape, dtype=np.int)
        idx1 = (ibound == 0)
        plotarray[idx1] = 1
        plotarray = np.ma.masked_equal(plotarray, 0)
        cmap = matplotlib.colors.ListedColormap(['0', color_noflow])
        bounds = [0, 1, 2]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        patches = self.plot_array(plotarray, cmap=cmap, norm=norm, **kwargs)

        return patches

    def plot_ibound(self, ibound=None, color_noflow='black', color_ch='blue',
                    color_vpt='red', head=None, **kwargs):
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
            if self.mg.idomain is not None:
                ibound = self.mg.idomain

            else:
                raise AssertionError("idomain must be supplied to use plot_ibound")

        plotarray = np.zeros(ibound.shape, dtype=np.int)
        idx1 = (ibound == 0)
        idx2 = (ibound < 0)
        plotarray[idx1] = 1
        plotarray[idx2] = 2
        plotarray = np.ma.masked_equal(plotarray, 0)
        cmap = matplotlib.colors.ListedColormap(['none', color_noflow,
                                                 color_vpt])
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

        pc = self.get_grid_line_collection(**kwargs)
        if pc is not None:
            ax.add_collection(pc)
            ax.set_xlim(self.extent[0], self.extent[1])
            ax.set_ylim(self.extent[2], self.extent[3])

        return pc

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
        arr_dict = p.stress_period_data.to_array(kper)
        if not arr_dict:
            return None

        for key in arr_dict:
            fluxes = arr_dict[key]
            break

        plotarray = np.zeros((self.mg.nlay, self.mg.ncpl))
        plotarray[fluxes != 0] = 1
        plotarray = np.ma.masked_equal(plotarray, 0)

        if color is None:
            if ftype in plotutil.bc_color_dict:
                c = plotutil.bc_color_dict[ftype]
            else:
                c = plotutil.bc_color_dict['default']
        else:
            c = color
        cmap = matplotlib.colors.ListedColormap(['none', c])
        bounds = [0, 1, 2]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

        patches = self.plot_array(plotarray, masked_values=[0],
                                  head=head, cmap=cmap, norm=norm, **kwargs)
        return patches

    def plot_discharge(self, fja=None, head=None, dis=None,
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

        delr = np.tile([np.max(i) - np.min(i) for i in self.mg.ygrid], (nlay, 1))
        delc = np.tile([np.max(i) - np.min(i) for i in self.mg.xgrid], (nlay, 1))

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
            if self.model.sto is not None:
                laytyp = self.model.sto.iconvert.array

        sat_thk = plotutil.PlotUtilities. \
            saturated_thickness(head, top,
                                botm, laytyp,
                                mask_values=[hnoflo, hdry])

        frf, fff, flf = plotutil.UnstructuredPlotUtilities. \
            vectorize_flow(fja, model_grid=self.mg,
                           idomain=self.mg.idomain)

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
        v = np.array([qz[cell] for cell
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

    def get_patch_collection(self, projpts, plotarray, **kwargs):
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


if __name__ == "__main__":
    import os
    import flopy as fp
    from flopy.plot.plotbase import PlotCrossSection
    import flopy.utils.binaryfile as bf
    from flopy.proposed_grid.proposed_vertex_mg import VertexModelGrid

    ws = "../../examples/data/mf6/triangles"
    name = "mfsim.nam"

    sim = fp.mf6.modflow.MFSimulation.load(sim_name=name, sim_ws=ws)

    print(sim.model_names)
    ml = sim.get_model("gwf_1")

    dis = ml.dis
    sto = ml.sto
    chd = ml.get_package("chd_left")

    t = VertexModelGrid(dis.vertices, dis.cell2d,
                        top=dis.top, botm=dis.botm,
                        idomain=dis.idomain, xoffset=10,
                        yoffset=0, rotation=-25)


    # todo: build out model grid methods!
    x = t.xgrid
    y = t.ygrid
    t0 = t.top
    t1 = t.botm
    z = t.zgrid
    xc = t.xcenters
    yc = t.ycenters
    zc = t.zcenters
    lc = t.grid_lines
    e = t.extent

    sr_x = t.sr.xgrid
    sr_y = t.sr.ygrid
    sr_xc = t.sr.xcenters
    sr_yc = t.sr.ycenters
    sr_lc = t.sr.grid_lines
    sr_e = t.sr.extent

    # line = np.array([(0,2.5), (5, 2.5), (10, 2.5)])
    line = np.array([(2.5, 0), (2.5, 10.01)])
    line = t.sr.transform(line.T[0], line.T[1])
    line = np.array(line).T

    cr = PlotCrossSection(modelgrid=t, dis=dis,
                          line={"line": line})

    #ax = cr.plot_grid()
    #ax = cr.plot_array(a=dis.botm.array, alpha=0.5)
    #plt.colorbar(ax)
    #ax = cr.plot_bc(package=chd)
    #plt.show()

    # ax = cr.plot_bc(package=chd)
    # plt.show()

    #idx = [np.random.randint(0, 399) for _ in range(100)]
    #idx2 = [np.random.randint(0, 399) for _ in range(200)]
    #idomain = np.zeros((400,), dtype=int)

    #idomain[idx] = 1
    #idomain[idx2] = -1

    #ax = cr.plot_ibound(ibound=idomain)
    #plt.show()


    # ax = cr.plot_grid()
    # plt.show()
    # ax = cr.plot_array(a=dis.botm.array)
    # plt.show()

    # arr = np.random.rand(400) * 100
    # ax = cr.contour_array(a=arr)
    # plt.show()

    # arr = np.random.rand(400)
    # plot = cr.plot_surface(arr)

    # for i in plot:
    #    plt.show()

    #idomain = np.ones(100, dtype=np.int)
    #r = np.random.randint(0, 100, size=25)
    #r1 = np.random.randint(0, 100, size=10)
    #idomain[r] = 0
    #idomain[r1] = -1
    #ax = map.plot_ibound(idomain)
    #plt.show()

    #ax = map.plot_grid()
    #plt.show()

    #chd = ml.get_package("CHD")
    #ax = map.plot_bc(package=chd)
    #plt.show()
    # todo: flip z-vector for flow!

    #cbc = os.path.join(ws, "expected_output/", "model_unch.cbc")
    #hds = os.path.join(ws, "expected_output/", "model_unch.hds")

    cbc = os.path.join(ws, "tri_model.cbc")
    hds = os.path.join(ws, "tri_model.hds")

    cbc = bf.CellBudgetFile(cbc, precision="double")
    hds = bf.HeadFile(hds)

    #print(cbc.get_unique_record_names())

    fja = cbc.get_data(text="FLOW-JA-FACE")
    #fja = cbc.get_data(text="FLOW JA FACE")
    head = hds.get_alldata()[0]
    head.shape = (4, -1)
    print(head.ndim)


    ax = cr.plot_discharge(fja=fja, head=head, kstep=2)
    plt.show()
    #print('break')
