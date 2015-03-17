import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import plotutil
from plotutil import bc_color_dict, rotate


class ModelCrossSection(object):
    """
    Class to create a map of the model.

    Parameters
    ----------
    ax : matplotlib.pyplot axis
        The plot axis.  If not provided it, plt.gca() will be used.
    model : flopy.modflow object
        flopy model object. (Default is None)
    dis : flopy discretization object
    line : dict
        Dictionary with either "row", "column", or "line" key. If key
        is "row" or "column" key value should be the zero-based row or
        column index for cross-section. If key is "line" value should
        be an array of (x, y) tuples with vertices of cross-section. 
        Vertices should be in map coordinates consistent with xul,
        yul, and rotation.
    layer : int
        Layer to plot.  Default is None.  If layer is not None, it must 
        be between 0 and nlay - 1. If layer is None all layers will be
        included in the cross-section.
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
    """
    def __init__(self, ax=None, model=None, dis=None, line=None, layer=None,
                 xul=None, yul=None, rotation=0., extent=None):
        self.model = model
        self.layer = layer
        if dis is None:
            if model is None:
                raise Exception('Cannot find discretization package')
            else:
                self.dis = model.get_package('DIS')
        else:
            self.dis = dis
            
        if line == None:
            s = 'line must be specified.'
            raise Exception(s)
        
        linekeys = [linekeys.lower() for linekeys in line.keys()]
        
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

        if self.layer != None:
            if self.layer < 0 or self.layer > self.dis.nlay - 1:
                s = 'Not a valid layer: {}.  Must be between 0 and {}.'.format(
                    self.layer, self.dis.nlay - 1)
                raise Exception(s)

        if ax is None:
            self.ax = plt.gca()
        else:
            self.ax = ax

        # Set origin and rotation
        if xul is None:
            self.xul = 0.
        else:
            self.xul = xul
        if yul is None:
            self.yul = np.add.reduce(self.dis.delc.array)
        else:
            self.yul = yul
        self.rotation = -rotation * np.pi / 180.

        # Create edge arrays and meshgrid for pcolormesh
        self.xedge = self.get_xedge_array()
        self.yedge = self.get_yedge_array()

        # Create x and y center arrays and meshgrid of centers
        self.xcenter = self.get_xcenter_array()
        self.ycenter = self.get_ycenter_array()
                                                         
        onkey = line.keys()[0]                      
        if 'row' in linekeys:
            self.direction = 'x'
            pts = [(self.xedge[0], self.ycenter[int(line[onkey])]), 
                   (self.xedge[-1], self.ycenter[int(line[onkey])])]
        elif 'column' in linekeys:
            self.direction = 'y'
            pts = [(self.xcenter[int(line[onkey])], self.yedge[0]), 
                   (self.xcenter[int(line[onkey])], self.yedge[-1])]
        else:
            self.direction = 'xy'
            xp = np.array(line[onkey][0])
            yp = np.array(line[onkey][1])
            # remove offset and rotation from line
            xp -= self.xul
            yp -= self.yul
            xp, yp = rotate(xp, yp, -self.rotation, 0, self.yedge[0])
            pts = []
            for xt, yt in zip(xp, yp):
                pts.append((xt, yt))
        # convert pts list to numpy array
        self.pts = np.array(pts)
            
        # get points along the line
        self.xpts = plotutil.line_intersect_grid(self.pts, self.xedge,
                                                 self.yedge)
        
        # set horizontal distance
        d = []
        for v in self.xpts:
            d.append(v[2])
        self.d = np.array(d)

        top = self.dis.top.array
        botm = self.dis.botm.array
        elev = [top.copy()]
        for k in xrange(self.dis.nlay):
            elev.append(botm[k, :, :])
        
        self.elev = np.array(elev)
        if self.layer == None:
            self.layer0 = 0
            self.layer1 = self.dis.nlay + 1
        else:
            self.layer0 = self.layer
            self.layer1 = min(self.layer + 2, self.dis.nlay + 1)
        
        zpts = []
        for k in xrange(self.layer0, self.layer1):
            zpts.append(plotutil.cell_value_points(self.xpts, self.xedge,
                                                   self.yedge,
                                                   self.elev[k, :, :]))
        self.zpts = np.array(zpts)
        
        xcentergrid = []
        zcentergrid = []
        nz = 0
        if self.dis.nlay == 1:
            for k in xrange(0, self.zpts.shape[0]):
                nz += 1
                nx = 0
                for i in xrange(0, self.xpts.shape[0], 2):
                    nx += 1
                    xp = 0.5 * (self.xpts[i][2] + self.xpts[i+1][2])
                    zp = self.zpts[k, i]
                    xcentergrid.append(xp)
                    zcentergrid.append(zp)
        else:
            for k in xrange(0, self.zpts.shape[0]-1):
                nz += 1
                nx = 0
                for i in xrange(0, self.xpts.shape[0], 2):
                    nx += 1
                    xp = 0.5 * (self.xpts[i][2] + self.xpts[i+1][2])
                    zp = 0.5 * (self.zpts[k, i] + self.zpts[k+1, i+1])
                    xcentergrid.append(xp)
                    zcentergrid.append(zp)
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
        Plot a three-dimensional array as a patch collection.  If there 
        is a layer tied to the class (self.layer), then the method will 
        plot this layer.

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

        plotarray = a
        if masked_values is not None:
            for mval in masked_values:
                plotarray = np.ma.masked_equal(plotarray, mval)

        vpts = []
        for k in xrange(self.dis.nlay):
            vpts.append(plotutil.cell_value_points(self.xpts, self.xedge,
                                                   self.yedge,
                                                   plotarray[k, :, :]))
        vpts = np.array(vpts)
        if self.layer != None:
            vpts = vpts[this.layer, :]
            vpts.reshape((1, vpts.shape[0], vpts.shape[1]))
            
        if isinstance(head, np.ndarray):
            zpts = self.set_zpts(head)
        else:
            zpts = self.zpts

        if masked_values is not None:
            for mval in masked_values:
                vpts = np.ma.masked_equal(vpts, mval)
        
        pc = self.get_grid_patch_collection(zpts, vpts, **kwargs)
        if pc != None:
            ax.add_collection(pc)
        return pc

    def plot_surface(self, a, masked_values=None, **kwargs):
        """
        Plot a three-dimensional array as lines.  If there is a layer 
        tied to the class (self.layer), then the method will plot 
        this layer.

        Parameters
        ----------
        a : numpy.ndarray
            Three-dimensional array to plot.
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
        for k in xrange(self.dis.nlay):
            vpts.append(plotutil.cell_value_points(self.xpts, self.xedge,
                                                   self.yedge,
                                                   plotarray[k, :, :]))
        vpts = np.array(vpts)
        
        if masked_values is not None:
            for mval in masked_values:
                vpts = np.ma.masked_equal(vpts, mval)

        if self.layer != None:
            vpts = vpts[this.layer, :]
            vpts.reshape((1, vpts.shape[0], vpts.shape[1]))
        plot = []
        for k in xrange(vpts.shape[0]):
            plot.append(ax.plot(self.d, vpts[k, :], **kwargs))
        return plot


    def plot_fill_between(self, a, colors=['blue', 'red'],
                            masked_values=None, **kwargs):
        """
        Plot a three-dimensional array as lines.  If there is a layer 
        tied to the class (self.layer), then the method will plot 
        this layer.

        Parameters
        ----------
        a : numpy.ndarray
            Three-dimensional array to plot.
        masked_values : iterable of floats, ints
            Values to mask.
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
        for k in xrange(self.dis.nlay):
            vpts.append(plotutil.cell_value_points(self.xpts, self.xedge,
                                                   self.yedge,
                                                   plotarray[k, :, :]))
        vpts = np.ma.array(vpts, mask=False)
        
        if masked_values is not None:
            for mval in masked_values:
                vpts = np.ma.masked_equal(vpts, mval)
        idxm = np.ma.getmask(vpts)

        #if self.layer != None:
        #    vpts = vpts[this.layer, :]
        #    vpts.reshape((1, vpts.shape[0], vpts.shape[1]))
        
        plot = []
        for k in xrange(self.dis.nlay):
            if self.layer != None:
                if k != self.layer:
                    continue
            idxmk = idxm[k, :]
            v = vpts[k, :]
            y1 = self.zpts[k, :]
            idx = v > y1
            v[idx] = y1[idx]
            y2 = v
            idx = v > y2
            y2[idx] =  v[idx]
            y1[idxmk] = np.nan
            y2[idxmk] = np.nan
            plot.append(ax.fill_between(self.d, y1=y1, y2=y2, color=colors[0],
                                        **kwargs))
            y1 = y2
            y2 = self.zpts[k+1, :]
            y2[idxmk] = np.nan
            plot.append(ax.fill_between(self.d, y1=y1, y2=y2, color=colors[1],
                                        **kwargs))
        return plot

    def contour_array(self, a, masked_values=None, head=None, **kwargs):
        """
        Contour a three-dimensional array. If there is a layer tied to 
        the class (self.layer), then the method will plot this layer.

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

#        if masked_values is not None:
#            for mval in masked_values:
#                plotarray = np.ma.masked_equal(plotarray, mval)

        vpts = []
        for k in xrange(self.dis.nlay):
            vpts.append(plotutil.cell_value_points(self.xpts, self.xedge,
                                                   self.yedge,
                                                   plotarray[k, :, :]))
        vpts = np.array(vpts)
        vpts = vpts[:, ::2]
        if self.dis.nlay == 1:
            vpts = np.vstack((vpts, vpts))

        if self.layer != None:
            vpts = vpts[this.layer, :]
            vpts.reshape((1, vpts.shape[0], vpts.shape[1]))

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
            ibound = bas.ibound
        plotarray = np.zeros(ibound.shape, dtype=np.int)
        idx1 = (ibound == 0)
        idx2 = (ibound < 0)
        plotarray[idx1] = 1
        plotarray[idx2] = 2
        plotarray = np.ma.masked_equal(plotarray, 0)
        cmap = matplotlib.colors.ListedColormap(['none', color_noflow,
                                                 color_ch])
        bounds=[0, 1, 2, 3]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        #--mask active cells
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
                  masked_values=None, head=None, **kwargs):
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
        # Find package to plot
        if package is not None:
            p = package
        elif self.model is not None:
            if ftype is None:
                raise Exception('ftype not specified')
            p = self.model.get_package(ftype)
        else:
            raise Exception('Cannot find package to plot')

        # Get the list data
        try:
            mflist = p.stress_period_data[kper]
        except:
            raise Exception('Not a list-style boundary package')

        # Return if mflist is None
        if mflist is None:
            return None

        # Plot the list locations
        plotarray = np.zeros(self.dis.botm.shape, dtype=np.int)
        idx = [mflist['k'], mflist['i'], mflist['j']]
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
        bounds=[0, 1, 2]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        patches = self.plot_array(plotarray, masked_values=masked_values,
                                    head=head, cmap=cmap, norm=norm, **kwargs)
        return patches

    def plot_discharge(self, frf, fff, flf=None, head=None,
                         kstep=1, hstep=1,
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
        kwargs : dictionary
            Keyword arguments passed to plt.quiver()

        Returns
        -------
        quiver : matplotlib.pyplot.quiver
            Vectors

        """

        # Calculate specific discharge
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
        sat_thk = plotutil.saturated_thickness(head, top, botm, laytyp,
                                               [hnoflo, hdry])

        # Calculate specific discharge
        qx, qy, qz = plotutil.centered_specific_discharge(frf, fff, flf, delr,
                                                          delc, sat_thk)
        
        if qz == None:
            qz = np.zeros((qx.shape), dtype=np.float)
        
        # Select correct specific discharge direction
        if self.direction == 'x':
            u = -qx[:, :, :]
            u2 = -qy[:, :, :]
            v = qz[:, :, :]
        elif self.direction == 'y':
            u = -qy[:, :, :]
            u2 = -qx[:, :, :]
            v = qz[:, :, :]
        elif self.direction == 'xy':
            print 'csplot_discharge does not support arbitrary cross-sections'
            return None
            
        if isinstance(head, np.ndarray):
            zcentergrid = self.set_zcentergrid(head)
        else:
            zcentergrid = self.zcentergrid
        
        if nlay == 1:
            x = []
            z = []
            for k in xrange(1):
                for i in xrange(self.xcentergrid.shape[1]):
                    x.append(self.xcentergrid[k, i])
                    z.append(0.5 * (zcentergrid[k, i] + zcentergrid[k+1, i]))
            x = np.array(x).reshape((1,self.xcentergrid.shape[1]))
            z = np.array(z).reshape((1,self.xcentergrid.shape[1]))
        else:
            x = self.xcentergrid
            z = zcentergrid
            
        upts = []
        u2pts = []
        vpts = []
        for k in xrange(self.dis.nlay):
            upts.append(plotutil.cell_value_points(self.xpts, self.xedge,
                                                   self.yedge, u[k, :, :]))
            u2pts.append(plotutil.cell_value_points(self.xpts, self.xedge,
                                                    self.yedge, u2[k, :, :]))
            vpts.append(plotutil.cell_value_points(self.xpts, self.xedge,
                                                   self.yedge, v[k, :, :]))
        upts = np.array(upts)
        u2pts = np.array(u2pts)
        vpts = np.array(vpts)
        if self.layer != None:
            upts = upts[this.layer, :]
            upts.reshape((1, upts.shape[0], upts.shape[1]))
            u2pts = u2pts[this.layer, :]
            u2pts.reshape((1, u2pts.shape[0], u2pts.shape[1]))
            vpts = vpts[this.layer, :]
            vpts.reshape((1, vpts.shape[0], vpts.shape[1]))
        
        x = x[::kstep, ::hstep]
        z = z[::kstep, ::hstep]
        upts = upts[::kstep, ::hstep]
        u2pts = u2pts[::kstep, ::hstep]
        vpts = vpts[::kstep, ::hstep]
            
        N = np.sqrt(upts**2. + u2pts**2. + vpts**2.)
        idx = N > 0.
        upts[idx] /= N[idx]
        u2pts[idx] /= N[idx]
        vpts[idx] /= N[idx]
        
        # plot the vectors
        quiver = self.ax.quiver(x, z, upts, vpts, **kwargs)

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

        v = []
        
        colors = []
        for k in xrange(zpts.shape[0]-1):
            for idx in xrange(0, len(self.xpts)-1, 2):
                ll = ((self.xpts[idx][2], zpts[k+1, idx]))
                try:
                    dx = self.xpts[idx+2][2] - self.xpts[idx][2]
                except:
                    dx = self.xpts[idx+1][2] - self.xpts[idx][2]
                dz = zpts[k, idx] - zpts[k+1, idx]
                pts = (ll, 
                      (ll[0], ll[1]+dz), (ll[0]+dx, ll[1]+dz),
                      (ll[0]+dx, ll[1])) #, ll) 
                if np.isnan(plotarray[k, idx]):
                    continue
                if plotarray[k, idx] is np.ma.masked:
                    continue
                rectcol.append(Polygon(pts, closed=True))
                colors.append(plotarray[k, idx])

        if len(rectcol) > 0:
            patches = PatchCollection(rectcol, **kwargs)
            patches.set_array(np.array(colors))
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
        for k in xrange(self.zpts.shape[0]-1):
            for idx in xrange(0, len(self.xpts)-1, 2):
                ll = ((self.xpts[idx][2], self.zpts[k+1, idx]))
                try:
                    dx = self.xpts[idx+2][2] - self.xpts[idx][2]
                except:
                    dx = self.xpts[idx+1][2] - self.xpts[idx][2]
                dz = self.zpts[k, idx] - self.zpts[k+1, idx]
                # horizontal lines
                linecol.append(((ll), (ll[0]+dx, ll[1])))
                linecol.append(((ll[0], ll[1]+dz), (ll[0]+dx, ll[1]+dz)))
                #vertical lines
                linecol.append(((ll), (ll[0], ll[1]+dz)))
                linecol.append(((ll[0]+dx, ll[1]), (ll[0]+dx, ll[1]+dz)))

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
        for k in xrange(self.layer0, self.layer1):
            e = self.elev[k, :, :]
            if k < self.dis.nlay:
                v = vs[k, :, :]
                idx =  v < e
                e[idx] = v[idx] 
            zpts.append(plotutil.cell_value_points(self.xpts, self.xedge,
                                                   self.yedge, e))
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
        for k in xrange(self.layer0, self.layer1):
            if k < self.dis.nlay:
                e = vs[k, :, :]
            else:
                e = self.elev[k, :, :]
            vpts.append(plotutil.cell_value_points(self.xpts, self.xedge,
                                                   self.yedge, e))
        vpts = np.array(vpts)

        zcentergrid = []
        nz = 0
        if self.dis.nlay == 1:
            for k in xrange(0, self.zpts.shape[0]):
                nz += 1
                nx = 0
                for i in xrange(0, self.xpts.shape[0], 2):
                    nx += 1
                    vp = vpts[k, i]
                    zp = self.zpts[k, i]
                    if k == 0:
                        if vp < zp:
                            zp = vp
                    zcentergrid.append(zp)
        else:
            for k in xrange(0, self.zpts.shape[0], 2):
                nz += 1
                nx = 0
                for i in xrange(0, self.xpts.shape[0], 2):
                    nx += 1
                    vp = vpts[k, i]
                    ep = self.zpts[k, i]
                    if vp < ep:
                        ep = vp
                    zp = 0.5 * (ep + self.zpts[k+1, i+1])
                    zcentergrid.append(zp)
        return np.array(zcentergrid).reshape((nz, nx)) 

    def get_xcenter_array(self):
        """
        Return a numpy one-dimensional float array that has the cell center x
        coordinate for every column in the grid.

        """
        x = np.add.accumulate(self.dis.delr.array) - 0.5 * self.dis.delr.array
        return x

    def get_ycenter_array(self):
        """
        Return a numpy one-dimensional float array that has the cell center x
        coordinate for every row in the grid.

        """
        Ly = np.add.reduce(self.dis.delc.array)
        y = Ly - (np.add.accumulate(self.dis.delc.array) - 0.5 *
                   self.dis.delc.array)
        return y

    def get_xedge_array(self):
        """
        Return a numpy one-dimensional float array that has the cell edge x
        coordinates for every column in the grid.  Array is of size (ncol + 1)

        """
        xedge = np.concatenate(([0.], np.add.accumulate(self.dis.delr.array)))
        return xedge

    def get_yedge_array(self):
        """
        Return a numpy one-dimensional float array that has the cell edge y
        coordinates for every row in the grid.  Array is of size (nrow + 1)

        """
        length_y = np.add.reduce(self.dis.delc.array)
        yedge = np.concatenate(([length_y], length_y -
                             np.add.accumulate(self.dis.delc.array)))
        return yedge

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



