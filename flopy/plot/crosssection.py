import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import plotutil

bc_color_dict = {'default': 'black', 'WEL': 'red', 'DRN': 'yellow',
                 'RIV': 'green', 'GHB': 'cyan', 'CHD': 'navy'}


def rotate(x, y, theta, xorigin=0., yorigin=0.):
    """
    Given x and y array-like values calculate the rotation about an
    arbitrary origin and then return the rotated coordinates.  theta is in
    radians.

    """
    xrot = xorigin + np.cos(theta) * (x - xorigin) - np.sin(theta) * \
                                                     (y - yorigin)
    yrot = yorigin + np.sin(theta) * (x - xorigin) + np.cos(theta) * \
                                                     (y - yorigin)
    return xrot, yrot


class ModelCrossSection(object):
    """
    Class to create a map of the model.

    Parameters
    ----------
    ax : matplotlib.pyplot axis
        The plot axis.  If not provided it, plt.gca() will be used.
    dis : flopy discretization object
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
        then these will be calculated based on grid, coordinates, and rotation
    """
    def __init__(self, ax=None, ml=None, dis=None, line=None, layer=None,
                 xul=None, yul=None, rotation=0., extent=None):
        self.ml = ml
        self.layer = layer
        if dis is None:
            if ml is None:
                raise Exception('Cannot find discretization package')
            else:
                self.dis = ml.get_package('DIS')
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
        self.xgrid, self.ygrid = np.meshgrid(self.xedge, self.yedge)
        

        # Create x and y center arrays and meshgrid of centers
        self.xcenter = self.get_xcenter_array()
        self.ycenter = self.get_ycenter_array()
        self.xcentergrid, self.ycentergrid = np.meshgrid(self.xcenter,
                                                         self.ycenter)
                                                         
        onkey = line.keys()[0]                      
        if 'row' in linekeys:
            pts = [(self.xedge[0], self.ycenter[int(line[onkey])]), 
                   (self.xedge[-1], self.ycenter[int(line[onkey])])]
        elif 'column' in linekeys:
            pts = [(self.xcenter[int(line[onkey])], self.yedge[0]), 
                   (self.xcenter[int(line[onkey])], self.yedge[-1])]
        else:
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
        self.xpts = plotutil.line_intersect_grid(self.pts, self.xedge, self.yedge)

        top = self.dis.top.array
        botm = self.dis.botm.array
        elev = [top.copy()]
        for k in xrange(self.dis.nlay):
            elev.append(botm[k, :, :])
        
        elev = np.array(elev)
        if self.layer == None:
            self.layer0 = 0
            self.layer1 = self.dis.nlay + 1
        else:
            self.layer0 = self.layer
            self.layer1 = min(self.layer + 2, self.dis.nlay + 1)
        
        zpts = []
        for k in xrange(self.layer0, self.layer1):
            zpts.append(plotutil.cell_value_points(self.xpts, self.xedge, self.yedge, elev[k, :, :]))
        
        self.zpts = np.array(zpts)
        
        
        ## Rotate xgrid and ygrid
        #self.xgrid, self.ygrid = rotate(self.xgrid, self.ygrid, self.rotation,
        #                                0, self.yedge[0])
        #self.xgrid += self.xul
        #self.ygrid += self.yul - self.yedge[0]
        #
        ## Rotate xcentergrid and ycentergrid
        #self.xcentergrid, self.ycentergrid = rotate(self.xcentergrid,
        #                                            self.ycentergrid,
        #                                            self.rotation,
        #                                            0, self.yedge[0])
        #self.xcentergrid += self.xul
        #self.ycentergrid += self.yul - self.yedge[0]

        # Create model extent
        if extent is None:
            self.extent = self.get_extent()
        else:
            self.extent = extent

        # Set axis limits
        self.ax.set_xlim(self.extent[0], self.extent[1])
        self.ax.set_ylim(self.extent[2], self.extent[3])

        return

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
        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax
        # if a.ndim == 3:
        #     plotarray = a[self.layer, :, :]
        # elif a.ndim == 2:
        #     plotarray = a
        # else:
        #     raise Exception('Array must be of dimension 2 or 3')
        plotarray = a
        if masked_values is not None:
            for mval in masked_values:
                plotarray = np.ma.masked_equal(plotarray, mval)

        vpts = []
        for k in xrange(self.dis.nlay):
            vpts.append(plotutil.cell_value_points(self.xpts, self.xedge, self.yedge, plotarray[k, :, :]))
        vpts = np.array(vpts)
        if self.layer != None:
            vpts = vpts[this.layer, :]
            vpts.reshape((1, vpts.shape[0], vpts.shape[1]))

        #quadmesh = self.ax.pcolormesh(self.xgrid, self.ygrid, plotarray, **kwargs)
        pc = self.get_grid_patch_collection(vpts, **kwargs)
        ax.add_collection(pc)
        return pc

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
        if a.ndim == 3:
            plotarray = a[self.layer, :, :]
        elif a.ndim == 2:
            plotarray = a
        else:
            raise Exception('Array must be of dimension 2 or 3')
        if masked_values is not None:
            for mval in masked_values:
                plotarray = np.ma.masked_equal(plotarray, mval)
        contour_set = self.ax.contour(self.xcentergrid, self.ycentergrid,
                                      plotarray, **kwargs)
        return contour_set

    def plot_ibound(self, ibound=None, color_noflow='black', color_ch='blue',
                    **kwargs):
        """
        Make a plot of ibound.  If not specified, then pull ibound from the
        self.ml

        Parameters
        ----------
        ibound : numpy.ndarray
            ibound array to plot.  (Default is ibound in 'BAS6' package.)
        color_noflow : string
            (Default is 'black')
        color_ch : string
            Color for constant heads (Default is 'blue'.)

        Returns
        -------
        quadmesh : matplotlib.collections.QuadMesh
        """
        if ibound is None:
            bas = self.ml.get_package('BAS6')
            ibound = bas.ibound
        plotarray = np.zeros(ibound.shape, dtype=np.int)
        idx1 = (ibound == 0)
        idx2 = (ibound < 0)
        plotarray[idx1] = 1
        plotarray[idx2] = 2
        plotarray = np.ma.masked_equal(plotarray, 0)
        cmap = matplotlib.colors.ListedColormap(['none', color_noflow, color_ch])
        bounds=[0, 1, 2, 3]
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
        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax

        if 'color' not in kwargs:
            kwargs['color'] = '0.5'
        
        lc = self.get_grid_line_collection(**kwargs)
        ax.add_collection(lc)
        return lc

    def plot_bc(self, ftype=None, package=None, kper=0, color=None, **kwargs):
        """
        Plot a boundary locations for a flopy model

        """
        # Find package to plot
        if package is not None:
            p = package
        elif self.ml is not None:
            if ftype is None:
                raise Exception('ftype not specified')
            p = self.ml.get_package(ftype)
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
        quadmesh = self.plot_array(plotarray, cmap=cmap, norm=norm, **kwargs)
        return quadmesh

    def plot_discharge(self, frf, fff, flf=None, head=None, istep=1, jstep=1,
                       **kwargs):
        """
        Use quiver to plot vectors.

        Parameters
        ----------
        frf : numpy.ndarray
            MODFLOW's 'flow right face'
        fff : numpy.ndarray
            MODFLOW's 'flow front face'
        fff : numpy.ndarray
            MODFLOW's 'flow lower face' (Default is None.)
        head : numpy.ndarray
            MODFLOW's head array.  If not provided, then will assume confined
            conditions in order to calculated saturated thickness.
        istep : int
            row frequency to plot. (Default is 1.)
        jstep : int
            column frequency to plot. (Default is 1.)
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
        if self.ml is not None:
            lpf = self.ml.get_package('LPF')
            if lpf is not None:
                laytyp = lpf.laytyp.array
                hdry = lpf.hdry
            bas = self.ml.get_package('BAS6')
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

        # Select correct slice and step
        x = self.xcentergrid[::istep, ::jstep]
        y = self.ycentergrid[::istep, ::jstep]
        u = qx[self.layer, :, :]
        v = qy[self.layer, :, :]
        u = u[::istep, ::jstep]
        v = v[::istep, ::jstep]

        # Rotate and plot
        urot, vrot = rotate(u, v, self.rotation)
        quiver = self.ax.quiver(x, y, urot, vrot, **kwargs)

        return quiver


    def get_grid_patch_collection(self, plotarray, **kwargs):
        """
        Get a PatchCollection of the grid
        """
        from matplotlib.patches import Polygon
        from matplotlib.collections import PatchCollection
        rectcol = []

        v = []
        
        colors = []
        for k in xrange(self.zpts.shape[0]-1):
            for idx in xrange(0, len(self.xpts)-1, 2):
                ll = ((self.xpts[idx][2], self.zpts[k+1, idx]))
                try:
                    dx = self.xpts[idx+2][2] - self.xpts[idx][2]
                except:
                    dx = self.xpts[idx+1][2] - self.xpts[idx][2]
                dz = self.zpts[k, idx] - self.zpts[k+1, idx]
                pts = (ll, 
                      (ll[0], ll[1]+dz), (ll[0]+dx, ll[1]+dz),
                      (ll[0]+dx, ll[1])) #, ll) 
                if np.isnan(plotarray[k, idx]):
                    continue
                rectcol.append(Polygon(pts, closed=True))
                colors.append(plotarray[k, idx])

        pc = PatchCollection(rectcol, **kwargs)
        pc.set_array(np.array(colors))
        return pc

    def get_grid_line_collection(self, **kwargs):
        """
        Get a LineCollection of the grid
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

        lc = LineCollection(linecol, **kwargs)
        return lc

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


