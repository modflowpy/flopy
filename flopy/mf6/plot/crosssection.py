import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from ..plot import plotutil
from ..plot.plotutil import bc_color_dict
from ..utils.decorators import deprecated

class StructuredModelCrossSection(object):
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
        indicates clockwise rotation.  Angles are in degrees.
    extent : tuple of floats
        (xmin, xmax, ymin, ymax) will be used to specify axes limits.  If None
        then these will be calculated based on grid, coordinates, and rotation.

    """
    def __init__(self, ax=None, model=None, dis=None, sr=None, line=None,
                 xul=None, yul=None, rotation=None, extent=None):
        self.model = model
        if dis is None:
            if model is None:
                raise Exception('Cannot find discretization package')
            else:
                self.dis = model.get_package('DIS')
                self.sr = copy.deepcopy(self.dis.sr)
        else:
            self.dis = dis
            self.sr = copy.deepcopy(dis.sr)
        if line is None:
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
        if 'row' in linekeys:
            self.direction = 'x'
            pts = [(self.sr.xedge[0]+0.1, self.sr.ycenter[int(line[onkey])]-0.1),
                   (self.sr.xedge[-1]-0.1, self.sr.ycenter[int(line[onkey])]+0.1)]
        elif 'column' in linekeys:
            self.direction = 'y'
            pts = [(self.sr.xcenter[int(line[onkey])]+0.1, self.sr.yedge[0]-0.1),
                   (self.sr.xcenter[int(line[onkey])]-0.1, self.sr.yedge[-1]+0.1)]
        else:
            self.direction = 'xy'
            verts = line[onkey]
            xp = []
            yp = []
            for [v1, v2] in verts:
                xp.append(v1)
                yp.append(v2)
            xp, yp = np.array(xp, dtype=np.float), np.array(yp, dtype=np.float)
            # remove offset and rotation from line
            xp -= self.sr.xul
            yp -= self.sr.yul
            xp, yp = self.sr.rotate(xp, yp, -self.sr.rotation, 0, self.sr.yedge[0])
            pts = []
            for xt, yt in zip(xp, yp):
                pts.append((xt, yt))
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

        top = self.dis.top #.array
        botm = self.dis.botm #.array
        elev = [top.copy()]

        for k in range(self.dis.nlay):
            elev.append(botm[k, :, :])
        
        self.elev = np.array(elev)
        self.layer0 = 0
        self.layer1 = self.dis.nlay + 1
        
        zpts = []
        for k in range(self.layer0, self.layer1):
            zpts.append(plotutil.cell_value_points(self.xpts, self.sr.xedge,
                                                   self.sr.yedge,
                                                   self.elev[k, :, :]))
        self.zpts = np.array(zpts)
        # zpts is array of top, then bottom elevations for each cell that crosses
        # the cross sectional line
        xcentergrid = []
        zcentergrid = []
        nz = 0
        if self.dis.nlay == 1:
            for k in range(0, self.zpts.shape[0]):
                nz += 1
                nx = 0
                for i in range(0, self.xpts.shape[0], 2):
                    try:
                        xp = 0.5 * (self.xpts[i][2] + self.xpts[i+1][2])
                        zp = self.zpts[k, i]
                        xcentergrid.append(xp)
                        zcentergrid.append(zp)
                        nx += 1
                    except:
                        break
        else:
            for k in range(0, self.zpts.shape[0]-1):
                nz += 1
                nx = 0
                for i in range(0, self.xpts.shape[0], 2):
                    try:
                        xp = 0.5 * (self.xpts[i][2] + self.xpts[i+1][2])
                        zp = 0.5 * (self.zpts[k, i] + self.zpts[k+1, i+1])
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

        plotarray = a
        if masked_values is not None:
            for mval in masked_values:
                plotarray = np.ma.masked_equal(plotarray, mval)

        vpts = []
        for k in range(self.dis.nlay):
            vpts.append(plotutil.cell_value_points(self.xpts, self.sr.xedge,
                                                   self.sr.yedge,
                                                   plotarray[k, :, :]))
        vpts = np.array(vpts)
            
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
            plotarray = np.reshape(plotarray, (1, plotarray.shape[0], plotarray.shape[1]))
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
            vpts.append(plotutil.cell_value_points(self.xpts, self.sr.xedge,
                                                   self.sr.yedge,
                                                   plotarray[k, :, :]))
        vpts = np.ma.array(vpts, mask=False)

        if isinstance(head, np.ndarray):
            zpts = self.set_zpts(head)
        else:
            zpts = self.zpts

        if masked_values is not None:
            for mval in masked_values:
                vpts = np.ma.masked_equal(vpts, mval)
        idxm = np.ma.getmask(vpts)

        plot = []
        for k in range(self.dis.nlay):
            idxmk = idxm[k, :]
            v = vpts[k, :]
            y1 = zpts[k, :]
            y2 = zpts[k+1, :]
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

    @deprecated('ibound array is not present in modflow 6'+\
                'please use <plot_idomain> and CHD.plot')
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
            try:
                # todo: create a self.model.get_package('CHD8')
                idomain = self.model.dis.idomain
            except:
                try:
                    idomain = self.dis.idomain
                except:
                    raise Exception
            try:
                chd = self.model.get_package('CHD').stress_period_data[self.kper]
            except:
                raise Exception

                # todo: update ibound to reflect current
            plotarray = np.zeros(ibound.shape, dtype=np.int)
            idx1 = (idomain == 0)
            idx2 = (chd != 0)
            idx3 = (idomain == -1)

            plotarray[idx1] = 1
            plotarray[idx2] = 2
            plotarray[idx3] = 3
        else:
            raise NotImplementedError('ibound passed as an array is not yet implemented')

        plotarray = np.ma.masked_equal(plotarray, 0)
        cmap = matplotlib.colors.ListedColormap(['0', color_noflow, color_ch,
                                                 color_vpt])
        bounds = [0, 1, 2, 3, 4]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        patches = self.plot_array(plotarray, cmap=cmap, norm=norm, head=head,
                                  masked_values=[0.], **kwargs)

        return patches

    def plot_idomain(self, idomain=None, color_noflow='black', color_vpt='red',
                     head=None, **kwargs):
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
            color_vpt: string
                Color for vertical pass through cells from the idomain array

            Returns
            -------
            quadmesh : matplotlib.collections.QuadMesh

        """
        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax
        if idomain is None:
            try:
                idomain = self.model.dis.idomain
            except:
                try:
                    idomain = self.dis.idomain
                except:
                    raise AssertionError('Discretation information not supplied to ModelMap()')

        plotarray = np.zeros(idomain.shape, dtype=np.int)
        idx1 = (idomain == 0)
        idx2 = (idomain == -1)
        plotarray[idx1] = 1
        plotarray[idx2] = 2
        plotarray = np.ma.masked_equal(plotarray, 0)
        cmap = matplotlib.colors.ListedColormap(['1', color_noflow, color_vpt])
        bounds = [0, 1, 2, 3]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        patches = self.plot_array(plotarray, cmap=cmap, norm=norm, head=head,
                                  masked_values=[0.], **kwargs)
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
        elif self.model is not None:
            if ftype is None:
                raise Exception('ftype not specified')
            p = self.model.get_package(ftype)
        else:
            raise Exception('Cannot find package to plot')

        # Get the list data
        try:
            #todo: return this to p.stress_period_data[kper] from development
            mflist = p.data[kper]
            # mflist = p.stress_period_data[kper]
        except:
            raise Exception('Not a list-style boundary package')

        # Return if MfList is None
        if mflist is None:
            return None

        # Plot the list locations
        plotarray = np.zeros(self.dis.botm.shape, dtype=np.int)
        # todo: check to see if data is zero based or 1 based if 0 based remove <-1>
        idx = [mflist['layer'] - 1, mflist['row'] - 1, mflist['column'] - 1]
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

    def plot_discharge(self, ja, dis=None, head=None, kstep=1, hstep=1,
                       normalize=False, **kwargs):
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

        if dis is None:
            try:
                dis = self.dis
            except:
                try:
                    dis = self.model.dis
                except:
                    raise AssertionError("ModelMap.plot_quiver() error: self.dis is None and dis arg is None ")

        # todo: update to reflect new ibound setup
        if len(head.shape) == 3:
            head = head[0]

        ja = np.array(ja)
        if len(ja.shape) == 4:
            ja = ja[0][0][0]

        # Calculate specific discharge
        delr = np.tile(dis.delr, (dis.ncol, 1))
        delc = np.tile(dis.delc, (dis.nrow, 1))
        top = np.copy(dis.top)
        botm = np.copy(dis.botm)
        nlay, nrow, ncol = botm.shape
        ncpl = nrow * ncol
        laytyp = None
        hnoflo = 999.
        hdry = 999.

        # If no access to head or laytyp, then calculate confined saturated
        # thickness by setting laytyp to zeros
        if head is None:
            head = np.zeros(botm.shape, np.float32)

        head.shape = (nlay, ncpl)
        top.shape = (ncpl)
        botm.shape = (nlay, ncpl)
        delr.shape = (ncpl)
        delc.shape = (ncpl)

        sat_thk = plotutil.saturated_thickness(head, top, botm,
                                               [hnoflo, hdry])

        # todo: adapt flf into the vectorize flow definition
        frf, fff, flf = plotutil.vectorize_flow(ja, dis)

        # Calculate specific discharge
        qx, qy, qz = plotutil.specific_discharge(frf, fff, flf, delr,
                                                          delc, sat_thk)

        qx.shape = (nlay, nrow, ncol)
        qy.shape = (nlay, nrow, ncol)
        qz.shape = (nlay, nrow, ncol)
        head.shape = (nlay, nrow, ncol)

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
                    z.append(0.5 * (zcentergrid[k, i] + zcentergrid[k+1, i]))
            x = np.array(x).reshape((1, self.xcentergrid.shape[1]))
            z = np.array(z).reshape((1, self.xcentergrid.shape[1]))
        else:
            x = self.xcentergrid
            z = zcentergrid
            
        upts = []
        u2pts = []
        vpts = []
        idpts = []
        idomain = dis.idomain
        # idx = (idomain[self.layer, ::istep, ::jstep] == 0)
        # idx[idomain[self.layer, ::istep, ::jstep] == -1] = 1
        for k in range(self.dis.nlay):
            upts.append(plotutil.cell_value_points(self.xpts, self.sr.xedge,
                                                   self.sr.yedge, u[k, :, :]))
            u2pts.append(plotutil.cell_value_points(self.xpts, self.sr.xedge,
                                                    self.sr.yedge, u2[k, :, :]))
            vpts.append(plotutil.cell_value_points(self.xpts, self.sr.xedge,
                                                   self.sr.yedge, v[k, :, :]))
            idpts.append(plotutil.cell_value_points(self.xpts, self.sr.xedge,
                                                    self.sr.yedge, dis.idomain[k, :, :]))
        # convert upts, u2pts, and vpts to numpy arrays
        upts = np.array(upts)
        u2pts = np.array(u2pts)
        vpts = np.array(vpts)
        idpts = np.array(idpts)

        # Select correct slice and apply step
        x = x[::kstep, ::hstep]
        z = z[::kstep, ::hstep]
        upts = upts[::kstep, ::hstep]
        u2pts = u2pts[::kstep, ::hstep]
        vpts = vpts[::kstep, ::hstep]
        idpts = idpts[::kstep, ::hstep]

        # normalize
        if normalize:
            if self.direction == 'xy':
                vmag = np.sqrt(upts**2. + u2pts**2. + vpts**2.)
            else:
                vmag = np.sqrt(upts**2. + vpts**2.)
            idx = vmag > 0.
            upts[idx] /= vmag[idx]
            u2pts[idx] /= vmag[idx]
            vpts[idx] /= vmag[idx]

        # upts and vpts has a value for the left and right
        # sides of a cell. Sample every other value for quiver
        upts = upts[0, ::2]
        vpts = vpts[0, ::2]
        idpts = idpts[0, ::2]

        # mask discharge in inactive cells
        idx = (idpts == 0)
        idx[idpts == -1] = 1
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

        v = []
        
        colors = []
        for k in range(zpts.shape[0]-1):
            for idx in range(0, len(self.xpts)-1, 2):
                try:
                    ll = ((self.xpts[idx][2], zpts[k+1, idx]))
                    try:
                        dx = self.xpts[idx+2][2] - self.xpts[idx][2]
                    except:
                        dx = self.xpts[idx+1][2] - self.xpts[idx][2]
                    dz = zpts[k, idx] - zpts[k+1, idx]
                    pts = (ll, 
                           (ll[0], ll[1]+dz), (ll[0]+dx, ll[1]+dz),
                           (ll[0]+dx, ll[1]))
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
        for k in range(self.zpts.shape[0]-1):
            for idx in range(0, len(self.xpts)-1, 2):
                try:
                    ll = ((self.xpts[idx][2], self.zpts[k+1, idx]))
                    try:
                        dx = self.xpts[idx+2][2] - self.xpts[idx][2]
                    except:
                        dx = self.xpts[idx+1][2] - self.xpts[idx][2]
                    dz = self.zpts[k, idx] - self.zpts[k+1, idx]
                    # horizontal lines
                    linecol.append(((ll), (ll[0]+dx, ll[1])))
                    linecol.append(((ll[0], ll[1]+dz), (ll[0]+dx, ll[1]+dz)))
                    # vertical lines
                    linecol.append(((ll), (ll[0], ll[1]+dz)))
                    linecol.append(((ll[0]+dx, ll[1]), (ll[0]+dx, ll[1]+dz)))
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
                    zp = 0.5 * (ep + self.zpts[k+1, i+1])
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


class VertexModelCrossSection(object):
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
            Dictionary with "line" key. If key
            is "row" or "column" key value should be the zero-based row or
            column index for cross-section. If key is "line" value should
            be an array of (x, y) tuples with vertices of cross-section.
            Vertices should be in map coordinates consistent with xul,
            yul, and rotation.
        extent : tuple of floats
            (xmin, xmax, ymin, ymax) will be used to specify axes limits.  If None
            then these will be calculated based on grid, coordinates, and rotation.

    """

    def __init__(self, ax=None, model=None, dis=None, sr=None, line=None,
                 extent=None, **kwargs):

        self.model = model

        if dis is None:
            if model is None:
                raise Exception('Cannot find discretization package')
            else:
                self.dis = model.get_package('DIS')
                self.sr = copy.deepcopy(self.dis.sr)
        else:
            self.dis = dis
            self.sr = copy.deepcopy(dis.sr)

        if line is None:
            s = 'line must be specified.'
            raise Exception(s)

        linekeys = [linekeys.lower() for linekeys in list(line.keys())]

        if len(linekeys) > 1:
            s = 'only row, column, or line can be specified in line dictionary.\n'
            s += 'keys specified: '
            for k in linekeys:
                s += '{} '.format(k)
            raise Exception(s)

        if 'row' in linekeys:
            s = 'row cannot be specified for vertex Discretization'
            raise Exception(s)

        if 'column' in linekeys:
            s = 'column cannot be specified for vertex Discretization.'
            raise Exception(s)

        if ax is None:
            self.ax = plt.gca()
        else:
            self.ax = ax

        onkey = list(line.keys())[0]

        assert 'line' in linekeys, 'cross section must be specified using\
         "line": [(x,y), (xn, yn)] in linekeys'

        self.direction = 'xy'
        pts = line[onkey]
        # convert pts list to numpy array
        self.pts = np.array(pts)

        # get points along the line
        self.xpts = plotutil.line_intersect_vertex_grid(self.pts, self.sr.xydict)
        # xpts returns xy intersection locations of line by cell

        if len(self.xpts) < 2:
            s = 'cross-section cannot be created\n.'
            s += '   less than 2 points intersect the model grid\n'
            s += '   {} points intersect the grid.'.format(len(self.xpts))
            raise Exception(s)

        top = self.dis.top
        botm = self.dis.botm
        elev = [top.copy()]
        for k in range(self.dis.nlay):
            elev.append(botm[k, :])

        self.elev = np.array(elev)
        self.layer0 = 0
        self.layer1 = self.dis.nlay + 1

        zpts = []
        for k in range(self.layer0, self.layer1):
            zpts.append(plotutil.cell_value_points_from_dict(self.xpts, self.elev[k,:]))
        self.zpts = zpts

        xcentergrid = []
        zcentergrid = []
        xparr = []
        zparr = []
        # todo: redo this section
        if self.dis.nlay == 1:
            for k in range(1, len(zpts)):
                xp = {}
                zp = {}
                for i, value in zpts[k].items():
                    try:
                        xparr.append([self.xpts[i][0][0], self.xpts[i][-1][0]])
                        zparr.append([value, zpts[k - 1][i]])
                        xp = 0.5 * (self.xpts[i][0][0] + self.xpts[i][-1][0])
                        zp = 0.5 * (value + self.zpts[k - 1][i])

                    except:
                        break
                xcentergrid.append(xp)
                zcentergrid.append(zp)

        else:
            for k in range(1, len(zpts)):
                xp = {}
                zp = {}
                for i, value in zpts[k].items():
                    try:
                        xparr.append([self.xpts[i][0][0], self.xpts[i][-1][0]])
                        zparr.append([value, zpts[k - 1][i]])
                        xp[i] = 0.5 * (self.xpts[i][0][0] + self.xpts[i][-1][0])
                        zp[i] = 0.5 * (value + zpts[k - 1][i])

                    except:
                        break
                xcentergrid.append(xp)
                zcentergrid.append(zp)

        # xparr and zparr are unordered arrays that are used to calculate extent
        self.xparr = np.array(xparr)
        self.zparr = np.array(zparr)

        self.xcentergrid = xcentergrid
        self.zcentergrid = zcentergrid

        # get distances and order of cell id's along distance array
        self.d, self.xc_ia = self._get_distance_index_array()
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
            Two-dimensional array to plot.
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
        mask = np.zeros(plotarray.shape, dtype=bool)

        if masked_values is not None:
            for mval in masked_values:
                plotarray = np.ma.masked_equal(plotarray, mval)
                mask = np.ma.getmask(plotarray)

        if type(mask) is np.bool_:
            mask = np.zeros(plotarray.shape, dtype=bool)

        vpts = []
        for k in range(self.dis.nlay):
            vpts.append(plotutil.cell_value_points_from_dict(self.xpts, plotarray[k, :]))

        if isinstance(head, np.ndarray):
            if len(head.shape) == 3:
                head = head[:, 0]
            zpts = self.set_zpts(head)
        else:
            zpts = self.zpts

        pc = self.get_patch_collection(self.xpts, zpts, vpts, mask, **kwargs)
        if pc != None:
            ax.add_collection(pc)
        return pc

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

        # if 'color' not in kwargs:
        #     kwargs['color'] = '0.5'

        if 'edgecolor' not in kwargs:
            kwargs['edgecolor'] = '0.5'

        if 'facecolor' not in kwargs:
            kwargs['facecolor'] = 'none'

        pc = self.get_patch_collection(self.xpts, self.zpts, grid=True, **kwargs)
        ax.add_collection(pc)

        ax.set_xlim(self.extent[0], self.extent[1])
        ax.set_ylim(self.extent[2], self.extent[3])

        return pc

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
        from scipy.interpolate import griddata

        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax

        plotarray = a

        vpts = []
        for k in range(self.dis.nlay):
            vpts.append(plotutil.cell_value_points_from_dict(self.xpts, plotarray[k, :]))

        if self.dis.nlay == 1:
            vpts = vpts.append(vpts[0])

        if masked_values is not None:
             for mval in masked_values:
                 vpts = np.ma.masked_equal(vpts, mval)

        if isinstance(head, np.ndarray):
            zpts = self.set_zpts(head)
        else:
            zpts = self.zpts

        xpts, zpts, vpts = self.set_arrays(self.xcentergrid, zpts, vpts)

        xi = np.linspace(np.min(xpts), np.max(xpts), 1000)
        yi = np.linspace(np.min(zpts), np.max(zpts), 1000)

        zi = griddata((xpts, zpts), vpts, (xi[None, :], yi[:, None]), method='cubic')

        contour_set = ax.contour(xi, yi, zi, **kwargs)
        return contour_set

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
        elif self.model is not None:
            if ftype is None:
                raise Exception('ftype not specified')
            p = self.model.get_package(ftype)
        else:
            raise Exception('Cannot find package to plot')

        # Get the list data
        try:
            # todo: return this to p.stress_period_data[kper] from development
            mflist = p.data[kper]
            # mflist = p.stress_period_data[kper]
        except:
            raise Exception('Not a list-style boundary package')

        # Return if MfList is None
        if mflist is None:
            return None

        # Plot the list locations
        plotarray = np.zeros(self.dis.botm.shape, dtype=np.int)
        # todo: check to see if data is zero based or 1 based if 0 based remove <-1>
        idx = [mflist['layer'] - 1, mflist['ncpl'] - 1]
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
        if len(plotarray.shape) == 1:
            nlay = 1
            plotarray.shape = (1, plotarray.shape[0])
        elif len(plotarray.shape) == 2:
            nlay = plotarray.shape[0]
        else:
            raise Exception('plot_array array must be a 2D or 3D array')
        for k in range(nlay):
            vpts.append(plotutil.cell_value_points_from_dict(self.xpts, plotarray[k, :]))
        # vpts = np.array(vpts)

        if masked_values is not None:
            for mval in masked_values:
                vpts = np.ma.masked_equal(vpts, mval)

        # order vpts based on the self.xc_ia <xc index array>
        vptarr = np.array([[layer[i] for i in self.xc_ia] for layer in vpts])

        plot = []
        for k in range(vpts.shape[0]):
            plot.append(ax.plot(self.d, vptarr[k, :], **kwargs))
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
            vpts.append(plotutil.cell_value_points_from_dict(self.xpts, plotarray[k, :]))

        if isinstance(head, np.ndarray):
            zpts = self.set_zpts(head)
        else:
            zpts = self.zpts

        # create sorted ordered arrays that correspond to self.d
        zpts = np.array([[layer[i] for i in self.xc_ia] for layer in zpts])
        vpts = np.array([[layer[i] for i in self.xc_ia] for layer in vpts])

        vpts = np.ma.array(vpts, mask=False)

        if masked_values is not None:
            for mval in masked_values:
                vpts = np.ma.masked_equal(vpts, mval)
        idxm = np.ma.getmask(vpts)

        plot = []
        for k in range(self.dis.nlay):
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
            plot.append(ax.fill_between(self.d, y1=y1, y2=y2, color=colors[0],
                                        **kwargs))
            y1 = y2
            y2 = zpts[k + 1, :]
            y2[idxmk] = np.nan
            plot.append(ax.fill_between(self.d, y1=y1, y2=y2, color=colors[1],
                                        **kwargs))
        return plot

    def plot_idomain(self, idomain=None, color_noflow='black', color_vpt='red',
                     head=None, **kwargs):
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
        color_vpt: string
            Color for vertical pass through cells from the idomain array

        Returns
        -------
        quadmesh : matplotlib.collections.QuadMesh

        """
        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            ax = self.ax
        if idomain is None:
            try:
                idomain = self.model.dis.idomain
            except:
                try:
                    idomain = self.dis.idomain
                except:
                    raise AssertionError('Discretation information not supplied to ModelMap()')

        plotarray = np.zeros(idomain.shape, dtype=np.int)
        idx1 = (idomain == 0)
        idx2 = (idomain == -1)
        plotarray[idx1] = 1
        plotarray[idx2] = 2
        plotarray = np.ma.masked_equal(plotarray, 0)
        cmap = matplotlib.colors.ListedColormap(['1', color_noflow, color_vpt])
        bounds = [0, 1, 2, 3]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        patches = self.plot_array(plotarray, cmap=cmap, norm=norm, head=head,
                                  masked_values=[0.], **kwargs)
        return patches

    @deprecated('ibound array is not present in modflow 6' + \
                'please use <plot_idomain> and CHD.plot')
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
            try:
                # todo: create a self.model.get_package('CHD8')
                idomain = self.model.dis.idomain
            except:
                try:
                    idomain = self.dis.idomain
                except:
                    raise Exception
            try:
                chd = self.model.get_package('CHD').stress_period_data[self.kper]
            except:
                raise Exception

            # todo: update ibound to reflect current
            plotarray = np.zeros(ibound.shape, dtype=np.int)
            idx1 = (idomain == 0)
            idx2 = (chd != 0)
            idx3 = (idomain == -1)

            plotarray[idx1] = 1
            plotarray[idx2] = 2
            plotarray[idx3] = 3
        else:
            raise NotImplementedError('ibound passed as an array is not yet implemented')

        plotarray = np.ma.masked_equal(plotarray, 0)
        cmap = matplotlib.colors.ListedColormap(['0', color_noflow, color_ch,
                                                 color_vpt])
        bounds = [0, 1, 2, 3, 4]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        patches = self.plot_array(plotarray, cmap=cmap, norm=norm, head=head,
                                  masked_values=[0.], **kwargs)

        return patches

    def get_patch_collection(self, xpts, zpts, vpts=None, mask=None, grid=False, **kwargs):
        """
        Method to create matplotlib.patches objects from verticies

        """
        from matplotlib.patches import Polygon
        from matplotlib.collections import PatchCollection

        if not grid:

            assert vpts is not None, 'plotarray must be provided to get_patch_collection'
            assert mask is not None, 'mask must be provided to get_patch_collection'

            patches = []
            a = np.array([])
            for k in range(1, len(zpts)):
                for idx, value in zpts[k].items():
                    if not mask[k - 1][idx]:
                        patch = np.array([[xpts[idx][0][0], zpts[k-1][idx]],
                                          [xpts[idx][0][0], zpts[k][idx]],
                                          [xpts[idx][1][0], zpts[k][idx]],
                                          [xpts[idx][1][0], zpts[k-1][idx]]])

                        # sort vertices by angle
                        patch = plotutil.arctan2(patch)
                        patches.append(Polygon(patch, True))
                        a = np.append(a, vpts[k-1][idx])

            p = PatchCollection(patches, **kwargs)
            p.set_array(a)

        else:
            # method to plot gridlines using the patch collection
            patches = []
            for k in range(1, len(zpts)):
                for idx, value in zpts[k].items():
                    patch = np.array([[xpts[idx][0][0], zpts[k - 1][idx]],
                                      [xpts[idx][0][0], zpts[k][idx]],
                                      [xpts[idx][1][0], zpts[k][idx]],
                                      [xpts[idx][1][0], zpts[k - 1][idx]]])

                    # sort vertices by angle
                    patch = plotutil.arctan2(patch)
                    patches.append(Polygon(patch, True))

            p = PatchCollection(patches, **kwargs)

        return p

    def set_arrays(self, xpts, zpts, vpts, center=False):
        """
        Method to order dictionary values into 1d arrays for plotting, if center
        is false array method extends data to bounds of the domain, which is useful
        for plotting contours.

        Parameters
        ----------
        vpts: (dict): value pts
        xpts: (dict): xpt locations
        zpts: (dict): zpt locations

        Returns
        -------
        vpt_arr (np.array), xpt_arr (np.array), zpt_arr (np.array)

        """
        vpt_arr = []
        xpt_arr = []
        zpt_arr = []

        for i, layer in enumerate(vpts):
            for j, val in layer.items():
                vpt_arr.append(val)
                xpt_arr.append(xpts[i][j])
                zpt_arr.append(zpts[i][j])

        if center is False:
            for j in vpts[-1]:
                vpt_arr.append(vpts[-1][j])
                xpt_arr.append(xpts[-1][j])
                zpt_arr.append(zpts[-1][j])

        return xpt_arr, zpt_arr, vpt_arr

    def _get_distance_index_array(self):
        """
        1 dimensional list of distances for along a line for plotting surfaces
        1 dimensional list of cell id's corresponding to those distance values
        Returns
        -------

        """
        import copy

        xpts = copy.copy(self.xpts)

        # lenxc = len(xpts) + 1
        d = []
        xc_ia = []
        evaluator = 0
        while len(xpts) > 0:
            lowest = 1e10
            c = None
            for cell, values in xpts.items():
                temp = values[0][0]
                if values[0][0] >=  evaluator:
                    if temp <= lowest:
                        lowest = temp
                        c = cell



            if len(xpts) != 1:
                d.append(xpts[c][0][0])
                xc_ia.append(c)

            else:
                d.append(xpts[c][0][0])
                xc_ia.append(c)
                d.append(xpts[c][1][0])
                xc_ia.append(c)

            xpts.pop(c)

        return d, xc_ia


    def get_extent(self):
        """
        Method to get the extent of the cross section requested for plotting
        """
        extent = [np.min(self.xparr), np.max(self.xparr),
                  np.min(self.zparr), np.max(self.zparr)]
        return extent

    def set_zpts(self, head):
        """
        Convienence method to adjust zpts based on head values in model output

        Returns
        -------
        """

        newpts = []
        for i in range(0, len(self.zpts) - 1):
            temp = {}
            for j, val in self.zpts[i].items():
                if head[i][j] < val:
                    temp[j] = head[i][j]
                else:
                    temp[j] = val
            newpts.append(temp)

        newpts.append(self.zpts[-1])
        return newpts

class ModelCrossSection(object):
    def __new__(cls, ax=None, dis=None, sr=None, model=None, line=None, xul=None,
                yul=None, rotation=None, extent=None, distype='structured' ):
        if distype == 'structured':
            new = object.__new__(StructuredModelCrossSection)
            new.__init__(ax=ax, model=model, dis=dis, sr=sr, line=line,
                         xul=xul, yul=yul, rotation=rotation, extent=extent)

        elif distype == 'vertex':
            new = object.__new__(VertexModelCrossSection)
            new.__init__(ax=ax, model=model, dis=dis, sr=sr, line=line,
                         extent=extent)

        elif distype == 'unstructured':
            raise NotImplementedError('Unstructured grid not yet implemented')

        else:
            raise TypeError('Discretization type {} not supported'.format(distype))

        return new
