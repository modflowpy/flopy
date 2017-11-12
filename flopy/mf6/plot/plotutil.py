"""
Module containing helper functions for plotting model data
using ModelMap and ModelCrossSection. Functions for plotting
shapefiles are also included.

"""
from __future__ import print_function
import os
import sys
import math
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from ..utils.binaryfile import VertexDisFile, StructuredDisFile


bc_color_dict = {'default': 'black', 'WEL': 'red', 'DRN': 'yellow',
                 'RIV': 'green', 'GHB': 'cyan', 'CHD': 'navy',
                 'STR': 'purple'}


def _plot_array_helper(plotarray, model=None, sr=None, axes=None,
                       names=None, filenames=None, fignum=None,
                       mflay=None, **kwargs):
    try:
        import matplotlib.pyplot as plt
    except:
        s = 'Could not import matplotlib.  Must install matplotlib ' +\
            ' in order to plot LayerFile data.'
        raise Exception(s)

    import flopy.plot.map as map
    

    # reshape 2d arrays to 3d for convenience
    if len(plotarray.shape) == 2:
        plotarray = plotarray.reshape((1, plotarray.shape[0],
                                       plotarray.shape[1]))

    # parse keyword arguments
    if 'figsize' in kwargs:
        figsize = kwargs.pop('figsize')
    else:
        figsize = None

    if 'masked_values' in kwargs:
        masked_values = kwargs.pop('masked_values')
    else:
        masked_values = None

    if 'pcolor' in kwargs:
        pcolor = kwargs.pop('pcolor')
    else:
        pcolor = True

    if 'inactive' in kwargs:
        inactive = kwargs.pop('inactive')
    else:
        inactive = True

    if 'contour' in kwargs:
        contourdata = kwargs.pop('contour')
    else:
        contourdata = False

    if 'clabel' in kwargs:
        clabel = kwargs.pop('clabel')
    else:
        clabel = False

    if 'colorbar' in kwargs:
        cb = kwargs.pop('colorbar')
    else:
        cb = False

    if 'grid' in kwargs:
        grid = kwargs.pop('grid')
    else:
        grid = False

    if 'levels' in kwargs:
        levels = kwargs.pop('levels')
    else:
        levels = None

    if 'colors' in kwargs:
        colors = kwargs.pop('colors')
    else:
        colors = 'black'
    
    if 'dpi' in kwargs:
        dpi = kwargs.pop('dpi')
    else:
        dpi = None
    
    if 'fmt' in kwargs:
        fmt = kwargs.pop('fmt')
    else:
        fmt = '%1.3f'
    
    if mflay is not None:
        i0 = int(mflay)
        if i0+1 >= plotarray.shape[0]:
            i0 = plotarray.shape[0] - 1
        i1 = i0 + 1
    else:
        i0 = 0
        i1 = plotarray.shape[0]
    
    if names is not None:
        if not isinstance(names, list):
            names = [names]
        assert len(names) == plotarray.shape[0]
    
    if filenames is not None:
        if not isinstance(filenames, list):
            filenames = [filenames]
        assert len(filenames) == plotarray.shape[0]
    
    if fignum is not None:
        if not isinstance(fignum, list):
            fignum = [fignum]
        assert len(fignum) == plotarray.shape[0]
        # check for existing figures
        f0 = fignum[0]
        for i in plt.get_fignums():
            if i >= f0:
                f0 = i + 1
        finc = f0 - fignum[0]
        for idx in range(len(fignum)):
            fignum[idx] += finc
    else:
        #fignum = np.arange(i0, i1)
        # check for existing figures
        f0 = 0
        for i in plt.get_fignums():
            if i >= f0:
                f0 += 1
        f1 = f0 + (i1 - i0)
        fignum = np.arange(f0, f1)


    if axes is not None:
        if not isinstance(axes, list):
            axes = [axes]
        assert len(axes) == plotarray.shape[0]
    # prepare some axis objects for use
    else:
        axes = []
        for idx, k in enumerate(range(i0, i1)):
            fig = plt.figure(figsize=figsize, num=fignum[idx])
            ax = plt.subplot(1, 1, 1, aspect='equal')
            if names is not None:
                title = names[k]
            else:
                klay = k
                if mflay is not None:
                    klay = int(mflay)
                title = '{} Layer {}'.format('data', klay+1)
            ax.set_title(title)
            axes.append(ax)
   
    for idx, k in enumerate(range(i0, i1)):
        fig = plt.figure(num=fignum[idx])
        mm = map.ModelMap(ax=axes[idx], model=model, sr=sr, layer=k)
        if pcolor:
            cm = mm.plot_array(plotarray[k], masked_values=masked_values,
                               ax=axes[idx], **kwargs)
            if cb:
                label = ''
                if not isinstance(cb,bool):
                    label = str(cb)
                plt.colorbar(cm, ax=axes[idx], shrink=0.5,label=label)

        if contourdata:
            cl = mm.contour_array(plotarray[k], masked_values=masked_values,
                                  ax=axes[idx], colors=colors, levels=levels, **kwargs)
            if clabel:
                axes[idx].clabel(cl, fmt=fmt,**kwargs)

        if grid:
            mm.plot_grid(ax=axes[idx])

        if inactive:
            try:
                ib = model.bas6.ibound.array
                mm.plot_inactive(ibound=ib, ax=axes[idx])
            except:
                pass

    if len(axes) == 1:
        axes = axes[0]
    if filenames is not None:
        for idx, k in enumerate(range(i0, i1)):
            fig = plt.figure(num=fignum[idx])
            fig.savefig(filenames[idx], dpi=dpi)
            print('    created...{}'.format(os.path.basename(filenames[idx])))
        # there will be nothing to return when done
        axes = None
        plt.close('all')
    return axes


def _plot_bc_helper(package, kper,
                    axes=None, names=None, filenames=None, fignum=None,
                    mflay=None, **kwargs):
    try:
        import matplotlib.pyplot as plt
    except:
        s = 'Could not import matplotlib.  Must install matplotlib ' +\
            ' in order to plot boundary condition data.'
        raise Exception(s)

    import flopy.plot.map as map

    # reshape 2d arrays to 3d for convenience
    ftype = package.name[0]

    nlay = package.parent.nlay

    # parse keyword arguments
    if 'figsize' in kwargs:
        figsize = kwargs.pop('figsize')
    else:
        figsize = None

    if 'inactive' in kwargs:
        inactive = kwargs.pop('inactive')
    else:
        inactive = True

    if 'grid' in kwargs:
        grid = kwargs.pop('grid')
    else:
        grid = False

    if 'dpi' in kwargs:
        dpi = kwargs.pop('dpi')
    else:
        dpi = None

    if 'masked_values' in kwargs:
        kwargs.pop('masked_values ')

    if mflay is not None:
        i0 = int(mflay)
        if i0+1 >= nlay:
            i0 = nlay - 1
        i1 = i0 + 1
    else:
        i0 = 0
        i1 = nlay

    if names is not None:
        if not isinstance(names, list):
            names = [names]
        assert len(names) == nlay

    if filenames is not None:
        if not isinstance(filenames, list):
            filenames = [filenames]
        assert len(filenames) == (i1 - i0)

    if fignum is not None:
        if not isinstance(fignum, list):
            fignum = [fignum]
        assert len(fignum) == (i1 - i0)
        # check for existing figures
        f0 = fignum[0]
        for i in plt.get_fignums():
            if i >= f0:
                f0 = i + 1
        finc = f0 - fignum[0]
        for idx in range(len(fignum)):
            fignum[idx] += finc
    else:
        #fignum = np.arange(i0, i1)
        # check for existing figures
        f0 = 0
        for i in plt.get_fignums():
            if i >= f0:
                f0 += 1
        f1 = f0 + (i1 - i0)
        fignum = np.arange(f0, f1)

    if axes is not None:
        if not isinstance(axes, list):
            axes = [axes]
        assert len(axes) == i1 - i0
    # prepare some axis objects for use
    else:
        axes = []
        for idx, k in enumerate(range(i0, i1)):
            fig = plt.figure(figsize=figsize, num=fignum[idx])
            ax = plt.subplot(1, 1, 1, aspect='equal')
            if names is not None:
                title = names[k]
            else:
                klay = k
                if mflay is not None:
                    klay = int(mflay)
                title = '{} Layer {}'.format('data', klay+1)
            ax.set_title(title)
            axes.append(ax)

    for idx, k in enumerate(range(i0, i1)):
        mm = map.ModelMap(ax=axes[idx], model=package.parent, layer=k)
        fig = plt.figure(num=fignum[idx])
        qm = mm.plot_bc(ftype=ftype, package=package, kper=kper, ax=axes[idx])

        if grid:
            mm.plot_grid(ax=axes[idx])

        if inactive:
            try:
                ib = package.parent.bas6.ibound.array
                mm.plot_inactive(ibound=ib, ax=axes[idx])
            except:
                pass

    if len(axes) == 1:
        axes = axes[0]

    if filenames is not None:
        for idx, k in enumerate(range(i0, i1)):
            fig = plt.figure(num=fignum[idx])
            fig.savefig(filenames[idx], dpi=dpi)
            plt.close(fignum[idx])
            print('    created...{}'.format(os.path.basename(filenames[idx])))
        # there will be nothing to return when done
        axes = None
        plt.close('all')
    return axes


class SwiConcentration():
    """
    The binary_header class is a class to create headers for MODFLOW
    binary files

    """
    def __init__(self, model=None, botm=None, istrat=1, nu=None):
        if model is None:
            if isinstance(botm, list):
                botm = np.array(botm)
            self.__botm = botm
            if isinstance(nu, list):
                nu = np.array(nu)
            self.__nu = nu
            self.__istrat = istrat
            if istrat == 1:
                self.__nsrf = self.nu.shape - 1
            else:
                self.__nsrf = self.nu.shape - 2
        else:
            try:
                dis = model.get_package('DIS')
            except:
                sys.stdout.write('Error: DIS package not available.\n')
            self.__botm = np.zeros((dis.nlay+1, dis.nrow, dis.ncol), np.float)
            self.__botm[0, :, :] = dis.top.array
            self.__botm[1:, :, :] = dis.botm.array
            try:
                swi = model.get_package('SWI2')
                self.__nu = swi.nu.array
                self.__istrat = swi.istrat
                self.__nsrf = swi.nsrf
            except:
                sys.stdout.write('Error: SWI2 package not available...\n')
        self.__nlay = self.__botm.shape[0] - 1
        self.__nrow = self.__botm[0, :, :].shape[0]
        self.__ncol = self.__botm[0, :, :].shape[1]
        self.__b = self.__botm[0:-1, :, :] - self.__botm[1:, :, :] 
     
    def calc_conc(self, zeta, layer=None):
        """
        Calculate concentrations for a given time step using passed zeta.

        Parameters
        ----------
        zeta : dictionary of numpy arrays
            Dictionary of zeta results. zeta keys are zero-based zeta surfaces.
        layer : int
            Concentration will be calculated for the specified layer.  If layer 
            is None, then the concentration will be calculated for all layers. 
            (default is None).

        Returns
        -------
        conc : numpy array
            Calculated concentration.

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow.load('test')
        >>> c = flopy.plot.SwiConcentration(model=m)
        >>> conc = c.calc_conc(z, layer=0)

        """
        conc = np.zeros((self.__nlay, self.__nrow, self.__ncol), np.float)
        
        pct = {}
        for isrf in range(self.__nsrf):
            z = zeta[isrf]
            pct[isrf] = (self.__botm[:-1, :, :] - z[:, :, :]) / self.__b[:, :, :]
        for isrf in range(self.__nsrf):
            p = pct[isrf]
            if self.__istrat == 1:
                conc[:, :, :] += self.__nu[isrf] * p[:, :, :]
                if isrf+1 == self.__nsrf:
                    conc[:, :, :] += self.__nu[isrf+1] * (1. - p[:, :, :])
            #TODO linear option
        if layer is None:
            return conc
        else:
            return conc[layer, :, :]



def shapefile_extents(shp):
    """
    Determine the extents of a shapefile

    Parameters
    ----------
    shp : string
        Name of the shapefile to convert to a PatchCollection.

    Returns
    -------
    extents : tuple
        tuple with xmin, xmax, ymin, ymax from shapefile.

    Examples
    --------

    >>> import flopy
    >>> fshp = 'myshapefile'
    >>> extent = flopy.plot.plotutil.shapefile_extents(fshp)

    """
    try:
        import shapefile
    except:
        s = 'Could not import shapefile.  Must install pyshp in order to plot shapefiles.'
        raise Exception(s)
    sf = shapefile.Reader(shp)
    shapes = sf.shapes()
    nshp = len(shapes)
    xmin, xmax, ymin, ymax = 1.e20, -1.e20, 1.e20, -1.e20
    ptchs = []
    for n in range(nshp):
        for p in shapes[n].points:
            xmin, xmax = min(xmin, p[0]), max(xmax, p[0])
            ymin, ymax = min(ymin, p[1]), max(ymax, p[1])
    return xmin, xmax, ymin, ymax


def shapefile_get_vertices(shp):
    """
    Get vertices for the features in a shapefile

    Parameters
    ----------
    shp : string
        Name of the shapefile to extract shapefile feature vertices.

    Returns
    -------
    vertices : list
        Vertices is a list with vertices for each feature in the shapefile. 
        Individual feature vertices are x, y tuples and contained in a list.
        A list with a single x, y tuple is returned for point shapefiles. A
        list with multiple x, y tuples is returned for polyline and polygon
        shapefiles.

    Examples
    --------

    >>> import flopy
    >>> fshp = 'myshapefile'
    >>> lines = flopy.plot.plotutil.shapefile_get_vertices(fshp)
    
    """
    try:
        import shapefile
    except:
        s = 'Could not import shapefile.  Must install pyshp in order to plot shapefiles.'
        raise Exception(s)
    sf = shapefile.Reader(shp)
    shapes = sf.shapes()
    nshp = len(shapes)
    vertices = []
    for n in range(nshp):
        st = shapes[n].shapeType
        if st in [1, 8, 11, 21]:
            #points
            for p in shapes[n].points:
                vertices.append([(p[0], p[1])])
        elif st in [3, 13, 23]:
            #line
            line = []
            for p in shapes[n].points:
                line.append((p[0], p[1]))
            line = np.array(line)
            vertices.append(line)
        elif st in [5, 25, 31]:
            #polygons
            pts = np.array(shapes[n].points)
            prt = shapes[n].parts
            par = list(prt) + [pts.shape[0]]
            for pij in range(len(prt)):
                vertices.append(pts[par[pij]:par[pij+1]])
    return vertices
    

def shapefile_to_patch_collection(shp, radius=500.):
    """
    Create a patch collection from the shapes in a shapefile

    Parameters
    ----------
    shp : string
        Name of the shapefile to convert to a PatchCollection.
    radius : float
        Radius of circle for points in the shapefile.  (Default is 500.)

    """
    try:
        import shapefile
    except:
        s = 'Could not import shapefile.  Must install pyshp in order to plot shapefiles.'
        raise Exception(s)
    from matplotlib.patches import Polygon, Circle, Path, PathPatch
    from matplotlib.collections import PatchCollection
    sf = shapefile.Reader(shp)
    shapes = sf.shapes()
    nshp = len(shapes)
    ptchs = []
    for n in range(nshp):
        st = shapes[n].shapeType
        if st in [1, 8, 11, 21]:
            #points
            for p in shapes[n].points:
                ptchs.append(Circle( (p[0], p[1]), radius=radius))
        elif st in [3, 13, 23]:
            #line
            vertices = []
            for p in shapes[n].points:
                vertices.append([p[0], p[1]])
            vertices = np.array(vertices)
            path = Path(vertices)
            ptchs.append(PathPatch(path, fill=False))
        elif st in [5, 25, 31]:
            #polygons
            pts = np.array(shapes[n].points)
            prt = shapes[n].parts
            par = list(prt) + [pts.shape[0]]
            for pij in range(len(prt)):
                ptchs.append(Polygon(pts[par[pij]:par[pij+1]]))
    pc = PatchCollection(ptchs)
    return pc

def plot_shapefile(shp, ax=None, radius=500., cmap='Dark2',
                   edgecolor='scaled', facecolor='scaled',
                   a=None, masked_values=None,
                   **kwargs):
    """
    Generic function for plotting a shapefile.

    Parameters
    ----------
    shp : string
        Name of the shapefile to plot.
    radius : float
        Radius of circle for points.  (Default is 500.)
    linewidth : float
        Width of all lines. (default is 1)
    cmap : string
        Name of colormap to use for polygon shading (default is 'Dark2')
    edgecolor : string
        Color name.  (Default is 'scaled' to scale the edge colors.)
    facecolor : string
        Color name.  (Default is 'scaled' to scale the face colors.)
    a : numpy.ndarray
        Array to plot.
    masked_values : iterable of floats, ints
        Values to mask.
    kwargs : dictionary
        Keyword arguments that are passed to PatchCollection.set(``**kwargs``).
        Some common kwargs would be 'linewidths', 'linestyles', 'alpha', etc.

    Returns
    -------
    pc : matplotlib.collections.PatchCollection

    Examples
    --------

    """

    try:
        import shapefile
    except:
        s = 'Could not import shapefile.  Must install pyshp in order to plot shapefiles.'
        raise Exception(s)
    import numpy as np
    import matplotlib.pyplot as plt

    if 'vmin' in kwargs:
        vmin = kwargs.pop('vmin')
    else:
        vmin = None

    if 'vmax' in kwargs:
        vmax = kwargs.pop('vmax')
    else:
        vmax = None

    if ax is None:
        ax = plt.gca()
    cm = plt.get_cmap(cmap)
    pc = shapefile_to_patch_collection(shp, radius=radius)
    pc.set(**kwargs)
    if a is None:
        nshp = len(pc.get_paths())
        cccol = cm(1. * np.arange(nshp) / nshp)
        if facecolor == 'scaled':
            pc.set_facecolor(cccol)
        else:
            pc.set_facecolor(facecolor)
        if edgecolor == 'scaled':
            pc.set_edgecolor(cccol)
        else:
            pc.set_edgecolor(edgecolor)
    else:
        pc.set_cmap(cm)
        if masked_values is not None:
            for mval in masked_values:
                a = np.ma.masked_equal(a, mval)
        if edgecolor == 'scaled':
            pc.set_edgecolor('none')
        else:
            pc.set_edgecolor(edgecolor)
        pc.set_array(a)
        pc.set_clim(vmin=vmin, vmax=vmax)
    # add the patch collection to the axis
    ax.add_collection(pc)
    return pc

def saturated_thickness(head, top, botm, mask_values=None):
    """
    Calculate the saturated thickness.

    Parameters
    ----------
    head : numpy.ndarray
        head array
    top : numpy.ndarray
        top array of shape (nrow, ncol)
    botm : numpy.ndarray
        botm array of shape (nlay, nrow, ncol)
    mask_values : list of floats
        If head is one of these values, then set sat to top - bot

    Returns
    -------
    sat_thk : numpy.ndarray
        Saturated thickness of shape (nlay, nrow, ncol).

    """
    # explanation: if confined (laytyp = 0) sat_thickness = top - botm,
    # else: if unconfined sat_thickness = head - botm if head < top
    nlay, ncpl = head.shape
    sat_thk = np.empty(head.shape, dtype=head.dtype)
    for k in range(nlay):
        if k == 0:
            t = top
        else:
            t = botm[k-1, :]
        sat_thk[k, :] = t - botm[k, :]
    for k in range(nlay):
        dh = np.zeros(ncpl, dtype=head.dtype)
        s = sat_thk[k, :]

        for mv in mask_values:
            idx = (head[k, :] == mv)
            dh[idx] = s[idx]

        if k == 0:
            t = top
        else:
            t = botm[k-1, :]
        t = np.where(head[k, :] > t, t, head[k, :])
        dh = np.where(dh == 0, t - botm[k, :], dh)
        sat_thk[k, :] = dh[:]
    return sat_thk

def specific_discharge(Qx, Qy, Qz, delr, delc, sat_thk):
    """
    Using the MODFLOW discharge, calculate the cell centered specific discharge
    by dividing by the flow width and then averaging to the cell center.

    Parameters
    ----------
    Qx : numpy.ndarray
        MODFLOW 'flow right face'
    Qy : numpy.ndarray
        MODFLOW 'flow front face'.  The sign on this array will be flipped
        by this function so that the y axis is positive to north.
    Qz : numpy.ndarray
        MODFLOW 'flow lower face'.  The sign on this array will be flipped by
        this function so that the z axis is positive in the upward direction.
    delr : numpy.ndarray
        MODFLOW delr array
    delc : numpy.ndarray
        MODFLOW delc array
    sat_thk : numpy.ndarray
        Saturated thickness for each cell

    Returns
    -------
    (qx, qy, qz) : tuple of numpy.ndarrays
        Specific discharge arrays that have been interpolated to cell centers.

    """
    qx = None
    qy = None
    qz = None

    if Qx is not None:

        if len(Qx.shape) != 1:
            nlay, ncpl = Qx.shape
        else:
            nlay = 1
            ncpl = Qx.shape[0]

        qx = np.zeros(Qx.shape, dtype=Qx.dtype)

        qx = Qx/(delc * sat_thk)
        qx = -qx

    if Qy is not None:

        if len(Qy.shape) !=1:
            nlay, ncpl = Qy.shape
        else:
            nlay = 1
            ncpl = Qy.shape[0]

        qy = np.zeros(Qy.shape, dtype=Qy.dtype)

        qy = Qy / (delr * sat_thk)

        qy = -qy


    if Qz is not None:

        if len(Qz.shape) != 1:
            nlay, ncpl = Qz.shape
        else:
            nlay = 1
            ncpl = Qz.shape[0]

        qz = np.zeros(Qz.shape, dtype=Qz.dtype)

        qz = Qz / (delr * sat_thk)

        qz = -qz

    return (qx, qy, qz)

def vectorize_flow(ja, dis):
    """
    Simple method to take discretization info. and a FLOW JA FACE array
    and create 1d list of fluid vectors in the flow right face, and flow front
    face directions. Used with the quiver plot. Can be extended to return cell
    averaged flow in vector directions as well.

    Parameters
    ----------
    ja: ndarray:
        1d flow ja face array from the cell by cell flow file
    dis: (object)
        Dis object <StructuredDisFile> or <VertexDisFile>

    Returns
    -------
    frf_arr = array of flow right face values
    fff_arr = array of flow forward face values
    """

    frf_arr = []
    fff_arr = []
    flf_arr = []

    disja = dis.ja - 1 # create a list based indexing option
    ia = dis.ia
    nlay = dis.nlay
    zcenter = dis.zcenter_array

    if isinstance(dis, VertexDisFile):
        xcenter = np.tile(dis.sr.xcenter_array, nlay)
        ycenter = np.tile(dis.sr.ycenter_array, nlay)
        ncpl = dis.sr.ncpl

    elif isinstance(dis, StructuredDisFile):
        ncpl = dis.nrow * dis.ncol
        xcenter = dis.sr.xcentergrid
        ycenter = dis.sr.ycentergrid
        xcenter = np.tile(xcenter.reshape(-1), nlay)
        ycenter = np.tile(ycenter.reshape(-1), nlay)

    else:
        raise AssertionError('distype not supported by quiver function')

    con_arr = []
    flux_arr = []
    for i in range(1, len(ia)):
        lji = ia[i-1]
        uji = ia[i] - 1
        con_arr.append(disja[lji:uji])
        flux_arr.append(ja[lji:uji])


    xcon_arr = []
    ycon_arr = []
    zcon_arr = []
    xy_angle_arr = []
    xz_angle_arr = []
    for i, j in enumerate(con_arr):
        xtmp = xcenter[j] - xcenter[i]
        ytmp = ycenter[j] - ycenter[i]
        ztmp = zcenter[j] - zcenter[i]
        compare = xtmp + ytmp
        xtmp[compare == 0.] = np.nan
        ytmp[compare == 0.] = np.nan
        ztmp[ztmp == 0.] = np.nan
        xcon_arr.append(xtmp)
        ycon_arr.append(ytmp)
        xy_angle_arr.append(np.arctan2(ytmp, xtmp) * -180 / np.pi)
        xz_angle_arr.append(np.arctan2(ztmp, xtmp) * -180 / np.pi)


    for i, cell in enumerate(xy_angle_arr):
        frf, fff = 0., 0.
        for j, angle in enumerate(cell):
            if angle < 0.:
                angle += 360

            if 0. <= angle < 90:
                frf += flux_arr[i][j] * ((90. - angle)/90.)
                fff += flux_arr[i][j] * (angle/90.)

            elif 90. <= angle < 180.:
                fff += flux_arr[i][j] * ((180. - angle)/90.)

            elif 270 < angle < 360.:
                frf += flux_arr[i][j] * ((angle - 270.)/90.)

            else:
                pass
        frf_arr.append(frf)
        fff_arr.append(fff)

    # todo: debug this relationship
    for i, cell in enumerate(xz_angle_arr):
        flf = 0.
        for j, angle in enumerate(cell):
            if 0. <= angle < 90:
                flf += flux_arr[i][j] * ((90. - angle)/90.)

            else:
                pass
        flf_arr.append(flf)

    return np.array(frf_arr), np.array(fff_arr), np.array(flf_arr)

def findrowcolumn(pt, xedge, yedge):
    """
    Find the MODFLOW cell containing the x- and y- point provided.

    Parameters
    ----------
    pt : list or tuple
        A list or tuple containing a x- and y- coordinate
    xedge : numpy.ndarray
        x-coordinate of the edge of each MODFLOW column. xedge is dimensioned
        to NCOL + 1. If xedge is not a numpy.ndarray it is converted to a
        numpy.ndarray.
    yedge : numpy.ndarray
        y-coordinate of the edge of each MODFLOW row. yedge is dimensioned
        to NROW + 1. If yedge is not a numpy.ndarray it is converted to a
        numpy.ndarray.

    Returns
    -------
    irow, jcol : int
        Row and column location containing x- and y- point passed to function.

    Examples
    --------
    >>> import flopy
    >>> irow, jcol = flopy.plotutil.findrowcolumn(pt, xedge, yedge)

    """

    # make sure xedge and yedge are numpy arrays
    if not isinstance(xedge, np.ndarray):
        xedge = np.array(xedge)
    if not isinstance(yedge, np.ndarray):
        yedge = np.array(yedge)

    # find column
    jcol = -100
    for jdx, xmf in enumerate(xedge):
        if xmf > pt[0]:
            jcol = jdx - 1
            break

    # find row
    irow = -100
    for jdx, ymf in enumerate(yedge):
        if ymf < pt[1]:
            irow = jdx - 1
            break
    return irow, jcol


def line_intersect_grid(ptsin, xedge, yedge, returnvertices=False):
    """
    Intersect a list of polyline vertices with a rectilinear MODFLOW
    grid. Vertices at the intersection of the polyline with the grid
    cell edges is returned. Optionally the original polyline vertices
    are returned.

    Parameters
    ----------
    ptsin : list
        A list of x, y points defining the vertices of a polyline that will be
        intersected with the rectilinear MODFLOW grid
    xedge : numpy.ndarray
        x-coordinate of the edge of each MODFLOW column. xedge is dimensioned
        to NCOL + 1. If xedge is not a numpy.ndarray it is converted to a
        numpy.ndarray.
    yedge : numpy.ndarray
        y-coordinate of the edge of each MODFLOW row. yedge is dimensioned
        to NROW + 1. If yedge is not a numpy.ndarray it is converted to a
        numpy.ndarray.
    returnvertices: bool
        Return the original polyline vertices in the list of numpy.ndarray
        containing vertices resulting from intersection of the provided
        polygon and the MODFLOW model grid if returnvertices=True.
        (default is False).

    Returns
    -------
    (x, y, dlen) : numpy.ndarray of tuples
        numpy.ndarray of tuples containing the x, y, and segment length of the
        intersection of the provided polyline with the rectilinear MODFLOW
        grid.

    Examples
    --------
    >>> import flopy
    >>> ptsout = flopy.plotutil.line_intersect_grid(ptsin, xedge, yedge)

    """

    small_value = 1.0e-1

    # make sure xedge and yedge are numpy arrays
    if not isinstance(xedge, np.ndarray):
        xedge = np.array(xedge)
    if not isinstance(yedge, np.ndarray):
        yedge = np.array(yedge)

    # build list of points along current line
    pts = []
    npts = len(ptsin)
    dlen = 0.
    for idx in range(1, npts):
        x0 = ptsin[idx - 1][0]
        x1 = ptsin[idx][0]
        y0 = ptsin[idx - 1][1]
        y1 = ptsin[idx][1]
        a = x1 - x0
        b = y1 - y0
        c = math.sqrt(math.pow(a, 2.) + math.pow(b, 2.))
        # find cells with (x0, y0) and (x1, y1)
        irow0, jcol0 = findrowcolumn((x0, y0), xedge, yedge)
        irow1, jcol1 = findrowcolumn((x1, y1), xedge, yedge)
        # determine direction to go in the x- and y-directions
        jx = 0
        incx = abs(small_value * a / c)
        iy = 0
        incy = -abs(small_value * b / c)
        if a == 0.:
            incx = 0.
        # go to the right
        elif a > 0.:
            jx = 1
            incx *= -1.
        if b == 0.:
            incy = 0.
        # go down
        elif b < 0.:
            iy = 1
            incy *= -1.
        # process data
        if irow0 >= 0 and jcol0 >= 0:
            iadd = True
            if idx > 1 and returnvertices: 
                iadd = False
            if iadd: 
                pts.append((x0, y0, dlen))
        icnt = 0
        while True:
            icnt += 1
            dx = xedge[jcol0 + jx] - x0
            dlx = 0.
            if a != 0.:
                dlx = c * dx / a
            dy = yedge[irow0 + iy] - y0 # changed to irow1 from irow0. irow0 caused it to crash!
            dly = 0.
            if b != 0.:
                dly = c * dy / b
            if dlx != 0. and dly != 0.:
                if abs(dlx) < abs(dly):
                    dy = dx * b / a
                else:
                    dx = dy * a / b
            xt = x0 + dx + incx
            yt = y0 + dy + incy
            dl = math.sqrt(math.pow((xt - x0), 2.) + math.pow((yt - y0), 2.))
            dlen += dl
            if not returnvertices: 
                pts.append((xt, yt, dlen))
            x0, y0 = xt, yt
            xt = x0 - 2. * incx
            yt = y0 - 2. * incy
            dl = math.sqrt(math.pow((xt - x0), 2.) + math.pow((yt - y0), 2.))
            dlen += dl
            x0, y0 = xt, yt
            irow0, jcol0 = findrowcolumn((x0, y0), xedge, yedge)
            if irow0 >= 0 and jcol0 >= 0:
                if not returnvertices: 
                    pts.append((xt, yt, dlen))
            elif irow1 < 0 or jcol1 < 0:
                dl = math.sqrt(math.pow((x1 - x0), 2.) + math.pow((y1 - y0), 2.))
                dlen += dl
                break
            if irow0 == irow1 and jcol0 == jcol1:
                dl = math.sqrt(math.pow((x1 - x0), 2.) + math.pow((y1 - y0), 2.))
                dlen += dl
                pts.append((x1, y1, dlen))
                break
    return np.array(pts)


def line_intersect_vertex_grid(pts, xyvdict):
    """
    Find the cells and verticies of intersection from a list of lines defined by
    the pts variable
    Parameters
    ----------
    pts: ndarray [(x,y), (x1, y1), (xn, yn)] list of points defining an arbitraty cross section
    xyvdict: (dict) dictionary of 2d lists associated with a cells xy vertices

    Returns
    -------

    """
    def get_line_intersection(lsline, cverts, ls):
        """
        Finds the intersection of two line segments if it exists.

        Parameters
        ----------
        lsline: [a, b, c] point slope equation of line segment
        cverts: [(x, y), (x1, y1)] set of verticies that define a cell edge
        ls: [(x, y), (x1, y1)] vertices that define the edges of the line segment

        Returns
        -------
        intersection: (dict) {1: [(x,y), (x1,y1)]}
        """
        from scipy.linalg import lu
        verts = None
        a1, b1, c1 = lsline
        a2, b2, c2 = get_line(cverts)
        matrix = [[a1, b1, c1],
                  [a2, b2, c2]]

        P, L, U = lu(matrix)

        r0, r1 = U.T[0]
        s0, s1 = U.T[1]
        c0, c1 = U.T[2]

        y = c1 / s1
        x = (c0 - (s0 * y)) / r0

        if np.min(cverts[:, 0]) <= x <= np.max(cverts[:, 0]):
            if np.min(cverts[:, 1]) <= y <= np.max(cverts[:, 1]):
                if np.min(ls[:, 0]) <= x <= np.max(ls[:, 0]):
                    if np.min(ls[:, 1]) <= y <= np.max(ls[:, 0]):
                        return x, y
        return

    def get_set(verts):
        verts = [i for i in verts if i is not None]
        verts = list(set(verts))
        if len(verts) > 1:
            return verts
        else:
            return

    def get_line(verts):
        """
        Gets the point slope form of a line
        Parameters
        ----------
        verts: verticies that define a line segment

        Returns
        -------
        a, b, c  in line form of ax + by = c
        """

        x2 = float(verts[1][0])
        x1 = float(verts[0][0])
        y2 = float(verts[1][1])
        y1 = float(verts[0][1])
        if x2 - x1 == 0.:
            a = 1
            b = 0
            c = x2
        elif y2 - y1 == 0.:
            a = 0
            b = 1
            c = y2
        else:
            a = - (y2 - y1) / (x2 - x1)
            b = 1
            c = y2 + (a * x2)
        return a, b, c

    intersection = {}
    for idx in range(1, len(pts)):
        ls = np.array([pts[idx - 1], pts[idx]])
        lxmax = np.max(ls[:, 0])
        lxmin = np.min(ls[:, 0])
        lymax = np.max(ls[:, 1])
        lymin = np.min(ls[:, 1])
        lsline = get_line(ls)

        for cellnum, cell in xyvdict.items():  # python 3 this is d.items()
            cxmax = np.max(cell[:, 0])
            cxmin = np.min(cell[:, 0])
            cymax = np.max(cell[:, 1])
            cymin = np.min(cell[:, 1])
            if cxmax < lxmin or cxmin > lxmax:
                pass
            else:
                if cymax < lymin or cymin > lymax:
                    pass
                else:
                    # compute line intersection and evaluate if line intersects
                    # within defined section of cell
                    verts = []
                    for idx in range(0, len(cell)):
                        cverts = np.array([cell[idx - 1], cell[idx]])
                        verts.append(get_line_intersection(lsline, cverts, ls))
                    verts = get_set(verts)
                    if verts is not None:
                        intersection[cellnum] = verts

    return intersection


def cell_value_points(pts, xedge, yedge, vdata):
    """
    Intersect a list of polyline vertices with a rectilinear MODFLOW
    grid. Vertices at the intersection of the polyline with the grid
    cell edges is returned. Optionally the original polyline vertices
    are returned.

    Parameters
    ----------
    pts : list
        A list of x, y points and polyline length to extract defining the
        vertices of a polyline that
    xedge : numpy.ndarray
        x-coordinate of the edge of each MODFLOW column. The shape of xedge is
        (NCOL + 1). If xedge is not a numpy.ndarray it is converted to a
        numpy.ndarray.
    yedge : numpy.ndarray
        y-coordinate of the edge of each MODFLOW row. The shape of yedge is
        (NROW + 1). If yedge is not a numpy.ndarray it is converted to a
        numpy.ndarray.
    vdata : numpy.ndarray
        Data (i.e., head, hk, etc.) for a rectilinear MODFLOW model grid. The
        shape of vdata is (NROW, NCOL). If vdata is not a numpy.ndarray it is
        converted to a numpy.ndarray.

    Returns
    -------
    vcell : numpy.ndarray
        numpy.ndarray of of data values from the vdata numpy.ndarray at x- and
        y-coordinate locations in pts.

    Examples
    --------
    >>> import flopy
    >>> vcell = flopy.plotutil.cell_value_points(xpts, xedge, yedge, head[0, :, :])

    """

    # make sure xedge and yedge are numpy arrays
    if not isinstance(xedge, np.ndarray):
        xedge = np.array(xedge)
    if not isinstance(yedge, np.ndarray):
        yedge = np.array(yedge)
    if not isinstance(vdata, np.ndarray):
        vdata = np.array(vdata)

    vcell = []
    for idx, [xt, yt, dlen] in enumerate(pts):
        # find the modflow cell containing point
        irow, jcol = findrowcolumn((xt, yt), xedge, yedge)
        if irow >= 0 and jcol >= 0:
            if np.isnan(vdata[irow, jcol]):
                vcell.append(np.nan)
            else:
                v = np.asarray(vdata[irow, jcol])
                vcell.append(v) 

    return np.array(vcell)


def cell_value_points_from_dict(xpts, elev):
    vdict = {}
    for idx, value in xpts.items():
        vdict[idx] = elev[idx]

    return vdict


def arctan2(verts):
    """
    Reads 2 dimensional set of verts and orders them using the arctan 2 method
    Parameters
    ----------
    verts: (np.array, float) Nx2 array of verts

    Returns
    -------
    verts: (np.array, float) Nx2 array of verts
    """
    center = verts.mean(axis=0)
    x = verts.T[0] - center[0]
    z = verts.T[1] - center[1]

    angles = np.arctan2(z, x) * 180 / np.pi
    angleidx = angles.argsort()

    verts = verts[angleidx]
    return verts