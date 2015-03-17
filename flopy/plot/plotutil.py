"""
Module to post-process MODFLOW binary output.  The module contains one
important classes that can be accessed by the user.

*  SwiConcentration (Process Zeta results to concentrations)

"""
import sys
import math
import numpy as np


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
        for isrf in xrange(self.__nsrf):
            z = zeta[isrf]
            pct[isrf] = (self.__botm[:-1, :, :] - z[:, :, :]) / self.__b[:, :, :]
        for isrf in xrange(self.__nsrf):
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
    for n in xrange(nshp):
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
            for pij in xrange(len(prt)):
                ptchs.append(Polygon(pts[par[pij]:par[pij+1]]))
    pc = PatchCollection(ptchs)
    return pc

def plot_shapefile(shp, ax=None, radius=500., cmap='Dark2',
                   edgecolor='scaled', facecolor='scaled', **kwargs):
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
    kwargs : dictionary
        Keyword arguments that are passed to PatchCollection.set(**kwargs).
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

    if ax is None:
        ax = plt.gca()
    cm = plt.get_cmap(cmap)
    pc = shapefile_to_patch_collection(shp, radius=radius)
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
    pc.set(**kwargs)
    ax.add_collection(pc)
    return pc

def saturated_thickness(head, top, botm, laytyp, mask_values=None):
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
    laytyp : numpy.ndarray
        confined (0) or convertible (1) of shape (nlay)
    mask_values : list of floats
        If head is one of these values, then set sat to top - bot

    Returns
    -------
    sat_thk : numpy.ndarray
        Saturated thickness of shape (nlay, nrow, ncol).

    """
    nlay, nrow, ncol = head.shape
    sat_thk = np.empty(head.shape, dtype=head.dtype)
    for k in xrange(nlay):
        if k == 0:
            t = top
        else:
            t = botm[k-1, :, :]
        sat_thk[k, :, :] = t - botm[k, :, :]
    for k in xrange(nlay):
        if laytyp[k] != 0:
            dh = np.zeros((nrow, ncol), dtype=head.dtype)
            s = sat_thk[k, :, :]

            for mv in mask_values:
                idx = (head[k, :, :] == mv)
                dh[idx] = s[idx]

            if k == 0:
                t = top
            else:
                t = botm[k-1, :, :]
            t = np.where(head[k, :, :] > t, t, head[k, :, :])
            dh = np.where(dh == 0, t - botm[k, :, :], dh)
            sat_thk[k, :, :] = dh[:, :]
    return sat_thk


def centered_specific_discharge(Qx, Qy, Qz, delr, delc, sat_thk):
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

        nlay, nrow, ncol = Qx.shape
        qx = np.zeros(Qx.shape, dtype=Qx.dtype)

        for k in xrange(nlay):
            for j in xrange(ncol-1):
                area = delc[:] * 0.5 * (sat_thk[k, :, j] + sat_thk[k, :, j + 1])
                qx[k, :, j] = Qx[k, :, j] / area

        qx[:, :, 1:] = 0.5 * (qx[:, :, 0:ncol-1] + qx[:, :, 1:ncol])
        qx[:, :, 0] = 0.5 * qx[:, :, 0]

    if Qy is not None:

        nlay, nrow, ncol = Qy.shape
        qy = np.zeros(Qy.shape, dtype=Qy.dtype)

        for k in xrange(nlay):
            for i in xrange(nrow-1):
                area = delr[:] * 0.5 * (sat_thk[k, i, :] + sat_thk[k, i + 1, :])
                qy[k, i, :] = Qy[k, i, :] / area

        qy[:, 1:, :] = 0.5 * (qy[:, 0:nrow-1, :] + qy[:, 1:nrow, :])
        qy[:, 0, :] = 0.5 * qy[:, 0, :]
        qy = -qy


    if Qz is not None:
        qz = np.zeros(Qz.shape, dtype=Qz.dtype)
        dr = delr.reshape((1, delr.shape[0]))
        dc = delc.reshape((delc.shape[0], 1))
        area = dr * dc
        for k in xrange(nlay):
            qz[k, :, :] = qz[k, :, :] / area[:, :]
        qz[1:, :, :] = 0.5 * (qz[0:nlay-1, :, :] + qz[1:nlay, :, :])
        qz[0, :, :] = 0.5 * qz[0, :, :]
        qz = -qz


    return (qx, qy, qz)
    


def findrowcolumn(pt, xedge, yedge):
    #--find the modflow cell containing the cross-section point
    jcol = -100
    for jdx, xmf in enumerate(xedge):
        if xmf > pt[0]:
            jcol = jdx - 1
            break
    irow = -100
    for jdx, ymf in enumerate(yedge):
        if ymf < pt[1]:
            irow = jdx - 1
            break
    return irow,jcol


def line_intersect_grid(ptsin, xedge, yedge, returnVertices=False):
    small_value = 1.0e-1
    
    #--build list of points along current line
    pts = []
    npts = len(ptsin)
    dlen = 0.
    for idx in xrange(1, npts):
        x0 = ptsin[idx-1][0]
        x1 = ptsin[idx][0]
        y0 = ptsin[idx-1][1]
        y1 = ptsin[idx][1]
        a  = x1 - x0
        b  = y1 - y0
        c  = math.sqrt( math.pow(a, 2.) + math.pow(b, 2.) )
        #--find cells with (x0, y0) and (x1, y1)
        irow0, jcol0 = findrowcolumn((x0, y0), xedge, yedge)
        irow1, jcol1 = findrowcolumn((x1, y1), xedge, yedge)
        #--determine direction to go in the x- and y-directions
        goRight = False
        jx   = 0
        incx =  abs( small_value * a / c )
        goDown = False
        iy   = 0
        incy = -abs(small_value * b / c )
        if a == 0.: 
            incx = 0.
        elif a > 0.: 
            goRight = True
            jx   = 1
            incx *= -1.
        if b == 0.: 
            incy = 0.
        elif b < 0.: 
            goDown = True
            iy   = 1
            incy *= -1.
        #--process data
        if irow0 >= 0 and jcol0 >= 0:
            iAdd = True
            if idx > 1 and returnVertices==True: iAdd = False
            if (iAdd==True): pts.append( (x0, y0, dlen) )
        icnt = 0
        while True:
            icnt += 1
            dx  = xedge[jcol0+jx] - x0
            dlx = 0.
            if a != 0.:
                dlx = c * dx / a
            dy  = yedge[irow0+iy] - y0
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
            dl = math.sqrt( math.pow((xt-x0),2.) + math.pow((yt-y0),2.) )
            dlen += dl
            if (returnVertices==False): pts.append( (xt, yt, dlen) )
            x0,y0 = xt,yt
            xt = x0 - 2. * incx
            yt = y0 - 2. * incy
            dl = math.sqrt( math.pow((xt-x0),2.) + math.pow((yt-y0), 2.) )
            dlen += dl
            x0,y0 = xt,yt
            irow0,jcol0 = findrowcolumn((x0, y0), xedge, yedge)
            if irow0 >= 0 and jcol0 >= 0:
                if (returnVertices==False): pts.append( (xt, yt, dlen) )
            elif irow1 < 0 or jcol1 < 0:
                dl = math.sqrt( math.pow((x1-x0), 2.) + math.pow((y1-y0), 2.) )
                dlen += dl
                break
            if irow0 == irow1 and jcol0 == jcol1:
                dl = math.sqrt( math.pow((x1-x0),2.) + math.pow((y1-y0), 2.) )
                dlen += dl
                pts.append( (x1, y1, dlen) )
                break
    return np.array(pts)


def cell_value_points(pts, xedge, yedge, vdata):
    vcell = []
    for idx, [xt, yt, dlen] in enumerate(pts):
        #--find the modflow cell containing point
        irow, jcol = findrowcolumn((xt, yt), xedge, yedge)
        if irow >= 0 and jcol >= 0:
            if np.isnan(vdata[irow, jcol]):
                vcell.append(np.nan)
            else:
                v = np.asarray(vdata[irow, jcol])
                vcell.append(v) 
    return np.array(vcell)
