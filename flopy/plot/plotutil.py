"""
Module to post-process MODFLOW binary output.  The module contains one
important classes that can be accessed by the user.

*  SwiConcentration (Process Zeta results to concentrations)

"""
import sys
import numpy as np

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

def centered_specific_discharge(Qx, Qy, Qz, delr, delc, botm):
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
    botm : numpy.ndarray
        MODFLOW botm array

    Returns
    -------
    (qx, qy, qz) : tuple of numpy.ndarrays
        Specific discharge arrays that have been interpolated to cell centers.

    NOT FINISHED YET!  STILL NEED TO
    1.  CALCULATE QZ
    2.  DIVIDE qx AND qy BY THE SATURATED THICKNESS
    3.  ROTATE FOR MAP PRESENTATION

    """
    qx = None
    qy = None
    qz = None

    if Qx is not None:

        nlay, nrow, ncol = Qx.shape
        qx = np.empty(Qx.shape, dtype=Qx.dtype)

        for k in xrange(nlay):
            for j in xrange(ncol):
                qx[k, :, j] = Qx[k, :, j] / delc[:]

        qx[:, :, 1:] = 0.5 * (qx[:, :, 0:ncol-1] + qx[:, :, 1:ncol])
        qx[:, :, 0] = 0.5 * qx[:, :, 0]

    if Qy is not None:

        nlay, nrow, ncol = Qy.shape
        qy = np.empty(Qy.shape, dtype=Qy.dtype)

        for k in xrange(nlay):
            for i in xrange(nrow):
                qy[k, i, :] = Qy[k, i, :] / delr[:]

        qy[:, 1:, :] = 0.5 * (qy[:, 0:nrow-1, :] + qx[:, 1:nrow, :])
        qy[:, 0, :] = 0.5 * qy[:, 0, :]


    if Qz is not None:
        raise NotImplementedError('Qz not implemented yet.  Sorry. Set Qz=None')
        qz = np.empty(Qz.shape, dtype=Qz.dtype)
    # qz_avg = np.empty(qz.shape, dtype=qz.dtype)
    # qz_avg[1:, :, :] = 0.5 * (qz[0:nlay-1, :, :] + qz[1:nlay, :, :])
    # qz_avg[0, :, :] = 0.5 * qz[0, :, :]

    return (qx, -qy, qz)