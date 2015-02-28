"""
Module to post-process MODFLOW binary output.  The module contains one
important classes that can be accessed by the user.

*  SwiConcentration (Process Zeta results to concentrations)

"""
import sys
import numpy as np
import matplotlib.pyplot as plt

def rotate(x, y, theta, xorigin=0., yorigin=0.):
    """
    Given x and y array-like values calculate the rotation about an
    arbitrary origin and then return the rotated coordinates.  theta is in
    radians.

    """
    xrot = xorigin + np.cos(theta) * (x - xorigin) - np.sin(theta) * (y - yorigin)
    yrot = yorigin + np.sin(theta) * (x - xorigin) + np.cos(theta) * (y - yorigin)
    return xrot, yrot


class MapPlanView(object):
    """
    Class to create a map of the model.

    Parameters
    ----------
    ax : matplotlib.pyplot axis
        The plot axis.  If not provided it, plt.gca() will be used.
    dis : flopy discretization object
    layer : int
        Layer to plot.  Default is 0.
    xul : float
        x coordinate for upper left corner
    yul : float
        y coordinate for upper left corner
    rotation : float
        Angle of grid rotation around the upper left corner.  A positive value
        indicates clockwise rotation.  Angles are in degrees.
    """
    def __init__(self, ax=None, dis=None, layer=0, xul=0., yul=0.,
                 rotation=0.):
        if ax is None:
            self.ax = plt.gca()
        else:
            self.ax = ax
        self.dis = dis
        self.layer = layer
        self.xul = xul
        self.yul = yul
        self.rotation = -rotation * np.pi / 180.
        self.xedge = self.get_xedge_array()
        self.yedge = self.get_yedge_array()
        self._grid_line_collection = None
        return

    def plot_grid(self, **kwargs):
        """
        Plot the grid lines on ax
        """
        if 'facecolors' not in kwargs:
            kwargs['facecolors'] = 'None'

        if 'edgecolors' not in kwargs:
            kwargs['edgecolors'] = 'black'

        if 'axes' not in kwargs:
            kwargs['axes'] = self.ax

        xgrid, ygrid = np.meshgrid(self.xedge, self.yedge)
        xgrid, ygrid = rotate(xgrid, ygrid, self.rotation, 0, self.yedge[0])
        xgrid += self.xul
        ygrid += self.yul - self.yedge[0]
        a = np.zeros(xgrid.shape)
        quadmesh = plt.pcolormesh(xgrid, ygrid, a, **kwargs)
        return quadmesh

    def get_xedge_array(self):
        """
        Return a numpy one-dimensional float array that has the cell edge x
        coordinates for every cell in the grid.  Array is of size (ncol + 1)

        """
        xedge = np.concatenate(([0.], np.add.accumulate(self.dis.delr.array)))
        return xedge

    def get_yedge_array(self):
        """
        Return a numpy one-dimensional float array that has the cell edge y
        coordinates for every cell in the grid.  Array is of size (nrow + 1)

        """
        length_y = np.add.reduce(self.dis.delc.array)
        yedge = np.concatenate(([length_y], length_y -
                             np.add.accumulate(self.dis.delc.array)))
        return yedge


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
              
