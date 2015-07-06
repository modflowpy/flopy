"""
Module to read MODPATH output files.  The module contains three
important classes that can be accessed by the user.

*  PathlineFile (Binary head file.  Can also be used for drawdown)

"""

import numpy as np
from collections import OrderedDict

class PathlineFile():
    """
    PathlineFile Class.

    Parameters
    ----------
    filename : string
        Name of the concentration file
    verbose : bool
        Write information to the screen.  Default is False.

    Attributes
    ----------

    Methods
    -------

    See Also
    --------

    Notes
    -----
    The PathlineFile class provides simple ways to retrieve MODPATH 6
    pathline data from a MODPATH 6 ascii pathline file.

    Examples
    --------

    >>> import flopy.utils.modpathfile as mpf
    >>> pthobj = flopy.utils.PathlineFile('model.mppth')
    >>> p1 = pthobj.get_data(partid=1)


    """
    def __init__(self, filename, verbose=False):
        self.fname = filename
        self.dtype = self._get_dtype()
        self._build_index()
        self.data = np.loadtxt(self.file, dtype=self.dtype, skiprows=self.skiprows)
        #--set number of particle ids
        self.nid = self.data['particleid'].max()
        #--convert layer, row, and column indices; particle id and group; and
        #  line segment indices to zero-based
        self.data['k'] -= 1
        self.data['i'] -= 1
        self.data['j'] -= 1
        self.data['particleid'] -= 1
        self.data['particlegroup'] -= 1
        self.data['linesegmentindex'] -= 1
        return
        
    def _build_index(self):
        """
           Set position of the start of the pathline data.
        """
        self.skiprows = 0
        self.file = open(self.fname, 'r')
        while True:
            line = self.file.readline()
            self.skiprows += 1
            if 'end header' in line.lower():
                break
        self.file.seek(0)

    def _get_dtype(self):
        """
           Build numpy dtype for the MODPATH 6 pathline file.
        """
        dtype = np.dtype([("particleid", np.int), ("particlegroup", np.int),
                          ("timepointindex", np.int), ("comulativetimestep", np.int),
                          ("time", np.float32), ("x", np.float32),
                          ("y", np.float32), ("z", np.float32),
                          ("k", np.int), ("i", np.int), ("j", np.int),
                          ("grid", np.int), ("xloc", np.float32), 
                          ("yloc", np.float32), ("zloc", np.float32),
                          ("linesegmentindex", np.int)])
        return dtype
                              

    def get_maxid(self):
        """
        Get the maximum pathline number in the file pathline file

        Returns
        ----------
        out : int
            Maximum pathline number.

        """
        return self.maxid

    def get_data(self, partid=0, totim=None):
        """
        get pathline data from the pathline file for a single pathline.

        Parameters
        ----------
        partid : int
            The zero-based particle id.  The first record is record 0.
        totim : float
            The simulation time. All pathline points for particle partid
            that are greater than or equal to totim will be returned

        Returns
        ----------
        ra : numpy record array
            A numpy recarray with the x, y, z, time, k, and particleid for
            pathline partid.


        See Also
        --------

        Notes
        -----

        Examples
        --------

        >>> import flopy.utils.modpathfile as mpf
        >>> pthobj = flopy.utils.PathlineFile('model.mppth')
        >>> p1 = pthobj.get_data(partid=1)

        """
        dtype = np.dtype([("x", np.float32), ("y", np.float32), ("z", np.float32),
                          ("time", np.float32), ("k", np.int), ("id", np.int)])
        if totim is not None:
            idx = self.data['particleid'] == partid & self.data['time'] > totime
        else:
            idx = self.data['particleid'] == partid
        ta = self.data[idx]
        #x, y, z, time, k, id = ta['x'], ta['y'], ta['z'], ta['time'], ta['k'], ta['particleid']
        ra = np.rec.fromarrays((ta['x'], ta['y'], ta['z'], ta['time'], ta['k'], ta['particleid']), dtype=dtype)
        return ra

    def get_alldata(self):
        """
        get pathline data from the pathline file for all pathlines and all times.

        Parameters
        ----------

        Returns
        ----------
        plist : a list of numpy record array
            A list of numpy recarrays with the x, y, z, time, k, and particleid for
            all pathlines.


        See Also
        --------

        Notes
        -----

        Examples
        --------

        >>> import flopy.utils.modpathfile as mpf
        >>> pthobj = flopy.utils.PathlineFile('model.mppth')
        >>> p = pthobj.get_alldata()

        """
        plist = []
        for partid in range(self.nid):
            plist.append(self.get_data(partid=partid))
        return plist
