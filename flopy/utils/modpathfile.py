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

    >>> import flopy
    >>> pthobj = flopy.utils.PathlineFile('model.mppth')
    >>> p1 = pthobj.get_data(partid=1)


    """
    def __init__(self, filename, verbose=False):
        self.fname = filename
        self.dtype, self.outdtype = self._get_dtypes()
        self._build_index()
        self._data = np.loadtxt(self.file, dtype=self.dtype, skiprows=self.skiprows)
        # set number of particle ids
        self.nid = self._data['particleid'].max()
        # convert layer, row, and column indices; particle id and group; and
        #  line segment indices to zero-based
        self._data['k'] -= 1
        self._data['i'] -= 1
        self._data['j'] -= 1
        self._data['particleid'] -= 1
        self._data['particlegroup'] -= 1
        self._data['linesegmentindex'] -= 1
        # close the input file
        self.file.close()
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

    def _get_dtypes(self):
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
        outdtype = np.dtype([("x", np.float32), ("y", np.float32), ("z", np.float32),
                             ("time", np.float32), ("k", np.int), ("id", np.int)])
        return dtype, outdtype
                              

    def get_maxid(self):
        """
        Get the maximum pathline number in the file pathline file

        Returns
        ----------
        out : int
            Maximum pathline number.

        """
        return self.maxid


    def get_maxtime(self):
        """
        Get the maximum time in pathline file

        Returns
        ----------
        out : float
            Maximum pathline timer.

        """
        return self.data['time'].max()


    def get_data(self, partid=0, totim=None, ge=True):
        """
        get pathline data from the pathline file for a single pathline.

        Parameters
        ----------
        partid : int
            The zero-based particle id.  The first record is record 0.
        totim : float
            The simulation time. All pathline points for particle partid
            that are greater than or equal to (ge=True) or less than or
            equal to (ge=False) totim will be returned. Default is None
        ge : bool
            Boolean that determines if pathline times greater than or equal
            to or less than or equal to totim is used to create a subset
            of pathlines. Default is True.

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
        idx = self._data['particleid'] == partid
        if totim is not None:
            if ge:
                idx = (self._data['time'] >= totim) & (self._data['particleid'] == partid)
            else:
                idx = (self._data['time'] <= totim) & (self._data['particleid'] == partid)
        else:
            idx = self._data['particleid'] == partid
        self._ta = self._data[idx]
        ra = np.rec.fromarrays((self._ta['x'], self._ta['y'], self._ta['z'],
                                self._ta['time'], self._ta['k'], self._ta['particleid']), dtype=self.outdtype)
        return ra

    def get_alldata(self, totim=None, ge=True):
        """
        get pathline data from the pathline file for all pathlines and all times.

        Parameters
        ----------
        totim : float
            The simulation time. All pathline points for particle partid
            that are greater than or equal to (ge=True) or less than or
            equal to (ge=False) totim will be returned. Default is None
        ge : bool
            Boolean that determines if pathline times greater than or equal
            to or less than or equal to totim is used to create a subset
            of pathlines. Default is True.

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
            plist.append(self.get_data(partid=partid, totim=totim, ge=ge))
        return plist


class EndpointFile():
    """
    EndpointFile Class.

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
    The EndpointeFile class provides simple ways to retrieve MODPATH 6
    endpoint data from a MODPATH 6 ascii endpoint file.

    Examples
    --------

    >>> import flopy
    >>> endobj = flopy.utils.EndpointFile('model.mpend')
    >>> e1 = endobj.get_data(partid=1)


    """
    def __init__(self, filename, verbose=False):
        self.fname = filename
        self.dtype, self.outdtype = self._get_dtypes()
        self._build_index()
        self._data = np.loadtxt(self.file, dtype=self.dtype, skiprows=self.skiprows)
        # set number of particle ids
        self.nid = self._data['particleid'].max()
        # convert layer, row, and column indices; particle id and group; and
        #  line segment indices to zero-based
        self._data['k0'] -= 1
        self._data['i0'] -= 1
        self._data['j0'] -= 1
        self._data['k'] -= 1
        self._data['i'] -= 1
        self._data['j'] -= 1
        self._data['particleid'] -= 1
        self._data['particlegroup'] -= 1
        # close the input file
        self.file.close()
        return

    def _build_index(self):
        """
           Set position of the start of the pathline data.
        """
        self.skiprows = 0
        self.file = open(self.fname, 'r')
        idx = 0
        while True:
            line = self.file.readline()
            self.skiprows += 1
            if idx == 1:
                t = line.strip()
                self.direction = 1
                if int(t[0]) == 2:
                    self.direction = -1
            if 'end header' in line.lower():
                break
        self.file.seek(0)

    def _get_dtypes(self):
        """
           Build numpy dtype for the MODPATH 6 endpoint file.
        """
        dtype = np.dtype([("particleid", np.int), ("particlegroup", np.int),
                          ('status', np.int), ('initialtime', np.float32),
                          ('finaltime', np.float32), ('initialgrid', np.int),
                          ('k0', np.int), ('i0', np.int),
                          ('j0', np.int), ('initialcellface', np.int),
                          ('initialzone', np.int), ('xloc0', np.float32),
                          ('yloc0', np.float32), ('zloc0', np.float32),
                          ('x0', np.float32), ('y0', np.float32), ('z0', np.float32),
                          ('finalgrid', np.int), ('k', np.int), ('i', np.int),
                          ('j', np.int), ('finalcellface', np.int),
                          ('finalzone', np.int), ('xloc', np.float32),
                          ('yloc', np.float32), ('zloc', np.float32),
                          ('x', np.float32), ('y', np.float32), ('z', np.float32),
                          ('label', '|S40')])
        outdtype = np.dtype([("x", np.float32), ("y", np.float32), ("z", np.float32),
                             ("time", np.float32), ("k", np.int), ("id", np.int)])
        return dtype, outdtype


    def get_maxid(self):
        """
        Get the maximum endpoint particle id in the file endpoint file

        Returns
        ----------
        out : int
            Maximum enpoint particle id.

        """
        return self.maxid


    def get_maxtime(self):
        """
        Get the maximum time in endpoint file

        Returns
        ----------
        out : float
            Maximum endpoint time.

        """
        return self.data['finaltime'].max()


    def get_data(self, partid=0, final=True):
        """
        get endpoint data from the endpoint file for a single particle.

        Parameters
        ----------
        partid : int
            The zero-based particle id.  The first record is record 0.
        final : bool
            Boolean flag used determine if the initial or final endpoint
            time should be returned, Default is True

        Returns
        ----------
        ra : numpy record array
            A numpy recarray with the final or initial (final=False)
            x, y, z, time, k, and particleid for endpoint partid.


        See Also
        --------

        Notes
        -----

        Examples
        --------

        >>> import flopy
        >>> endobj = flopy.utils.EndpointFile('model.mpend')
        >>> e1 = endobj.get_data(partid=1)

        """
        idx = self._data['particleid'] == partid
        self._ta = self._data[idx]
        if final:
            ra = np.rec.fromarrays((self._ta['x'], self._ta['y'], self._ta['z'],
                                    self._ta['finaltime'], self._ta['k'],
                                    self._ta['particleid']), dtype=self.outdtype)
        else:
            ra = np.rec.fromarrays((self._ta['x0'], self._ta['y0'], self._ta['z0'],
                                    self._ta['initialtime'], self._ta['k0'],
                                    self._ta['particleid']), dtype=self.outdtype)
        return ra

    def get_alldata(self, final=True):
        """
        get endpoint data from the endpoint file for all endpoints.

        Parameters
        ----------
        final : bool
            Boolean flag used determine if the initial or final endpoint
            time should be returned, Default is True

        Returns
        ----------
        ra : numpy record array
            A numpy recarray with the final or initial (final=False)
            x, y, z, time, k, and particleid for endpoint partid.


        See Also
        --------

        Notes
        -----

        Examples
        --------

        >>> import flopy
        >>> endobj = flopy.utils.EndpointFile('model.mpend')
        >>> e = endobj.get_alldata()

        """
        if final:
            ra = np.rec.fromarrays((self._data['x'], self._data['y'], self._data['z'],
                                    self._data['finaltime'], self._data['k'],
                                    self._data['particleid']), dtype=self.outdtype)
        else:
            ra = np.rec.fromarrays((self._data['x0'], self._data['y0'], self._data['z0'],
                                    self._data['initialtime'], self._data['k0'],
                                    self._data['particleid']), dtype=self.outdtype)
        return ra
