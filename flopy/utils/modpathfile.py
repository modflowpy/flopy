"""
Module to read MODPATH output files.  The module contains two
important classes that can be accessed by the user.

*  EndpointFile (ascii endpoint file)
*  PathlineFile (ascii pathline file)

"""

import numpy as np
from ..utils.flopy_io import loadtxt
from ..utils.recarray_utils import ra_slice

class PathlineFile():
    """
    PathlineFile Class.

    Parameters
    ----------
    filename : string
        Name of the pathline file
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
    kijnames = ['k', 'i', 'j', 'particleid', 'particlegroup', 'linesegmentindex']

    def __init__(self, filename, verbose=False):
        """
        Class constructor.

        """
        self.fname = filename
        self.dtype, self.outdtype = self._get_dtypes()
        self._build_index()
        self._data = loadtxt(self.file, dtype=self.dtype, skiprows=self.skiprows)
        # set number of particle ids
        self.nid = self._data['particleid'].max()
        # convert layer, row, and column indices; particle id and group; and
        #  line segment indices to zero-based
        for n in self.kijnames:
            self._data[n] -= 1
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
            if isinstance(line, bytes):
                line = line.decode()
            if self.skiprows < 1:
                if 'MODPATH_PATHLINE_FILE 6' not in line.upper():
                    errmsg = '{} is not a valid pathline file'.format(self.fname)
                    raise Exception(errmsg)
            self.skiprows += 1
            if 'end header' in line.lower():
                break
        self.file.seek(0)

    def _get_dtypes(self):
        """
           Build numpy dtype for the MODPATH 6 pathline file.
        """
        dtype = np.dtype([("particleid", np.int), ("particlegroup", np.int),
                          ("timepointindex", np.int), ("cumulativetimestep", np.int),
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
            Maximum pathline time.

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

    def get_destination_pathline_data(self, dest_cells):
        """Get pathline data for set of destination cells.

        Parameters
        ----------
        dest_cells : list or array of tuples
            (k, i, j) of each destination cell (zero-based)

        Returns
        -------
        pthldest : np.recarray
            Slice of pathline data array (e.g. PathlineFile._data)
            containing only pathlines with final k,i,j in dest_cells.
        """
        ra = self._data.view(np.recarray)
        # find the intersection of endpoints and dest_cells
        # convert dest_cells to same dtype for comparison
        raslice = ra[['k', 'i', 'j']]
        dest_cells = np.array(dest_cells, dtype=raslice.dtype)
        inds = np.in1d(raslice, dest_cells)
        epdest = ra[inds].copy().view(np.recarray)

        # use particle ids to get the rest of the paths
        inds = np.in1d(ra.particleid, epdest.particleid)
        pthldes = ra[inds].copy()
        pthldes.sort(order=['particleid', 'time'])
        return pthldes

    def write_shapefile(self, pathline_data=None,
                        one_per_particle=True,
                        direction='ending',
                        shpname='endpoings.shp',
                        sr=None, epsg=None,
                        **kwargs):
        """Write pathlines to shapefile.

        pathline_data : np.recarry
            Record array of same form as that returned by EndpointFile.get_alldata.
            (if none, EndpointFile.get_alldata() is exported).
        one_per_particle : boolean (default True)
            True writes a single LineString with a single set of attribute data for each
            particle. False writes a record/geometry for each pathline segment
            (each row in the PathLine file). This option can be used to visualize
            attribute information (time, model layer, etc.) across a pathline in a GIS.
        direction : str
            String defining if starting or ending particle locations should be
            included in shapefile attribute information. Only used if one_per_particle=False.
            (default is 'ending')
        shpname : str
            File path for shapefile
        sr : flopy.utils.reference.SpatialReference instance
            Used to scale and rotate Global x,y,z values in MODPATH Endpoint file
        epsg : int
            EPSG code for writing projection (.prj) file. If this is not supplied,
            the proj4 string or epgs code associated with sr will be used.
        kwargs : keyword arguments to flopy.export.shapefile_utils.recarray2shp
        """
        from ..utils.reference import SpatialReference
        from ..utils.geometry import LineString
        from ..export.shapefile_utils import recarray2shp

        pth = pathline_data
        if pth is None:
            pth = self._data.view(np.recarray)
        pth = pth.copy()
        pth.sort(order=['particleid', 'time'])

        if sr is None:
            sr = SpatialReference()

        particles = np.unique(pth.particleid)
        geoms = []

        # 1 geometry for each path
        if one_per_particle:

            loc_inds = 0
            if direction == 'ending':
                loc_inds = -1

            pthdata = []
            for pid in particles:
                ra = pth[pth.particleid == pid]

                x, y = sr.transform(ra.x, ra.y)
                z = ra.z
                geoms.append(LineString(list(zip(x, y, z))))
                pthdata.append((pid,
                                ra.particlegroup[0],
                                ra.time.max(),
                                ra.k[loc_inds],
                                ra.i[loc_inds],
                                ra.j[loc_inds]))
            pthdata = np.array(pthdata, dtype=[('particleid', np.int),
                                               ('particlegroup', np.int),
                                               ('time', np.float),
                                               ('k', np.int),
                                               ('i', np.int),
                                               ('j', np.int)
                                               ]).view(np.recarray)
        # geometry for each row in PathLine file
        else:
            dtype = pth.dtype
            #pthdata = np.empty((0, len(dtype)), dtype=dtype).view(np.recarray)
            pthdata = []
            for pid in particles:
                ra = pth[pth.particleid == pid]
                x, y = sr.transform(ra.x, ra.y)
                z = ra.z
                geoms += [LineString([(x[i-1], y[i-1], z[i-1]),
                                          (x[i], y[i], z[i])])
                             for i in np.arange(1, (len(ra)))]
                #pthdata = np.append(pthdata, ra[1:]).view(np.recarray)
                pthdata += ra[1:].tolist()
            pthdata = np.array(pthdata, dtype=dtype).view(np.recarray)
        # convert back to one-based
        for n in set(self.kijnames).intersection(set(pthdata.dtype.names)):
            pthdata[n] += 1
        recarray2shp(pthdata, geoms, shpname=shpname, epsg=sr.epsg, **kwargs)


class EndpointFile():
    """
    EndpointFile Class.

    Parameters
    ----------
    filename : string
        Name of the endpoint file
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
    The EndpointFile class provides simple ways to retrieve MODPATH 6
    endpoint data from a MODPATH 6 ascii endpoint file.

    Examples
    --------

    >>> import flopy
    >>> endobj = flopy.utils.EndpointFile('model.mpend')
    >>> e1 = endobj.get_data(partid=1)


    """
    kijnames = ['k0', 'i0', 'j0', 'k', 'i', 'j', 'particleid', 'particlegroup']

    def __init__(self, filename, verbose=False):
        """
        Class constructor.

        """
        self.fname = filename
        self.dtype = self._get_dtypes()
        self._build_index()
        self._data = loadtxt(self.file, dtype=self.dtype, skiprows=self.skiprows)
        # set number of particle ids
        self.nid = self._data['particleid'].max()
        # convert layer, row, and column indices; particle id and group; and
        #  line segment indices to zero-based
        for n in self.kijnames:
            self._data[n] -= 1

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
            if isinstance(line, bytes):
                line = line.decode()
            if self.skiprows < 1:
                if 'MODPATH_ENDPOINT_FILE 6' not in line.upper():
                    errmsg = '{} is not a valid endpoint file'.format(self.fname)
                    raise Exception(errmsg)
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
        return dtype

    def get_maxid(self):
        """
        Get the maximum endpoint particle id in the file endpoint file

        Returns
        ----------
        out : int
            Maximum endpoint particle id.

        """
        return self.maxid

    def get_maxtime(self):
        """
        Get the maximum time in the endpoint file

        Returns
        ----------
        out : float
            Maximum endpoint time.

        """
        return self.data['finaltime'].max()


    def get_maxtraveltime(self):
        """
        Get the maximum travel time in the endpoint file

        Returns
        ----------
        out : float
            Maximum endpoint travel time.

        """
        return (self.data['finaltime'] - self.data['initialtime']).max()

    def get_data(self, partid=0):
        """
        Get endpoint data from the endpoint file for a single particle.

        Parameters
        ----------
        partid : int
            The zero-based particle id.  The first record is record 0.
            (default is 0)

        Returns
        ----------
        ra : numpy record array
            A numpy recarray with the endpoint particle data for
            endpoint partid.


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
        ra = self._data[idx]
        return ra

    def get_alldata(self):
        """
        Get endpoint data from the endpoint file for all endpoints.

        Parameters
        ----------

        Returns
        ----------
        ra : numpy record array
            A numpy recarray with the endpoint particle data


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
        ra = self._data.view(np.recarray).copy()
        # if final:
        #     ra = np.rec.fromarrays((self._data['x'], self._data['y'], self._data['z'],
        #                             self._data['finaltime'], self._data['k'],
        #                             self._data['particleid']), dtype=self.outdtype)
        # else:
        #     ra = np.rec.fromarrays((self._data['x0'], self._data['y0'], self._data['z0'],
        #                             self._data['initialtime'], self._data['k0'],
        #                             self._data['particleid']), dtype=self.outdtype)
        return ra

    def get_destination_endpoint_data(self, dest_cells):
        """Get endpoint data for set of destination cells.

        Parameters
        ----------
        dest_cells : list or array of tuples
            (k, i, j) of each destination cell (zero-based)

        Returns
        -------
        epdest : np.recarray
            Slice of endpoint data array (e.g. EndpointFile.get_alldata)
            containing only data with final k,i,j in dest_cells.
        """
        ra = self.get_alldata()
        # find the intersection of endpoints and dest_cells
        # convert dest_cells to same dtype for comparison
        raslice = ra_slice(ra, ['k', 'i', 'j'])
        dest_cells = np.array(dest_cells, dtype=[('k', int),
                                                 ('i', int),
                                                 ('j', int)])
        inds = np.in1d(raslice, dest_cells)
        epdest = ra[inds].copy().view(np.recarray)
        return epdest

    def write_shapefile(self, endpoint_data=None,
                        shpname='endpoings.shp',
                        direction='ending', sr=None, epsg=None,
                        **kwargs):
        """Write particle starting / ending locations to shapefile.

        endpoint_data : np.recarry
            Record array of same form as that returned by EndpointFile.get_alldata.
            (if none, EndpointFile.get_alldata() is exported).
        shpname : str
            File path for shapefile
        direction : str
            String defining if starting or ending particle locations should be
            considered. (default is 'ending')
        sr : flopy.utils.reference.SpatialReference instance
            Used to scale and rotate Global x,y,z values in MODPATH Endpoint file
        epsg : int
            EPSG code for writing projection (.prj) file. If this is not supplied,
            the proj4 string or epgs code associated with sr will be used.
        kwargs : keyword arguments to flopy.export.shapefile_utils.recarray2shp
        """
        from ..utils.reference import SpatialReference
        from ..utils.geometry import Point
        from ..export.shapefile_utils import recarray2shp

        epd = endpoint_data.copy()
        if epd is None:
            epd = self.get_alldata()

        if direction.lower() == 'ending':
            xcol, ycol, zcol = 'x', 'y', 'z'
        elif direction.lower() == 'starting':
            xcol, ycol, zcol = 'x0', 'y0', 'z0'
        else:
            errmsg = 'flopy.map.plot_endpoint direction must be "ending" ' + \
                     'or "starting".'
            raise Exception(errmsg)
        if sr is None:
            sr = SpatialReference()
        x, y = sr.transform(epd[xcol], epd[ycol])
        z = epd[zcol]

        geoms = [Point(x[i], y[i], z[i]) for i in range(len(epd))]
        # convert back to one-based
        for n in self.kijnames:
            epd[n] += 1
        recarray2shp(epd, geoms, shpname=shpname, epsg=epsg, **kwargs)