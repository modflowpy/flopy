"""
Module spatial referencing for flopy model objects

"""
import sys
import os
import numpy as np


class SpatialReference(object):
    """
    a simple class to locate the model grid in x-y space

    Parameters
    ----------

    delr : numpy ndarray
        the model discretization delr vector

    delc : numpy ndarray
        the model discretization delc vector

    lenuni : int
        the length units flag from the discretization package

    xul : float
        the x coordinate of the upper left corner of the grid
        Enter either xul and yul or xll and yll.
    yul : float
        the y coordinate of the upper left corner of the grid
        Enter either xul and yul or xll and yll.
    xll : float
        the x coordinate of the lower left corner of the grid
        Enter either xul and yul or xll and yll.
    yll : float
        the y coordinate of the lower left corner of the grid
        Enter either xul and yul or xll and yll.
    rotation : float
        the counter-clockwise rotation (in degrees) of the grid

    proj4_str: str
        a PROJ4 string that identifies the grid in space. warning: case sensitive!

    epsg : int
        EPSG code that identifies the grid in space. Can be used in lieu of proj4.
        PROJ4 attribute will auto-populate if there is an internet connection
        (via get_proj4 method).
        See https://www.epsg-registry.org/ or spatialreference.org

    length_multiplier : float
        multiplier to convert model units to spatial reference units.
        delr and delc above will be multiplied by this value. (default=1.)

    Attributes
    ----------
    xedge : ndarray
        array of column edges

    yedge : ndarray
        array of row edges

    xgrid : ndarray
        numpy meshgrid of xedges

    ygrid : ndarray
        numpy meshgrid of yedges

    xcenter : ndarray
        array of column centers

    ycenter : ndarray
        array of row centers

    xcentergrid : ndarray
        numpy meshgrid of column centers

    ycentergrid : ndarray
        numpy meshgrid of row centers

    Notes
    -----

    xul and yul can be explicitly (re)set after SpatialReference
    instantiation, but only before any of the other attributes and methods are
    accessed
        
    """

    def __init__(self, delr=np.array([]), delc=np.array([]), lenuni=1, xul=None, yul=None, xll=None, yll=None, rotation=0.0,
                 proj4_str="EPSG:4326", epsg=None, units=None, length_multiplier=1.):

        for delrc in [delr, delc]:
            if isinstance(delrc, float) or isinstance(delrc, int):
                raise TypeError("delr and delcs must be an array or sequences equal in length to the number of rows/columns.")

        self.delc = np.atleast_1d(np.array(delc)) * length_multiplier
        self.delr = np.atleast_1d(np.array(delr)) * length_multiplier

        if self.delr.sum() == 0 or self.delc.sum() == 0:
            if xll is None or yll is None:
                print('Warning: no grid spacing or lower-left corner supplied. Origin will be set to zero.')
                xll, yll = 0, 0

        self.lenuni = lenuni
        self._proj4_str = proj4_str

        self.epsg = epsg
        if epsg is not None:
            self._proj4_str = getproj4(epsg)

        self.supported_units = ["feet","meters"]
        self._units = units
        self._reset()
        self.set_spatialreference(xul, yul, xll, yll, rotation)
        self.length_multiplier = length_multiplier

    @property
    def proj4_str(self):
        if "epsg" in self._proj4_str.lower() and \
           "init" not in self._proj4_str.lower():
            proj4_str = "+init=" + self._proj4_str
        else:
            proj4_str = self._proj4_str
        return proj4_str

    @property
    def units(self):
        units = None
        if self._units is not None:
            units = self._units.lower()
        else:
            try:
                # need this because preserve_units doesn't seem to be
                # working for complex proj4 strings.  So if an
                # epsg code was passed, we have no choice, but if a
                # proj4 string was passed, we can just parse it
                if "EPSG" in self.proj4_str.upper():
                    import pyproj

                    crs = pyproj.Proj(self.proj4_str,
                                      preseve_units=True,
                                      errcheck=True)
                    proj_str = crs.srs
                else:
                    proj_str = self.proj4_str
                if "units=m" in proj_str:
                    units = "meters"
                elif "units=ft" in proj_str or \
                   "to_meters:0.3048" in proj_str:
                    units = "feet"
            except:
                pass
                
        if units is None:
            print("warning: assuming SpatialReference units are meters")
            units = 'meters'
        assert units in self.supported_units
        return units

    @staticmethod
    def attribs_from_namfile_header(namefile):
        # check for reference info in the nam file header
        header = []
        with open(namefile,'r') as f:
            for line in f:
                if not line.startswith('#'):
                    break
                header.extend(line.strip().replace('#','').split(';'))

        xul, yul = None, None
        rotation = 0.0
        proj4_str = "EPSG:4326"
        start_datetime = "1/1/1970"
        units = None

        for item in header:
            if "xul" in item.lower():
                try:
                    xul = float(item.split(':')[1])
                except:
                    pass
            elif "yul" in item.lower():
                try:
                    yul = float(item.split(':')[1])
                except:
                    pass
            elif "rotation" in item.lower():
                try:
                    rotation = float(item.split(':')[1])
                except:
                    pass
            elif "proj4_str" in item.lower():
                try:
                    proj4_str = ':'.join(item.split(':')[1:]).strip()
                except:
                    pass
            elif "start" in item.lower():
                try:
                    start_datetime = item.split(':')[1].strip()
                except:
                    pass
            elif "units" in item.lower():
                units = item.split(':')[1].strip()

        return {"xul":xul,"yul":yul,"rotation":rotation,
                "proj4_str":proj4_str,"start_datetime":start_datetime,
                "units":units}

    def __setattr__(self, key, value):
        reset = True
        if key == "delr":
            super(SpatialReference,self).\
                __setattr__("delr",np.atleast_1d(np.array(value)))
        elif key == "delc":
            super(SpatialReference,self).\
                __setattr__("delc",np.atleast_1d(np.array(value)))
        elif key == "xul":
            super(SpatialReference,self).\
                __setattr__("xul",float(value))
        elif key == "yul":
            super(SpatialReference,self).\
                __setattr__("yul",float(value))
        elif key == "rotation":
            super(SpatialReference,self).\
                __setattr__("rotation",float(value))
        elif key == "lenuni":
            super(SpatialReference,self).\
                __setattr__("lenuni",int(value))
        elif key == "units":
            value = value.lower()
            assert value in self.supported_units
            super(SpatialReference,self).\
                __setattr__("_units",value)
        elif key == "proj4_str":
            super(SpatialReference,self).\
                __setattr__("_proj4_str",value)
        else:
            super(SpatialReference,self).__setattr__(key,value)
            reset = False
        if reset:
            self._reset()

    def reset(self,**kwargs):
        for key,value in kwargs.items():
            setattr(self,key,value)


    def _reset(self):
        self._xgrid = None
        self._ygrid = None
        self._ycentergrid = None
        self._xcentergrid = None

    @property
    def nrow(self):
        return self.delc.shape[0]

    @property
    def ncol(self):
        return self.delr.shape[0]

    def __eq__(self, other):
        if not isinstance(other,SpatialReference):
            return False
        if other.xul != self.xul:
            return False
        if other.yul != self.yul:
            return False
        if other.rotation != self.rotation:
            return False
        if other.proj4_str != self.proj4_str:
            return False
        return True


    @classmethod
    def from_namfile(cls,namefile):
        attribs = SpatialReference.attribs_from_namfile_header(namefile)
        try:
            attribs.pop("start_datetime")
        except:
            pass
        return SpatialReference(**attribs)


    @classmethod
    def from_gridspec(cls,gridspec_file,lenuni=0):
        f = open(gridspec_file,'r')
        #lines = f.readlines()
        raw = f.readline().strip().split()
        nrow = int(raw[0])
        ncol = int(raw[1])
        raw = f.readline().strip().split()
        xul, yul, rot = float(raw[0]), float(raw[1]), float(raw[2])
        delr = []
        j = 0
        while j < ncol:
            raw = f.readline().strip().split()
            for r in raw:
                if '*' in r:
                    rraw = r.split('*')
                    for n in range(int(rraw[0])):
                        delr.append(float(rraw[1]))
                        j += 1
                else:
                    delr.append(float(r))
                    j += 1
        delc = []
        i = 0
        while i < nrow:
            raw = f.readline().strip().split()
            for r in raw:
                if '*' in r:
                    rraw = r.split('*')
                    for n in range(int(rraw[0])):
                        delc.append(float(rraw[1]))
                        i += 1
                else:
                    delc.append(float(r))
                    i += 1
        f.close()
        return cls(np.array(delr), np.array(delc),
                   lenuni, xul=xul, yul=yul, rotation=rot)


    @property
    def attribute_dict(self):
        return {"xul":self.xul,"yul":self.yul,"rotation":self.rotation,
                "proj4_str":self.proj4_str}

    def set_spatialreference(self, xul=None, yul=None, xll=None, yll=None, rotation=0.0):
        """
            set spatial reference - can be called from model instance
        """
        if xul is not None and xll is not None:
            raise ValueError('both xul and xll entered. Please enter either xul, yul or xll, yll.')
        if yul is not None and yll is not None:
            raise ValueError('both yul and yll entered. Please enter either xul, yul or xll, yll.')

        theta = -rotation * np.pi / 180.
        # Set origin and rotation
        if xul is None:
            if xll is not None:
                self.xul = xll - np.sin(theta) * self.yedge[0]
            else:
                self.xul = 0.
        else:
            self.xul = xul
        if yul is None:
            if yll is not None:
                self.yul = yll + np.cos(theta) * self.yedge[0]
            else:
                self.yul = np.add.reduce(self.delc)
        else:
            self.yul = yul
        if xll is None:
            self.xll = self.xul + np.sin(theta) * self.yedge[0]
        else:
            self.xll = xll
        if yll is None:
            self.yll = self.yul - np.cos(theta) * self.yedge[0]
        else:
            self.yll = yll
        self.rotation = rotation
        self._reset()

    def __repr__(self):
        s = "xul:{0:<G}; yul:{1:<G}; rotation:{2:<G}; ".\
            format(self.xul,self.yul,self.rotation)
        s += "proj4_str:{0}; ".format(self.proj4_str)
        s += "units:{0}; ".format(self.units)
        s += "lenuni:{0}".format(self.lenuni)
        return s

    @property
    def xedge(self):
        return self.get_xedge_array()

    @property
    def yedge(self):
        return self.get_yedge_array()

    @property
    def xgrid(self):
        if self._xgrid is None:
            self._set_xygrid()
        return self._xgrid

    @property
    def ygrid(self):
        if self._ygrid is None:
            self._set_xygrid()
        return self._ygrid

    @property
    def xcenter(self):
        return self.get_xcenter_array()

    @property
    def ycenter(self):
        return self.get_ycenter_array()

    @property
    def ycentergrid(self):
        if self._ycentergrid is None:
            self._set_xycentergrid()
        return self._ycentergrid

    @property
    def xcentergrid(self):
        if self._xcentergrid is None:
            self._set_xycentergrid()
        return self._xcentergrid

    def _set_xycentergrid(self):
        self._xcentergrid, self._ycentergrid = np.meshgrid(self.xcenter,
                                                          self.ycenter)
        self._xcentergrid, self._ycentergrid = self.transform(self._xcentergrid,
                                                              self._ycentergrid)

    def _set_xygrid(self):
        self._xgrid, self._ygrid = np.meshgrid(self.xedge, self.yedge)
        self._xgrid, self._ygrid = self.transform(self._xgrid, self._ygrid)


    @staticmethod
    def rotate(x, y, theta, xorigin=0., yorigin=0.):
        """
        Given x and y array-like values calculate the rotation about an
        arbitrary origin and then return the rotated coordinates.  theta is in
        degrees.

        """
        theta = -theta * np.pi / 180.
        xrot = xorigin + np.cos(theta) * (x - xorigin) - np.sin(theta) * \
                                                         (y - yorigin)
        yrot = yorigin + np.sin(theta) * (x - xorigin) + np.cos(theta) * \
                                                         (y - yorigin)
        return xrot, yrot

    def transform(self, x, y):
        """
        Given x and y array-like values, apply rotation, scale and offset,
        to convert them from model coordinates to real-world coordinates.
        """
        x += self.xll
        y += self.yll
        x, y = SpatialReference.rotate(x * self.length_multiplier,
                                       y * self.length_multiplier,
                                       theta=self.rotation,
                                       xorigin=self.xll, yorigin=self.yll)

        return x, y


    def get_extent(self):
        """
        Get the extent of the rotated and offset grid

        Return (xmin, xmax, ymin, ymax)

        """
        x0 = self.xedge[0]
        x1 = self.xedge[-1]
        y0 = self.yedge[0]
        y1 = self.yedge[-1]

        # upper left point
        x0r, y0r = self.transform(x0, y0)

        # upper right point
        x1r, y1r = self.transform(x1, y0)

        # lower right point
        x2r, y2r = self.transform(x1, y1)

        # lower left point
        x3r, y3r = self.transform(x0, y1)

        xmin = min(x0r, x1r, x2r, x3r)
        xmax = max(x0r, x1r, x2r, x3r)
        ymin = min(y0r, y1r, y2r, y3r)
        ymax = max(y0r, y1r, y2r, y3r)

        return (xmin, xmax, ymin, ymax)

    def get_grid_lines(self):
        """
            get the grid lines as a list
        """
        xmin = self.xedge[0]
        xmax = self.xedge[-1]
        ymin = self.yedge[-1]
        ymax = self.yedge[0]
        lines = []
        # Vertical lines
        for j in range(self.ncol + 1):
            x0 = self.xedge[j]
            x1 = x0
            y0 = ymin
            y1 = ymax
            x0r, y0r = self.transform(x0, y0)
            x1r, y1r = self.transform(x1, y1)
            lines.append([(x0r, y0r), (x1r, y1r)])

        #horizontal lines
        for i in range(self.nrow + 1):
            x0 = xmin
            x1 = xmax
            y0 = self.yedge[i]
            y1 = y0
            x0r, y0r = self.transform(x0, y0)
            x1r, y1r = self.transform(x1, y1)
            lines.append([(x0r, y0r), (x1r, y1r)])
        return lines


    def get_xcenter_array(self):
        """
        Return a numpy one-dimensional float array that has the cell center x
        coordinate for every column in the grid in model space - not offset or rotated.

        """
        x = np.add.accumulate(self.delr) - 0.5 * self.delr
        return x

    def get_ycenter_array(self):
        """
        Return a numpy one-dimensional float array that has the cell center x
        coordinate for every row in the grid in model space - not offset of rotated.

        """
        Ly = np.add.reduce(self.delc)
        y = Ly - (np.add.accumulate(self.delc) - 0.5 *
                   self.delc)
        return y

    def get_xedge_array(self):
        """
        Return a numpy one-dimensional float array that has the cell edge x
        coordinates for every column in the grid in model space - not offset
        or rotated.  Array is of size (ncol + 1)

        """
        xedge = np.concatenate(([0.], np.add.accumulate(self.delr)))
        return xedge

    def get_yedge_array(self):
        """
        Return a numpy one-dimensional float array that has the cell edge y
        coordinates for every row in the grid in model space - not offset or
        rotated. Array is of size (nrow + 1)

        """
        length_y = np.add.reduce(self.delc)
        yedge = np.concatenate(([length_y], length_y -
                np.add.accumulate(self.delc)))
        return yedge


    def write_gridSpec(self, filename):
        """ write a PEST-style grid specification file
        """
        f = open(filename,'w')
        f.write("{0:10d} {1:10d}\n".format(self.delc.shape[0], self.delr.shape[0]))
        f.write("{0:15.6E} {1:15.6E} {2:15.6E}\n".format(self.xul, self.yul, self.rotation))
        for c in self.delc:
            f.write("{0:15.6E} ".format(c))
        f.write('\n')
        for r in self.delr:
            f.write("{0:15.6E} ".format(r))
        f.write('\n')
        return

    def get_vertices(self, i, j):
        pts = []
        xgrid, ygrid = self.xgrid, self.ygrid
        pts.append([xgrid[i, j], ygrid[i, j]])
        pts.append([xgrid[i+1, j], ygrid[i+1, j]])
        pts.append([xgrid[i+1, j+1], ygrid[i+1, j+1]])
        pts.append([xgrid[i, j+1], ygrid[i, j+1]])
        pts.append([xgrid[i, j], ygrid[i, j]])
        return pts

    def interpolate(self, a, xi, method='nearest'):
        """
        Use the griddata method to interpolate values from an array onto the
        points defined in xi.  For any values outside of the grid, use
        'nearest' to find a value for them.

        Parameters
        ----------
        a : numpy.ndarray
            array to interpolate from.  It must be of size nrow, ncol
        xi : numpy.ndarray
            array containing x and y point coordinates of size (npts, 2). xi
            also works with broadcasting so that if a is a 2d array, then
            xi can be passed in as (xgrid, ygrid).
        method : {'linear', 'nearest', 'cubic'}
            method to use for interpolation (default is 'nearest')

        Returns
        -------
        b : numpy.ndarray
            array of size (npts)

        """
        from scipy.interpolate import griddata

        # Create a 2d array of points for the grid centers
        points = np.empty((self.ncol * self.nrow, 2))
        points[:, 0] = self.xcentergrid.flatten()
        points[:, 1] = self.ycentergrid.flatten()

        # Use the griddata function to interpolate to the xi points
        b = griddata(points, a.flatten(), xi, method=method, fill_value=np.nan)

        # if method is linear or cubic, then replace nan's with a value
        # interpolated using nearest
        if method != 'nearest':
            bn = griddata(points, a.flatten(), xi, method='nearest')
            idx = np.isnan(b)
            b[idx] = bn[idx]

        return b

class epsgRef:
    """Sets up a local database of projection file text referenced by epsg code.
    The database is located in the site packages folder in epsgref.py, which
    contains a dictionary, prj, of projection file text keyed by epsg value.
    """
    def __init__(self):
        sp = [f for f in sys.path if f.endswith('site-packages')][0]
        self.location = os.path.join(sp, 'epsgref.py')

    def _remove_pyc(self):
        try: # get rid of pyc file
            os.remove(self.location + 'c')
        except:
            pass
    def make(self):
        if not os.path.exists(self.location):
            newfile = open(self.location, 'w')
            newfile.write('prj = {}\n')
            newfile.close()
    def reset(self, verbose=True):
        os.remove(self.location)
        self._remove_pyc()
        self.make()
        if verbose:
            print('Resetting {}'.format(self.location))
    def add(self, epsg, prj):
        """add an epsg code to epsgref.py"""
        with open(self.location, 'a') as epsgfile:
            epsgfile.write("prj[{:d}] = '{}'\n".format(epsg, prj))
    def remove(self, epsg):
        """removes an epsg entry from epsgref.py"""
        from epsgref import prj
        self.reset(verbose=False)
        if epsg in prj.keys():
            del prj[epsg]
        for epsg, prj in prj.items():
            self.add(epsg, prj)
    @staticmethod
    def show():
        import importlib
        import epsgref
        importlib.reload(epsgref)
        from epsgref import prj
        for k, v in prj.items():
            print('{}:\n{}\n'.format(k, v))


def getprj(epsg, addlocalreference=True):
    """Gets projection file (.prj) text for given epsg code from spatialreference.org
    See: https://www.epsg-registry.org/

    Parameters
    ----------
    epsg : int
        epsg code for coordinate system
    addlocalreference : boolean
        adds the projection file text associated with epsg to a local
        database, epsgref.py, located in site-packages.

    Returns
    -------
    prj : str
        text for a projection (*.prj) file.
    """
    epsgfile = epsgRef()
    prj = None
    try:
        from epsgref import prj
        prj = prj.get(epsg)
    except:
        epsgfile.make()

    if prj is None:
        prj = get_spatialreference(epsg, text='prettywkt')
    if addlocalreference:
        epsgfile.add(epsg, prj)
    return prj

def get_spatialreference(epsg, text='prettywkt'):
    """Gets text for given epsg code and text format from spatialreference.org
    Fetches the reference text using the url:
        http://spatialreference.org/ref/epsg/<epsg code>/<text>/

    See: https://www.epsg-registry.org/

    Parameters
    ----------
    epsg : int
        epsg code for coordinate system
    text : str
        string added to url

    Returns
    -------
    url : str

    """
    url = "http://spatialreference.org/ref/epsg/{0}/{1}/".format(epsg, text)
    try:
        # For Python 3.0 and later
        from urllib.request import urlopen
    except ImportError:
        # Fall back to Python 2's urllib2
        from urllib2 import urlopen
    try:
        urlobj = urlopen(url)
        text = urlobj.read().decode()
    except:
        e = sys.exc_info()
        print(e)
        print('Need an internet connection to look up epsg on spatialreference.org.')
        return
    text = text.replace("\n", "")
    return text

def getproj4(epsg):
    """Gets projection file (.prj) text for given epsg code from spatialreference.org
    See: https://www.epsg-registry.org/

    Parameters
    ----------
    epsg : int
        epsg code for coordinate system

    Returns
    -------
    prj : str
        text for a projection (*.prj) file.
    """
    return get_spatialreference(epsg, text='proj4')