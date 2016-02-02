"""
Module spatial referencing for flopy model objects

"""
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

    yul : float
        the y coordinate of the upper left corner of the grid

    rotation : float
        the counter-clockwise rotation (in degrees) of the grid

    proj4_str: str
        a PROJ4 string that identifies the grid in space. warning: case sensitive!

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

    def __init__(self, delr=1.0, delc=1.0, lenuni=1, xul=None, yul=None, rotation=0.0,
                 proj4_str="EPSG:4326"):
        self.delc = np.atleast_1d(np.array(delc))
        self.delr = np.atleast_1d(np.array(delr))

        self.lenuni = lenuni
        self.proj4_str = proj4_str
        self._reset()
        self.set_spatialreference(xul, yul, rotation)


    @classmethod
    def from_namfile_header(cls,namefile):
        # check for reference info in the nam file header
        header = []
        with open(namefile,'r') as f:
            for line in f:
                if not line.startswith('#'):
                    break
                header.extend(line.strip().replace('#','').split(','))

        xul, yul = None, None
        rotation = 0.0
        proj4_str = "EPSG:4326"
        start_datetime = "1/1/1970"

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

        return cls(xul=xul,yul=yul,rotation=rotation,proj4_str=proj4_str),\
               start_datetime

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
    def from_gridspec(cls,gridspec_file,lenuni=0):
        f = open(gridspec_file,'r')
        lines = f.readlines()
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
                        delr.append(int(rraw[1]))
                        j += 1
                else:
                    delr.append(int(r))
                    j += 1
        delc = []
        i = 0
        while i < nrow:
            raw = f.readline().strip().split()
            for r in raw:
                if '*' in r:
                    rraw = r.split('*')
                    for n in range(int(rraw[0])):
                        delc.append(int(rraw[1]))
                        i += 1
                else:
                    delc.append(int(r))
                    i += 1
        f.close()
        return cls(np.array(delr), np.array(delc),
                   lenuni, xul=xul, yul=yul, rotation=rot)


    @property
    def attribute_dict(self):
        return {"xul":self.xul,"yul":self.yul,"rotation":self.rotation,
                "proj4_str":self.proj4_str}

    def set_spatialreference(self, xul=None, yul=None, rotation=0.0):
        """
            set spatial reference - can be called from model instance
        """

        # Set origin and rotation
        if xul is None:
            self.xul = 0.
        else:
            self.xul = xul
        if yul is None:
            self.yul = np.add.reduce(self.delc)
        else:
            self.yul = yul
        self.rotation = rotation
        self._reset()

    def __repr__(self):
        s = "xul:{0:<G}, yul:{1:<G}, rotation:{2:<G}, ".\
            format(self.xul,self.yul,self.rotation)
        s += "proj4_str:{0}".format(self.proj4_str)
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
        self._xcentergrid, self._ycentergrid = self.rotate(self._xcentergrid,
                                                          self._ycentergrid,
                                                          self.rotation,
                                                          0, self.yedge[0])
        self._xcentergrid += self.xul
        self._ycentergrid += self.yul - self.yedge[0]

    def _set_xygrid(self):
        self._xgrid, self._ygrid = np.meshgrid(self.xedge, self.yedge)
        self._xgrid, self._ygrid = self.rotate(self._xgrid, self._ygrid, self.rotation,
                                               0, self.yedge[0])
        self._xgrid += self.xul
        self._ygrid += self.yul - self.yedge[0]


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
        x0r, y0r = self.rotate(x0, y0, self.rotation, 0, self.yedge[0])
        x0r += self.xul
        y0r += self.yul - self.yedge[0]

        # upper right point
        x1r, y1r = self.rotate(x1, y0, self.rotation, 0, self.yedge[0])
        x1r += self.xul
        y1r += self.yul - self.yedge[0]

        # lower right point
        x2r, y2r = self.rotate(x1, y1, self.rotation, 0, self.yedge[0])
        x2r += self.xul
        y2r += self.yul - self.yedge[0]

        # lower left point
        x3r, y3r = self.rotate(x0, y1, self.rotation, 0, self.yedge[0])
        x3r += self.xul
        y3r += self.yul - self.yedge[0]

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
            x0r, y0r = self.rotate(x0, y0, self.rotation, 0, self.yedge[0])
            x0r += self.xul
            y0r += self.yul - self.yedge[0]
            x1r, y1r = self.rotate(x1, y1, self.rotation, 0, self.yedge[0])
            x1r += self.xul
            y1r += self.yul - self.yedge[0]
            lines.append([(x0r, y0r), (x1r, y1r)])

        #horizontal lines
        for i in range(self.nrow + 1):
            x0 = xmin
            x1 = xmax
            y0 = self.yedge[i]
            y1 = y0
            x0r, y0r = self.rotate(x0, y0, self.rotation, 0, self.yedge[0])
            x0r += self.xul
            y0r += self.yul - self.yedge[0]
            x1r, y1r = self.rotate(x1, y1, self.rotation, 0, self.yedge[0])
            x1r += self.xul
            y1r += self.yul - self.yedge[0]
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


