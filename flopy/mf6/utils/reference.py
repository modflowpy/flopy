"""
Module spatial referencing for flopy model objects

"""
import numpy as np


class StructuredSpatialReference(object):
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
        a PROJ4 string that identifies the grid in space. warning: case
        sensitive!

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

    def __init__(
        self,
        delr=1.0,
        delc=1.0,
        lenuni=1,
        nlay=1,
        xul=None,
        yul=None,
        rotation=0.0,
        proj4_str=None,
        **kwargs
    ):
        self.delc = np.atleast_1d(np.array(delc))
        self.delr = np.atleast_1d(np.array(delr))
        self.nlay = nlay
        self.lenuni = lenuni
        self.proj4_str = proj4_str
        self._reset()
        self.set_spatialreference(xul, yul, rotation)

    @classmethod
    def from_namfile_header(cls, namefile):
        # check for reference info in the nam file header
        header = []
        with open(namefile, "r") as f:
            for line in f:
                if not line.startswith("#"):
                    break
                header.extend(line.strip().replace("#", "").split(","))

        xul, yul = None, None
        rotation = 0.0
        proj4_str = None
        start_datetime = "1/1/1970"

        for item in header:
            if "xul" in item.lower():
                try:
                    xul = float(item.split(":")[1])
                except:
                    pass
            elif "yul" in item.lower():
                try:
                    yul = float(item.split(":")[1])
                except:
                    pass
            elif "rotation" in item.lower():
                try:
                    rotation = float(item.split(":")[1])
                except:
                    pass
            elif "proj4_str" in item.lower():
                try:
                    proj4_str = ":".join(item.split(":")[1:]).strip()
                except:
                    pass
            elif "start" in item.lower():
                try:
                    start_datetime = item.split(":")[1].strip()
                except:
                    pass

        return (
            cls(xul=xul, yul=yul, rotation=rotation, proj4_str=proj4_str),
            start_datetime,
        )

    def __setattr__(self, key, value):
        reset = True
        if key == "delr":
            super(StructuredSpatialReference, self).__setattr__(
                "delr", np.atleast_1d(np.array(value))
            )
        elif key == "delc":
            super(StructuredSpatialReference, self).__setattr__(
                "delc", np.atleast_1d(np.array(value))
            )
        elif key == "xul":
            super(StructuredSpatialReference, self).__setattr__(
                "xul", float(value)
            )
        elif key == "yul":
            super(StructuredSpatialReference, self).__setattr__(
                "yul", float(value)
            )
        elif key == "rotation":
            super(StructuredSpatialReference, self).__setattr__(
                "rotation", float(value)
            )
        elif key == "lenuni":
            super(StructuredSpatialReference, self).__setattr__(
                "lenuni", int(value)
            )
        elif key == "nlay":
            super(StructuredSpatialReference, self).__setattr__(
                "nlay", int(value)
            )
        else:
            super(StructuredSpatialReference, self).__setattr__(key, value)
            reset = False
        if reset:
            self._reset()

    def reset(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

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
        if not isinstance(other, StructuredSpatialReference):
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
    def from_gridspec(cls, gridspec_file, lenuni=0):
        f = open(gridspec_file, "r")
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
                if "*" in r:
                    rraw = r.split("*")
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
                if "*" in r:
                    rraw = r.split("*")
                    for n in range(int(rraw[0])):
                        delc.append(int(rraw[1]))
                        i += 1
                else:
                    delc.append(int(r))
                    i += 1
        f.close()
        return cls(
            np.array(delr),
            np.array(delc),
            lenuni,
            xul=xul,
            yul=yul,
            rotation=rot,
        )

    @property
    def attribute_dict(self):
        return {
            "xul": self.xul,
            "yul": self.yul,
            "rotation": self.rotation,
            "proj4_str": self.proj4_str,
        }

    def set_spatialreference(self, xul=None, yul=None, rotation=0.0):
        """
        set spatial reference - can be called from model instance
        """

        # Set origin and rotation
        if xul is None:
            self.xul = 0.0
        else:
            self.xul = xul
        if yul is None:
            self.yul = np.add.reduce(self.delc)
        else:
            self.yul = yul
        self.rotation = rotation
        self._reset()

    def __repr__(self):
        s = "xul:{0:<G}, yul:{1:<G}, rotation:{2:<G}, ".format(
            self.xul, self.yul, self.rotation
        )
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
        self._xcentergrid, self._ycentergrid = np.meshgrid(
            self.xcenter, self.ycenter
        )
        self._xcentergrid, self._ycentergrid = self.rotate(
            self._xcentergrid,
            self._ycentergrid,
            self.rotation,
            0,
            self.yedge[0],
        )
        self._xcentergrid += self.xul
        self._ycentergrid += self.yul - self.yedge[0]

    def _set_xygrid(self):
        self._xgrid, self._ygrid = np.meshgrid(self.xedge, self.yedge)
        self._xgrid, self._ygrid = self.rotate(
            self._xgrid, self._ygrid, self.rotation, 0, self.yedge[0]
        )
        self._xgrid += self.xul
        self._ygrid += self.yul - self.yedge[0]

    @staticmethod
    def rotate(x, y, theta, xorigin=0.0, yorigin=0.0):
        """
        Given x and y array-like values calculate the rotation about an
        arbitrary origin and then return the rotated coordinates.  theta is in
        degrees.

        """
        theta = -theta * np.pi / 180.0
        xrot = (
            xorigin
            + np.cos(theta) * (x - xorigin)
            - np.sin(theta) * (y - yorigin)
        )
        yrot = (
            yorigin
            + np.sin(theta) * (x - xorigin)
            + np.cos(theta) * (y - yorigin)
        )
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

        return xmin, xmax, ymin, ymax

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

        # horizontal lines
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
        coordinate for every column in the grid in model space - not offset
        or rotated.

        """
        x = np.add.accumulate(self.delr) - 0.5 * self.delr
        return x

    def get_ycenter_array(self):
        """
        Return a numpy one-dimensional float array that has the cell center x
        coordinate for every row in the grid in model space - not offset
        of rotated.

        """
        Ly = np.add.reduce(self.delc)
        y = Ly - (np.add.accumulate(self.delc) - 0.5 * self.delc)

        return y

    def get_xedge_array(self):
        """
        Return a numpy one-dimensional float array that has the cell edge x
        coordinates for every column in the grid in model space - not offset
        or rotated.  Array is of size (ncol + 1)

        """
        xedge = np.concatenate(([0.0], np.add.accumulate(self.delr)))
        return xedge

    def get_yedge_array(self):
        """
        Return a numpy one-dimensional float array that has the cell edge y
        coordinates for every row in the grid in model space - not offset or
        rotated. Array is of size (nrow + 1)

        """
        length_y = np.add.reduce(self.delc)
        yedge = np.concatenate(
            ([length_y], length_y - np.add.accumulate(self.delc))
        )
        return yedge

    def write_gridSpec(self, filename):
        """write a PEST-style grid specification file"""
        f = open(filename, "w")
        f.write(
            "{0:10d} {1:10d}\n".format(self.delc.shape[0], self.delr.shape[0])
        )
        f.write(
            "{0:15.6E} {1:15.6E} {2:15.6E}\n".format(
                self.xul, self.yul, self.rotation
            )
        )
        for c in self.delc:
            f.write("{0:15.6E} ".format(c))
        f.write("\n")
        for r in self.delr:
            f.write("{0:15.6E} ".format(r))
        f.write("\n")
        return

    def get_vertices(self, i, j):
        pts = []
        xgrid, ygrid = self.xgrid, self.ygrid
        pts.append([xgrid[i, j], ygrid[i, j]])
        pts.append([xgrid[i + 1, j], ygrid[i + 1, j]])
        pts.append([xgrid[i + 1, j + 1], ygrid[i + 1, j + 1]])
        pts.append([xgrid[i, j + 1], ygrid[i, j + 1]])
        pts.append([xgrid[i, j], ygrid[i, j]])
        return pts

    def interpolate(self, a, xi, method="nearest"):
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
        try:
            from scipy.interpolate import griddata
        except:
            print("scipy not installed\ntry pip install scipy")
            return None

        # Create a 2d array of points for the grid centers
        points = np.empty((self.ncol * self.nrow, 2))
        points[:, 0] = self.xcentergrid.flatten()
        points[:, 1] = self.ycentergrid.flatten()

        # Use the griddata function to interpolate to the xi points
        b = griddata(points, a.flatten(), xi, method=method, fill_value=np.nan)

        # if method is linear or cubic, then replace nan's with a value
        # interpolated using nearest
        if method != "nearest":
            bn = griddata(points, a.flatten(), xi, method="nearest")
            idx = np.isnan(b)
            b[idx] = bn[idx]

        return b


class VertexSpatialReference(object):
    """
    a simple class to locate the model grid in x-y space

    Parameters
    ----------
    xvdict: dictionary
        dictionary of x-vertices {1: (0,1,1,0)}
    yvdict: dictionary
        dictionary of y-vertices {1: (1,0,1,0)}
    lenuni : int
        the length units flag from the discretization package
    xadj : float
        the x coordinate of the upper left corner of the grid
    yadj : float
        the y coordinate of the upper left corner of the grid
    rotation : float
        the counter-clockwise rotation (in degrees) of the grid
    proj4_str: str
        a PROJ4 string that identifies the grid in space. warning:
        case sensitive!
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

    xadj and yuadj can be explicitly (re)set after SpatialReference
    instantiation, but only before any of the other attributes and methods are
    accessed

    """

    def __init__(
        self,
        xvdict=None,
        yvdict=None,
        nlay=1,
        xadj=0,
        yadj=0,
        rotation=0.0,
        lenuni=1.0,
        proj4_str=None,
        **kwargs
    ):

        assert len(xvdict) == len(
            yvdict
        ), "len(xvdict): {} != len(yvdict): {}".format(
            len(xvdict), len(yvdict)
        )

        self._xv = np.array([xvdict[idx] for idx, key in enumerate(xvdict)])
        self._yv = np.array([yvdict[idx] for idx, key in enumerate(yvdict)])
        self.nlay = nlay
        self.lenuni = lenuni
        self.proj4_str = proj4_str
        self._reset()
        self.set_spatialreference(xadj, yadj, rotation)

    def _reset(self):
        self._xvarr = None
        self._yvarr = None
        self._xvdict = None
        self._yvdict = None
        self._xyvdict = None
        self._xcenter_array = None
        self._ycenter_array = None
        self._ncpl = None

    @classmethod
    def from_namfile_header(cls, namefile):
        # check for reference info in the nam file header
        header = []
        with open(namefile, "r") as f:
            for line in f:
                if not line.startswith("#"):
                    break
                header.extend(line.strip().replace("#", "").split(","))

        xadj, yadj = None, None
        proj4_str = None
        start_datetime = "1/1/1970"

        for item in header:
            if "xadj" in item.lower():
                try:
                    xadj = float(item.split(":")[1])
                except:
                    pass
            elif "yadj" in item.lower():
                try:
                    yadj = float(item.split(":")[1])
                except:
                    pass
            elif "proj4_str" in item.lower():
                try:
                    proj4_str = ":".join(item.split(":")[1:]).strip()
                except:
                    pass
            elif "start" in item.lower():
                try:
                    start_datetime = item.split(":")[1].strip()
                except:
                    pass

        return cls(xdaj=xadj, yadj=yadj, proj4_str=proj4_str), start_datetime

    def __setattr__(self, key, value):
        reset = True
        if key == "xvdict":
            super(VertexSpatialReference, self).__setattr__(
                "xvdict", dict(value)
            )
        elif key == "yvdict":
            super(VertexSpatialReference, self).__setattr__(
                "yvdict", dict(value)
            )
        elif key == "xyvdict":
            super(VertexSpatialReference, self).__setattr__("xyvdict", value)
        elif key == "xadj":
            super(VertexSpatialReference, self).__setattr__(
                "xadj", float(value)
            )
        elif key == "yadj":
            super(VertexSpatialReference, self).__setattr__(
                "yadj", float(value)
            )
        elif key == "rotation":
            super(VertexSpatialReference, self).__setattr__(
                "rotation", float(value)
            )
        elif key == "lenuni":
            super(VertexSpatialReference, self).__setattr__(
                "lenuni", int(value)
            )
        else:
            super(VertexSpatialReference, self).__setattr__(key, value)
            reset = False
        if reset:
            self._reset()

    def set_spatialreference(self, xadj=0.0, yadj=0.0, rotation=0.0):
        """
        set spatial reference - can be called from model instance
        xadj, yadj should be named xadj, yadj since they represent an
        adjustment factor
        """

        # Set origin and rotation
        self.xadj = xadj
        self.yadj = yadj
        self.rotation = rotation
        self._reset()

    @staticmethod
    def rotate(x, y, theta, xorigin=0.0, yorigin=0.0):
        """
        Given x and y array-like values calculate the rotation about an
        arbitrary origin and then return the rotated coordinates.  theta is in
        degrees.

        """
        theta = -theta * np.pi / 180.0
        xrot = (
            xorigin
            + np.cos(theta) * (x - xorigin)
            - np.sin(theta) * (y - yorigin)
        )
        yrot = (
            yorigin
            + np.sin(theta) * (x - xorigin)
            + np.cos(theta) * (y - yorigin)
        )
        return xrot, yrot

    def get_extent(self):
        """
        Get the extent of the rotated and offset grid

        Return (xmin, xmax, ymin, ymax)

        """
        xvarr, yvarr = self._get_rotated_vertices()

        xmin = np.min(xvarr)
        xmax = np.max(xvarr)
        ymin = np.min(yvarr)
        ymax = np.max(yvarr)
        return xmin, xmax, ymin, ymax

    @property
    def ncpl(self):
        if self._ncpl is None:
            self._ncpl = self.xarr.size / self.nlay
        return self._ncpl

    @property
    def xdict(self):
        if self._xvdict is None:
            self._xvdict = self._set_vertices()[2]
        return self._xvdict

    @property
    def ydict(self):
        if self._yvdict is None:
            self._yvdict = self._set_vertices()[3]
        return self._yvdict

    @property
    def xydict(self):
        if self._xyvdict is None:
            self._xyvdict = self._set_vertices()[4]
        return self._xyvdict

    @property
    def xarr(self):
        if self._xvarr is None:
            self._xvarr = self._set_vertices()[0]
        return self._xvarr

    @property
    def yarr(self):
        if self._yvarr is None:
            self._yvarr = self._set_vertices()[1]
        return self._yvarr

    @property
    def xcenter_array(self):
        if self._xcenter_array is None:
            self._set_xcenter_array()
        return self._xcenter_array

    @property
    def ycenter_array(self):
        if self._ycenter_array is None:
            self._set_ycenter_array()
        return self._ycenter_array

    def _get_rotated_vertices(self):
        """
        Adjusts position and rotates vertices if applicable

        Returns
        -------
        xvarr, yvarr:
            rotated and adjusted np.arrays of vertices
        """
        xvadj = self._xv + self.xadj
        yvadj = self._yv + self.yadj
        xvarr, yvarr = self.rotate(xvadj, yvadj, self.rotation)

        return xvarr, yvarr

    def _set_vertices(self):
        """
        Sets variables _xvarr, _yvarr, _xvdict and _yvdict to be accessed by
        property instances.

        Returns
        -------

        """
        self._xvarr, self._yvarr = self._get_rotated_vertices()
        self._xvdict = {idx: verts for idx, verts in enumerate(self._xvarr)}
        self._yvdict = {idx: verts for idx, verts in enumerate(self._yvarr)}
        self._xyvdict = {
            idx: np.array(list(zip(self._xvdict[idx], self._yvdict[idx])))
            for idx in self._xvdict
        }
        return (
            self._xvarr,
            self._yvarr,
            self._xvdict,
            self._yvdict,
            self._xyvdict,
        )

    def _set_xcenter_array(self):
        """
        Gets the x vertex center location of all cells sets to a 1d array for
        further interpolation. Useful when using Scipy.griddata to contour data

        """
        if self._xvarr is None:
            self._set_vertices()

        self._xcenter_array = np.array([])
        for cell in self.xarr:
            self._xcenter_array = np.append(self._xcenter_array, np.mean(cell))

    def _set_ycenter_array(self):
        """
        Gets the x vertex center location of all cells sets to a 1d array for
        further interpolation. Useful when using Scipy.griddata to contour data

        Returns
        -------
        """

        if self._yvarr is None:
            self._set_vertices()

        self._ycenter_array = np.array([])
        for cell in self.yarr:
            self._ycenter_array = np.append(self._ycenter_array, np.mean(cell))


class SpatialReference(object):
    """
    A dynamic inheritance class that locates a gridded model in space

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
            a PROJ4 string that identifies the grid in space. warning:
            case sensitive!
        xadj : float
            vertex grid: x vertex adjustment factor
        yadj : float
            vertex grid: y vertex adjustment factor
        xvdict: dict
            dictionary of x-vertices by cellnum ex. {0: (0,1,1,0)}
        yvdict: dict
            dictionary of y-vertices by cellnum ex. {0: (1,1,0,0)}
        distype: str
            model grid discretization type
    """

    def __new__(
        cls,
        delr=1.0,
        delc=1.0,
        xvdict=None,
        yvdict=None,
        lenuni=1,
        nlay=1,
        xul=None,
        yul=None,
        xadj=0.0,
        yadj=0.0,
        rotation=0.0,
        proj4_str=None,
        distype="structured",
    ):

        if distype == "structured":
            new = object.__new__(StructuredSpatialReference)
            new.__init__(
                delr=delr,
                delc=delc,
                lenuni=lenuni,
                nlay=nlay,
                xul=xul,
                yul=yul,
                rotation=rotation,
                proj4_str=proj4_str,
            )

        elif distype == "vertex":
            new = object.__new__(VertexSpatialReference)
            new.__init__(
                xvdict=xvdict,
                yvdict=yvdict,
                xadj=xadj,
                yadj=yadj,
                rotation=rotation,
                proj4_str=proj4_str,
                nlay=nlay,
            )

        elif distype == "unstructured":
            raise NotImplementedError(
                "Unstructured discretization not yet " "implemented"
            )

        else:
            raise TypeError(
                "Discretization type {} not " "supported".format(distype)
            )

        return new
