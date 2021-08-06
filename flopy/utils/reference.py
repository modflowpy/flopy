"""
Module spatial referencing for flopy model objects

"""
import json
import numpy as np
import os
import warnings

from collections import OrderedDict

__all__ = ["TemporalReference"]
# all other classes and methods in this module are deprecated

# web address of spatial reference dot org
srefhttp = "https://spatialreference.org"


class SpatialReference:
    """
    a class to locate a structured model grid in x-y space

    .. deprecated:: 3.2.11
        This class will be removed in version 3.3.5. Use
        :py:class:`flopy.discretization.structuredgrid.StructuredGrid` instead.

    Parameters
    ----------

    delr : numpy ndarray
        the model discretization delr vector
        (An array of spacings along a row)
    delc : numpy ndarray
        the model discretization delc vector
        (An array of spacings along a column)
    lenuni : int
        the length units flag from the discretization package
        (default 2)
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
        a PROJ4 string that identifies the grid in space. warning: case
        sensitive!

    units : string
        Units for the grid.  Must be either feet or meters

    epsg : int
        EPSG code that identifies the grid in space. Can be used in lieu of
        proj4. PROJ4 attribute will auto-populate if there is an internet
        connection(via get_proj4 method).
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

    vertices : 1D array
        1D array of cell vertices for whole grid in C-style (row-major) order
        (same as np.ravel())


    Notes
    -----

    xul and yul can be explicitly (re)set after SpatialReference
    instantiation, but only before any of the other attributes and methods are
    accessed

    """

    xul, yul = None, None
    xll, yll = None, None
    rotation = 0.0
    length_multiplier = 1.0
    origin_loc = "ul"  # or ll

    defaults = {
        "xul": None,
        "yul": None,
        "rotation": 0.0,
        "proj4_str": None,
        "units": None,
        "lenuni": 2,
        "length_multiplier": None,
        "source": "defaults",
    }

    lenuni_values = {"undefined": 0, "feet": 1, "meters": 2, "centimeters": 3}
    lenuni_text = {v: k for k, v in lenuni_values.items()}

    def __init__(
        self,
        delr=np.array([]),
        delc=np.array([]),
        lenuni=2,
        xul=None,
        yul=None,
        xll=None,
        yll=None,
        rotation=0.0,
        proj4_str=None,
        epsg=None,
        prj=None,
        units=None,
        length_multiplier=None,
    ):
        warnings.warn(
            "SpatialReference has been deprecated and will be removed in "
            "version 3.3.5. Use StructuredGrid instead.",
            category=DeprecationWarning,
        )

        for delrc in [delr, delc]:
            if isinstance(delrc, float) or isinstance(delrc, int):
                msg = (
                    "delr and delcs must be an array or sequences equal in "
                    "length to the number of rows/columns."
                )
                raise TypeError(msg)

        self.delc = np.atleast_1d(np.array(delc)).astype(
            np.float64
        )  # * length_multiplier
        self.delr = np.atleast_1d(np.array(delr)).astype(
            np.float64
        )  # * length_multiplier

        if self.delr.sum() == 0 or self.delc.sum() == 0:
            if xll is None or yll is None:
                msg = (
                    "Warning: no grid spacing or lower-left corner "
                    "supplied. Setting the offset with xul, yul requires "
                    "arguments for delr and delc. Origin will be set to "
                    "zero."
                )
                print(msg)
                xll, yll = 0, 0
                xul, yul = None, None

        self._lenuni = lenuni
        self._proj4_str = proj4_str

        self._epsg = epsg
        if epsg is not None:
            self._proj4_str = getproj4(self._epsg)
        self.prj = prj
        self._wkt = None
        self.crs = crs(prj=prj, epsg=epsg)

        self.supported_units = ["feet", "meters"]
        self._units = units
        self._length_multiplier = length_multiplier
        self._reset()
        self.set_spatialreference(xul, yul, xll, yll, rotation)

    @property
    def xll(self):
        if self.origin_loc == "ll":
            xll = self._xll if self._xll is not None else 0.0
        elif self.origin_loc == "ul":
            # calculate coords for lower left corner
            xll = self._xul - (
                np.sin(self.theta) * self.yedge[0] * self.length_multiplier
            )
        return xll

    @property
    def yll(self):
        if self.origin_loc == "ll":
            yll = self._yll if self._yll is not None else 0.0
        elif self.origin_loc == "ul":
            # calculate coords for lower left corner
            yll = self._yul - (
                np.cos(self.theta) * self.yedge[0] * self.length_multiplier
            )
        return yll

    @property
    def xul(self):
        if self.origin_loc == "ll":
            # calculate coords for upper left corner
            xul = self._xll + (
                np.sin(self.theta) * self.yedge[0] * self.length_multiplier
            )
        if self.origin_loc == "ul":
            # calculate coords for lower left corner
            xul = self._xul if self._xul is not None else 0.0
        return xul

    @property
    def yul(self):
        if self.origin_loc == "ll":
            # calculate coords for upper left corner
            yul = self._yll + (
                np.cos(self.theta) * self.yedge[0] * self.length_multiplier
            )

        if self.origin_loc == "ul":
            # calculate coords for lower left corner
            yul = self._yul if self._yul is not None else 0.0
        return yul

    @property
    def proj4_str(self):
        proj4_str = None
        if self._proj4_str is not None:
            if "epsg" in self._proj4_str.lower():
                proj4_str = self._proj4_str
                # set the epsg if proj4 specifies it
                tmp = [
                    i for i in self._proj4_str.split() if "epsg" in i.lower()
                ]
                self._epsg = int(tmp[0].split(":")[1])
            else:
                proj4_str = self._proj4_str
        elif self.epsg is not None:
            proj4_str = "epsg:{}".format(self.epsg)
        return proj4_str

    @property
    def epsg(self):
        # don't reset the proj4 string here
        # because proj4 attribute may already be populated
        # (with more details than getproj4 would return)
        # instead reset proj4 when epsg is set
        # (on init or setattr)
        return self._epsg

    @property
    def wkt(self):
        if self._wkt is None:
            if self.prj is not None:
                with open(self.prj) as src:
                    wkt = src.read()
            elif self.epsg is not None:
                wkt = getprj(self.epsg)
            else:
                return None
            return wkt
        else:
            return self._wkt

    @property
    def lenuni(self):
        return self._lenuni

    def _parse_units_from_proj4(self):
        units = None
        try:
            # need this because preserve_units doesn't seem to be
            # working for complex proj4 strings.  So if an
            # epsg code was passed, we have no choice, but if a
            # proj4 string was passed, we can just parse it
            if "EPSG" in self.proj4_str.upper():
                import pyproj

                crs = pyproj.Proj(self.proj4_str, preserve_units=True)
                proj_str = crs.srs
            else:
                proj_str = self.proj4_str
            # http://proj4.org/parameters.html#units
            # from proj4 source code
            # "us-ft", "0.304800609601219", "U.S. Surveyor's Foot",
            # "ft", "0.3048", "International Foot",
            if "units=m" in proj_str:
                units = "meters"
            elif (
                "units=ft" in proj_str
                or "units=us-ft" in proj_str
                or "to_meter=0.3048" in proj_str
            ):
                units = "feet"
            return units
        except:
            if self.proj4_str is not None:
                print(
                    "   could not parse units from {}".format(self.proj4_str)
                )

    @property
    def units(self):
        if self._units is not None:
            units = self._units.lower()
        else:
            units = self._parse_units_from_proj4()
        if units is None:
            # print("warning: assuming SpatialReference units are meters")
            units = "meters"
        assert units in self.supported_units
        return units

    @property
    def length_multiplier(self):
        """
        Attempt to identify multiplier for converting from
        model units to sr units, defaulting to 1.
        """
        lm = None
        if self._length_multiplier is not None:
            lm = self._length_multiplier
        else:
            if self.model_length_units == "feet":
                if self.units == "meters":
                    lm = 0.3048
                elif self.units == "feet":
                    lm = 1.0
            elif self.model_length_units == "meters":
                if self.units == "feet":
                    lm = 1 / 0.3048
                elif self.units == "meters":
                    lm = 1.0
            elif self.model_length_units == "centimeters":
                if self.units == "meters":
                    lm = 1 / 100.0
                elif self.units == "feet":
                    lm = 1 / 30.48
            else:  # model units unspecified; default to 1
                lm = 1.0
        return lm

    @property
    def model_length_units(self):
        return self.lenuni_text[self.lenuni]

    @property
    def bounds(self):
        """
        Return bounding box in shapely order.
        """
        xmin, xmax, ymin, ymax = self.get_extent()
        return xmin, ymin, xmax, ymax

    @classmethod
    def load(cls, namefile=None, reffile="usgs.model.reference"):
        """
        Attempts to load spatial reference information from
        the following files (in order):
        1) usgs.model.reference
        2) NAM file (header comment)
        3) SpatialReference.default dictionary
        """
        reffile = os.path.join(os.path.split(namefile)[0], reffile)
        d = SpatialReference.read_usgs_model_reference_file(reffile)
        if d is not None:
            return d
        d = SpatialReference.attribs_from_namfile_header(namefile)
        if d is not None:
            return d
        else:
            return SpatialReference.defaults

    @staticmethod
    def attribs_from_namfile_header(namefile):
        # check for reference info in the nam file header
        d = SpatialReference.defaults.copy()
        d["source"] = "namfile"
        if namefile is None:
            return None
        header = []
        with open(namefile, "r") as f:
            for line in f:
                if not line.startswith("#"):
                    break
                header.extend(line.strip().replace("#", "").split(";"))

        for item in header:
            if "xul" in item.lower():
                try:
                    d["xul"] = float(item.split(":")[1])
                except:
                    print("   could not parse xul in {}".format(namefile))
            elif "yul" in item.lower():
                try:
                    d["yul"] = float(item.split(":")[1])
                except:
                    print("   could not parse yul in {}".format(namefile))
            elif "rotation" in item.lower():
                try:
                    d["rotation"] = float(item.split(":")[1])
                except:
                    print("   could not parse rotation in {}".format(namefile))
            elif "proj4_str" in item.lower():
                try:
                    proj4_str = ":".join(item.split(":")[1:]).strip()
                    if proj4_str.lower() == "none":
                        proj4_str = None
                    d["proj4_str"] = proj4_str
                except:
                    print(
                        "   could not parse proj4_str in {}".format(namefile)
                    )
            elif "start" in item.lower():
                try:
                    d["start_datetime"] = item.split(":")[1].strip()
                except:
                    print("   could not parse start in {}".format(namefile))

            # spatial reference length units
            elif "units" in item.lower():
                d["units"] = item.split(":")[1].strip()
            # model length units
            elif "lenuni" in item.lower():
                d["lenuni"] = int(item.split(":")[1].strip())
            # multiplier for converting from model length units to sr length units
            elif "length_multiplier" in item.lower():
                d["length_multiplier"] = float(item.split(":")[1].strip())
        return d

    @staticmethod
    def read_usgs_model_reference_file(reffile="usgs.model.reference"):
        """
        read spatial reference info from the usgs.model.reference file
        https://water.usgs.gov/ogw/policy/gw-model/modelers-setup.html
        """

        ITMUNI = {
            0: "undefined",
            1: "seconds",
            2: "minutes",
            3: "hours",
            4: "days",
            5: "years",
        }
        itmuni_values = {v: k for k, v in ITMUNI.items()}

        d = SpatialReference.defaults.copy()
        d["source"] = "usgs.model.reference"
        # discard default to avoid confusion with epsg code if entered
        d.pop("proj4_str")
        if os.path.exists(reffile):
            with open(reffile) as fref:
                for line in fref:
                    if len(line) > 1:
                        if line.strip()[0] != "#":
                            info = line.strip().split("#")[0].split()
                            if len(info) > 1:
                                d[info[0].lower()] = " ".join(info[1:])
            d["xul"] = float(d["xul"])
            d["yul"] = float(d["yul"])
            d["rotation"] = float(d["rotation"])

            # convert the model.reference text to a lenuni value
            # (these are the model length units)
            if "length_units" in d.keys():
                d["lenuni"] = SpatialReference.lenuni_values[d["length_units"]]
            if "time_units" in d.keys():
                d["itmuni"] = itmuni_values[d["time_units"]]
            if "start_date" in d.keys():
                start_datetime = d.pop("start_date")
                if "start_time" in d.keys():
                    start_datetime += " {}".format(d.pop("start_time"))
                d["start_datetime"] = start_datetime
            if "epsg" in d.keys():
                try:
                    d["epsg"] = int(d["epsg"])
                except Exception as e:
                    raise Exception(
                        "error reading epsg code from file:\n" + str(e)
                    )
            # this prioritizes epsg over proj4 if both are given
            # (otherwise 'proj4' entry will be dropped below)
            elif "proj4" in d.keys():
                d["proj4_str"] = d["proj4"]

            # drop any other items that aren't used in sr class
            d = {
                k: v
                for k, v in d.items()
                if k.lower() in SpatialReference.defaults.keys()
                or k.lower() in {"epsg", "start_datetime", "itmuni", "source"}
            }
            return d
        else:
            return None

    def __setattr__(self, key, value):
        reset = True
        if key == "delr":
            super().__setattr__("delr", np.atleast_1d(np.array(value)))
        elif key == "delc":
            super().__setattr__("delc", np.atleast_1d(np.array(value)))
        elif key == "xul":
            super().__setattr__("_xul", float(value))
            self.origin_loc = "ul"
        elif key == "yul":
            super().__setattr__("_yul", float(value))
            self.origin_loc = "ul"
        elif key == "xll":
            super().__setattr__("_xll", float(value))
            self.origin_loc = "ll"
        elif key == "yll":
            super().__setattr__("_yll", float(value))
            self.origin_loc = "ll"
        elif key == "length_multiplier":
            super().__setattr__("_length_multiplier", float(value))
            # self.set_origin(xul=self.xul, yul=self.yul, xll=self.xll,
            #                yll=self.yll)
        elif key == "rotation":
            super().__setattr__("rotation", float(value))
            # self.set_origin(xul=self.xul, yul=self.yul, xll=self.xll,
            #                yll=self.yll)
        elif key == "lenuni":
            super().__setattr__("_lenuni", int(value))
            # self.set_origin(xul=self.xul, yul=self.yul, xll=self.xll,
            #                yll=self.yll)
        elif key == "units":
            value = value.lower()
            assert value in self.supported_units
            super().__setattr__("_units", value)
        elif key == "proj4_str":
            super().__setattr__("_proj4_str", value)
            # reset the units and epsg
            units = self._parse_units_from_proj4()
            if units is not None:
                self._units = units
            self._epsg = None
        elif key == "epsg":
            super().__setattr__("_epsg", value)
            # reset the units and proj4
            self._units = None
            self._proj4_str = getproj4(self._epsg)
            self.crs = crs(epsg=value)
        elif key == "prj":
            super().__setattr__("prj", value)
            # translation to proj4 strings in crs class not robust yet
            # leave units and proj4 alone for now.
            self.crs = crs(prj=value, epsg=self.epsg)
        else:
            super().__setattr__(key, value)
            reset = False
        if reset:
            self._reset()

    def reset(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return

    def _reset(self):
        self._xgrid = None
        self._ygrid = None
        self._ycentergrid = None
        self._xcentergrid = None
        self._vertices = None
        return

    @property
    def nrow(self):
        return self.delc.shape[0]

    @property
    def ncol(self):
        return self.delr.shape[0]

    def __eq__(self, other):
        if not isinstance(other, SpatialReference):
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
    def from_namfile(cls, namefile):
        attribs = SpatialReference.attribs_from_namfile_header(namefile)
        try:
            attribs.pop("start_datetime")
        except:
            print("   could not remove start_datetime")

        return cls(**attribs)

    @classmethod
    def from_gridspec(cls, gridspec_file, lenuni=0):
        f = open(gridspec_file, "r")
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
                if "*" in r:
                    rraw = r.split("*")
                    for n in range(int(rraw[0])):
                        delc.append(float(rraw[1]))
                        i += 1
                else:
                    delc.append(float(r))
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

    def set_spatialreference(
        self, xul=None, yul=None, xll=None, yll=None, rotation=0.0
    ):
        """
        set spatial reference - can be called from model instance

        """
        if xul is not None and xll is not None:
            msg = (
                "Both xul and xll entered. Please enter either xul, yul or "
                "xll, yll."
            )
            raise ValueError(msg)
        if yul is not None and yll is not None:
            msg = (
                "Both yul and yll entered. Please enter either xul, yul or "
                "xll, yll."
            )
            raise ValueError(msg)
        # set the origin priority based on the left corner specified
        # (the other left corner will be calculated).  If none are specified
        # then default to upper left
        if xul is None and yul is None and xll is None and yll is None:
            self.origin_loc = "ul"
            xul = 0.0
            yul = self.delc.sum()
        elif xll is not None:
            self.origin_loc = "ll"
        else:
            self.origin_loc = "ul"

        self.rotation = rotation
        self._xll = xll if xll is not None else 0.0
        self._yll = yll if yll is not None else 0.0
        self._xul = xul if xul is not None else 0.0
        self._yul = yul if yul is not None else 0.0
        # self.set_origin(xul, yul, xll, yll)
        return

    def __repr__(self):
        s = "xul:{0:<.10G}; yul:{1:<.10G}; rotation:{2:<G}; ".format(
            self.xul, self.yul, self.rotation
        )
        s += "proj4_str:{0}; ".format(self.proj4_str)
        s += "units:{0}; ".format(self.units)
        s += "lenuni:{0}; ".format(self.lenuni)
        s += "length_multiplier:{}".format(self.length_multiplier)
        return s

    def set_origin(self, xul=None, yul=None, xll=None, yll=None):
        if self.origin_loc == "ll":
            # calculate coords for upper left corner
            self._xll = xll if xll is not None else 0.0
            self.yll = yll if yll is not None else 0.0
            self.xul = self._xll + (
                np.sin(self.theta) * self.yedge[0] * self.length_multiplier
            )
            self.yul = self.yll + (
                np.cos(self.theta) * self.yedge[0] * self.length_multiplier
            )

        if self.origin_loc == "ul":
            # calculate coords for lower left corner
            self.xul = xul if xul is not None else 0.0
            self.yul = yul if yul is not None else 0.0
            self._xll = self.xul - (
                np.sin(self.theta) * self.yedge[0] * self.length_multiplier
            )
            self.yll = self.yul - (
                np.cos(self.theta) * self.yedge[0] * self.length_multiplier
            )
        self._reset()
        return

    @property
    def theta(self):
        return -self.rotation * np.pi / 180.0

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
        self._xcentergrid, self._ycentergrid = self.transform(
            self._xcentergrid, self._ycentergrid
        )

    def _set_xygrid(self):
        self._xgrid, self._ygrid = np.meshgrid(self.xedge, self.yedge)
        self._xgrid, self._ygrid = self.transform(self._xgrid, self._ygrid)

    @staticmethod
    def rotate(x, y, theta, xorigin=0.0, yorigin=0.0):
        """
        Given x and y array-like values calculate the rotation about an
        arbitrary origin and then return the rotated coordinates.  theta is in
        degrees.

        """
        # jwhite changed on Oct 11 2016 - rotation is now positive CCW
        # theta = -theta * np.pi / 180.
        theta = theta * np.pi / 180.0

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

    def transform(self, x, y, inverse=False):
        """
        Given x and y array-like values, apply rotation, scale and offset,
        to convert them from model coordinates to real-world coordinates.
        """
        if isinstance(x, list):
            x = np.array(x)
            y = np.array(y)
        if not np.isscalar(x):
            x, y = x.copy(), y.copy()

        if not inverse:
            x *= self.length_multiplier
            y *= self.length_multiplier
            x += self.xll
            y += self.yll
            x, y = SpatialReference.rotate(
                x, y, theta=self.rotation, xorigin=self.xll, yorigin=self.yll
            )
        else:
            x, y = SpatialReference.rotate(
                x, y, -self.rotation, self.xll, self.yll
            )
            x -= self.xll
            y -= self.yll
            x /= self.length_multiplier
            y /= self.length_multiplier
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
        Get the grid lines as a list

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

        # horizontal lines
        for i in range(self.nrow + 1):
            x0 = xmin
            x1 = xmax
            y0 = self.yedge[i]
            y1 = y0
            x0r, y0r = self.transform(x0, y0)
            x1r, y1r = self.transform(x1, y1)
            lines.append([(x0r, y0r), (x1r, y1r)])
        return lines

    def get_grid_line_collection(self, **kwargs):
        """
        Get a LineCollection of the grid

        """
        from ..plot import ModelMap

        map = ModelMap(sr=self)
        lc = map.plot_grid(**kwargs)
        return lc

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
                self.xul * self.length_multiplier,
                self.yul * self.length_multiplier,
                self.rotation,
            )
        )

        for r in self.delr:
            f.write("{0:15.6E} ".format(r))
        f.write("\n")
        for c in self.delc:
            f.write("{0:15.6E} ".format(c))
        f.write("\n")
        return

    def write_shapefile(self, filename="grid.shp", epsg=None, prj=None):
        """Write a shapefile of the grid with just the row and column attributes"""
        from ..export.shapefile_utils import write_grid_shapefile

        if epsg is None and prj is None:
            epsg = self.epsg
        write_grid_shapefile(
            filename, self, array_dict={}, nan_val=-1.0e9, epsg=epsg, prj=prj
        )

    def get_vertices(self, i, j):
        """Get vertices for a single cell or sequence if i, j locations."""
        pts = []
        xgrid, ygrid = self.xgrid, self.ygrid
        pts.append([xgrid[i, j], ygrid[i, j]])
        pts.append([xgrid[i + 1, j], ygrid[i + 1, j]])
        pts.append([xgrid[i + 1, j + 1], ygrid[i + 1, j + 1]])
        pts.append([xgrid[i, j + 1], ygrid[i, j + 1]])
        pts.append([xgrid[i, j], ygrid[i, j]])
        if np.isscalar(i):
            return pts
        else:
            vrts = np.array(pts).transpose([2, 0, 1])
            return [v.tolist() for v in vrts]

    def get_rc(self, x, y):
        return self.get_ij(x, y)

    def get_ij(self, x, y):
        """Return the row and column of a point or sequence of points
        in real-world coordinates.

        Parameters
        ----------
        x : scalar or sequence of x coordinates
        y : scalar or sequence of y coordinates

        Returns
        -------
        i : row or sequence of rows (zero-based)
        j : column or sequence of columns (zero-based)
        """
        if np.isscalar(x):
            c = (np.abs(self.xcentergrid[0] - x)).argmin()
            r = (np.abs(self.ycentergrid[:, 0] - y)).argmin()
        else:
            xcp = np.array([self.xcentergrid[0]] * (len(x)))
            ycp = np.array([self.ycentergrid[:, 0]] * (len(x)))
            c = (np.abs(xcp.transpose() - x)).argmin(axis=0)
            r = (np.abs(ycp.transpose() - y)).argmin(axis=0)
        return r, c

    def get_grid_map_plotter(self, **kwargs):
        """
        Create a QuadMesh plotting object for this grid

        Returns
        -------
        quadmesh : matplotlib.collections.QuadMesh

        """
        try:
            from matplotlib.collections import QuadMesh
        except:
            raise ImportError(
                "matplotlib must be installed to use get_grid_map_plotter()"
            )

        verts = np.vstack((self.xgrid.flatten(), self.ygrid.flatten())).T
        qm = QuadMesh(self.ncol, self.nrow, verts)
        return qm

    def plot_array(self, a, ax=None, **kwargs):
        """
        Create a QuadMesh plot of the specified array using pcolormesh

        Parameters
        ----------
        a : np.ndarray

        Returns
        -------
        quadmesh : matplotlib.collections.QuadMesh

        """
        try:
            import matplotlib.pyplot as plt
        except:
            raise ImportError(
                "matplotlib must be installed to use reference.plot_array()"
            )

        if ax is None:
            ax = plt.gca()
        qm = ax.pcolormesh(self.xgrid, self.ygrid, a, **kwargs)
        return qm

    def export_array(
        self, filename, a, nodata=-9999, fieldname="value", **kwargs
    ):
        """
        Write a numpy array to Arc Ascii grid or shapefile with the
        model reference.

        Parameters
        ----------
        filename : str
            Path of output file. Export format is determined by
            file extension.
            '.asc'  Arc Ascii grid
            '.tif'  GeoTIFF (requires rasterio package)
            '.shp'  Shapefile
        a : 2D numpy.ndarray
            Array to export
        nodata : scalar
            Value to assign to np.nan entries (default -9999)
        fieldname : str
            Attribute field name for array values (shapefile export only).
            (default 'values')
        kwargs:
            keyword arguments to np.savetxt (ascii)
            rasterio.open (GeoTIFF)
            or flopy.export.shapefile_utils.write_grid_shapefile2

        Notes
        -----
        Rotated grids will be either be unrotated prior to export,
        using scipy.ndimage.rotate (Arc Ascii format) or rotation will be
        included in their transform property (GeoTiff format). In either case
        the pixels will be displayed in the (unrotated) projected geographic coordinate system,
        so the pixels will no longer align exactly with the model grid
        (as displayed from a shapefile, for example). A key difference between
        Arc Ascii and GeoTiff (besides disk usage) is that the
        unrotated Arc Ascii will have a different grid size, whereas the GeoTiff
        will have the same number of rows and pixels as the original.

        """

        if filename.lower().endswith(".asc"):
            if (
                len(np.unique(self.delr)) != len(np.unique(self.delc)) != 1
                or self.delr[0] != self.delc[0]
            ):
                raise ValueError("Arc ascii arrays require a uniform grid.")

            xll, yll = self.xll, self.yll
            cellsize = self.delr[0] * self.length_multiplier
            fmt = kwargs.get("fmt", "%.18e")
            a = a.copy()
            a[np.isnan(a)] = nodata
            if self.rotation != 0:
                try:
                    from scipy.ndimage import rotate

                    a = rotate(a, self.rotation, cval=nodata)
                    height_rot, width_rot = a.shape
                    xmin, ymin, xmax, ymax = self.bounds
                    dx = (xmax - xmin) / width_rot
                    dy = (ymax - ymin) / height_rot
                    cellsize = np.max((dx, dy))
                    # cellsize = np.cos(np.radians(self.rotation)) * cellsize
                    xll, yll = xmin, ymin
                except ImportError:
                    print("scipy package required to export rotated grid.")

            filename = (
                ".".join(filename.split(".")[:-1]) + ".asc"
            )  # enforce .asc ending
            nrow, ncol = a.shape
            a[np.isnan(a)] = nodata
            txt = "ncols  {:d}\n".format(ncol)
            txt += "nrows  {:d}\n".format(nrow)
            txt += "xllcorner  {:f}\n".format(xll)
            txt += "yllcorner  {:f}\n".format(yll)
            txt += "cellsize  {}\n".format(cellsize)
            # ensure that nodata fmt consistent w values
            txt += "NODATA_value  {}\n".format(fmt) % (nodata)
            with open(filename, "w") as output:
                output.write(txt)
            with open(filename, "ab") as output:
                np.savetxt(output, a, **kwargs)
            print("wrote {}".format(filename))

        elif filename.lower().endswith(".tif"):
            if (
                len(np.unique(self.delr)) != len(np.unique(self.delc)) != 1
                or self.delr[0] != self.delc[0]
            ):
                raise ValueError("GeoTIFF export require a uniform grid.")
            try:
                import rasterio
                from rasterio import Affine
            except:
                print("GeoTIFF export requires the rasterio package.")
                return
            dxdy = self.delc[0] * self.length_multiplier
            trans = (
                Affine.translation(self.xul, self.yul)
                * Affine.rotation(self.rotation)
                * Affine.scale(dxdy, -dxdy)
            )

            # third dimension is the number of bands
            a = a.copy()
            if len(a.shape) == 2:
                a = np.reshape(a, (1, a.shape[0], a.shape[1]))
            if a.dtype.name == "int64":
                a = a.astype("int32")
                dtype = rasterio.int32
            elif a.dtype.name == "int32":
                dtype = rasterio.int32
            elif a.dtype.name == "float64":
                dtype = rasterio.float64
            elif a.dtype.name == "float32":
                dtype = rasterio.float32
            else:
                msg = 'ERROR: invalid dtype "{}"'.format(a.dtype.name)
                raise TypeError(msg)

            meta = {
                "count": a.shape[0],
                "width": a.shape[2],
                "height": a.shape[1],
                "nodata": nodata,
                "dtype": dtype,
                "driver": "GTiff",
                "crs": self.proj4_str,
                "transform": trans,
            }
            meta.update(kwargs)
            with rasterio.open(filename, "w", **meta) as dst:
                dst.write(a)
            print("wrote {}".format(filename))

        elif filename.lower().endswith(".shp"):
            from ..export.shapefile_utils import write_grid_shapefile

            epsg = kwargs.get("epsg", None)
            prj = kwargs.get("prj", None)
            if epsg is None and prj is None:
                epsg = self.epsg
            write_grid_shapefile(
                filename,
                self,
                array_dict={fieldname: a},
                nan_val=nodata,
                epsg=epsg,
                prj=prj,
            )

    def export_contours(
        self,
        filename,
        contours,
        fieldname="level",
        epsg=None,
        prj=None,
        **kwargs
    ):
        """
        Convert matplotlib contour plot object to shapefile.

        Parameters
        ----------
        filename : str
            path of output shapefile
        contours : matplotlib.contour.QuadContourSet or list of them
            (object returned by matplotlib.pyplot.contour)
        epsg : int
            EPSG code. See https://www.epsg-registry.org/ or spatialreference.org
        prj : str
            Existing projection file to be used with new shapefile.
        **kwargs : key-word arguments to flopy.export.shapefile_utils.recarray2shp

        Returns
        -------
        df : dataframe of shapefile contents

        """
        from flopy.utils.geometry import LineString
        from flopy.export.shapefile_utils import recarray2shp

        if not isinstance(contours, list):
            contours = [contours]

        if epsg is None:
            epsg = self._epsg
        if prj is None:
            prj = self.proj4_str

        geoms = []
        level = []
        for ctr in contours:
            levels = ctr.levels
            for i, c in enumerate(ctr.collections):
                paths = c.get_paths()
                geoms += [LineString(p.vertices) for p in paths]
                level += list(np.ones(len(paths)) * levels[i])

        # convert the dictionary to a recarray
        ra = np.array(level, dtype=[(fieldname, float)]).view(np.recarray)

        recarray2shp(ra, geoms, filename, epsg=epsg, prj=prj, **kwargs)

    def export_array_contours(
        self,
        filename,
        a,
        fieldname="level",
        interval=None,
        levels=None,
        maxlevels=1000,
        epsg=None,
        prj=None,
        **kwargs
    ):
        """
        Contour an array using matplotlib; write shapefile of contours.

        Parameters
        ----------
        filename : str
            Path of output file with '.shp' extension.
        a : 2D numpy array
            Array to contour
        epsg : int
            EPSG code. See https://www.epsg-registry.org/ or spatialreference.org
        prj : str
            Existing projection file to be used with new shapefile.
        **kwargs : key-word arguments to flopy.export.shapefile_utils.recarray2shp

        """
        try:
            import matplotlib.pyplot as plt
        except:
            raise ImportError(
                "matplotlib must be installed to "
                "use cvfd_to_patch_collection()"
            )

        if epsg is None:
            epsg = self._epsg
        if prj is None:
            prj = self.proj4_str

        if interval is not None:
            vmin = np.nanmin(a)
            vmax = np.nanmax(a)
            nlevels = np.round(np.abs(vmax - vmin) / interval, 2)
            msg = "{:.0f} levels at interval of {} > " "maxlevels = {}".format(
                nlevels, interval, maxlevels
            )
            assert nlevels < maxlevels, msg
            levels = np.arange(vmin, vmax, interval)
        fig, ax = plt.subplots()
        ctr = self.contour_array(ax, a, levels=levels)
        self.export_contours(filename, ctr, fieldname, epsg, prj, **kwargs)
        plt.close()

    def contour_array(self, ax, a, **kwargs):
        """
        Create a QuadMesh plot of the specified array using pcolormesh

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            ax to add the contours

        a : np.ndarray
            array to contour

        Returns
        -------
        contour_set : ContourSet

        """
        from flopy.plot import ModelMap

        kwargs["ax"] = ax
        mm = ModelMap(sr=self)
        contour_set = mm.contour_array(a=a, **kwargs)

        return contour_set

    @property
    def vertices(self):
        """
        Returns a list of vertices for
        """
        if self._vertices is None:
            self._set_vertices()
        return self._vertices

    def _set_vertices(self):
        """
        Populate vertices for the whole grid
        """
        jj, ii = np.meshgrid(range(self.ncol), range(self.nrow))
        jj, ii = jj.ravel(), ii.ravel()
        self._vertices = self.get_vertices(ii, jj)

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

    def get_2d_vertex_connectivity(self):
        """
        Create the cell 2d vertices array and the iverts index array.  These
        are the same form as the ones used to instantiate an unstructured
        spatial reference.

        Returns
        -------

        verts : ndarray
            array of x and y coordinates for the grid vertices

        iverts : list
            a list with a list of vertex indices for each cell in clockwise
            order starting with the upper left corner

        """
        x = self.xgrid.flatten()
        y = self.ygrid.flatten()
        nrowvert = self.nrow + 1
        ncolvert = self.ncol + 1
        npoints = nrowvert * ncolvert
        verts = np.empty((npoints, 2), dtype=float)
        verts[:, 0] = x
        verts[:, 1] = y
        iverts = []
        for i in range(self.nrow):
            for j in range(self.ncol):
                iv1 = i * ncolvert + j  # upper left point number
                iv2 = iv1 + 1
                iv4 = (i + 1) * ncolvert + j
                iv3 = iv4 + 1
                iverts.append([iv1, iv2, iv3, iv4])
        return verts, iverts

    def get_3d_shared_vertex_connectivity(self, nlay, botm, ibound=None):

        # get the x and y points for the grid
        x = self.xgrid.flatten()
        y = self.ygrid.flatten()

        # set the size of the vertex grid
        nrowvert = self.nrow + 1
        ncolvert = self.ncol + 1
        nlayvert = nlay + 1
        nrvncv = nrowvert * ncolvert
        npoints = nrvncv * nlayvert

        # create and fill a 3d points array for the grid
        verts = np.empty((npoints, 3), dtype=float)
        verts[:, 0] = np.tile(x, nlayvert)
        verts[:, 1] = np.tile(y, nlayvert)
        istart = 0
        istop = nrvncv
        for k in range(nlay + 1):
            verts[istart:istop, 2] = self.interpolate(
                botm[k], verts[istart:istop, :2], method="linear"
            )
            istart = istop
            istop = istart + nrvncv

        # create the list of points comprising each cell. points must be
        # listed a specific way according to vtk requirements.
        iverts = []
        for k in range(nlay):
            koffset = k * nrvncv
            for i in range(self.nrow):
                for j in range(self.ncol):
                    if ibound is not None:
                        if ibound[k, i, j] == 0:
                            continue
                    iv1 = i * ncolvert + j + koffset
                    iv2 = iv1 + 1
                    iv4 = (i + 1) * ncolvert + j + koffset
                    iv3 = iv4 + 1
                    iverts.append(
                        [
                            iv4 + nrvncv,
                            iv3 + nrvncv,
                            iv1 + nrvncv,
                            iv2 + nrvncv,
                            iv4,
                            iv3,
                            iv1,
                            iv2,
                        ]
                    )

        # renumber and reduce the vertices if ibound_filter
        if ibound is not None:

            # go through the vertex list and mark vertices that are used
            ivertrenum = np.zeros(npoints, dtype=int)
            for vlist in iverts:
                for iv in vlist:
                    # mark vertices that are actually used
                    ivertrenum[iv] = 1

            # renumber vertices that are used, skip those that are not
            inum = 0
            for i in range(npoints):
                if ivertrenum[i] > 0:
                    inum += 1
                    ivertrenum[i] = inum
            ivertrenum -= 1

            # reassign the vertex list using the new vertex numbers
            iverts2 = []
            for vlist in iverts:
                vlist2 = []
                for iv in vlist:
                    vlist2.append(ivertrenum[iv])
                iverts2.append(vlist2)
            iverts = iverts2
            idx = np.where(ivertrenum >= 0)
            verts = verts[idx]

        return verts, iverts

    def get_3d_vertex_connectivity(self, nlay, top, bot, ibound=None):
        if ibound is None:
            ncells = nlay * self.nrow * self.ncol
            ibound = np.ones((nlay, self.nrow, self.ncol), dtype=int)
        else:
            ncells = (ibound != 0).sum()
        npoints = ncells * 8
        verts = np.empty((npoints, 3), dtype=float)
        iverts = []
        ipoint = 0
        for k in range(nlay):
            for i in range(self.nrow):
                for j in range(self.ncol):
                    if ibound[k, i, j] == 0:
                        continue

                    ivert = []
                    pts = self.get_vertices(i, j)
                    pt0, pt1, pt2, pt3, pt0 = pts

                    z = bot[k, i, j]

                    verts[ipoint, 0:2] = np.array(pt1)
                    verts[ipoint, 2] = z
                    ivert.append(ipoint)
                    ipoint += 1

                    verts[ipoint, 0:2] = np.array(pt2)
                    verts[ipoint, 2] = z
                    ivert.append(ipoint)
                    ipoint += 1

                    verts[ipoint, 0:2] = np.array(pt0)
                    verts[ipoint, 2] = z
                    ivert.append(ipoint)
                    ipoint += 1

                    verts[ipoint, 0:2] = np.array(pt3)
                    verts[ipoint, 2] = z
                    ivert.append(ipoint)
                    ipoint += 1

                    z = top[k, i, j]

                    verts[ipoint, 0:2] = np.array(pt1)
                    verts[ipoint, 2] = z
                    ivert.append(ipoint)
                    ipoint += 1

                    verts[ipoint, 0:2] = np.array(pt2)
                    verts[ipoint, 2] = z
                    ivert.append(ipoint)
                    ipoint += 1

                    verts[ipoint, 0:2] = np.array(pt0)
                    verts[ipoint, 2] = z
                    ivert.append(ipoint)
                    ipoint += 1

                    verts[ipoint, 0:2] = np.array(pt3)
                    verts[ipoint, 2] = z
                    ivert.append(ipoint)
                    ipoint += 1

                    iverts.append(ivert)

        return verts, iverts


class SpatialReferenceUnstructured(SpatialReference):
    """
    a class to locate an unstructured model grid in x-y space

    .. deprecated:: 3.2.11
        This class will be removed in version 3.3.5. Use
        :py:class:`flopy.discretization.vertexgrid.VertexGrid` instead.

    Parameters
    ----------

    verts : ndarray
        2d array of x and y points.

    iverts : list of lists
        should be of len(ncells) with a list of vertex numbers for each cell

    ncpl : ndarray
        array containing the number of cells per layer.  ncpl.sum() must be
        equal to the total number of cells in the grid.

    layered : boolean
        flag to indicated that the grid is layered.  In this case, the vertices
        define the grid for single layer, and all layers use this same grid.
        In this case the ncpl value for each layer must equal len(iverts).
        If not layered, then verts and iverts are specified for all cells and
        all layers in the grid.  In this case, npcl.sum() must equal
        len(iverts).

    lenuni : int
        the length units flag from the discretization package

    proj4_str: str
        a PROJ4 string that identifies the grid in space. warning: case
        sensitive!

    units : string
        Units for the grid.  Must be either feet or meters

    epsg : int
        EPSG code that identifies the grid in space. Can be used in lieu of
        proj4. PROJ4 attribute will auto-populate if there is an internet
        connection(via get_proj4 method).
        See https://www.epsg-registry.org/ or spatialreference.org

    length_multiplier : float
        multiplier to convert model units to spatial reference units.
        delr and delc above will be multiplied by this value. (default=1.)

    Attributes
    ----------
    xcenter : ndarray
        array of x cell centers

    ycenter : ndarray
        array of y cell centers

    Notes
    -----

    """

    def __init__(
        self,
        xc,
        yc,
        verts,
        iverts,
        ncpl,
        layered=True,
        lenuni=1,
        proj4_str=None,
        epsg=None,
        units=None,
        length_multiplier=1.0,
    ):
        warnings.warn(
            "SpatialReferenceUnstructured has been deprecated and will be "
            "removed in version 3.3.5. Use VertexGrid instead.",
            category=DeprecationWarning,
        )
        self.xc = xc
        self.yc = yc
        self.verts = verts
        self.iverts = iverts
        self.ncpl = ncpl
        self.layered = layered
        self._lenuni = lenuni
        self._proj4_str = proj4_str
        self._epsg = epsg
        if epsg is not None:
            self._proj4_str = getproj4(epsg)
        self.supported_units = ["feet", "meters"]
        self._units = units
        self._length_multiplier = length_multiplier

        # set defaults
        self._xul = 0.0
        self._yul = 0.0
        self.rotation = 0.0

        if self.layered:
            assert all([n == len(iverts) for n in ncpl])
            assert self.xc.shape[0] == self.ncpl[0]
            assert self.yc.shape[0] == self.ncpl[0]
        else:
            msg = "Length of iverts must equal ncpl.sum ({} {})".format(
                len(iverts), ncpl
            )
            assert len(iverts) == ncpl.sum(), msg
            assert self.xc.shape[0] == self.ncpl.sum()
            assert self.yc.shape[0] == self.ncpl.sum()
        return

    @property
    def grid_type(self):
        return "unstructured"

    def write_shapefile(self, filename="grid.shp"):
        """
        Write shapefile of the grid

        Parameters
        ----------
        filename : string
            filename for shapefile

        Returns
        -------

        """
        raise NotImplementedError()
        return

    def write_gridSpec(self, filename):
        """
        Write a PEST-style grid specification file

        Parameters
        ----------
        filename : string
            filename for grid specification file

        Returns
        -------

        """
        raise NotImplementedError()
        return

    @classmethod
    def from_gridspec(cls, fname):
        """
        Create a new SpatialReferenceUnstructured grid from an PEST
        grid specification file

        Parameters
        ----------
        fname : string
            File name for grid specification file

        Returns
        -------
            sru : flopy.utils.reference.SpatialReferenceUnstructured

        """
        raise NotImplementedError()
        return

    @classmethod
    def from_argus_export(cls, fname, nlay=1):
        """
        Create a new SpatialReferenceUnstructured grid from an Argus One
        Trimesh file

        Parameters
        ----------
        fname : string
            File name

        nlay : int
            Number of layers to create

        Returns
        -------
            sru : flopy.utils.reference.SpatialReferenceUnstructured

        """
        from ..utils.geometry import get_polygon_centroid

        f = open(fname, "r")
        line = f.readline()
        ll = line.split()
        ncells, nverts = ll[0:2]
        ncells = int(ncells)
        nverts = int(nverts)
        verts = np.empty((nverts, 2), dtype=float)
        xc = np.empty((ncells), dtype=float)
        yc = np.empty((ncells), dtype=float)

        # read the vertices
        f.readline()
        for ivert in range(nverts):
            line = f.readline()
            ll = line.split()
            c, iv, x, y = ll[0:4]
            verts[ivert, 0] = x
            verts[ivert, 1] = y

        # read the cell information and create iverts, xc, and yc
        iverts = []
        for icell in range(ncells):
            line = f.readline()
            ll = line.split()
            ivlist = []
            for ic in ll[2:5]:
                ivlist.append(int(ic) - 1)
            if ivlist[0] != ivlist[-1]:
                ivlist.append(ivlist[0])
            iverts.append(ivlist)
            xc[icell], yc[icell] = get_polygon_centroid(verts[ivlist, :])

        # close file and return spatial reference
        f.close()
        return cls(xc, yc, verts, iverts, np.array(nlay * [len(iverts)]))

    def __setattr__(self, key, value):
        super(SpatialReference, self).__setattr__(key, value)
        return

    def get_extent(self):
        """
        Get the extent of the grid

        Returns
        -------
        extent : tuple
            min and max grid coordinates

        """
        xmin = self.verts[:, 0].min()
        xmax = self.verts[:, 0].max()
        ymin = self.verts[:, 1].min()
        ymax = self.verts[:, 1].max()
        return (xmin, xmax, ymin, ymax)

    def get_xcenter_array(self):
        """
        Return a numpy one-dimensional float array that has the cell center x
        coordinate for every cell in the grid in model space - not offset or
        rotated.

        """
        return self.xc

    def get_ycenter_array(self):
        """
        Return a numpy one-dimensional float array that has the cell center x
        coordinate for every cell in the grid in model space - not offset of
        rotated.

        """
        return self.yc

    def plot_array(self, a, ax=None):
        """
        Create a QuadMesh plot of the specified array using patches

        Parameters
        ----------
        a : np.ndarray

        Returns
        -------
        pc : matplotlib.collections.PatchCollection

        """
        from ..plot import ModelMap

        pmv = ModelMap(sr=self, ax=ax)
        pc = pmv.plot_array(a)

        return pc

    def get_grid_line_collection(self, **kwargs):
        """
        Get a patch collection of the grid

        """
        from ..plot import ModelMap

        ax = kwargs.pop("ax", None)
        pmv = ModelMap(sr=self, ax=ax)
        pc = pmv.plot_grid(**kwargs)
        return pc

    def contour_array(self, ax, a, **kwargs):
        """
        Create a QuadMesh plot of the specified array using pcolormesh

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            ax to add the contours

        a : np.ndarray
            array to contour

        Returns
        -------
        contour_set : ContourSet

        """
        contour_set = ax.tricontour(self.xcenter, self.ycenter, a, **kwargs)
        return contour_set


class TemporalReference:
    """
    For now, just a container to hold start time and time units files
    outside of DIS package.
    """

    defaults = {"itmuni": 4, "start_datetime": "01-01-1970"}

    itmuni_values = {
        "undefined": 0,
        "seconds": 1,
        "minutes": 2,
        "hours": 3,
        "days": 4,
        "years": 5,
    }

    itmuni_text = {v: k for k, v in itmuni_values.items()}

    def __init__(self, itmuni=4, start_datetime=None):
        self.itmuni = itmuni
        self.start_datetime = start_datetime

    @property
    def model_time_units(self):
        return self.itmuni_text[self.itmuni]


class epsgRef:
    """
    Sets up a local database of text representations of coordinate reference
    systems, keyed by EPSG code.

    The database is epsgref.json, located in the user's data directory. If
    optional 'appdirs' package is available, this is in the platform-dependent
    user directory, otherwise in the user's 'HOME/.flopy' directory.

    .. deprecated:: 3.2.11
        This class will be removed in version 3.3.5.
    """

    def __init__(self):
        warnings.warn(
            "epsgRef has been deprecated and will be removed in version "
            "3.3.5.",
            category=DeprecationWarning,
        )
        try:
            from appdirs import user_data_dir
        except ImportError:
            user_data_dir = None
        if user_data_dir:
            datadir = user_data_dir("flopy")
        else:
            # if appdirs is not installed, use user's home directory
            datadir = os.path.join(os.path.expanduser("~"), ".flopy")
        if not os.path.isdir(datadir):
            os.makedirs(datadir)
        dbname = "epsgref.json"
        self.location = os.path.join(datadir, dbname)

    def to_dict(self):
        """
        Returns dict with EPSG code integer key, and WKT CRS text
        """
        data = OrderedDict()
        if os.path.exists(self.location):
            with open(self.location, "r") as f:
                loaded_data = json.load(f, object_pairs_hook=OrderedDict)
            # convert JSON key from str to EPSG integer
            for key, value in loaded_data.items():
                try:
                    data[int(key)] = value
                except ValueError:
                    data[key] = value
        return data

    def _write(self, data):
        with open(self.location, "w") as f:
            json.dump(data, f, indent=0)
            f.write("\n")

    def reset(self, verbose=True):
        if os.path.exists(self.location):
            os.remove(self.location)
        if verbose:
            print("Resetting {}".format(self.location))

    def add(self, epsg, prj):
        """
        add an epsg code to epsgref.json
        """
        data = self.to_dict()
        data[epsg] = prj
        self._write(data)

    def get(self, epsg):
        """
        returns prj from a epsg code, otherwise None if not found
        """
        data = self.to_dict()
        return data.get(epsg)

    def remove(self, epsg):
        """
        removes an epsg entry from epsgref.json
        """
        data = self.to_dict()
        if epsg in data:
            del data[epsg]
            self._write(data)

    @staticmethod
    def show():
        ep = epsgRef()
        prj = ep.to_dict()
        for k, v in prj.items():
            print("{}:\n{}\n".format(k, v))


class crs:
    """
    Container to parse and store coordinate reference system parameters,
    and translate between different formats.

    .. deprecated:: 3.2.11
        This class will be removed in version 3.3.5. Use
        :py:class:`flopy.export.shapefile_utils.CRS` instead.
    """

    def __init__(self, prj=None, esri_wkt=None, epsg=None):
        warnings.warn(
            "crs has been deprecated and will be removed in version 3.3.5. "
            "Use CRS in shapefile_utils instead.",
            category=DeprecationWarning,
        )
        self.wktstr = None
        if prj is not None:
            with open(prj) as fprj:
                self.wktstr = fprj.read()
        elif esri_wkt is not None:
            self.wktstr = esri_wkt
        elif epsg is not None:
            wktstr = getprj(epsg)
            if wktstr is not None:
                self.wktstr = wktstr
        if self.wktstr is not None:
            self.parse_wkt()

    @property
    def crs(self):
        """
        Dict mapping crs attributes to proj4 parameters
        """
        proj = None
        if self.projcs is not None:
            # projection
            if "mercator" in self.projcs.lower():
                if (
                    "transverse" in self.projcs.lower()
                    or "tm" in self.projcs.lower()
                ):
                    proj = "tmerc"
                else:
                    proj = "merc"
            elif (
                "utm" in self.projcs.lower() and "zone" in self.projcs.lower()
            ):
                proj = "utm"
            elif "stateplane" in self.projcs.lower():
                proj = "lcc"
            elif "lambert" and "conformal" and "conic" in self.projcs.lower():
                proj = "lcc"
            elif "albers" in self.projcs.lower():
                proj = "aea"
        elif self.projcs is None and self.geogcs is not None:
            proj = "longlat"

        # datum
        datum = None
        if (
            "NAD" in self.datum.lower()
            or "north" in self.datum.lower()
            and "america" in self.datum.lower()
        ):
            datum = "nad"
            if "83" in self.datum.lower():
                datum += "83"
            elif "27" in self.datum.lower():
                datum += "27"
        elif "84" in self.datum.lower():
            datum = "wgs84"

        # ellipse
        ellps = None
        if "1866" in self.spheroid_name:
            ellps = "clrk66"
        elif "grs" in self.spheroid_name.lower():
            ellps = "grs80"
        elif "wgs" in self.spheroid_name.lower():
            ellps = "wgs84"

        # prime meridian
        pm = self.primem[0].lower()

        return {
            "proj": proj,
            "datum": datum,
            "ellps": ellps,
            "a": self.semi_major_axis,
            "rf": self.inverse_flattening,
            "lat_0": self.latitude_of_origin,
            "lat_1": self.standard_parallel_1,
            "lat_2": self.standard_parallel_2,
            "lon_0": self.central_meridian,
            "k_0": self.scale_factor,
            "x_0": self.false_easting,
            "y_0": self.false_northing,
            "units": self.projcs_unit,
            "zone": self.utm_zone,
        }

    @property
    def grid_mapping_attribs(self):
        """
        Map parameters for CF Grid Mappings
        http://http://cfconventions.org/cf-conventions/cf-conventions.html,
        Appendix F: Grid Mappings
        """
        if self.wktstr is not None:
            sp = [
                p
                for p in [self.standard_parallel_1, self.standard_parallel_2]
                if p is not None
            ]
            sp = sp if len(sp) > 0 else None
            proj = self.crs["proj"]
            names = {
                "aea": "albers_conical_equal_area",
                "aeqd": "azimuthal_equidistant",
                "laea": "lambert_azimuthal_equal_area",
                "longlat": "latitude_longitude",
                "lcc": "lambert_conformal_conic",
                "merc": "mercator",
                "tmerc": "transverse_mercator",
                "utm": "transverse_mercator",
            }
            attribs = {
                "grid_mapping_name": names[proj],
                "semi_major_axis": self.crs["a"],
                "inverse_flattening": self.crs["rf"],
                "standard_parallel": sp,
                "longitude_of_central_meridian": self.crs["lon_0"],
                "latitude_of_projection_origin": self.crs["lat_0"],
                "scale_factor_at_projection_origin": self.crs["k_0"],
                "false_easting": self.crs["x_0"],
                "false_northing": self.crs["y_0"],
            }
            return {k: v for k, v in attribs.items() if v is not None}

    @property
    def proj4(self):
        """
        Not implemented yet
        """
        return None

    def parse_wkt(self):

        self.projcs = self._gettxt('PROJCS["', '"')
        self.utm_zone = None
        if self.projcs is not None and "utm" in self.projcs.lower():
            self.utm_zone = self.projcs[-3:].lower().strip("n").strip("s")
        self.geogcs = self._gettxt('GEOGCS["', '"')
        self.datum = self._gettxt('DATUM["', '"')
        tmp = self._getgcsparam("SPHEROID")
        self.spheroid_name = tmp.pop(0)
        self.semi_major_axis = tmp.pop(0)
        self.inverse_flattening = tmp.pop(0)
        self.primem = self._getgcsparam("PRIMEM")
        self.gcs_unit = self._getgcsparam("UNIT")
        self.projection = self._gettxt('PROJECTION["', '"')
        self.latitude_of_origin = self._getvalue("latitude_of_origin")
        self.central_meridian = self._getvalue("central_meridian")
        self.standard_parallel_1 = self._getvalue("standard_parallel_1")
        self.standard_parallel_2 = self._getvalue("standard_parallel_2")
        self.scale_factor = self._getvalue("scale_factor")
        self.false_easting = self._getvalue("false_easting")
        self.false_northing = self._getvalue("false_northing")
        self.projcs_unit = self._getprojcs_unit()

    def _gettxt(self, s1, s2):
        s = self.wktstr.lower()
        strt = s.find(s1.lower())
        if strt >= 0:  # -1 indicates not found
            strt += len(s1)
            end = s[strt:].find(s2.lower()) + strt
            return self.wktstr[strt:end]

    def _getvalue(self, k):
        s = self.wktstr.lower()
        strt = s.find(k.lower())
        if strt >= 0:
            strt += len(k)
            end = s[strt:].find("]") + strt
            try:
                return float(self.wktstr[strt:end].split(",")[1])
            except:
                print("   could not typecast wktstr to a float")

    def _getgcsparam(self, txt):
        nvalues = 3 if txt.lower() == "spheroid" else 2
        tmp = self._gettxt('{}["'.format(txt), "]")
        if tmp is not None:
            tmp = tmp.replace('"', "").split(",")
            name = tmp[0:1]
            values = list(map(float, tmp[1:nvalues]))
            return name + values
        else:
            return [None] * nvalues

    def _getprojcs_unit(self):
        if self.projcs is not None:
            tmp = self.wktstr.lower().split('unit["')[-1]
            uname, ufactor = tmp.strip().strip("]").split('",')[0:2]
            ufactor = float(ufactor.split("]")[0].split()[0].split(",")[0])
            return uname, ufactor
        return None, None


def getprj(epsg, addlocalreference=True, text="esriwkt"):
    """
    Gets projection file (.prj) text for given epsg code from
    spatialreference.org

    .. deprecated:: 3.2.11
        This function will be removed in version 3.3.5. Use
        :py:class:`flopy.discretization.structuredgrid.StructuredGrid` instead.

    Parameters
    ----------
    epsg : int
        epsg code for coordinate system
    addlocalreference : boolean
        adds the projection file text associated with epsg to a local
        database, epsgref.json, located in the user's data directory.

    References
    ----------
    https://www.epsg-registry.org/

    Returns
    -------
    prj : str
        text for a projection (*.prj) file.

    """
    warnings.warn(
        "SpatialReference has been deprecated and will be removed in version "
        "3.3.5. Use StructuredGrid instead.",
        category=DeprecationWarning,
    )
    epsgfile = epsgRef()
    wktstr = epsgfile.get(epsg)
    if wktstr is None:
        wktstr = get_spatialreference(epsg, text=text)
    if addlocalreference and wktstr is not None:
        epsgfile.add(epsg, wktstr)
    return wktstr


def get_spatialreference(epsg, text="esriwkt"):
    """
    Gets text for given epsg code and text format from spatialreference.org

    Fetches the reference text using the url:
        https://spatialreference.org/ref/epsg/<epsg code>/<text>/

    See: https://www.epsg-registry.org/

    .. deprecated:: 3.2.11
        This function will be removed in version 3.3.5. Use
        :py:class:`flopy.discretization.structuredgrid.StructuredGrid` instead.

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
    from flopy.utils.flopy_io import get_url_text

    warnings.warn(
        "SpatialReference has been deprecated and will be removed in version "
        "3.3.5. Use StructuredGrid instead.",
        category=DeprecationWarning,
    )

    epsg_categories = ["epsg", "esri"]
    for cat in epsg_categories:
        url = "{}/ref/{}/{}/{}/".format(srefhttp, cat, epsg, text)
        result = get_url_text(url)
        if result is not None:
            break
    if result is not None:
        return result.replace("\n", "")
    elif result is None and text != "epsg":
        for cat in epsg_categories:
            error_msg = (
                "No internet connection or epsg code {} ".format(epsg)
                + "not found at {}/ref/".format(srefhttp)
                + "{}/{}/{}".format(cat, cat, epsg)
            )
            print(error_msg)
    # epsg code not listed on spatialreference.org
    # may still work with pyproj
    elif text == "epsg":
        return "epsg:{}".format(epsg)


def getproj4(epsg):
    """
    Get projection file (.prj) text for given epsg code from
    spatialreference.org. See: https://www.epsg-registry.org/

    .. deprecated:: 3.2.11
        This function will be removed in version 3.3.5. Use
        :py:class:`flopy.discretization.structuredgrid.StructuredGrid` instead.

    Parameters
    ----------
    epsg : int
        epsg code for coordinate system

    Returns
    -------
    prj : str
        text for a projection (*.prj) file.

    """
    warnings.warn(
        "SpatialReference has been deprecated and will be removed in version "
        "3.3.5. Use StructuredGrid instead.",
        category=DeprecationWarning,
    )

    return get_spatialreference(epsg, text="proj4")
