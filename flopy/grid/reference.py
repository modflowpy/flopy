"""
Module spatial referencing for flopy model objects

"""
import sys
import os
import numpy as np


class SpatialReference(object):
    """
    a class to locate a structured model grid in x-y space

    Parameters
    ----------

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
    rotation = 0.
    length_multiplier = 1.
    origin_loc = 'ul'  # or ll

    defaults = {"xul": None, "yul": None, "rotation": 0.,
                "proj4_str": None,
                "units": None, "lenuni": 2,
                "length_multiplier": None,
                "source": 'defaults'}

    lenuni_values = {'undefined': 0,
                     'feet': 1,
                     'meters': 2,
                     'centimeters': 3}
    lenuni_text = {v: k for k, v in lenuni_values.items()}

    def __init__(self, lenuni=2, xul=None, yul=None, xll=None, yll=None,
                 rotation=0.0, proj4_str=None, epsg=None, prj=None, units=None,
                 length_multiplier=None):
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
        if self.origin_loc == 'll':
            xll = self._xll if self._xll is not None else 0.
        elif self.origin_loc == 'ul':
            # calculate coords for lower left corner
            xll = self._xul - (np.sin(self.theta) * self.yedge[0] *
                               self.length_multiplier)
        return xll

    @property
    def yll(self):
        if self.origin_loc == 'll':
            yll = self._yll if self._yll is not None else 0.
        elif self.origin_loc == 'ul':
            # calculate coords for lower left corner
            yll = self._yul - (np.cos(self.theta) * self.yedge[0] *
                               self.length_multiplier)
        return yll

    @property
    def xul(self):
        if self.origin_loc == 'll':
            # calculate coords for upper left corner
            xul = self._xll + (np.sin(self.theta) * self.yedge[0] *
                               self.length_multiplier)
        if self.origin_loc == 'ul':
            # calculate coords for lower left corner
            xul = self._xul if self._xul is not None else 0.
        return xul

    @property
    def yul(self):
        if self.origin_loc == 'll':
            # calculate coords for upper left corner
            yul = self._yll + (np.cos(self.theta) * self.yedge[0] *
                               self.length_multiplier)

        if self.origin_loc == 'ul':
            # calculate coords for lower left corner
            yul = self._yul if self._yul is not None else 0.
        return yul

    @property
    def proj4_str(self):
        proj4_str = None
        if self._proj4_str is not None:
            if "epsg" in self._proj4_str.lower():
                if "init" not in self._proj4_str.lower():
                    proj4_str = "+init=" + self._proj4_str
                else:
                    proj4_str = self._proj4_str
                # set the epsg if proj4 specifies it
                tmp = [i for i in self._proj4_str.split() if
                       'epsg' in i.lower()]
                self._epsg = int(tmp[0].split(':')[1])
            else:
                proj4_str = self._proj4_str
        elif self.epsg is not None:
            proj4_str = '+init=epsg:{}'.format(self.epsg)
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

                crs = pyproj.Proj(self.proj4_str,
                                  preseve_units=True,
                                  errcheck=True)
                proj_str = crs.srs
            else:
                proj_str = self.proj4_str
            # http://proj4.org/parameters.html#units
            # from proj4 source code
            # "us-ft", "0.304800609601219", "U.S. Surveyor's Foot",
            # "ft", "0.3048", "International Foot",
            if "units=m" in proj_str:
                units = "meters"
            elif "units=ft" in proj_str or \
                            "units=us-ft" in proj_str or \
                            "to_meters:0.3048" in proj_str:
                units = "feet"
            return units
        except:
            pass

    @property
    def units(self):
        if self._units is not None:
            units = self._units.lower()
        else:
            units = self._parse_units_from_proj4()
        if units is None:
            # print("warning: assuming SpatialReference units are meters")
            units = 'meters'
        assert units in self.supported_units
        return units

    @property
    def length_multiplier(self):
        """Attempt to identify multiplier for converting from
        model units to sr units, defaulting to 1."""
        lm = None
        if self._length_multiplier is not None:
            lm = self._length_multiplier
        else:
            if self.model_length_units == 'feet':
                if self.units == 'meters':
                    lm = 0.3048
                elif self.units == 'feet':
                    lm = 1.
            elif self.model_length_units == 'meters':
                if self.units == 'feet':
                    lm = 1 / .3048
                elif self.units == 'meters':
                    lm = 1.
            elif self.model_length_units == 'centimeters':
                if self.units == 'meters':
                    lm = 1 / 100.
                elif self.units == 'feet':
                    lm = 1 / 30.48
            else:  # model units unspecified; default to 1
                lm = 1.
        return lm

    @property
    def model_length_units(self):
        return self.lenuni_text[self.lenuni]

    @property
    def bounds(self):
        """Return bounding box in shapely order."""
        xmin, xmax, ymin, ymax = self.get_extent()
        return xmin, ymin, xmax, ymax

    @staticmethod
    def load(namefile=None, reffile='usgs.model.reference'):
        """Attempts to load spatial reference information from
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
        d['source'] = 'namfile'
        if namefile is None:
            return None
        header = []
        with open(namefile, 'r') as f:
            for line in f:
                if not line.startswith('#'):
                    break
                header.extend(line.strip().replace('#', '').split(';'))

        for item in header:
            if "xul" in item.lower():
                try:
                    d['xul'] = float(item.split(':')[1])
                except:
                    pass
            elif "yul" in item.lower():
                try:
                    d['yul'] = float(item.split(':')[1])
                except:
                    pass
            elif "rotation" in item.lower():
                try:
                    d['rotation'] = float(item.split(':')[1])
                except:
                    pass
            elif "proj4_str" in item.lower():
                try:
                    proj4_str = ':'.join(item.split(':')[1:]).strip()
                    if proj4_str.lower() == 'none':
                        proj4_str = None
                    d['proj4_str'] = proj4_str

                except:
                    pass
            elif "start" in item.lower():
                try:
                    d['start_datetime'] = item.split(':')[1].strip()
                except:
                    pass
            # spatial reference length units
            elif "units" in item.lower():
                d['units'] = item.split(':')[1].strip()
            # model length units
            elif "lenuni" in item.lower():
                d['lenuni'] = int(item.split(':')[1].strip())
            # multiplier for converting from model length units to sr length units
            elif "length_multiplier" in item.lower():
                d['length_multiplier'] = float(item.split(':')[1].strip())
        return d

    @staticmethod
    def read_usgs_model_reference_file(reffile='usgs.model.reference'):
        """read spatial reference info from the usgs.model.reference file
        https://water.usgs.gov/ogw/policy/gw-model/modelers-setup.html"""

        ITMUNI = {0: "undefined", 1: "seconds", 2: "minutes", 3: "hours",
                  4: "days",
                  5: "years"}
        itmuni_values = {v: k for k, v in ITMUNI.items()}

        d = SpatialReference.defaults.copy()
        d['source'] = 'usgs.model.reference'
        d.pop(
            'proj4_str')  # discard default to avoid confusion with epsg code if entered
        if os.path.exists(reffile):
            with open(reffile) as input:
                for line in input:
                    if len(line) > 1:
                        if line.strip()[0] != '#':
                            info = line.strip().split('#')[0].split()
                            if len(info) > 1:
                                d[info[0].lower()] = ' '.join(info[1:])
            d['xul'] = float(d['xul'])
            d['yul'] = float(d['yul'])
            d['rotation'] = float(d['rotation'])

            # convert the model.reference text to a lenuni value
            # (these are the model length units)
            if 'length_units' in d.keys():
                d['lenuni'] = SpatialReference.lenuni_values[d['length_units']]
            if 'time_units' in d.keys():
                d['itmuni'] = itmuni_values[d['time_units']]
            if 'start_date' in d.keys():
                start_datetime = d.pop('start_date')
                if 'start_time' in d.keys():
                    start_datetime += ' {}'.format(d.pop('start_time'))
                d['start_datetime'] = start_datetime
            if 'epsg' in d.keys():
                try:
                    d['epsg'] = int(d['epsg'])
                except Exception as e:
                    raise Exception(
                        "error reading epsg code from file:\n" + str(e))
            # this prioritizes epsg over proj4 if both are given
            # (otherwise 'proj4' entry will be dropped below)
            elif 'proj4' in d.keys():
                d['proj4_str'] = d['proj4']

            # drop any other items that aren't used in sr class
            d = {k: v for k, v in d.items() if
                 k.lower() in SpatialReference.defaults.keys()
                 or k.lower() in {'epsg', 'start_datetime', 'itmuni',
                                  'source'}}
            return d
        else:
            return None

    def __setattr__(self, key, value):
        reset = True
        if key == "xul":
            super(SpatialReference, self). \
                __setattr__("_xul", float(value))
            self.origin_loc = 'ul'
        elif key == "yul":
            super(SpatialReference, self). \
                __setattr__("_yul", float(value))
            self.origin_loc = 'ul'
        elif key == "xll":
            super(SpatialReference, self). \
                __setattr__("_xll", float(value))
            self.origin_loc = 'll'
        elif key == "yll":
            super(SpatialReference, self). \
                __setattr__("_yll", float(value))
            self.origin_loc = 'll'
        elif key == "length_multiplier":
            super(SpatialReference, self). \
                __setattr__("_length_multiplier", float(value))
            # self.set_origin(xul=self.xul, yul=self.yul, xll=self.xll,
            #                yll=self.yll)
        elif key == "rotation":
            super(SpatialReference, self). \
                __setattr__("rotation", float(value))
            # self.set_origin(xul=self.xul, yul=self.yul, xll=self.xll,
            #                yll=self.yll)
        elif key == "lenuni":
            super(SpatialReference, self). \
                __setattr__("_lenuni", int(value))
            # self.set_origin(xul=self.xul, yul=self.yul, xll=self.xll,
            #                yll=self.yll)
        elif key == "units":
            value = value.lower()
            assert value in self.supported_units
            super(SpatialReference, self). \
                __setattr__("_units", value)
        elif key == "proj4_str":
            super(SpatialReference, self). \
                __setattr__("_proj4_str", value)
            # reset the units and epsg
            units = self._parse_units_from_proj4()
            if units is not None:
                self._units = units
            self._epsg = None
        elif key == "epsg":
            super(SpatialReference, self). \
                __setattr__("_epsg", value)
            # reset the units and proj4
            self._units = None
            self._proj4_str = getproj4(self._epsg)
            self.crs = crs(epsg=value)
        elif key == "prj":
            super(SpatialReference, self). \
                __setattr__("prj", value)
            # translation to proj4 strings in crs class not robust yet
            # leave units and proj4 alone for now.
            self.crs = crs(prj=value, epsg=self.epsg)
        else:
            super(SpatialReference, self).__setattr__(key, value)
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
            pass
        return SpatialReference(**attribs)

    @classmethod
    def from_gridspec(cls, gridspec_file, lenuni=0):
        f = open(gridspec_file, 'r')
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
        return {"xul": self.xul, "yul": self.yul, "rotation": self.rotation,
                "proj4_str": self.proj4_str}

    def set_spatialreference(self, xul=None, yul=None, xll=None, yll=None,
                             rotation=0.0):
        """
            set spatial reference - can be called from model instance

        """
        if xul is not None and xll is not None:
            msg = ('Both xul and xll entered. Please enter either xul, yul or '
                   'xll, yll.')
            raise ValueError(msg)
        if yul is not None and yll is not None:
            msg = ('Both yul and yll entered. Please enter either xul, yul or '
                   'xll, yll.')
            raise ValueError(msg)
        # set the origin priority based on the left corner specified
        # (the other left corner will be calculated).  If none are specified
        # then default to upper left
        if xul is None and yul is None and xll is None and yll is None:
            self.origin_loc = 'ul'
            xul = 0.
            yul = self.delc.sum()
        elif xll is not None:
            self.origin_loc = 'll'
        else:
            self.origin_loc = 'ul'

        self.rotation = rotation
        self._xll = xll if xll is not None else 0.
        self._yll = yll if yll is not None else 0.
        self._xul = xul if xul is not None else 0.
        self._yul = yul if yul is not None else 0.
        # self.set_origin(xul, yul, xll, yll)
        return

    def __repr__(self):
        s = "xul:{0:<.10G}; yul:{1:<.10G}; rotation:{2:<G}; ". \
            format(self.xul, self.yul, self.rotation)
        s += "proj4_str:{0}; ".format(self.proj4_str)
        s += "units:{0}; ".format(self.units)
        s += "lenuni:{0}; ".format(self.lenuni)
        s += "length_multiplier:{}".format(self.length_multiplier)
        return s

    def set_origin(self, xul=None, yul=None, xll=None, yll=None):
        if self.origin_loc == 'll':
            # calculate coords for upper left corner
            self._xll = xll if xll is not None else 0.
            self.yll = yll if yll is not None else 0.
            self.xul = self._xll + (np.sin(self.theta) * self.yedge[0] *
                                    self.length_multiplier)
            self.yul = self.yll + (np.cos(self.theta) * self.yedge[0] *
                                   self.length_multiplier)

        if self.origin_loc == 'ul':
            # calculate coords for lower left corner
            self.xul = xul if xul is not None else 0.
            self.yul = yul if yul is not None else 0.
            self._xll = self.xul - (np.sin(self.theta) * self.yedge[0] *
                                    self.length_multiplier)
            self.yll = self.yul - (np.cos(self.theta) * self.yedge[0] *
                                   self.length_multiplier)
        self._reset()
        return

    @property
    def theta(self):
        return -self.rotation * np.pi / 180.

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
        self._xcentergrid, self._ycentergrid = self.transform(
            self._xcentergrid,
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
        # jwhite changed on Oct 11 2016 - rotation is now positive CCW
        # theta = -theta * np.pi / 180.
        theta = theta * np.pi / 180.

        xrot = xorigin + np.cos(theta) * (x - xorigin) - np.sin(theta) * \
                                                         (y - yorigin)
        yrot = yorigin + np.sin(theta) * (x - xorigin) + np.cos(theta) * \
                                                         (y - yorigin)
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
            x, y = SpatialReference.rotate(x, y, theta=self.rotation,
                                           xorigin=self.xll, yorigin=self.yll)
        else:
            x, y = SpatialReference.rotate(x, y, -self.rotation,
                                           self.xll, self.yll)
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
        from matplotlib.collections import LineCollection

        lc = LineCollection(self.get_grid_lines(), **kwargs)
        return lc

    def write_shapefile(self, filename='grid.shp', epsg=None, prj=None):
        """Write a shapefile of the grid with just the row and column attributes"""
        from ..export.shapefile_utils import write_grid_shapefile2
        if epsg is None and prj is None:
            epsg = self.epsg
        write_grid_shapefile2(filename, self, array_dict={}, nan_val=-1.0e9,
                              epsg=epsg, prj=prj)

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
        """Return the row and column of a point or sequence of points
        in real-world coordinates.

        Parameters
        ----------
        x : scalar or sequence of x coordinates
        y : scalar or sequence of y coordinates

        Returns
        -------
        r : row or sequence of rows (zero-based)
        c : column or sequence of columns (zero-based)
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


class TemporalReference(object):
    """For now, just a container to hold start time and time units files
    outside of DIS package."""

    defaults = {'itmuni': 4,
                'start_datetime': '01-01-1970'}

    itmuni_values = {'undefined': 0,
                     'seconds': 1,
                     'minutes': 2,
                     'hours': 3,
                     'days': 4,
                     'years': 5}

    itmuni_text = {v: k for k, v in itmuni_values.items()}

    def __init__(self, itmuni=4, start_datetime=None):
        self.itmuni = itmuni
        self.start_datetime = start_datetime

    @property
    def model_time_units(self):
        return self.itmuni_text[self.itmuni]


class epsgRef:
    """Sets up a local database of projection file text referenced by epsg code.
    The database is located in the site packages folder in epsgref.py, which
    contains a dictionary, prj, of projection file text keyed by epsg value.
    """

    def __init__(self):
        sp = [f for f in sys.path if f.endswith('site-packages')][0]
        self.location = os.path.join(sp, 'epsgref.py')

    def _remove_pyc(self):
        try:  # get rid of pyc file
            os.remove(self.location + 'c')
        except:
            pass

    def make(self):
        if not os.path.exists(self.location):
            newfile = open(self.location, 'w')
            newfile.write('prj = {}\n')
            newfile.close()

    def reset(self, verbose=True):
        if os.path.exists(self.location):
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
        try:
            from importlib import reload
        except:
            from imp import reload
        import epsgref
        from epsgref import prj
        reload(epsgref)
        for k, v in prj.items():
            print('{}:\n{}\n'.format(k, v))


class crs(object):
    """Container to parse and store coordinate reference system parameters,
    and translate between different formats."""

    def __init__(self, prj=None, esri_wkt=None, epsg=None):

        self.wktstr = None
        if prj is not None:
            with open(prj) as input:
                self.wktstr = input.read()
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
        """Dict mapping crs attibutes to proj4 parameters"""
        proj = None
        if self.projcs is not None:
            # projection
            if 'mercator' in self.projcs.lower():
                if 'transvers' in self.projcs.lower() or \
                                'tm' in self.projcs.lower():
                    proj = 'tmerc'
                else:
                    proj = 'merc'
            elif 'utm' in self.projcs.lower() and \
                            'zone' in self.projcs.lower():
                proj = 'utm'
            elif 'stateplane' in self.projcs.lower():
                proj = 'lcc'
            elif 'lambert' and 'conformal' and 'conic' in self.projcs.lower():
                proj = 'lcc'
            elif 'albers' in self.projcs.lower():
                proj = 'aea'
        elif self.projcs is None and self.geogcs is not None:
            proj = 'longlat'

        # datum
        if 'NAD' in self.datum.lower() or \
                                'north' in self.datum.lower() and \
                                'america' in self.datum.lower():
            datum = 'nad'
            if '83' in self.datum.lower():
                datum += '83'
            elif '27' in self.datum.lower():
                datum += '27'
        elif '84' in self.datum.lower():
            datum = 'wgs84'

        # ellipse
        if '1866' in self.spheriod_name:
            ellps = 'clrk66'
        elif 'grs' in self.spheriod_name.lower():
            ellps = 'grs80'
        elif 'wgs' in self.spheriod_name.lower():
            ellps = 'wgs84'

        # prime meridian
        pm = self.primem[0].lower()

        return {'proj': proj,
                'datum': datum,
                'ellps': ellps,
                'a': self.semi_major_axis,
                'rf': self.inverse_flattening,
                'lat_0': self.latitude_of_origin,
                'lat_1': self.standard_parallel_1,
                'lat_2': self.standard_parallel_2,
                'lon_0': self.central_meridian,
                'k_0': self.scale_factor,
                'x_0': self.false_easting,
                'y_0': self.false_northing,
                'units': self.projcs_unit,
                'zone': self.utm_zone}

    @property
    def grid_mapping_attribs(self):
        """Map parameters for CF Grid Mappings
        http://http://cfconventions.org/cf-conventions/cf-conventions.html,
        Appendix F: Grid Mappings
        """
        if self.wktstr is not None:
            sp = [p for p in [self.standard_parallel_1,
                              self.standard_parallel_2]
                  if p is not None]
            sp = sp if len(sp) > 0 else None
            proj = self.crs['proj']
            names = {'aea': 'albers_conical_equal_area',
                     'aeqd': 'azimuthal_equidistant',
                     'laea': 'lambert_azimuthal_equal_area',
                     'longlat': 'latitude_longitude',
                     'lcc': 'lambert_conformal_conic',
                     'merc': 'mercator',
                     'tmerc': 'transverse_mercator',
                     'utm': 'transverse_mercator'}
            attribs = {'grid_mapping_name': names[proj],
                       'semi_major_axis': self.crs['a'],
                       'inverse_flattening': self.crs['rf'],
                       'standard_parallel': sp,
                       'longitude_of_central_meridian': self.crs['lon_0'],
                       'latitude_of_projection_origin': self.crs['lat_0'],
                       'scale_factor_at_projection_origin': self.crs['k_0'],
                       'false_easting': self.crs['x_0'],
                       'false_northing': self.crs['y_0']}
            return {k: v for k, v in attribs.items() if v is not None}

    @property
    def proj4(self):
        """Not implemented yet"""
        return None

    def parse_wkt(self):

        self.projcs = self._gettxt('PROJCS["', '"')
        self.utm_zone = None
        if self.projcs is not None and 'utm' in self.projcs.lower():
            self.utm_zone = self.projcs[-3:].lower().strip('n').strip('s')
        self.geogcs = self._gettxt('GEOGCS["', '"')
        self.datum = self._gettxt('DATUM["', '"')
        tmp = self._getgcsparam('SPHEROID')
        self.spheriod_name = tmp.pop(0)
        self.semi_major_axis = tmp.pop(0)
        self.inverse_flattening = tmp.pop(0)
        self.primem = self._getgcsparam('PRIMEM')
        self.gcs_unit = self._getgcsparam('UNIT')
        self.projection = self._gettxt('PROJECTION["', '"')
        self.latitude_of_origin = self._getvalue('latitude_of_origin')
        self.central_meridian = self._getvalue('central_meridian')
        self.standard_parallel_1 = self._getvalue('standard_parallel_1')
        self.standard_parallel_2 = self._getvalue('standard_parallel_2')
        self.scale_factor = self._getvalue('scale_factor')
        self.false_easting = self._getvalue('false_easting')
        self.false_northing = self._getvalue('false_northing')
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
            end = s[strt:].find(']') + strt
            try:
                return float(self.wktstr[strt:end].split(',')[1])
            except:
                pass

    def _getgcsparam(self, txt):
        nvalues = 3 if txt.lower() == 'spheroid' else 2
        tmp = self._gettxt('{}["'.format(txt), ']')
        if tmp is not None:
            tmp = tmp.replace('"', '').split(',')
            name = tmp[0:1]
            values = list(map(float, tmp[1:nvalues]))
            return name + values
        else:
            return [None] * nvalues

    def _getprojcs_unit(self):
        if self.projcs is not None:
            tmp = self.wktstr.lower().split('unit["')[-1]
            uname, ufactor = tmp.strip().strip(']').split('",')[0:2]
            ufactor = float(ufactor.split(']')[0].split()[0].split(',')[0])
            return uname, ufactor
        return None, None


def getprj(epsg, addlocalreference=True, text='esriwkt'):
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
    wktstr = None
    try:
        from epsgref import prj
        wktstr = prj.get(epsg)
    except:
        epsgfile.make()
    if wktstr is None:
        wktstr = get_spatialreference(epsg, text=text)
    if addlocalreference and wktstr is not None:
        epsgfile.add(epsg, wktstr)
    return wktstr


def get_spatialreference(epsg, text='esriwkt'):
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
    from flopy.utils.flopy_io import get_url_text

    epsg_categories = ['epsg', 'esri']
    for cat in epsg_categories:
        url = "http://spatialreference.org/ref/{2}/{0}/{1}/".format(epsg,
                                                                    text,
                                                                    cat)
        result = get_url_text(url)
        if result is not None:
            break
    if result is not None:
        return result.replace("\n", "")
    elif result is None and text != 'epsg':
        for cat in epsg_categories:
            error_msg = 'No internet connection or epsg code {0} ' \
                        'not found at http://spatialreference.org/ref/{2}/{0}/{1}'.format(
                epsg,
                text,
                cat)
            print(error_msg)
    elif text == 'epsg':  # epsg code not listed on spatialreference.org may still work with pyproj
        return '+init=epsg:{}'.format(epsg)


def getproj4(epsg):
    """Gets projection file (.prj) text for given epsg code from
    spatialreference.org. See: https://www.epsg-registry.org/

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
