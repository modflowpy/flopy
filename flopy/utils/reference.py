"""
Module spatial referencing for flopy model objects

"""
import json
import os
import warnings

from collections import OrderedDict

__all__ = ["TemporalReference"]
# all other classes and methods in this module are deprecated

# web address of spatial reference dot org
srefhttp = "https://spatialreference.org"


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
