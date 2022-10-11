"""
Module for exporting and importing flopy model attributes

"""
import copy
import json
import shutil
import sys
import warnings
from os.path import expandvars
from pathlib import Path

import numpy as np

from ..datbase import DataInterface, DataType
from ..utils import Util3d, import_optional_dependency

# web address of spatial reference dot org
srefhttp = "https://spatialreference.org"


def write_gridlines_shapefile(filename, mg):
    """
    Write a polyline shapefile of the grid lines - a lightweight alternative
    to polygons.

    Parameters
    ----------
    filename : Path or str
        name of the shapefile to write
    mg : model grid

    Returns
    -------
    None

    """
    shapefile = import_optional_dependency("shapefile")
    wr = shapefile.Writer(str(filename), shapeType=shapefile.POLYLINE)
    wr.field("number", "N", 18, 0)
    if mg.__class__.__name__ == "SpatialReference":
        grid_lines = mg.get_grid_lines()
        warnings.warn(
            "SpatialReference has been deprecated. Use StructuredGrid"
            " instead.",
            category=DeprecationWarning,
        )
    else:
        grid_lines = mg.grid_lines
    for i, line in enumerate(grid_lines):
        wr.line([line])
        wr.record(i)

    wr.close()
    write_prj(filename, mg=mg)
    return


def write_grid_shapefile(
    path,
    mg,
    array_dict,
    nan_val=np.nan,
    epsg=None,
    prj=None,  # -1.0e9,
):
    """
    Method to write a shapefile of gridded input data

    Parameters
    ----------
    path : Path or str
        shapefile file path
    mg : flopy.discretization.Grid object
        flopy model grid
    array_dict : dict
        dictionary of model input arrays
    nan_val : float
        value to fill nans
    epsg : str, int
        epsg code
    prj : Path or str
        projection file name path

    Returns
    -------
    None

    """
    shapefile = import_optional_dependency("shapefile")
    w = shapefile.Writer(str(path), shapeType=shapefile.POLYGON)
    w.autoBalance = 1

    if mg.__class__.__name__ == "SpatialReference":
        verts = copy.deepcopy(mg.vertices)
        warnings.warn(
            "SpatialReference has been deprecated. Use StructuredGrid"
            " instead.",
            category=DeprecationWarning,
        )
        mg.grid_type = "structured"
    elif mg.grid_type == "structured":
        verts = [
            mg.get_cell_vertices(i, j)
            for i in range(mg.nrow)
            for j in range(mg.ncol)
        ]
    elif mg.grid_type == "vertex":
        verts = [mg.get_cell_vertices(cellid) for cellid in range(mg.ncpl)]
    elif mg.grid_type == "unstructured":
        verts = [mg.get_cell_vertices(cellid) for cellid in range(mg.nnodes)]
    else:
        raise Exception(f"Grid type {mg.grid_type} not supported.")

    # set up the attribute fields and arrays of attributes
    if mg.grid_type == "structured":
        names = ["node", "row", "column"] + list(array_dict.keys())
        dtypes = [
            ("node", np.dtype("int")),
            ("row", np.dtype("int")),
            ("column", np.dtype("int")),
        ] + [
            (enforce_10ch_limit([name])[0], array_dict[name].dtype)
            for name in names[3:]
        ]
        node = list(range(1, mg.ncol * mg.nrow + 1))
        col = list(range(1, mg.ncol + 1)) * mg.nrow
        row = sorted(list(range(1, mg.nrow + 1)) * mg.ncol)
        at = np.vstack(
            [node, row, col] + [array_dict[name].ravel() for name in names[3:]]
        ).transpose()

        names = enforce_10ch_limit(names)

    elif mg.grid_type == "vertex":
        names = ["node"] + list(array_dict.keys())
        dtypes = [("node", np.dtype("int"))] + [
            (enforce_10ch_limit([name])[0], array_dict[name].dtype)
            for name in names[1:]
        ]
        node = list(range(1, mg.ncpl + 1))
        at = np.vstack(
            [node] + [array_dict[name].ravel() for name in names[1:]]
        ).transpose()

        names = enforce_10ch_limit(names)

    elif mg.grid_type == "unstructured":
        if mg.nlay is None:
            names = ["node"] + list(array_dict.keys())
            dtypes = [("node", np.dtype("int"))] + [
                (enforce_10ch_limit([name])[0], array_dict[name].dtype)
                for name in names[1:]
            ]
            node = list(range(1, mg.nnodes + 1))
            at = np.vstack(
                [node] + [array_dict[name].ravel() for name in names[1:]]
            ).transpose()
        else:
            names = ["node", "layer"] + list(array_dict.keys())
            dtypes = [
                ("node", np.dtype("int")),
                ("layer", np.dtype("int")),
            ] + [
                (enforce_10ch_limit([name])[0], array_dict[name].dtype)
                for name in names[2:]
            ]
            node = list(range(1, mg.nnodes + 1))
            layer = np.zeros(mg.nnodes)
            for ilay in range(mg.nlay):
                istart, istop = mg.get_layer_node_range(ilay)
                layer[istart:istop] = ilay + 1
            at = np.vstack(
                [node]
                + [layer]
                + [array_dict[name].ravel() for name in names[2:]]
            ).transpose()

        names = enforce_10ch_limit(names)

    # flag nan values and explicitly set the dtypes
    if at.dtype in [float, np.float32, np.float64]:
        at[np.isnan(at)] = nan_val
    at = np.array([tuple(i) for i in at], dtype=dtypes)

    # write field information
    fieldinfo = {
        name: get_pyshp_field_info(dtype.name) for name, dtype in dtypes
    }
    for n in names:
        w.field(n, *fieldinfo[n])

    for i, r in enumerate(at):
        # check if polygon is closed, if not close polygon for QGIS
        if verts[i][-1] != verts[i][0]:
            verts[i] = verts[i] + [verts[i][0]]
        w.poly([verts[i]])
        w.record(*r)

    # close
    w.close()
    print(f"wrote {path}")
    # write the projection file
    write_prj(path, mg, epsg, prj)
    return


def model_attributes_to_shapefile(
    path, ml, package_names=None, array_dict=None, **kwargs
):
    """
    Wrapper function for writing a shapefile of model data.  If package_names
    is not None, then search through the requested packages looking for arrays
    that can be added to the shapefile as attributes

    Parameters
    ----------
    path : Path or str
        path to write the shapefile to
    ml : flopy.mbase
        model instance
    package_names : list of package names (e.g. ["dis","lpf"])
        Packages to export data arrays to shapefile. (default is None)
    array_dict : dict of {name:2D array} pairs
       Additional 2D arrays to add as attributes to the shapefile.
       (default is None)

    **kwargs : keyword arguments
        modelgrid : fp.modflow.Grid object
            if modelgrid is supplied, user supplied modelgrid is used in lieu
            of the modelgrid attached to the modflow model object
        epsg : int
            epsg projection information
        prj : Path or str
            user supplied prj file

    Returns
    -------
    None

    Examples
    --------

    >>> import flopy
    >>> m = flopy.modflow.Modflow()
    >>> flopy.utils.model_attributes_to_shapefile('model.shp', m)

    """

    if array_dict is None:
        array_dict = {}

    if package_names is not None:
        if not isinstance(package_names, list):
            package_names = [package_names]
    else:
        package_names = [pak.name[0] for pak in ml.packagelist]

    if "modelgrid" in kwargs:
        grid = kwargs.pop("modelgrid")
    else:
        grid = ml.modelgrid

    horz_shape = grid.get_plottable_layer_shape()
    for pname in package_names:
        pak = ml.get_package(pname)
        attrs = dir(pak)
        if pak is not None:
            if "sr" in attrs:
                attrs.remove("sr")
            if "start_datetime" in attrs:
                attrs.remove("start_datetime")
            for attr in attrs:
                a = pak.__getattribute__(attr)
                if (
                    a is None
                    or not hasattr(a, "data_type")
                    or a.name == "thickness"
                ):
                    continue
                if (
                    a.data_type == DataType.array2d
                    and a.array.shape == horz_shape
                ):
                    name = shape_attr_name(a.name, keep_layer=True)
                    # name = a.name.lower()
                    array_dict[name] = a.array
                elif a.data_type == DataType.array3d:
                    # Not sure how best to check if an object has array data
                    try:
                        assert a.array is not None
                    except:
                        print(
                            "Failed to get data for "
                            f"{a.name} array, {pak.name[0]} package"
                        )
                        continue
                    if isinstance(a.name, list) and a.name[0] == "thickness":
                        continue

                    if a.array.shape == horz_shape:
                        if hasattr(a, "shape"):
                            if a.shape[1] is None:  # usg unstructured Util3d
                                # return a flattened array, with a.name[0] (a per-layer list)
                                array_dict[a.name[0]] = a.array
                            else:
                                array_dict[a.name] = a.array
                        else:
                            array_dict[a.name] = a.array
                    else:
                        # array is not the same shape as the layer shape
                        for ilay in range(a.array.shape[0]):
                            try:
                                arr = a.array[ilay]
                            except:
                                arr = a[ilay]

                            if isinstance(a, Util3d):
                                aname = shape_attr_name(a[ilay].name)
                            else:
                                aname = a.name

                            if arr.shape == (1,) + horz_shape:
                                # fix for mf6 case
                                arr = arr[0]
                            assert arr.shape == horz_shape
                            name = f"{aname}_{ilay + 1}"
                            array_dict[name] = arr
                elif (
                    a.data_type == DataType.transient2d
                ):  # elif isinstance(a, Transient2d):
                    # Not sure how best to check if an object has array data
                    try:
                        assert a.array is not None
                    except:
                        print(
                            "Failed to get data for "
                            f"{a.name} array, {pak.name[0]} package"
                        )
                        continue
                    for kper in range(a.array.shape[0]):
                        name = f"{shape_attr_name(a.name)}{kper + 1}"
                        arr = a.array[kper][0]
                        assert arr.shape == horz_shape
                        array_dict[name] = arr
                elif (
                    a.data_type == DataType.transientlist
                ):  # elif isinstance(a, MfList):
                    try:
                        list(a.masked_4D_arrays_itr())
                    except:
                        continue
                    for name, array in a.masked_4D_arrays_itr():
                        for kper in range(array.shape[0]):
                            for k in range(array.shape[1]):
                                n = shape_attr_name(name, length=4)
                                aname = f"{n}{k + 1}{kper + 1}"
                                arr = array[kper][k]
                                assert arr.shape == horz_shape
                                if np.all(np.isnan(arr)):
                                    continue
                                array_dict[aname] = arr
                elif isinstance(a, list):
                    for v in a:
                        if (
                            isinstance(a, DataInterface)
                            and v.data_type == DataType.array3d
                        ):
                            for ilay in range(a.model.modelgrid.nlay):
                                u2d = a[ilay]
                                name = (
                                    f"{shape_attr_name(u2d.name)}_{ilay + 1}"
                                )
                                arr = u2d.array
                                assert arr.shape == horz_shape
                                array_dict[name] = arr

    # write data arrays to a shapefile
    write_grid_shapefile(path, grid, array_dict)
    epsg = kwargs.get("epsg", None)
    prj = kwargs.get("prj", None)
    write_prj(path, grid, epsg, prj)


def shape_attr_name(name, length=6, keep_layer=False):
    """
    Function for to format an array name to a maximum of 10 characters to
    conform with ESRI shapefile maximum attribute name length

    Parameters
    ----------
    name : str
        data array name
    length : int
        maximum length of string to return. Value passed to function is
        overridden and set to 10 if keep_layer=True. (default is 6)
    keep_layer : bool
        Boolean that determines if layer number in name should be retained.
        (default is False)


    Returns
    -------
    str

    Examples
    --------

    >>> import flopy
    >>> name = flopy.utils.shape_attr_name('averylongstring')
    >>> name
    >>> 'averyl'

    """
    # kludges
    if name == "model_top":
        name = "top"
    # replace spaces with "_"
    n = name.lower().replace(" ", "_")
    # exclude "_layer_X" portion of string
    if keep_layer:
        length = 10
        n = n.replace("_layer", "_")
    else:
        try:
            idx = n.index("_layer")
            n = n[:idx]
        except:
            pass

    if len(n) > length:
        n = n[:length]
    return n


def enforce_10ch_limit(names):
    """Enforce 10 character limit for fieldnames.
    Add suffix for duplicate names starting at 0.

    Parameters
    ----------
    names : list of strings

    Returns
    -------
    list
        list of unique strings of len <= 10.
    """
    names = [n[:5] + n[-4:] + "_" if len(n) > 10 else n for n in names]
    dups = {x: names.count(x) for x in names}
    suffix = {n: list(range(cnt)) for n, cnt in dups.items() if cnt > 1}
    for i, n in enumerate(names):
        if dups[n] > 1:
            names[i] = n[:9] + str(suffix[n].pop(0))
    return names


def get_pyshp_field_info(dtypename):
    """Get pyshp dtype information for a given numpy dtype."""
    fields = {
        "int": ("N", 18, 0),
        "<i": ("N", 18, 0),
        "float": ("F", 20, 12),
        "<f": ("F", 20, 12),
        "bool": ("L", 1),
        "b1": ("L", 1),
        "str": ("C", 50),
        "object": ("C", 50),
    }
    k = [k for k in fields.keys() if k in dtypename.lower()]
    if len(k) == 1:
        return fields[k[0]]
    else:
        return fields["str"]


def get_pyshp_field_dtypes(code):
    """Returns a numpy dtype for a pyshp field type."""
    dtypes = {
        "N": int,
        "F": float,
        "L": bool,
        "C": object,
    }
    return dtypes.get(code, object)


def shp2recarray(shpname):
    """Read a shapefile into a numpy recarray.

    Parameters
    ----------
    shpname : Path or str
        ESRI Shapefile.

    Returns
    -------
    np.recarray

    """
    from ..utils.geospatial_utils import GeoSpatialCollection

    sf = import_optional_dependency("shapefile")

    sfobj = sf.Reader(str(shpname))
    dtype = [
        (str(f[0]), get_pyshp_field_dtypes(f[1])) for f in sfobj.fields[1:]
    ]

    geoms = GeoSpatialCollection(sfobj).flopy_geometry
    records = [
        tuple(r) + (geoms[i],) for i, r in enumerate(sfobj.iterRecords())
    ]
    dtype += [("geometry", object)]

    recarray = np.array(records, dtype=dtype).view(np.recarray)
    return recarray


def recarray2shp(
    recarray,
    geoms,
    shpname="recarray.shp",
    mg=None,
    epsg=None,
    prj=None,
    **kwargs,
):
    """
    Write a numpy record array to a shapefile, using a corresponding
    list of geometries. Method supports list of flopy geometry objects,
    flopy Collection object, shapely Collection object, and geojson
    Geometry Collection objects

    Parameters
    ----------
    recarray : np.recarray
        Numpy record array with attribute information that will go in the
        shapefile
    geoms : list of flopy.utils.geometry, shapely geometry collection,
            flopy geometry collection, shapefile.Shapes,
            list of shapefile.Shape objects, or geojson geometry collection
        The number of geometries in geoms must equal the number of records in
        recarray.
    shpname : Path or str, default "recarray.shp"
        Path for the output shapefile
    epsg : int
        EPSG code. See https://www.epsg-registry.org/ or spatialreference.org
    prj : Path or str
        Existing projection file to be used with new shapefile.

    Notes
    -----
    Uses pyshp.
    epsg code requires an internet connection the first time to get the
    projection file text from spatialreference.org, but then stashes the text
    in the file epsgref.json (located in the user's data directory) for
    subsequent use. See flopy.reference for more details.

    """
    from ..utils.geospatial_utils import GeoSpatialCollection

    if len(recarray) != len(geoms):
        raise IndexError(
            "Number of geometries must equal the number of records!"
        )

    if len(recarray) == 0:
        raise Exception("Recarray is empty")

    geomtype = None

    geoms = GeoSpatialCollection(geoms).flopy_geometry

    for g in geoms:
        try:
            geomtype = g.shapeType
        except AttributeError:
            continue

    # set up for pyshp 2
    shapefile = import_optional_dependency("shapefile")
    w = shapefile.Writer(str(shpname), shapeType=geomtype)
    w.autoBalance = 1

    # set up the attribute fields
    names = enforce_10ch_limit(recarray.dtype.names)
    for i, npdtype in enumerate(recarray.dtype.descr):
        key = names[i]
        if not isinstance(key, str):
            key = str(key)
        w.field(key, *get_pyshp_field_info(npdtype[1]))

    # write the geometry and attributes for each record
    ralist = recarray.tolist()
    if geomtype == shapefile.POLYGON:
        for i, r in enumerate(ralist):
            w.poly(geoms[i].pyshp_parts)
            w.record(*r)
    elif geomtype == shapefile.POLYLINE:
        for i, r in enumerate(ralist):
            w.line(geoms[i].pyshp_parts)
            w.record(*r)
    elif geomtype == shapefile.POINT:
        # pyshp version 2.x w.point() method can only take x and y
        # code will need to be refactored in order to write POINTZ
        # shapes with the z attribute.
        for i, r in enumerate(ralist):
            w.point(*geoms[i].pyshp_parts[:2])
            w.record(*r)

    w.close()
    write_prj(shpname, mg, epsg, prj)
    print(f"wrote {shpname}")
    return


def write_prj(shpname, mg=None, epsg=None, prj=None, wkt_string=None):
    # projection file name
    prjname = Path(shpname).with_suffix(".prj")

    # figure which CRS option to use
    # prioritize args over grid reference
    # no proj4 option because it is too difficult
    # to create prjfile from proj4 string without OGR
    prjtxt = wkt_string
    if epsg is not None:
        prjtxt = CRS.getprj(epsg)
    # copy a supplied prj file
    elif prj is not None:
        if prjname.exists():
            print(f".prj file {prjname} already exists")
        else:
            shutil.copy(str(prj), str(prjname))

    elif mg is not None:
        if mg.epsg is not None:
            prjtxt = CRS.getprj(mg.epsg)

    else:
        print(
            "No CRS information for writing a .prj file.\n"
            "Supply an epsg code or .prj file path to the "
            "model spatial reference or .export() method."
            "(writing .prj files from proj4 strings not supported)"
        )
    if prjtxt is not None:
        prjname.write_text(prjtxt)


class CRS:
    """
    Container to parse and store coordinate reference system parameters,
    and translate between different formats.
    """

    def __init__(self, prj=None, esri_wkt=None, epsg=None):

        self.wktstr = None
        if prj is not None:
            with open(prj) as prj_input:
                self.wktstr = prj_input.read()
        elif esri_wkt is not None:
            self.wktstr = esri_wkt
        elif epsg is not None:
            wktstr = CRS.getprj(epsg)
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
                    "transvers" in self.projcs.lower()
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
        https://cfconventions.org/cf-conventions/cf-conventions.html#appendix-grid-mappings,
        Appendix F: Grid Mappings

        """
        if self.wktstr is not None:
            sp = [
                p
                for p in [
                    self.standard_parallel_1,
                    self.standard_parallel_2,
                ]
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
            except (
                IndexError,
                TypeError,
                ValueError,
                AttributeError,
            ):
                pass

    def _getgcsparam(self, txt):
        nvalues = 3 if txt.lower() == "spheroid" else 2
        tmp = self._gettxt(f'{txt}["', "]")
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

    @staticmethod
    def getprj(epsg, addlocalreference=True, text="esriwkt"):
        """
        Gets projection file (.prj) text for given epsg code from
        spatialreference.org
        See: https://www.epsg-registry.org/

        Parameters
        ----------
        epsg : int
            epsg code for coordinate system
        addlocalreference : boolean
            adds the projection file text associated with epsg to a local
            database, epsgref.json, located in the user's data directory.

        Returns
        -------
        str
            text for a projection (*.prj) file.

        """
        epsgfile = EpsgReference()
        wktstr = epsgfile.get(epsg)
        if wktstr is None:
            wktstr = CRS.get_spatialreference(epsg, text=text)
        if addlocalreference and wktstr is not None:
            epsgfile.add(epsg, wktstr)
        return wktstr

    @staticmethod
    def get_spatialreference(epsg, text="esriwkt"):
        """
        Gets text for given epsg code and text format from spatialreference.org
        Fetches the reference text using the url:
            https://spatialreference.org/ref/epsg/<epsg code>/<text>/
        See: https://www.epsg-registry.org/

        Parameters
        ----------
        epsg : int
            epsg code for coordinate system
        text : str
            string added to url

        Returns
        -------
        str

        """
        from ..utils.flopy_io import get_url_text

        epsg_categories = (
            "epsg",
            "esri",
        )
        urls = []
        for cat in epsg_categories:
            url = f"{srefhttp}/ref/{cat}/{epsg}/{text}/"
            urls.append(url)
            result = get_url_text(url)
            if result is not None:
                break
        if result is not None:
            return result.replace("\n", "")
        elif result is None and text != "epsg":
            error_msg = (
                f"No internet connection or epsg code {epsg} not found at:\n"
            )
            for idx, url in enumerate(urls):
                error_msg += f"  {idx + 1:>2d}: {url}\n"
            print(error_msg)
        # epsg code not listed on spatialreference.org
        # may still work with pyproj
        elif text == "epsg":
            return f"epsg:{epsg}"

    @staticmethod
    def getproj4(epsg):
        """
        Gets projection file (.prj) text for given epsg code from
        spatialreference.org. See: https://www.epsg-registry.org/

        Parameters
        ----------
        epsg : int
            epsg code for coordinate system

        Returns
        -------
        str
            text for a projection (*.prj) file.
        """
        return CRS.get_spatialreference(epsg, text="proj4")


class EpsgReference:
    r"""
    Sets up a local database of text representations of coordinate reference
    systems, keyed by EPSG code.

    The database is epsgref.json, located in either "%LOCALAPPDATA%\flopy"
    for Windows users, or $HOME/.local/share/flopy for others.
    """

    def __init__(self):
        if sys.platform.startswith("win"):
            flopy_appdata = Path(expandvars(r"%LOCALAPPDATA%\flopy"))
        else:
            flopy_appdata = Path.home() / ".local" / "share" / "flopy"
        if not flopy_appdata.exists():
            flopy_appdata.mkdir(parents=True, exist_ok=True)
        dbname = "epsgref.json"
        self.location = flopy_appdata / dbname

    def to_dict(self):
        """
        returns dict with EPSG code integer key, and WKT CRS text
        """
        data = {}
        if self.location.exists():
            loaded_data = json.loads(self.location.read_text())
            # convert JSON key from str to EPSG integer
            for key, value in loaded_data.items():
                try:
                    data[int(key)] = value
                except ValueError:
                    data[key] = value
        return data

    def _write(self, data):
        with self.location.open("w") as f:
            json.dump(data, f, indent=0)
            f.write("\n")

    def reset(self, verbose=True):
        if self.location.exists():
            if verbose:
                print(f"Resetting {self.location}")
            self.location.unlink()
        elif verbose:
            print(f"{self.location} does not exist, no reset required")

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
        ep = EpsgReference()
        prj = ep.to_dict()
        for k, v in prj.items():
            print(f"{k}:\n{v}\n")
