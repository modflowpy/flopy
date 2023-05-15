"""
Module for exporting and importing flopy model attributes

"""
import copy
import json
import os
import shutil
import sys
import warnings
from pathlib import Path
from typing import Optional, Union
from warnings import warn

import numpy as np

from ..datbase import DataInterface, DataType
from ..utils import Util3d, flopy_io, import_optional_dependency
from ..utils.crs import get_crs

# web address of spatial reference dot org
srefhttp = "https://spatialreference.org"


def write_gridlines_shapefile(filename: Union[str, os.PathLike], mg):
    """
    Write a polyline shapefile of the grid lines - a lightweight alternative
    to polygons.

    Parameters
    ----------
    filename : str or PathLike
        path of the shapefile to write
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
    write_prj(filename, modelgrid=mg)
    return


def write_grid_shapefile(
    path: Union[str, os.PathLike],
    mg,
    array_dict,
    nan_val=np.nan,
    crs=None,
    prjfile=None,
    epsg=None,
    prj: Optional[Union[str, os.PathLike]] = None,
    verbose=False,
):
    """
    Method to write a shapefile of gridded input data

    Parameters
    ----------
    path : str or PathLike
        shapefile file path
    mg : flopy.discretization.Grid object
        flopy model grid
    array_dict : dict
        dictionary of model input arrays
    nan_val : float
        value to fill nans
    crs : pyproj.CRS, optional if `prjfile` is specified
        Coordinate reference system (CRS) for the model grid
        (must be projected; geographic CRS are not supported).
        The value can be anything accepted by
        :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
        such as an authority string (eg "EPSG:26916") or a WKT string.
    prjfile : str or pathlike, optional if `crs` is specified
        ESRI-style projection file with well-known text defining the CRS
        for the model grid (must be projected; geographic CRS are not supported).

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
    if verbose:
        print(f"wrote {flopy_io.relpath_safe(path)}")

    # write the projection file
    write_prj(path, mg, crs=crs, epsg=epsg, prj=prj, prjfile=prjfile)
    return


def model_attributes_to_shapefile(
    path: Union[str, os.PathLike],
    ml,
    package_names=None,
    array_dict=None,
    verbose=False,
    **kwargs,
):
    """
    Wrapper function for writing a shapefile of model data.  If package_names
    is not None, then search through the requested packages looking for arrays
    that can be added to the shapefile as attributes

    Parameters
    ----------
    path : str or PathLike
        path to write the shapefile to
    ml : flopy.mbase
        model instance
    package_names : list of package names (e.g. ["dis","lpf"])
        Packages to export data arrays to shapefile. (default is None)
    array_dict : dict of {name:2D array} pairs
       Additional 2D arrays to add as attributes to the shapefile.
       (default is None)
    verbose : bool, optional, default False
        whether to print verbose output
    **kwargs : keyword arguments
        modelgrid : fp.modflow.Grid object
            if modelgrid is supplied, user supplied modelgrid is used in lieu
            of the modelgrid attached to the modflow model object
        crs : pyproj.CRS, optional if `prjfile` is specified
            Coordinate reference system (CRS) for the model grid
            (must be projected; geographic CRS are not supported).
            The value can be anything accepted by
            :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
            such as an authority string (eg "EPSG:26916") or a WKT string.
        prjfile : str or pathlike, optional if `crs` is specified
            ESRI-style projection file with well-known text defining the CRS
            for the model grid (must be projected; geographic CRS are not supported).

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
                if a.data_type == DataType.array2d:
                    if a.array is None or a.array.shape != horz_shape:
                        warn(
                            "Failed to get data for "
                            f"{a.name} array, {pak.name[0]} package"
                        )
                        continue
                    name = shape_attr_name(a.name, keep_layer=True)
                    # name = a.name.lower()
                    array_dict[name] = a.array
                elif a.data_type == DataType.array3d:
                    # Not sure how best to check if an object has array data
                    if a.array is None:
                        warn(
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
                        warn(
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
    crs = kwargs.get("crs", None)
    prjfile = kwargs.get("prjfile", None)
    write_prj(path, grid, crs=crs, prjfile=prjfile)


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


def shp2recarray(shpname: Union[str, os.PathLike]):
    """Read a shapefile into a numpy recarray.

    Parameters
    ----------
    shpname : str or PathLike
        ESRI Shapefile path

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
    shpname: Union[str, os.PathLike] = "recarray.shp",
    mg=None,
    crs=None,
    prjfile=None,
    epsg=None,
    prj: Optional[Union[str, os.PathLike]] = None,
    verbose=False,
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
    shpname : str or PathLike, default "recarray.shp"
        Path for the output shapefile
    crs : pyproj.CRS, optional if `prjfile` is specified
        Coordinate reference system (CRS) for the model grid
        (must be projected; geographic CRS are not supported).
        The value can be anything accepted by
        :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
        such as an authority string (eg "EPSG:26916") or a WKT string.
    prjfile : str or pathlike, optional if `crs` is specified
        ESRI-style projection file with well-known text defining the CRS
        for the model grid (must be projected; geographic CRS are not supported).

    Notes
    -----
    Uses pyshp.
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
    write_prj(shpname, mg, crs=crs, epsg=epsg, prj=prj, prjfile=prjfile)
    print(f"wrote {flopy_io.relpath_safe(os.getcwd(), shpname)}")
    return


def write_prj(
    shpname,
    modelgrid=None,
    crs=None,
    epsg=None,
    prj=None,
    prjfile=None,
    wkt_string=None,
):
    # projection file name
    output_projection_file = Path(shpname).with_suffix(".prj")

    crs = get_crs(
        prjfile=prjfile, prj=prj, epsg=epsg, crs=crs, wkt_string=wkt_string
    )
    if crs is None and modelgrid is not None:
        crs = modelgrid.crs
    if crs is not None:
        with open(output_projection_file, "w", encoding="utf-8") as dest:
            write_text = crs.to_wkt()
            dest.write(write_text)
    else:
        print(
            "No CRS information for writing a .prj file.\n"
            "Supply an valid coordinate system reference to the attached modelgrid object "
            "or .export() method."
        )
