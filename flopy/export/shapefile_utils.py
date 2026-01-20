"""
Module for exporting and importing flopy model attributes

"""

import copy
import json
import os
import shutil
import sys
import warnings
from os import PathLike
from pathlib import Path
from typing import Optional, Union
from warnings import warn

import numpy as np

from ..datbase import DataInterface, DataType
from ..utils import Util3d, flopy_io, import_optional_dependency
from ..utils.crs import get_crs


def write_gridlines_shapefile(filename: Union[str, PathLike], modelgrid):
    """
    Write a polyline shapefile of the grid lines - a lightweight alternative
    to polygons.

    Parameters
    ----------
    filename : str or PathLike
        path of the shapefile to write
    modelgrid : flopy.discretization.grid.Grid object
        flopy model grid

    Returns
    -------
    None

    """
    warnings.warn(
        "the write_gridlines_shapefile() utility will be deprecated and is "
        "replaced by modelgrid.to_geodataframe()",
        DeprecationWarning,
    )

    from ..discretization.grid import Grid

    if not isinstance(modelgrid, Grid):
        raise ValueError(
            f"'modelgrid' must be a flopy Grid subclass instance; "
            f"found '{type(modelgrid)}'"
        )

    gdf = modelgrid.grid_line_geodataframe()
    gdf.write_file(filename)


def write_grid_shapefile(
    filename: Union[str, PathLike],
    mg,
    array_dict,
    nan_val=np.nan,
    crs=None,
    prjfile: Union[str, PathLike, None] = None,
    verbose=False,
    **kwargs,
):
    """
    DEPRECATED -- removal planned in flopy version 3.11. Functionality replaced by
    `to_geodataframe` on modelgrid object

    Method to write a shapefile of gridded input data

    Parameters
    ----------
    filename : str or PathLike
        shapefile file path
    mg : flopy.discretization.grid.Grid object
        flopy model grid
    array_dict : dict
        dictionary of model input arrays
    nan_val : float
        value to fill nans
    crs : pyproj.CRS, int, str, optional if `prjfile` is specified
        Coordinate reference system (CRS) for the model grid
        (must be projected; geographic CRS are not supported).
        The value can be anything accepted by
        :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
        such as an authority string (eg "EPSG:26916") or a WKT string.
    prjfile : str or PathLike, optional if `crs` is specified
        ESRI-style projection file with well-known text defining the CRS
        for the model grid (must be projected; geographic CRS are not supported).
    **kwargs : dict, optional
        Support deprecated keyword options.

        .. deprecated:: 3.5
           The following keyword options will be removed for FloPy 3.6:

             - ``epsg`` (int): use ``crs`` instead.
             - ``prj`` (str or PathLike): use ``prjfile`` instead.

    Returns
    -------
    None

    """
    warnings.warn(
        "the write_grid_shapefile() utility will be deprecated and is "
        "replaced by modelgrid.to_geodataframe()",
        DeprecationWarning,
    )

    from ..discretization.grid import Grid

    if not isinstance(mg, Grid):
        raise ValueError(
            f"'modelgrid' must be a flopy Grid subclass instance; found '{type(mg)}'"
        )
    gdf = mg.to_geodataframe()
    names = list(array_dict.keys())
    at = np.vstack([array_dict[name].ravel() for name in names])
    names = enforce_10ch_limit(names)

    for ix, name in enumerate(names):
        gdf[name] = at[ix]

    gdf = gdf.fillna(nan_val)

    if "epsg" in kwargs:
        epsg = kwargs.pop("epsg")
        crs = f"EPSG:{epsg}"

    if crs is not None:
        if gdf.crs is None:
            gdf = gdf.set_crs(crs)
        else:
            gdf = gdf.to_crs(crs)

    gdf.to_file(filename)

    if verbose:
        print(f"wrote {flopy_io.relpath_safe(os.getcwd(), filename)}")

    if "prj" in kwargs or "prjfile" in kwargs or "wkt_string" in kwargs:
        try:
            write_prj(filename, mg, crs=crs, prjfile=prjfile, **kwargs)
        except ImportError:
            if verbose:
                print("projection file not written")


def model_attributes_to_shapefile(
    path: Union[str, PathLike],
    ml,
    package_names=None,
    array_dict=None,
    verbose=False,
    **kwargs,
):
    """
    DEPRECATED -- removal planned in flopy version 3.11. Functionality replaced by
    `to_geodataframe` on model object

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
        crs : pyproj.CRS, int, str, optional if `prjfile` is specified
            Coordinate reference system (CRS) for the model grid
            (must be projected; geographic CRS are not supported).
            The value can be anything accepted by
            :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
            such as an authority string (eg "EPSG:26916") or a WKT string.
        prjfile : str or PathLike, optional if `crs` is specified
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
    warnings.warn(
        "model_attributes_to_shapefile is deprecated, please use the built in "
        "to_geodataframe() method on the model object",
        DeprecationWarning,
    )

    if array_dict is None:
        array_dict = {}

    if package_names is not None:
        if not isinstance(package_names, list):
            package_names = [package_names]
    else:
        package_names = [pak.name[0] for pak in ml.packagelist]

    gdf = ml.to_geodataframe(package_names=package_names, shorten_attr=True)

    if array_dict:
        modelgrid = ml.modelgrid
        for name, array in array_dict.items():
            if modelgrid.grid_type() == "unstructured":
                gdf[name] = array.ravel()
            else:
                if array.size == modelgrid.ncpl:
                    gdf[name] = array.ravel()
                elif array.size % modelgrid.ncpl == 0:
                    array = array.reshape((-1, modelgrid.ncpl))
                    for ix, lay in enumerate(array):
                        gdf[f"{name}_{ix}"] = lay
                else:
                    raise ValueError(
                        f"{name} array size is not a multiple of ncpl {modelgrid.ncpl}"
                    )

    crs = kwargs.get("crs", None)
    if crs is not None:
        if gdf.crs is None:
            gdf = gdf.set_crs(crs)
        else:
            gdf = gdf.to_crs(crs)

    gdf.to_file(path)

    prjfile = kwargs.get("prjfile", None)
    if prjfile is not None:
        try:
            write_prj(path, ml.modelgrid, crs=crs, prjfile=prjfile)
        except ImportError:
            pass


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


def enforce_10ch_limit(names: list[str], warnings: bool = True) -> list[str]:
    """Enforce 10 character limit for fieldnames.
    Add suffix for duplicate names starting at 0.

    Parameters
    ----------
    names : list of strings
    warnings : whether to warn if names are truncated

    Returns
    -------
    list
        list of unique strings of len <= 10.
    """

    def truncate(s):
        name = s[:5] + s[-4:] + "_"
        if warnings:
            warn(f"Truncating shapefile fieldname {s} to {name}")
        return name

    names = [truncate(n) if len(n) > 10 else n for n in names]
    dups = {x: names.count(x) for x in names}
    suffix = {n: list(range(cnt)) for n, cnt in dups.items() if cnt > 1}
    for i, n in enumerate(names):
        if dups[n] > 1:
            names[i] = n[:9] + str(suffix[n].pop(0))
    return names


def shp2recarray(shpname: Union[str, PathLike]):
    """
    DEPRECATED - Functionality can be recreated with geopandas `read_file()` and
    `to_records()` methods.

    Read a shapefile into a numpy recarray.

    Parameters
    ----------
    filename : str or PathLike
        ESRI Shapefile path

    Returns
    -------
    np.recarray

    """
    warnings.warn(
        "shp2recarray will be deprecated, shapefiles can be read in using "
        "geopandas standard methods. e.g., gpd.read_file(filename)",
        DeprecationWarning,
    )

    from ..utils.geospatial_utils import GeoSpatialCollection

    gpd = import_optional_dependency("geopandas")
    gdf = gpd.read_file(shpname)
    recarray = gdf.to_records()
    return recarray


def recarray2shp(
    recarray,
    geoms,
    shpname: Union[str, PathLike] = "recarray.shp",
    mg=None,
    crs=None,
    prjfile: Union[str, PathLike, None] = None,
    verbose=False,
    **kwargs,
):
    """
    DEPRECATED -- Functionality can be recreated using `to_geodataframe` on
    data objects.

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
    mg : flopy.discretization.Grid object
        flopy model grid
    crs : pyproj.CRS, int, str, optional if `prjfile` is specified
        Coordinate reference system (CRS) for the model grid
        (must be projected; geographic CRS are not supported).
        The value can be anything accepted by
        :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
        such as an authority string (eg "EPSG:26916") or a WKT string.
    prjfile : str or PathLike, optional if `crs` is specified
        ESRI-style projection file with well-known text defining the CRS
        for the model grid (must be projected; geographic CRS are not supported).
    **kwargs : dict, optional
        Support deprecated keyword options.

        .. deprecated:: 3.5
           The following keyword options will be removed for FloPy 3.6:

             - ``epsg`` (int): use ``crs`` instead.
             - ``prj`` (str or PathLike): use ``prjfile`` instead.

    Notes
    -----
    Uses pyshp and optionally pyproj.
    """
    from ..utils.geospatial_utils import GeoSpatialCollection

    if len(recarray) != len(geoms):
        raise IndexError("Number of geometries must equal the number of records!")

    if len(recarray) == 0:
        raise Exception("Recarray is empty")

    # set up for geopandas
    gdf = GeoSpatialCollection(geoms).geodataframe
    names = enforce_10ch_limit(recarray.dtype.names)

    for ix, name in enumerate(names):
        ra_name = recarray.dtype.names[ix]
        gdf[name] = recarray[ra_name]

    if "epsg" in kwargs:
        epsg = kwargs.pop("epsg")
        crs = f"EPSG:{epsg}"

    if crs is not None:
        gdf = gdf.set_crs(crs)

    gdf.to_file(shpname)

    if verbose:
        print(f"wrote {flopy_io.relpath_safe(os.getcwd(), shpname)}")

    if "prj" in kwargs or "prjfile" in kwargs or "wkt_string" in kwargs:
        try:
            write_prj(shpname, mg, crs=crs, prjfile=prjfile, **kwargs)
        except ImportError:
            if verbose:
                print("projection file not written")


def write_prj(
    shpname,
    modelgrid=None,
    crs=None,
    prjfile=None,
    **kwargs,
):
    # projection file name
    output_projection_file = Path(shpname).with_suffix(".prj")

    # handle deprecated projection kwargs; warnings are raised in crs.py
    get_crs_args = {}
    if "epsg" in kwargs:
        get_crs_args["epsg"] = kwargs.pop("epsg")
    if "prj" in kwargs:
        get_crs_args["prj"] = kwargs.pop("prj")
    if "wkt_string" in kwargs:
        get_crs_args["wkt_string"] = kwargs.pop("wkt_string")
    if kwargs:
        raise TypeError(f"unhandled keywords: {kwargs}")
    crs = get_crs(prjfile=prjfile, crs=crs, **get_crs_args)
    if crs is None and modelgrid is not None:
        crs = modelgrid.crs
    if crs is not None:
        output_projection_file.write_text(crs.to_wkt(), encoding="utf-8")
    else:
        print(
            "No CRS information for writing a .prj file.\n"
            "Supply an valid coordinate system reference to the attached "
            "modelgrid object or .export() method."
        )
