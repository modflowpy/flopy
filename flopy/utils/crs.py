"""Utilities related to coordinate reference system handling."""

import warnings
from pathlib import Path

from ..utils import import_optional_dependency


def get_authority_crs(crs):
    """Try to get the authority representation for a
    coordinate reference system (CRS), for more robust
    comparison with other CRS objects.

    Parameters
    ----------
    crs : pyproj.CRS, int, str
        Coordinate reference system (CRS) for the model grid
        (must be projected; geographic CRS are not supported).
        The value can be anything accepted by
        :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
        such as an authority string (eg "EPSG:26916") or a WKT string.

    Returns
    -------
    pyproj.CRS instance
        CRS instance initialized with the name
        and authority code (e.g. "epsg:5070") produced by
        :meth:`pyproj.crs.CRS.to_authority`

    Notes
    -----
    :meth:`pyproj.crs.CRS.to_authority` will return None if a matching
    authority name and code can't be found. In this case,
    the input crs instance will be returned.

    References
    ----------
    http://pyproj4.github.io/pyproj/stable/api/crs/crs.html

    """
    pyproj = import_optional_dependency("pyproj")
    if crs is not None:
        crs = pyproj.crs.CRS.from_user_input(crs)
        authority = crs.to_authority()
        if authority is not None:
            return pyproj.CRS.from_user_input(authority)
        return crs


def get_shapefile_crs(shapefile):
    """Get the coordinate reference system for a shapefile.

    Parameters
    ----------
    shapefile : str or pathlike
        Path to a shapefile or an associated projection (.prj) file.

    Returns
    -------
    pyproj.CRS instance

    """
    shapefile = Path(shapefile)
    prjfile = shapefile.with_suffix(".prj")
    if prjfile.exists():
        pyproj = import_optional_dependency("pyproj")
        wkt = prjfile.read_text(encoding="utf-8")
        crs = pyproj.crs.CRS.from_wkt(wkt)
        return get_authority_crs(crs)
    else:
        raise FileNotFoundError(f"could not find {prjfile}")


def get_crs(prjfile=None, crs=None, **kwargs):
    """Helper function to produce a pyproj.CRS object from various input.

    Notes
    -----
    Longer-term, this would just handle the ``crs``
    and ``prjfile`` arguments, but in the near term, we need to
    warn users about deprecating the ``prj``, ``epsg``, ``proj4``
    and ``wkt_string`` inputs.

    Parameters
    ----------
    prjfile : str or pathlike, optional
        ESRI-style projection file with well-known text defining the CRS
        for the model grid (must be projected; geographic CRS are not supported).
    crs : pyproj.CRS, int, str, optional if `prjfile` is specified
        Coordinate reference system (CRS) for the model grid
        (must be projected; geographic CRS are not supported).
        The value can be anything accepted by
        :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
        such as an authority string (eg "EPSG:26916") or a WKT string.
    **kwargs : dict, optional
        Support deprecated keyword options.

        .. deprecated:: 3.4
           The following keyword options will be removed for FloPy 3.6:

             - ``prj`` (str or pathlike): use ``prjfile`` instead.
             - ``epsg`` (int): use ``crs`` instead.
             - ``proj4`` (str): use ``crs`` instead.
             - ``wkt_string`` (str): use ``crs`` instead.

    Returns
    -------
    pyproj.CRS instance

    """
    if crs is not None:
        crs = get_authority_crs(crs)
    if "prj" in kwargs:
        warnings.warn(
            "the prj argument will be deprecated and will be removed in "
            "version 3.6. Use prjfile instead.",
            PendingDeprecationWarning,
        )
        prjfile = kwargs.pop("prj")
    if "epsg" in kwargs:
        warnings.warn(
            "the epsg argument will be deprecated and will be removed in "
            "version 3.6. Use crs instead.",
            PendingDeprecationWarning,
        )
        epsg = kwargs.pop("epsg")
        if crs is None:
            crs = get_authority_crs(epsg)
    if prjfile is not None:
        prjfile_crs = get_shapefile_crs(prjfile)
        if (crs is not None) and (crs != prjfile_crs):
            raise ValueError(
                "Different coordinate reference systems "
                f"in crs argument and supplied projection file: {prjfile}\n"
                f"\nuser supplied crs: {crs}  !=\ncrs from projection file: {prjfile_crs}"
            )
        else:
            crs = prjfile_crs
    if "proj4" in kwargs:
        warnings.warn(
            "the proj4 argument will be deprecated and will be removed in "
            "version 3.6. Use crs instead.",
            PendingDeprecationWarning,
        )
        proj4 = kwargs.pop("proj4")
        if crs is None:
            crs = get_authority_crs(proj4)
    if "wkt_string" in kwargs:
        wkt_string = kwargs.pop("wkt_string")
        if crs is None:
            crs = get_authority_crs(wkt_string)
    if kwargs:
        raise TypeError(f"unhandled keywords: {kwargs}")
    if crs is not None and not crs.is_projected:
        raise ValueError(
            f"Only projected coordinate reference systems are supported.\n{crs}"
        )
    return crs
