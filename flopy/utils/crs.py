"""Utilities related to coordinate reference system handling.
"""
import warnings
from pathlib import Path

from ..utils import import_optional_dependency


def get_authority_crs(crs):
    """Try to get the authority representation for a
    coordinate reference system (CRS), for more robust
    comparison with other CRS objects.

    Parameters
    ----------
    crs : pyproj.CRS, optional if `prj` is specified
        Coordinate reference system (CRS) for the model grid
        (must be projected; geographic CRS are not supported).
        The value can be anything accepted by
        :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
        such as an authority string (eg "EPSG:26916") or a WKT string.

    Returns
    -------
    authority_crs : pyproj.CRS instance
        CRS instance initiallized with the name
        and authority code (e.g. epsg: 5070) produced by
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
        Path to a shapefile or an associated
        projection (.prj) file.

    Returns
    -------
    crs : pyproj.CRS instance

    """
    pyproj = import_optional_dependency("pyproj")
    shapefile = Path(shapefile)
    prjfile = shapefile.with_suffix(".prj")
    if prjfile.exists():
        with open(prjfile) as src:
            wkt = src.read()
            crs = pyproj.crs.CRS.from_wkt(wkt)
            return get_authority_crs(crs)


def get_crs(
    prjfile=None, prj=None, epsg=None, proj4=None, crs=None, wkt_string=None
):
    """Helper function to produce a pyproj.CRS object from
    various input. Longer-term, this would just handle the ``crs``
    and ``prjfile`` arguments, but in the near term, we need to
    warn users about deprecating epsg and proj_str."""
    if crs is not None:
        crs = get_authority_crs(crs)
    if prj is not None:
        warnings.warn(
            "the prj argument will be deprecated and will be removed in version "
            "3.4. Use prjfile instead.",
            PendingDeprecationWarning,
        )
        prjfile = prj
    if epsg is not None:
        warnings.warn(
            "the epsg argument will be deprecated and will be removed in version "
            "3.4. Use crs instead.",
            PendingDeprecationWarning,
        )
        if crs is None:
            crs = get_authority_crs(epsg)
    elif prjfile is not None:
        prjfile_crs = get_shapefile_crs(prjfile)
        if (crs is not None) and (crs != prjfile_crs):
            raise ValueError(
                "Different coordinate reference systems "
                f"in crs argument and supplied projection file: {prjfile}\n"
                f"\nuser supplied crs: {crs}  !=\ncrs from projection file: {prjfile_crs}"
            )
        else:
            crs = prjfile_crs
    elif proj4 is not None:
        warnings.warn(
            "the proj4 argument will be deprecated and will be removed in version "
            "3.4. Use crs instead.",
            PendingDeprecationWarning,
        )
        if crs is None:
            crs = get_authority_crs(proj4)
    elif wkt_string is not None:
        if crs is None:
            crs = get_authority_crs(wkt_string)
    if crs is not None and not crs.is_projected:
        raise ValueError(
            f"Only projected coordinate reference systems are supported.\n{crs}"
        )
    return crs
