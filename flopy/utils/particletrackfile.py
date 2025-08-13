"""
Utilities for parsing particle tracking output files.
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import numpy as np
from numpy.lib.recfunctions import stack_arrays

from .utl_import import import_optional_dependency

MIN_PARTICLE_TRACK_DTYPE = np.dtype(
    [
        ("x", np.float32),
        ("y", np.float32),
        ("z", np.float32),
        ("time", np.float32),
        ("k", np.int32),
        ("particleid", np.int32),
    ]
)


class ParticleTrackFile(ABC):
    """
    Abstract base class for particle track output files. Exposes a unified API
    supporting MODPATH versions 3, 5, 6 and 7, as well as MODFLOW 6 PRT models.

    Notes
    -----


    Parameters
    ----------
    filename : str or PathLike
        Path of output file
    verbose : bool
        Show verbose output. Default is False.

    """

    outdtype = MIN_PARTICLE_TRACK_DTYPE
    """
    Minimal information shared by all particle track file formats.
    Track data are converted to this dtype for internal storage
    and for return from (sub-)class methods.
    """

    def __init__(
        self,
        filename: Union[str, os.PathLike],
        verbose: bool = False,
    ):
        self.fname = Path(filename).expanduser().absolute()
        self.verbose = verbose

    def get_maxid(self) -> int:
        """
        Get the maximum particle ID.

        Returns
        -------
        out : int
            Maximum particle ID.

        """
        return self._data["particleid"].max()

    def get_maxtime(self) -> float:
        """
        Get the maximum tracking time.

        Returns
        -------
        out : float
            Maximum tracking time.

        """
        return self._data["time"].max()

    def get_data(self, partid=0, totim=None, ge=True, minimal=False) -> np.recarray:
        """
        Get a single particle track, optionally filtering by time.

        Parameters
        ----------
        partid : int
            Zero-based particle id. Default is 0.
        totim : float
            The simulation time. Default is None.
        ge : bool
            Filter tracking times greater than or equal
            to or less than or equal to totim.
            Only used if totim is not None.
        minimal : bool
            Whether to return only the minimal, canonical fields. Default is False.

        Returns
        -------
        data : np.recarray
            Recarray with dtype ParticleTrackFile.outdtype

        """
        data = self._data[list(self.outdtype.names)] if minimal else self._data
        idx = (
            np.asarray(data["particleid"] == partid).nonzero()[0]
            if totim is None
            else (
                np.asarray(
                    (data["time"] >= totim) & (data["particleid"] == partid)
                ).nonzero()[0]
                if ge
                else np.asarray(
                    (data["time"] <= totim) & (data["particleid"] == partid)
                ).nonzero()[0]
            )
        )

        return data[idx]

    def get_alldata(self, totim=None, ge=True, minimal=False):
        """
        Get all particle tracks separately, optionally filtering by time.

        Parameters
        ----------
        totim : float
            The simulation time.
        ge : bool
            Boolean that determines if pathline times greater than or equal
            to or less than or equal to totim.
        minimal : bool
            Whether to return only the minimal, canonical fields. Default is False.

        Returns
        -------
        data : list of numpy record arrays
            List of recarrays with dtype ParticleTrackFile.outdtype

        """
        nids = np.unique(self._data["particleid"])
        data = self._data[list(self.outdtype.names)] if minimal else self._data
        if totim is not None:
            idx = (
                np.asarray(data["time"] >= totim).nonzero()[0]
                if ge
                else np.asarray(data["time"] <= totim).nonzero()[0]
            )
            if len(idx) > 0:
                data = data[idx]
        return [data[data["particleid"] == i] for i in nids]

    def get_destination_data(self, dest_cells, to_recarray=True) -> np.recarray:
        """
        Get data for set of destination cells.

        Parameters
        ----------
        dest_cells : list or array of tuples
            (k, i, j) of each destination cell for MODPATH versions less than
            MODPATH 7 or node number of each destination cell. (zero based)
        to_recarray : bool
            Boolean that controls returned series. If to_recarray is True,
            a single recarray with all of the pathlines that intersect
            dest_cells are returned. If to_recarray is False, a list of
            recarrays (the same form as returned by get_alldata method)
            that intersect dest_cells are returned (default is False).

        Returns
        -------
        data : np.recarray
            Slice of data array (e.g. PathlineFile._data, TimeseriesFile._data)
            containing endpoint, pathline, or timeseries data that intersect
            (k,i,j) or (node) dest_cells.

        """

        return self.intersect(dest_cells, to_recarray)

    @abstractmethod
    def intersect(self, cells, to_recarray) -> np.recarray:
        """Find intersection of pathlines with cells."""
        pass

    def to_geo_dataframe(
        self,
        modelgrid,
        data=None,
        one_per_particle=True,
        direction="ending",
    ):
        """
        Create a geodataframe of particle tracks.

        Parameters
        ----------
        modelgrid : flopy.discretization.Grid instance
            Used to scale and rotate Global x, y, z values.
        data : np.recarray
            Record array of same form as that returned by
            get_alldata(). (if none, get_alldata() is exported).
        one_per_particle : boolean (default True)
            True writes a single LineString with a single set of attribute
            data for each particle. False writes a record/geometry for each
            pathline segment (each row in the PathLine file). This option can
            be used to visualize attribute information (time, model layer,
            etc.) across a pathline in a GIS.
        direction : str
            String defining if starting or ending particle locations should be
            included in shapefile attribute information. Only used if
            one_per_particle=False. (default is 'ending')

        Returns
        -------
            GeoDataFrame
        """
        from . import geometry

        shapely_geo = import_optional_dependency("shapely.geometry")
        gpd = import_optional_dependency("geopandas")

        if data is None:
            data = self._data.view(np.recarray)
        else:
            # convert pathline list to a single recarray
            if isinstance(data, list):
                s = data[0]
                print(s.dtype)
                for n in range(1, len(data)):
                    s = stack_arrays((s, data[n]))
                data = s.view(np.recarray)

        data = data.copy()
        data.sort(order=["particleid", "time"])

        particles = np.unique(data.particleid)
        geoms = []

        # create a dict of attrs?
        headings = ["particleid", "particlegroup", "time", "k", "i", "j", "node"]
        attrs = []
        for h in headings:
            if h in data.dtype.names:
                attrs.append(h)

        if one_per_particle:
            dfdata = {a: [] for a in attrs}
            if direction == "ending":
                idx = -1
            else:
                idx = 0

            for p in particles:
                ra = data[data.particleid == p]
                for k in dfdata.keys():
                    if k == "time":
                        dfdata[k].append(np.max(ra[k]))
                    else:
                        dfdata[k].append(ra[k][idx])

                x, y = geometry.transform(
                    ra.x,
                    ra.y,
                    modelgrid.xoffset,
                    modelgrid.yoffset,
                    modelgrid.angrot_radians,
                )
                z = ra.z

                line = list(zip(x, y, z))
                geoms.append(shapely_geo.LineString(line))

        else:
            dfdata = {a: [] for a in data.dtype.names}
            for pid in particles:
                ra = data[data.particleid == pid]
                x, y = geometry.transform(
                    ra.x,
                    ra.y,
                    modelgrid.xoffset,
                    modelgrid.yoffset,
                    modelgrid.angrot_radians,
                )
                z = ra.z
                geoms += [
                    shapely_geo.LineString(
                        [(x[i - 1], y[i - 1], z[i - 1]), (x[i], y[i], z[i])]
                    )
                    for i in np.arange(1, (len(ra)))
                ]
                for k in dfdata.keys():
                    dfdata[k].extend(ra[k][1:])

        # now create a geodataframe
        gdf = gpd.GeoDataFrame(dfdata, geometry=geoms, crs=modelgrid.crs)

        # adjust to 1 based node numbers
        for col in list(gdf):
            if col in self.kijnames:
                gdf[col] += 1

        return gdf

    def write_shapefile(
        self,
        data=None,
        one_per_particle=True,
        direction="ending",
        shpname="endpoints.shp",
        mg=None,
        crs=None,
        **kwargs,
    ):
        """
        Write particle track data to a shapefile.

        data : np.recarray
            Record array of same form as that returned by
            get_alldata(). (if none, get_alldata() is exported).
        one_per_particle : boolean (default True)
            True writes a single LineString with a single set of attribute
            data for each particle. False writes a record/geometry for each
            pathline segment (each row in the PathLine file). This option can
            be used to visualize attribute information (time, model layer,
            etc.) across a pathline in a GIS.
        direction : str
            String defining if starting or ending particle locations should be
            included in shapefile attribute information. Only used if
            one_per_particle=False. (default is 'ending')
        shpname : str
            File path for shapefile
        mg : flopy.discretization.grid instance
            Used to scale and rotate Global x,y,z values.
        crs : pyproj.CRS, int, str, optional
            Coordinate reference system (CRS) for the model grid
            (must be projected; geographic CRS are not supported).
            The value can be anything accepted by
            :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
            such as an authority string (eg "EPSG:26916") or a WKT string.
        kwargs : keyword arguments to flopy.export.shapefile_utils.recarray2shp

          .. deprecated:: 3.5
             The following keyword options will be removed for FloPy 3.6:

               - ``epsg`` (int): use ``crs`` instead.

        """
        import warnings

        warnings.warn(
            "write_shapefile will be Deprecated, please use to_geo_dataframe()",
            DeprecationWarning,
        )
        gdf = self.to_geo_dataframe(
            modelgrid=mg,
            data=data,
            one_per_particle=one_per_particle,
            direction=direction,
        )
        if crs is not None:
            if gdf.crs is None:
                gdf = gdf.set_crs(crs)
            else:
                gdf = gdf.to_crs(crs)

        gdf.to_file(shpname)
