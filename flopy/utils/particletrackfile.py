"""
Utilities for parsing particle tracking output files.
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import numpy as np
from numpy.lib.recfunctions import stack_arrays

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
        ----------
        out : int
            Maximum particle ID.

        """
        return self._data["particleid"].max()

    def get_maxtime(self) -> float:
        """
        Get the maximum tracking time.

        Returns
        ----------
        out : float
            Maximum tracking time.

        """
        return self._data["time"].max()

    def get_data(
        self, partid=0, totim=None, ge=True, minimal=False
    ) -> np.recarray:
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
        ----------
        data : np.recarray
            Recarray with dtype ParticleTrackFile.outdtype

        """
        data = self._data[list(self.outdtype.names)] if minimal else self._data
        idx = (
            np.where(data["particleid"] == partid)[0]
            if totim is None
            else (
                np.where(
                    (data["time"] >= totim) & (data["particleid"] == partid)
                )[0]
                if ge
                else np.where(
                    (data["time"] <= totim) & (data["particleid"] == partid)
                )[0]
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
        ----------
        data : list of numpy record arrays
            List of recarrays with dtype ParticleTrackFile.outdtype

        """
        nids = np.unique(self._data["particleid"]).size
        data = self._data[list(self.outdtype.names)] if minimal else self._data
        if totim is not None:
            idx = (
                np.where(data["time"] >= totim)[0]
                if ge
                else np.where(data["time"] <= totim)[0]
            )
            if len(idx) > 0:
                data = data[idx]
        return [data[data["particleid"] == i] for i in range(nids)]

    def get_destination_data(
        self, dest_cells, to_recarray=True
    ) -> np.recarray:
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
        from ..export.shapefile_utils import recarray2shp
        from . import geometry
        from .geometry import LineString

        series = data
        if series is None:
            series = self._data.view(np.recarray)
        else:
            # convert pathline list to a single recarray
            if isinstance(series, list):
                s = series[0]
                print(s.dtype)
                for n in range(1, len(series)):
                    s = stack_arrays((s, series[n]))
                series = s.view(np.recarray)

        series = series.copy()
        series.sort(order=["particleid", "time"])

        if mg is None:
            raise ValueError("A modelgrid object was not provided.")

        particles = np.unique(series.particleid)
        geoms = []

        # create dtype with select attributes in pth
        names = series.dtype.names
        dtype = []
        atts = ["particleid", "particlegroup", "time", "k", "i", "j", "node"]
        for att in atts:
            if att in names:
                t = np.int32
                if att == "time":
                    t = np.float32
                dtype.append((att, t))
        dtype = np.dtype(dtype)

        # reset names to the selected names in the created dtype
        names = dtype.names

        # 1 geometry for each path
        if one_per_particle:
            loc_inds = 0
            if direction == "ending":
                loc_inds = -1

            sdata = []
            for pid in particles:
                ra = series[series.particleid == pid]

                x, y = geometry.transform(
                    ra.x, ra.y, mg.xoffset, mg.yoffset, mg.angrot_radians
                )
                z = ra.z
                geoms.append(LineString(list(zip(x, y, z))))

                t = [pid]
                if "particlegroup" in names:
                    t.append(ra.particlegroup[0])
                t.append(ra.time.max())
                if "node" in names:
                    t.append(ra.node[loc_inds])
                else:
                    if "k" in names:
                        t.append(ra.k[loc_inds])
                    if "i" in names:
                        t.append(ra.i[loc_inds])
                    if "j" in names:
                        t.append(ra.j[loc_inds])
                sdata.append(tuple(t))

            sdata = np.array(sdata, dtype=dtype).view(np.recarray)

        # geometry for each row in PathLine file
        else:
            dtype = series.dtype
            sdata = []
            for pid in particles:
                ra = series[series.particleid == pid]
                if mg is not None:
                    x, y = geometry.transform(
                        ra.x, ra.y, mg.xoffset, mg.yoffset, mg.angrot_radians
                    )
                else:
                    x, y = geometry.transform(ra.x, ra.y, 0, 0, 0)
                z = ra.z
                geoms += [
                    LineString(
                        [(x[i - 1], y[i - 1], z[i - 1]), (x[i], y[i], z[i])]
                    )
                    for i in np.arange(1, (len(ra)))
                ]
                sdata += ra[1:].tolist()
            sdata = np.array(sdata, dtype=dtype).view(np.recarray)

        # convert back to one-based
        for n in set(self.kijnames).intersection(set(sdata.dtype.names)):
            sdata[n] += 1

        # write the final recarray to a shapefile
        recarray2shp(sdata, geoms, shpname=shpname, crs=crs, **kwargs)
