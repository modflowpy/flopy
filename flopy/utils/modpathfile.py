"""
Support for MODPATH output files.
"""

import itertools
import os
from typing import List, Optional, Tuple, Union

import numpy as np
from numpy.lib.recfunctions import append_fields, repack_fields

from flopy.utils.particletrackfile import ParticleTrackFile

from ..utils.flopy_io import loadtxt


class ModpathFile(ParticleTrackFile):
    """Provides MODPATH output file support."""

    def __init__(
        self, filename: Union[str, os.PathLike], verbose: bool = False
    ):
        super().__init__(filename, verbose)
        self.output_type = self.__class__.__name__.lower().replace("file", "")
        (
            self.modpath,
            self.compact,
            self.skiprows,
            self.version,
            self.direction,
        ) = self.parse(filename, self.output_type)

    @staticmethod
    def parse(
        file_path: Union[str, os.PathLike], file_type: str
    ) -> Tuple[bool, int, int, Optional[int]]:
        """
        Extract preliminary information from a MODPATH output file:
            - whether in compact format
            - how many rows to skip
            - the MODPATH version

        Parameters
        ----------
        file_path : str or PathLike
            The output file path
        file_type : str
            The output file type: pathline, endpoint, or timeseries

        Returns
        -------
        out : bool, int, int
            A tuple (compact, skiprows, version)
        """

        modpath = True
        compact = False
        idx = 0
        skiprows = 0
        version = None
        direction = None
        with open(file_path, "r") as f:
            while True:
                line = f.readline()
                if isinstance(line, bytes):
                    line = line.decode()
                if skiprows < 1:
                    if f"MODPATH_{file_type.upper()}_FILE 6" in line.upper():
                        version = 6
                    elif (
                        f"MODPATH_{file_type.upper()}_FILE         7"
                        in line.upper()
                    ):
                        version = 7
                    elif "MODPATH 5.0" in line.upper():
                        version = 5
                        if "COMPACT" in line.upper():
                            compact = True
                    elif "MODPATH Version 3.00" in line.upper():
                        version = 3
                    if version is None:
                        modpath = False
                skiprows += 1
                if version in [6, 7]:
                    if file_type.lower() == "endpoint":
                        if idx == 1:
                            direction = 1
                            if int(line.strip()[0]) == 2:
                                direction = -1
                        idx += 1
                    if "end header" in line.lower():
                        break
                else:
                    break

        return modpath, compact, skiprows, version, direction

    def intersect(
        self, cells, to_recarray
    ) -> Union[List[np.recarray], np.recarray]:
        if self.version < 7:
            try:
                raslice = self._data[["k", "i", "j"]]
            except (KeyError, ValueError):
                raise KeyError(
                    "could not extract 'k', 'i', and 'j' keys "
                    "from {} data".format(self.output_type.lower())
                )
        else:
            try:
                raslice = self._data[["node"]]
            except (KeyError, ValueError):
                msg = "could not extract 'node' key from {} data".format(
                    self.output_type.lower()
                )
                raise KeyError(msg)
            if isinstance(cells, (list, tuple)):
                allint = all(isinstance(el, int) for el in cells)
                # convert to a list of tuples
                if allint:
                    t = []
                    for el in cells:
                        t.append((el,))
                        cells = t

        cells = np.array(cells, dtype=raslice.dtype)
        inds = np.in1d(raslice, cells)
        epdest = self._data[inds].copy().view(np.recarray)

        if to_recarray:
            # use particle ids to get the rest of the paths
            inds = np.in1d(self._data["particleid"], epdest.particleid)
            series = self._data[inds].copy()
            series.sort(order=["particleid", "time"])
            series = series.view(np.recarray)
        else:
            # collect unique particleids in selection
            partids = np.unique(epdest["particleid"])
            series = [self.get_data(partid) for partid in partids]

        return series


class PathlineFile(ModpathFile):
    """
    Particle pathline file.

    Parameters
    ----------
    filename : str or PathLike
        Path of the pathline file
    verbose : bool
        Show verbose output. Default is False.

    Examples
    --------

    >>> import flopy
    >>> pl_file = flopy.utils.PathlineFile('model.mppth')
    >>> pl1 = pthobj.get_data(partid=1)

    """

    dtypes = {
        **dict.fromkeys(
            [3, 5],
            np.dtype(
                [
                    ("particleid", np.int32),
                    ("x", np.float32),
                    ("y", np.float32),
                    ("zloc", np.float32),
                    ("z", np.float32),
                    ("time", np.float32),
                    ("j", np.int32),
                    ("i", np.int32),
                    ("k", np.int32),
                    ("cumulativetimestep", np.int32),
                ]
            ),
        ),
        6: np.dtype(
            [
                ("particleid", np.int32),
                ("particlegroup", np.int32),
                ("timepointindex", np.int32),
                ("cumulativetimestep", np.int32),
                ("time", np.float32),
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("k", np.int32),
                ("i", np.int32),
                ("j", np.int32),
                ("grid", np.int32),
                ("xloc", np.float32),
                ("yloc", np.float32),
                ("zloc", np.float32),
                ("linesegmentindex", np.int32),
            ]
        ),
        7: np.dtype(
            [
                ("particleid", np.int32),
                ("particlegroup", np.int32),
                ("sequencenumber", np.int32),
                ("particleidloc", np.int32),
                ("time", np.float32),
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("k", np.int32),
                ("node", np.int32),
                ("xloc", np.float32),
                ("yloc", np.float32),
                ("zloc", np.float32),
                ("stressperiod", np.int32),
                ("timestep", np.int32),
            ]
        ),
    }

    kijnames = [
        "k",
        "i",
        "j",
        "node",
        "particleid",
        "particlegroup",
        "linesegmentindex",
        "particleidloc",
        "sequencenumber",
    ]

    def __init__(
        self, filename: Union[str, os.PathLike], verbose: bool = False
    ):
        super().__init__(filename, verbose=verbose)
        self.dtype, self._data = self._load()
        self.nid = np.unique(self._data["particleid"])

    def _load(self) -> Tuple[np.dtype, np.ndarray]:
        dtype = self.dtypes[self.version]
        if self.version == 7:
            dtyper = np.dtype(
                [
                    ("node", np.int32),
                    ("x", np.float32),
                    ("y", np.float32),
                    ("z", np.float32),
                    ("time", np.float32),
                    ("xloc", np.float32),
                    ("yloc", np.float32),
                    ("zloc", np.float32),
                    ("k", np.int32),
                    ("stressperiod", np.int32),
                    ("timestep", np.int32),
                ]
            )
            idx = 0
            particle_pathlines = {}
            nrows = 0
            with open(self.fname) as f:
                while True:
                    if idx == 0:
                        for n in range(self.skiprows):
                            line = f.readline()
                    # read header line
                    try:
                        line = f.readline().strip()
                        if self.verbose:
                            print(line)
                        if len(line) < 1:
                            break
                    except:
                        break
                    t = [int(s) for j, s in enumerate(line.split()) if j < 4]
                    sequencenumber, group, particleid, pathlinecount = t[0:4]
                    nrows += pathlinecount
                    # read in the particle data
                    d = np.loadtxt(
                        itertools.islice(f, 0, pathlinecount), dtype=dtyper
                    )
                    key = (
                        idx,
                        sequencenumber,
                        group,
                        particleid,
                        pathlinecount,
                    )
                    particle_pathlines[key] = d.copy()
                    idx += 1

            # create data array
            data = np.zeros(nrows, dtype=dtype)

            # fill data
            ipos0 = 0
            for key, value in particle_pathlines.items():
                idx, sequencenumber, group, particleid, pathlinecount = key[
                    0:5
                ]
                ipos1 = ipos0 + pathlinecount
                # fill constant items for particle
                # particleid is not necessarily unique for all pathlines - use
                # sequencenumber which is unique
                data["particleid"][ipos0:ipos1] = sequencenumber
                # set particlegroup and sequence number
                data["particlegroup"][ipos0:ipos1] = group
                data["sequencenumber"][ipos0:ipos1] = sequencenumber
                # save particleidloc to particleid
                data["particleidloc"][ipos0:ipos1] = particleid
                # fill particle data
                for name in value.dtype.names:
                    data[name][ipos0:ipos1] = value[name]
                ipos0 = ipos1
        else:
            data = loadtxt(self.fname, dtype=dtype, skiprows=self.skiprows)

        # convert indices to zero-based
        for n in self.kijnames:
            if n in data.dtype.names:
                data[n] -= 1

        # sort by particle ID and time
        data.sort(order=["particleid", "time"])

        return dtype, data

    def get_destination_pathline_data(self, dest_cells, to_recarray=False):
        """
        Get pathline data that pass through a set of destination cells.

        Parameters
        ----------
        dest_cells : list or array of tuples
            (k, i, j) of each destination cell for MODPATH versions less than
            MODPATH 7 or node number of each destination cell. (zero based)
        to_recarray : bool
            Boolean that controls returned pthldest. If to_recarray is True,
            a single recarray with all of the pathlines that intersect
            dest_cells are returned. If to_recarray is False, a list of
            recarrays (the same form as returned by get_alldata method)
            that intersect dest_cells are returned (default is False).

        Returns
        -------
        np.recarray
            Slice of pathline data array (e.g. PathlineFile._data)
            containing only pathlines that pass through (k,i,j) or (node)
            dest_cells.

        Examples
        --------

        >>> import flopy
        >>> p = flopy.utils.PathlineFile('modpath.pathline')
        >>> p0 = p.get_destination_pathline_data([(0, 0, 0),
        ...                                       (1, 0, 0)])

        """
        return super().get_destination_data(
            dest_cells=dest_cells, to_recarray=to_recarray
        )

    def write_shapefile(
        self,
        data=None,
        pathline_data=None,
        one_per_particle=True,
        direction="ending",
        shpname="pathlines.shp",
        mg=None,
        crs=None,
        **kwargs,
    ):
        """
        Write pathlines to a shapefile.

        Parameters
        ----------
        data : np.recarray
            Record array of same form as that returned by
            .get_alldata() (if None, .get_alldata() is exported).
        timeseries_data : np.recarray
            Record array of same form as that returned by
            .get_alldata() (if None, .get_alldata() is exported).

            .. deprecated:: 3.7
                The ``timeseries_data`` option will be removed for FloPy 4. Use ``data`` instead.
        one_per_particle : boolean (default True)
            True writes a single LineString with a single set of attribute
            data for each particle. False writes a record/geometry for each
            pathline segment (each row in the Timeseries file). This option can
            be used to visualize attribute information (time, model layer,
            etc.) across a pathline in a GIS.
        direction : str
            String defining if starting or ending particle locations should be
            included in shapefile attribute information. Only used if
            one_per_particle=False. (default is 'ending')
        shpname : str
            File path for shapefile
        mg : flopy.discretization.grid instance
            Used to scale and rotate Global x,y,z values in MODPATH Timeseries
            file.
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
        ParticleTrackFile.write_shapefile(
            self,
            data=data if data is not None else pathline_data,
            one_per_particle=one_per_particle,
            direction=direction,
            shpname=shpname,
            mg=mg,
            crs=crs,
            **kwargs,
        )


class EndpointFile(ModpathFile):
    """
    Particle endpoint file.

    Parameters
    ----------
    filename : str or PathLike
        Path of the endpoint file
    verbose : bool
        Show verbose output. Default is False.

    Examples
    --------

    >>> import flopy
    >>> ep_file = flopy.utils.EndpointFile('model.mpend')
    >>> ep1 = endobj.get_data(partid=1)

    """

    dtypes = {
        **dict.fromkeys(
            [3, 5],
            np.dtype(
                [
                    ("zone", np.int32),
                    ("j", np.int32),
                    ("i", np.int32),
                    ("k", np.int32),
                    ("x", np.float32),
                    ("y", np.float32),
                    ("z", np.float32),
                    ("zloc", np.float32),
                    ("time", np.float32),
                    ("x0", np.float32),
                    ("y0", np.float32),
                    ("zloc0", np.float32),
                    ("j0", np.int32),
                    ("i0", np.int32),
                    ("k0", np.int32),
                    ("zone0", np.int32),
                    ("cumulativetimestep", np.int32),
                    ("ipcode", np.int32),
                    ("time0", np.float32),
                ]
            ),
        ),
        6: np.dtype(
            [
                ("particleid", np.int32),
                ("particlegroup", np.int32),
                ("status", np.int32),
                ("time0", np.float32),
                ("time", np.float32),
                ("initialgrid", np.int32),
                ("k0", np.int32),
                ("i0", np.int32),
                ("j0", np.int32),
                ("cellface0", np.int32),
                ("zone0", np.int32),
                ("xloc0", np.float32),
                ("yloc0", np.float32),
                ("zloc0", np.float32),
                ("x0", np.float32),
                ("y0", np.float32),
                ("z0", np.float32),
                ("finalgrid", np.int32),
                ("k", np.int32),
                ("i", np.int32),
                ("j", np.int32),
                ("cellface", np.int32),
                ("zone", np.int32),
                ("xloc", np.float32),
                ("yloc", np.float32),
                ("zloc", np.float32),
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("label", "|S40"),
            ]
        ),
        7: np.dtype(
            [
                ("particleid", np.int32),
                ("particlegroup", np.int32),
                ("particleidloc", np.int32),
                ("status", np.int32),
                ("time0", np.float32),
                ("time", np.float32),
                ("node0", np.int32),
                ("k0", np.int32),
                ("xloc0", np.float32),
                ("yloc0", np.float32),
                ("zloc0", np.float32),
                ("x0", np.float32),
                ("y0", np.float32),
                ("z0", np.float32),
                ("zone0", np.int32),
                ("initialcellface", np.int32),
                ("node", np.int32),
                ("k", np.int32),
                ("xloc", np.float32),
                ("yloc", np.float32),
                ("zloc", np.float32),
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("zone", np.int32),
                ("cellface", np.int32),
            ]
        ),
    }

    kijnames = [
        "k0",
        "i0",
        "j0",
        "node0",
        "k",
        "i",
        "j",
        "node",
        "particleid",
        "particlegroup",
        "particleidloc",
        "zone0",
        "zone",
    ]

    def __init__(
        self, filename: Union[str, os.PathLike], verbose: bool = False
    ):
        super().__init__(filename, verbose)
        self.dtype, self._data = self._load()
        self.nid = np.unique(self._data["particleid"])

    def _load(self) -> Tuple[np.dtype, np.ndarray]:
        dtype = self.dtypes[self.version]
        data = loadtxt(self.fname, dtype=dtype, skiprows=self.skiprows)

        # convert indices to zero-based
        for n in self.kijnames:
            if n in data.dtype.names:
                data[n] -= 1

        # add particle ids for earlier version of MODPATH
        if self.version < 6:
            shape = data.shape[0]
            pids = np.arange(1, shape + 1, 1, dtype=np.int32)
            data = append_fields(data, "particleid", pids)

        return dtype, data

    def get_maxtraveltime(self):
        """
        Get the maximum travel time.

        Returns
        ----------
        out : float
            Maximum travel time.

        """
        return (self._data["time"] - self._data["time0"]).max()

    def get_alldata(self):
        """
        Get endpoint data from the endpoint file for all endpoints.

        Parameters
        ----------

        Returns
        ----------
        data : numpy record array
            A numpy recarray with the endpoint particle data


        See Also
        --------

        Notes
        -----

        Examples
        --------

        >>> import flopy
        >>> endobj = flopy.utils.EndpointFile('model.mpend')
        >>> e = endobj.get_alldata()

        """
        return self._data.view(np.recarray).copy()

    def get_destination_endpoint_data(self, dest_cells, source=False):
        """
        Get endpoint data that terminate in a set of destination cells.

        Parameters
        ----------
        dest_cells : list or array of tuples
            (k, i, j) of each destination cell for MODPATH versions less than
            MODPATH 7 or node number of each destination cell. (zero based)
        source : bool
            Boolean to specify is dest_cells applies to source or
            destination cells (default is False).

        Returns
        -------
        np.recarray
            Slice of endpoint data array (e.g. EndpointFile.get_alldata)
            containing only endpoint data with final locations in (k,i,j) or
            (node) dest_cells.

        Examples
        --------

        >>> import flopy
        >>> e = flopy.utils.EndpointFile('modpath.endpoint')
        >>> e0 = e.get_destination_endpoint_data([(0, 0, 0),
        ...                                       (1, 0, 0)])

        """

        # create local copy of _data
        data = self.get_alldata()

        # find the intersection of endpoints and dest_cells
        # convert dest_cells to same dtype for comparison
        if self.version < 7:
            if source:
                keys = ["k0", "i0", "j0"]
            else:
                keys = ["k", "i", "j"]
            try:
                raslice = repack_fields(data[keys])
            except (KeyError, ValueError):
                raise KeyError(
                    "could not extract "
                    + "', '".join(keys)
                    + " from endpoint data."
                )
        else:
            if source:
                keys = ["node0"]
            else:
                keys = ["node"]
            try:
                raslice = repack_fields(data[keys])
            except (KeyError, ValueError):
                msg = f"could not extract '{keys[0]}' key from endpoint data"
                raise KeyError(msg)
            if isinstance(dest_cells, (list, tuple)):
                allint = all(isinstance(el, int) for el in dest_cells)
                # convert to a list of tuples
                if allint:
                    t = []
                    for el in dest_cells:
                        t.append((el,))
                        dest_cells = t
        dtype = []
        for key in keys:
            dtype.append((key, np.int32))
        dtype = np.dtype(dtype)
        dest_cells = np.array(dest_cells, dtype=dtype)

        inds = np.in1d(raslice, dest_cells)
        return data[inds].copy().view(np.recarray)

    def write_shapefile(
        self,
        data=None,
        endpoint_data=None,
        shpname="endpoints.shp",
        direction="ending",
        mg=None,
        crs=None,
        **kwargs,
    ):
        """
        Write particle starting / ending locations to shapefile.

        data : np.recarray
            Record array of same form as that returned by EndpointFile.get_alldata.
            (if none, EndpointFile.get_alldata() is exported).
        endpoint_data : np.recarray
            Record array of same form as that returned by EndpointFile.get_alldata.
            (if none, EndpointFile.get_alldata() is exported).

            .. deprecated:: 3.7
                The ``endpoint_data`` option will be removed for FloPy 4. Use ``data`` instead.
        shpname : str
            File path for shapefile
        direction : str
            String defining if starting or ending particle locations should be
            considered. (default is 'ending')
        mg : flopy.discretization.grid instance
            Used to scale and rotate Global x,y,z values in MODPATH Endpoint
            file.
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
        from ..discretization import StructuredGrid
        from ..export.shapefile_utils import recarray2shp
        from ..utils import geometry
        from ..utils.geometry import Point

        epd = (data if data is not None else endpoint_data).copy()
        if epd is None:
            epd = self.get_alldata()

        if direction.lower() == "ending":
            xcol, ycol, zcol = "x", "y", "z"
        elif direction.lower() == "starting":
            xcol, ycol, zcol = "x0", "y0", "z0"
        else:
            raise Exception(
                'flopy.map.plot_endpoint direction must be "ending" '
                'or "starting".'
            )
        if mg is None:
            raise ValueError("A modelgrid object was not provided.")

        if isinstance(mg, StructuredGrid):
            x, y = geometry.transform(
                epd[xcol],
                epd[ycol],
                xoff=mg.xoffset,
                yoff=mg.yoffset,
                angrot_radians=mg.angrot_radians,
            )
        else:
            x, y = mg.get_coords(epd[xcol], epd[ycol])
        z = epd[zcol]

        geoms = [Point(x[i], y[i], z[i]) for i in range(len(epd))]
        # convert back to one-based
        for n in self.kijnames:
            if n in epd.dtype.names:
                epd[n] += 1
        recarray2shp(epd, geoms, shpname=shpname, crs=crs, **kwargs)


class TimeseriesFile(ModpathFile):
    """
    Particle timeseries file.

    Parameters
    ----------
    filename : str or PathLike
        Path of the timeseries file
    verbose : bool
        Show verbose output. Default is False.

    Examples
    --------

    >>> import flopy
    >>> ts_file = flopy.utils.TimeseriesFile('model.timeseries')
    >>> ts1 = tsobj.get_data(partid=1)
    """

    dtypes = {
        **dict.fromkeys(
            [3, 5],
            np.dtype(
                [
                    ("timestepindex", np.int32),
                    ("particleid", np.int32),
                    ("node", np.int32),
                    ("x", np.float32),
                    ("y", np.float32),
                    ("z", np.float32),
                    ("zloc", np.float32),
                    ("time", np.float32),
                    ("timestep", np.int32),
                ]
            ),
        ),
        6: np.dtype(
            [
                ("timepointindex", np.int32),
                ("timestep", np.int32),
                ("time", np.float32),
                ("particleid", np.int32),
                ("particlegroup", np.int32),
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("grid", np.int32),
                ("k", np.int32),
                ("i", np.int32),
                ("j", np.int32),
                ("xloc", np.float32),
                ("yloc", np.float32),
                ("zloc", np.float32),
            ]
        ),
        7: np.dtype(
            [
                ("timepointindex", np.int32),
                ("timestep", np.int32),
                ("time", np.float32),
                ("particleid", np.int32),
                ("particlegroup", np.int32),
                ("particleidloc", np.int32),
                ("node", np.int32),
                ("xloc", np.float32),
                ("yloc", np.float32),
                ("zloc", np.float32),
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("k", np.int32),
            ]
        ),
    }

    kijnames = [
        "k",
        "i",
        "j",
        "node",
        "particleid",
        "particlegroup",
        "particleidloc",
        "timestep",
        "timestepindex",
        "timepointindex",
    ]

    def __init__(self, filename, verbose=False):
        super().__init__(filename, verbose)
        self.dtype, self._data = self._load()
        self.nid = np.unique(self._data["particleid"])

    def _load(self) -> Tuple[np.dtype, np.ndarray]:
        dtype = self.dtypes[self.version]
        if self.version in [3, 5] and not self.compact:
            dtype = np.dtype(
                [
                    ("timestepindex", np.int32),
                    ("particleid", np.int32),
                    ("j", np.int32),
                    ("i", np.int32),
                    ("k", np.int32),
                    ("x", np.float32),
                    ("y", np.float32),
                    ("z", np.float32),
                    ("zloc", np.float32),
                    ("time", np.float32),
                    ("timestep", np.int32),
                ]
            )

        data = loadtxt(self.fname, dtype=dtype, skiprows=self.skiprows)

        # convert indices to zero-based
        for n in self.kijnames:
            if n in data.dtype.names:
                data[n] -= 1

        # sort by particle ID and time
        data.sort(order=["particleid", "time"])

        return dtype, data

    def get_destination_timeseries_data(self, dest_cells):
        """
        Get timeseries data that pass through a set of destination cells.

        Parameters
        ----------
        dest_cells : list or array of tuples
            (k, i, j) of each destination cell for MODPATH versions less than
            MODPATH 7 or node number of each destination cell. (zero based)

        Returns
        -------
        np.recarray
            Slice of timeseries data array (e.g. TmeseriesFile._data)
            containing only timeseries that pass through (k,i,j) or
            (node) dest_cells.

        Examples
        --------

        >>> import flopy
        >>> ts = flopy.utils.TimeseriesFile('modpath.timeseries')
        >>> tss = ts.get_destination_timeseries_data([(0, 0, 0),
        ...                                           (1, 0, 0)])

        """
        return super().get_destination_data(dest_cells=dest_cells)

    def write_shapefile(
        self,
        data=None,
        timeseries_data=None,
        one_per_particle=True,
        direction="ending",
        shpname="pathlines.shp",
        mg=None,
        crs=None,
        **kwargs,
    ):
        """
        Write timeseries to a shapefile

        data : np.recarray
            Record array of same form as that returned by
            Timeseries.get_alldata(). (if none, Timeseries.get_alldata()
            is exported).
        timeseries_data : np.recarray
            Record array of same form as that returned by
            Timeseries.get_alldata(). (if none, Timeseries.get_alldata()
            is exported).

            .. deprecated:: 3.7
                The ``timeseries_data`` option will be removed for FloPy 4. Use ``data`` instead.
        one_per_particle : boolean (default True)
            True writes a single LineString with a single set of attribute
            data for each particle. False writes a record/geometry for each
            pathline segment (each row in the Timeseries file). This option can
            be used to visualize attribute information (time, model layer,
            etc.) across a pathline in a GIS.
        direction : str
            String defining if starting or ending particle locations should be
            included in shapefile attribute information. Only used if
            one_per_particle=False. (default is 'ending')
        shpname : str
            File path for shapefile
        mg : flopy.discretization.grid instance
            Used to scale and rotate Global x,y,z values in MODPATH Timeseries
            file.
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
        ParticleTrackFile.write_shapefile(
            self,
            data=data if data is not None else timeseries_data,
            one_per_particle=one_per_particle,
            direction=direction,
            shpname=shpname,
            mg=mg,
            crs=crs,
            **kwargs,
        )
