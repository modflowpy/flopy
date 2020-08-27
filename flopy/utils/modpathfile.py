"""
Module to read MODPATH output files.  The module contains two
important classes that can be accessed by the user.

*  EndpointFile (ascii endpoint file)
*  PathlineFile (ascii pathline file)

"""

import itertools
import collections
import warnings
import numpy as np

from numpy.lib.recfunctions import append_fields, stack_arrays

from ..utils.flopy_io import loadtxt
from ..utils.recarray_utils import ra_slice


class PathlineFile:
    """
    PathlineFile Class.

    Parameters
    ----------
    filename : string
        Name of the pathline file
    verbose : bool
        Write information to the screen.  Default is False.

    Examples
    --------

    >>> import flopy
    >>> pthobj = flopy.utils.PathlineFile('model.mppth')
    >>> p1 = pthobj.get_data(partid=1)

    """

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

    def __init__(self, filename, verbose=False):
        """
        Class constructor.

        """
        self.fname = filename
        self.verbose = verbose

        # build index
        self._build_index()

        # set output dtype
        self.outdtype = self._get_outdtype()

        # set data dtype and read pathline data
        if self.version == 7:
            self.dtype, self._data = self._get_mp7data()
        else:
            self.dtype = self._get_dtypes()
            self._data = loadtxt(
                self.file, dtype=self.dtype, skiprows=self.skiprows
            )

        # convert layer, row, and column indices; particle id and group; and
        #  line segment indices to zero-based
        for n in self.kijnames:
            if n in self._data.dtype.names:
                self._data[n] -= 1

        # set number of particle ids
        self.nid = np.unique(self._data["particleid"])

        # close the input file
        self.file.close()
        return

    def _build_index(self):
        """
        Set position of the start of the pathline data.
        """
        self.skiprows = 0
        self.file = open(self.fname, "r")
        while True:
            line = self.file.readline()
            if isinstance(line, bytes):
                line = line.decode()
            if self.skiprows < 1:
                if "MODPATH_PATHLINE_FILE 6" in line.upper():
                    self.version = 6
                elif "MODPATH_PATHLINE_FILE         7" in line.upper():
                    self.version = 7
                elif "MODPATH 5.0" in line.upper():
                    self.version = 5
                elif "MODPATH Version 3.00" in line.upper():
                    self.version = 3
                else:
                    self.version = None
                if self.version is None:
                    errmsg = "{} is not a valid pathline file".format(
                        self.fname
                    )
                    raise Exception(errmsg)
            self.skiprows += 1
            if self.version == 6 or self.version == 7:
                if "end header" in line.lower():
                    break
            elif self.version == 3 or self.version == 5:
                break
        self.file.seek(0)
        return

    def _get_dtypes(self):
        """
        Build numpy dtype for the MODPATH 6 pathline file.
        """
        if self.version == 3 or self.version == 5:
            dtype = np.dtype(
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
            )
        elif self.version == 6:
            dtype = np.dtype(
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
            )
        elif self.version == 7:
            msg = (
                "_get_dtypes() should not be called for "
                + "MODPATH 7 pathline files"
            )
            raise TypeError(msg)
        return dtype

    def _get_outdtype(self):
        outdtype = np.dtype(
            [
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("time", np.float32),
                ("k", np.int32),
                ("particleid", np.int32),
            ]
        )
        return outdtype

    def _get_mp7data(self):
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
        dtype = np.dtype(
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
        )
        idx = 0
        part_dict = collections.OrderedDict()
        ndata = 0
        while True:
            if idx == 0:
                for n in range(self.skiprows):
                    line = self.file.readline()
            # read header line
            try:
                line = self.file.readline().strip()
                if self.verbose:
                    print(line)
                if len(line) < 1:
                    break
            except:
                break
            t = [int(s) for j, s in enumerate(line.split()) if j < 4]
            sequencenumber, group, particleid, pathlinecount = t[0:4]
            ndata += pathlinecount
            # read in the particle data
            d = np.loadtxt(
                itertools.islice(self.file, 0, pathlinecount), dtype=dtyper
            )
            key = (idx, sequencenumber, group, particleid, pathlinecount)
            part_dict[key] = d.copy()
            idx += 1

        # create data array
        data = np.zeros(ndata, dtype=dtype)

        # fill data
        ipos0 = 0
        for key, value in part_dict.items():
            idx, sequencenumber, group, particleid, pathlinecount = key[0:5]
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

        return dtype, data

    def get_maxid(self):
        """
        Get the maximum pathline number in the file pathline file

        Returns
        ----------
        out : int
            Maximum pathline number.

        """
        return self._data["particleid"].max()

    def get_maxtime(self):
        """
        Get the maximum time in pathline file

        Returns
        ----------
        out : float
            Maximum pathline time.

        """
        return self._data["time"].max()

    def get_data(self, partid=0, totim=None, ge=True):
        """
        get pathline data from the pathline file for a single pathline.

        Parameters
        ----------
        partid : int
            The zero-based particle id.  The first record is record 0.
        totim : float
            The simulation time. All pathline points for particle partid
            that are greater than or equal to (ge=True) or less than or
            equal to (ge=False) totim will be returned. Default is None
        ge : bool
            Boolean that determines if pathline times greater than or equal
            to or less than or equal to totim is used to create a subset
            of pathlines. Default is True.

        Returns
        ----------
        ra : numpy record array
            A numpy recarray with the x, y, z, time, k, and particleid for
            pathline partid.


        See Also
        --------

        Notes
        -----

        Examples
        --------

        >>> import flopy.utils.modpathfile as mpf
        >>> pthobj = flopy.utils.PathlineFile('model.mppth')
        >>> p1 = pthobj.get_data(partid=1)

        """
        # idx = self._data['particleid'] == partid
        if totim is not None:
            if ge:
                idx = (self._data["time"] >= totim) & (
                    self._data["particleid"] == partid
                )
            else:
                idx = (self._data["time"] <= totim) & (
                    self._data["particleid"] == partid
                )
        else:
            idx = self._data["particleid"] == partid
        self._ta = self._data[idx]
        names = ["x", "y", "z", "time", "k", "particleid"]
        return np.rec.fromarrays(
            (self._ta[name] for name in names), dtype=self.outdtype
        )

    def get_alldata(self, totim=None, ge=True):
        """
        get pathline data from the pathline file for all pathlines and all times.

        Parameters
        ----------
        totim : float
            The simulation time. All pathline points for particle partid
            that are greater than or equal to (ge=True) or less than or
            equal to (ge=False) totim will be returned. Default is None
        ge : bool
            Boolean that determines if pathline times greater than or equal
            to or less than or equal to totim is used to create a subset
            of pathlines. Default is True.

        Returns
        ----------
        plist : a list of numpy record array
            A list of numpy recarrays with the x, y, z, time, k, and particleid for
            all pathlines.


        See Also
        --------

        Notes
        -----

        Examples
        --------

        >>> import flopy.utils.modpathfile as mpf
        >>> pthobj = flopy.utils.PathlineFile('model.mppth')
        >>> p = pthobj.get_alldata()

        """
        # plist = []
        # for partid in self.nid:
        #     plist.append(self.get_data(partid=partid, totim=totim, ge=ge))
        return [
            self.get_data(partid=partid, totim=totim, ge=ge)
            for partid in self.nid
        ]

    def get_destination_pathline_data(self, dest_cells, to_recarray=False):
        """
        Get pathline data for set of destination cells.

        Parameters
        ----------
        dest_cells : list or array of tuples
            (k, i, j) of each destination cell (zero-based)
        to_recarray : bool
            Boolean that controls returned pthldest. If to_recarray is True,
            a single recarray with all of the pathlines that intersect
            dest_cells are returned. If to_recarray is False, a list of
            recarrays (the same form as returned by get_alldata method)
            that intersect dest_cells are returned (default is False).

        Returns
        -------
        pthldest : np.recarray
            Slice of pathline data array (e.g. PathlineFile._data)
            containing only pathlines with final k,i,j in dest_cells.

        Examples
        --------

        >>> import flopy
        >>> p = flopy.utils.PathlineFile('modpath.pathline')
        >>> p0 = p.get_destination_pathline_data([(0, 0, 0),
        ...                                       (1, 0, 0)])

        """

        # create local copy of _data
        ra = np.array(self._data)

        # find the intersection of pathlines and dest_cells
        # convert dest_cells to same dtype for comparison
        if self.version < 7:
            try:
                raslice = ra[["k", "i", "j"]]
            except:
                msg = (
                    "could not extract 'k', 'i', and 'j' keys "
                    + "from pathline data"
                )
                raise KeyError(msg)
        else:
            try:
                raslice = ra[["node"]]
            except:
                msg = "could not extract 'node' key from pathline data"
                raise KeyError(msg)
            if isinstance(dest_cells, (list, tuple)):
                allint = all(isinstance(el, int) for el in dest_cells)
                # convert to a list of tuples
                if allint:
                    t = []
                    for el in dest_cells:
                        t.append((el,))
                        dest_cells = t

        dest_cells = np.array(dest_cells, dtype=raslice.dtype)
        inds = np.in1d(raslice, dest_cells)
        epdest = ra[inds].copy().view(np.recarray)

        if to_recarray:
            # use particle ids to get the rest of the paths
            inds = np.in1d(ra["particleid"], epdest.particleid)
            pthldes = ra[inds].copy()
            pthldes.sort(order=["particleid", "time"])
            pthldes = pthldes.view(np.recarray)
        else:

            # get list of unique particleids in selection
            partids = np.unique(epdest["particleid"])

            # build list of unique particleids in selection
            pthldes = [self.get_data(partid) for partid in partids]

        return pthldes

    def write_shapefile(
        self,
        pathline_data=None,
        one_per_particle=True,
        direction="ending",
        shpname="endpoints.shp",
        mg=None,
        epsg=None,
        sr=None,
        **kwargs
    ):
        """
        Write pathlines to a shapefile

        pathline_data : np.recarray
            Record array of same form as that returned by
            EndpointFile.get_alldata(). (if none, EndpointFile.get_alldata()
            is exported).
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
            Used to scale and rotate Global x,y,z values in MODPATH Endpoint
            file.
        epsg : int
            EPSG code for writing projection (.prj) file. If this is not
            supplied, the proj4 string or epgs code associated with mg will be
            used.
        kwargs : keyword arguments to flopy.export.shapefile_utils.recarray2shp

        """
        from ..utils.reference import SpatialReference
        from ..utils import geometry
        from ..discretization import StructuredGrid
        from ..utils.geometry import LineString
        from ..export.shapefile_utils import recarray2shp

        pth = pathline_data
        if pth is None:
            pth = self._data.view(np.recarray)
        else:
            # convert pathline list to a single recarray
            if isinstance(pth, list):
                s = pth[0]
                print(s.dtype)
                for n in range(1, len(pth)):
                    s = stack_arrays((s, pth[n]))
                pth = s.view(np.recarray)

        pth = pth.copy()
        pth.sort(order=["particleid", "time"])

        if isinstance(mg, SpatialReference) or isinstance(
            sr, SpatialReference
        ):
            warnings.warn(
                "Deprecation warning: SpatialReference is deprecated."
                "Use the Grid class instead.",
                DeprecationWarning,
            )
            if isinstance(mg, SpatialReference):
                sr = mg
            mg = StructuredGrid(sr.delc, sr.delr)
            mg.set_coord_info(
                xoff=sr.xll,
                yoff=sr.yll,
                angrot=sr.rotation,
                epsg=sr.epsg,
                proj4=sr.proj4_str,
            )

        if epsg is None:
            epsg = mg.epsg

        particles = np.unique(pth.particleid)
        geoms = []

        # create dtype with select attributes in pth
        names = pth.dtype.names
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

            pthdata = []
            for pid in particles:
                ra = pth[pth.particleid == pid]

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
                pthdata.append(tuple(t))

            pthdata = np.array(pthdata, dtype=dtype).view(np.recarray)

        # geometry for each row in PathLine file
        else:
            dtype = pth.dtype
            # pthdata = np.empty((0, len(dtype)), dtype=dtype).view(np.recarray)
            pthdata = []
            for pid in particles:
                ra = pth[pth.particleid == pid]
                if isinstance(mg, StructuredGrid):
                    x, y = geometry.transform(
                        ra.x, ra.y, mg.xoffset, mg.yoffset, mg.angrot_radians
                    )
                else:
                    x, y = mg.transform(ra.x, ra.y)
                z = ra.z
                geoms += [
                    LineString(
                        [(x[i - 1], y[i - 1], z[i - 1]), (x[i], y[i], z[i])]
                    )
                    for i in np.arange(1, (len(ra)))
                ]
                # pthdata = np.append(pthdata, ra[1:]).view(np.recarray)
                pthdata += ra[1:].tolist()
            pthdata = np.array(pthdata, dtype=dtype).view(np.recarray)

        # convert back to one-based
        for n in set(self.kijnames).intersection(set(pthdata.dtype.names)):
            pthdata[n] += 1

        # write the final recarray to a shapefile
        recarray2shp(pthdata, geoms, shpname=shpname, epsg=epsg, **kwargs)


class EndpointFile:
    """
    EndpointFile Class.

    Parameters
    ----------
    filename : string
        Name of the endpoint file
    verbose : bool
        Write information to the screen.  Default is False.

    Examples
    --------

    >>> import flopy
    >>> endobj = flopy.utils.EndpointFile('model.mpend')
    >>> e1 = endobj.get_data(partid=1)


    """

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

    def __init__(self, filename, verbose=False):
        """
        Class constructor.

        """
        self.fname = filename
        self.verbose = verbose
        self._build_index()
        self.dtype = self._get_dtypes()
        self._data = loadtxt(
            self.file, dtype=self.dtype, skiprows=self.skiprows
        )
        # add particleid if required
        self._add_particleid()

        # convert layer, row, and column indices; particle id and group; and
        #  line segment indices to zero-based
        for n in self.kijnames:
            if n in self._data.dtype.names:
                self._data[n] -= 1

        # set number of particle ids
        self.nid = np.unique(self._data["particleid"]).shape[0]

        # close the input file
        self.file.close()
        return

    def _build_index(self):
        """
        Set position of the start of the pathline data.
        """
        self.skiprows = 0
        self.file = open(self.fname, "r")
        idx = 0
        while True:
            line = self.file.readline()
            if isinstance(line, bytes):
                line = line.decode()
            if self.skiprows < 1:
                if "MODPATH_ENDPOINT_FILE 6" in line.upper():
                    self.version = 6
                elif "MODPATH_ENDPOINT_FILE         7" in line.upper():
                    self.version = 7
                elif "MODPATH 5.0" in line.upper():
                    self.version = 5
                elif "MODPATH Version 3.00" in line.upper():
                    self.version = 3
                else:
                    self.version = None
                if self.version is None:
                    errmsg = "{} is not a valid endpoint file".format(
                        self.fname
                    )
                    raise Exception(errmsg)
            self.skiprows += 1
            if self.version == 6 or self.version == 7:
                if idx == 1:
                    t = line.strip()
                    self.direction = 1
                    if int(t[0]) == 2:
                        self.direction = -1
                idx += 1
                if "end header" in line.lower():
                    break
            else:
                break
        self.file.seek(0)

        if self.verbose:
            print("MODPATH version {} endpoint file".format(self.version))

    def _get_dtypes(self):
        """
        Build numpy dtype for the MODPATH 6 endpoint file.
        """
        if self.version == 3 or self.version == 5:
            dtype = self._get_mp35_dtype()
        elif self.version == 6:
            dtype = self._get_mp6_dtype()
        elif self.version == 7:
            dtype = self._get_mp7_dtype()
        return dtype

    def _get_mp35_dtype(self, add_id=False):
        dtype = [
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
        if add_id:
            dtype.insert(0, ("particleid", np.int32))
        return np.dtype(dtype)

    def _get_mp6_dtype(self):
        dtype = [
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
        return np.dtype(dtype)

    def _get_mp7_dtype(self):
        dtype = [
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
        return np.dtype(dtype)

    def _add_particleid(self):

        # add particle ids for earlier version of MODPATH
        if self.version < 6:
            # create particle ids
            shaped = self._data.shape[0]
            pids = np.arange(1, shaped + 1, 1, dtype=np.int32)

            # determine numpy version
            npv = np.__version__
            v = [int(s) for s in npv.split(".")]
            if self.verbose:
                print("numpy version {}".format(npv))

            # for numpy version 1.14 and higher
            if v[0] > 1 or (v[0] == 1 and v[1] > 13):
                self._data = append_fields(self._data, "particleid", pids)
            # numpy versions prior to 1.14
            else:
                if self.verbose:
                    print(self._data.dtype)

                # convert pids to structured array
                pids = np.array(
                    pids, dtype=np.dtype([("particleid", np.int32)])
                )

                # create new dtype
                dtype = self._get_mp35_dtype(add_id=True)
                if self.verbose:
                    print(dtype)

                # create new array with new dtype and fill with available data
                data = np.zeros(shaped, dtype=dtype)
                if self.verbose:
                    print("new data shape {}".format(data.shape))
                    print("\nFilling new structured data array")

                # add particle id to new array
                if self.verbose:
                    msg = (
                        "writing particleid (pids) to new "
                        + "structured data array"
                    )
                    print(msg)
                data["particleid"] = pids["particleid"]

                # add remaining data to the new array
                if self.verbose:
                    msg = (
                        "writing remaining data to new "
                        + "structured data array"
                    )
                    print(msg)
                for name in self._data.dtype.names:
                    data[name] = self._data[name]
                if self.verbose:
                    print("replacing data with copy of new data array")
                self._data = data.copy()
        return

    def get_maxid(self):
        """
        Get the maximum endpoint particle id in the file endpoint file

        Returns
        ----------
        out : int
            Maximum endpoint particle id.

        """
        return np.unique(self._data["particleid"]).max()

    def get_maxtime(self):
        """
        Get the maximum time in the endpoint file

        Returns
        ----------
        out : float
            Maximum endpoint time.

        """
        return self._data["time"].max()

    def get_maxtraveltime(self):
        """
        Get the maximum travel time in the endpoint file

        Returns
        ----------
        out : float
            Maximum endpoint travel time.

        """
        return (self._data["time"] - self._data["time0"]).max()

    def get_data(self, partid=0):
        """
        Get endpoint data from the endpoint file for a single particle.

        Parameters
        ----------
        partid : int
            The zero-based particle id.  The first record is record 0.
            (default is 0)

        Returns
        ----------
        ra : numpy record array
            A numpy recarray with the endpoint particle data for
            endpoint partid.


        See Also
        --------

        Notes
        -----

        Examples
        --------

        >>> import flopy
        >>> endobj = flopy.utils.EndpointFile('model.mpend')
        >>> e1 = endobj.get_data(partid=1)

        """
        idx = self._data["particleid"] == partid
        ra = self._data[idx]
        return ra

    def get_alldata(self):
        """
        Get endpoint data from the endpoint file for all endpoints.

        Parameters
        ----------

        Returns
        ----------
        ra : numpy record array
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
        Get endpoint data for set of destination cells.

        Parameters
        ----------
        dest_cells : list or array of tuples
            (k, i, j) or (node,) of each destination cell (zero-based)
        source : bool
            Boolean to specify is dest_cells applies to source or
            destination cells (default is False).

        Returns
        -------
        epdest : np.recarray
            Slice of endpoint data array (e.g. EndpointFile.get_alldata)
            containing only data with final k,i,j in dest_cells.

        Examples
        --------

        >>> import flopy
        >>> e = flopy.utils.EndpointFile('modpath.endpoint')
        >>> e0 = e.get_destination_endpoint_data([(0, 0, 0),
        ...                                       (1, 0, 0)])

        """

        # create local copy of _data
        ra = self.get_alldata()

        # find the intersection of endpoints and dest_cells
        # convert dest_cells to same dtype for comparison
        if self.version < 7:
            if source:
                keys = ["k0", "i0", "j0"]
            else:
                keys = ["k", "i", "j"]
            try:
                raslice = ra_slice(ra, keys)
            except:
                msg = (
                    "could not extract"
                    + "'"
                    + "', '".join(keys)
                    + "'"
                    + "from endpoint data."
                )
                raise KeyError(msg)
        else:
            if source:
                keys = ["node0"]
            else:
                keys = ["node"]
            try:
                raslice = ra_slice(ra, keys)
            except:
                msg = (
                    "could not extract '{}' ".format(keys[0])
                    + "key from endpoint data"
                )
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
        epdest = ra[inds].copy().view(np.recarray)
        return epdest

    def write_shapefile(
        self,
        endpoint_data=None,
        shpname="endpoints.shp",
        direction="ending",
        mg=None,
        epsg=None,
        sr=None,
        **kwargs
    ):
        """
        Write particle starting / ending locations to shapefile.

        endpoint_data : np.recarray
            Record array of same form as that returned by EndpointFile.get_alldata.
            (if none, EndpointFile.get_alldata() is exported).
        shpname : str
            File path for shapefile
        direction : str
            String defining if starting or ending particle locations should be
            considered. (default is 'ending')
        sr : flopy.utils.reference.SpatialReference instance
            Used to scale and rotate Global x,y,z values in MODPATH Endpoint file
        epsg : int
            EPSG code for writing projection (.prj) file. If this is not supplied,
            the proj4 string or epgs code associated with sr will be used.
        kwargs : keyword arguments to flopy.export.shapefile_utils.recarray2shp

        """
        from ..utils.reference import SpatialReference
        from ..utils import geometry
        from ..discretization import StructuredGrid
        from ..utils.geometry import Point
        from ..export.shapefile_utils import recarray2shp

        epd = endpoint_data.copy()
        if epd is None:
            epd = self.get_alldata()

        if direction.lower() == "ending":
            xcol, ycol, zcol = "x", "y", "z"
        elif direction.lower() == "starting":
            xcol, ycol, zcol = "x0", "y0", "z0"
        else:
            errmsg = (
                'flopy.map.plot_endpoint direction must be "ending" '
                + 'or "starting".'
            )
            raise Exception(errmsg)
        if isinstance(mg, SpatialReference) or isinstance(
            sr, SpatialReference
        ):
            warnings.warn(
                "Deprecation warning: SpatialReference is deprecated."
                "Use the Grid class instead.",
                DeprecationWarning,
            )
            if isinstance(mg, SpatialReference):
                sr = mg
            mg = StructuredGrid(sr.delc, sr.delr)
            mg.set_coord_info(
                xoff=sr.xll,
                yoff=sr.yll,
                angrot=sr.rotation,
                epsg=sr.epsg,
                proj4=sr.proj4_str,
            )
        if epsg is None:
            epsg = mg.epsg

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
        recarray2shp(epd, geoms, shpname=shpname, epsg=epsg, **kwargs)


class TimeseriesFile:
    """
    TimeseriesFile Class.

    Parameters
    ----------
    filename : string
        Name of the timeseries file
    verbose : bool
        Write information to the screen.  Default is False.

    Examples
    --------

    >>> import flopy
    >>> tsobj = flopy.utils.TimeseriesFile('model.timeseries')
    >>> ts1 = tsobj.get_data(partid=1)
    """

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
        """
        Class constructor.

        """
        self.fname = filename
        self.verbose = verbose

        # build index
        self._build_index()

        # set output dtype
        self.outdtype = self._get_outdtype()

        # set dtype
        self.dtype = self._get_dtypes()

        # read data
        self._data = loadtxt(
            self.file, dtype=self.dtype, skiprows=self.skiprows
        )

        # convert layer, row, and column indices; particle id and group; and
        #  line segment indices to zero-based
        for n in self.kijnames:
            if n in self._data.dtype.names:
                self._data[n] -= 1

        # set number of particle ids
        self.nid = np.unique(self._data["particleid"])

        # close the input file
        self.file.close()
        return

    def _build_index(self):
        """
        Set position of the start of the timeseries data.
        """
        compact = False
        self.skiprows = 0
        self.file = open(self.fname, "r")
        while True:
            line = self.file.readline()
            if isinstance(line, bytes):
                line = line.decode()
            if self.skiprows < 1:
                if "MODPATH_TIMESERIES_FILE 6" in line.upper():
                    self.version = 6
                elif "MODPATH_TIMESERIES_FILE         7" in line.upper():
                    self.version = 7
                elif "MODPATH 5.0" in line.upper():
                    self.version = 5
                    if "COMPACT" in line.upper():
                        compact = True
                elif "MODPATH Version 3.00" in line.upper():
                    self.version = 3
                else:
                    self.version = None
                if self.version is None:
                    errmsg = (
                        "{} ".format(self.fname)
                        + "is not a valid timeseries file"
                    )
                    raise Exception(errmsg)
            self.skiprows += 1
            if self.version == 6 or self.version == 7:
                if "end header" in line.lower():
                    break
            elif self.version == 3 or self.version == 5:
                break

        # set compact
        self.compact = compact

        # return to the top of the file
        self.file.seek(0)

    def _get_dtypes(self):
        """
        Build numpy dtype for the MODPATH 6 timeseries file.
        """
        if self.version == 3 or self.version == 5:
            if self.compact:
                dtype = np.dtype(
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
                )
            else:
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
        elif self.version == 6:
            dtype = np.dtype(
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
            )
        elif self.version == 7:
            dtype = np.dtype(
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
            )
        return dtype

    def _get_outdtype(self):
        outdtype = np.dtype(
            [
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("time", np.float32),
                ("k", np.int32),
                ("particleid", np.int32),
            ]
        )
        return outdtype

    def get_maxid(self):
        """
        Get the maximum timeseries number in the file timeseries file

        Returns
        ----------
        out : int
            Maximum pathline number.

        """
        return self._data["particleid"].max()

    def get_maxtime(self):
        """
        Get the maximum time in timeseries file

        Returns
        ----------
        out : float
            Maximum pathline time.

        """
        return self._data["time"].max()

    def get_data(self, partid=0, totim=None, ge=True):
        """
        get timeseries data from the timeseries file for a single timeseries
        particleid.

        Parameters
        ----------
        partid : int
            The zero-based particle id.  The first record is record 0.
        totim : float
            The simulation time. All timeseries points for particle partid
            that are greater than or equal to (ge=True) or less than or
            equal to (ge=False) totim will be returned. Default is None
        ge : bool
            Boolean that determines if timeseries times greater than or equal
            to or less than or equal to totim is used to create a subset
            of timeseries. Default is True.

        Returns
        ----------
        ra : numpy record array
            A numpy recarray with the x, y, z, time, k, and particleid for
            timeseries partid.


        See Also
        --------

        Notes
        -----

        Examples
        --------

        >>> import flopy
        >>> tsobj = flopy.utils.TimeseriesFile('model.timeseries')
        >>> ts1 = tsobj.get_data(partid=1)

        """
        if totim is not None:
            if ge:
                idx = (self._data["time"] >= totim) & (
                    self._data["particleid"] == partid
                )
            else:
                idx = (self._data["time"] <= totim) & (
                    self._data["particleid"] == partid
                )
        else:
            idx = self._data["particleid"] == partid
        self._ta = self._data[idx]
        names = ["x", "y", "z", "time", "k", "particleid"]
        return np.rec.fromarrays(
            (self._ta[name] for name in names), dtype=self.outdtype
        )

    def get_alldata(self, totim=None, ge=True):
        """
        get timeseries data from the timeseries file for all timeseries
        and all times.

        Parameters
        ----------
        totim : float
            The simulation time. All timeseries points for timeseries partid
            that are greater than or equal to (ge=True) or less than or
            equal to (ge=False) totim will be returned. Default is None
        ge : bool
            Boolean that determines if timeseries times greater than or equal
            to or less than or equal to totim is used to create a subset
            of timeseries. Default is True.

        Returns
        ----------
        tlist : a list of numpy record array
            A list of numpy recarrays with the x, y, z, time, k, and
            particleid for all timeseries.


        See Also
        --------

        Notes
        -----

        Examples
        --------

        >>> import flopy
        >>> tsobj = flopy.utils.TimeseriesFile('model.timeseries')
        >>> ts = tsobj.get_alldata()

        """
        return [
            self.get_data(partid=partid, totim=totim, ge=ge)
            for partid in self.nid
        ]

    def get_destination_timeseries_data(self, dest_cells):
        """
        Get timeseries data for set of destination cells.

        Parameters
        ----------
        dest_cells : list or array of tuples
            (k, i, j) or nodes of each destination cell (zero-based)

        Returns
        -------
        tsdest : np.recarray
            Slice of timeseries data array (e.g. TmeseriesFile._data)
            containing only pathlines with final k,i,j in dest_cells.

        Examples
        --------

        >>> import flopy
        >>> ts = flopy.utils.TimeseriesFile('modpath.timeseries')
        >>> tss = ts.get_destination_timeseries_data([(0, 0, 0),
        ...                                           (1, 0, 0)])

        """

        # create local copy of _data
        ra = np.array(self._data)

        # find the intersection of timeseries and dest_cells
        # convert dest_cells to same dtype for comparison
        if self.version < 7:
            try:
                raslice = ra[["k", "i", "j"]]
            except:
                msg = (
                    "could not extract 'k', 'i', and 'j' keys "
                    + "from timeseries data"
                )
                raise KeyError(msg)
        else:
            try:
                raslice = ra[["node"]]
            except:
                msg = "could not extract 'node' key from timeseries data"
                raise KeyError(msg)
            if isinstance(dest_cells, (list, tuple)):
                allint = all(isinstance(el, int) for el in dest_cells)
                # convert to a list of tuples
                if allint:
                    t = []
                    for el in dest_cells:
                        t.append((el,))
                        dest_cells = t

        dest_cells = np.array(dest_cells, dtype=raslice.dtype)
        inds = np.in1d(raslice, dest_cells)
        epdest = ra[inds].copy().view(np.recarray)

        # use particle ids to get the rest of the timeseries
        inds = np.in1d(ra["particleid"], epdest.particleid)
        tsdes = ra[inds].copy()
        tsdes.sort(order=["particleid", "time"])
        return tsdes.view(np.recarray)
