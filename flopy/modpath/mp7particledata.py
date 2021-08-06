"""
mp7particledata module. Contains the ParticleData, CellDataType,
    FaceDataType, and NodeParticleData classes.


"""

import os
import numpy as np
from ..utils.util_array import Util2d
from ..utils.recarray_utils import create_empty_recarray


class ParticleData:
    """
    Class to create the most basic particle data type (starting location
    input style 1). Input style 1 is the most general input style and
    provides the most flexibility in customizing starting locations.

    Parameters
    ----------
    partlocs : list/tuple of int, list/tuple of list/tuple, or np.ndarray
        Particle locations (zero-based) that are either layer, row, column
        locations or nodes.
    structured : bool
        Boolean defining if a structured (True) or unstructured
        particle recarray will be created (default is True).
    particleids : list, tuple, or np.ndarray
        Particle ids for the defined particle locations. If particleids
        is None, MODPATH 7 will define the particle ids to each particle
        location. If particleids is provided a particle
        id must be provided for each partloc (default is None).
    localx : float, list, tuple, or np.ndarray
        Local x-location of the particle in the cell. If a single value is
        provided all particles will have the same localx position. If
        a list, tuple, or np.ndarray is provided a localx position must
        be provided for each partloc. If localx is None, a value of
        0.5 (center of the cell) will be used (default is None).
    localy : float, list, tuple, or np.ndarray
        Local y-location of the particle in the cell. If a single value is
        provided all particles will have the same localy position. If
        a list, tuple, or np.ndarray is provided a localy position must
        be provided for each partloc. If localy is None, a value of
        0.5 (center of the cell) will be used (default is None).
    localz : float, list, tuple, or np.ndarray
        Local z-location of the particle in the cell. If a single value is
        provided all particles will have the same localz position. If
        a list, tuple, or np.ndarray is provided a localz position must
        be provided for each partloc. If localy is None, a value of
        0.5 (center of the cell) will be used (default is None).
    timeoffset : float, list, tuple, or np.ndarray
        Timeoffset of the particle relative to the release time. If a
        single value is provided all particles will have the same
        timeoffset. If a list, tuple, or np.ndarray is provided a
        timeoffset must be provided for each partloc. If timeoffset is
        None, a value of 0. (equal to the release time) will be used
        (default is None).
    drape : int, list, tuple, or np.ndarray
        Drape indicates how particles are treated when starting locations
        are specified for cells that are dry. If drape is 0, Particles are
        placed in the specified cell. If the cell is dry at the time of
        release, the status of the particle is set to unreleased and
        removed from the simulation. If drape is 1, particles are placed
        in the upper most active grid cell directly beneath the specified
        layer, row, column or node location. If a single value is provided
        all particles will have the same drape value. If a list, tuple, or
        np.ndarray is provided a drape value must be provided for each
        partloc. If drape is None, a value of 0 will be used (default
        is None).


    Examples
    --------

    >>> import flopy
    >>> locs = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
    >>> pd = flopy.modpath.ParticleData(locs, structured=True, drape=0,
    ...                                 localx=0.5, localy=0.5, localz=1)


    """

    def __init__(
        self,
        partlocs=None,
        structured=False,
        particleids=None,
        localx=None,
        localy=None,
        localz=None,
        timeoffset=None,
        drape=None,
    ):
        """
        Class constructor

        """
        self.name = "ParticleData"

        if structured:
            locationstyle = 1
        else:
            locationstyle = 2

        if partlocs is None:
            if structured:
                partlocs = [(0, 0, 0)]
            else:
                partlocs = [(0,)]

        # create dtype
        dtype = []
        if structured:
            dtype.append(("k", np.int32))
            dtype.append(("i", np.int32))
            dtype.append(("j", np.int32))
        else:
            dtype.append(("node", np.int32))
        dtype = np.dtype(dtype)

        if isinstance(partlocs, (list, tuple)):
            # determine if the list or tuple contains lists or tuples
            alllsttup = all(isinstance(el, (list, tuple)) for el in partlocs)
            if structured:
                if alllsttup:
                    alllen3 = all(len(el) == 3 for el in partlocs)
                    if not alllen3:
                        raise ValueError(
                            "{}: all partlocs entries  must have 3 items for "
                            "structured particle data".format(self.name)
                        )
                else:
                    raise ValueError(
                        "{}: partlocs list or tuple "
                        "for structured particle data should "
                        "contain list or tuple entries".format(self.name)
                    )
            else:
                allint = all(
                    isinstance(el, (int, np.int32, np.int64))
                    for el in partlocs
                )
                # convert to a list of tuples
                if allint:
                    t = []
                    for el in partlocs:
                        t.append((el,))
                    partlocs = t
                    alllsttup = all(
                        isinstance(el, (list, tuple)) for el in partlocs
                    )
                if alllsttup:
                    alllen1 = all(len(el) == 1 for el in partlocs)
                    if not alllen1:
                        raise ValueError(
                            "{}: all entries of partlocs must have 1 items "
                            "for unstructured particle data".format(self.name)
                        )
                else:
                    raise ValueError(
                        "{}: partlocs list or tuple for unstructured particle "
                        "data should contain integers or a list or tuple with "
                        "one entry".format(self.name)
                    )

            # convert partlocs composed of a lists/tuples of lists/tuples
            # to a numpy array
            partlocs = np.array(partlocs, dtype=dtype)
        elif isinstance(partlocs, np.ndarray):
            dtypein = partlocs.dtype
            if dtypein != dtype:
                partlocs = np.array(partlocs, dtype=dtype)
        else:
            raise ValueError(
                "{}: partlocs must be a list or "
                "tuple with lists or tuples".format(self.name)
            )

        # localx
        if localx is None:
            localx = 0.5
        else:
            if isinstance(localx, (float, int)):
                localx = np.ones(partlocs.shape[0], dtype=np.float32) * localx
            elif isinstance(localx, (list, tuple)):
                localx = np.array(localx, dtype=np.float32)
            if isinstance(localx, np.ndarray):
                if localx.shape[0] != partlocs.shape[0]:
                    raise ValueError(
                        "{}:shape of localx ({}) is not equal to the shape "
                        "of partlocs ({}).".format(
                            self.name, localx.shape[0], partlocs.shape[0]
                        )
                    )

        # localy
        if localy is None:
            localy = 0.5
        else:
            if isinstance(localy, (float, int)):
                localy = np.ones(partlocs.shape[0], dtype=np.float32) * localy
            elif isinstance(localy, (list, tuple)):
                localy = np.array(localy, dtype=np.float32)
            if isinstance(localy, np.ndarray):
                if localy.shape[0] != partlocs.shape[0]:
                    raise ValueError(
                        "{}:shape of localy ({}) is not equal to the shape "
                        "of partlocs ({}).".format(
                            self.name, localy.shape[0], partlocs.shape[0]
                        )
                    )

        # localz
        if localz is None:
            localz = 0.5
        else:
            if isinstance(localz, (float, int)):
                localz = np.ones(partlocs.shape[0], dtype=np.float32) * localz
            elif isinstance(localz, (list, tuple)):
                localz = np.array(localz, dtype=np.float32)
            if isinstance(localz, np.ndarray):
                if localz.shape[0] != partlocs.shape[0]:
                    raise ValueError(
                        "{}:shape of localz ({}) is not equal to the shape "
                        "of partlocs ({}).".format(
                            self.name, localz.shape[0], partlocs.shape[0]
                        )
                    )
        # timeoffset
        if timeoffset is None:
            timeoffset = 0.0
        else:
            if isinstance(timeoffset, (float, int)):
                timeoffset = (
                    np.ones(partlocs.shape[0], dtype=np.float32) * timeoffset
                )
            elif isinstance(timeoffset, (list, tuple)):
                timeoffset = np.array(timeoffset, dtype=np.float32)
            if isinstance(timeoffset, np.ndarray):
                if timeoffset.shape[0] != partlocs.shape[0]:
                    raise ValueError(
                        "{}:shape of timeoffset ({}) is not equal to the "
                        "shape of partlocs ({}).".format(
                            self.name, timeoffset.shape[0], partlocs.shape[0]
                        )
                    )

        # drape
        if drape is None:
            drape = 0
        else:
            if isinstance(drape, (float, int)):
                drape = np.ones(partlocs.shape[0], dtype=np.float32) * drape
            elif isinstance(drape, (list, tuple)):
                drape = np.array(drape, dtype=np.int32)
            if isinstance(drape, np.ndarray):
                if drape.shape[0] != partlocs.shape[0]:
                    raise ValueError(
                        "{}:shape of drape ({}) is not equal to the shape "
                        "of partlocs ({}).".format(
                            self.name, drape.shape[0], partlocs.shape[0]
                        )
                    )

        # particleids
        if particleids is None:
            particleid = False
            particleidoption = 0
        else:
            particleid = True
            particleidoption = 1
            if isinstance(particleids, (int, float)):
                raise TypeError(
                    "{}:A particleid must be provided for each partloc "
                    "as a list/tuple/np.ndarray of size {}. "
                    "A single particleid has been provided.".format(
                        self.name, partlocs.shape[0]
                    )
                )
            elif isinstance(particleids, (list, tuple)):
                particleids = np.array(particleids, dtype=np.int32)
            if isinstance(particleids, np.ndarray):
                if particleids.shape[0] != partlocs.shape[0]:
                    raise ValueError(
                        "{}:shape of particleids ({}) is not equal to the "
                        "shape of partlocs ({}).".format(
                            self.name, particleids.shape[0], partlocs.shape[0]
                        )
                    )

        # create empty particle
        ncells = partlocs.shape[0]
        self.dtype = self._get_dtype(structured, particleid)
        particledata = create_empty_recarray(
            ncells, self.dtype, default_value=0
        )

        # fill particle
        if structured:
            particledata["k"] = partlocs["k"]
            particledata["i"] = partlocs["i"]
            particledata["j"] = partlocs["j"]
        else:
            particledata["node"] = partlocs["node"]
        particledata["localx"] = localx
        particledata["localy"] = localy
        particledata["localz"] = localz
        particledata["timeoffset"] = timeoffset
        particledata["drape"] = drape
        if particleid:
            particledata["id"] = particleids

        self.particlecount = particledata.shape[0]
        self.particleidoption = particleidoption
        self.locationstyle = locationstyle
        self.particledata = particledata

        return

    def write(self, f=None):
        """

        Parameters
        ----------
        f : fileobject
            Fileobject that is open with write access

        Returns
        -------

        """
        # validate that a valid file object was passed
        if not hasattr(f, "write"):
            raise ValueError(
                "{}: cannot write data for template without passing a valid "
                "file object ({}) open for writing".format(self.name, f)
            )

        # particle data item 4 and 5
        d = np.recarray.copy(self.particledata)
        lnames = [name.lower() for name in d.dtype.names]
        # Add one to the kij and node indices
        for idx in (
            "k",
            "i",
            "j",
            "node",
        ):
            if idx in lnames:
                d[idx] += 1
        # Add one to the particle id if required
        if self.particleidoption == 0 and "id" in lnames:
            d["id"] += 1

        # write the particle data
        fmt = self._fmt_string + "\n"
        for v in d:
            f.write(fmt.format(*v))

        return

    def _get_dtype(self, structured, particleid):
        """
        define the dtype for a structured or unstructured
        particledata recarray. Optionally, include a particleid column in
        the dtype.


        Parameters
        ----------
        structured : bool
            Boolean defining if a structured (True) or unstructured
            particle dtype will be created.
        particleid : bool
            Boolean defining if the dtype will include a particle id
            column.

        Returns
        -------
        dtype : numpy dtype

        Examples
        --------

        >>> import flopy.modpath as fmp
        >>> dtype = fmp.ParticleGroup.get_particledata_dtype(structured=True,
        ...                                                  particleid=True)

        """
        dtype = []
        if particleid:
            dtype.append(("id", np.int32))
        if structured:
            dtype.append(("k", np.int32))
            dtype.append(("i", np.int32))
            dtype.append(("j", np.int32))
        else:
            dtype.append(("node", np.int32))
        dtype.append(("localx", np.float32))
        dtype.append(("localy", np.float32))
        dtype.append(("localz", np.float32))
        dtype.append(("timeoffset", np.float32))
        dtype.append(("drape", np.int32))
        return np.dtype(dtype)

    @property
    def _fmt_string(self):
        """
        Returns a python-style fmt string to write particle data
        that corresponds to the dtype

        Parameters
        ----------

        Returns
        -------
        fmt : str
            python format string with space delimited entries


        """
        fmts = []
        for field in self.particledata.dtype.descr:
            vtype = field[1][1].lower()
            if vtype == "i" or vtype == "b":
                fmts.append("{:9d}")
            elif vtype == "f":
                if field[1][2] == 8:
                    fmts.append("{:23.16g}")
                else:
                    fmts.append("{:15.7g}")
            elif vtype == "o":
                fmts.append("{:9s}")
            elif vtype == "s":
                raise TypeError(
                    "Particles.fmt_string error: 'str' type found in dtype. "
                    "This gives unpredictable results when recarray to file - "
                    "change to 'object' type"
                )
            else:
                raise TypeError(
                    "MfList.fmt_string error: unknown vtype in "
                    "field: {}".format(field)
                )
        return " " + " ".join(fmts)


class FaceDataType:
    """
    Face data type class to create a MODPATH 7 particle location template for
    input style 2, 3, and 4 on cell faces (templatesubdivisiontype = 2).

    Parameters
    ----------
    drape : int
        Drape indicates how particles are treated when starting locations
        are specified for cells that are dry. If drape is 0, Particles are
        placed in the specified cell. If the cell is dry at the time of
        release, the status of the particle is set to unreleased and
        removed from the simulation. If drape is 1, particles are placed
        in the upper most active grid cell directly beneath the specified
        layer, row, column or node location (default is 0).
    verticaldivisions1 : int
        The number of vertical subdivisions that define the two-dimensional
        array of particles on cell face 1 (default is 3).
    horizontaldivisions1 : int
        The number of horizontal subdivisions that define the two-dimensional
        array of particles on cell face 1 (default is 3).
    verticaldivisions2 : int
        The number of vertical subdivisions that define the two-dimensional
        array of particles on cell face 2 (default is 3).
    horizontaldivisions2 : int
        The number of horizontal subdivisions that define the two-dimensional
        array of particles on cell face 2 (default is 3).
    verticaldivisions3 : int
        The number of vertical subdivisions that define the two-dimensional
        array of particles on cell face 3 (default is 3).
    horizontaldivisions3 : int
        The number of horizontal subdivisions that define the two-dimensional
        array of particles on cell face 3 (default is 3).
    verticaldivisions4 : int
        The number of vertical subdivisions that define the two-dimensional
        array of particles on cell face 4 (default is 3).
    horizontaldivisions4 : int
        The number of horizontal subdivisions that define the two-dimensional
        array of particles on cell face 4 (default is 3).
    rowdivisions5 : int
        The number of row subdivisions that define the two-dimensional array
        of particles on the bottom cell face (face 5) (default is 3).
    columndivisions5 : int
        The number of column subdivisions that define the two-dimensional array
        of particles on the bottom cell face (face 5) (default is 3).
    rowdivisions6 : int
        The number of row subdivisions that define the two-dimensional array
        of particles on the top cell face (face 6) (default is 3).
    columndivisions6 : int
        The number of column subdivisions that define the two-dimensional array
        of particles on the top cell face (face 6) (default is 3).

    Examples
    --------

    >>> import flopy
    >>> fd = flopy.modpath.FaceDataType()

    """

    def __init__(
        self,
        drape=0,
        verticaldivisions1=3,
        horizontaldivisions1=3,
        verticaldivisions2=3,
        horizontaldivisions2=3,
        verticaldivisions3=3,
        horizontaldivisions3=3,
        verticaldivisions4=3,
        horizontaldivisions4=3,
        rowdivisions5=3,
        columndivisions5=3,
        rowdivisions6=3,
        columndivisions6=3,
    ):
        """
        Class constructor

        """
        self.name = "FaceDataType"

        # assign attributes
        self.templatesubdivisiontype = 1
        self.drape = drape
        self.verticaldivisions1 = verticaldivisions1
        self.horizontaldivisions1 = horizontaldivisions1
        self.verticaldivisions2 = verticaldivisions2
        self.horizontaldivisions2 = horizontaldivisions2
        self.verticaldivisions3 = verticaldivisions3
        self.horizontaldivisions3 = horizontaldivisions3
        self.verticaldivisions4 = verticaldivisions4
        self.horizontaldivisions4 = horizontaldivisions4
        self.rowdivisions5 = rowdivisions5
        self.columndivisions5 = columndivisions5
        self.rowdivisions6 = rowdivisions6
        self.columndivisions6 = columndivisions6
        return

    def write(self, f=None):
        """

        Parameters
        ----------
        f : fileobject
            Fileobject that is open with write access

        Returns
        -------

        """
        # validate that a valid file object was passed
        if not hasattr(f, "write"):
            raise ValueError(
                "{}: cannot write data for template "
                "without passing a valid file object ({}) "
                "open for writing".format(self.name, f)
            )

        # item 4
        fmt = 12 * " {}" + "\n"
        line = fmt.format(
            self.verticaldivisions1,
            self.horizontaldivisions1,
            self.verticaldivisions2,
            self.horizontaldivisions2,
            self.verticaldivisions3,
            self.horizontaldivisions3,
            self.verticaldivisions4,
            self.horizontaldivisions4,
            self.rowdivisions5,
            self.columndivisions5,
            self.rowdivisions6,
            self.columndivisions6,
        )
        f.write(line)

        return


class CellDataType:
    """
    Cell data type class to create a MODPATH 7 particle location template for
    input style 2, 3, and 4 in cells (templatesubdivisiontype = 2).

    Parameters
    ----------
    drape : int
        Drape indicates how particles are treated when starting locations
        are specified for cells that are dry. If drape is 0, Particles are
        placed in the specified cell. If the cell is dry at the time of
        release, the status of the particle is set to unreleased and
        removed from the simulation. If drape is 1, particles are placed
        in the upper most active grid cell directly beneath the specified
        layer, row, column or node location (default is 0).
    columncelldivisions : int
        Number of particles in a cell in the column (x-coordinate)
        direction (default is 3).
    rowcelldivisions : int
        Number of particles in a cell in the row (y-coordinate)
        direction (default is 3).
    layercelldivisions : int
        Number of particles in a cell in the layer (z-coordinate)
        direction (default is 3).

    Examples
    --------

    >>> import flopy
    >>> cd = flopy.modpath.CellDataType()

    """

    def __init__(
        self,
        drape=0,
        columncelldivisions=3,
        rowcelldivisions=3,
        layercelldivisions=3,
    ):
        """
        Class constructor

        """
        self.name = "CellDataType"

        # assign attributes
        self.templatesubdivisiontype = 2
        self.drape = drape
        self.columncelldivisions = columncelldivisions
        self.rowcelldivisions = rowcelldivisions
        self.layercelldivisions = layercelldivisions
        return

    def write(self, f=None):
        """

        Parameters
        ----------
        f : fileobject
            Fileobject that is open with write access

        Returns
        -------

        """
        # validate that a valid file object was passed
        if not hasattr(f, "write"):
            raise ValueError(
                "{}: cannot write data for template "
                "without passing a valid file object ({}) "
                "open for writing".format(self.name, f)
            )

        # item 5
        fmt = " {} {} {}\n"
        line = fmt.format(
            self.columncelldivisions,
            self.rowcelldivisions,
            self.layercelldivisions,
        )
        f.write(line)

        return


class LRCParticleData:
    """
    Layer, row, column particle data template class to create MODPATH 7
    particle location input style 2 on cell faces (templatesubdivisiontype = 1)
    and/or in cells (templatesubdivisiontype = 2). Particle locations for this
    template are specified by layer, row, column regions.

    Parameters
    ----------
    subdivisiondata : FaceDataType, CellDataType or list of FaceDataType
                      and/or CellDataType types
        FaceDataType, CellDataType, or a list of FaceDataType and/or
        CellDataTypes that are used to create one or more particle templates
        in a particle group. If subdivisiondata is None, a default CellDataType
        with 27 particles per cell will be created (default is None).
    lrcregions : list of lists tuples or np.ndarrays
        Layer, row, column (zero-based) regions with particles created using
        the specified template parameters. A region is defined as a list/tuple
        of minlayer, minrow, mincolumn, maxlayer, maxrow, maxcolumn values.
        If subdivisiondata is a list, a list/tuple or array of layer, row,
        column regions with the same length as subdivision data must be
        provided. If lrcregions is None, particles will be placed in
        the first model cell (default is None).

    Examples
    --------

    >>> import flopy
    >>> pg = flopy.modpath.LRCParticleData(lrcregions=[0, 0, 0, 3, 10, 10])

    """

    def __init__(self, subdivisiondata=None, lrcregions=None):
        """
        Class constructor

        """
        self.name = "LRCParticleData"

        if subdivisiondata is None:
            subdivisiondata = CellDataType()

        if lrcregions is None:
            lrcregions = [[0, 0, 0, 0, 0, 0]]

        if isinstance(subdivisiondata, (CellDataType, FaceDataType)):
            subdivisiondata = [subdivisiondata]

        for idx, fd in enumerate(subdivisiondata):
            if not isinstance(fd, (CellDataType, FaceDataType)):
                raise TypeError(
                    "{}: facedata item {} is of type {} instead of an "
                    "instance of CellDataType or FaceDataType".format(
                        self.name, idx, type(fd)
                    )
                )

        # validate lrcregions data
        if isinstance(lrcregions, (list, tuple)):
            # determine if the list or tuple contains lists or tuples
            alllsttup = all(
                isinstance(el, (list, tuple, np.ndarray)) for el in lrcregions
            )
            if not alllsttup:
                raise TypeError(
                    "{}: lrcregions should be "
                    "a list with lists, tuples, or arrays".format(self.name)
                )
            t = []
            for lrcregion in lrcregions:
                t.append(np.array(lrcregion, dtype=np.int32))
            lrcregions = t
        else:
            raise TypeError(
                "{}: lrcregions should be a list of lists, tuples, or arrays "
                "not a {}.".format(self.name, type(lrcregions))
            )

        # validate size of nodes relative to subdivisiondata
        shape = len(subdivisiondata)
        if len(lrcregions) != shape:
            raise ValueError(
                "{}: lrcregions data must have {} rows but a total of {} rows "
                "were provided.".format(self.name, shape, lrcregions.shape[0])
            )

        # validate that there are 6 columns in each lrcregions entry
        for idx, lrcregion in enumerate(lrcregions):
            shapel = lrcregion.shape
            if len(shapel) == 1:
                lrcregions[idx] = lrcregion.reshape(1, shapel)
                shapel = lrcregion[idx].shape
            if shapel[1] != 6:
                raise ValueError(
                    "{}: Each lrcregions entry must "
                    "have 6 columns passed lrcregions has "
                    "{} columns".format(self.name, shapel[1])
                )

        #
        totalcellregioncount = 0
        for lrcregion in lrcregions:
            totalcellregioncount += lrcregion.shape[0]

        # assign attributes
        self.particletemplatecount = shape
        self.totalcellregioncount = totalcellregioncount
        self.subdivisiondata = subdivisiondata
        self.lrcregions = lrcregions
        return

    def write(self, f=None):
        """

        Parameters
        ----------
        f : fileobject
            Fileobject that is open with write access

        Returns
        -------

        """
        # validate that a valid file object was passed
        if not hasattr(f, "write"):
            raise ValueError(
                "{}: cannot write data for template "
                "without passing a valid file object ({}) "
                "open for writing".format(self.name, f)
            )

        # item 2
        f.write(
            "{} {}\n".format(
                self.particletemplatecount, self.totalcellregioncount
            )
        )

        for sd, lrcregion in zip(self.subdivisiondata, self.lrcregions):
            # item 3
            f.write(
                "{} {} {}\n".format(
                    sd.templatesubdivisiontype, lrcregion.shape[0], sd.drape
                )
            )

            # item 4 or 5
            sd.write(f)

            # item 6
            for row in lrcregion:
                line = ""
                for lrc in row:
                    line += "{} ".format(lrc + 1)
                line += "\n"
                f.write(line)

        return


class NodeParticleData:
    """
    Node particle data template class to create MODPATH 7 particle location
    input style 3 on cell faces (templatesubdivisiontype = 1) and/or in cells
    (templatesubdivisiontype = 2). Particle locations for this template are
    specified by nodes.

    Parameters
    ----------
    subdivisiondata : FaceDataType, CellDataType or list of FaceDataType
                      and/or CellDataType types
        FaceDataType, CellDataType, or a list of FaceDataType and/or
        CellDataTypes that are used to create one or more particle templates
        in a particle group. If subdivisiondata is None, a default CellDataType
        with 27 particles per cell will be created (default is None).
    nodes : int, list of ints, tuple of ints, or np.ndarray
        Nodes (zero-based) with particles created using the specified template
        parameters. If subdivisiondata is a list, a list of nodes with the same
        length as subdivision data must be provided. If nodes is None,
        particles will be placed in the first model cell (default is None).

    Examples
    --------

    >>> import flopy
    >>> pg = flopy.modpath.NodeParticleData(nodes=[100, 101])

    """

    def __init__(self, subdivisiondata=None, nodes=None):
        """
        Class constructor

        """
        self.name = "NodeParticleData"

        if subdivisiondata is None:
            subdivisiondata = CellDataType()

        if nodes is None:
            nodes = 0

        if isinstance(subdivisiondata, (CellDataType, FaceDataType)):
            subdivisiondata = [subdivisiondata]

        if isinstance(nodes, (int, np.int32, np.int64)):
            nodes = [(nodes,)]
        elif isinstance(nodes, (float, np.float32, np.float64)):
            raise TypeError(
                "{}: nodes is of type {} but must be an int if a "
                "single value is passed".format(self.name, type(nodes))
            )

        for idx, fd in enumerate(subdivisiondata):
            if not isinstance(fd, (CellDataType, FaceDataType)):
                raise TypeError(
                    "{}: facedata item {} is of type {} instead of an "
                    "instance of CellDataType or FaceDataType".format(
                        self.name, idx, type(fd)
                    )
                )

        # validate nodes data
        if isinstance(nodes, np.ndarray):
            if len(nodes.shape) == 1:
                nodes = nodes.reshape(1, nodes.shape[0])
            # convert to a list of numpy arrays
            t = []
            for idx in range(nodes.shape[0]):
                t.append(np.array(nodes[idx, :], dtype=np.int32))
            nodes = t
        elif isinstance(nodes, (list, tuple)):
            # convert a single list/tuple to a list of tuples if only one
            # entry in subdivisiondata
            if len(subdivisiondata) == 1:
                if len(nodes) > 1:
                    nodes = [tuple(nodes)]
            # determine if the list or tuple contains lists or tuples
            alllsttup = all(
                isinstance(el, (list, tuple, np.ndarray)) for el in nodes
            )
            if not alllsttup:
                raise TypeError(
                    "{}: nodes should be "
                    "a list or tuple with lists or tuple if a single "
                    "int or numpy array is not provided".format(self.name)
                )
            t = []
            for idx in range(len(nodes)):
                t.append(np.array(nodes[idx], dtype=np.int32))
            nodes = t
        else:
            raise TypeError(
                "{}: nodes should be a single integer, a numpy array, or a "
                "list/tuple or lists/tuples.".format(self.name)
            )

        # validate size of nodes relative to subdivisiondata
        shape = len(subdivisiondata)
        if len(nodes) != shape:
            raise ValueError(
                "{}: node data must have {} rows but a total of {} rows were "
                "provided.".format(self.name, shape, nodes.shape[0])
            )

        totalcellcount = 0
        for t in nodes:
            totalcellcount += t.shape[0]

        # assign attributes
        self.particletemplatecount = shape
        self.totalcellcount = totalcellcount
        self.subdivisiondata = subdivisiondata
        self.nodedata = nodes
        return

    def write(self, f=None):
        """

        Parameters
        ----------
        f : fileobject
            Fileobject that is open with write access

        Returns
        -------

        """
        # validate that a valid file object was passed
        if not hasattr(f, "write"):
            raise ValueError(
                "{}: cannot write data for template "
                "without passing a valid file object ({}) "
                "open for writing".format(self.name, f)
            )

        # item 2
        f.write(
            "{} {}\n".format(self.particletemplatecount, self.totalcellcount)
        )

        for sd, nodes in zip(self.subdivisiondata, self.nodedata):
            # item 3
            f.write(
                "{} {} {}\n".format(
                    sd.templatesubdivisiontype, nodes.shape[0], sd.drape
                )
            )

            # item 4 or 5
            sd.write(f)

            # item 6
            line = ""
            for idx, node in enumerate(nodes):
                line += " {}".format(node + 1)
                lineend = False
                if idx > 0:
                    if idx % 10 == 0 or idx == nodes.shape[0] - 1:
                        lineend = True
                if lineend:
                    line += "\n"
            f.write(line)

        return
