"""
mp7particledata module. Contains the ParticleData, NodeParticleDataFace,
    NodeParticleDataCell, LRCParticleDataFace, and LRCParticleDataCell classes.


"""

import os
import numpy as np
from ..utils.util_array import Util2d
from ..utils.recarray_utils import create_empty_recarray


class ParticleData(object):
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

    def __init__(self, partlocs=None, structured=False, particleids=None,
                 localx=None, localy=None, localz=None, timeoffset=None,
                 drape=None):
        """
        Class constructor

        """
        self.name = 'ParticleData'

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
            dtype.append(('k', np.int32))
            dtype.append(('i', np.int32))
            dtype.append(('j', np.int32))
        else:
            dtype.append(('node', np.int32))
        dtype = np.dtype(dtype)

        if isinstance(partlocs, (list, tuple)):
            # determine if the list or tuple contains lists or tuples
            alllsttup = all(isinstance(el, (list, tuple)) for el in partlocs)
            if structured:
                if alllsttup:
                    alllen3 = all(len(el) == 3 for el in partlocs)
                    if not alllen3:
                        msg = '{}: all partlocs entries '.format(self.name) + \
                              ' must have 3 items for structured particle data'
                        raise ValueError(msg)
                else:
                    msg = '{}: partlocs list or tuple '.format(self.name) + \
                          'for structured particle data should ' + \
                          'contain list or tuple entries'
                    raise ValueError(msg)
            else:
                allint = all(isinstance(el, (int, np.int32, np.int64))
                             for el in partlocs)
                # convert to a list of tuples
                if allint:
                    t = []
                    for el in partlocs:
                        t.append((el,))
                    partlocs = t
                    alllsttup = all(isinstance(el, (list, tuple))
                                    for el in partlocs)
                if alllsttup:
                    alllen1 = all(len(el) == 1 for el in partlocs)
                    if not alllen1:
                        msg = '{}: all entries of '.format(self.name) + \
                              'partlocs must have 1 items ' + \
                              'for unstructured particle data'
                        raise ValueError(msg)
                else:
                    msg = '{}: partlocs list or tuple '.format(self.name) + \
                          'for unstructured particle data should ' + \
                          'contain integers or a list or tuple with one entry'
                    raise ValueError(msg)

            # convert partlocs composed of a lists/tuples of lists/tuples
            # to a numpy array
            partlocs = np.array(partlocs, dtype=dtype)
        elif isinstance(partlocs, np.ndarray):
            dtypein = partlocs.dtype
            if dtypein != partlocs.dtype:
                partlocs = np.array(partlocs, dtype=dtype)
        else:
            msg = '{}: partlocs must be a list or '.format(self.name) + \
                  'tuple with lists or tuples'
            raise ValueError(msg)

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
                    msg = '{}:'.format(self.name) + \
                          'shape of localx ({}) '.format(localx.shape[0]) + \
                          'is not equal to the shape ' + \
                          'of partlocs ({}).'.format(partlocs.shape[0])
                    raise ValueError(msg)

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
                    msg = '{}:'.format(self.name) + \
                          'shape of localy ({}) '.format(localy.shape[0]) + \
                          'is not equal to the shape ' + \
                          'of partlocs ({}).'.format(partlocs.shape[0])
                    raise ValueError(msg)

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
                    msg = '{}:'.format(self.name) + \
                          'shape of localz ({}) '.format(localz.shape[0]) + \
                          'is not equal to the shape ' + \
                          'of partlocs ({}).'.format(partlocs.shape[0])
                    raise ValueError(msg)
        # timeoffset
        if timeoffset is None:
            timeoffset = 0.
        else:
            if isinstance(timeoffset, (float, int)):
                timeoffset = np.ones(partlocs.shape[0], dtype=np.float32) * \
                             timeoffset
            elif isinstance(timeoffset, (list, tuple)):
                timeoffset = np.array(timeoffset, dtype=np.float32)
            if isinstance(timeoffset, np.ndarray):
                if timeoffset.shape[0] != partlocs.shape[0]:
                    msg = '{}:'.format(self.name) + \
                          'shape of timeoffset ' + \
                          '({}) '.format(timeoffset.shape[0]) + \
                          'is not equal to the shape ' + \
                          'of partlocs ({}).'.format(partlocs.shape[0])
                    raise ValueError(msg)

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
                    msg = '{}:'.format(self.name) + \
                          'shape of drape ({}) '.format(drape.shape[0]) + \
                          'is not equal to the shape ' + \
                          'of partlocs ({}).'.format(partlocs.shape[0])
                    raise ValueError(msg)

        # particleids
        if particleids is None:
            particleid = False
            particleidoption = 0
        else:
            particleid = True
            particleidoption = 1
            if isinstance(particleids, (int, float)):
                msg = '{}:'.format(self.name) + \
                      'A particleid must be provided for each partloc ' + \
                      'as a list/tuple/np.ndarray of size ' + \
                      '{}. '.format(partlocs.shape[0]) + \
                      'A single particleid has been provided.'
                raise TypeError(msg)
            elif isinstance(particleids, (list, tuple)):
                particleids = np.array(particleids, dtype=np.int32)
            if isinstance(particleids, np.ndarray):
                if particleids.shape[0] != partlocs.shape[0]:
                    msg = '{}:'.format(self.name) + \
                          'shape of particleids ' + \
                          '({}) '.format(particleids.shape[0]) + \
                          'is not equal to the shape ' + \
                          'of partlocs ({}).'.format(partlocs.shape[0])
                    raise ValueError(msg)

        # create empty particle
        ncells = partlocs.shape[0]
        self.dtype = self._get_dtype(structured, particleid)
        particledata = create_empty_recarray(ncells, self.dtype,
                                             default_value=0)

        # fill particle
        if structured:
            particledata['k'] = partlocs['k']
            particledata['i'] = partlocs['i']
            particledata['j'] = partlocs['j']
        else:
            particledata['node'] = partlocs['node']
        particledata['localx'] = localx
        particledata['localy'] = localy
        particledata['localz'] = localz
        particledata['timeoffset'] = timeoffset
        particledata['drape'] = drape
        if particleid:
            particledata['id'] = particleids

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
        if not hasattr(f, 'write'):
            msg = '{}: cannot write data for template '.format(self.name) + \
                  'without passing a valid file object ({}) '.format(f) + \
                  'open for writing'
            raise ValueError(msg)

        # particle data item 4 and 5
        d = np.recarray.copy(self.particledata)
        lnames = [name.lower() for name in d.dtype.names]
        # Add one to the kij and node indices
        for idx in ['k', 'i', 'j', 'node', 'id']:
            if idx in lnames:
                d[idx] += 1

        # write the particle data
        fmt = self._fmt_string + '\n'
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
            dtype.append(('id', np.int32))
        if structured:
            dtype.append(('k', np.int32))
            dtype.append(('i', np.int32))
            dtype.append(('j', np.int32))
        else:
            dtype.append(('node', np.int32))
        dtype.append(('localx', np.float32))
        dtype.append(('localy', np.float32))
        dtype.append(('localz', np.float32))
        dtype.append(('timeoffset', np.float32))
        dtype.append(('drape', np.int32))
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
            if vtype == 'i' or vtype == 'b':
                fmts.append('{:9d}')
            elif vtype == 'f':
                if field[1][2] == 8:
                    fmts.append('{:23.16g}')
                else:
                    fmts.append('{:15.7g}')
            elif vtype == 'o':
                fmts.append('{:9s}')
            elif vtype == 's':
                msg = "Particles.fmt_string error: 'str' " + \
                      "type found in dtype. This gives unpredictable " + \
                      "results when recarray to file - change to 'object' type"
                raise TypeError(msg)
            else:
                msg = "MfList.fmt_string error: unknown vtype in " + \
                      "field: {}".format(field)
                raise TypeError(msg)
        return ' ' + ' '.join(fmts)


class FaceDataType(object):
    """
    Face data type class to create MODPATH 7 particle location
    input style 2, 3, and 4 on cell faces (templatesubdivisiontype = 1).

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
    columndivisons5 : int
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

    def __init__(self, drape=0,
                 verticaldivisions1=3, horizontaldivisions1=3,
                 verticaldivisions2=3, horizontaldivisions2=3,
                 verticaldivisions3=3, horizontaldivisions3=3,
                 verticaldivisions4=3, horizontaldivisions4=3,
                 rowdivisions5=3, columndivisons5=3,
                 rowdivisions6=3, columndivisions6=3):
        """
        Class constructor

        """
        self.name = 'FaceDataType'

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
        self.columndivisons5 = columndivisons5
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
        if not hasattr(f, 'write'):
            msg = '{}: cannot write data for template '.format(self.name) + \
                  'without passing a valid file object ({}) '.format(f) + \
                  'open for writing'
            raise ValueError(msg)

        # item 4
        fmt = 12 * ' {}' + '\n'
        line = fmt.format(self.verticaldivisions1, self.horizontaldivisions1,
                          self.verticaldivisions2, self.horizontaldivisions2,
                          self.verticaldivisions3, self.horizontaldivisions3,
                          self.verticaldivisions4, self.horizontaldivisions4,
                          self.rowdivisions5, self.columndivisons5,
                          self.rowdivisions6, self.columndivisions6)
        f.write(line)

        return

class NodeParticleDataFace(object):
    """
    Node particle data template class to create MODPATH 7 particle location
    input style 3 on cell faces (templatesubdivisiontype = 1). Particle
    locations for this template are specified by nodes.

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
    columndivisons5 : int
        The number of column subdivisions that define the two-dimensional array
        of particles on the bottom cell face (face 5) (default is 3).
    rowdivisions6 : int
        The number of row subdivisions that define the two-dimensional array
        of particles on the top cell face (face 6) (default is 3).
    columndivisions6 : int
        The number of column subdivisions that define the two-dimensional array
        of particles on the top cell face (face 6) (default is 3).
    nodes : int, list of ints, tuple of ints, or np.ndarray
        Nodes (zero-based) with particles created using the specified template
        parameters (default is node 0).

    Examples
    --------

    >>> import flopy
    >>> pf = flopy.modpath.NodeParticleDataFace(nodes=[100, 101])

    """

    def __init__(self, drape=0,
                 verticaldivisions1=3, horizontaldivisions1=3,
                 verticaldivisions2=3, horizontaldivisions2=3,
                 verticaldivisions3=3, horizontaldivisions3=3,
                 verticaldivisions4=3, horizontaldivisions4=3,
                 rowdivisions5=3, columndivisons5=3,
                 rowdivisions6=3, columndivisions6=3,
                 nodes=(0,)):
        """
        Class constructor

        """
        self.name = 'NodeParticleDataFace'

        # validate nodes
        if not isinstance(nodes, np.ndarray):
            if isinstance(nodes, (int, np.int32, np.int64)):
                nodes = np.array([nodes], dtype=np.int32)
            elif isinstance(nodes, (list, tuple)):
                nodes = np.array(nodes, dtype=np.int32)
            else:
                msg = '{}: node data must be a '.format(self.name) + \
                      'integer, list of integers, or tuple of integers'
                raise TypeError(msg)

        # validate shape of nodes
        templatecellcount = nodes.shape[0]
        if len(nodes.shape) > 1:
            msg = '{}: processed node data must be a '.format(self.name) + \
                  'numpy array has a shape of {} '.format(nodes.shape) + \
                  'but should have a shape of ({}) '.format(nodes.shape[0])
            raise TypeError(msg)

        # assign attributes
        self.templatesubdivisiontype = 1
        self.templatecellcount = templatecellcount
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
        self.columndivisons5 = columndivisons5
        self.rowdivisions6 = rowdivisions6
        self.columndivisions6 = columndivisions6
        self.nodes = nodes
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
        if not hasattr(f, 'write'):
            msg = '{}: cannot write data for template '.format(self.name) + \
                  'without passing a valid file object ({}) '.format(f) + \
                  'open for writing'
            raise ValueError(msg)

        # item 3
        f.write('{} {} {}\n'.format(self.templatesubdivisiontype,
                                    self.templatecellcount,
                                    self.drape))

        # item 4
        fmt = 12 * ' {}' + '\n'
        line = fmt.format(self.verticaldivisions1, self.horizontaldivisions1,
                          self.verticaldivisions2, self.horizontaldivisions2,
                          self.verticaldivisions3, self.horizontaldivisions3,
                          self.verticaldivisions4, self.horizontaldivisions4,
                          self.rowdivisions5, self.columndivisons5,
                          self.rowdivisions6, self.columndivisions6)
        f.write(line)

        # item 6
        line = ''
        for idx, node in enumerate(self.nodes):
            line += ' {}'.format(node + 1)
            lineend = False
            if idx > 0:
                if idx % 10 == 0 or idx == self.nodes.shape[0] - 1:
                    lineend = True
            if lineend:
                line += '\n'
        f.write(line)

        return



class NodeParticleDataCell(object):
    """
    Node particle data template class to create MODPATH 7 particle location
    input style 3 within cells (templatesubdivisiontype = 2). Particle
    locations for this template are specified by nodes.

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
        Number of oarticles in a cell in the layer (z-coordinate)
        direction (default is 3).
    nodes : int, list of ints, tuple of ints, or np.ndarray
        Nodes (zero-based) with particles created using the specified template
        parameters (default is node 0).

    Examples
    --------

    >>> import flopy
    >>> pc = flopy.modpath.NodeParticleDataCell(nodes=[100, 101])

    """

    def __init__(self, drape=0,
                 columncelldivisions=3, rowcelldivisions=3,
                 layercelldivisions=3, nodes=(0,)):
        """
        Class constructor

        """
        self.name = 'NodeParticleDataCell'

        # validate nodes
        if not isinstance(nodes, np.ndarray):
            if isinstance(nodes, (int, np.int32, np.int64)):
                nodes = np.array([nodes], dtype=np.int32)
            elif isinstance(nodes, (list, tuple)):
                nodes = np.array(nodes, dtype=np.int32)
            else:
                msg = '{}: node data must be a integer, '.format(self.name) + \
                      'list of integers or tuple of integers'
                raise TypeError(msg)

        # validate shape of nodes
        templatecellcount = nodes.shape[0]
        if len(nodes.shape) > 1:
            msg = '{}: processed node data must be a '.format(self.name) + \
                  'numpy array has a shape of {} '.format(nodes.shape) + \
                  'but should have a shape of ({}) '.format(nodes.shape[0])
            raise TypeError(msg)

        # assign attributes
        self.templatesubdivisiontype = 2
        self.templatecellcount = templatecellcount
        self.drape = drape
        self.columncelldivisions = columncelldivisions
        self.rowcelldivisions = rowcelldivisions
        self.layercelldivisions = layercelldivisions
        self.nodes = nodes
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
        if not hasattr(f, 'write'):
            msg = '{}: cannot write data for template '.format(self.name) + \
                  'without passing a valid file object ({}) '.format(f) + \
                  'open for writing'
            raise ValueError(msg)

        # item 3
        f.write('{} {} {}\n'.format(self.templatesubdivisiontype,
                                    self.templatecellcount,
                                    self.drape))

        # item 5
        fmt = ' {} {} {}\n'
        line = fmt.format(self.columncelldivisions, self.rowcelldivisions,
                          self.layercelldivisions)
        f.write(line)

        # item 6
        line = ''
        for idx, node in enumerate(self.nodes):
            line += ' {}'.format(node + 1)
            lineend = False
            if idx > 0:
                if idx % 10 == 0 or idx == self.nodes.shape[0] - 1:
                    lineend = True
            if lineend:
                line += '\n'
        f.write(line)

        return