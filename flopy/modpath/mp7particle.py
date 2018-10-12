"""
mp7particle module.  Contains the Modpath7Particle, Particles,
    NodeParticleTemplate, ParticleFaceNodeData, and ParticleCellNodeData
    classes.


"""

import os
import numpy as np
from ..utils.util_array import Util2d
from ..utils.recarray_utils import create_empty_recarray


class _Modpath7ParticleGroup(object):
    """
    Base particle group class that defines common data to all particle
    input styles (MODPATH 7 simulation file items 26 through 32).
    _Modpath7ParticleGroup should not be called directly.

    Parameters
    ----------
    particlegroupname : str
       Name of particle group
    filename : str
        Name of the external file that will contain the particle data.
        If filename is '' or None the particle information for the
        particle group will be written to the MODPATH7 simulation
        file.
    releasedata : float, int, list, or tuple
        If releasedata is a float or an int or a list/tuple with a single
        float or int, releaseoption is set to 1 and release data is the
        particle release time.

    """

    def __init__(self, particlegroupname, filename, releasedata):
        """
        Class constructor

        """
        self.particlegroupname = particlegroupname
        if filename == '':
            filename = None
        self.filename = filename
        if self.filename is None:
            self.external = False
        else:
            self.external = True

        if releasedata is None:
            msg = 'releasedata must be provided to instantiate ' + \
                  'a MODPATH 7 particle group'
            raise ValueError(msg)

        # convert releasedata to a list, if required
        if isinstance(releasedata, (float, int)):
            releasedata = [releasedata]
        elif isinstance(releasedata, np.ndarray):
            releasedata = releasedata.tolist()

        # validate that releasedata is a list or tuple
        if not isinstance(releasedata, (list, tuple)):
            msg = 'releasedata must be a float, int, list, or tuple'
            raise ValueError(msg)

        # process releasedata
        if len(releasedata) == 1:
            releaseoption = 1
            releasetimecount = 0
            releaseinterval = 0
            releasetimes = np.array(releasedata, dtype=np.float32)
        elif len(releasedata) == 3:
            releaseoption = 2
            releasetimecount = int(releasedata[0])
            releaseinterval = int(releasedata[2])
            releasetimes = np.array(releasedata[1], dtype=np.float32)
        elif len(releasedata) == 2:
            releaseoption = 3
            releasetimecount = int(releasedata[0])
            releaseinterval = 0
            # convert releasetimes list or tuple to a numpy array
            if isinstance(releasedata[1], list) \
                    or isinstance(releasedata[1], tuple):
                releasedata[1] = np.array(releasedata[1])
            if releasedata[1].shape[0] != releasetimecount:
                msg = 'The number of releasetimes data ' + \
                      '({}) '.format(releasedata[1].shape[0]) + \
                      'is not equal to releasetimecount ' + \
                      '({}).'.format(releasetimecount)
                raise ValueError(msg)
            releasetimes = np.array(releasedata[1], dtype=np.float32)
        else:
            msg = 'releasedata must have 1, 2, or 3 entries'
            raise ValueError(msg)

        # set release data
        self.releaseoption = releaseoption
        self.releasetimecount = releasetimecount
        self.releaseinterval = releaseinterval
        self.releasetimes = releasetimes

    def write(self, fp=None, ws='.'):
        """
        Common write of MODPATH 7 simulation file items 26 through 32

        Parameters
        ----------
        fp : fileobject
            Fileobject that is open with write access
        ws : str
            Workspace for particle data

        Returns
        -------

        """

        # validate that a valid file object was passed
        if not hasattr(fp, 'write'):
            msg = 'Cannot write data for particle group ' + \
                  '{} '.format(self.particlegroupname) + \
                  'without passing a valid file object ({}) '.format(fp) + \
                  'open for writing'
            raise ValueError(msg)

        # item 26
        fp.write('{}\n'.format(self.particlegroupname))

        # item 27
        fp.write('{}\n'.format(self.releaseoption))

        if self.releaseoption == 1:
            # item 28
            fp.write('{}\n'.format(self.releasetimes[0]))
        elif self.releaseoption == 2:
            # item 29
            fp.write('{} {} {}\n'.format(self.releasetimecount,
                                         self.releasetimes[0],
                                         self.releaseinterval))
        elif self.releaseoption == 3:
            # item 30
            fp.write('{}\n'.format(self.releasetimecount))
            # item 31
            tp = self.releasetimes
            v = Util2d(self, (tp.shape[0],),
                       np.float32, tp,
                       name='temp',
                       locat=0)
            fp.write(v.string)

        # item 32
        if self.external:
            line = 'EXTERNAL {}\n'.format(self.filename)
        else:
            line = 'INTERNAL\n'
        fp.write(line)

        return


class ParticleGroup(_Modpath7ParticleGroup):
    """
    ParticleGroup class to create MODPATH 7 particle group data for location
    input style 1. Location input style 1 is the most general type of particle
    group that requires the user to define the location of all particles and
    associated data (relative release time, drap, and optionally particle ids).
    Particledata locations can be specified by layer, row, column
    (locationstyle=1) or nodes (locationstyle=2).

    Parameters
    ----------
    particlegroupname : str
       Name of particle group (default is 'PG1')
    filename : str
        Name of the external file that will contain the particle data.
        If filename is '' or None the particle information for the
        particle group will be written to the MODPATH7 simulation
        file (default is None).
    releasedata : float, int, list, or tuple
        If releasedata is a float or an int or a list/tuple with a single
        float or int, releaseoption is set to 1 and release data is the
        particle release time (default is 0.0).
    particledata : list, tuple, or numpy array
        list, tuple, or numpy array with particle data. Each row of
        particledata must have 6, 7, 8, or 9 entries. particledata with
        6 and 7 entries per row are used to define particles with starting
        positions defined by node. particledata with 8 and 9 entries per
        row are used to define particles with starting positions defined
        by layer, row, column. particledata with 7 and 9 entries also
        include user defined particle ids. Layer, row, column locations and
        nodes are zero-based. If particledata is None, a node-based particle
        will be placed at the center of the first node in the model
        (default is None).

    Examples
    --------

    >>> import flopy
    >>> p = [(2, 0, 0), (0, 20, 0)]
    >>> p = flopy.modpath.ParticleGroup.create_particledata(p)
    >>> pg = flopy.modpath.ParticleGroup(particledata=p)

    """

    def __init__(self, particlegroupname='PG1', filename=None,
                 releasedata=0.0,
                 particledata=None):
        """
        Class constructor

        """

        # instantiate base class
        _Modpath7ParticleGroup.__init__(self, particlegroupname, filename,
                                        releasedata)
        self.inputstyle = 1

        # create default node-based particle data if not passed
        if particledata is None:
            particledata = [(0, 0.5, 0.5, 0.5, 0., 0)]

        # convert list, tuples, and numpy array to
        dtypein = None
        if isinstance(particledata, (list, tuple)):
            ncells = len(particledata)
            v = particledata[0]
        elif isinstance(particledata, np.ndarray):
            dtypein = particledata.dtype
            ncells = particledata.shape[0]
            v = particledata[0].tolist()
        else:
            msg = 'particledata must be a list, tuple, ' + \
                  'numpy ndarray, or an instance of ' + \
                  'Particles'
            raise ValueError(msg)

        # parse the first row to determine particleidoption, locationstyle,
        # if the particle id is user-defined, and if the data are layer, row,
        # column based (structured=True) or node-based (structured=False)
        if len(v) == 6:
            particleidoption = 0
            locationstyle = 2
            partid = False
            structured = False
        elif len(v) == 7:
            particleidoption = 1
            locationstyle = 2
            partid = True
            structured = False
        elif len(v) == 8:
            particleidoption = 0
            locationstyle = 1
            partid = False
            structured = True
        elif len(v) == 9:
            particleidoption = 1
            locationstyle = 1
            partid = True
            structured = True
        else:
            msg = 'particledata should have 6, 7, 8, or 9 columns.' + \
                  'Specified data for particle group ' + \
                  '{} '.format(particlegroupname) + \
                  'only has {} columns.'.format(ncells)
            raise ValueError(msg)
        dtype = self.get_particledata_dtype(structured=structured,
                                            particleid=partid)
        if dtype != dtypein:
            particledata = np.array(particledata, dtype=dtype)

        # set attributes
        self.particleidoption = particleidoption
        self.locationstyle = locationstyle
        self.particledata = particledata

        return

    def write(self, fp=None, ws='.'):
        """
        Write MODPATH 7 particle data items 1 through 5

        Parameters
        ----------
        fp : fileobject
            Fileobject that is open with write access
        ws : str
            Workspace for particle data

        Returns
        -------

        """

        # call base class write method to write common data
        _Modpath7ParticleGroup.write(self, fp, ws)

        # open external file if required
        if self.external:
            fpth = os.path.join(ws, self.filename)
            f = open(fpth, 'w')
        else:
            f = fp

        # particle data item 1
        f.write('{}\n'.format(self.inputstyle))

        # particle data item 2
        f.write('{}\n'.format(self.locationstyle))

        # particle data item 3
        f.write('{} {}\n'.format(self.particledata.shape[0],
                                 self.particleidoption))

        # particle data item 4 and 5
        d = np.recarray.copy(self.particledata)
        lnames = [name.lower() for name in d.dtype.names]
        # Add one to the kij and node indices
        for idx in ['k', 'i', 'j', 'node', 'id']:
            if idx in lnames:
                d[idx] += 1

        # save the particle data
        # np.savetxt(f, d, fmt=self.fmt_string, delimiter='')
        fmt = self.fmt_string + '\n'
        for v in d:
            f.write(fmt.format(*v))

        # close the external file
        if self.external:
            f.close()

        return

    @property
    def fmt_string(self):
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

    @staticmethod
    def get_particledata_dtype(structured=True, particleid=False):
        """
        Static method to get the dtype for a structured or unstructured
        particledata recarray. Optionally, include a particleid column in
        the dtype.


        Parameters
        ----------
        structured : bool
            Boolean defining if a structured (True) or unstructured
            particle dtype will be created (default is True).
        particleid : bool
            Boolean defining if the dtype will include a particle id
            column (default is False).

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

    @staticmethod
    def get_particledata_empty(ncells=0, structured=True, particleid=False):
        """
        Static method to get an empty structured or unstructured particledata
        recarray. Optionally, include a particleid column in the recarray.

        Parameters
        ----------
        ncells : int
            Number of particles (default is 0).
        structured : bool
            Boolean defining if a structured (True) or unstructured
            particle recarray will be created (default is True).
        particleid : bool
            Boolean defining if the particle recarray will include a
            particle id column (default is False).

        Returns
        -------
        recarray : numpy recarray

        Examples
        --------

        >>> import flopy.modpath as fmp
        >>> pd = fmp.ParticleGroup.get_particledata_empty(ncells=21,
        ...                                               particleid=True)


        """
        # get an empty recarray that corresponds to dtype
        dtype = ParticleGroup.get_particledata_dtype(structured=structured,
                                                     particleid=particleid)
        return create_empty_recarray(ncells, dtype, default_value=0)

    @staticmethod
    def create_particledata(partlocs, structured=True, particleids=None,
                            localx=None, localy=None, localz=None,
                            timeoffset=None, drape=None):
        """
        Static method to create a particledata recarray based on passed
        particle locations and optional local cell location, timeoffset,
        drape, and particleid data.

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

        Returns
        -------
        particledata : particledata recarray

        Examples
        --------

        >>> import flopy.modpath as fmp
        >>> locs = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
        >>> pd = fmp.ParticleGroup.create_particledata(locs, structured=True,
        ...                                            drape=0, localx=0.5,
        ...                                            localy=0.5, localz=1)

        """
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
                        msg = 'all entries of partlocs must have 3 items ' + \
                              'for structured particle data'
                        raise ValueError(msg)
                else:
                    msg = 'partlocs list or tuple for structured ' + \
                          'particle data should contain list or tuple entries'
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
                        msg = 'all entries of partlocs must have 1 items ' + \
                              'for unstructured particle data'
                        raise ValueError(msg)
                else:
                    msg = 'partlocs list or tuple for unstructured ' + \
                          ' particle datashould contain integers or a ' + \
                          'list or tuple with one entry'
                    raise ValueError(msg)

            # convert partlocs composed of a lists/tuples of lists/tuples
            # to a numpy array
            partlocs = np.array(partlocs, dtype=dtype)
        elif isinstance(partlocs, np.ndarray):
            dtypein = partlocs.dtype
            if dtypein != partlocs.dtype:
                partlocs = np.array(partlocs, dtype=dtype)
        else:
            msg = 'create_Particles partlocs must be a list or ' + \
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
                    msg = 'shape of localx ({}) '.format(localx.shape[0]) + \
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
                    msg = 'shape of localy ({}) '.format(localy.shape[0]) + \
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
                    msg = 'shape of localz ({}) '.format(localz.shape[0]) + \
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
                    msg = 'shape of timeoffset ' + \
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
                    msg = 'shape of drape ({}) '.format(drape.shape[0]) + \
                          'is not equal to the shape ' + \
                          'of partlocs ({}).'.format(partlocs.shape[0])
                    raise ValueError(msg)

        # particleids
        if particleids is None:
            particleid = False
        else:
            particleid = True
            if isinstance(particleids, (int, float)):
                msg = 'A particleid must be provided for each partloc ' + \
                      'as a list/tuple/np.ndarray of size ' + \
                      '{}. '.format(partlocs.shape[0]) + \
                      'A single particleid has been provided.'
                raise TypeError(msg)
            elif isinstance(particleids, (list, tuple)):
                particleids = np.array(particleids, dtype=np.int32)
            if isinstance(particleids, np.ndarray):
                if particleids.shape[0] != partlocs.shape[0]:
                    msg = 'shape of particleids ' + \
                          '({}) '.format(particleids.shape[0]) + \
                          'is not equal to the shape ' + \
                          'of partlocs ({}).'.format(partlocs.shape[0])
                    raise ValueError(msg)

        # create empty particle
        ncells = partlocs.shape[0]
        particledata = ParticleGroup.get_particledata_empty(ncells=ncells,
                                                            structured=structured,
                                                            particleid=particleid)
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

        # return particle instance
        return particledata


class _ParticleGroupTemplate(_Modpath7ParticleGroup):
    """
    Base particle group template that defines all data for particle
    group items 1 through 6. _ParticleGroupTemplate should not be
    called directly.

    """

    def __init__(self, particlegroupname, filename,
                 releasedata):
        """
        Base class constructor

        """
        # instantiate base class
        _Modpath7ParticleGroup.__init__(self, particlegroupname, filename,
                                        releasedata)

    def write(self, fp=None, ws='.'):
        """

        Parameters
        ----------
        fp : fileobject
            Fileobject that is open with write access
        ws : str
            Workspace for particle data

        Returns
        -------

        """
        return


class ParticleGroupNodeTemplate(_ParticleGroupTemplate):
    """
    Node particle template class to create MODPATH 7 particle location
    input style 3. Particle locations for this template are specified
    by nodes.


    Parameters
    ----------
    particlegroupname : str
       Name of particle group
    filename : str
        Name of the external file that will contain the particle data.
        If filename is '' or None the particle information for the
        particle group will be written to the MODPATH7 simulation
        file.
    releasedata : float, int, list, or tuple
        If releasedata is a float or an int or a list/tuple with a single
        float or int, releaseoption is set to 1 and release data is the
        particle release time.
    particledata : list of ParticleNodeData or ParticleCellNodeData objects
        List or tuple containing ParticleFaceNodeData or ParticleCellNodeData
        objects with input style 3 face and/or node particles


    Returns
    -------

    """

    def __init__(self, particlegroupname='PG1', filename=None,
                 releasedata=(0.0,),
                 particledata=None):
        """
        Class constructor

        """

        # instantiate base class
        _ParticleGroupTemplate.__init__(self, particlegroupname, filename,
                                        releasedata)
        # validate particledata
        if particledata is None:
            msg = 'FaceNode: valid ParticleNodeData item must be passed'
            raise ValueError(msg)

        if isinstance(particledata, (NodeParticleDataFace,
                                     NodeParticleDataCell)):
            particledata = [particledata]

        totalcellcount = 0
        for idx, td in enumerate(particledata):
            if not isinstance(td, (NodeParticleDataFace,
                                   NodeParticleDataCell)):
                msg = 'FaceNode: valid ParticleNodeData or ' + \
                      'ParticlecCellData item must be passed' + \
                      'for particledata item {}'.format(idx)
                raise ValueError(msg)
            totalcellcount += td.templatecellcount

        self.inputstyle = 3
        self.particletemplatecount = len(particledata)
        self.totalcellcount = totalcellcount
        self.particledata = particledata

    def write(self, fp=None, ws='.'):
        """

        Parameters
        ----------
        fp : fileobject
            Fileobject that is open with write access
        ws : str
            Workspace for particle data

        Returns
        -------

        """
        # validate that a valid file object was passed
        if not hasattr(fp, 'write'):
            msg = 'FaceNode: cannot write data for template ' + \
                  'without passing a valid file object ({}) '.format(fp) + \
                  'open for writing'
            raise ValueError(msg)

        # call base class write method to write common data
        _Modpath7ParticleGroup.write(self, fp, ws)

        # open external file if required
        if self.external:
            fpth = os.path.join(ws, self.filename)
            f = open(fpth, 'w')
        else:
            f = fp

        # item 1
        f.write('{}\n'.format(self.inputstyle))

        # item 2
        f.write('{} {}\n'.format(self.particletemplatecount,
                                 self.totalcellcount))
        # items 3, 4 or 5, and 6
        for td in self.particledata:
            td.write(f)

        # close the external file
        if self.external:
            f.close()

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
    >>> pf = flopy.modpath.ParticleFaceNodeData(nodes=[100, 101])

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

        # validate nodes
        if not isinstance(nodes, np.ndarray):
            if isinstance(nodes, (int, np.int32, np.int64)):
                nodes = np.array([nodes], dtype=np.int32)
            elif isinstance(nodes, (list, tuple)):
                nodes = np.array(nodes, dtype=np.int32)
            else:
                msg = 'ParticleFaceNodeData: node data must be a ' + \
                      'integer, list of integers, or tuple of integers'
                raise TypeError(msg)

        # validate shape of nodes
        templatecellcount = nodes.shape[0]
        if len(nodes.shape) > 1:
            msg = 'ParticleFaceNodeData: processed node data must be a ' + \
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
            msg = 'ParticleFaceNodeData: cannot write data for template ' + \
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
    >>> pc = flopy.modpath.ParticleCellNodeData(nodes=[100, 101])

    """

    def __init__(self, drape=0,
                 columncelldivisions=3, rowcelldivisions=3,
                 layercelldivisions=3, nodes=(0,)):
        """
        Class constructor

        """

        # validate nodes
        if not isinstance(nodes, np.ndarray):
            if isinstance(nodes, (int, np.int32, np.int64)):
                nodes = np.array([nodes], dtype=np.int32)
            elif isinstance(nodes, (list, tuple)):
                nodes = np.array(nodes, dtype=np.int32)
            else:
                msg = 'ParticleCellNodeData: node data must be a integer, ' + \
                      'list of integers or tuple of integers'
                raise TypeError(msg)

        # validate shape of nodes
        templatecellcount = nodes.shape[0]
        if len(nodes.shape) > 1:
            msg = 'ParticleCellNodeData: processed node data must be a ' + \
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
            msg = 'ParticleCellNodeData: cannot write data for template ' + \
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
