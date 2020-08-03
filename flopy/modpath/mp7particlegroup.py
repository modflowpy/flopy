"""
mp7particlegroup module.  Contains the ParticleGroup, and
    ParticleGroupNodeTemplate classes.


"""

import os
import numpy as np
from ..utils.util_array import Util2d
from .mp7particledata import ParticleData, NodeParticleData


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
        if filename == "":
            filename = None
        self.filename = filename
        if self.filename is None:
            self.external = False
        else:
            self.external = True

        if releasedata is None:
            msg = (
                "releasedata must be provided to instantiate "
                + "a MODPATH 7 particle group"
            )
            raise ValueError(msg)

        # convert releasedata to a list, if required
        if isinstance(releasedata, (float, int)):
            releasedata = [releasedata]
        elif isinstance(releasedata, np.ndarray):
            releasedata = releasedata.tolist()

        # validate that releasedata is a list or tuple
        if not isinstance(releasedata, (list, tuple)):
            msg = "releasedata must be a float, int, list, or tuple"
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
            if isinstance(releasedata[1], list) or isinstance(
                releasedata[1], tuple
            ):
                releasedata[1] = np.array(releasedata[1])
            if releasedata[1].shape[0] != releasetimecount:
                msg = (
                    "The number of releasetimes data "
                    + "({}) ".format(releasedata[1].shape[0])
                    + "is not equal to releasetimecount "
                    + "({}).".format(releasetimecount)
                )
                raise ValueError(msg)
            releasetimes = np.array(releasedata[1], dtype=np.float32)
        else:
            msg = "releasedata must have 1, 2, or 3 entries"
            raise ValueError(msg)

        # set release data
        self.releaseoption = releaseoption
        self.releasetimecount = releasetimecount
        self.releaseinterval = releaseinterval
        self.releasetimes = releasetimes

    def write(self, fp=None, ws="."):
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
        if not hasattr(fp, "write"):
            msg = (
                "Cannot write data for particle group "
                + "{} ".format(self.particlegroupname)
                + "without passing a valid file object ({}) ".format(fp)
                + "open for writing"
            )
            raise ValueError(msg)

        # item 26
        fp.write("{}\n".format(self.particlegroupname))

        # item 27
        fp.write("{}\n".format(self.releaseoption))

        if self.releaseoption == 1:
            # item 28
            fp.write("{}\n".format(self.releasetimes[0]))
        elif self.releaseoption == 2:
            # item 29
            fp.write(
                "{} {} {}\n".format(
                    self.releasetimecount,
                    self.releasetimes[0],
                    self.releaseinterval,
                )
            )
        elif self.releaseoption == 3:
            # item 30
            fp.write("{}\n".format(self.releasetimecount))
            # item 31
            tp = self.releasetimes
            v = Util2d(
                self, (tp.shape[0],), np.float32, tp, name="temp", locat=0
            )
            fp.write(v.string)

        # item 32
        if self.external:
            line = "EXTERNAL {}\n".format(self.filename)
        else:
            line = "INTERNAL\n"
        fp.write(line)

        return


class ParticleGroup(_Modpath7ParticleGroup):
    """
    ParticleGroup class to create MODPATH 7 particle group data for location
    input style 1. Location input style 1 is the most general type of particle
    group that requires the user to define the location of all particles and
    associated data (relative release time, drape, and optionally particle
    ids). Particledata locations can be specified by layer, row, column
    (locationstyle=1) or nodes (locationstyle=2) and are created with the
    ParticleData class.

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
    particledata : ParticleData
        ParticleData instance with particle data. If particledata is None,
        a ParticleData instance will be created with a node-based particle
        in the center of the first node in the model (default is None).

    Examples
    --------

    >>> import flopy
    >>> p = [(2, 0, 0), (0, 20, 0)]
    >>> p = flopy.modpath.ParticleData(p)
    >>> pg = flopy.modpath.ParticleGroup(particledata=p)

    """

    def __init__(
        self,
        particlegroupname="PG1",
        filename=None,
        releasedata=0.0,
        particledata=None,
    ):
        """
        Class constructor

        """

        # instantiate base class
        _Modpath7ParticleGroup.__init__(
            self, particlegroupname, filename, releasedata
        )
        self.name = "ParticleGroup"

        # create default node-based particle data if not passed
        if particledata is None:
            particledata = ParticleData(structured=False)

        # convert particledata to a list if a ParticleData type
        if not isinstance(particledata, ParticleData):
            msg = "{}: particledata must be a".format(
                self.name
            ) + " ParticleData instance not a {}".format(type(particledata))
            raise TypeError(msg)

        # set attributes
        self.inputstyle = 1
        self.particlecount = particledata.particlecount
        self.particleidoption = particledata.particleidoption
        self.locationstyle = particledata.locationstyle
        self.particledata = particledata

        return

    def write(self, fp=None, ws="."):
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
            f = open(fpth, "w")
        else:
            f = fp

        # particle data item 1
        f.write("{}\n".format(self.inputstyle))

        # particle data item 2
        f.write("{}\n".format(self.locationstyle))

        # particle data item 3
        f.write("{} {}\n".format(self.particlecount, self.particleidoption))

        # particle data item 4 and 5
        # call the write method in ParticleData
        self.particledata.write(f=f)

        # close the external file
        if self.external:
            f.close()

        return


class _ParticleGroupTemplate(_Modpath7ParticleGroup):
    """
    Base particle group template that defines all data for particle
    group items 1 through 6. _ParticleGroupTemplate should not be
    called directly.

    """

    def __init__(self, particlegroupname, filename, releasedata):
        """
        Base class constructor

        """
        # instantiate base class
        _Modpath7ParticleGroup.__init__(
            self, particlegroupname, filename, releasedata
        )

    def write(self, fp=None, ws="."):
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


class ParticleGroupLRCTemplate(_ParticleGroupTemplate):
    """
    Layer, row, column particle template class to create MODPATH 7 particle
    location input style 2. Particle locations for this template are specified
    by layer, row, column regions.

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
    particledata :
        LRCParticleData object with input style 2 face and/or node particle
        data. If particledata is None a default LRCParticleData object is
        created (default is None).


    Returns
    -------

    """

    def __init__(
        self,
        particlegroupname="PG1",
        filename=None,
        releasedata=(0.0,),
        particledata=None,
    ):
        """
        Class constructor

        """
        self.name = "ParticleGroupLRCTemplate"

        # instantiate base class
        _ParticleGroupTemplate.__init__(
            self, particlegroupname, filename, releasedata
        )
        # validate particledata
        if particledata is None:
            particledata = NodeParticleData()

        self.inputstyle = 2
        self.particledata = particledata

    def write(self, fp=None, ws="."):
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
        if not hasattr(fp, "write"):
            msg = (
                "{}: cannot write data for ".format(self.name)
                + "template without passing a valid file object "
                + "({}) ".format(fp)
                + "open for writing"
            )
            raise ValueError(msg)

        # call base class write method to write common data
        _Modpath7ParticleGroup.write(self, fp, ws)

        # open external file if required
        if self.external:
            fpth = os.path.join(ws, self.filename)
            f = open(fpth, "w")
        else:
            f = fp

        # item 1
        f.write("{}\n".format(self.inputstyle))

        # items 2, 3, 4 or 5, and 6
        self.particledata.write(f)

        # close the external file
        if self.external:
            f.close()

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
    particledata :
        NodeParticleData object with input style 3 face and/or node particle
        data. If particledata is None a default NodeParticleData object is
        created (default is None).


    Returns
    -------

    """

    def __init__(
        self,
        particlegroupname="PG1",
        filename=None,
        releasedata=(0.0,),
        particledata=None,
    ):
        """
        Class constructor

        """
        self.name = "ParticleGroupNodeTemplate"

        # instantiate base class
        _ParticleGroupTemplate.__init__(
            self, particlegroupname, filename, releasedata
        )
        # validate particledata
        if particledata is None:
            particledata = NodeParticleData()

        self.inputstyle = 3
        self.particledata = particledata

    def write(self, fp=None, ws="."):
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
        if not hasattr(fp, "write"):
            msg = (
                "{}: cannot write data for ".format(self.name)
                + "template without passing a valid file object "
                + "({}) ".format(fp)
                + "open for writing"
            )
            raise ValueError(msg)

        # call base class write method to write common data
        _Modpath7ParticleGroup.write(self, fp, ws)

        # open external file if required
        if self.external:
            fpth = os.path.join(ws, self.filename)
            f = open(fpth, "w")
        else:
            f = fp

        # item 1
        f.write("{}\n".format(self.inputstyle))

        # items 2, 3, 4 or 5, and 6
        self.particledata.write(f)

        # close the external file
        if self.external:
            f.close()

        return
