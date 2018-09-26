import os
import numpy as np
from ..utils.recarray_utils import create_empty_recarray

try:
    from numpy.lib import NumpyVersion

    numpy114 = NumpyVersion(np.__version__) >= '1.14.0'
except ImportError:
    numpy114 = False


class Modpath7Particle(object):
    def __init__(self, ParticleGroupName, filename, ReleaseData):
        self.ParticleGroupName = ParticleGroupName
        if filename == '':
            filename = None
        self.filename = filename

        if ReleaseData is None:
            msg = 'ReleaseData must be provided to instantiate ' + \
                  'a MODPATH 7 particle group'
            raise ValueError(msg)

        if isinstance(ReleaseData, float) or isinstance(ReleaseData, int):
            ReleaseData = [ReleaseData]
        else:
            # validate that ReleaseData is a list or tuple
            if not isinstance(ReleaseData, list) \
                    and not isinstance(ReleaseData, tuple):
                msg = 'ReleaseData must be a list or tuple'
                raise ValueError(msg)
            # process ReleaseData
            if len(ReleaseData) == 1:
                ReleaseOption = 1
                ReleaseTimeCount = 0
                ReleaseInterval = 0
                ReleaseTimes = np.array(ReleaseData, dtype=np.float32)
            elif len(ReleaseData) == 3:
                ReleaseOption = 2
                ReleaseTimeCount = int(ReleaseData[0])
                ReleaseInterval = 0
                ReleaseTimes = np.array(ReleaseData[1], dtype=np.float32)
                ReleaseInterval = int(ReleaseData[2])
            elif len(ReleaseData) == 2:
                ReleaseOption = 3
                ReleaseTimeCount = int(ReleaseData[0])
                # convert ReleaseTimes list or tuple to a numpy array
                if isinstance(ReleaseData[1], list) \
                        or isinstance(ReleaseData[1], tuple):
                    ReleaseData[1] = np.array(ReleaseData[1])
                if ReleaseData[1].shape[0] != ReleaseTimeCount:
                    msg = 'The number of ReleaseTimes data ' + \
                          '({}) '.format(ReleaseData[1].shape[0]) + \
                          'is not equal to ReleaseTimeCount ' + \
                          '({}).'.format(ReleaseTimeCount)
                    raise ValueError(msg)
                ReleaseTimes = np.array(ReleaseData[1], dtype=np.float32)
        # set release data
        self.ReleaseOption = ReleaseOption
        self.ReleaseTimeCount = ReleaseTimeCount
        self.ReleaseInterval = ReleaseInterval
        self.ReleaseTimes = ReleaseTimes

    def write(self, fp=None, ws='.'):
        return


class LayerRowColumnParticles(Modpath7Particle):
    def __init__(self, ParticleGroupName='PG1', filename=None,
                 ReleaseData=[0.0],
                 ParticleData=[[0, 0, 0, 0.5, 0.5, 0.5, 0., 0]]):

        # instantiate base class
        Modpath7Particle.__init__(self, ParticleGroupName, filename,
                                  ReleaseData)
        self.InputStyle = 1
        # convert list, tuples, and numpy array to
        v = None
        dtypein = None
        if isinstance(ParticleData, list) or isinstance(ParticleData, tuple):
            ncells = len(ParticleData)
            v = ParticleData[0]
        elif isinstance(ParticleData, np.ndarray):
            dtypein = ParticleData.dtype
            ncells = ParticleData.shape[0]
            v = ParticleData[0].tolist()
        else:
            msg = 'ParticleData must be a list, tuple, ' + \
                  'numpy ndarray, or an instance of ' + \
                  'LayerRowColumnParticles'
            raise ValueError(msg)
        if v is not None:
            if len(v) == 6:
                ParticleIdOption = 0
                LocationStyle = 2
                partid = False
                structured = False
            elif len(v) == 7:
                ParticleIdOption = 1
                LocationStyle = 2
                partid = True
                structured = False
            elif len(v) == 8:
                ParticleIdOption = 0
                LocationStyle = 1
                partid = False
                structured = True
            elif len(v) == 9:
                ParticleIdOption = 1
                LocationStyle = 1
                partid = True
                structured = True
            else:
                msg = 'ParticleData should have 6, 7, 8, or 9 columns.' + \
                      'Specified data for particle group ' + \
                      '{} '.format(ParticleGroupName) + \
                      'only has {} columns.'.format(ncells)
                raise ValueError(msg)
            dtype = self.get_default_dtype(structured=structured,
                                           particleid=partid)
            if dtypein is None:
                dtypein = dtype
            if dtype != dtypein:
                ParticleData = np.array(ParticleData, dtype=dtype)

        # set attributes
        self.ParticleIdOption = ParticleIdOption
        self.LocationStyle = LocationStyle
        self.ParticleData = ParticleData

        return

    def write(self, fp=None, ws='.'):
        # validate that a valid file object was passed
        if not hasattr(fp, 'write'):
            msg = 'Cannot write data for particle group ' + \
                  '{} '.format(self.ParticleGroupName) + \
                  'without passing a valid file object ({}) '.format(fp) + \
                  'open for writing'
            raise ValueError(msg)

        # item 26
        fp.write('{}\n'.format(self.ParticleGroupName))

        # item 27
        fp.write('{}\n'.format(self.ReleaseOption))

        if self.ReleaseOption == 1:
            # item 28
            fp.write('{}\n'.format(self.ReleaseTimes[0]))
        elif self.ReleaseOption == 2:
            # item 29
            fp.write('{} {} {}\n'.format(self.ReleaseTimeCount,
                                         self.ReleaseTimes[0],
                                         self.ReleaseInterval))
        elif self.ReleaseOption == 3:
            # item 30
            fp.write('{}\n'.format(self.ReleaseTimeCount))
            # item 31
            tp = self.ReleaseTimes
            v = Util2d(self.parent, (tp.shape[0],),
                       np.float32, tp,
                       name='temp',
                       locat=self.unit_number[0])
            fp.write(v.string)

        # item 32
        if self.filename is not None:
            fp.write('EXTERNAL {}\n'.format(self.filename))
            fpth = os.path.join(ws, self.filename)
            f = open(fpth, 'w')
        else:
            fp.write('INTERNAL\n')
            f = fp

        # particle data item 1
        f.write('{}\n'.format(self.InputStyle))

        # particle data item 2
        f.write('{}\n'.format(self.LocationStyle))

        # particle data item 3
        f.write('{} {}\n'.format(self.ParticleData.shape[0],
                                 self.ParticleIdOption))

        # particle data item 4 and 5
        d = np.recarray.copy(self.ParticleData)
        lnames = [name.lower() for name in d.dtype.names]
        # Add one to the kij and node indices
        for idx in ['k', 'i', 'j', 'node', 'id']:
            if idx in lnames:
                d[idx] += 1
        # save the particle data
        np.savetxt(f, d, fmt=self.fmt_string, delimiter='')

        if self.filename is not None:
            f.close()

        return

    @property
    def fmt_string(self):
        """Returns a C-style fmt string for numpy savetxt that corresponds to
        the dtype"""
        fmts = []
        for field in self.ParticleData.dtype.descr:
            vtype = field[1][1].lower()
            if vtype == 'i' or vtype == 'b':
                fmts.append('%9d')
            elif vtype == 'f':
                if numpy114:
                    # Use numpy's floating-point formatter (Dragon4)
                    fmts.append('%15s')
                else:
                    fmts.append('%15.7E')
            elif vtype == 'o':
                fmts.append('%9s')
            elif vtype == 's':
                msg = "LayerRowColumnParticles.fmt_string error: 'str' " + \
                      "type found in dtype. This gives unpredictable " + \
                      "results when recarray to file - change to 'object' type"
                raise TypeError(msg)
            else:
                msg = "MfList.fmt_string error: unknown vtype in " + \
                      "field: {}".format(field)
                raise TypeError(msg)
        return ' ' + ' '.join(fmts)

    @staticmethod
    def get_default_dtype(structured=True, particleid=False):
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
    def get_empty(ncells=0, structured=True, particleid=False):
        # get an empty recarray that corresponds to dtype
        dtype = LayerRowColumnParticles.get_default_dtype(
            structured=structured,
            particleid=particleid)
        return create_empty_recarray(ncells, dtype, default_value=0)
