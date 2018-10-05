import os
import numpy as np
from ..utils.util_array import Util2d
from ..utils.recarray_utils import create_empty_recarray


class Modpath7Particle(object):
    def __init__(self, particlegroupname, filename, releasedata):
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

        if isinstance(releasedata, float) or isinstance(releasedata, int):
            releasedata = [releasedata]
        else:
            # validate that releasedata is a list or tuple
            if not isinstance(releasedata, list) \
                    and not isinstance(releasedata, tuple):
                msg = 'releasedata must be a list or tuple'
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

        # set release data
        self.releaseoption = releaseoption
        self.releasetimecount = releasetimecount
        self.releaseinterval = releaseinterval
        self.releasetimes = releasetimes

    def write(self, fp=None, ws='.'):

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
            v = Util2d(self.parent, (tp.shape[0],),
                       np.float32, tp,
                       name='temp',
                       locat=self.unit_number[0])
            fp.write(v.string)

        # item 32
        if self.external:
            line = 'EXTERNAL {}\n'.format(self.filename)
        else:
            line = 'INTERNAL\n'
        fp.write(line)

        return


class Particles(Modpath7Particle):
    def __init__(self, particlegroupname='PG1', filename=None,
                 releasedata=[0.0],
                 particledata=[[0, 0, 0, 0.5, 0.5, 0.5, 0., 0]]):

        # instantiate base class
        Modpath7Particle.__init__(self, particlegroupname, filename,
                                  releasedata)
        self.inputstyle = 1
        # convert list, tuples, and numpy array to
        v = None
        dtypein = None
        if isinstance(particledata, list) or isinstance(particledata, tuple):
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
        if v is not None:
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
            dtype = self.get_default_dtype(structured=structured,
                                           particleid=partid)
            if dtypein is None:
                dtypein = dtype
            if dtype != dtypein:
                particledata = np.array(particledata, dtype=dtype)

        # set attributes
        self.particleidoption = particleidoption
        self.locationstyle = locationstyle
        self.particledata = particledata

        return

    def write(self, fp=None, ws='.'):

        # call base class write method to write common data
        Modpath7Particle.write(self, fp, ws)

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
        Returns a C-style fmt string for numpy savetxt that corresponds to
        the dtype
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
        dtype = Particles.get_default_dtype(
            structured=structured,
            particleid=particleid)
        return create_empty_recarray(ncells, dtype, default_value=0)

    @staticmethod
    def create_particles(v, structured=True, particleids=None,
                         localx=None, localy=None, localz=None,
                         timeoffset=None, drape=None):
        dtype = []
        if structured:
            dtype.append(('k', np.int32))
            dtype.append(('i', np.int32))
            dtype.append(('j', np.int32))
        else:
            dtype.append(('node', np.int32))
        dtype = np.dtype(dtype)
        if isinstance(v, (list, tuple)):
            # determine if the list or tuple contains lists or tuples
            alllsttup = all(isinstance(el, (list, tuple)) for el in v)
            if structured:
                if alllsttup:
                    alllen3 = all(len(el) == 3 for el in v)
                    if not alllen3:
                        msg = 'all entries of v must have 3 items ' + \
                              'for structured particle data'
                        raise ValueError(msg)
                else:
                    msg = 'v list or tuple for structured particle data ' + \
                          'should contain list or tuple entries'
                    raise ValueError(msg)
            else:
                allint = all(isinstance(el, (int, np.int32, np.int64))
                             for el in v)
                # convert to a list of tuples
                if allint:
                    t = []
                    for el in v:
                        t.append((el,))
                    v = t
                    alllsttup = all(isinstance(el, (list, tuple)) for el in v)
                if alllsttup:
                    alllen1 = all(len(el) == 1 for el in v)
                    if not alllen1:
                        msg = 'all entries of v must have 1 items ' + \
                              'for unstructured particle data'
                        raise ValueError(msg)
                else:
                    msg = 'v list or tuple for unstructured particle data ' + \
                          'should contain integers or a list or tuple ' + \
                          'with one entry'
                    raise ValueError(msg)

            # convert v composed of a lists/tuples of lists/tuples
            # to a numpy array
            v = np.array(v, dtype=dtype)
        elif isinstance(v, np.ndarray):
            dtypein = v.dtype
            if dtypein != v.dtype:
                v = np.array(v, dtype=dtype)
        else:
            msg = 'create_Particles v must be a list or ' + \
                  'tuple with lists or tuples'
            raise ValueError(msg)

        # localx
        if localx is None:
            localx = 0.5
        else:
            if isinstance(localx, (list, tuple)):
                localx = np.array(localx, dtype=np.float32)
            if isinstance(localx, np.ndarray):
                if localx.shape[0] != v.shape[0]:
                    msg = 'shape of localx ({}) '.format(localx.shape[0]) + \
                          'is not equal to the shape ' + \
                          'of v ({}).'.format(v.shape[0])
                    raise ValueError(msg)

        # localy
        if localy is None:
            localy = 0.5
        else:
            if isinstance(localy, (list, tuple)):
                localy = np.array(localy, dtype=np.float32)
            if isinstance(localy, np.ndarray):
                if localy.shape[0] != v.shape[0]:
                    msg = 'shape of localy ({}) '.format(localy.shape[0]) + \
                          'is not equal to the shape ' + \
                          'of v ({}).'.format(v.shape[0])
                    raise ValueError(msg)

        # localz
        if localz is None:
            localz = 0.5
        else:
            if isinstance(localz, (list, tuple)):
                localz = np.array(localz, dtype=np.float32)
            if isinstance(localz, np.ndarray):
                if localz.shape[0] != v.shape[0]:
                    msg = 'shape of localz ({}) '.format(localz.shape[0]) + \
                          'is not equal to the shape ' + \
                          'of v ({}).'.format(v.shape[0])
                    raise ValueError(msg)
        # timeoffset
        if timeoffset is None:
            timeoffset = 0.
        else:
            if isinstance(timeoffset, (list, tuple)):
                timeoffset = np.array(timeoffset, dtype=np.float32)
            if isinstance(timeoffset, np.ndarray):
                if timeoffset.shape[0] != v.shape[0]:
                    msg = 'shape of timeoffset ' + \
                          '({}) '.format(timeoffset.shape[0]) + \
                          'is not equal to the shape ' + \
                          'of v ({}).'.format(v.shape[0])
                    raise ValueError(msg)

        # drape
        if drape is None:
            drape = 0
        else:
            if isinstance(drape, (list, tuple)):
                drape = np.array(drape, dtype=np.int32)
            if isinstance(drape, np.ndarray):
                if drape.shape[0] != v.shape[0]:
                    msg = 'shape of drape ({}) '.format(drape.shape[0]) + \
                          'is not equal to the shape ' + \
                          'of v ({}).'.format(v.shape[0])
                    raise ValueError(msg)

        # particleids
        if particleids is None:
            particleid = False
        else:
            particleid = True
            if isinstance(particleids, (list, tuple)):
                particleids = np.array(particleids, dtype=np.int32)
            if isinstance(particleids, np.ndarray):
                if particleids.shape[0] != v.shape[0]:
                    msg = 'shape of particleids ' + \
                          '({}) '.format(particleids.shape[0]) + \
                          'is not equal to the shape ' + \
                          'of v ({}).'.format(v.shape[0])
                    raise ValueError(msg)

        # create empty particle
        part = Particles.get_empty(ncells=v.shape[0],
                                   structured=structured,
                                   particleid=particleid)
        # fill part
        if structured:
            part['k'] = v['k']
            part['i'] = v['i']
            part['j'] = v['j']
        else:
            part['node'] = v['node']
        part['localx'] = localx
        part['localy'] = localy
        part['localz'] = localz
        part['timeoffset'] = timeoffset
        part['drape'] = drape
        if particleid:
            part['id'] = particleids
        # return particle instance
        return part


class _ParticleTemplate(Modpath7Particle):
    def __init__(self, particlegroupname, filename,
                 releasedata):
        # instantiate base class
        Modpath7Particle.__init__(self, particlegroupname, filename,
                                  releasedata)

    def write(self, fp=None, ws='.'):
        return


class NodeParticleTemplate(_ParticleTemplate):
    def __init__(self, particlegroupname='PG1', filename=None,
                 releasedata=[0.0],
                 particledata=None):

        # instantiate base class
        _ParticleTemplate.__init__(self, particlegroupname, filename,
                                   releasedata)
        # validate particledata
        if particledata is None:
            msg = 'FaceNode: valid ParticleNodeData item must be passed'
            raise ValueError(msg)

        if isinstance(particledata, (ParticleNodeData, ParticleCellData)):
            particledata = [particledata]

        totalcellcount = 0
        for idx, td in enumerate(particledata):
            if not isinstance(td, (ParticleNodeData, ParticleCellData)):
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
        # validate that a valid file object was passed
        if not hasattr(fp, 'write'):
            msg = 'FaceNode: cannot write data for template ' + \
                  'without passing a valid file object ({}) '.format(fp) + \
                  'open for writing'
            raise ValueError(msg)

        # call base class write method to write common data
        Modpath7Particle.write(self, fp, ws)

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


class ParticleNodeData(object):
    def __init__(self, drape=0,
                 verticaldivisions1=3, horizontaldivisions1=3,
                 verticaldivisions2=3, horizontaldivisions2=3,
                 verticaldivisions3=3, horizontaldivisions3=3,
                 verticaldivisions4=3, horizontaldivisions4=3,
                 rowdivisions5=3, columndivisons5=3,
                 rowdivisions6=3, columndivisions6=3,
                 nodes=[0]):

        # validate nodes
        if not isinstance(nodes, np.ndarray):
            if isinstance(nodes, (int, np.int32, np.int64)):
                nodes = np.array([nodes], dtype=np.int32)
            elif isinstance(nodes, (list, tuple)):
                nodes = np.array(nodes, dtype=np.int32)
            else:
                msg = 'ParticleNodeData: node data must be a integer, ' + \
                      'list of integers or tuple of integers'
                raise TypeError(msg)

        # validate shape of nodes
        templatecellcount = nodes.shape[0]
        if len(nodes.shape) > 1:
            msg = 'ParticleNodeData: processed node data must be a ' + \
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
        # validate that a valid file object was passed
        if not hasattr(f, 'write'):
            msg = 'ParticleNodeData: cannot write data for template ' + \
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


class ParticleCellData(object):
    def __init__(self, drape=0,
                 columncelldivisions=3, rowcelldivisions=3,
                 layercelldivisions=3, nodes=[0]):

        # validate nodes
        if not isinstance(nodes, np.ndarray):
            if isinstance(nodes, (int, np.int32, np.int64)):
                nodes = np.array([nodes], dtype=np.int32)
            elif isinstance(nodes, (list, tuple)):
                nodes = np.array(nodes, dtype=np.int32)
            else:
                msg = 'ParticleCellData: node data must be a integer, ' + \
                      'list of integers or tuple of integers'
                raise TypeError(msg)

        # validate shape of nodes
        templatecellcount = nodes.shape[0]
        if len(nodes.shape) > 1:
            msg = 'ParticleCellData: processed node data must be a ' + \
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
        # validate that a valid file object was passed
        if not hasattr(f, 'write'):
            msg = 'ParticleCellData: cannot write data for template ' + \
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
