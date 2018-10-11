import numpy as np
import collections

from ..utils.utils_def import FlopyBinaryData
from ..discretization.structuredgrid import StructuredGrid


class MfGrdFile(FlopyBinaryData):

    def __init__(self, filename, precision='double', verbose=False):
        """
        Class constructor.

        """


        super(MfGrdFile, self).__init__()
        self.set_float(precision=precision)
        self.verbose = verbose
        self._initial_len = 50
        self._recorddict = collections.OrderedDict()
        self._datadict = collections.OrderedDict()
        self._recordkeys = []

        if self.verbose:
            print('\nProcessing binary grid file: {}'.format(filename))
        self.file = open(filename, 'rb')
        """
        # read header information
        GRID DISV
        VERSION 1
        NTXT 13
        LENTXT 100
        """

        # grid type
        line = self.read_text(self._initial_len).strip()
        t = line.split()
        self._grid = t[1]

        # version
        line = self.read_text(self._initial_len).strip()
        t = line.split()
        self._version = t[1]

        # version
        line = self.read_text(self._initial_len).strip()
        t = line.split()
        self._ntxt = int(t[1])

        # length of text
        line = self.read_text(self._initial_len).strip()
        t = line.split()
        self._lentxt = int(t[1])

        # read text strings
        for idx in range(self._ntxt):
            line = self.read_text(self._lentxt).strip()
            t = line.split()
            key = t[0]
            dt = t[1]
            if dt == 'INTEGER':
                dtype = np.int32
            elif dt == 'SINGLE':
                dtype = np.float32
            elif dt == 'DOUBLE':
                dtype = np.float64
            else:
                dtype = None
            nd = int(t[3])
            if nd > 0:
                shp = [int(v) for v in t[4:]]
                shp = tuple(shp[::-1])
            else:
                shp = (0,)
            self._recorddict[key] = (dtype, nd, shp)
            self._recordkeys.append(key)
            if self.verbose:
                s = ''
                if nd > 0:
                    s = shp
                print('  File contains data for {} with shape {}'.format(key, s))

        if self.verbose:
            print('Attempting to read {} records from {}'.format(self._ntxt,
                                                                 filename))

        for key in self._recordkeys:
            if self.verbose:
                print('  Reading {}'.format(key))
            dt, nd, shp = self._recorddict[key]
            # read array data
            if nd > 0:
                count = 1
                for v in shp:
                    count *= v
                v = self.read_record(count=count, dtype=dt)
            # read variable data
            else:
                if dt == np.int32:
                    v = self.read_integer()
                elif dt == np.float32:
                    v = self.read_real()
                elif dt == np.float64:
                    v = self.read_real()
            self._datadict[key] = v

        # set the model grid
        self.mg = self._set_modelgrid()

    def _set_modelgrid(self):
        try:
            if self._grid == 'DISV':
                mg = None
            elif self._grid == 'DIS':
                nlay, nrow, ncol = self._datadict["NLAY"], self._datadict["NROW"], self._datadict["NCOL"]
                delr, delc = self._datadict['DELR'], self._datadict['DELC']
                top, botm = self._datadict['TOP'], self._datadict['BOTM']
                top.shape = (nrow, ncol)
                botm.shape = (nlay, nrow, ncol)
                xorigin, yorigin, rot = self._datadict['XORIGIN'], \
                                        self._datadict['YORIGIN'], \
                                        self._datadict['ANGROT']
                mg = StructuredGrid(delc, delr, top, botm, xoff=xorigin,
                                    yoff=yorigin, angrot=rot)
            else:
                mg = None
        except:
            mg = None
            print('could not set spatial reference for {}'.format(self.file.name))

        return mg

    def get_centroids(self):
        x, y = None, None
        try:
            if self._grid in ['DISV', 'DISU']:
                x = self._datadict['CELLX']
                y = self._datadict['CELLY']
            elif self._grid == 'DIS':
                nlay = self._datadict['NLAY']
                x = np.tile(self.mg.xcell_centers().flatten(), nlay)
                y = np.tile(self.mg.xcell_centers().flatten(), nlay)
        except:
            print('could not return centroids' +
                  ' for {}'.format(self.file.name))
        return np.column_stack((x, y))

    def get_verts(self):
        if self._grid == 'DISV':
            try:
                iverts = []
                iavert = self._datadict['IAVERT']
                javert = self._datadict['JAVERT']
                shpvert = self._recorddict['VERTICES'][2]
                for ivert in range(self._datadict['NCPL']):
                    i0 = iavert[ivert] - 1
                    i1 = iavert[ivert+1] - 1
                    iverts.append((javert[i0:i1]-1).tolist())
                if self.verbose:
                    print('returning vertices for {}'.format(self.file.name))
                return iverts, self._datadict['VERTICES'].reshape(shpvert)
            except:
                print('could not return vertices for {}'.format(self.file.name))
        elif self._grid == 'DISU':
            try:
                iverts = []
                iavert = self._datadict['IAVERT']
                javert = self._datadict['JAVERT']
                shpvert = self._recorddict['VERTICES'][2]
                for ivert in range(self._datadict['NODES']):
                    i0 = iavert[ivert] - 1
                    i1 = iavert[ivert + 1] - 1
                    iverts.append((javert[i0:i1] - 1).tolist())
                if self.verbose:
                    print('returning vertices for {}'.format(
                        self.file.name))
                return iverts, self._datadict['VERTICES'].reshape(
                    shpvert)
            except:
                print('could not return vertices for {}'.format(
                    self.file.name))
        elif self._grid == 'DIS':
            try:
                nlay, nrow, ncol = self._datadict['NLAY'], \
                                   self._datadict['NROW'], \
                                   self._datadict['NCOL']
                iv = 0
                verts = []
                iverts = []
                for k in range(nlay):
                    for i in range(nrow):
                        for j in range(ncol):
                            ivlist = []
                            v = self.mg.get_cell_vertices(i, j)
                            for (x, y) in v:
                                verts.append((x, y))
                                ivlist.append(iv)
                                iv += 1
                            iverts.append(ivlist)
                verts = np.array(verts)
                return iverts, verts
            except:
                print('could not return vertices for {}'.format(self.file.name))
        return



