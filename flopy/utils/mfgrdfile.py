import numpy as np
import collections

from ..utils.utils_def import FlopyBinaryData


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
            print('read {} records from {}'.format(self._ntxt, filename))

        for key in self._recordkeys:
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

    def get_verts(self):
        if self._grid == 'DISV':
            try:
                iverts = []
                iavert = self._datadict['IAVERT']
                javert = self._datadict['JAVERT']
                shpvert = self._recorddict['VERTICES'][2]
                for ivert in range(self._datadict['NCELLS']):
                    i0 = iavert[ivert] - 1
                    i1 = iavert[ivert+1] - 1
                    iverts.append((javert[i0:i1]-1).tolist())
                if self.verbose:
                    print('returning vertices for {}'.format(self.file.name))
                return iverts, self._datadict['VERTICES'].reshape(shpvert)
            except:
                print('could not return vertices for {}'.format(self.file.name))




