import numpy as np
from pylab import find

class BinaryReader(object):
    'Generic class for reading binary output from MODFLOW/MT3DMS models'
    def __init__(self, parent, compiler='gfortran',double_precision=False):
        self.parent = parent # To be able to access the parent modflow object's attributes
        self.byte_skip = 0
        if (compiler[0] == 'g'): # gfortran
            self.byte_skip = 2
        elif (compiler[0] == 'i'): # Intel, still to check!
            self.byte_skip = 0
        elif (compiler[0] == 'l'): # LF95
            self.byte_skip = 4
        self.eof = False
        self.fp = None
        if double_precision:
            self.float_format = np.float64
        else:
            self.float_format = np.float32
        self.int_format = np.int32
        self.header_structure = np.dtype([('kstep', self.int_format), ('kper', self.int_format), ('pertim', self.float_format), ('totim', self.float_format), ('text', np.uint8, 16), ('ncol', self.int_format), ('nrow', self.int_format), ('ilay)', self.int_format)])
        self.message = 'records'
        self.nrow, self.ncol, self.nlay, self.nper = self.parent.nrow_ncol_nlay_nper
        self.silent = self.parent.silent
    def __iter__(self):
        return self
    def next(self):
        if (self.eof):
            raise StopIteration

        t = 0.
        if (isinstance(self, ModflowCbcRead)):
            data = []
        else:
            data = np.empty((self.nrow, self.ncol, self.nlay))
        strings = []

        prev_ntrans = 9999
        prev_kstep = 9999
        prev_kper = 9999

        while (True): # infinite loop
            ntrans, kstep, kper, pertim, totim, text, ncol, nrow, ilay, success = self.read_header()
            self.eof = not success
            if ((ntrans > prev_ntrans) or (kstep > prev_kstep) or (kper > prev_kper) or (self.eof)):
                self.rewind()
                return t, data, strings
            t = totim
            if (not (text in strings)):
                strings = strings + [text]
            if (isinstance(self, ModflowCbcRead)):
                data = data + [self.read_values(self.nlay)]
            else:
                data[:, :, ilay - 1] = self.read_values(1)[ :, :, 0]

            prev_ntrans = ntrans
            prev_kstep = kstep
            prev_kper = kper
    def read_all(self, filename):
        tt = []
        dt = []
        m = []

        if self.silent == 0: print ('Please wait. Reading %s from file %s.' % (self.message, filename))
        self.fp = open(filename, 'rb')
        self.eof = False
        for tds in self:
            tt = tt + [tds[0]]
            #dt = dt + [tds[1]]
            dt.append(tds[1])
            m.append(tds[2])
        self.fp.close()

        if self.silent == 0: print str(len(dt)), ' timesteps read.'
        return tt, dt, m
    def read_header(self):
        # Initialize return values
        ntrans = 1; kstep = 1; kper = 1
        pertim = 0.; totim = 0.
        text = ''
        ncol = 1; nrow = 1; nlay = 1
        success = True
        try:
            # Skip compiler-dependent record header
            self.skip_record_header()
            # Read data as string and store in array hdr
            hdr = np.fromstring(self.fp.read(self.header_structure.itemsize), self.header_structure)[0]
            # Skip compiler-dependent record header
            self.skip_record_header()

            # Extract return values
            if (isinstance(self, ModflowHdsRead)):
                kstep = hdr[0]; kper = hdr[1]
                pertim = hdr[2]; totim = hdr[3]
                text = hdr[4].tostring()
                ncol = hdr[5]; nrow = hdr[6]; nlay = hdr[7] # is in fact ilay but for convenience we use nlay
            elif (isinstance(self, ModflowCbcRead)):
                kstep = hdr[0]; kper = hdr[1]
                text = hdr[2].tostring()
                ncol = hdr[3]; nrow = hdr[4]; nlay = hdr[5]
                # if nlay < 0 then records were savd with either UBDSV1 or UBDSV2 (see utl6.f)
                # in that case, an extra record follows with an identifier that is either 1 for
                # UBDSV1 or 2 for UBDSV2. The latter is not supported (yet?). Also, a record
                # containing delt, pertim and totim must be read in this case
                if (nlay < 0):
                    nlay = -nlay
                    # Define the record structure
                    dt = np.dtype([('ubdsv_nr', np.int16), ('kper', self.int_format), ('delt', self.float_format), ('pertim', self.float_format), ('totim)', self.float_format)])
                    # Skip compiler-dependent record header
                    self.skip_record_header()
                    # Read data as string and store in array hdr
                    hdr = np.fromstring(self.fp.read(dt.itemsize), dt)[0]
                    # Skip compiler-dependent record header
                    self.skip_record_header()
                    ubdsv_nr = hdr[0]
                    delt = hdr[1]; pertim = hdr[2]; totim = hdr[3]
                    assert ubdsv_nr == 1, 'Format of cbc file not (yet) supported'
            elif (isinstance(self, ModflowUcnRead)):
                ntrans = hdr[0]; kstep = hdr[1]; kper = hdr[2]
                totim = hdr[3]
                text = hdr[4].tostring()
                ncol = hdr[5]; nrow = hdr[6]; nlay = hdr[7] # is in fact ilay but for convenience we use nlay
                # print ntrans, kstep, kper, totim, text, ncol, nrow, nlay
            # Check grid dimensions
            assert (ncol == self.ncol), 'Number of columns incompatible with data in binary file!'
            assert (nrow == self.nrow), 'Number of rows incompatible with data in binary file!'
            assert (nlay <= self.nlay), 'Number of layers incompatible with data in binary file!'
        except:
            success = False
        finally:
            return ntrans, kstep, kper, pertim, totim, text, ncol, nrow, nlay, success
    def read_values(self, nl):
        # Skip compiler-dependent header
        self.skip_record_header()
        layer_values = np.fromfile(file = self.fp, dtype = self.float_format, count = nl * self.nrow * self.ncol)
        layer_values.shape = (nl, self.nrow, self.ncol)
        layer_values = np.swapaxes(layer_values, 0, 2)
        layer_values = np.swapaxes(layer_values, 0, 1)
        # Skip compiler-dependent footer
        self.skip_record_header()
        return layer_values
    def rewind(self):
        rewind_bytes = 2 * self.byte_skip + self.header_structure.itemsize
        self.fp.seek(-rewind_bytes, 1)
    def skip_record_header(self):
        self.fp.read(self.byte_skip)
        '''if (self.byte_skip > 0):
            dummy = np.fromfile(file = self.fp, dtype = np.int16, count = self.byte_skip)'''

class ModflowCbcRead(BinaryReader):
    def __init__(self, parent, compiler='gfortran',double_precision=False):
        print 'Deprecation Warning: ModflowCbcRead is being deprecated. Use CellBudgetFile instead'
        BinaryReader.__init__(self, parent, compiler,double_precision)
        self.header_structure = np.dtype([('kstep', self.int_format), ('kper', self.int_format), ('text', np.uint8, 16), ('ncol', self.int_format), ('nrow', self.int_format), ('nlay)', self.int_format)])
        self.message = 'cbc terms'

class ModflowHdsRead(BinaryReader):
    def __init__(self, parent, compiler='gfortran',double_precision=False):
        print 'Deprecation Warning: ModflowHdsRead is being deprecated. Use HeadFile instead'
        BinaryReader.__init__(self, parent, compiler,double_precision)
        self.header_structure = np.dtype([('kstep', self.int_format), ('kper', self.int_format), ('pertim', self.float_format), ('totim', self.float_format), ('text', np.uint8, 16), ('ncol', self.int_format), ('nrow', self.int_format), ('ilay)', self.int_format)])
        self.message = 'heads'
    def extract_time_series(self, filename, node_yxz):
        dis = self.parent.get_package('DIS')
        if (dis != None):
            yy, xx, zz = dis.get_node_coordinates()

            times, heads, strings = self.read_all(filename)

            if self.silent == 0: print ('Please wait. Extracting time series.')
            ht = np.empty((1, len(node_yxz)))
            rr = 0
            for h in heads:
                if (rr > 0):
                    ht = np.vstack((ht, np.empty(len(node_yxz))))
                cc = 0
                for yxz in node_yxz:
                    y = yxz[0]
                    x = yxz[1]
                    z = yxz[2]
                    r = find(abs(yy - y) < abs(y) / 1e6)[0]
                    c = find(abs(xx - x) < abs(x) / 1e6)[0]
                    l = find(abs(zz[:, r, c] - z) < abs(z) / 1e6)[0]
                    ht[rr, cc] = h[r, c, l]
                    cc = cc + 1
                rr = rr + 1
            return times, ht

class Mt3dUcnRead(BinaryReader):
    def __init__(self, parent, compiler='gfortran',double_precision=False):
        print 'Deprecation Warning: Mt3dUcnRead is being deprecated. Use UcnFile instead'
        BinaryReader.__init__(self, parent, compiler, double_precision)
        self.header_structure = np.dtype([('ntrans', self.int_format), ('kstep', self.int_format), ('kper', self.int_format), ('totim', self.float_format), ('text', np.uint8, 16), ('ncol', self.int_format), ('nrow', self.int_format), ('ilay)', self.int_format)])
        self.message = 'concentrations'
