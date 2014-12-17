import numpy as np
from flopy.utils.mfreadbinaries import BinaryReader

class Mt3dPsRead(BinaryReader):
    'Reads binary output from MT3DMS models'
    def __init__(self, parent, compiler='gfortran'):
        BinaryReader.__init__(self, parent, compiler)
        self.header_structure = np.dtype([('obsnam', np.uint8, 12), ('TimeObs', np.float), ('ObsVal', np.float)])
        self.message = 'observations'
    def next(self):
        if (self.eof):
            raise StopIteration

        obsnam, TimeObs, ObsVal, success = self.read_header()
        self.eof = not success
        if (not self.eof):
            return obsnam, TimeObs, ObsVal
    def read_all(self, filename):
        tt = []
        dt = []
        nt = []

        print ('Please wait. Reading %s from file %s.' % (self.message, filename))
        self.fp = open(filename, 'rb')
        # Skip compiler-dependent record header
        self.skip_record_header()
        # Read data as string and store in array hdr
        dummy = np.fromfile(file = self.fp, dtype=np.uint8, count=15).tostring()
        # Skip compiler-dependent record header
        self.skip_record_header()

        self.eof = False
        for tds in self:
            if tds:
                tt = tt + [tds[1]]
                dt = dt + [tds[2]]
                nt = nt + [tds[0]]
        self.fp.close()

        print str(len(dt)), ' timesteps read.'
        return tt, dt, nt
    def read_header(self):
        # Initialize return values
        obsnam = ''
        TimeObs = 0.
        ObsVal = 0.
        success = True
        try:
            # Skip compiler-dependent record header
            self.skip_record_header()
            # Read data as string and store in array hdr
            hdr = np.fromstring(self.fp.read(self.header_structure.itemsize), self.header_structure)[0]
            # Skip compiler-dependent record header
            self.skip_record_header()

            # Extract return values
            obsnam = hdr[0].tostring()
            TimeObs = hdr[1]
            ObsVal = hdr[2]
        except:
            success = False
        finally:
            return obsnam, TimeObs, ObsVal, success
#    def read_pst_single(self, f_p):
        '''Reads a single time step from an unformatted output file
        containing calculated concentrations and mass fluxes at
        user-defined observation points and mass-flux objects.
        Arguments:
        f_p: Handle to an open PST file
        Output:
        obsnam: The name of the observation point
        TimeObs: The time since the beginning of the simulations
        ObsVal: The concentration or mass-flux at the observation point at time TimeObs
        last_item_read: Boolean indicating if the end of the file has been reached'''
        '''     try:
            # Skip compiler-dependent header
            if (self.byte_skip > 0):
                dummy = np.fromfile(file = f_p, dtype=np.int16, count=self.byte_skip)
            obsnam = np.fromfile(file = f_p, dtype=np.uint8, count=12).tostring()
            # Time, observed value (concentration or flux)
            TimeObs, ObsVal = np.fromfile(file = f_p, dtype=np.float, count=2)
            # Skip compiler-dependent header
            if (self.byte_skip > 0):
                dummy = np.fromfile(file = f_p, dtype=np.int16, count=self.byte_skip)
            return obsnam, TimeObs, ObsVal, False
        except MemoryError:
            return '', 0., 0., True
    def read_pst_all(self, filename):'''
        '''Read binary output file containing calculated concentrations
        and mass fluxes at user-defined observation points and mass-flux
        objects.with Mt3dms.'''
'''        print ('Please wait. Reading observations from file %s.' % filename)
        f_p = open(filename, 'rb')

        # Skip compiler-dependent header
        if (self.byte_skip > 0):
            dummy = np.fromfile(file = f_p, dtype=np.int16, count=self.byte_skip)
        # Text header
        header = np.fromfile(file = f_p, dtype=np.uint8, count=15).tostring()
        # Skip compiler-dependent header
        if (self.byte_skip > 0):
            dummy = np.fromfile(file = f_p, dtype=np.int16, count=self.byte_skip)

        nt = []
        tt = []
        ot = []
        last_item_read = False

        while (not last_item_read):
            n, t, o, last_item_read = self.read_pst_single(f_p)
            if (not last_item_read): # and ((len(tt) == 0) or (tt[-1] != t))):
                nt = nt + [n]
                tt = tt + [t]
                ot = ot + [o]
        f_p.close()

        print str(len(ot)), ' timesteps read.'
        return nt, tt, ot'''
