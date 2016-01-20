import sys
import numpy as np
import struct as strct
from collections import OrderedDict


class SwrBinaryStatements:
    integer = np.int32
    real = np.float64
    character = np.uint8
    integerbyte = 4
    realbyte = 8
    textbyte = 4

    def read_integer(self):
        intvalue = \
            strct.unpack('i',
                         self.file.read(1 * SwrBinaryStatements.integerbyte))[
                0]
        return intvalue

    def read_real(self):
        realvalue = \
            strct.unpack('d',
                         self.file.read(1 * SwrBinaryStatements.realbyte))[0]
        return realvalue

    def read_text(self):
        textvalue = np.fromfile(file=self.file,
                                dtype=SwrBinaryStatements.character,
                                count=16).tostring()
        return textvalue

    def read_obs_text(self, nchar=20):
        textvalue = np.fromfile(file=self.file,
                                dtype=SwrBinaryStatements.character,
                                count=nchar).tostring()
        return textvalue

    def read_record(self):
        if self.skip == True:
            lpos = self.file.tell() + (
                self.nrecord * self.items * SwrBinaryStatements.realbyte)
            self.file.seek(lpos)
            x = np.zeros((self.nrecord * self.items), SwrBinaryStatements.real)
        else:
            x = np.fromfile(file=self.file, dtype=SwrBinaryStatements.real,
                            count=self.nrecord * self.items)
        x.resize(self.nrecord, self.items)
        return x

    def read_items(self):
        if self.skip == True:
            lpos = self.file.tell() + (
                self.items * SwrBinaryStatements.realbyte)
            self.file.seek(lpos)
            x = np.zeros((self.items), SwrBinaryStatements.real)
        else:
            x = np.fromfile(file=self.file, dtype=SwrBinaryStatements.real,
                            count=self.items)
        return x

    def read_1dintegerarray(self):
        i = np.fromfile(file=self.file, dtype=SwrBinaryStatements.integer,
                        count=self.nrecord)
        return i


class SwrObs(SwrBinaryStatements):
    """
    Read binary SWR observations output from MODFLOW SWR Process binary observation files

    Parameters
    ----------
    filename : string
        Name of the cell budget file
    verbose : bool
        Write information to the screen.  Default is False.

    Attributes
    ----------

    Methods
    -------

    See Also
    --------

    Notes
    -----

    Examples
    --------

    >>> import flopy
    >>> so = flopy.utils.SwrObs('mymodel.swr.obs')

    """

    def __init__(self, filename, verbose=False):
        """
        Class constructor.

        """
        # initialize class information
        self.skip = False
        self.verbose = verbose
        # open binary head file
        self.file = open(filename, 'rb')
        # NOBS
        self.nobs = self.read_integer()
        self.v = np.empty((self.nobs), dtype='float')
        self.v.fill(1.0E+32)
        # read obsnames
        obsnames = []
        for idx in range(0, self.nobs):
            cid = self.read_obs_text()
            obsnames.append(cid)
        self.obsnames = np.array(obsnames)
        # set position
        self.datastart = self.file.tell()
        # get times
        self.times = self._set_time_list()

    def get_times(self):
        return self.times

    def get_num_items(self):
        return self.nobs

    def get_obs_labels(self):
        return self.obsnames

    def _rewind_file(self):
        self.file.seek(self.datastart)
        return True

    def _set_time_list(self):
        self.skip = True
        self.file.seek(self.datastart)
        times = []
        while True:
            current_position = self.file.tell()
            totim, v, success = self._get_data()
            if success:
                times.append([totim, current_position])
            else:
                self.file.seek(self.datastart)
                times = np.array(times)
                self.skip = False
                return times

    def __iter__(self):
        return self

    def _read_header(self):
        try:
            totim = self.read_real()
            return totim, True
        except:
            return -999., False

    def _get_data(self):
        totim, success = self._read_header()
        if (success):
            for idx in range(0, self.nobs):
                self.v[idx] = self.read_real()
        else:
            if self.verbose == True:
                print('_BinaryObs object._get_data() reached end of file.')
            self.v.fill(1.0E+32)
        return totim, self.v, success

    def get_values(self, idx):
        iposition = int(self.times[idx, 1])
        self.file.seek(iposition)
        totim, v, success = self._get_data()
        if success:
            return totim, v, True
        else:
            self.v.fill(1.0E+32)
            return 0.0, self.v, False

    def get_time_gage(self, record):
        idx = -1
        try:
            idx = int(record) - 1
            if self.verbose == True:
                print(
                        'retrieving SWR observation record [{0}]'.format(
                                idx + 1))
        except:
            for icnt, cid in enumerate(self.obsnames):
                if record.strip().lower() == cid.strip().lower():
                    idx = icnt
                    if self.verbose == True:
                        print(
                                'retrieving SWR observation record [{0}] {1}'.format(
                                        idx + 1, record.strip().lower()))
                    break
        gage_record = np.zeros((2))  # tottime plus observation
        if idx != -1 and idx < self.nobs:
            # --find offset to position
            ilen = self._get_point_offset(idx)
            # get data
            for time_data in self.times:
                self.file.seek(int(time_data[1]) + ilen)
                v = self.read_real()
                this_entry = np.array([float(time_data[0])])
                this_entry = np.hstack((this_entry, v))
                gage_record = np.vstack((gage_record, this_entry))
            # delete the first 'zeros' element
            gage_record = np.delete(gage_record, 0, axis=0)
        return gage_record

    def _get_point_offset(self, ipos):
        self.file.seek(self.datastart)
        lpos0 = self.file.tell()
        point_offset = int(0)
        totim, success = self._read_header()
        idx = (ipos)
        lpos1 = self.file.tell() + idx * SwrBinaryStatements.realbyte
        self.file.seek(lpos1)
        point_offset = self.file.tell() - lpos0
        return point_offset


class SwrFile(SwrBinaryStatements):
    """
    Read binary SWR output from MODFLOW SWR Process binary output files

    Parameters
    ----------
    filename : string
        Name of the swr output file
    swrtype : str
        swr data type. Valid data types are 'stage', 'reachgroup',
        'qm', or 'qaq'. (default is 'stage')
    verbose : bool
        Write information to the screen.  Default is False.

    Attributes
    ----------

    Methods
    -------

    See Also
    --------

    Notes
    -----

    Examples
    --------

    >>> import flopy
    >>> so = flopy.utils.SwrFile('mymodel.swr.stage.bin')

    """

    def __init__(self, filename, swrtype='stage', verbose=False):
        """
        Class constructor.

        """
        self.floattype = 'f8'
        self.header_dtype = np.dtype([('totim', self.floattype),
                                      ('kswr', 'i4'), ('kstp', 'i4'),
                                      ('kper', 'i4')])
        self._recordarray = []

        self.file = open(filename, 'rb')
        self.types = ('stage', 'budget', 'qm', 'qaq')
        if swrtype.lower() in self.types:
            self.type = swrtype.lower()
        else:
            err = 'SWR type ({}) is not defined. '.format(type) + \
                  'Available types are:\n'
            for t in self.types:
                err = '{}  {}\n'.format(err, t)
            raise Exception(err)

        # set data dtypes
        self._set_dtypes()

        # debug
        self.verbose = verbose

        # Read the dimension data
        self.nrgout = 0
        if self.type == 'qm':
            self.nrgout = self.read_integer()
        self.nrecord = self.read_integer()

        # set-up
        self.items = len(self.dtype) - 1

        # read connectivity for velocity data if necessary
        self.conn_dtype = None
        if self.type == 'qm':
            self.connectivity = self._read_connectivity()
            if self.verbose:
                print('Connectivity: ')
                print(self.connectivity)

        # initialize reachlayers and nqaqentries for qaq data
        self.nqaqentries = {}
        self.qaq_dtype = np.dtype([('layer', 'i4'),
                                   ('bottom', 'f8'), ('stage', 'f8'),
                                   ('depth', 'f8'), ('head', 'f8'),
                                   ('wetper', 'f8'), ('cond', 'f8'),
                                   ('headdiff', 'f8'), ('qaq', 'f8')])

        self.datastart = self.file.tell()

        # build index
        self._build_index()

    def get_connectivity(self):
        if self.type == 'qm':
            return self.connectivity
        else:
            return None

    def get_nrecords(self):
        return self.nrgout, self.nrecord

    def get_kswrkstpkper(self):
        return self._kswrkstpkper

    def get_ntimes(self):
        return self._ntimes

    def get_times(self):
        return self._times

    def get_record_names(self):
        return self.dtype.names

    def get_data(self, kswrkstpkper=None, idx=None, totim=None):

        if kswrkstpkper is not None:
            kswr1 = kswrkstpkper[0]
            kstp1 = kswrkstpkper[1]
            kper1 = kswrkstpkper[2]

            totim1 = self._recordarray[np.where(
                    (self._recordarray['kswr'] == kswr1) &
                    (self._recordarray['kstp'] == kstp1) &
                    (self._recordarray['kper'] == kper1))]["totim"][0]
        elif totim is not None:
            totim1 = totim
        elif idx is not None:
            totim1 = self._recordarray['totim'][idx]
        else:
            totim1 = self._times[-1]

        try:
            ipos = self.recorddict[totim1]
            self.file.seek(ipos)
            if self.type == 'qaq':
                self.nqaq, self.reachlayers = self.nqaqentries[totim1]
                r = self._read_qaq()
            else:
                r = self.read_record()
            # expand data to add totim so that self.dtype can be
            # used to return a numpy recarry
            s = np.zeros((r.shape[0], r.shape[1] + 1), np.float)
            s[:, 1:] = r[:, :]
            s[:, 0] = totim1
            return s.view(dtype=self.dtype)
        except:
            return None

    def get_ts(self, irec=0, iconn=0, klay=0):
        """
        Get a time series from a swr binary file.

        Parameters
        ----------
        irec : int
            is the zero-based reach (stage, qm, qaq) or reach group number
            (budget) to retrieve. (default is 0)
        iconn : int
            is the zero-based connection number for reach (irch) to retrieve
            qm data. iconn is only used if qm data is being read.
            (default is 0)
        klay : int
            is the zero-based layer number for reach (irch) to retrieve
            qaq data . klay is only used if qaq data is being read.
            (default is 0)

        Returns
        ----------
        out : numpy recarray
            Array has size (ntimes, nitems).  The first column in the
            data array will contain time (totim). nitems is 2 for stage
            data, 15 for budget data, 3 for qm data, and 11 for qaq
            data.

        See Also
        --------

        Notes
        -----

        The irec, iconn, and klay values must be zero-based.

        Examples
        --------

        """

        if irec + 1 > self.nrecord:
            err = 'Error: specified irec ({}) '.format(irec) + \
                  'exceeds the total number of records ()'.format(self.nrecord)
            raise Exception(err)

        gage_record = None
        # stage and budget
        if self.type == self.types[0] or self.type == self.types[1]:
            gage_record = self._get_ts(irec=irec)
        # qm
        elif self.type == self.types[2]:
            gage_record = self._get_ts_qm(irec=irec, iconn=iconn)
        # qaq
        elif self.type == self.types[3]:
            gage_record = self._get_ts_qaq(irec=irec, klay=klay)

        return gage_record

    def _read_connectivity(self):
        self.conn_dtype = np.dtype([('reach', 'i4'),
                                    ('from', 'i4'), ('to', 'i4')])
        conn = np.zeros((self.nrecord, 3), np.int)
        icount = 0
        for nrg in range(self.nrgout):
            nconn = self.read_integer()
            for ic in range(nconn):
                conn[icount, 0] = nrg
                conn[icount, 1] = self.read_integer() - 1
                conn[icount, 2] = self.read_integer() - 1
                icount += 1
        return conn

    def _set_dtypes(self):
        if self.type == self.types[0]:
            self.read_dtype = np.dtype([('stage', self.floattype)])
            self.dtype = np.dtype([('totim', self.floattype),
                                   ('stage', self.floattype)])
        elif self.type == self.types[1]:
            self.read_dtype = np.dtype([('stage', self.floattype),
                                        ('qsflow', self.floattype),
                                        ('qlatflow', self.floattype),
                                        ('quzflow', self.floattype),
                                        ('rain', self.floattype),
                                        ('evap', self.floattype),
                                        ('qbflow', self.floattype),
                                        ('qeflow', self.floattype),
                                        ('qexflow', self.floattype),
                                        ('qbcflow', self.floattype),
                                        ('qcrflow', self.floattype),
                                        ('dv', self.floattype),
                                        ('inf-out', self.floattype),
                                        ('volume', self.floattype)])
            self.dtype = np.dtype([('totim', self.floattype),
                                   ('stage', self.floattype),
                                   ('qsflow', self.floattype),
                                   ('qlatflow', self.floattype),
                                   ('quzflow', self.floattype),
                                   ('rain', self.floattype),
                                   ('evap', self.floattype),
                                   ('qbflow', self.floattype),
                                   ('qeflow', self.floattype),
                                   ('qexflow', self.floattype),
                                   ('qbcflow', self.floattype),
                                   ('qcrflow', self.floattype),
                                   ('dv', self.floattype),
                                   ('inf-out', self.floattype),
                                   ('volume', self.floattype)])
        elif self.type == self.types[2]:
            self.read_dtype = np.dtype([('flow', self.floattype),
                                        ('velocity', self.floattype)])
            self.dtype = np.dtype([('totim', self.floattype),
                                   ('flow', self.floattype),
                                   ('velocity', self.floattype)])
        elif self.type == self.types[3]:
            self.read_dtype = np.dtype([('layer', 'i4'),
                                        ('bottom', 'f8'), ('stage', 'f8'),
                                        ('depth', 'f8'), ('head', 'f8'),
                                        ('wetper', 'f8'), ('cond', 'f8'),
                                        ('headdiff', 'f8'), ('qaq', 'f8')])
            self.dtype = np.dtype([('totim', self.floattype),
                                   ('reach', self.floattype),
                                   ('layer', self.floattype),
                                   ('bottom', self.floattype),
                                   ('stage', self.floattype),
                                   ('depth', self.floattype),
                                   ('head', self.floattype),
                                   ('wetper', self.floattype),
                                   ('cond', self.floattype),
                                   ('headdiff', self.floattype),
                                   ('qaq', self.floattype)])
        return

    def _read_header(self):
        nqaq = 0
        if self.type == 'qaq':
            reachlayers = np.zeros(self.nrecord, np.int)
            try:
                for i in range(self.nrecord):
                    reachlayers[i] = self.read_integer()
                    nqaq += reachlayers[i]
                    # print i+1, self.reachlayers[i]
                    # print self.nqaqentries
                self.nqaq = nqaq
            except:
                if self.verbose:
                    sys.stdout.write('\nCould not read reachlayers')
                return 0.0, 0.0, 0, 0, 0, False
        try:
            totim = self.read_real()
            dt = self.read_real()
            kper = self.read_integer() - 1
            kstp = self.read_integer() - 1
            kswr = self.read_integer() - 1
            if self.type == 'qaq':
                self.nqaqentries[totim] = (nqaq, reachlayers)
            return totim, dt, kper, kstp, kswr, True
        except:
            return 0.0, 0.0, 0, 0, 0, False

    def _get_ts(self, irec=0):

        # create array
        gage_record = np.zeros((self._ntimes, self.items + 1), dtype=np.float)

        # iterate through the record dictionary
        idx = 0
        for key, value in self.recorddict.items():
            self.file.seek(value)
            r = self._get_data()
            header = np.array(key)
            this_entry = np.hstack((header, r[irec]))
            gage_record[idx, :] = this_entry
            idx += 1

        return gage_record.view(dtype=self.dtype)

    def _get_ts_qm(self, irec=0, iconn=0):

        # create array
        gage_record = np.zeros((self._ntimes, self.items + 1), dtype=np.float)

        # iterate through the record dictionary
        idx = 0
        for key, value in self.recorddict.items():
            self.file.seek(value)
            r = self._get_data()
            header = np.array(key)
            ipos = irec

            # find correct entry for reach and connection
            ifound = 0
            for i in range(self.nrecord):
                inode = self.connectivity[i, 1]
                ic = self.connectivity[i, 2]
                if irec == inode and ic == iconn:
                    ifound = 1
                    ipos = i
                    break
            if ifound < 1:
                r[ipos, :] = 0.0

            this_entry = np.hstack((header, r[ipos]))
            gage_record[idx, :] = this_entry
            idx += 1

        return gage_record.view(dtype=self.dtype)

    def _get_ts_qaq(self, irec=0, klay=0):

        # create array
        gage_record = np.zeros((self._ntimes, self.items + 1), dtype=np.float)

        # iterate through the record dictionary
        idx = 0
        for key, value in self.recorddict.items():
            self.file.seek(value)
            self.nqaq, self.reachlayers = self.nqaqentries[key]
            r = self._get_data()
            header = np.array(key)
            ipos = irec

            # find correct entry for record and layer
            ifound = 0
            ilen = np.shape(r)[0]
            # print np.shape(r)
            for i in range(0, ilen):
                ir = int(r[i, 0]) - 1
                il = int(r[i, 1]) - 1
                if ir == irec and il == klay:
                    ifound = 1
                    ipos = i
                    break
            if ifound < 1:
                r[ipos, :] = 0.0

            this_entry = np.hstack((header, r[ipos]))
            gage_record[idx, :] = this_entry
            idx += 1

        return gage_record.view(dtype=self.dtype)

    def _get_data(self):
        if self.type == 'qaq':
            return self._read_qaq()
        else:
            return self.read_record()

    def _read_qaq(self):
        r = np.zeros((self.nqaq, self.items), SwrBinaryStatements.real)

        #bd = np.fromfile(self.file, dtype=self.qaq_dtype,
        #                 count=self.nqaq)
        bd = np.fromfile(self.file, dtype=self.read_dtype,
                         count=self.nqaq)
        ientry = 0
        for irch in range(self.nrecord):
            klay = self.reachlayers[irch]
            for k in range(klay):
                r[ientry, 0] = irch + 1
                ientry += 1
        for idx, k in enumerate(self.qaq_dtype.names):
            r[:, idx + 1] = bd[k]
        # print 'shape x: {}'.format(x.shape)
        return r

    def _build_index(self):
        """
        Build the recordarray recarray and recorddict dictionary, which map
        the header information to the position in the binary file.
        """
        self.skip = True
        self.file.seek(self.datastart)
        idx = 0
        sys.stdout.write('Generating SWR binary data time list\n')
        self._ntimes = 0
        self._times = []
        self._kswrkstpkper = []
        self.recorddict = OrderedDict()
        while True:
            # --output something to screen so it is possible to determine
            #  that the time list is being created
            idx += 1
            if self.verbose:
                v = divmod(float(idx), 72.)
                if v[1] == 0.0:
                    sys.stdout.write('.')
            # read header
            totim, dt, kper, kstp, kswr, success = self._read_header()
            if success:
                if self.type == 'qaq':
                    bytes = self.nqaq * \
                            (SwrBinaryStatements.integerbyte +
                             8 * SwrBinaryStatements.realbyte)
                else:
                    bytes = self.nrecord * self.items * \
                            SwrBinaryStatements.realbyte
                ipos = self.file.tell()
                self.file.seek(bytes, 1)
                # save data
                self._ntimes += 1
                self._times.append(totim)
                self._kswrkstpkper.append((kswr, kstp, kper))
                header = (totim, kswr, kstp, kper)
                self.recorddict[totim] = ipos
                self._recordarray.append(header)
            else:
                self.skip = False
                if self.verbose:
                    sys.stdout.write('\n')
                self._recordarray = np.array(self._recordarray,
                                             dtype=self.header_dtype)
                self._times = np.array(self._times)
                self._kswrkstpkper = np.array(self._kswrkstpkper)
                return
