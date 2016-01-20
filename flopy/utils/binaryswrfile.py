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

    def read_obs_text(self, nchar=20):
        textvalue = np.fromfile(file=self.file,
                                dtype=SwrBinaryStatements.character,
                                count=nchar).tostring()
        return textvalue

    def read_record(self, count=None):
        if count is None:
            count = self.nrecord
        return np.fromfile(self.file, dtype=self.read_dtype, count=count)


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
        self.floattype = 'f8'
        self.verbose = verbose
        # open binary head file
        self.file = open(filename, 'rb')

        # read header information
        self._read_header()

        # read data
        self.data = None
        self._read_data()

    def get_times(self):
        return self._get_selection(['totim'])

    def get_ntimes(self):
        return self.data['totim'].shape[0]

    def get_nobs(self):
        return self.nobs

    def get_obsnames(self):
        return self.data.dtype.names[1:]

    def get_data(self, obsname=None, idx=None):
        if obsname is None and idx is None:
            return self.data
        else:
            r = None
            if obsname is not None:
                if obsname not in self.data.dtype.names:
                    obsname = None
            elif idx is not None:
                idx += 1
                if idx < len(self.data.dtype.names):
                    obsname = self.data.dtype.names[idx]
            if obsname is not None:
                r = self._get_selection(['totim', obsname])
            return r

    def _read_header(self):
        # NOBS
        self.nobs = self.read_integer()
        self.v = np.empty((self.nobs), dtype='float')
        self.v.fill(1.0E+32)
        # read obsnames
        obsnames = []
        for idx in range(0, self.nobs):
            cid = self.read_obs_text()
            if isinstance(cid, bytes):
                cid = cid.decode()
            obsnames.append(cid.strip())
        #
        vdata = [('totim', self.floattype)]
        for name in obsnames:
            vdata.append((name, self.floattype))
        self.read_dtype = np.dtype(vdata)

        # set position of data start
        self.datastart = self.file.tell()

    def _read_data(self):

        if self.data is not None:
            return

        while True:
            try:
                r = self.read_record(count=1)
                if self.data is None:
                    self.data = r.copy()
                else:
                    self.data = np.vstack((self.data, r))
            except:
                break

        return

    def _get_selection(self, names):
        if not isinstance(names, list):
            names = [names]
        dtype2 = np.dtype({name:self.data.dtype.fields[name] for name in names})
        return np.ndarray(self.data.shape, dtype2, self.data, 0, self.data.strides)


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

            # add totim to data record array
            s = np.zeros(r.shape[0], dtype=self.dtype)
            s['totim'] = totim1
            for name in r.dtype.names:
                s[name] = r[name]
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
        self.vtotim = ('totim', self.floattype)
        if self.type == self.types[0]:
            vtype = [('stage', self.floattype)]
        elif self.type == self.types[1]:
            vtype = [('stage', self.floattype), ('qsflow', self.floattype),
                     ('qlatflow', self.floattype), ('quzflow', self.floattype),
                     ('rain', self.floattype), ('evap', self.floattype),
                     ('qbflow', self.floattype), ('qeflow', self.floattype),
                     ('qexflow', self.floattype), ('qbcflow', self.floattype),
                     ('qcrflow', self.floattype), ('dv', self.floattype),
                     ('inf-out', self.floattype), ('volume', self.floattype)]
        elif self.type == self.types[2]:
            vtype = [('flow', self.floattype),
                     ('velocity', self.floattype)]
        elif self.type == self.types[3]:
            vtype = [('layer', 'i4'), ('bottom', 'f8'), ('stage', 'f8'),
                     ('depth', 'f8'), ('head', 'f8'), ('wetper', 'f8'),
                     ('cond', 'f8'), ('headdiff', 'f8'), ('qaq', 'f8')]
        self.read_dtype = np.dtype(vtype)
        temp = list(vtype)
        if self.type == self.types[3]:
            temp.insert(0, ('reach', 'i4'))
            self.qaq_dtype = np.dtype(temp)
        temp.insert(0, self.vtotim)
        self.dtype = np.dtype(temp)
        return

    def _read_header(self):
        nqaq = 0
        if self.type == 'qaq':
            reachlayers = np.zeros(self.nrecord, np.int)
            try:
                for i in range(self.nrecord):
                    reachlayers[i] = self.read_integer()
                    nqaq += reachlayers[i]
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
        gage_record = np.zeros(self._ntimes, dtype=self.dtype)

        # iterate through the record dictionary
        idx = 0
        for key, value in self.recorddict.items():
            totim = np.array(key)
            gage_record['totim'][idx] = totim

            self.file.seek(value)
            r = self._get_data()
            for name in r.dtype.names:
                gage_record[name][idx] = r[name][irec]
            idx += 1

        return gage_record.view(dtype=self.dtype)

    def _get_ts_qm(self, irec=0, iconn=0):

        # create array
        gage_record = np.zeros(self._ntimes, dtype=self.dtype)

        # iterate through the record dictionary
        idx = 0
        for key, value in self.recorddict.items():
            totim = key
            gage_record['totim'][idx] = totim

            self.file.seek(value)
            r = self._get_data()

            # find correct entry for reach and connection
            for i in range(self.nrecord):
                inode = self.connectivity[i, 1]
                ic = self.connectivity[i, 2]
                if irec == inode and ic == iconn:
                    for name in r.dtype.names:
                        gage_record[name][idx] = r[name][i]
                    break
            idx += 1

        return gage_record.view(dtype=self.dtype)

    def _get_ts_qaq(self, irec=0, klay=0):

        # create array
        gage_record = np.zeros(self._ntimes, dtype=self.dtype)

        # iterate through the record dictionary
        idx = 0
        for key, value in self.recorddict.items():
            totim = key
            gage_record['totim'][idx] = totim

            self.file.seek(value)
            r = self._get_data()

            self.nqaq, self.reachlayers = self.nqaqentries[key]

            # find correct entry for record and layer
            ilen = np.shape(r)[0]
            for i in range(ilen):
                ir = r['reach'][i]
                il = r['layer'][i]
                if ir == irec and il == klay:
                    for name in r.dtype.names:
                        gage_record[name][idx] = r[name][i]
                    break
            idx += 1

        return gage_record.view(dtype=self.dtype)

    def _get_data(self):
        if self.type == 'qaq':
            return self._read_qaq()
        else:
            return self.read_record()

    def _read_qaq(self):

        # read qaq data using standard record reader
        bd = self.read_record(count=self.nqaq)
        bd['layer'] -= 1

        # add reach number to qaq data
        r = np.zeros(self.nqaq, dtype=self.qaq_dtype)

        # build array with reach numbers
        reaches = np.zeros(self.nqaq, dtype=np.int32)
        idx = 0
        for irch in range(self.nrecord):
            klay = self.reachlayers[irch]
            for k in range(klay):
                # r[idx, 0] = irch
                reaches[idx] = irch
                idx += 1

        # add reach to array returned
        r['reach'] = reaches.copy()

        # add read data to array returned
        for idx, k in enumerate(self.read_dtype.names):
            r[k] = bd[k]
        return r

    def _build_index(self):
        """
        Build the recordarray recarray and recorddict dictionary, which map
        the header information to the position in the binary file.
        """
        self.file.seek(self.datastart)
        if self.verbose:
            sys.stdout.write('Generating SWR binary data time list\n')
        self._ntimes = 0
        self._times = []
        self._kswrkstpkper = []
        self.recorddict = OrderedDict()

        idx = 0
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
                if self.verbose:
                    sys.stdout.write('\n')
                self._recordarray = np.array(self._recordarray,
                                             dtype=self.header_dtype)
                self._times = np.array(self._times)
                self._kswrkstpkper = np.array(self._kswrkstpkper)
                return
