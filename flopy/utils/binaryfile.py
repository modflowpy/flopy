"""
Module to read MODFLOW binary output files.  The module contains three
important classes that can be accessed by the user.

*  HeadFile (Binary head file.  Can also be used for drawdown)
*  UcnFile (Binary concentration file from MT3DMS)
*  CellBudgetFile (Binary cell-by-cell flow file)

"""
from __future__ import print_function
import numpy as np
from collections import OrderedDict
from flopy.utils.datafile import Header, LayerFile

class BinaryHeader():
    """
    The binary_header class is a class to create headers for MODFLOW
    binary files.

    Parameters
    ----------
        bintype is the type of file being opened (head and unc file currently supported)
        precision is the precision of the floating point data in the file
    """
    def __init__(self, bintype=None, precision='single'):
        super(BinaryHeader, self).__init__(bintype, precision)

    def set_values(self, **kwargs):
        """
        Set values using kwargs
        """
        ikey = ['ntrans', 'kstp', 'kper', 'ncol', 'nrow', 'ilay']
        fkey = ['pertim', 'totim']
        ckey = ['text']
        for k in ikey:
            if kwargs.has_key(k):
                try:
                    self.header[0][k] = int(kwargs[k])
                except:
                    print('{0} key not available in {1} header dtype'.format(k, self.header_type))
        for k in fkey:
            if kwargs.has_key(k):
                try:
                    self.header[0][k] = float(kwargs[k])
                except:
                    print('{0} key not available in {1} header dtype'.format(k, self.header_type))
        for k in ckey:
            if kwargs.has_key(k):
                # Convert to upper case to be consistent case used by MODFLOW
                # text strings. Necessary to work with HeadFile and UcnFile
                # routines
                ttext = kwargs[k].upper()
                if len(ttext) > 16:
                    text = text[0:16]
                else:
                    text = ttext
                self.header[0][k] = text
            else:
                self.header[0][k] = 'DUMMY TEXT'

    @staticmethod
    def set_dtype(bintype=None, precision='single'):
        """
        Set the dtype
        """
        header = Header(filetype=bintype, precision=precision)
        return header.dtype

    @staticmethod
    def create(bintype=None, **kwargs):
        """
        Create a binary header
        """
        header = Header(filetype=bintype)
        if header.get_dtype() is not None:
            header.set_values(**kwargs)
        return header.get_values()

def binaryread_struct(file, vartype, shape=(1), charlen=16):
    """
    Read text, a scalar value, or an array of values from a binary file.
        file is an open file object
        vartype is the return variable type: str, numpy.int32, numpy.float32,
            or numpy.float64
        shape is the shape of the returned array (shape(1) returns a single
            value) for example, shape = (nlay, nrow, ncol)
        charlen is the length of the text string.  Note that string arrays
            cannot be returned, only multi-character strings.  Shape has no
            affect on strings.

    """
    import struct
    import numpy as np
    
    # store the mapping from type to struct format (fmt)
    typefmtd = {np.int32:'i', np.float32:'f', np.float64:'d'}
        
    # read a string variable of length charlen
    if vartype == str:
        result = file.read(charlen*1)
        
    # read other variable types
    else:
        fmt = typefmtd[vartype]
        # find the number of bytes for one value
        numbytes = vartype(1).nbytes
        # find the number of values
        nval = np.core.fromnumeric.prod(shape)
        fmt = str(nval) + fmt
        s = file.read(numbytes * nval)
        result = struct.unpack(fmt, s)
        if nval == 1:
            result = vartype(result[0])
        else:
            result = np.array(result, dtype=vartype)
            result = np.reshape(result, shape)
    return result


def binaryread(file, vartype, shape=(1), charlen=16):
    """
    Uses numpy to read from binary file.  This was found to be faster than the
        struct approach and is used as the default.

    """
    
    # read a string variable of length charlen
    if vartype == str:
        result = file.read(charlen*1)     
    else:
        # find the number of values
        nval = np.core.fromnumeric.prod(shape)
        result = np.fromfile(file,vartype,nval)
        if nval == 1:
            result = result  # [0]
        else:
            result = np.reshape(result, shape)
    return result

def join_struct_arrays(arrays):
    """
    Simple function that can join two numpy structured arrays.

    """
    newdtype = sum((a.dtype.descr for a in arrays), [])
    newrecarray = np.empty(len(arrays[0]), dtype = newdtype)
    for a in arrays:
        for name in a.dtype.names:
            newrecarray[name] = a[name]
    return newrecarray


class BinaryLayerFile(LayerFile):
    """
    The BinaryLayerFile class is the super class from which specific derived
    classes are formed.  This class should not be instantiated directly

    """
    def __init__(self, filename, precision, verbose, kwargs):
        super(BinaryLayerFile, self).__init__(filename, precision, verbose, kwargs)
        return
   

    def _build_index(self):
        """
        Build the recordarray and iposarray, which maps the header information
        to the position in the binary file.
        """
        header = self._get_header()
        self.nrow = header['nrow']
        self.ncol = header['ncol']
        self.file.seek(0, 2)
        self.totalbytes = self.file.tell()
        self.file.seek(0, 0)        
        self.databytes = header['ncol'] * header['nrow'] * self.realtype(1).nbytes
        ipos = 0
        while ipos < self.totalbytes:           
            header = self._get_header()
            self.recordarray.append(header)
            if self.text.upper() not in header['text']:
                continue
            if ipos == 0:
                self.times.append(header['totim'])
                kstpkper = (header['kstp'], header['kper'])
                self.kstpkper.append(kstpkper)
            else:
                totim = header['totim']
                if totim != self.times[-1]:
                    self.times.append(totim)
                    kstpkper = (header['kstp'], header['kper'])
                    self.kstpkper.append(kstpkper)
            ipos = self.file.tell()
            self.iposarray.append(ipos)
            self.file.seek(self.databytes, 1)
            ipos = self.file.tell()

        # self.recordarray contains a recordarray of all the headers.
        self.recordarray = np.array(self.recordarray, dtype=self.header_dtype)
        self.iposarray = np.array(self.iposarray)
        self.nlay = np.max(self.recordarray['ilay'])
        return

    def _read_data(self):
        return binaryread(self.file, self.realtype, shape=(self.nrow, self.ncol))

    def _get_header(self):
        """
        Read the file header

        """
        header = binaryread(self.file, self.header_dtype, (1,))
        return header[0]

    def get_ts(self, idx):
        """
        Get a time series from the binary file.

        Parameters
        ----------
        idx : tuple of ints, or a list of a tuple of ints
            idx can be (layer, row, column) or it can be a list in the form
            [(layer, row, column), (layer, row, column), ...].  The layer,
            row, and column values must be zero based.

        Returns
        ----------
        out : numpy array
            Array has size (ntimes, ncells + 1).  The first column in the
            data array will contain time (totim).

        See Also
        --------

        Notes
        -----

        The layer, row, and column values must be zero-based, and must be
        within the following ranges: 0 <= k < nlay; 0 <= i < nrow; 0 <= j < ncol

        Examples
        --------

        """
        kijlist = self._build_kijlist(idx)
        nstation = self._get_nstation(idx, kijlist)

        # Initialize result array and put times in first column
        result = self._init_result(nstation)

        istat = 1
        for k, i, j in kijlist:
            recordlist = []
            ioffset = (i * self.ncol + j) * self.realtype(1).nbytes
            for irec, header in enumerate(self.recordarray):
                ilay = header['ilay'] - 1  # change ilay from header to zero-based
                if ilay != k:
                    continue
                ipos = self.iposarray[irec]

                # Calculate offset necessary to reach intended cell
                self.file.seek(ipos + np.long(ioffset), 0)

                # Find the time index and then put value into result in the
                # correct location.
                itim = np.where(result[:, 0] == header['totim'])[0]
                result[itim, istat] = binaryread(self.file, self.realtype)
            istat += 1
        return result

class HeadFile(BinaryLayerFile):
    """
    HeadFile Class.

    Parameters
    ----------
    filename : string
        Name of the concentration file
    text : string
        Name of the text string in the head file.  Default is 'head'
    precision : string
        'single' or 'double'.  Default is 'single'.
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
    The HeadFile class provides simple ways to retrieve 2d and 3d
    head arrays from a MODFLOW binary head file and time series
    arrays for one or more cells.

    The BinaryLayerFile class is built on a record array consisting of
    headers, which are record arrays of the modflow header information
    (kstp, kper, pertim, totim, text, nrow, ncol, ilay)
    and long integers, which are pointers to first bytes of data for
    the corresponding data array.

    Examples
    --------

    >>> import flopy.utils.binaryfile as bf
    >>> hdobj = bf.HeadFile('model.hds', precision='single')
    >>> hdobj.list_records()
    >>> rec = hdobj.get_data(kstpkper=(1, 50))

    >>> ddnobj = bf.HeadFile('model.ddn', text='drawdown', precision='single')
    >>> ddnobj.list_records()
    >>> rec = ddnobj.get_data(totim=100.)


    """
    def __init__(self, filename, text='head', precision='single',
                 verbose=False, **kwargs):
        self.text = text.encode()
        self.header_dtype = BinaryHeader.set_dtype(bintype='Head',
                                                   precision=precision)
        super(HeadFile, self).__init__(filename, precision, verbose, kwargs)
        return


class UcnFile(BinaryLayerFile):
    """
    UcnFile Class.

    Parameters
    ----------
    filename : string
        Name of the concentration file
    text : string
        Name of the text string in the ucn file.  Default is 'CONCENTRATION'
    precision : string
        'single' or 'double'.  Default is 'single'.
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
    The UcnFile class provides simple ways to retrieve 2d and 3d
    concentration arrays from a MT3D binary head file and time series
    arrays for one or more cells.

    The BinaryLayerFile class is built on a record array consisting of
    headers, which are record arrays of the modflow header information
    (kstp, kper, pertim, totim, text, nrow, ncol, ilay)
    and long integers, which are pointers to first bytes of data for
    the corresponding data array.

    Examples
    --------

    >>> import flopy.utils.binaryfile as bf
    >>> ucnobj = bf.UcnFile('MT3D001.UCN', precision='single')
    >>> ucnobj.list_records()
    >>> rec = ucnobj.get_data(kstpkper=(1,1))

    """
    def __init__(self, filename, text='concentration', precision='single',
                 verbose=False, **kwargs):
        self.text = text.encode()
        self.header_dtype = BinaryHeader.set_dtype(bintype='Ucn',
                                                   precision=precision)
        super(UcnFile, self).__init__(filename, precision, verbose, kwargs)
        return


class CellBudgetFile(object):
    """
    CellBudgetFile Class.

    Parameters
    ----------
    filename : string
        Name of the cell budget file
    precision : string
        'single' or 'double'.  Default is 'single'.
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

    >>> import flopy.utils.binaryfile as bf
    >>> cbb = bf.CellBudgetFile('mymodel.cbb')
    >>> cbb.list_records()
    >>> rec = cbb.get_data(kstpkper=(0,0), text='RIVER LEAKAGE')

    """

    def __init__(self, filename, precision='single', verbose=False, **kwargs):
        self.filename = filename
        self.precision = precision
        self.verbose = verbose
        self.file = open(self.filename, 'rb')
        self.nrow = 0
        self.ncol = 0
        self.nlay = 0
        self.times = []
        self.kstpkper = []
        self.recordarray = []
        self.iposarray = []
        self.textlist = []
        self.nrecords = 0
        h1dt = [('kstp', 'i4'), ('kper', 'i4'), ('text', 'a16'),
                ('ncol', 'i4'), ('nrow', 'i4'), ('nlay', 'i4')]

        if precision == 'single':
            self.realtype = np.float32
            h2dt = [('imeth', 'i4'), ('delt', 'f4'), ('pertim', 'f4'),
                    ('totim', 'f4')]
        elif precision == 'double':
            self.realtype = np.float64
            h2dt = [('imeth', 'i4'),('delt', 'f8'), ('pertim', 'f8'),
                    ('totim', 'f8')]
        else:
            raise Exception('Unknown precision specified: ' + precision)

        if len(kwargs.keys()) > 0:
            raise NotImplementedError()

        self.header1_dtype = np.dtype(h1dt)
        self.header2_dtype = np.dtype(h2dt)
        hdt = h1dt + h2dt
        self.header_dtype = np.dtype(hdt)

        # read through the file and build the pointer index
        self._build_index()
        
        # allocate the value array
        self.value = np.empty((self.nlay, self.nrow, self.ncol),
                              dtype=self.realtype)
        return
   
    def _build_index(self):
        """
        Build the ordered dictionary, which maps the header information
        to the position in the binary file.
        """
        header = self._get_header()
        self.file.seek(0, 2)
        self.totalbytes = self.file.tell()
        self.file.seek(0, 0)
        self.databytes = (header['ncol'] * header['nrow'] * header['nlay'] 
                          * self.realtype(1).nbytes)
        self.recorddict = OrderedDict()
        ipos = 0
        while ipos < self.totalbytes:           
            header = self._get_header()
            if self.verbose:
                print(header)
            self.nrecords += 1
            totim = header['totim']
            if totim > 0 and totim not in self.times:
                self.times.append(totim)
            kstpkper = (header['kstp'], header['kper'])
            if kstpkper not in self.kstpkper:
                self.kstpkper.append( kstpkper )
            if header['text'] not in self.textlist:
                self.textlist.append(header['text'])
            ipos = self.file.tell()

            if self.verbose:
                for itxt in ['kstp', 'kper', 'text', 'ncol', 'nrow', 'nlay',
                             'imeth', 'delt', 'pertim', 'totim']:
                    print(itxt + ': ' + str(header[itxt]))
                print('file position: ', ipos)
                if int(header['imeth']) != 5:
                    print('\n')

            # store record and byte position mapping
            self.recorddict[tuple(header)] = ipos    # store the position right after header2
            self.recordarray.append(header)
            self.iposarray.append(ipos)  # store the position right after header2

            # skip over the data to the next record and set ipos
            self._skip_record(header)
            ipos = self.file.tell()

        # convert to numpy arrays
        self.recordarray = np.array(self.recordarray, dtype=self.header_dtype)
        self.iposarray = np.array(self.iposarray, dtype=np.int64)

        return

    def _skip_record(self, header):
        """
        Skip over this record, not counting header and header2.
        """
        nlay = abs(header['nlay'])
        nrow = header['nrow']
        ncol = header['ncol']
        imeth = header['imeth']
        if imeth == 0:
            nbytes = (nrow * ncol * nlay * self.realtype(1).nbytes)
        elif imeth == 1:
            nbytes = (nrow * ncol * nlay * self.realtype(1).nbytes)
        elif imeth == 2:
            nlist = binaryread(self.file, np.int32)[0]
            nbytes = nlist * (np.int32(1).nbytes + self.realtype(1).nbytes)
        elif imeth == 3:
            nbytes = (nrow * ncol * self.realtype(1).nbytes)
            nbytes += (nrow * ncol * np.int32(1).nbytes)
        elif imeth == 4:
            nbytes = (nrow * ncol * self.realtype(1).nbytes)
        elif imeth == 5:
            nauxp1 = binaryread(self.file, np.int32)[0]
            naux = nauxp1 - 1
            for i in range(naux):
                temp = binaryread(self.file, str, charlen=16)
            nlist = binaryread(self.file, np.int32)[0]
            if self.verbose: 
                print('nlist: ', nlist)
                print('\n')
            nbytes = nlist * (np.int32(1).nbytes + self.realtype(1).nbytes + 
                              naux * self.realtype(1).nbytes)
        else:
            raise Exception('invalid method code ' + str(imeth))
        if nbytes != 0:
            self.file.seek(nbytes, 1)
        return
                          
    def _get_header(self):
        """
        Read the file header
        """
        header1 = binaryread(self.file, self.header1_dtype, (1,))
        nlay = header1['nlay']
        if  nlay < 0:
            header2 = binaryread(self.file, self.header2_dtype, (1,))
        else:
            header2 = np.array([(0, 0., 0., 0.)], dtype=self.header2_dtype)
        fullheader = join_struct_arrays([header1, header2])
        return fullheader[0]

    def list_records(self):
        """
        Print a list of all of the records in the file
        """
        for rec in self.recordarray:
            print(rec)
        return

    def unique_record_names(self):
        """
        Get a list of unique record names in the file

        Returns
        ----------
        out : list of strings
            List of unique text names in the binary file.

        """
        return self.textlist

    def get_kstpkper(self):
        """
        Get a list of unique stress periods and time steps in the file

        Returns
        ----------
        out : list of (kstp, kper) tuples
            List of unique kstp, kper combinations in binary file.  kstp and
            kper values are zero-based.

        """
        kstpkper = []
        for kstp, kper in self.kstpkper:
            kstpkper.append((kstp - 1, kper - 1))
        return kstpkper

    def get_data(self, idx=None, kstpkper=None, totim=None, text=None,
                 verbose=False, full3D=False):
        """
        get data from the budget file.

        Parameters
        ----------
        idx : int
            The zero-based record number.  The first record is record 0.
        kstpkper : tuple of ints
            A tuple containing the time step and stress period (kstp, kper).
            The kstp and kper values are zero based.
        totim : float
            The simulation time.
        text : str
            The text identifier for the record.  Examples include
            'RIVER LEAKAGE', 'STORAGE', 'FLOW RIGHT FACE', etc.
        verbose : boolean
            If true, print additional information to to the screen during the
            extraction.  (Default is False).
        full3D : boolean
            If true, then return the record as a three dimensional numpy
            array, even for those list-style records writen as part of a
            'COMPACT BUDGET' MODFLOW budget file.  (Default is False.)

        Returns
        ----------
        recordlist : list of records
            A list of budget objects.  The structure of the returned object
            depends on the structure of the data in the cbb file.

            If full3D is True, then this method will return a numpy masked
            array of size (nlay, nrow, ncol) for those list-style
            'COMPACT BUDGET' records written by MODFLOW.

        See Also
        --------

        Notes
        -----

        Examples
        --------

        """
        # trap for totim error
        if totim is not None:
            if len(self.times) == 0:
                errmsg = '''This is an older style budget file that
                         does not have times in it.  Use the MODFLOW 
                         compact budget format if you want to work with 
                         times.  Or you may access this file using the
                         kstp and kper arguments or the idx argument.'''
                raise Exception(errmsg)

        # check and make sure that text is in file
        if text is not None:
            text16 = None
            for t in self.unique_record_names():
                if text.encode().upper() in t:
                    text16 = t
                    break
            if text16 is None:
                errmsg = 'The specified text string is not in the budget file.'
                raise Exception(errmsg)

        if kstpkper is not None:
            kstp1 = kstpkper[0] + 1
            kper1 = kstpkper[1] + 1
            if text is None:
                select_indices = np.where(
                    (self.recordarray['kstp'] == kstp1) &
                    (self.recordarray['kper'] == kper1))
            else:
                select_indices = np.where(
                    (self.recordarray['kstp'] == kstp1) &
                    (self.recordarray['kper'] == kper1) &
                    (self.recordarray['text'] == text16))

        elif totim is not None:
            if text is None:
                select_indices = np.where(
                    (self.recordarray['totim'] == totim))
            else:
                select_indices = np.where(
                    (self.recordarray['totim'] == totim) &
                    (self.recordarray['text'] == text16))

        # allow for idx to be a list or a scalar
        elif idx is not None:
            if isinstance(idx, list):
                select_indices = idx
            else:
                select_indices = [idx]

        # case where only text is entered
        elif text is not None:
            select_indices = np.where((self.recordarray['text'] == text16))

        # build and return the record list
        if isinstance(select_indices, tuple):
            select_indices = select_indices[0]
        recordlist = []
        for idx in select_indices:
            rec = self.get_record(idx, full3D=full3D, verbose=verbose)
            recordlist.append(rec)

        return recordlist

    def get_record(self, idx, full3D=False, verbose=False):
        """
        Get a single data record from the budget file.

        Parameters
        ----------
        idx : int
            The zero-based record number.  The first record is record 0.
        verbose : boolean
            If true, print additional information to to the screen during the
            extraction.  (Default is False).
        full3D : boolean
            If true, then return the record as a three dimensional numpy
            array, even for those list-style records writen as part of a
            'COMPACT BUDGET' MODFLOW budget file.  (Default is False.)

        Returns
        ----------
        record : a single data record
            The structure of the returned object depends on the structure of
            the data in the cbb file.

            If full3D is True, then this method will return a numpy masked
            array of size (nlay, nrow, ncol) for those list-style
            'COMPACT BUDGET' records written by MODFLOW.

        See Also
        --------

        Notes
        -----

        Examples
        --------

        """
        # idx must be an ndarray
        if np.isscalar(idx):
            idx = np.array([idx])

        header = self.recordarray[idx]
        ipos = self.iposarray[idx]
        self.file.seek(ipos, 0)
        imeth = header['imeth'][0]

        t = header['text'][0]
        s = 'Returning ' + str(t).strip() + ' as '

        nlay = abs(header['nlay'][0])
        nrow = header['nrow'][0]
        ncol = header['ncol'][0]

        # default method
        if imeth == 0:
            if verbose:
                s += 'an array of shape ' + str((nlay, nrow, ncol))
                print(s)
            return binaryread(self.file, self.realtype(1),
                              shape=(nlay, nrow, ncol))
        # imeth 1
        elif imeth == 1:
            if verbose:
                s += 'an array of shape ' + str( (nlay, nrow, ncol) )
                print(s)
            return binaryread(self.file, self.realtype(1),
                              shape=(nlay, nrow, ncol))

        # imeth 2
        elif imeth == 2:
            nlist = binaryread(self.file, np.int32)
            dtype = np.dtype([('node', np.int32), ('q', self.realtype)])
            if verbose:
                if full3D:
                    s += 'a numpy masked array of size ({}{}{})'.format(nlay,
                                                                        nrow,
                                                                        ncol)
                else:
                    s += 'a dictionary of size ' + str(nlist)
                print(s)
            data = binaryread(self.file, dtype, shape=(nlist,))
            if full3D:
                return self.create3D(data, nlay, nrow, ncol)
            else:
                return dict(zip(data['node'], data['q']))

        # imeth 3
        elif imeth == 3:
            ilayer = binaryread(self.file, np.int32, shape=(nrow, ncol))
            data = binaryread(self.file, self.realtype(1), shape=(nrow, ncol))
            if verbose:
                if full3D:
                    s += 'a numpy masked array of size ({}{}{})'.format(nlay,
                                                                        nrow,
                                                                        ncol)
                else:
                    s += 'a list of two 2D arrays.  '
                    s += 'The first is an integer layer array of shape  ' + str(
                                                            (nrow, ncol))
                    s += 'The second is real data array of shape  ' + str(
                                                        (nrow, ncol) )
                print(s)
            if full3D:
                out = np.ma.zeros((nlay, nrow, ncol), dtype=np.float32)
                out.mask = True
                vertical_layer = ilayer[0] - 1  # This is always the top layer
                out[vertical_layer, :, :] = data
                return out
            else:
                return [ilayer, data]

        # imeth 4
        elif imeth == 4:
            if verbose:
                s += 'a 2d array of shape ' + str((nrow, ncol))
                print(s)
            return binaryread(self.file, self.realtype(1), shape=(nrow, ncol))

        # imeth 5
        elif imeth == 5:
            nauxp1 = binaryread(self.file, np.int32)
            naux = nauxp1 - 1
            l = [('node', np.int32), ('q', self.realtype)]
            for i in range(naux):
                auxname = binaryread(self.file, str, charlen=16)
                l.append( (auxname, self.realtype))
            dtype = np.dtype(l)                
            nlist = binaryread(self.file, np.int32)
            data = binaryread(self.file, dtype, shape=(nlist,))
            if full3D:
                if verbose:
                    s += 'a list array of shape ({}, {}, {})'.format(nlay, nrow, ncol)
                    print(s)
                return self.create3D(data, nlay, nrow, ncol)
            else:
                if verbose:
                    s += 'a dictionary of size ' + str(nlist)
                    print(s)
                return dict(zip(data['node'], data['q']))

        # should not reach this point
        return

    def create3D(self, data, nlay, nrow, ncol):
        """
        Convert a dictionary of {node: q, ...} into a numpy masked array.
        In most cases this should not be called directly by the user unless
        you know what you're doing.  Instead, it is used as part of the
        full3D keyword for get_data.

        Parameters
        ----------
        data : dictionary
            Dictionary with node keywords and flows (q) items.

        nlay, nrow, ncol : int
            Number of layers, rows, and columns of the model grid.

        Returns
        ----------
        out : numpy masked array
            List contains unique simulation times (totim) in binary file.

        """
        out = np.ma.zeros((nlay*nrow*ncol), dtype=np.float32)
        out.mask = True
        for [node, q] in zip(data['node'], data['q']):
            idx = node - 1
            out.data[idx] += q
            out.mask[idx] = False
        return np.ma.reshape(out, (nlay, nrow, ncol))

    def get_times(self):
        """
        Get a list of unique times in the file

        Returns
        ----------
        out : list of floats
            List contains unique simulation times (totim) in binary file.

        """
        return self.times

    def get_nrecords(self):
        """
        Return the number of records in the file

        Returns
        -------

        out : int
            Number of records in the file.

        """
        return self.recordarray.shape[0]

    def close(self):
        """
        Close the file handle
        """
        self.file.close()
        return
