import numpy as np
from collections import OrderedDict

class binaryheader():
    """
    The binary_header class is a class to create headers for MODFLOW
    binary files    
    """
    def __init__(self, bintype=None, precision='single'):
        floattype = 'f4'
        if precision is 'double':
            floattype = 'f8'
        self.header_types = ['head','ucn']
        if bintype is None:
            self.header_type = None
        else:
            self.header_type = bintype.lower()
        if self.header_type in self.header_types:
            if self.header_type == 'head':
                self.dtype = np.dtype([('kstp','i4'),('kper','i4'),\
                                       ('pertim',floattype),('totim',floattype),\
                                       ('text','a16'),\
                                       ('ncol','i4'),('nrow','i4'),('ilay','i4')])
            elif self.header_type == 'ucn':
                self.dtype = np.dtype([('ntrans','i4'),('kstp','i4'),('kper','i4'),\
                                       ('totim',floattype),('text','a16'),\
                                       ('ncol','i4'),('nrow','i4'),('ilay','i4')])
            self.header = np.ones(1, self.dtype)
        else:
            self.dtype = None
            self.header = None
            print 'Specified {0} bintype is not available. Available bintypes are:'.format(self.header_type)
            for idx, t in enumerate(self.header_types):
                print '  {0} {1}'.format(idx+1, t)
        return

    def set_values(self, **kwargs):
        ikey = ['ntrans', 'kstp', 'kper', 'ncol', 'nrow', 'ilay']
        fkey = ['pertim', 'totim']
        ckey = ['text']
        for k in ikey:
            if kwargs.has_key(k):
                try:
                    self.header[0][k] = int(kwargs[k])
                except:
                    print '{0} key not available in {1} header dtype'.format(k, self.header_type)
        for k in fkey:
            if kwargs.has_key(k):
                try:
                    self.header[0][k] = float(kwargs[k])
                except:
                    print '{0} key not available in {1} header dtype'.format(k, self.header_type)
        for k in ckey:
            if kwargs.has_key(k):
                #--Convert to upper case to be consistent case used by MODFLOW text strings.
                #  Necessary to work with HeadFile and UcnFile routines
                ttext = kwargs[k].upper()
                if len(ttext) > 16:
                    text = text[0:16]
                else:
                    text = ttext
                self.header[0][k] = text
            else:
                self.header[0][k] = 'DUMMY TEXT'

    def get_dtype(self):
        return self.dtype

    def get_names(self):
        return self.dtype.names
        
    def get_values(self):
        if self.header is None:
            return None
        else:
            return self.header[0]

    @staticmethod
    def set_dtype(bintype=None, precision='single'):
        header = binaryheader(bintype=bintype, precision=precision)
        return header.dtype

    @staticmethod
    def create(bintype=None, **kwargs):
        header = binaryheader(bintype=bintype)
        if header.get_dtype() is not None:
            header.set_values(**kwargs)
        return header.get_values()

def binaryread_struct(file, vartype, shape=(1), charlen=16):
    '''Read text, a scalar value, or an array of values from a binary file.
       file is an open file object
       vartype is the return variable type: str, numpy.int32, numpy.float32, 
           or numpy.float64
       shape is the shape of the returned array (shape(1) returns a single value)
           for example, shape = (nlay, nrow, ncol)
       charlen is the length of the text string.  Note that string arrays cannot
           be returned, only multi-character strings.  Shape has no affect on strings.
    '''
    import struct
    import numpy as np
    
    #store the mapping from type to struct format (fmt)
    typefmtd = {np.int32:'i', np.float32:'f', np.float64:'d'}
        
    #read a string variable of length charlen
    if vartype is str:
        result = file.read(charlen*1)
        
    #read other variable types
    else:
        fmt = typefmtd[vartype]
        #find the number of bytes for one value
        numbytes = vartype(1).nbytes
        #find the number of values
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
    '''uses numpy to read from binary file.  This
       was found to be faster than the struct
       approach and is used as the default.
    '''
    
    #read a string variable of length charlen
    if vartype is str:
        result = file.read(charlen*1)     
    else:
        #find the number of values
        nval = np.core.fromnumeric.prod(shape)
        result = np.fromfile(file,vartype,nval)
        if nval == 1:
            result = result #[0]
        else:
            result = np.reshape(result, shape)
    return result

def join_struct_arrays(arrays):
    '''
    Simple function that can join two numpy structured arrays.
    '''
    newdtype = sum((a.dtype.descr for a in arrays), [])
    newrecarray = np.empty(len(arrays[0]), dtype = newdtype)
    for a in arrays:
        for name in a.dtype.names:
            newrecarray[name] = a[name]
    return newrecarray


class BinaryLayerFile(object):
    '''
    The BinaryLayerFile class is the super class from which specific derived
    classes are formed.  This class should not be instantiated directly
    '''
    def __init__(self, filename, precision, verbose):        
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

        if precision is 'single':
            self.realtype = np.float32
        elif precision is 'double':
            self.realtype = np.float64
        else:
            raise Exception('Unknown precision specified: ' + precision)
        
        #read through the file and build the pointer index
        self._build_index()
        
        #allocate the value array
        self.value = np.empty((self.nlay, self.nrow, self.ncol),
                              dtype=self.realtype)
        return
   

    def _build_index(self):
        '''
        Build the recordarray and iposarray, which maps the header information
        to the position in the binary file.
        '''        
        header = self.get_header()
        self.nrow = header['nrow']
        self.ncol = header['ncol']
        self.file.seek(0, 2)
        self.totalbytes = self.file.tell()
        self.file.seek(0, 0)        
        self.databytes = header['ncol'] * header['nrow'] * self.realtype(1).nbytes
        ipos = 0
        while ipos < self.totalbytes:           
            header = self.get_header()
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
            #key = (kstp, kper, pertim, totim, text, nrow, ncol, ilay)
            ipos = self.file.tell()
            self.iposarray.append(ipos)
            self.file.seek(self.databytes, 1)
            ipos = self.file.tell()

        #self.recordarray contains a recordarray of all the headers.
        self.recordarray = np.array(self.recordarray, dtype=self.header_dtype)
        self.iposarray = np.array(self.iposarray)
        self.nlay = np.max(self.recordarray['ilay'])
        return

    def get_header(self):
        '''
        Read the file header
        '''        
        header = binaryread(self.file, self.header_dtype, (1,))
        return header[0]

    def list_records(self):
        '''
        Print a list of all of the records in the file
        obj.list_records()
        '''
        for header in self.recordarray:
            print header
        return

    def _fill_value_array(self, kstp=0, kper=0, totim=0):
        '''
        Fill the three dimensional value array, self.value, for the
        specified kstp and kper value or totim value.
        '''

        if totim > 0.:
            keyindices = np.where((self.recordarray['totim'] == totim))[0]
        elif kstp > 0 and kper > 0:
            keyindices = np.where(
                                  (self.recordarray['kstp'] == kstp) &
                                  (self.recordarray['kper'] == kper))[0]
        else:
            raise Exception('Data not found...')

        #initialize head with nan and then fill it
        self.value[:, :, :] = np.nan
        for idx in keyindices:
            ipos = self.iposarray[idx]
            ilay = self.recordarray['ilay'][idx]
            if self.verbose:
                print 'Byte position in file: {0}'.format(ipos)
            self.file.seek(ipos, 0)
            self.value[ilay - 1, :, :] = binaryread(self.file, self.realtype, 
                                                shape=(self.nrow, self.ncol))
        return
    
    def get_times(self):
        '''
        Return a list of unique times in the file
        '''
        return self.times

    def get_kstpkper(self):
        '''
        Return a list of unique stress periods and time steps in the file
        '''
        return self.kstpkper

    def get_data(self, kstp=0, kper=0, idx=None, totim=0, mflay=None):
        '''
        Return a three dimensional value array for the specified kstp, kper
        pair or totim value, or return a two dimensional head array
        if the mflay argument is specified, where mflay is the MODFLOW layer
        number (starting at 1).
        '''
        if idx is not None:
            totim = self.recordarray['totim'][idx]
        self._fill_value_array(kstp, kper, totim)
        if mflay is None:
            return self.value
        else:
            return self.value[mflay-1, :, :]
        return

    def get_alldata(self, mflay=None, nodata = -9999):
        '''
        Return a four-dimensional array (ntimes,nlay,nrow,ncol) if mflay = None
        Return a three-dimensional arrayn (ntimes,nrow,ncol) if mflay is
        specified.  mflay is the MODFLOW layer number (i.e., it starts at 1)
        '''
        if mflay is None:
            h = np.zeros((self.nlay,self.nrow,self.ncol),dtype=np.float)
        else:
            h = np.zeros((self.nrow,self.ncol),dtype=np.float)
        rv = []
        for totim in self.times:
            h[:] = self.get_data(totim=totim, mflay=mflay)
            rv.append(h)
        rv = np.array(rv)
        rv[ rv == nodata ] = np.nan
        return rv

    def get_ts(self, k=0, i=0, j=0):
        '''
        Create and return a time series array of size [ntimes, nstations].
        
        The get_ts method can be called as ts = hdobj.get_ts(1, 2, 3),
        which will get the time series for layer 1, row 2, and column 3.
        The time value will be in the first column of the array.
        
        Alternatively, the get_ts method can be called with a list like
        ts = hdobj.get_ts( [(1, 2, 3), (2, 3, 4)] ), which will return
        the time series for two different cells.
        
        '''
        if isinstance(k, list):
            kijlist = k
            nstation = len(kijlist)
        else:
            kijlist = [ (k, i, j) ]
            nstation = 1
        result = np.empty( (len(self.times), nstation + 1), dtype=self.realtype)
        result[:, :] = np.nan
        result[:, 0] = np.array(self.times)

        istat = 1
        for k, i, j in kijlist:
            recordlist = []
            ioffset = ((i - 1) * self.ncol + j - 1) * self.realtype(1).nbytes
            for irec, header in enumerate(self.recordarray):
                ilay = header['ilay']
                if ilay != k:
                    continue
                ipos = self.iposarray[irec]
                self.file.seek(ipos + np.long(ioffset), 0)
                itim = np.where(result[:, 0] == header['totim'])[0]
                result[itim, istat] = binaryread(self.file, self.realtype)
            istat += 1
        return result

    def close(self):
        """
        close the file handle
        """
        self.file.close()
        return


class HeadFile(BinaryLayerFile):
    '''
    The HeadFile class provides simple ways to retrieve 2d and 3d 
    head arrays from a MODFLOW binary head file and time series
    arrays for one or more cells.
    
    A HeadFile object is created as
    hdobj = HeadFile(filename, precision='single')
    
    This class can also be used for a binary drawdown file as
    ddnobj = HeadFile(filename, precision='single', text='drawdown')
    
    The BinaryLayerFile class is built on a record array consisting of
    headers, which are record arrays of the modflow header information
    (kstp, kper, pertim, totim, text, nrow, ncol, ilay)
    and long integers, which are pointers to first bytes of data for
    the corresponding data array.
    '''
    def __init__(self, filename, text='head',precision='single', verbose=False):
        self.text = text
        self.header_dtype = binaryheader.set_dtype(bintype='Head',
                                                   precision=precision)
        super(HeadFile,self).__init__(filename, precision, verbose)


class UcnFile(BinaryLayerFile):
    '''
    The UcnFile class provides simple ways to retrieve 2d and 3d 
    concentration arrays from a MT3D binary head file and time series
    arrays for one or more cells.
    
    A UcnFile object is created as
    ucnobj = UcnFile(filename, precision='single')
    
    The BinaryLayerFile class is built on a record array consisting of
    headers, which are record arrays of the modflow header information
    (kstp, kper, pertim, totim, text, nrow, ncol, ilay)
    and long integers, which are pointers to first bytes of data for
    the corresponding data array.
    '''
    def __init__(self, filename, text='concentration',precision='single', verbose=False):
        self.text = text
        self.header_dtype = binaryheader.set_dtype(bintype='Ucn',
                                                   precision=precision)
        super(UcnFile,self).__init__(filename, precision, verbose)


class CellBudgetFile(object):
    '''
    The CellBudgetFile ...    
    '''
    def __init__(self, filename, precision='single', verbose=False):        
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

        if precision is 'single':
            self.realtype = np.float32
            h2dt = [('imeth', 'i4'), ('delt', 'f4'), ('pertim', 'f4'),
                    ('totim', 'f4')]
        elif precision is 'double':
            self.realtype = np.float64
            h2dt = [('imeth', 'i4'),('delt', 'f8'), ('pertim', 'f8'),
                    ('totim', 'f8')]
        else:
            raise Exception('Unknown precision specified: ' + precision)

        self.header1_dtype = np.dtype(h1dt)
        self.header2_dtype = np.dtype(h2dt)
        hdt = h1dt + h2dt
        self.header_dtype = np.dtype(hdt)

        #read through the file and build the pointer index
        self._build_index()
        
        #allocate the value array
        self.value = np.empty((self.nlay, self.nrow, self.ncol),
                              dtype=self.realtype)
        return
   
    def _build_index(self):
        '''
        Build the ordered dictionary, which maps the header information
        to the position in the binary file.
        '''        
        header = self.get_header()
        self.file.seek(0, 2)
        self.totalbytes = self.file.tell()
        self.file.seek(0, 0)
        self.databytes = (header['ncol'] * header['nrow'] * header['nlay'] 
                          * self.realtype(1).nbytes)
        self.recorddict = OrderedDict()
        ipos = 0
        while ipos < self.totalbytes:           
            header = self.get_header()
            if self.verbose:
                print header
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
                    print itxt + ': ' + str(header[itxt])
                print 'file position: ', ipos
                if int(header['imeth']) != 5:
                    print '\n'

            #store record and byte position mapping
            self.recorddict[tuple(header)] = ipos    #store the position right after header2
            self.recordarray.append(header)
            self.iposarray.append(ipos)  #store the position right after header2

            #skip over the data to the next record and set ipos
            self.skip_record(header)
            ipos = self.file.tell()

        #convert to numpy arrays
        self.recordarray = np.array(self.recordarray, dtype=self.header_dtype)
        self.iposarray = np.array(self.iposarray, dtype=np.int64)

        return

    def skip_record(self, header):
        '''
        Skip over this record, not counting header and header2.
        '''
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
            for i in xrange(naux):
                temp = binaryread(self.file, str, charlen=16)
            nlist = binaryread(self.file, np.int32)[0]
            if self.verbose: 
                print 'nlist: ', nlist
                print '\n'
            nbytes = nlist * (np.int32(1).nbytes + self.realtype(1).nbytes + 
                              naux * self.realtype(1).nbytes)
        else:
            raise Exception('invalid method code ' + str(imeth))
        if nbytes != 0:
            self.file.seek(nbytes, 1)
        return
                          
    def get_header(self):
        '''
        Read the file header
        '''
        header1 = binaryread(self.file, self.header1_dtype, (1,))
        nlay = header1['nlay']
        if  nlay < 0:
            header2 = binaryread(self.file, self.header2_dtype, (1,))
        else:
            header2 = np.array([(0, 0., 0., 0.)], dtype=self.header2_dtype)
        fullheader = join_struct_arrays([header1, header2])
        return fullheader[0]

    def list_records(self):
        '''
        Print a list of all of the records in the file
        '''
        for rec in self.recordarray:
            print rec
        return

    def unique_record_names(self):
        """
        Returns all unique record names
        """
        return self.textlist

    def get_kstpkper(self):
        """
        Return a list of unique stress periods and time steps in the file
        """
        return self.kstpkper

    def get_data(self, idx=None, kstpkper=None, totim=None, text=None,
                 verbose=False, full3D=False):
        """
        get data from the budget file.

        Parameters
        ----------
        idx : int
            The zero-based record number.  The first record is record 0.
        kstpkper : tuple of ints
            A tuple containing the time step and stress period (kstp, kper)
        totim : float
            The simulation time.

        Returns
        ----------
        A list of budget objects.  The structure of the returned object
        depends on the structure of the data in the cbb file.

        See Also
        --------

        Notes
        -----

        Examples
        --------

        """

        #trap for totim error
        if totim is not None:
            if len(self.times) == 0:
                errmsg = '''This is an older style budget file that
                         does not have times in it.  Use the MODFLOW 
                         compact budget format if you want to work with 
                         times.  Or you may access this file using the
                         kstp and kper arguments or the idx argument.'''
                raise Exception(errmsg)

        #check and make sure that text is in file
        if text is not None:
            text16 = None
            for t in self.unique_record_names():
                if text.upper() in t:
                    text16 = t
                    break
            if text16 is None:
                errmsg = 'The specified text string is not in the budget file.'
                raise Exception(errmsg)

        if kstpkper is not None:
            kstp = kstpkper[0]
            kper = kstpkper[1]
            if text is None:
                select_indices = np.where(
                    (self.recordarray['kstp'] == kstp) &
                    (self.recordarray['kper'] == kper))
            else:
                select_indices = np.where(
                    (self.recordarray['kstp'] == kstp) &
                    (self.recordarray['kper'] == kper) &
                    (self.recordarray['text'] == text16))

        elif totim is not None:
            if text is None:
                select_indices = np.where(
                    (self.recordarray['totim'] == totim))
            else:
                select_indices = np.where(
                    (self.recordarray['totim'] == totim) &
                    (self.recordarray['text'] == text16))

        #allow for idx to be a list or a scalar
        elif idx is not None:
            if isinstance(idx, list):
                select_indices = idx
            else:
                select_indices = [idx]

        #case where only text is entered
        elif text is not None:
            select_indices = np.where((self.recordarray['text'] == text16))

        #build and return the record list
        recordlist = []
        for idx in select_indices[0]:
            rec = self.get_record(idx, full3D=full3D, verbose=verbose)
            recordlist.append(rec)

        return recordlist

    def get_record(self, idx, full3D=False, verbose=False):
        """
        Get one record from the cell by cell flow file.
        idx is the record number to get.

        """

        #idx must be an ndarray
        if np.isscalar(idx):
            idx = np.array([idx])

        header = self.recordarray[idx]
        ipos = self.iposarray[idx]
        self.file.seek(ipos, 0)
        imeth = header['imeth'][0]

        t = header['text'][0]
        s = 'Returning ' + t.strip() + ' as '

        nlay = abs(header['nlay'][0])
        nrow = header['nrow'][0]
        ncol = header['ncol'][0]

        #default method
        if imeth == 0:
            if verbose:
                s += 'an array of shape ' + str((nlay, nrow, ncol))
                print s
            return binaryread(self.file, self.realtype(1),
                              shape=(nlay, nrow, ncol))
        #imeth 1
        elif imeth == 1:
            if verbose:
                s += 'an array of shape ' + str( (nlay, nrow, ncol) )
                print s           
            return binaryread(self.file, self.realtype(1),
                              shape=(nlay, nrow, ncol))

        #imeth 2
        elif imeth == 2:
            nlist = binaryread(self.file, np.int32)
            dtype = np.dtype([('node', np.int32), ('q', self.realtype)])
            if verbose:
                s += 'a list array of shape ' + str( nlist ) 
                print s  
            data = binaryread(self.file, dtype, shape=(nlist,))
            if full3D:
                return self.create3D(data, nlay, nrow, ncol)
            else:
                return dict(zip(data['node'],data['q']))

        #imeth 3
        elif imeth == 3:
            ilayer = binaryread(self.file, np.int32, shape=(nrow, ncol))
            data = binaryread(self.file, self.realtype(1), shape=(nrow, ncol))
            if verbose:
                s += 'a list of two 2D arrays.  '
                s += 'The first is an integer layer array of shape  ' + str( 
                                                        (nrow, ncol) )
                s += 'The second is real data array of shape  ' + str( 
                                                        (nrow, ncol) )
                print s
            return [ilayer, data]

        #imeth 4
        elif imeth == 4:
            if verbose:
                s += 'a 2d array of shape ' + str( (nrow, ncol) )
                print s
            return binaryread(self.file, self.realtype(1), shape=(nrow, ncol))

        #imeth 5
        elif imeth == 5:
            nauxp1 = binaryread(self.file, np.int32)
            naux = nauxp1 - 1
            l = [('node', np.int32), ('q', self.realtype)]
            for i in xrange(naux):
                auxname = binaryread(self.file, str, charlen=16)
                l.append( (auxname, self.realtype))
            dtype = np.dtype(l)                
            nlist = binaryread(self.file, np.int32)
            if verbose:
                s += 'a list array of shape ' + str(nlist)
                print s
            data = binaryread(self.file, dtype, shape=(nlist,))
            if full3D:
                return self.create3D(data, nlay, nrow, ncol)
            else:
                return dict(zip(data['node'], data['q']))

        #should not reach this point
        return

    def get_data_by_text(self, text):
        '''Returns one array of size (Ndata,nlay,nrow,ncol) of all
        records with text. Trailing spaces can be ignored'''
        idxlist = []
        keys = self.recorddict.keys()
        for i,h in enumerate(keys):
            # Remove any spaces. That avoids any problems with too many, etc.
            if h[2].replace(' ','') == text.replace(' ',''):
                idxlist.append(i)
        Ndata = len(idxlist)
        if Ndata == 0:
            print 'No such string: ', text
            return
        h = keys[idxlist[0]]
        nlay = abs(h[5])
        nrow = h[4]
        ncol = h[3]
        rv = np.empty((Ndata,nlay,nrow,ncol))
        for i,idx in enumerate(idxlist):
            header = self.recorddict.keys()[idx]
            ipos = self.recorddict[header]
            self.file.seek(ipos, 0)
            rv[i] = binaryread(self.file, self.realtype(1),
                               shape=(nlay, nrow, ncol))
        return rv

    def create3D(self, data, nlay, nrow, ncol):
        out = np.zeros((nlay*nrow*ncol), dtype=np.float32)
        for [node, q] in zip(data['node'], data['q']):
            idx = node - 1
            out[idx] += q
        return np.reshape(out, (nlay, nrow, ncol))

    def get_times(self):
        '''
        Return a list of unique times in the file
        '''
        return self.times

    def get_nrecords(self):
        """
        Return the number of records in the file
        """
        return self.recordarray.shape[0]

    def close(self):
        """
        close the file handle
        """
        self.file.close()
        return
