"""
Module to read MODFLOW output files.  The module contains shared
abstract classes that should not be directly accessed.

"""
from __future__ import print_function
import numpy as np

class Header():
    """
    The header class is an abstract base class to create headers for MODFLOW files
    """
    def __init__(self, filetype=None, precision='single'):
        floattype = 'f4'
        if precision == 'double':
            floattype = 'f8'
        self.header_types = ['head','ucn']
        if filetype is None:
            self.header_type = None
        else:
            self.header_type = filetype.lower()
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
            print('Specified {0} type is not available. Available types are:'.format(self.header_type))
            for idx, t in enumerate(self.header_types):
                print('  {0} {1}'.format(idx+1, t))
        return

    def get_dtype(self):
        """
        Return the dtype
        """
        return self.dtype

    def get_names(self):
        """
        Return the dtype names
        """
        return self.dtype.names
        
    def get_values(self):
        """
        Return the header values
        """
        if self.header is None:
            return None
        else:
            return self.header[0]


class LayerFile(object):
    """
    The LayerFile class is the abstract base class from which specific derived
    classes are formed.  LayerFile This class should not be instantiated directly.

    """
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

        if precision == 'single':
            self.realtype = np.float32
        elif precision == 'double':
            self.realtype = np.float64
        else:
            raise Exception('Unknown precision specified: ' + precision)
        
        #read through the file and build the pointer index
        self._build_index()
        return
   

    def _build_index(self):
        """
        Build the recordarray and iposarray, which maps the header information
        to the position in the formatted file.
        """
        raise Exception('Abstract method _build_index called in LayerFile.  This method needs to be overridden.')

    def list_records(self):
        """
        Print a list of all of the records in the file
        obj.list_records()

        """
        for header in self.recordarray:
            print(header)
        return
    
    def _get_data_array(self, kstp=0, kper=0, totim=0):
        """
        Get the three dimensional data array for the
        specified kstp and kper value or totim value.

        """

        if totim > 0.:
            keyindices = np.where((self.recordarray['totim'] == totim))[0]
        elif kstp > 0 and kper > 0:
            keyindices = np.where(
                                  (self.recordarray['kstp'] == kstp) &
                                  (self.recordarray['kper'] == kper))[0]
        else:
            raise Exception('Data not found...')

        #initialize head with nan and then fill it
        data = np.empty((self.nlay, self.nrow, self.ncol),
                         dtype=self.realtype)
        data[:, :, :] = np.nan
        for idx in keyindices:
            ipos = self.iposarray[idx]
            ilay = self.recordarray['ilay'][idx]
            if self.verbose:
                print('Byte position in file: {0}'.format(ipos))
            self.file.seek(ipos, 0)
            data[ilay - 1, :, :] = self._read_data()
        return data

    def get_times(self):
        """
        Get a list of unique times in the file

        Returns
        ----------
        out : list of floats
            List contains unique simulation times (totim) in binary file.

        """
        return self.times

    def get_kstpkper(self):
        """
        Get a list of unique stress periods and time steps in the file

        Returns
        ----------
        out : list of (kstp, kper) tuples
            List of unique kstp, kper combinations in binary file.  kstp and
            kper values are presently zero-based.

        """
        kstpkper = []
        for kstp, kper in self.kstpkper:
            kstpkper.append((kstp - 1, kper - 1))
        return kstpkper

    def get_data(self, kstpkper=(0, 0), idx=None, totim=0, mflay=None):
        """
        Get data from the file for the specified conditions.

        Parameters
        ----------
        idx : int
            The zero-based record number.  The first record is record 0.
        kstpkper : tuple of ints
            A tuple containing the time step and stress period (kstp, kper).
            These are zero-based kstp and kper values.
        totim : float
            The simulation time.
        mflay : integer
           MODFLOW zero-based layer number to return.  If None, then all
           all layers will be included. (Default is None.)

        Returns
        ----------
        data : numpy array
            Array has size (nlay, nrow, ncol) if mflay is None or it has size
            (nrow, ncol) if mlay is specified.

        See Also
        --------

        Notes
        -----

        Examples
        --------

        """
        # One-based kstp and kper for pulling out of recarray
        kstp1 = kstpkper[0] + 1
        kper1 = kstpkper[1] + 1
        if idx is not None:
            totim = self.recordarray['totim'][idx]
        data = self._get_data_array(kstp1, kper1, totim)
        if mflay is None:
            return data
        else:
            return data[mflay, :, :]

    def get_alldata(self, mflay=None, nodata=-9999):
        """
        Get all of the data from the file.

        Parameters
        ----------
        mflay : integer
           MODFLOW zero-based layer number to return.  If None, then all
           all layers will be included. (Default is None.)

        nodata : float
           The nodata value in the data array.  All array values that have the
           nodata value will be assigned np.nan.

        Returns
        ----------
        data : numpy array
            Array has size (ntimes, nlay, nrow, ncol) if mflay is None or it
            has size (ntimes, nrow, ncol) if mlay is specified.

        See Also
        --------

        Notes
        -----

        Examples
        --------

        """
        rv = []
        for totim in self.times:
            h = self.get_data(totim=totim, mflay=mflay)
            rv.append(h)
        rv = np.array(rv)
        rv[rv == nodata] = np.nan
        return rv

    def _read_data(self):
        """
        Read data from file

        """
        raise Exception('Abstract method _read_data called in LayerFile.  This method needs to be overridden.')

    def _build_kijlist(self, idx):
        if isinstance(idx, list):
            kijlist = idx
        elif isinstance(idx, tuple):
            kijlist = [idx]

        # Check to make sure that k, i, j are within range, otherwise
        # the seek approach won't work.  Can't use k = -1, for example.
        for k, i, j in kijlist:
            fail = False
            errmsg = 'Invalid cell index. Cell ' + str((k, i, j)) + ' not within model grid: ' + \
                     str((self.nlay, self.nrow, self.ncol))
            if k < 0 or k > self.nlay - 1:
                fail = True
            if i < 0 or i > self.nrow - 1:
                fail = True
            if j < 0 or j > self.ncol - 1:
                fail = True
            if fail:
                raise Exception(errmsg)
        return kijlist

    def _get_nstation(self, idx, kijlist):
        if isinstance(idx, list):
            return len(kijlist)
        elif isinstance(idx, tuple):
            return 1

    def _init_result(self, nstation):
        # Initialize result array and put times in first column
        result = np.empty((len(self.times), nstation + 1),
                           dtype=self.realtype)
        result[:, :] = np.nan
        result[:, 0] = np.array(self.times)
        return result

    def close(self):
        """
        Close the file handle.

        """
        self.file.close()
        return
