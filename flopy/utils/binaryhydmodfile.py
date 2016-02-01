import sys
import numpy as np
import struct as strct


class HydmodBinaryData(object):
    """
    The HydmodBinaryData class is a class to that defines the data types for
    integer, floating point, and character data in HYDMOD binary
    files. The HydmodBinaryData class is the super class from which the
    specific derived class HydmodObs is formed.  This class should not be
    instantiated directly.

    """
    def __init__(self):

        self.integer = np.int32
        self.integerbyte = self.integer(1).nbytes

        self.character = np.uint8
        self.textbyte = 1

        return

    def set_float(self, precision):
        self.precision = precision
        if precision.lower() == 'double':
            self.real = np.float64
            self.floattype = 'f8'
        else:
            self.real = np.float32
            self.floattype = 'f4'
        self.realbyte = self.real(1).nbytes
        return


    def read_hyd_text(self, nchar=20):
        textvalue = self._read_values(self.character, nchar).tostring()
        if not isinstance(textvalue, str):
            textvalue = textvalue.decode().strip()
        else:
            textvalue = textvalue.strip()
        return textvalue

    def read_integer(self):
        return self._read_values(self.integer, 1)[0]

    def read_real(self):
        return self._read_values(self.real, 1)[0]

    def read_record(self, count):
        return self._read_values(self.dtype, count)

    def _read_values(self, dtype, count):
        return np.fromfile(self.file, dtype, count)


class HydmodObs(HydmodBinaryData):
    """
    HydmodObs Class - used to read binary MODFLOW HYDMOD package output

    Parameters
    ----------
    filename : str
        Name of the hydmod output file
    verbose : boolean
        If true, print additional information to to the screen during the
        extraction.  (default is False)
    hydlbl_len : int
        Length of hydmod labels. (default is 20)

    Returns
    -------
    None

    """

    def __init__(self, filename, verbose=False, hydlbl_len=20):
        """
        Class constructor.

        """
        super(HydmodObs, self).__init__()
        # initialize class information
        self.verbose = verbose
        # --open binary head file
        self.file = open(filename, 'rb')
        # NHYDTOT,ITMUNI
        self.nhydtot = self.read_integer()
        precision = 'single'
        if self.nhydtot < 0:
            self.nhydtot = abs(self.nhydtot)
            precision = 'double'
        self.set_float(precision)

        # continue reading the file
        self.itmuni = self.read_integer()
        self.v = np.empty(self.nhydtot, dtype=np.float)
        self.v.fill(1.0E+32)
        ctime = self.read_hyd_text(nchar=4)
        self.hydlbl_len = int(hydlbl_len)
        # read HYDLBL
        hydlbl = []
        for idx in range(0, self.nhydtot):
            cid = self.read_hyd_text(self.hydlbl_len)
            hydlbl.append(cid)
        self.hydlbl = np.array(hydlbl)

        # create dtype
        dtype = [('totim', self.floattype)]
        for site in self.hydlbl:
            if not isinstance(site, str):
                site_name = site.decode().strip()
            else:
                site_name = site.strip()
            dtype.append((site_name, self.floattype))
        self.dtype = np.dtype(dtype)

        self.data = None
        self._read_data()

    def get_times(self):
        """
        Get a list of unique times in the file

        Returns
        ----------
        out : list of floats
            List contains unique simulation times (totim) in binary file.

        """
        return self._get_selection(['totim']).tolist()

    def get_ntimes(self):
        """
        Get the number of times in the file

        Returns
        ----------
        out : int
            The number of simulation times (totim) in binary file.

        """
        return self.data['totim'].shape[0]

    def get_nobs(self):
        """
        Get the number of observations in the file

        Returns
        ----------
        out : tuple of int
            A tupe with the number of records and number of flow items
            in the file. The number of flow items is non-zero only if
            swrtype='flow'.

        """
        return self.nhydtot

    def get_obsnames(self):
        """
        Get a list of observation names in the file

        Returns
        ----------
        out : list of strings
            List of observation names in the binary file. totim is not
            included in the list of observation names.

        """
        return self.data.dtype.names[1:]

    def get_data(self, idx=None, obsname=None):
        """
        Get data from the observation file.

        Parameters
        ----------
        idx : int
            The zero-based record number.  The first record is record 0.
            (default is None)
        obsname : string
            The name of the observation to return. (default is None)

        Returns
        ----------
        data : numpy record array
            Array has size (ntimes, nitems). totim is always returned. nitems
            is 2 if idx or obsname is not None or nobs+1.

        See Also
        --------

        Notes
        -----
        If both idx and obsname are None, will return all of the observation
        data.

        Examples
        --------

        """
        if obsname is None and idx is None:
            return self.data.view(dtype=self.dtype)
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
        dtype2 = np.dtype(
                {name: self.data.dtype.fields[name] for name in names})
        return np.ndarray(self.data.shape, dtype2, self.data, 0,
                          self.data.strides)

