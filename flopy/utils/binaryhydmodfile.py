import sys
import numpy as np
import struct as strct


class HydmodBinaryStatements:
    """
    Class for reading integer, real, double, and character data
    from hydmod output files.
    """
    # --byte definition
    integer = np.int32
    real = np.float32
    double = np.float64
    character = np.uint8
    integerbyte = 4
    realbyte = 4
    doublebyte = 8
    textbyte = 1

    def _read_integer(self):
        intvalue = strct.unpack('i', self.file.read(1 * self.integerbyte))[0]
        return intvalue

    def _read_real(self):
        realvalue = strct.unpack('f', self.file.read(1 * self.realbyte))[0]
        return realvalue

    def _read_double(self):
        doublevalue = strct.unpack('f', self.file.read(1 * self.doublebyte))[0]
        return doublevalue

    def _read_hyd_text(self, nchar=20):
        # textvalue=strct.unpack('cccccccccccccccc',self.file.read(16*self.textbyte))
        textvalue = np.fromfile(file=self.file, dtype=self.character, count=nchar).tostring()
        if not isinstance(textvalue, str):
            textvalue = textvalue.decode().strip()
        else:
            textvalue = textvalue.strip()

        return textvalue


class HydmodObs(HydmodBinaryStatements):
    """
    HydmodObs Class - used to read binary MODFLOW HYDMOD package output

    Parameters
    ----------
    filename : str
        Name of the hydmod output file
    double : boolean
        If true, set flag used to read a hydmod output file with double precision
        data. If false, set flag used to read hydmod output file with single
        precision data. (default is False)
    slurp : boolean
        If true, HydmodObs will be instantiated to access all data from the hydmod
        output file at once using the .slurp() method. (default is False)
    verbose : boolean
        If true, print additional information to to the screen during the
        extraction.  (default is False)
    hydlbl_len : int
        Length of hydmod labels. (default is 20)

    Returns
    -------
    None

    """

    def __init__(self, filename, double=False, slurp=False, verbose=False, hydlbl_len=20):
        """
        Class constructor.

        """
        # initialize class information
        self.skip = True
        self.double = bool(double)
        self.verbose = verbose
        # --open binary head file
        self.file = open(filename, 'rb')
        # NHYDTOT,ITMUNI
        self.nhydtot = self._read_integer()
        self.itmuni = self._read_integer()
        if self.nhydtot < 0:
            self.double = True
            self.nhydtot = abs(self.nhydtot)
        self.v = np.empty((self.nhydtot), dtype='float')
        self.v.fill(1.0E+32)
        ctime = self._read_hyd_text(nchar=4)
        self.hydlbl_len = int(hydlbl_len)
        # read HYDLBL
        hydlbl = []
        for idx in range(0, self.nhydtot):
            cid = self._read_hyd_text(self.hydlbl_len)
            hydlbl.append(cid)
        self.hydlbl = np.array(hydlbl)
        if self.verbose:
            print(self.hydlbl)
        if not slurp:
            self._allow_slurp = False
            # set position
            self.datastart = self.file.tell()
            # get times
            self.times = self._time_list()
        else:
            self._allow_slurp = True

    def get_time_list(self):
        """
        Get the times stored in a hydmod output file.

        Returns
        -------
        out : list of floats
            A list of times stored in a hydmod output file.

        Examples
        --------

        >>> import flopy
        >>> h = flopy.utils.HydmodObs('model.hyd.bin')
        >>> times = h.get_time_list()

        """
        return [time for time, ipos in self.times]

    def get_num_items(self):
        """
        Get the number of observations locations in a hydmod
        output file.

        Returns
        -------
        out : int
            The number of observations in a hydmod output file.

        Examples
        --------

        >>> import flopy
        >>> h = flopy.utils.HydmodObs('model.hyd.bin')
        >>> nitems = h.get_num_items()

        """
        return self.nhydtot

    def get_hyd_labels(self):
        """
        Get the observation labels in a hydmod output file.

        Returns
        -------
        out : list of strings
            A list of the observation labels in a hydmod output
            file.

        Examples
        --------

        >>> import flopy
        >>> h = flopy.utils.HydmodObs('model.hyd.bin')
        >>> labels = h.get_hyd_labels()

        """
        return self.hydlbl

    def slurp(self):
        """

        Returns
        -------
        data : numpy structured array (times, nhyd+1)
            Simulated values for all times in the hydmod output file. The totime for all
            time steps is also included in data.

        Examples
        --------

        >>> import flopy
        >>> h = flopy.utils.HydmodObs('model.hyd.bin', slurp=True)
        >>> times = h.slurp()

        """
        if not self._allow_slurp:
            raise ValueError('Cannot use .slurp() method if slurp=False in instantiation.')

        if self.double:
            float_type = 'f8'
        else:
            float_type = 'f4'
        dtype_list = [('totim', float_type)]
        for site in self.hydlbl:
            if not isinstance(site, str):
                site_name = site.decode().strip()
            else:
                site_name = site.strip()
            dtype_list.append((site_name, float_type))
        dtype = np.dtype(dtype_list)
        data = np.fromfile(self.file, dtype, count=-1)
        return data

    def get_values(self, idx=0, totim=None):
        """

        Parameters
        ----------
        idx : int
            The zero-based time step number to extract from the
            hydmod output file. (default is 0)
        totim : float
            The simulation time to extract from the hydmod output
            file. (default is None)

        Returns
        -------
        totim : float
            Simulation time extracted from the hydmod output file.
        v : numpy array (nhyd)
            Simulated values at the specified idx or totim
        success : boolean
            Boolean indicating if the data extraction was successful.

        Examples
        --------

        >>> import flopy
        >>> h = flopy.utils.HydmodObs('model.hyd.bin')
        >>> t, data, success = h.get_values(idx=0)

        """
        iposition = None
        if totim is None:
            try:
                iposition = int(self.times[idx, 1])
            except:
                print('Error: could not find the specified time step (idx) in the hydmod file')
        else:
            for t, ipos in self.times:
                if t == totim:
                    iposition = int(ipos)
                    break
            if iposition is None:
                print('Error: could not find the specified totim [{}] in the hydmod file')
        self.file.seek(iposition)
        totim, v, success = self.__next__()
        if success:
            return totim, v, True
        else:
            self.v.fill(1.0E+32)
            return 0.0, self.v, False

    def get_time_gage(self, record=None, idx=0, lblstrip=6):
        """
        Get data for a selected observation location using the
        observation name or observation number.

        Parameters
        ----------
        record : str
            observation name. (default is None)
        idx : int
            zero-based observation number. (default is 0)
        lblstrip : int
            number of characters to strip from the beginning of the
            hydmod label. (default is 6)

        Returns
        -------
        gage_record : numpy array (len(times), 2)
            value for the record for all times saved in the hydmod output file


        Examples
        --------

        >>> import flopy
        >>> h = flopy.utils.HydmodObs('model.hyd.bin')
        >>> data = h.get_time_gage(record='OBS1')
        >>> data2 = h.get_time_gage(idx=2)

        """

        if idx is None:
            idx = -1
            try:
                idx = int(record) - 1
                if idx >= 0 and idx < self.nhydtot:
                    if self.verbose:
                        print('retrieving HYDMOD observation record [{0}]'.format(idx + 1))
                else:
                    print('Error: HYDMOD observation record {0} not found'.format(record.strip().lower()))
            except:
                for icnt, cid in enumerate(self.hydlbl):
                    if lblstrip > 0:
                        tcid = cid[lblstrip:len(cid)]
                    else:
                        tcid = cid
                    if record.strip().lower() == tcid.strip().lower():
                        idx = icnt
                        if self.verbose:
                            print('retrieving HYDMOD observation record [{0}] {1}'.format(idx + 1, record.strip().lower()))
                        break
                if idx == -1:
                    print('Error: HYDMOD observation record {0} not found'.format(record.strip().lower()))
        else:
            if idx < 0 or idx+1 > self.nhydtot:
                print('Error: HYDMOD observation index {0} not found'.format(idx))
                print('Error: HYDMOD observation index must be between 0 and {0} not found'.format(self.nhydtot-1))
                idx = -1

        gage_record = np.zeros(2, dtype=np.float)  # tottime plus observation
        if idx != -1 and idx < self.nhydtot:
            # --find offset to position
            ilen = self._get_point_offset(idx)
            # --get data
            for time_data in self.times:
                self.file.seek(int(time_data[1]) + ilen)
                if self.double:
                    v = float(self._read_double())
                else:
                    v = self._read_real()
                this_entry = np.array([float(time_data[0])])
                this_entry = np.hstack((this_entry, v))
                gage_record = np.vstack((gage_record, this_entry))
            # delete the first 'zeros' element
            gage_record = np.delete(gage_record, 0, axis=0)
        return gage_record

    def __iter__(self):
        return self

    def __next__(self):
        totim, success = self._read_header()
        if success:
            for idx in range(0, self.nhydtot):
                if self.double:
                    self.v[idx] = float(self._read_double())
                else:
                    self.v[idx] = self._read_real()
        else:
            if self.verbose:
                print('MODFLOW_HYDMOD object.next() reached end of file.')
            self.v.fill(1.0E+32)
        return totim, self.v, success

    def _get_point_offset(self, ipos):
        self.file.seek(self.datastart)
        lpos0 = self.file.tell()
        point_offset = int(0)
        totim, success = self._read_header()
        idx = (ipos)
        if self.double:
            lpos1 = self.file.tell() + idx * HydmodBinaryStatements.doublebyte
        else:
            lpos1 = self.file.tell() + idx * HydmodBinaryStatements.realbyte
        self.file.seek(lpos1)
        point_offset = self.file.tell() - lpos0
        return point_offset

    def _time_list(self):
        self.skip = True
        self.file.seek(self.datastart)
        times = []
        while True:
            current_position = self.file.tell()
            totim, v, success = self.__next__()
            if success:
                times.append([totim, current_position])
            else:
                self.file.seek(self.datastart)
                times = np.array(times)
                self.skip = False
                return times

    def _read_header(self):
        try:
            totim = self._read_real()
            return totim, True
        except:
            return -999., False
