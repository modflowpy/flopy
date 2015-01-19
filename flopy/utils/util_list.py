import os
import warnings
import numpy as np

class mflist(object):
    """
    a generic object for handling transient boundary condition lists

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.mf.Modflow`) to which
        this package will be added.
    dtype : np.dtype
        a numpy dtype describing the columns of the list data
    data : varies
        the data of the transient list (optional). (the default is None)

    Attributes
    ----------
    mxact : int
        the max number of active bc for any stress period

    Methods
    -------
    add_record(kper,index,value) : None
        add a record to stress period kper at index location
    write_transient(f) : None
        write the transient sequence to the model input file f
    check_kij() : None
        checks for boundaries outside of model domain - issues warnings only

    See Also
    --------

    Notes
    -----

    Examples
    --------

    """

    def __init__(self, model, dtype, data = None):
        self.model = model
        assert isinstance(dtype, np.dtype)
        self.__dtype = dtype
        self.__vtype = {}
        self.__data = {}
        if data is not None:
            self.__cast_data(data)

    @property
    def data(self):
        return self.__data

    @property
    def vtype(self):
        return self.__vtype

    @property
    def dtype(self):
        return self.__dtype

    # Get the itmp for a given kper
    def get_itmp(self, kper):
        if (kper not in self.__data.keys()):
            return None
        # If an external file, have to load it
        if (self.__vtype[kper] == str):
            return self.__fromfile(self.__data[kper]).shape[0]
        if (self.__vtype[kper] == np.recarray):
            return self.__data[kper].shape[0]
        # If not any of the above, it must be an int
        return self.__data[kper]

    @property
    def mxact(self):
        mxact = 0
        for kper in self.__data.keys():
            mxact = max(mxact, self.get_itmp(kper))
        return mxact

    # Get the numpy savetxt-style fmt string that corresponds to the dtype
    @property
    def fmt_string(self):
        fmt_string = ''
        for field in self.dtype.descr:
            vtype = field[1][1].lower()
            if (vtype == 'i'):
                fmt_string += ' %9d'
            elif (vtype == 'f'):
                fmt_string += ' %9f'
            elif (vtype == 'o'):
                fmt_string += ' %s'
            elif (vtype == 's'):
                raise Exception("mflist error: '\str\' type found it dtype." +\
                                " This gives unpredictable results when " +\
                                "recarray to file - change to \'object\' type")
            else:
                raise Exception("mflist.fmt_string error: unknown vtype " +\
                                "in dtype:" + vtype)
        return fmt_string


    # Private method to cast the data argument
    # Should only be called by the constructor
    def __cast_data(self, data):
        # If data is a list, then all we can do is try to cast it to
        # an ndarray, then cast again to a recarray
        if isinstance(data, list):
            #warnings.warn("mflist casting list to array")
            try:
                data = np.array(data)
            except Exception as e:
                raise Exception("mflist error: casting list to ndarray: " +\
                                str(e))

        # If data is a dict, the we have to assume it is keyed on kper
        if isinstance(data, dict):
            for kper, d in data.iteritems():
                assert isinstance(kper, int), "mflist error: data dict key " +\
                                              " \'{0:s}\' " +\
                                              "not integer: ".format(kper) +\
                                              str(type(kper))
                # Same as before, just try...
                if isinstance(d, list):
                    #warnings.warn("mflist: casting list to array at " +\
                    #               "kper {0:d}".format(kper))
                    try:
                        d = np.array(d)
                    except Exception as e:
                        raise Exception("mflist error: casting list " +\
                                        "to ndarray: " + str(e))

                if isinstance(d, np.recarray):
                    self.__cast_recarray(kper, d)
                elif isinstance(d, np.ndarray):
                    self.__cast_ndarray(kper, d)
                elif isinstance(d, int):
                    self.__cast_int(kper, d)
                elif isinstance(d, str):
                    self.__cast_str(kper, d)
                else:
                    raise Exception("mflist error: unsupported data type: " +
                                    str(type(d)) + " at kper " +
                                    "{0:d}".format(kper))

        # A single recarray - same mflist for all stress periods
        elif isinstance(data, np.recarray):
            self.__cast_recarray(0, data)
        # A single ndarray
        elif isinstance(data, np.ndarray):
            self.__cast_ndarray(0, data)
        # A single filename
        elif isinstance(data, str):
            self.__cast_str(0, data)
        else:
            raise Exception("mflist error: unsupported data type: " +\
                            str(type(data)))

    def __cast_str(self, kper, d):
        # If d is a string, assume it is a filename and check that it exists
        assert os.path.exists(d), "mflist error: dict filename (string) \'" +\
                                  d + "\' value for " +\
                                  "kper {0:d} not found".format(kper)
        self.__data[kper] = d
        self.__vtype[kper] = str

    def __cast_int(self, kper, d):
        # If d is an integer, then it must be 0 or -1
        if (d > 0):
            raise Exception("mflist error: dict integer value for " +\
                            "kper {0:10d} must be 0 or -1, " +\
                            "not {1:10d}".format(kper, d))
        if (d == 0):
            self.__data[kper] = 0
            self.__vtype[kper] = None
        else:
            if (kper == 0):
                raise Exception("mflist error: dict integer value for " +\
                                "kper 0 for cannot be negative")
            self.__data[kper] = -1
            self.__vtype[kper]= None

    def __cast_recarray(self, kper, d):
        assert d.dtype == self.__dtype, "mflist error: recarray dtype: " +\
                                        str(d.dtype) + " doesn't match " +\
                                        "self dtype: " + str(self.dtype)
        self.__data[kper] = d
        self.__vtype[kper] = np.recarray

    def __cast_ndarray(self, kper, d):
        d = np.atleast_2d(d)
        if (d.dtype != self.__dtype):
            assert d.shape[1] == len(self.dtype), "mflist error: ndarray " +\
                                                  "shape " + str(d.shape) +\
                                                  " doesn't match dtype " +\
                                                  "len: " +\
                                                  str(len(self.dtype))
            #warnings.warn("mflist: ndarray dtype does not match self " +\
            #               "dtype, trying to cast")
        try:
            self.__data[kper] = np.core.records.fromarrays(d.transpose(),
                                dtype = self.dtype)
        except Exception as e:
            raise Exception("mflist error: casting ndarray to recarray: " +\
                            str(e))
        self.__vtype[kper] = np.recarray

    def add_record(self, kper, index, values):
        # Add a record to possible already set list for a given kper
        assert len(index) + len(values) == len(self.dtype), \
            "mflist.add_record() error: length of index arg +" +\
            "length of value arg != length of self dtype"
        # If we already have something for this kper, then add to it
        if (kper in self.__data.keys()):
            # If a 0 or -1, reset
            if (self.vtype[kper] == int):
                self.__data[kper] = self.get_empty(1)
                self.__vtype[kper] = np.recarray
            # If filename, load into recarray
            if (self.vtype[kper] == str):
                d = self.__fromfile(self.data[kper])
                d.resize(d.shape[0], d.shape[1])
                self.__data[kper] = d
                self.__vtype[kper] = np.recarray
            # Extend the recarray
            if (self.vtype[kper] == np.recarray):
                shape = self.__data[kper].shape
                self.__data[kper].resize(shape[0] + 1, shape[1])
        else:
            self.__data[kper] = self.get_empty(1)
            self.__vtype[kper] = np.recarray
        rec = list(index)
        rec.extend(list(values))
        try:
            self.__data[kper][-1] = tuple(rec)
        except Exception as e:
            raise Exception("mflist.add_record() error: adding record to " +\
                            "recarray: " + str(e))

    def __getitem__(self, kper):
        # Get the recarray for a given kper
        # If the data entry for kper is a string, 
        # return the corresponding recarray,
        # but don't reset the value in the data dict
        print self.data.keys(), kper, kper in self.data.keys()
        assert kper in self.data.keys(), "mflist.__getitem__() kper " +\
                                         str(kper) + " not in data.keys()"
        if (self.vtype[kper] == int):
            if (self.data[kper] == 0):
                return self.get_empty()
            else:
                return self.data[self.__find_last_kper(kper)]
        if (self.vtype[kper] == str):
            return self.__fromfile(self.data[kper])
        if (self.vtype[kper] == np.recarray):
            return self.data[kper]

    def __setitem__(self, key, value):
        raise NotImplementedError("mflist.__setitem__() not implemented")

    def __fromfile(self, f):
        #d = np.fromfile(f,dtype=self.dtype,count=count)
        try:
            d = np.genfromtxt(f, dtype = self.dtype)
        except Exception as e:
            raise Exception("mflist.__fromfile() error reading recarray " +\
                            "from file " + str(e))
        return d

    def write_transient(self, f, single_per = None):
        #write the transient sequence described by the data dict
        nr, nc, nl, nper = self.model.get_nrow_ncol_nlay_nper()
        assert isinstance(f, file), "mflist.write() error: " +\
                                    "f argument must be a file handle"
        kpers = self.data.keys()
        kpers.sort()
        # Assert 0 in kpers,"mflist.write() error: kper 0 not defined"
        first = kpers[0]
        if (single_per == None):
            loop_over_kpers = range(0, max(nper, max(kpers)) + 1)
        else:
            if (not isinstance(single_per, list)):
                single_per = [single_per]
            loop_over_kpers = single_per

        for kper in loop_over_kpers:
            # Fill missing early kpers with 0
            if (kper < first):
                itmp = 0
                kper_vtype = int
            elif (kper in kpers):
                kper_data = self.__data[kper]
                kper_vtype = self.__vtype[kper]
                if (kper_vtype == str):
                    if (not self.model.free_format):
                        kper_data = self.__fromfile(kper_data)
                        kper_vtype = np.recarray
                    itmp = self.get_itmp(kper)
                if (kper_vtype == np.recarray):
                    itmp = kper_data.shape[0]
                elif (kper_vtype == int) or (kper_vtype is None):
                    itmp = kper_data
            # Fill late missing kpers with -1
            else:
                itmp = -1
                kper_vtype = int
            
            f.write(" {0:9d} {1:9d} # stress period {2:d}\n".format(itmp,
                    0, kper))

            if (kper_vtype == np.recarray):
                self.__tofile(f, kper_data)
            elif (kper_vtype == str):
                f.write("         open/close " + kper_data + '\n')

    def __tofile(self, f, data):
        # Write the recarray (data) to the file (or file handle) f
        assert isinstance(data, np.recarray), "mflist.__tofile() data arg " +\
                                              "not a recarray"

        # Add one to the kij indices
        names = self.dtype.names
        lnames = []
        [lnames.append(name.lower()) for name in names]
        for idx in ['k', 'i', 'j']:
            if (idx in lnames):
                data[idx] += 1
        np.savetxt(f, data, fmt = self.fmt_string, delimiter = '')

    def check_kij(self):
        names = self.dtype.names
        if ('k' not in names) or ('i' not in names) or ('j' not in names):
            warnings.warn("mflist.check_kij(): index fieldnames \'k,i,j\' " +\
                          "not found in self.dtype names: " + str(names))
            return
        nr, nc, nl, nper = self.model.get_nrow_ncol_nlay_nper()
        if (nl == 0):
            warnings.warn("mflist.check_kij(): unable to get dis info from " +\
                          "model")
            return
        for kper in self.data.keys():
            out_idx = []
            data = self[kper]
            if (data is not None):
                k = data['k']
                k_idx = np.where(np.logical_or(k < 0, k >= nl))
                if (k_idx[0].shape[0] > 0):
                    out_idx.extend(list(k_idx[0]))
                i = data['i']
                i_idx = np.where(np.logical_or(i < 0, i >= nr))
                if (i_idx[0].shape[0] > 0):
                    out_idx.extend(list(i_idx[0]))
                j = data['j']
                j_idx = np.where(np.logical_or(j < 0, j >= nc))
                if (j_idx[0].shape[0]):
                    out_idx.extend(list(j_idx[0]))

                if (len(out_idx) > 0):
                    warn_str = "mflist.check_kij(): warning the following " +\
                               "indices are out of bounds in kper " +\
                               str(kper) + ':\n'
                    for idx in out_idx:
                        d = data[idx]
                        warn_str += " {0:9d} {1:9d} {2:9d}\n".format(d['k']
                                    + 1, d['i'] + 1, d['j'] + 1)
                    warnings.warn(warn_str)

    def __find_last_kper(self, kper):
        kpers = self.data.keys()
        kpers.sort()
        last = kpers[0]
        for kper in kpers:
            if (kper >= last):
                break
            if (self.vtype[kper] != int) or (self.data[kper] != -1):
                last = kper

        return kper
