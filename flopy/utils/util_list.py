"""
util_list module.  Contains the mflist class.
 This classes encapsulates modflow-style list inputs away
 from the individual packages.  The end-user should not need to
 instantiate this class directly.

    some more info

"""
from __future__ import division, print_function

import os
import warnings
import numpy as np

try:
    from numpy.lib import NumpyVersion
    numpy114 = NumpyVersion(np.__version__) >= '1.14.0'
except ImportError:
    numpy114 = False


class MfList(object):
    """
    a generic object for handling transient boundary condition lists

    Parameters
    ----------
    package : package object
        The package object (of type :class:`flopy.pakbase.Package`) to which
        this MfList will be added.
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

    def __init__(self, package, data=None, dtype=None, model=None,
                 list_free_format=None, binary=False):

        if isinstance(data, MfList):
            for attr in data.__dict__.items():
                setattr(self, attr[0], attr[1])
            if model is None:
                self.model = package.parent
            else:
                self.model = model
            self.package = package
            return

        self.package = package
        if model is None:
            self.model = package.parent
        else:
            self.model = model
        try:
            self.sr = self.model.sr
        except:
            self.sr = None
        if dtype is None:
            assert isinstance(self.package.dtype, np.dtype)
            self.__dtype = self.package.dtype
        else:
            self.__dtype = dtype
        self.__binary = binary
        self.__vtype = {}
        self.__data = {}
        if data is not None:
            self.__cast_data(data)
        self.__df = None
        if list_free_format is None:
            if package.parent.version == "mf2k":
                list_free_format = False
        self.list_free_format = list_free_format
        return

    def get_empty(self, ncell=0):
        d = np.zeros((ncell, len(self.dtype)), dtype=self.dtype)
        d[:, :] = -1.0E+10
        return d

    def export(self, f, **kwargs):
        from flopy import export
        return export.utils.mflist_helper(f, self, **kwargs)

    def append(self,other):
        """ append the recarrays from one MfList to another
        Parameters
        ----------
            other: variable: an item that can be cast in to an MfList
                that corresponds with self
        Returns
        -------
            dict of {kper:recarray}
        """
        if not isinstance(other,MfList):
            other = MfList(self.package,data=other,dtype=self.dtype,
                           model=self.model,
                           list_free_format=self.list_free_format)
        assert isinstance(other,MfList),"MfList.append(): other arg must be "+\
                                        "MfList or dict, not {0}".format(type(other))

        other_kpers = list(other.data.keys())
        other_kpers.sort()

        self_kpers = list(self.data.keys())
        self_kpers.sort()

        new_dict = {}
        for kper in range(self.model.nper):
            other_data = other[kper].copy()
            self_data = self[kper].copy()

            other_len = other_data.shape[0]
            self_len = self_data.shape[0]



            if (other_len == 0 and self_len == 0) or\
               (kper not in self_kpers and kper not in other_kpers):
                continue


            elif self_len == 0:
                new_dict[kper] = other_data
            elif other_len == 0:
                new_dict[kper] = self_data
            else:
                new_len = other_data.shape[0] + self_data.shape[0]
                new_data = np.recarray(new_len,dtype=self.dtype)
                new_data[:self_len] = self_data
                new_data[self_len:self_len+other_len] = other_data
                new_dict[kper] = new_data


        return new_dict

    def drop(self, fields):
        """drop fields from an MfList

        Parameters
        ----------
        fields : list or set of field names to drop

        Returns
        -------
        dropped : MfList without the dropped fields
        """
        if not isinstance(fields, list):
            fields = [fields]
        names = [n for n in self.dtype.names if n not in fields]
        dtype = np.dtype([(k, d) for k, d in self.dtype.descr if k not in fields])
        spd = {}
        for k, v in self.data.items():
            # because np 1.9 doesn't support indexing by list of columns
            newarr = np.array([self.data[k][n] for n in names]).transpose()
            newarr = np.array(list(map(tuple, newarr)), dtype=dtype).view(np.recarray)
            for n in dtype.names:
                newarr[n] = self.data[k][n]
            spd[k] = newarr
        return MfList(self.package, spd, dtype=dtype)

    @property
    def data(self):
        return self.__data

    @property
    def df(self):
        if self.__df is None:
            self.__df = self.get_dataframe()
        return self.__df

    @property
    def vtype(self):
        return self.__vtype

    @property
    def dtype(self):
        return self.__dtype

    # Get the itmp for a given kper
    def get_itmp(self, kper):
        if kper not in list(self.__data.keys()):
            return None
        # If an external file, have to load it
        if self.__vtype[kper] == str:
            return self.__fromfile(self.__data[kper]).shape[0]
        if self.__vtype[kper] == np.recarray:
            return self.__data[kper].shape[0]
        # If not any of the above, it must be an int
        return self.__data[kper]

    @property
    def mxact(self):
        mxact = 0
        for kper in list(self.__data.keys()):
            mxact = max(mxact, self.get_itmp(kper))
        return mxact

    @property
    def fmt_string(self):
        """Returns a C-style fmt string for numpy savetxt that corresponds to
        the dtype"""
        if self.list_free_format is not None:
            use_free = self.list_free_format
        else:
            use_free = True
            if self.package.parent.bas6 is not None:
                use_free = self.package.parent.bas6.ifrefm
            # mt3d list data is fixed format
            if 'mt3d' in self.package.parent.version.lower():
                use_free = False
        fmts = []
        for field in self.dtype.descr:
            vtype = field[1][1].lower()
            if vtype == 'i' or vtype == 'b':
                if use_free:
                    fmts.append('%9d')
                else:
                    fmts.append('%10d')
            elif vtype == 'f':
                if use_free:
                    if numpy114:
                        # Use numpy's floating-point formatter (Dragon4)
                        fmts.append('%15s')
                    else:
                        fmts.append('%15.7E')
                else:
                    fmts.append('%10G')
            elif vtype == 'o':
                if use_free:
                    fmts.append('%9s')
                else:
                    fmts.append('%10s')
            elif vtype == 's':
                raise TypeError(
                        "MfList.fmt_string error: 'str' type found in dtype. "
                        "This gives unpredictable results when "
                        "recarray to file - change to 'object' type")
            else:
                raise TypeError("MfList.fmt_string error: unknown vtype in "
                                "field: {}".format(field))
        if use_free:
            fmt_string = ' ' + ' '.join(fmts)
        else:
            fmt_string = ''.join(fmts)
        return fmt_string

    # Private method to cast the data argument
    # Should only be called by the constructor
    def __cast_data(self, data):
        # If data is a list, then all we can do is try to cast it to
        # an ndarray, then cast again to a recarray
        if isinstance(data, list):
            # warnings.warn("MfList casting list to array")
            try:
                data = np.array(data)
            except Exception as e:
                raise Exception("MfList error: casting list to ndarray: " + \
                                str(e))

        # If data is a dict, the we have to assume it is keyed on kper
        if isinstance(data, dict):
            if len(list(data.keys())) == 0:
                raise Exception("MfList error: data dict is empty")
            for kper, d in data.items():
                try:
                    kper = int(kper)
                except Exception as e:
                    raise Exception("MfList error: data dict key " + \
                                    "{0:s} not integer: ".format(kper) + \
                                    str(type(kper)) + "\n" + str(e))
                # Same as before, just try...
                if isinstance(d, list):
                    # warnings.warn("MfList: casting list to array at " +\
                    #               "kper {0:d}".format(kper))
                    try:
                        d = np.array(d)
                    except Exception as e:
                        raise Exception("MfList error: casting list " + \
                                        "to ndarray: " + str(e))

                #super hack - sick of recarrays already
                #if (isinstance(d,np.ndarray) and len(d.dtype.fields) > 1):
                #    d = d.view(np.recarray)

                if isinstance(d, np.recarray):
                    self.__cast_recarray(kper, d)
                elif isinstance(d, np.ndarray):
                    self.__cast_ndarray(kper, d)
                elif isinstance(d, int):
                    self.__cast_int(kper, d)
                elif isinstance(d, str):
                    self.__cast_str(kper, d)
                else:
                    raise Exception("MfList error: unsupported data type: " +
                                    str(type(d)) + " at kper " +
                                    "{0:d}".format(kper))

        # A single recarray - same MfList for all stress periods
        elif isinstance(data, np.recarray):
            self.__cast_recarray(0, data)
        # A single ndarray
        elif isinstance(data, np.ndarray):
            self.__cast_ndarray(0, data)
        # A single filename
        elif isinstance(data, str):
            self.__cast_str(0, data)
        else:
            raise Exception("MfList error: unsupported data type: " + \
                            str(type(data)))

    def __cast_str(self, kper, d):
        # If d is a string, assume it is a filename and check that it exists
        assert os.path.exists(d), "MfList error: dict filename (string) \'" + \
                                  d + "\' value for " + \
                                  "kper {0:d} not found".format(kper)
        self.__data[kper] = d
        self.__vtype[kper] = str

    def __cast_int(self, kper, d):
        # If d is an integer, then it must be 0 or -1
        if (d > 0):
            raise Exception("MfList error: dict integer value for " + \
                            "kper {0:10d} must be 0 or -1, " + \
                            "not {1:10d}".format(kper, d))
        if (d == 0):
            self.__data[kper] = 0
            self.__vtype[kper] = None
        else:
            if (kper == 0):
                raise Exception("MfList error: dict integer value for " + \
                                "kper 0 for cannot be negative")
            self.__data[kper] = -1
            self.__vtype[kper] = None

    def __cast_recarray(self, kper, d):
        assert d.dtype == self.__dtype, "MfList error: recarray dtype: " + \
                                        str(d.dtype) + " doesn't match " + \
                                        "self dtype: " + str(self.dtype)
        self.__data[kper] = d
        self.__vtype[kper] = np.recarray

    def __cast_ndarray(self, kper, d):
        d = np.atleast_2d(d)
        if (d.dtype != self.__dtype):
            assert d.shape[1] == len(self.dtype), "MfList error: ndarray " + \
                                                  "shape " + str(d.shape) + \
                                                  " doesn't match dtype " + \
                                                  "len: " + \
                                                  str(len(self.dtype))
            # warnings.warn("MfList: ndarray dtype does not match self " +\
            #               "dtype, trying to cast")
        try:
            self.__data[kper] = np.core.records.fromarrays(d.transpose(),
                                                           dtype=self.dtype)
        except Exception as e:
            raise Exception("MfList error: casting ndarray to recarray: " + \
                            str(e))
        self.__vtype[kper] = np.recarray

    def get_dataframe(self, squeeze=True):
        """
        Cast recarrays for stress periods into single
        dataframe containing all stress periods. 
        
        Parameters
        ----------
        squeeze : bool
            Reduce number of columns in dataframe to only include
            stress periods where a variable changes.
        
        Returns
        -------
        df : dataframe
            Dataframe of shape nrow = ncells, ncol = nvar x nper. If 
            the squeeze option is choosen, nper is the number of 
            stress periods where at least one cells is different, 
            otherwise it is equal to the number of keys in MfList.data.
        
        Notes
        -----
        Requires pandas.
        """
        try:
            import pandas as pd
        except Exception as e:
            msg = 'MfList.get_dataframe() requires pandas'
            raise ImportError(msg)

        # make a dataframe of all data for all stress periods
        names = ['k', 'i', 'j']
        if 'MNW2' in self.package.name:
            names += ['wellid']

        # find relevant variable names
        # may have to iterate over the first stress period
        for per in range(self.model.nper):
            if hasattr(self.data[per], 'dtype'):
                varnames = list([n for n in self.data[per].dtype.names
                                 if n not in names])
                break

        # create list of dataframes for each stress period
        # each with index of k, i, j
        dfs = []
        for per in range(self.model.nper):
            recs = self.data[per]
            if recs is None or recs is 0:
                # add an empty dataframe if a stress period is
                # set to 0 (e.g. no pumping during a predevelopment
                # period)
                columns = names + list(['{}{}'.format(c, per)
                                        for c in varnames])
                dfi = pd.DataFrame(data=None, columns=columns)
                dfi = dfi.set_index(names)
            else:
                dfi = pd.DataFrame.from_records(recs)
                dfi = dfi.set_index(names)
                dfi.columns = list(['{}{}'.format(c, per) for c in varnames])
            dfs.append(dfi)
        df = pd.concat(dfs, axis=1)
        if squeeze:
            keep = []
            for var in varnames:
                diffcols = list([n for n in df.columns if var in n])
                diff = df[diffcols].diff(axis=1)
                diff['{}0'.format(var)] = 1  # always return the first stress period
                changed = diff.sum(axis=0) != 0
                keep.append(df.loc[:, changed.index[changed]])
            df = pd.concat(keep, axis=1)
        df = df.reset_index()
        df.insert(len(names), 'node', df.i * self.model.ncol + df.j)
        return df

    def add_record(self, kper, index, values):
        # Add a record to possible already set list for a given kper
        # index is a list of k,i,j or nodes.
        # values is a list of floats.
        # The length of index + values must be equal to the number of names
        # in dtype
        assert len(index) + len(values) == len(self.dtype), \
            "MfList.add_record() error: length of index arg +" + \
            "length of value arg != length of self dtype"
        # If we already have something for this kper, then add to it
        if (kper in list(self.__data.keys())):
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
            raise Exception("MfList.add_record() error: adding record to " + \
                            "recarray: " + str(e))

    def __getitem__(self, kper):
        # Get the recarray for a given kper
        # If the data entry for kper is a string, 
        # return the corresponding recarray,
        # but don't reset the value in the data dict
        # assert kper in list(self.data.keys()), "MfList.__getitem__() kper " + \
        #                                       str(kper) + " not in data.keys()"
        try:
            kper = int(kper)
        except Exception as e:
            raise Exception(
                "MfList error: _getitem__() passed invalid kper index:"
                + str(kper))
        if kper not in list(self.data.keys()):
            if kper == 0:
                return self.get_empty()
            else:
                return self.data[self.__find_last_kper(kper)]
        if (self.vtype[kper] == int):
            if (self.data[kper] == 0):
                return self.get_empty()
            else:
                return self.data[self.__find_last_kper(kper)]
        if (self.vtype[kper] == str):
            return self.__fromfile(self.data[kper])
        if (self.vtype[kper] == np.recarray):
            return self.data[kper]

    def __setitem__(self, kper, data):
        if (kper in list(self.__data.keys())):
            if self.model.verbose:
                print('removing existing data for kper={}'.format(kper))
            self.data.pop(kper)
        # If data is a list, then all we can do is try to cast it to
        # an ndarray, then cast again to a recarray
        if isinstance(data, list):
            # warnings.warn("MfList casting list to array")
            try:
                data = np.array(data)
            except Exception as e:
                raise Exception("MfList error: casting list to ndarray: " + \
                                str(e))
        # cast data
        if isinstance(data, int):
            self.__cast_int(kper, data)
        elif isinstance(data, np.recarray):
            self.__cast_recarray(kper, data)
        # A single ndarray
        elif isinstance(data, np.ndarray):
            self.__cast_ndarray(kper, data)
        # A single filename
        elif isinstance(data, str):
            self.__cast_str(kper, data)
        else:
            raise Exception("MfList error: unsupported data type: " + \
                            str(type(data)))

            # raise NotImplementedError("MfList.__setitem__() not implemented")

    def __fromfile(self, f):
        # d = np.fromfile(f,dtype=self.dtype,count=count)
        try:
            d = np.genfromtxt(f, dtype=self.dtype)
        except Exception as e:
            raise Exception("MfList.__fromfile() error reading recarray " + \
                            "from file " + str(e))
        return d

    def get_filenames(self):
        kpers = list(self.data.keys())
        kpers.sort()
        filenames = []
        first = kpers[0]
        for kper in list(range(0, max(self.model.nper, max(kpers) + 1))):
            # Fill missing early kpers with 0
            if (kper < first):
                itmp = 0
                kper_vtype = int
            elif (kper in kpers):
                kper_vtype = self.__vtype[kper]

            if self.model.array_free_format and self.model.external_path is not None:

                # py_filepath = ''
                # py_filepath = os.path.join(py_filepath,
                #                            self.model.external_path)
                filename = self.package.name[0] + \
                            "_{0:04d}.dat".format(kper)
                # py_filepath = os.path.join(py_filepath, filename)
                # filenames.append(py_filepath)
                filenames.append(filename)
        return filenames

    def get_filename(self,kper):
        ext = "dat"
        if self.binary:
            ext = 'bin'
        return self.package.name[0] + '_{0:04d}.{1}'.format(kper,ext)

    @property
    def binary(self):
        return bool(self.__binary)

    def write_transient(self, f, single_per=None, forceInternal=False):
        # forceInteral overrides isExternal (set below) for cases where
        # external arrays are not supported (oh hello MNW1!)
        # write the transient sequence described by the data dict
        nr, nc, nl, nper = self.model.get_nrow_ncol_nlay_nper()
        assert hasattr(f, "read"), "MfList.write() error: " + \
                                   "f argument must be a file handle"
        kpers = list(self.data.keys())
        kpers.sort()
        first = kpers[0]
        if (single_per == None):
            loop_over_kpers = list(range(0, max(nper, max(kpers) + 1)))
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
                    if (not self.model.array_free_format):
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

            f.write(" {0:9d} {1:9d} # stress period {2:d}\n"
                    .format(itmp, 0, kper+1))

            isExternal = False
            if self.model.array_free_format and \
                            self.model.external_path is not None and \
                            forceInternal is False:
                isExternal = True
            if self.__binary:
                isExternal = True
            if isExternal:
                if kper_vtype == np.recarray:
                    py_filepath = ''
                    if self.model.model_ws is not None:
                        py_filepath = self.model.model_ws
                    if self.model.external_path is not None:
                        py_filepath = os.path.join(py_filepath,
                                                   self.model.external_path)
                    filename = self.get_filename(kper)
                    py_filepath = os.path.join(py_filepath, filename)
                    model_filepath = filename
                    if self.model.external_path is not None:
                        model_filepath = os.path.join(self.model.external_path,
                                                      filename)
                    self.__tofile(py_filepath, kper_data)
                    kper_vtype = str
                    kper_data = model_filepath

            if kper_vtype == np.recarray:
                name = f.name
                f.close()
                f = open(name, 'ab+')
                # print(f)
                self.__tofile(f, kper_data)
                f.close()
                f = open(name, 'a')
                # print(f)
            elif kper_vtype == str:
                f.write('         open/close ' + kper_data)
                if self.__binary:
                    f.write(' (BINARY)')
                f.write('\n')

    def __tofile(self, f, data):
        # Write the recarray (data) to the file (or file handle) f
        assert isinstance(data, np.recarray), "MfList.__tofile() data arg " + \
                                              "not a recarray"

        # Add one to the kij indices
        lnames = [name.lower() for name in self.dtype.names]
        # --make copy of data for multiple calls
        d = np.recarray.copy(data)
        for idx in ['k', 'i', 'j', 'node']:
            if idx in lnames:
                d[idx] += 1
        if self.__binary:
            dtype2 = []
            for name in self.dtype.names:
                dtype2.append((name, np.float32))
            dtype2 = np.dtype(dtype2)
            d = np.array(d, dtype=dtype2)
            d.tofile(f)
        else:
            np.savetxt(f, d, fmt=self.fmt_string, delimiter='')

    def check_kij(self):
        names = self.dtype.names
        if ('k' not in names) or ('i' not in names) or ('j' not in names):
            warnings.warn("MfList.check_kij(): index fieldnames \'k,i,j\' " +
                          "not found in self.dtype names: " + str(names))
            return
        nr, nc, nl, nper = self.model.get_nrow_ncol_nlay_nper()
        if (nl == 0):
            warnings.warn("MfList.check_kij(): unable to get dis info from " +
                          "model")
            return
        for kper in list(self.data.keys()):
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
                    warn_str = "MfList.check_kij(): warning the following " + \
                               "indices are out of bounds in kper " + \
                               str(kper) + ':\n'
                    for idx in out_idx:
                        d = data[idx]
                        warn_str += " {0:9d} {1:9d} {2:9d}\n".format(d['k']
                                                                     + 1, d[
                                                                         'i'] + 1,
                                                                     d[
                                                                         'j'] + 1)
                    warnings.warn(warn_str)

    def __find_last_kper(self, kper):
        kpers = list(self.data.keys())
        kpers.sort()
        last = 0
        for kkper in kpers[::-1]:
            # if this entry is valid
            if self.vtype[kkper] != int or self.data[kkper] != -1:
                last = kkper
                if kkper <= kper:
                    break
        return kkper

    def get_indices(self):
        """
            a helper function for plotting - get all unique indices
        """
        names = self.dtype.names
        lnames = []
        [lnames.append(name.lower()) for name in names]
        if 'k' not in lnames or 'j' not in lnames:
            raise NotImplementedError("MfList.get_indices requires kij")
        kpers = list(self.data.keys())
        kpers.sort()
        indices = None
        for i, kper in enumerate(kpers):
            kper_vtype = self.__vtype[kper]
            if (kper_vtype != int) or (kper_vtype is not None):
                d = self.data[kper]
                if indices is None:
                    indices = list(zip(d['k'], d['i'], d['j']))
                else:
                    new_indices = list(zip(d['k'], d['i'], d['j']))
                    for ni in new_indices:
                        if ni not in indices:
                            indices.append(ni)
        return indices

    def attribute_by_kper(self, attr, function=np.mean, idx_val=None):
        assert attr in self.dtype.names
        if idx_val is not None:
            assert idx_val[0] in self.dtype.names
        kpers = list(self.data.keys())
        kpers.sort()
        values = []
        for kper in range(0, max(self.model.nper, max(kpers))):

            if kper < min(kpers):
                values.append(0)
            elif kper > max(kpers) or kper not in kpers:
                values.append(values[-1])
            else:
                kper_data = self.__data[kper]
                if idx_val is not None:
                    kper_data = kper_data[
                        np.where(kper_data[idx_val[0]] == idx_val[1])]
                # kper_vtype = self.__vtype[kper]
                v = function(kper_data[attr])
                values.append(v)
        return values

    def plot(self, key=None, names=None, kper=0,
             filename_base=None, file_extension=None, mflay=None,
             **kwargs):
        """
        Plot stress period boundary condition (MfList) data for a specified
        stress period

        Parameters
        ----------
        key : str
            MfList dictionary key. (default is None)
        names : list
            List of names for figure titles. (default is None)
        kper : int
            MODFLOW zero-based stress period number to return. (default is zero)
        filename_base : str
            Base file name that will be used to automatically generate file
            names for output image files. Plots will be exported as image
            files if file_name_base is not None. (default is None)
        file_extension : str
            Valid matplotlib.pyplot file extension for savefig(). Only used
            if filename_base is not None. (default is 'png')
        mflay : int
            MODFLOW zero-based layer number to return.  If None, then all
            all layers will be included. (default is None)
        **kwargs : dict
            axes : list of matplotlib.pyplot.axis
                List of matplotlib.pyplot.axis that will be used to plot
                data for each layer. If axes=None axes will be generated.
                (default is None)
            pcolor : bool
                Boolean used to determine if matplotlib.pyplot.pcolormesh
                plot will be plotted. (default is True)
            colorbar : bool
                Boolean used to determine if a color bar will be added to
                the matplotlib.pyplot.pcolormesh. Only used if pcolor=True.
                (default is False)
            inactive : bool
                Boolean used to determine if a black overlay in inactive
                cells in a layer will be displayed. (default is True)
            contour : bool
                Boolean used to determine if matplotlib.pyplot.contour
                plot will be plotted. (default is False)
            clabel : bool
                Boolean used to determine if matplotlib.pyplot.clabel
                will be plotted. Only used if contour=True. (default is False)
            grid : bool
                Boolean used to determine if the model grid will be plotted
                on the figure. (default is False)
            masked_values : list
                List of unique values to be excluded from the plot.

        Returns
        ----------
        out : list
            Empty list is returned if filename_base is not None. Otherwise
            a list of matplotlib.pyplot.axis is returned.

        See Also
        --------

        Notes
        -----

        Examples
        --------
        >>> import flopy
        >>> ml = flopy.modflow.Modflow.load('test.nam')
        >>> ml.wel.stress_period_data.plot(ml.wel, kper=1)

        """

        import flopy.plot.plotutil as pu

        if file_extension is not None:
            fext = file_extension
        else:
            fext = 'png'

        filenames = None
        if filename_base is not None:
            if mflay is not None:
                i0 = int(mflay)
                if i0 + 1 >= self.model.nlay:
                    i0 = self.model.nlay - 1
                i1 = i0 + 1
            else:
                i0 = 0
                i1 = self.model.nlay
            # build filenames
            pn = self.package.name[0].upper()
            filenames = [
                '{}_{}_StressPeriod{}_Layer{}.{}'.format(filename_base, pn,
                                                         kper + 1, k + 1, fext)
                for k in range(i0, i1)]
        if names is None:
            if key is None:
                names = ['{} location stress period: {} layer: {}'.format(
                    self.package.name[0], kper + 1, k + 1)
                         for k in range(self.model.nlay)]
            else:
                names = ['{} {} stress period: {} layer: {}'.format(
                    self.package.name[0], key, kper + 1, k + 1)
                         for k in range(self.model.nlay)]

        if key is None:
            axes = pu._plot_bc_helper(self.package, kper,
                                      names=names, filenames=filenames,
                                      mflay=mflay, **kwargs)
        else:
            arr_dict = self.to_array(kper, mask=True)

            try:
                arr = arr_dict[key]
            except:
                p = 'Cannot find key to plot\n'
                p += '  Provided key={}\n  Available keys='.format(key)
                for name, arr in arr_dict.items():
                    p += '{}, '.format(name)
                p += '\n'
                raise Exception(p)

            axes = pu._plot_array_helper(arr, model=self.model,
                                         names=names, filenames=filenames,
                                         mflay=mflay, **kwargs)
        return axes

    def to_shapefile(self, filename, kper=None):
        """
        Export stress period boundary condition (MfList) data for a specified
        stress period

        Parameters
        ----------
        filename : str
            Shapefile name to write
        kper : int
            MODFLOW zero-based stress period number to return. (default is None)

        Returns
        ----------
        None

        See Also
        --------

        Notes
        -----

        Examples
        --------
        >>> import flopy
        >>> ml = flopy.modflow.Modflow.load('test.nam')
        >>> ml.wel.to_shapefile('test_hk.shp', kper=1)
        """
        import warnings
        warnings.warn(
            "Deprecation warning: to_shapefile() is deprecated. use .export()")

        # if self.sr is None:
        #     raise Exception("MfList.to_shapefile: SpatialReference not set")
        # import flopy.utils.flopy_io as fio
        # if kper is None:
        #     keys = self.data.keys()
        #     keys.sort()
        # else:
        #     keys = [kper]
        # array_dict = {}
        # for kk in keys:
        #     arrays = self.to_array(kk)
        #     for name, array in arrays.items():
        #         for k in range(array.shape[0]):
        #             #aname = name+"{0:03d}_{1:02d}".format(kk, k)
        #             n = fio.shape_attr_name(name, length=4)
        #             aname = "{}{:03d}{:03d}".format(n, k+1, int(kk)+1)
        #             array_dict[aname] = array[k]
        # fio.write_grid_shapefile(filename, self.sr, array_dict)
        self.export(filename, kper=kper)

    def to_array(self, kper=0, mask=False):
        """
        Convert stress period boundary condition (MfList) data for a
        specified stress period to a 3-D numpy array

        Parameters
        ----------
        kper : int
            MODFLOW zero-based stress period number to return. (default is zero)
        mask : boolean
            return array with np.NaN instead of zero
        Returns
        ----------
        out : dict of numpy.ndarrays
            Dictionary of 3-D numpy arrays containing the stress period data for
            a selected stress period. The dictonary keys are the MfList dtype
            names for the stress period data ('cond', 'flux', 'bhead', etc.).

        See Also
        --------

        Notes
        -----

        Examples
        --------
        >>> import flopy
        >>> ml = flopy.modflow.Modflow.load('test.nam')
        >>> v = ml.wel.stress_period_data.to_array(kper=1)

        """
        i0 = 3
        if 'inode' in self.dtype.names:
            raise NotImplementedError()
        arrays = {}
        for name in self.dtype.names[i0:]:
            if not self.dtype.fields[name][0] == object:
                arr = np.zeros((self.model.nlay, self.model.nrow, self.model.ncol))
                arrays[name] = arr.copy()

        # if this kper is not found
        if kper not in self.data.keys():
            kpers = list(self.data.keys())
            kpers.sort()
            # if this kper is before the first entry,
            # (maybe) mask and return
            if kper < kpers[0]:
                if mask:
                    for name, arr in arrays.items():
                        arrays[name][:] = np.NaN
                return arrays
            # find the last kper
            else:
                kper = self.__find_last_kper(kper)

        sarr = self.data[kper]

        if np.isscalar(sarr):
            # if there are no entries for this kper
            if sarr == 0:
                if mask:
                    for name, arr in arrays.items():
                        arrays[name][:] = np.NaN
                return arrays
            else:
                raise Exception("MfList: something bad happened")

        for name, arr in arrays.items():
            cnt = np.zeros((self.model.nlay, self.model.nrow, self.model.ncol),
                           dtype=np.float)
            #print(name,kper)
            for rec in sarr:
                arr[rec['k'], rec['i'], rec['j']] += rec[name]
                cnt[rec['k'], rec['i'], rec['j']] += 1.
            # average keys that should not be added
            if name != 'cond' and name != 'flux':
                idx = cnt > 0.
                arr[idx] /= cnt[idx]
            if mask:
                arr = np.ma.masked_where(cnt == 0., arr)
                arr[cnt == 0.] = np.NaN

            arrays[name] = arr.copy()
        # elif mask:
        #     for name, arr in arrays.items():
        #         arrays[name][:] = np.NaN
        return arrays

    @property
    def masked_4D_arrays(self):
        # get the first kper
        arrays = self.to_array(kper=0, mask=True)

        # initialize these big arrays
        m4ds = {}
        for name, array in arrays.items():
            m4d = np.zeros((self.model.nper, self.model.nlay,
                            self.model.nrow, self.model.ncol))
            m4d[0, :, :, :] = array
            m4ds[name] = m4d
        for kper in range(1, self.model.nper):
            arrays = self.to_array(kper=kper, mask=True)
            for name, array in arrays.items():
                m4ds[name][kper, :, :, :] = array
        return m4ds

    def masked_4D_arrays_itr(self):
        # get the first kper
        arrays = self.to_array(kper=0, mask=True)

        # initialize these big arrays
        for name, array in arrays.items():
            m4d = np.zeros((self.model.nper, self.model.nlay,
                            self.model.nrow, self.model.ncol))
            m4d[0, :, :, :] = array
            for kper in range(1, self.model.nper):
                arrays = self.to_array(kper=kper, mask=True)
                for tname, array in arrays.items():
                    if tname == name:
                        m4d[kper, :, :, :] = array
            yield name, m4d

    @property
    def array(self):
        return self.masked_4D_arrays

    @classmethod
    def from_4d(cls, model, pak_name, m4ds):
        """construct an MfList instance from a dict of
        (attribute_name,masked 4D ndarray
        Parameters
        ----------
            model : mbase dervied type
            pak_name : str package name (e.g GHB)
            m4ds : {attibute name:4d masked numpy.ndarray}
        Returns
        -------
            MfList instance
        """
        sp_data = MfList.masked4D_arrays_to_stress_period_data(
            model.get_package(pak_name).get_default_dtype(), m4ds)
        return cls(model.get_package(pak_name), data=sp_data)

    @staticmethod
    def masked4D_arrays_to_stress_period_data(dtype, m4ds):
        """ convert a dictionary of 4-dim masked arrays to
            a stress_period_data style dict of recarray
        Parameters
        ----------
            dtype : numpy dtype

            m4ds : dict {name:masked numpy 4-dim ndarray}
        Returns
        -------
            dict {kper:recarray}
        """
        assert isinstance(m4ds, dict)
        for name, m4d in m4ds.items():
            assert isinstance(m4d, np.ndarray)
            assert name in dtype.names
            assert m4d.ndim == 4
        keys = list(m4ds.keys())

        for i1, key1 in enumerate(keys):
            a1 = np.isnan(m4ds[key1])
            for i2, key2 in enumerate(keys[i1:]):
                a2 = np.isnan(m4ds[key2])
                if not np.array_equal(a1, a2):
                    raise Exception("Transient2d error: masking not equal" + \
                                    " for {0} and {1}".format(key1, key2))

        sp_data = {}
        for kper in range(m4d.shape[0]):
            vals = {}
            for name, m4d in m4ds.items():
                arr = m4d[kper, :, :, :]
                isnan = np.argwhere(~np.isnan(arr))
                v = []
                for k, i, j in isnan:
                    v.append(arr[k, i, j])
                vals[name] = v
                kk = isnan[:, 0]
                ii = isnan[:, 1]
                jj = isnan[:, 2]

            spd = np.recarray(shape=isnan.shape[0], dtype=dtype)
            spd["i"] = ii
            spd["k"] = kk
            spd["j"] = jj
            for n, v in vals.items():
                spd[n] = v
            sp_data[kper] = spd
        return sp_data
