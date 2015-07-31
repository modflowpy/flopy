"""
util_list module.  Contains the mflist class.
 This classes encapsulates modflow-style list inputs away
 from the individual packages.  The end-user should not need to
 instantiate this class directly.

"""
from __future__ import division, print_function

import os
import warnings
import numpy as np
from flopy.utils import reference


class mflist(object):
    """
    a generic object for handling transient boundary condition lists

    Parameters
    ----------
    package : package object
        The package object (of type :class:`flopy.mbase.Package`) to which
        this mflist will be added.
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

    def __init__(self, package, data=None, model=None):
        self.package = package
        if model is None:
            self.model = package.parent
        else:
            self.model = model
        try:
            self.sr = self.model.dis.sr
        except:
            self.sr = None
        assert isinstance(self.package.dtype, np.dtype)
        self.__dtype = self.package.dtype
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
        if (kper not in list(self.__data.keys())):
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
        for kper in list(self.__data.keys()):
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
                raise Exception("mflist error: '\str\' type found it dtype." + \
                                " This gives unpredictable results when " + \
                                "recarray to file - change to \'object\' type")
            else:
                raise Exception("mflist.fmt_string error: unknown vtype " + \
                                "in dtype:" + vtype)
        return fmt_string


    # Private method to cast the data argument
    # Should only be called by the constructor
    def __cast_data(self, data):
        # If data is a list, then all we can do is try to cast it to
        # an ndarray, then cast again to a recarray
        if isinstance(data, list):
            # warnings.warn("mflist casting list to array")
            try:
                data = np.array(data)
            except Exception as e:
                raise Exception("mflist error: casting list to ndarray: " + \
                                str(e))

        # If data is a dict, the we have to assume it is keyed on kper
        if isinstance(data, dict):
            if len(list(data.keys())) == 0:
                raise Exception("mflist error: data dict is empty")
            for kper, d in data.items():
                assert isinstance(kper, int), "mflist error: data dict key " + \
                                              " \'{0:s}\' " + \
                                              "not integer: ".format(kper) + \
                                              str(type(kper))
                # Same as before, just try...
                if isinstance(d, list):
                    # warnings.warn("mflist: casting list to array at " +\
                    #               "kper {0:d}".format(kper))
                    try:
                        d = np.array(d)
                    except Exception as e:
                        raise Exception("mflist error: casting list " + \
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
            raise Exception("mflist error: unsupported data type: " + \
                            str(type(data)))

    def __cast_str(self, kper, d):
        # If d is a string, assume it is a filename and check that it exists
        assert os.path.exists(d), "mflist error: dict filename (string) \'" + \
                                  d + "\' value for " + \
                                  "kper {0:d} not found".format(kper)
        self.__data[kper] = d
        self.__vtype[kper] = str

    def __cast_int(self, kper, d):
        # If d is an integer, then it must be 0 or -1
        if (d > 0):
            raise Exception("mflist error: dict integer value for " + \
                            "kper {0:10d} must be 0 or -1, " + \
                            "not {1:10d}".format(kper, d))
        if (d == 0):
            self.__data[kper] = 0
            self.__vtype[kper] = None
        else:
            if (kper == 0):
                raise Exception("mflist error: dict integer value for " + \
                                "kper 0 for cannot be negative")
            self.__data[kper] = -1
            self.__vtype[kper] = None

    def __cast_recarray(self, kper, d):
        assert d.dtype == self.__dtype, "mflist error: recarray dtype: " + \
                                        str(d.dtype) + " doesn't match " + \
                                        "self dtype: " + str(self.dtype)
        self.__data[kper] = d
        self.__vtype[kper] = np.recarray

    def __cast_ndarray(self, kper, d):
        d = np.atleast_2d(d)
        if (d.dtype != self.__dtype):
            assert d.shape[1] == len(self.dtype), "mflist error: ndarray " + \
                                                  "shape " + str(d.shape) + \
                                                  " doesn't match dtype " + \
                                                  "len: " + \
                                                  str(len(self.dtype))
            # warnings.warn("mflist: ndarray dtype does not match self " +\
            #               "dtype, trying to cast")
        try:
            self.__data[kper] = np.core.records.fromarrays(d.transpose(),
                                                           dtype=self.dtype)
        except Exception as e:
            raise Exception("mflist error: casting ndarray to recarray: " + \
                            str(e))
        self.__vtype[kper] = np.recarray

    def add_record(self, kper, index, values):
        # Add a record to possible already set list for a given kper
        # index is a list of k,i,j or nodes.
        # values is a list of floats.
        # The length of index + values must be equal to the number of names
        # in dtype
        assert len(index) + len(values) == len(self.dtype), \
            "mflist.add_record() error: length of index arg +" + \
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
            raise Exception("mflist.add_record() error: adding record to " + \
                            "recarray: " + str(e))

    def __getitem__(self, kper):
        # Get the recarray for a given kper
        # If the data entry for kper is a string, 
        # return the corresponding recarray,
        # but don't reset the value in the data dict
        assert kper in list(self.data.keys()), "mflist.__getitem__() kper " + \
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

    def __setitem__(self, kper, data):
        if (kper in list(self.__data.keys())):
            if self.model.verbose:
                print('removing existing data for kper={}'.format(kper))
            self.data.pop(kper)
        # If data is a list, then all we can do is try to cast it to
        # an ndarray, then cast again to a recarray
        if isinstance(data, list):
            # warnings.warn("mflist casting list to array")
            try:
                data = np.array(data)
            except Exception as e:
                raise Exception("mflist error: casting list to ndarray: " + \
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
            raise Exception("mflist error: unsupported data type: " + \
                            str(type(data)))

            # raise NotImplementedError("mflist.__setitem__() not implemented")

    def __fromfile(self, f):
        # d = np.fromfile(f,dtype=self.dtype,count=count)
        try:
            d = np.genfromtxt(f, dtype=self.dtype)
        except Exception as e:
            raise Exception("mflist.__fromfile() error reading recarray " + \
                            "from file " + str(e))
        return d

    def write_transient(self, f, single_per=None):
        # write the transient sequence described by the data dict
        nr, nc, nl, nper = self.model.get_nrow_ncol_nlay_nper()
        # assert isinstance(f, file), "mflist.write() error: " +\
        #                             "f argument must be a file handle"
        assert hasattr(f, "read"), "mflist.write() error: " + \
                                   "f argument must be a file handle"
        kpers = list(self.data.keys())
        kpers.sort()
        # Assert 0 in kpers,"mflist.write() error: kper 0 not defined"
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

            f.write(" {0:9d} {1:9d} # stress period {2:d}\n"
                    .format(itmp,0, kper))

            if (kper_vtype == np.recarray):
                name = f.name
                f.close()
                f = open(name, 'ab+')
                #print(f)
                self.__tofile(f, kper_data)
                f.close()
                f = open(name, 'a')
                #print(f)
            elif (kper_vtype == str):
                f.write("         open/close " + kper_data + '\n')

    def __tofile(self, f, data):
        # Write the recarray (data) to the file (or file handle) f
        assert isinstance(data, np.recarray), "mflist.__tofile() data arg " + \
                                              "not a recarray"

        # Add one to the kij indices
        names = self.dtype.names
        lnames = []
        [lnames.append(name.lower()) for name in names]
        # --make copy of data for multiple calls
        d = np.recarray.copy(data)
        for idx in ['k', 'i', 'j', 'node']:
            if (idx in lnames):
                d[idx] += 1
        np.savetxt(f, d, fmt=self.fmt_string, delimiter='')

    def check_kij(self):
        names = self.dtype.names
        if ('k' not in names) or ('i' not in names) or ('j' not in names):
            warnings.warn("mflist.check_kij(): index fieldnames \'k,i,j\' " +
                          "not found in self.dtype names: " + str(names))
            return
        nr, nc, nl, nper = self.model.get_nrow_ncol_nlay_nper()
        if (nl == 0):
            warnings.warn("mflist.check_kij(): unable to get dis info from " +
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
                    warn_str = "mflist.check_kij(): warning the following " + \
                               "indices are out of bounds in kper " + \
                               str(kper) + ':\n'
                    for idx in out_idx:
                        d = data[idx]
                        warn_str += " {0:9d} {1:9d} {2:9d}\n".format(d['k']
                                                                     + 1, d['i'] + 1, d['j'] + 1)
                    warnings.warn(warn_str)

    def __find_last_kper(self, kper):
        kpers = list(self.data.keys())
        kpers.sort()
        last = kpers[0]
        for kper in kpers:
            if (kper >= last):
                break
            if (self.vtype[kper] != int) or (self.data[kper] != -1):
                last = kper
        return kper

    def get_indices(self):
        """
            a helper function for plotting - get all unique indices
        """
        names = self.dtype.names
        lnames = []
        [lnames.append(name.lower()) for name in names]
        if 'k' not in lnames or 'j' not in lnames:
            raise NotImplementedError("mflist.get_indices requires kij")
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
        Plot stress period boundary condition (mflist) data for a specified
        stress period

        Parameters
        ----------
        key : str
            mflist dictionary key. (default is None)
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
                if i0+1 >= self.model.nlay:
                    i0 = self.model.nlay - 1
                i1 = i0 + 1
            else:
                i0 = 0
                i1 = self.model.nlay
            # build filenames
            pn = self.package.name[0].upper()
            filenames = ['{}_{}_StressPeriod{}_Layer{}.{}'.format(filename_base, pn,
                                                                  kper+1, k+1, fext) for k in range(i0, i1)]
        if names is None:
            if key is None:
                names = ['{} location stress period: {} layer: {}'.format(self.package.name[0], kper+1, k+1)
                         for k in range(self.model.nlay)]
            else:
                names = ['{} {} stress period: {} layer: {}'.format(self.package.name[0], key, kper+1, k+1)
                         for k in range(self.model.nlay)]

        if key is None:
            axes = pu._plot_bc_helper(self.package, kper,
                                      names=names, filenames=filenames,
                                      mflay=mflay, **kwargs)
        else:
            arr_dict = self.to_array(kper)

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

    def to_shapefile(self, filename, kper=0):
        """
        Export stress period boundary condition (mflist) data for a specified
        stress period

        Parameters
        ----------
        filename : str
            Shapefile name to write
        kper : int
            MODFLOW zero-based stress period number to return. (default is zero)

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

        if self.sr is None:
            raise Exception("mflist.to_shapefile: SpatialReference not set")
        import flopy.utils.flopy_io as fio
        arrays = self.to_array(kper)
        array_dict = {}
        for name, array in arrays.items():
            for k in range(array.shape[0]):
                aname = name+"{0:03d}_{1:02d}".format(kper, k)
                array_dict[aname] = array[k]
        fio.write_grid_shapefile(filename, self.sr, array_dict)

    def to_array(self, kper=0):
        """
        Convert stress period boundary condition (mflist) data for a
        specified stress period to a 3-D numpy array

        Parameters
        ----------
        kper : int
            MODFLOW zero-based stress period number to return. (default is zero)

        Returns
        ----------
        out : dict of numpy.ndarrays
            Dictionary of 3-D numpy arrays containing the stress period data for
            a selected stress period. The dictonary keys are the mflist dtype
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
            arr = np.zeros((self.model.nlay, self.model.nrow, self.model.ncol))
            arrays[name] = arr.copy()
        if kper in self.data.keys():
            sarr = self.data[kper]
            for name, arr in arrays.items():
                cnt = np.zeros((self.model.nlay, self.model.nrow, self.model.ncol), dtype=np.float)
                for rec in sarr:
                    arr[rec['k'], rec['i'], rec['j']] += rec[name]
                    if name != 'cond' and name != 'flux':
                        cnt[rec['k'], rec['i'], rec['j']] += 1.
                # average keys that should not be added
                if name != 'cond' and name != 'flux':
                    idx = cnt > 0.
                    arr[idx] /= cnt[idx]
                arrays[name] = arr

        return arrays
