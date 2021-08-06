"""
util_array module.  Contains the util_2d, util_3d and transient_2d classes.
 These classes encapsulate modflow-style array inputs away
 from the individual packages.  The end-user should not need to
 instantiate these classes directly.

"""
from __future__ import division, print_function

# from future.utils import with_metaclass

import os
import shutil
import copy
import numpy as np
from warnings import warn
from ..utils.binaryfile import BinaryHeader
from ..utils.flopy_io import line_parse
from ..datbase import DataType, DataInterface


class ArrayFormat:
    """
    ArrayFormat class for handling various output format types for both
    MODFLOW and flopy

    Parameters
    ----------
    u2d : Util2d instance
    python : str (optional)
        python-style output format descriptor e.g. {0:15.6e}
    fortran : str (optional)
        fortran style output format descriptor e.g. (2E15.6)


    Attributes
    ----------
    fortran : str
        fortran format output descriptor (e.g. (100G15.6)
    py : str
        python format output descriptor (e.g. "{0:15.6E}")
    numpy : str
        numpy format output descriptor (e.g. "%15.6e")
    npl : int
        number if items per line of output
    width : int
        the width of the formatted numeric output
    decimal : int
        the number of decimal digits in the numeric output
    format : str
        the output format type e.g. I, G, E, etc
    free : bool
        free format flag
    binary : bool
        binary format flag


    Methods
    -------
    get_default_numpy_fmt : (dtype : [np.int32, np.float32])
        a static method to get a default numpy dtype - used for loading
    decode_fortran_descriptor : (fd : str)
        a static method to decode fortran descriptors into npl, format,
        width, decimal.

    See Also
    --------

    Notes
    -----

    Examples
    --------

    """

    def __init__(self, u2d, python=None, fortran=None, array_free_format=None):

        assert isinstance(
            u2d, Util2d
        ), "ArrayFormat only supports Util2d, not {0}".format(type(u2d))
        if len(u2d.shape) == 1:
            self._npl_full = u2d.shape[0]
        else:
            self._npl_full = u2d.shape[1]
        self.dtype = u2d.dtype
        self._npl = None
        self._format = None
        self._width = None
        self._decimal = None
        if array_free_format is not None:
            self._freeformat_model = bool(array_free_format)
        else:
            self._freeformat_model = bool(u2d.model.array_free_format)

        self.default_float_width = 15
        self.default_int_width = 10
        self.default_float_format = "E"
        self.default_int_format = "I"
        self.default_float_decimal = 6
        self.default_int_decimal = 0

        self._fmts = ["I", "G", "E", "F"]

        self._isbinary = False
        self._isfree = False

        if python is not None and fortran is not None:
            raise Exception(
                "only one of [python,fortran] can be passed "
                "to ArrayFormat constructor"
            )

        if python is not None:
            self._parse_python_format(python)

        elif fortran is not None:
            self._parse_fortran_format(fortran)

        else:
            self._set_defaults()

    @property
    def array_free_format(self):
        return bool(self._freeformat_model)

    def _set_defaults(self):
        if self.dtype == np.int32:
            self._npl = self._npl_full
            self._format = self.default_int_format
            self._width = self.default_int_width
            self._decimal = None

        elif self.dtype in [np.float32, bool]:
            self._npl = self._npl_full
            self._format = self.default_float_format
            self._width = self.default_float_width
            self._decimal = self.default_float_decimal
        else:
            raise Exception(
                "ArrayFormat._set_defaults() error: "
                "unsupported dtype: {0}".format(str(self.dtype))
            )

    def __str__(self):
        s = "ArrayFormat: npl:{0},format:{1},width:{2},decimal{3}".format(
            self.npl, self.format, self.width, self.decimal
        )
        s += ",isfree:{0},isbinary:{1}".format(self._isfree, self._isbinary)
        return s

    @staticmethod
    def get_default_numpy_fmt(dtype):
        if dtype == np.int32:
            return "%10d"
        elif dtype == np.float32:
            return "%15.6E"
        else:
            raise Exception(
                "ArrayFormat.get_default_numpy_fmt(): unrecognized "
                "dtype, must be np.int32 or np.float32"
            )

    @classmethod
    def integer(cls):
        raise NotImplementedError()

    @classmethod
    def float(cls):
        raise NotImplementedError()

    @property
    def binary(self):
        return bool(self._isbinary)

    @property
    def free(self):
        return bool(self._isfree)

    def __eq__(self, other):
        if isinstance(other, str):
            if other.lower() == "free":
                return self.free
            if other.lower() == "binary":
                return self.binary
        else:
            super().__eq__(other)

    @property
    def npl(self):
        return copy.copy(self._npl)

    @property
    def format(self):
        return copy.copy(self._format)

    @property
    def width(self):
        return copy.copy(self._width)

    @property
    def decimal(self):
        return copy.copy(self._decimal)

    def __setattr__(self, key, value):
        if key == "format":
            value = value.upper()
            assert value.upper() in self._fmts
            if value == "I":
                assert self.dtype == np.int32, self.dtype
                self._format = value
                self._decimal = None
            else:
                if value == "G":
                    print("'G' format being reset to 'E'")
                    value = "E"
                self._format = value
                if self.decimal is None:
                    self._decimal = self.default_float_decimal

        elif key == "width":
            width = int(value)
            if self.dtype == np.float32 and width < self.decimal:
                raise Exception("width cannot be less than decimal")
            elif self.dtype == np.float32 and width < self.default_float_width:
                print(
                    "ArrayFormat warning:setting width less "
                    "than default of {0}".format(self.default_float_width)
                )
                self._width = width
        elif key == "decimal":
            if self.dtype == np.int32:
                raise Exception("cannot set decimal for integer dtypes")
            elif self.dtype == np.float32:
                value = int(value)
                if value < self.default_float_decimal:
                    print(
                        "ArrayFormat warning: setting decimal less than "
                        "default of {0}".format(self.default_float_decimal)
                    )
                if value < self.decimal:
                    print(
                        "ArrayFormat warning: setting decimal "
                        "less than current value of "
                        "{0}".format(self.default_float_decimal)
                    )
                self._decimal = int(value)
            else:
                raise TypeError(self.dtype)

        elif key == "entries" or key == "entires_per_line" or key == "npl":
            value = int(value)
            assert value <= self._npl_full, "cannot set npl > shape"
            self._npl = value

        elif key.lower() == "binary":
            value = bool(value)
            if value and self.free:
                #    raise Exception("cannot switch from 'free' to 'binary' format")
                self._isfree = False
            self._isbinary = value
            self._set_defaults()

        elif key.lower() == "free":
            value = bool(value)
            if value and self.binary:
                #    raise Exception("cannot switch from 'binary' to 'free' format")
                self._isbinary = False
            self._isfree = bool(value)
            self._set_defaults()

        elif key.lower() == "fortran":
            self._parse_fortran_format(value)

        elif key.lower() == "python" or key.lower() == "py":
            self._parse_python_format(value)

        else:
            super().__setattr__(key, value)

    @property
    def py(self):
        return self._get_python_format()

    def _get_python_format(self):

        if self.format == "I":
            fmt = "d"
        else:
            fmt = self.format
        pd = "{0:" + str(self.width)
        if self.decimal is not None:
            pd += "." + str(self.decimal) + fmt + "}"
        else:
            pd += fmt + "}"

        if self.npl is None:
            if self._isfree:
                return (self._npl_full, pd)
            else:
                raise Exception(
                    "ArrayFormat._get_python_format() error: "
                    "format is not 'free' and npl is not set"
                )

        return (self.npl, pd)

    def _parse_python_format(self, arg):
        raise NotImplementedError()

    @property
    def fortran(self):
        return self._get_fortran_format()

    def _get_fortran_format(self):
        if self._isfree:
            return "(FREE)"
        if self._isbinary:
            return "(BINARY)"

        fd = "({0:d}{1:s}{2:d}".format(self.npl, self.format, self.width)
        if self.decimal is not None:
            fd += ".{0:d})".format(self.decimal)
        else:
            fd += ")"
        return fd

    def _parse_fortran_format(self, arg):
        """Decode fortran descriptor

        Parameters
        ----------
        arg : str

        Returns
        -------
        npl, fmt, width, decimal : int, str, int, int

        """
        # strip off any quotes around format string

        npl, fmt, width, decimal = ArrayFormat.decode_fortran_descriptor(arg)
        if isinstance(npl, str):
            if "FREE" in npl.upper():
                self._set_defaults()
                self._isfree = True
                return

            elif "BINARY" in npl.upper():
                self._set_defaults()
                self._isbinary = True
                return
        self._npl = int(npl)
        self._format = fmt
        self._width = int(width)
        if decimal is not None:
            self._decimal = int(decimal)

    @property
    def numpy(self):
        return self._get_numpy_format()

    def _get_numpy_format(self):
        return "%{0}{1}.{2}".format(self.width, self.format, self.decimal)

    @staticmethod
    def decode_fortran_descriptor(fd):
        """Decode fortran descriptor

        Parameters
        ----------
        fd : str

        Returns
        -------
        npl, fmt, width, decimal : int, str, int, int

        """
        # strip off any quotes around format string
        fd = fd.replace("'", "")
        fd = fd.replace('"', "")
        # strip off '(' and ')'
        fd = fd.strip()[1:-1]
        if str("FREE") in str(fd.upper()):
            return "free", None, None, None
        elif str("BINARY") in str(fd.upper()):
            return "binary", None, None, None
        if str(".") in str(fd):
            raw = fd.split(".")
            decimal = int(raw[1])
        else:
            raw = [fd]
            decimal = None
        fmts = ["ES", "EN", "I", "G", "E", "F"]
        raw = raw[0].upper()
        for fmt in fmts:
            if fmt in raw:
                raw = raw.split(fmt)
                # '(F9.0)' will return raw = ['', '9']
                #  try and except will catch this
                try:
                    npl = int(raw[0])
                    width = int(raw[1])
                except:
                    npl = 1
                    width = int(raw[1])
                if fmt == "G":
                    fmt = "E"
                elif fmt == "ES":
                    fmt = "E"
                elif fmt == "EN":
                    fmt = "E"
                return npl, fmt, width, decimal
        raise Exception(
            "Unrecognized format type: {} looking for: {}".format(fd, fmts)
        )


def read1d(f, a):
    """
    Fill the 1d array, a, with the correct number of values.  Required in
    case lpf 1d arrays (chani, layvka, etc) extend over more than one line

    """
    if len(a.shape) != 1:
        raise ValueError(
            "read1d: expected 1 dimension, found shape {0}".format(a.shape)
        )
    values = []
    while len(values) < a.shape[0]:
        line = f.readline()
        if len(line) == 0:
            raise ValueError("read1d: no data found")
        values += line_parse(line)
    a[:] = np.fromiter(values, dtype=a.dtype, count=a.shape[0])
    return a


def new_u2d(old_util2d, value):
    new_util2d = Util2d(
        old_util2d.model,
        old_util2d.shape,
        old_util2d.dtype,
        value,
        old_util2d.name,
        old_util2d.format.fortran,
        old_util2d.cnstnt,
        old_util2d.iprn,
        old_util2d.ext_filename,
        old_util2d.locat,
        old_util2d.format.binary,
        array_free_format=old_util2d.format.array_free_format,
    )
    return new_util2d


class Util3d(DataInterface):
    """
    Util3d class for handling 3-D model arrays.  just a thin wrapper around
        Util2d

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.mf.Modflow`) to which
        this package will be added.
    shape : length 3 tuple
        shape of the 3-D array, typically (nlay,nrow,ncol)
    dtype : [np.int32, np.float32, bool]
        the type of the data
    value : variable
        the data to be assigned to the 3-D array.
        can be a scalar, list, or ndarray
    name : string
        name of the property, used for writing comments to input files
    fmtin : string
        modflow fmtin variable (optional).  (the default is None)
    cnstnt : string
        modflow cnstnt variable (optional) (the default is 1.0)
    iprn : int
        modflow iprn variable (optional) (the default is -1)
    locat : int
        modflow locat variable (optional) (the default is None).  If the model
        instance does not support free format and the
        external flag is not set and the value is a simple scalar,
        then locat must be explicitly passed as it is the unit number
        to read the array from
    ext_filename : string
        the external filename to write the array representation to
        (optional) (the default is None) .
        If type(value) is a string and is an accessible filename, the
        ext_filename is reset to value.
    bin : bool
        flag to control writing external arrays as binary (optional)
        (the defaut is False)

    Attributes
    ----------
    array : np.ndarray
        the array representation of the 3-D object


    Methods
    -------
    get_file_entry : string
        get the model input file string including the control record for the
        entire 3-D property

    See Also
    --------

    Notes
    -----

    Examples
    --------

    """

    def __init__(
        self,
        model,
        shape,
        dtype,
        value,
        name,
        fmtin=None,
        cnstnt=1.0,
        iprn=-1,
        locat=None,
        ext_unit_dict=None,
        array_free_format=None,
    ):
        """
        3-D wrapper from Util2d - shape must be 3-D
        """
        self.array_free_format = array_free_format
        if isinstance(value, Util3d):
            for attr in value.__dict__.items():
                setattr(self, attr[0], attr[1])
            self._model = model
            self.array_free_format = array_free_format
            for i, u2d in enumerate(self.util_2ds):
                self.util_2ds[i] = Util2d(
                    model,
                    u2d.shape,
                    u2d.dtype,
                    u2d._array,
                    name=u2d.name,
                    fmtin=u2d.format.fortran,
                    locat=locat,
                    cnstnt=u2d.cnstnt,
                    ext_filename=u2d.filename,
                    array_free_format=array_free_format,
                )

            return
        if len(shape) != 3:
            raise ValueError(
                "Util3d: expected 3 dimensions, found shape {0}".format(shape)
            )
        self._model = model
        self.shape = shape
        self._dtype = dtype
        self.__value = value
        isnamespecified = False
        if isinstance(name, list):
            self._name = name
            isnamespecified = True
            isnamespecified = True
            isnamespecified = True
        else:
            t = []
            for k in range(shape[0]):
                t.append(name)
            self._name = t
        self.name_base = []
        for k in range(shape[0]):
            if isnamespecified:
                self.name_base.append(self.name[k])
            else:
                if "Layer" not in self.name[k]:
                    self.name_base.append(self.name[k] + " Layer ")
                else:
                    self.name_base.append(self.name[k])
        self.fmtin = fmtin
        self.cnstnt = cnstnt
        self.iprn = iprn
        self.locat = locat

        self.ext_filename_base = []
        if model.external_path is not None:
            for k in range(shape[0]):
                self.ext_filename_base.append(
                    os.path.join(
                        model.external_path,
                        self.name_base[k].replace(" ", "_"),
                    )
                )
        else:
            for k in range(shape[0]):
                self.ext_filename_base.append(
                    self.name_base[k].replace(" ", "_")
                )

        self.util_2ds = self.build_2d_instances()

    def __setitem__(self, k, value):
        if isinstance(k, int):
            assert k in range(
                0, self.shape[0]
            ), "Util3d error: k not in range nlay"
            self.util_2ds[k] = new_u2d(self.util_2ds[k], value)
        else:
            raise NotImplementedError(
                "Util3d doesn't support setitem indices" + str(k)
            )

    def __setattr__(self, key, value):
        if hasattr(self, "util_2ds") and key == "cnstnt":
            # set the cnstnt for each u2d
            for u2d in self.util_2ds:
                u2d.cnstnt = value
        elif hasattr(self, "util_2ds") and key == "fmtin":
            for u2d in self.util_2ds:
                u2d.format = ArrayFormat(
                    u2d,
                    fortran=value,
                    array_free_format=self.array_free_format,
                )
            super().__setattr__("fmtin", value)
        elif hasattr(self, "util_2ds") and key == "how":
            for u2d in self.util_2ds:
                u2d.how = value
        else:
            # set the attribute for u3d
            super().__setattr__(key, value)

    @property
    def name(self):
        return self._name

    @property
    def dtype(self):
        return self._dtype

    @property
    def model(self):
        return self._model

    @property
    def data_type(self):
        return DataType.array3d

    @property
    def plottable(self):
        return True

    def export(self, f, **kwargs):
        from flopy import export

        return export.utils.array3d_export(f, self, **kwargs)

    def to_shapefile(self, filename):
        """
        Export 3-D model data to shapefile (polygons).  Adds an
            attribute for each Util2d in self.u2ds

        Parameters
        ----------
        filename : str
            Shapefile name to write

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
        >>> ml.lpf.hk.to_shapefile('test_hk.shp')
        """
        warn(
            "Deprecation warning: to_shapefile() is deprecated. use .export()",
            DeprecationWarning,
        )
        self.export(filename)

    def plot(
        self,
        filename_base=None,
        file_extension=None,
        mflay=None,
        fignum=None,
        **kwargs
    ):
        """
        Plot 3-D model input data

        Parameters
        ----------
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
        >>> ml.lpf.hk.plot()

        """
        from flopy.plot import PlotUtilities

        axes = PlotUtilities._plot_util3d_helper(
            self,
            filename_base=filename_base,
            file_extension=file_extension,
            mflay=mflay,
            fignum=fignum,
            **kwargs
        )
        return axes

    def __getitem__(self, k):
        if isinstance(k, int) or np.issubdtype(
            getattr(k, "dtype", None), np.integer
        ):
            return self.util_2ds[k]
        elif len(k) == 3:
            return self.array[k[0], k[1], k[2]]
        else:
            raise Exception("Util3d error: unsupported indices:" + str(k))

    def get_file_entry(self):
        s = ""
        for u2d in self.util_2ds:
            s += u2d.get_file_entry()
        return s

    def get_value(self):
        value = []
        for u2d in self.util_2ds:
            value.append(u2d.get_value())
        return value

    @property
    def array(self):
        """
        Return a numpy array of the 3D shape.  If an unstructured model, then
        return an array of size nodes.

        """
        nlay, nrow, ncol = self.shape
        if nrow is not None:
            # typical 3D case
            a = np.empty((self.shape), dtype=self._dtype)
            # for i,u2d in self.uds:
            for i, u2d in enumerate(self.util_2ds):
                a[i] = u2d.array
        else:
            # unstructured case
            nodes = ncol.sum()
            a = np.empty((nodes), dtype=self._dtype)
            istart = 0
            for i, u2d in enumerate(self.util_2ds):
                istop = istart + ncol[i]
                a[istart:istop] = u2d.array
                istart = istop
        return a

    def build_2d_instances(self):
        u2ds = []
        # if value is not enumerable, then make a list of something
        if not isinstance(self.__value, list) and not isinstance(
            self.__value, np.ndarray
        ):
            self.__value = [self.__value] * self.shape[0]

        # if this is a list or 1-D array with constant values per layer
        if isinstance(self.__value, list) or (
            isinstance(self.__value, np.ndarray) and (self.__value.ndim == 1)
        ):

            assert (
                len(self.__value) == self.shape[0]
            ), "length of 3d enumerable: {} != to shape[0]: {}".format(
                len(self.__value), self.shape[0]
            )

            for i, item in enumerate(self.__value):
                if isinstance(item, Util2d):
                    # we need to reset the external name because most of the
                    # load() methods don't use layer-specific names
                    item._ext_filename = self.ext_filename_base[
                        i
                    ] + "{0}.ref".format(i + 1)
                    # reset the model instance in cases these Util2d's
                    # came from another model instance
                    item.model = self._model
                    u2ds.append(item)
                else:
                    name = self.name_base[i] + str(i + 1)
                    ext_filename = None
                    if self._model.external_path is not None:
                        ext_filename = (
                            self.ext_filename_base[i] + str(i + 1) + ".ref"
                        )
                    shape = self.shape[1:]
                    if shape[0] is None:
                        # allow for unstructured so that ncol changes by layer
                        shape = (self.shape[2][i],)
                    u2d = Util2d(
                        self.model,
                        shape,
                        self.dtype,
                        item,
                        fmtin=self.fmtin,
                        name=name,
                        ext_filename=ext_filename,
                        locat=self.locat,
                        array_free_format=self.array_free_format,
                    )
                    u2ds.append(u2d)

        elif isinstance(self.__value, np.ndarray):
            # if an array of shape nrow,ncol was passed, tile it out for each layer
            if self.__value.shape[0] != self.shape[0]:
                if self.__value.shape == (self.shape[1], self.shape[2]):
                    self.__value = [self.__value] * self.shape[0]
                else:
                    raise Exception(
                        "value shape[0] != to self.shape[0] and"
                        "value.shape[[1,2]] != self.shape[[1,2]] "
                        "{} {}".format(self.__value.shape, self.shape)
                    )
            for i, a in enumerate(self.__value):
                a = np.atleast_2d(a)
                ext_filename = None
                name = self.name_base[i] + str(i + 1)
                if self._model.external_path is not None:
                    ext_filename = (
                        self.ext_filename_base[i] + str(i + 1) + ".ref"
                    )
                u2d = Util2d(
                    self._model,
                    self.shape[1:],
                    self._dtype,
                    a,
                    fmtin=self.fmtin,
                    name=name,
                    ext_filename=ext_filename,
                    locat=self.locat,
                    array_free_format=self.array_free_format,
                )
                u2ds.append(u2d)

        else:
            raise Exception(
                "util_array_3d: value attribute must be list "
                "or ndarray, not {}".format(type(self.__value))
            )
        return u2ds

    @classmethod
    def load(
        cls,
        f_handle,
        model,
        shape,
        dtype,
        name,
        ext_unit_dict=None,
        array_format=None,
    ):
        if len(shape) != 3:
            raise ValueError(
                "Util3d: expected 3 dimensions, found shape {0}".format(shape)
            )
        nlay, nrow, ncol = shape
        u2ds = []
        for k in range(nlay):
            u2d_name = name + "_Layer_{0}".format(k)
            if nrow is None:
                nr = 1
                nc = ncol[k]
            else:
                nr = nrow
                nc = ncol
            u2d = Util2d.load(
                f_handle,
                model,
                (nr, nc),
                dtype,
                u2d_name,
                ext_unit_dict=ext_unit_dict,
                array_format=array_format,
            )
            u2ds.append(u2d)
        return cls(model, shape, dtype, u2ds, name)

    def __mul__(self, other):
        if np.isscalar(other):
            new_u2ds = []
            for u2d in self.util_2ds:
                new_u2ds.append(u2d * other)
            return Util3d(
                self._model,
                self.shape,
                self._dtype,
                new_u2ds,
                self._name,
                self.fmtin,
                self.cnstnt,
                self.iprn,
                self.locat,
            )
        elif isinstance(other, list):
            assert len(other) == self.shape[0]
            new_u2ds = []
            for u2d, item in zip(self.util_2ds, other):
                new_u2ds.append(u2d * item)
            return Util3d(
                self._model,
                self.shape,
                self._dtype,
                new_u2ds,
                self._name,
                self.fmtin,
                self.cnstnt,
                self.iprn,
                self.locat,
            )


class Transient3d(DataInterface):
    """
    Transient3d class for handling time-dependent 3-D model arrays.
    just a thin wrapper around Util3d

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.mf.Modflow`) to which
        this package will be added.
    shape : length 3 tuple
        shape of the 3-D transient arrays, typically (nlay,nrow,ncol)
    dtype : [np.int32, np.float32, bool]
        the type of the data
    value : variable
        the data to be assigned to the 3-D arrays. Typically a dict
        of {kper:value}, where kper is the zero-based stress period
        to assign a value to.  Value should be cast-able to Util2d instance
        can be a scalar, list, or ndarray is the array value is constant in
        time.
    name : string
        name of the property, used for writing comments to input files and
        for forming external files names (if needed)
    fmtin : string
        modflow fmtin variable (optional).  (the default is None)
    cnstnt : string
        modflow cnstnt variable (optional) (the default is 1.0)
    iprn : int
        modflow iprn variable (optional) (the default is -1)
    locat : int
        modflow locat variable (optional) (the default is None).  If the model
        instance does not support free format and the
        external flag is not set and the value is a simple scalar,
        then locat must be explicitly passed as it is the unit number
         to read the array from
    ext_filename : string
        the external filename to write the array representation to
        (optional) (the default is None) .
        If type(value) is a string and is an accessible filename,
        the ext_filename is reset to value.
    bin : bool
        flag to control writing external arrays as binary (optional)
        (the default is False)

    Attributes
    ----------
    transient_3ds : dict{kper:Util3d}
        the transient sequence of Util3d objects

    Methods
    -------
    get_kper_entry : (itmp,string)
        get the itmp value and the Util2d file entry of the value in
        transient_2ds in bin kper.  if kper < min(Transient2d.keys()),
        return (1,zero_entry<Util2d>).  If kper > < min(Transient2d.keys()),
        but is not found in Transient2d.keys(), return (-1,'')

    See Also
    --------

    Notes
    -----

    Examples
    --------

    """

    def __init__(
        self,
        model,
        shape,
        dtype,
        value,
        name,
        fmtin=None,
        cnstnt=1.0,
        iprn=-1,
        ext_filename=None,
        locat=None,
        bin=False,
        array_free_format=None,
    ):

        if isinstance(value, Transient3d):
            for attr in value.__dict__.items():
                setattr(self, attr[0], attr[1])
            self._model = model
            return

        self._model = model
        if len(shape) != 3:
            raise ValueError(
                "Transient3d: expected 3 dimensions (nlay, nrow, ncol), found "
                "shape {0}".format(shape)
            )
        self.shape = shape
        self._dtype = dtype
        self.__value = value
        self.name_base = name
        self.fmtin = fmtin
        self.cnstnt = cnstnt
        self.iprn = iprn
        self.locat = locat
        self.array_free_format = array_free_format
        self.transient_3ds = self.build_transient_sequence()
        return

    def __setattr__(self, key, value):
        # set the attribute for u3d, even for cnstnt
        super().__setattr__(key, value)

    @property
    def model(self):
        return self._model

    @property
    def name(self):
        return self.name_base

    @property
    def dtype(self):
        return self._dtype

    @property
    def data_type(self):
        return DataType.transient3d

    @property
    def plottable(self):
        return False

    def get_zero_3d(self, kper):
        name = self.name_base + str(kper + 1) + "(filled zero)"
        return Util3d(
            self._model,
            self.shape,
            self._dtype,
            0.0,
            name=name,
            array_free_format=self.array_free_format,
        )

    def __getitem__(self, kper):
        if kper in list(self.transient_3ds.keys()):
            return self.transient_3ds[kper]
        elif kper < min(self.transient_3ds.keys()):
            return self.get_zero_3d(kper)
        else:
            for i in range(kper, -1, -1):
                if i in list(self.transient_3ds.keys()):
                    return self.transient_3ds[i]
            raise Exception(
                "Transient2d.__getitem__(): error: "
                "could not find an entry before kper {0:d}".format(kper)
            )

    def __setitem__(self, key, value):
        try:
            key = int(key)
        except Exception as e:
            raise Exception(
                "Transient3d.__setitem__() error: "
                "'key'could not be cast to int:{0}".format(str(e))
            )
        nper = self._model.nper
        if key > self._model.nper or key < 0:
            raise Exception(
                "Transient3d.__setitem__() error: "
                "key {0} not in nper range {1}:{2}".format(key, 0, nper)
            )

        self.transient_3ds[key] = self.__get_3d_instance(key, value)

    @property
    def array(self):
        arr = np.zeros(
            (self._model.nper, self.shape[0], self.shape[1], self.shape[2]),
            dtype=self._dtype,
        )
        for kper in range(self._model.nper):
            u3d = self[kper]
            for k in range(self.shape[0]):
                arr[kper, k, :, :] = u3d[k].array
        return arr

    def get_kper_entry(self, kper):
        """
        get the file entry info for a given kper
        returns (itmp,file entry string from Util3d)
        """
        if kper in self.transient_3ds:
            s = ""
            for k in range(self.shape[0]):
                s += self.transient_3ds[kper][k].get_file_entry()
            return 1, s
        elif kper < min(self.transient_3ds.keys()):
            t = self.get_zero_3d(kper).get_file_entry()
            s = ""
            for k in range(self.shape[0]):
                s += t[k].get_file_entry()
            return 1, s
        else:
            return -1, ""

    def build_transient_sequence(self):
        """
        parse self.__value into a dict{kper:Util3d}
        """

        # a dict keyed on kper (zero-based)
        if isinstance(self.__value, dict):
            tran_seq = {}
            for key, val in self.__value.items():
                try:
                    key = int(key)
                except:
                    raise Exception(
                        "Transient3d error: can't cast key: "
                        "{} to kper integer".format(key)
                    )
                if key < 0:
                    raise Exception(
                        "Transient3d error: key can't be negative: "
                        "{}".format(key)
                    )
                try:
                    u3d = self.__get_3d_instance(key, val)
                except Exception as e:
                    raise Exception(
                        "Transient3d error building Util3d instance from "
                        "value at kper: {}\n{}".format(key, e)
                    )
                tran_seq[key] = u3d
            return tran_seq

        # these are all for single entries - use the same Util2d for all kper
        # an array of shape (nrow,ncol)
        elif isinstance(self.__value, np.ndarray):
            return {0: self.__get_3d_instance(0, self.__value)}

        # a filename
        elif isinstance(self.__value, str):
            return {0: self.__get_3d_instance(0, self.__value)}

        # a scalar
        elif np.isscalar(self.__value):
            return {0: self.__get_3d_instance(0, self.__value)}

        # lists aren't allowed
        elif isinstance(self.__value, list):
            raise Exception(
                "Transient3d error: value cannot be a list "
                "anymore.  try a dict{kper,value}"
            )
        else:
            raise Exception(
                "Transient3d error: value type not recognized: "
                "{}".format(type(self.__value))
            )

    def __get_3d_instance(self, kper, arg):
        """
        parse an argument into a Util3d instance
        """
        name = "{}_period{}".format(self.name_base, kper + 1)
        u3d = Util3d(
            self._model,
            self.shape,
            self._dtype,
            arg,
            fmtin=self.fmtin,
            name=name,
            #                     ext_filename=ext_filename,
            locat=self.locat,
            array_free_format=self.array_free_format,
        )
        return u3d


class Transient2d(DataInterface):
    """
    Transient2d class for handling time-dependent 2-D model arrays.
    just a thin wrapper around Util2d

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.mf.Modflow`) to which
        this package will be added.
    shape : length 2 tuple
        shape of the 2-D transient arrays, typically (nrow,ncol)
    dtype : [np.int32, np.float32, bool]
        the type of the data
    value : variable
        the data to be assigned to the 2-D arrays. Typically a dict
        of {kper:value}, where kper is the zero-based stress period
        to assign a value to.  Value should be cast-able to Util2d instance
        can be a scalar, list, or ndarray is the array value is constant in
        time.
    name : string
        name of the property, used for writing comments to input files and
        for forming external files names (if needed)
    fmtin : string
        modflow fmtin variable (optional).  (the default is None)
    cnstnt : string
        modflow cnstnt variable (optional) (the default is 1.0)
    iprn : int
        modflow iprn variable (optional) (the default is -1)
    locat : int
        modflow locat variable (optional) (the default is None).  If the model
        instance does not support free format and the
        external flag is not set and the value is a simple scalar,
        then locat must be explicitly passed as it is the unit number
         to read the array from
    ext_filename : string
        the external filename to write the array representation to
        (optional) (the default is None) .
        If type(value) is a string and is an accessible filename,
        the ext_filename is reset to value.
    bin : bool
        flag to control writing external arrays as binary (optional)
        (the default is False)

    Attributes
    ----------
    transient_2ds : dict{kper:Util2d}
        the transient sequence of Util2d objects

    Methods
    -------
    get_kper_entry : (itmp,string)
        get the itmp value and the Util2d file entry of the value in
        transient_2ds in bin kper.  if kper < min(Transient2d.keys()),
        return (1,zero_entry<Util2d>).  If kper > < min(Transient2d.keys()),
        but is not found in Transient2d.keys(), return (-1,'')

    See Also
    --------

    Notes
    -----

    Examples
    --------

    """

    def __init__(
        self,
        model,
        shape,
        dtype,
        value,
        name,
        fmtin=None,
        cnstnt=1.0,
        iprn=-1,
        ext_filename=None,
        locat=None,
        bin=False,
        array_free_format=None,
    ):

        if isinstance(value, Transient2d):
            for attr in value.__dict__.items():
                setattr(self, attr[0], attr[1])
            for kper, u2d in self.transient_2ds.items():
                self.transient_2ds[kper] = Util2d(
                    model,
                    u2d.shape,
                    u2d.dtype,
                    u2d._array,
                    name=u2d.name,
                    fmtin=u2d.format.fortran,
                    locat=locat,
                    cnstnt=u2d.cnstnt,
                    ext_filename=u2d.filename,
                    array_free_format=array_free_format,
                )

            self._model = model
            return

        self._model = model
        if len(shape) != 2:
            raise ValueError(
                "Transient2d: expected 2 dimensions (nrow, ncol), found "
                "shape {0}".format(shape)
            )
        if shape[0] is None:
            # allow for unstructured so that ncol changes by layer
            shape = (1, shape[1][0])

        self.shape = shape
        self._dtype = dtype
        self.__value = value
        self.name_base = name
        self.fmtin = fmtin
        self.cnstnt = cnstnt
        self.iprn = iprn
        self.locat = locat
        self.array_free_format = array_free_format
        if model.external_path is not None:
            self.ext_filename_base = os.path.join(
                model.external_path, self.name_base.replace(" ", "_")
            )
        else:
            self.ext_filename_base = self.name_base.replace(" ", "_")
        self.transient_2ds = self.build_transient_sequence()
        return

    @property
    def name(self):
        return self.name_base

    @property
    def dtype(self):
        return self._dtype

    @property
    def model(self):
        return self._model

    @property
    def data_type(self):
        return DataType.transient2d

    @property
    def plottable(self):
        return True

    @staticmethod
    def masked4d_array_to_kper_dict(m4d):
        assert m4d.ndim == 4
        kper_dict = {}
        for kper, arr in enumerate(m4d):
            if np.all(np.isnan(arr)):
                continue
            elif np.any(np.isnan(arr)):
                raise Exception("masked value found in array")
            kper_dict[kper] = arr.copy()
        return kper_dict

    @classmethod
    def from_4d(cls, model, pak_name, m4ds):
        """construct a Transient2d instance from a
        dict(name: (masked) 4d numpy.ndarray
        Parameters
        ----------
            model : flopy.mbase derived type
            pak_name : str package name (e.g. RCH)
            m4ds : dict(name,(masked) 4d numpy.ndarray)
                each ndarray must have shape (nper,1,nrow,ncol).
                if an entire (nrow,ncol) slice is np.NaN, then
                that kper is skipped.
        Returns
        -------
            Transient2d instance
        """

        assert isinstance(m4ds, dict)
        keys = list(m4ds.keys())
        assert len(keys) == 1
        name = keys[0]
        m4d = m4ds[name]

        assert m4d.ndim == 4
        assert m4d.shape[0] == model.nper
        assert m4d.shape[1] == 1
        assert m4d.shape[2] == model.nrow
        assert m4d.shape[3] == model.ncol
        m4d = m4d.astype(np.float32)
        kper_dict = Transient2d.masked4d_array_to_kper_dict(m4d)
        return cls(
            model=model,
            shape=(model.nrow, model.ncol),
            value=kper_dict,
            dtype=m4d.dtype.type,
            name=name,
        )

    def __setattr__(self, key, value):
        if hasattr(self, "transient_2ds") and key == "cnstnt":
            # set cnstnt for each u2d
            for kper, u2d in self.transient_2ds.items():
                self.transient_2ds[kper].cnstnt = value
        elif hasattr(self, "transient_2ds") and key == "fmtin":
            # set fmtin for each u2d
            for kper, u2d in self.transient_2ds.items():
                self.transient_2ds[kper].format = ArrayFormat(
                    u2d, fortran=value
                )
        elif hasattr(self, "transient_2ds") and key == "how":
            # set how for each u2d
            for kper, u2d in self.transient_2ds.items():
                self.transient_2ds[kper].how = value
        # set the attribute for u3d, even for cnstnt
        super().__setattr__(key, value)

    def get_zero_2d(self, kper):
        name = self.name_base + str(kper + 1) + "(filled zero)"
        return Util2d(
            self._model,
            self.shape,
            self._dtype,
            0.0,
            name=name,
            array_free_format=self.array_free_format,
        )

    def to_shapefile(self, filename):
        """
        Export transient 2D data to a shapefile (as polygons). Adds an
            attribute for each unique Util2d instance in self.data

        Parameters
        ----------
        filename : str
            Shapefile name to write

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
        >>> ml.rch.rech.as_shapefile('test_rech.shp')
        """
        warn(
            "Deprecation warning: to_shapefile() is deprecated. use .export()",
            DeprecationWarning,
        )
        self.export(filename)

    def plot(
        self,
        filename_base=None,
        file_extension=None,
        kper=0,
        fignum=None,
        **kwargs
    ):
        """
        Plot transient 2-D model input data

        Parameters
        ----------
        filename_base : str
            Base file name that will be used to automatically generate file
            names for output image files. Plots will be exported as image
            files if file_name_base is not None. (default is None)
        file_extension : str
            Valid matplotlib.pyplot file extension for savefig(). Only used
            if filename_base is not None. (default is 'png')
        kper : int or str
            model stress period. if 'all' is provided, all stress periods
            will be plotted
        fignum: list or int
            Figure numbers for plot title

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
            kper : str
                MODFLOW zero-based stress period number to return. If
                kper='all' then data for all stress period will be
                extracted. (default is zero).

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
        >>> ml.rch.rech.plot()

        """
        from flopy.plot import PlotUtilities

        axes = PlotUtilities._plot_transient2d_helper(
            self,
            filename_base=filename_base,
            file_extension=file_extension,
            kper=kper,
            fignum=fignum,
            **kwargs
        )

        return axes

    def __getitem__(self, kper):
        if kper in list(self.transient_2ds.keys()):
            return self.transient_2ds[kper]
        elif kper < min(self.transient_2ds.keys()):
            return self.get_zero_2d(kper)
        else:
            for i in range(kper, -1, -1):
                if i in list(self.transient_2ds.keys()):
                    return self.transient_2ds[i]
            raise Exception(
                "Transient2d.__getitem__(): error: "
                "could not find an entry before kper {0:d}".format(kper)
            )

    def __setitem__(self, key, value):
        try:
            key = int(key)
        except Exception as e:
            raise Exception(
                "Transient2d.__setitem__() error: "
                "'key'could not be cast to int:{0}".format(str(e))
            )
        nper = self._model.nper
        if key > self._model.nper or key < 0:
            raise Exception(
                "Transient2d.__setitem__() error: "
                "key {0} not in nper range {1}:{2}".format(key, 0, nper)
            )

        self.transient_2ds[key] = self.__get_2d_instance(key, value)

    @property
    def array(self):
        arr = np.zeros(
            (self._model.nper, 1, self.shape[0], self.shape[1]),
            dtype=self._dtype,
        )
        for kper in range(self._model.nper):
            u2d = self[kper]
            arr[kper, 0, :, :] = u2d.array
        return arr

    def export(self, f, **kwargs):
        from flopy import export

        return export.utils.transient2d_export(f, self, **kwargs)

    def get_kper_entry(self, kper):
        """
        Get the file entry info for a given kper
        returns (itmp,file entry string from Util2d)
        """
        if kper in self.transient_2ds:
            return (1, self.transient_2ds[kper].get_file_entry())
        elif kper < min(self.transient_2ds.keys()):
            return (1, self.get_zero_2d(kper).get_file_entry())
        else:
            return (-1, "")

    def build_transient_sequence(self):
        """
        parse self.__value into a dict{kper:Util2d}
        """

        # a dict keyed on kper (zero-based)
        if isinstance(self.__value, dict):
            tran_seq = {}
            for key, val in self.__value.items():
                try:
                    key = int(key)
                except:
                    raise Exception(
                        "Transient2d error: can't cast key: "
                        "{} to kper integer".format(key)
                    )
                if key < 0:
                    raise Exception(
                        "Transient2d error: key can't be negative: "
                        "{}".format(key)
                    )
                try:
                    u2d = self.__get_2d_instance(key, val)
                except Exception as e:
                    raise Exception(
                        "Transient2d error building Util2d instance from "
                        "value at kper: {}\n{}".format(key, e)
                    )
                tran_seq[key] = u2d
            return tran_seq

        # these are all for single entries - use the same Util2d for all kper
        # an array of shape (nrow,ncol)
        elif isinstance(self.__value, np.ndarray):
            return {0: self.__get_2d_instance(0, self.__value)}

        # a filename
        elif isinstance(self.__value, str):
            return {0: self.__get_2d_instance(0, self.__value)}

        # a scalar
        elif np.isscalar(self.__value):
            return {0: self.__get_2d_instance(0, self.__value)}

        # lists aren't allowed
        elif isinstance(self.__value, list):
            raise Exception(
                "Transient2d error: value cannot be a list "
                "anymore.  try a dict{kper,value}"
            )
        else:
            raise Exception(
                "Transient2d error: value type not recognized: "
                "{}".format(type(self.__value))
            )

    def __get_2d_instance(self, kper, arg):
        """
        parse an argument into a Util2d instance
        """
        ext_filename = None
        name = self.name_base + str(kper + 1)
        ext_filename = self.ext_filename_base + str(kper) + ".ref"
        u2d = Util2d(
            self._model,
            self.shape,
            self._dtype,
            arg,
            fmtin=self.fmtin,
            name=name,
            ext_filename=ext_filename,
            locat=self.locat,
            array_free_format=self.array_free_format,
        )
        return u2d


class Util2d(DataInterface):
    """
    Util2d class for handling 1- or 2-D model arrays

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.mf.Modflow`) to which
        this package will be added.
    shape : tuple
        Shape of the 1- or 2-D array
    dtype : [np.int32, np.float32, bool]
        the type of the data
    value : variable
        the data to be assigned to the 1- or 2-D array.
        can be a scalar, list, ndarray, or filename
    name : string
        name of the property (optional). (the default is None
    fmtin : string
        modflow fmtin variable (optional).  (the default is None)
    cnstnt : string
        modflow cnstnt variable (optional) (the default is 1.0)
    iprn : int
        modflow iprn variable (optional) (the default is -1)
    locat : int
        modflow locat variable (optional) (the default is None).  If the model
        instance does not support free format and the
        external flag is not set and the value is a simple scalar,
        then locat must be explicitly passed as it is the unit number
         to read the array from)
    ext_filename : string
        the external filename to write the array representation to
        (optional) (the default is None) .
        If type(value) is a string and is an accessible filename,
        the ext_filename is reset to value.
    bin : bool
        flag to control writing external arrays as binary (optional)
        (the default is False)

    Attributes
    ----------
    array : np.ndarray
        the array representation of the 2-D object
    how : str
        the str flag to control how the array is written to the model
        input files e.g. "constant","internal","external","openclose"
    format : ArrayFormat object
        controls the ASCII representation of the numeric array

    Methods
    -------
    get_file_entry : string
        get the model input file string including the control record

    See Also
    --------

    Notes
    -----
    If value is a valid filename and model.external_path is None, then a copy
    of the file is made and placed in model.model_ws directory.

    If value is a valid filename and model.external_path is not None, then
    a copy of the file is made a placed in the external_path directory.

    If value is a scalar, it is always written as a constant, regardless of
    the model.external_path setting.

    If value is an array and model.external_path is not None, then the array
    is written out in the external_path directory.  The name of the file that
    holds the array is created from the name attribute.  If the model supports
    "free format", then the array is accessed via the "open/close" approach.
    Otherwise, a unit number and filename is added to the name file.

    If value is an array and model.external_path is None, then the array is
    written internally to the model input file.

    Examples
    --------

    """

    def __init__(
        self,
        model,
        shape,
        dtype,
        value,
        name,
        fmtin=None,
        cnstnt=1.0,
        iprn=-1,
        ext_filename=None,
        locat=None,
        bin=False,
        how=None,
        array_free_format=None,
    ):
        """Create 1- or 2-d array

        Parameters
        ----------
        model : model object
        shape : tuple
            Dimensions of 1- or 2-D array, e.g. (nrow, ncol)
        dtype : int or np.float32
        value : str, list, np.int32, np.float32, bool or np.ndarray
        name : str
            Array name or description
        fmtin : str, optional
        cnstnt : np.int32 or np.float32, optional
            Array constant; default 1.0
        iprn : int, optional
            Modflow printing option; default -1
        ext_filename : str, optional
            Name of external files name where arrays are written
        locat : int, optional
        bin : bool, optional
            If True, writes unformatted files; default False writes formatted
        how : str, optional
            One of "constant", "internal", "external", or "openclose"
        array_free_format : bool, optional
            used for generating control record

        Notes
        -----
        Support with minimum of mem footprint, only creates arrays as needed,
        otherwise functions with strings or constants.

        Model instance string attribute "external_path" used to determine
        external array writing
        """
        if isinstance(value, Util2d):
            for attr in value.__dict__.items():
                setattr(self, attr[0], attr[1])
            self._model = model
            self._name = name
            self._ext_filename = self._name.replace(" ", "_") + ".ref"
            if ext_filename is not None:
                self.ext_filename = ext_filename.lower()
            else:
                self.ext_filename = None
            if locat is not None:
                self.locat = locat
            return

        # some defense
        if dtype != np.int32 and np.issubdtype(dtype, np.integer):
            # Modflow only uses 4-byte integers
            dtype = np.dtype(dtype)
            if np.dtype(int).itemsize != 4:
                # show warning for platforms where int is not 4-bytes
                warn(
                    "Util2d: setting integer dtype from {} to int32 for array {}".format(
                        dtype, name
                    )
                )
            dtype = np.int32
        if dtype not in [np.int32, np.float32, bool]:
            raise TypeError("Util2d:unsupported dtype: " + str(dtype))

        if name is not None:
            name = name.lower()
        if ext_filename is not None:
            ext_filename = ext_filename.lower()

        self._model = model
        if len(shape) not in (1, 2):
            raise ValueError(
                "Util2d: shape must describe 1- or 2-dimensions, "
                "e.g. (nrow, ncol)"
            )
        if min(shape) < 1:
            raise ValueError("Util2d: each shape dimension must be at least 1")
        self.shape = shape
        self._dtype = dtype
        self._name = name
        self.locat = locat
        self.parse_value(value)
        if self.vtype == str:
            fmtin = "(FREE)"
        self.__value_built = None
        self.cnstnt = dtype(cnstnt)

        self.iprn = iprn
        self._format = ArrayFormat(
            self, fortran=fmtin, array_free_format=array_free_format
        )
        self._format._isbinary = bool(bin)
        self.ext_filename = ext_filename
        self._ext_filename = self._name.replace(" ", "_") + ".ref"

        self._acceptable_hows = [
            "constant",
            "internal",
            "external",
            "openclose",
        ]

        if how is not None:
            how = how.lower()
            assert how in self._acceptable_hows
            self._how = how
        else:
            self._decide_how()

    @property
    def name(self):
        return self._name

    @property
    def dtype(self):
        return self._dtype

    @property
    def model(self):
        return self._model

    @property
    def data_type(self):
        return DataType.array2d

    @property
    def plottable(self):
        return True

    def _decide_how(self):
        # if a constant was passed in
        if self.vtype in [np.int32, np.float32]:
            self._how = "constant"
        # if a filename was passed in or external path was set
        elif self._model.external_path is not None or self.vtype == str:
            if self.format.array_free_format:
                self._how = "openclose"
            else:
                self._how = "external"
        else:
            self._how = "internal"

    def plot(
        self,
        title=None,
        filename_base=None,
        file_extension=None,
        fignum=None,
        **kwargs
    ):
        """
        Plot 2-D model input data

        Parameters
        ----------
        title : str
            Plot title. If a plot title is not provide one will be
            created based on data name (self._name). (default is None)
        filename_base : str
            Base file name that will be used to automatically generate file
            names for output image files. Plots will be exported as image
            files if file_name_base is not None. (default is None)
        file_extension : str
            Valid matplotlib.pyplot file extension for savefig(). Only used
            if filename_base is not None. (default is 'png')
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
        >>> ml.dis.top.plot()

        """
        from flopy.plot import PlotUtilities

        axes = PlotUtilities._plot_util2d_helper(
            self,
            title=title,
            filename_base=filename_base,
            file_extension=file_extension,
            fignum=fignum,
            **kwargs
        )
        return axes

    def export(self, f, **kwargs):
        from flopy import export

        return export.utils.array2d_export(f, self, **kwargs)

    def to_shapefile(self, filename):
        """
        Export 2-D model data to a shapefile (as polygons) of self.array

        Parameters
        ----------
        filename : str
            Shapefile name to write

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
        >>> ml.dis.top.as_shapefile('test_top.shp')
        """

        warn(
            "Deprecation warning: to_shapefile() is deprecated. use .export()",
            DeprecationWarning,
        )
        self.export(filename)

    def set_fmtin(self, fmtin):
        self._format = ArrayFormat(
            self,
            fortran=fmtin,
            array_free_format=self.format.array_free_format,
        )

    def get_value(self):
        return copy.deepcopy(self.__value)

    # overloads, tries to avoid creating arrays if possible
    def __add__(self, other):
        if self.vtype in [np.int32, np.float32] and self.vtype == other.vtype:
            return self.__value + other.get_value()
        else:
            return self.array + other.array

    def __sub__(self, other):
        if self.vtype in [np.int32, np.float32] and self.vtype == other.vtype:
            return self.__value - other.get_value()
        else:
            return self.array - other.array

    def __mul__(self, other):
        if np.isscalar(other):
            return Util2d(
                self._model,
                self.shape,
                self._dtype,
                self._array * other,
                self._name,
                self.format.fortran,
                self.cnstnt,
                self.iprn,
                self.ext_filename,
                self.locat,
                self.format.binary,
            )
        else:
            raise NotImplementedError(
                "Util2d.__mul__() not implemented for non-scalars"
            )

    def __eq__(self, other):
        if not isinstance(other, Util2d):
            return False
        if not np.array_equal(other.array, self.array):
            return False
        if other.cnstnt != self.cnstnt:
            return False
        return True

    def __getitem__(self, k):
        if isinstance(k, int):
            if len(self.shape) == 1:
                return self.array[k]
            elif self.shape[0] == 1:
                return self.array[0, k]
            elif self.shape[1] == 1:
                return self.array[k, 0]
            else:
                raise Exception(
                    "Util2d.__getitem__() error: an integer was passed, "
                    "self.shape > 1 in both dimensions"
                )
        else:
            if isinstance(k, tuple):
                if len(k) == 2:
                    return self.array[k[0], k[1]]
                if len(k) == 1:
                    return self.array[k]
            else:
                return self.array[(k,)]

    def __setitem__(self, k, value):
        """
        this one is dangerous because it resets __value
        """
        a = self.array
        a[k] = value
        a = a.astype(self._dtype)
        self.__value = a
        if self.__value_built is not None:
            self.__value_built = None

    def __setattr__(self, key, value):
        if key == "fmtin":
            self._format = ArrayFormat(self, fortran=value)
        elif key == "format":
            assert isinstance(value, ArrayFormat)
            self._format = value
        elif key == "how":
            value = value.lower()
            assert value in self._acceptable_hows
            self._how = value
        elif key == "model":
            self._model = value
        else:
            super().__setattr__(key, value)

    def all(self):
        return self.array.all()

    def __len__(self):
        return self.shape[0]

    def sum(self):
        return self.array.sum()

    def unique(self):
        return np.unique(self.array)

    @property
    def format(self):
        # don't return a copy because we want to allow
        # access to the attributes of ArrayFormat
        return self._format

    @property
    def how(self):
        return copy.copy(self._how)

    @property
    def vtype(self):
        return type(self.__value)

    @property
    def python_file_path(self):
        """
        where python is going to write the file
        Returns
        -------
            file_path (str) : path relative to python: includes model_ws
        """
        # if self.vtype != str:
        #    raise Exception("Util2d call to python_file_path " +
        #                    "for vtype != str")
        python_file_path = ""
        if self._model.model_ws != ".":
            python_file_path = os.path.join(self._model.model_ws)
        if self._model.external_path is not None:
            python_file_path = os.path.join(
                python_file_path, self._model.external_path
            )
        python_file_path = os.path.join(python_file_path, self.filename)
        return python_file_path

    @property
    def filename(self):
        if self.vtype != str:
            if self.ext_filename is not None:
                filename = os.path.split(self.ext_filename)[-1]
            else:
                filename = os.path.split(self._ext_filename)[-1]
        else:
            filename = os.path.split(self.__value)[-1]
        return filename

    @property
    def model_file_path(self):
        """
        where the model expects the file to be

        Returns
        -------
            file_path (str): path relative to the name file

        """

        model_file_path = ""
        if self._model.external_path is not None:
            model_file_path = os.path.join(
                model_file_path, self._model.external_path
            )
        model_file_path = os.path.join(model_file_path, self.filename)
        return model_file_path

    def get_constant_cr(self, value):

        if self.format.array_free_format:
            lay_space = "{0:>27s}".format("")
            if self.vtype in [int, np.int32]:
                lay_space = "{0:>32s}".format("")
            cr = "CONSTANT " + self.format.py[1].format(value)
            cr = "{0:s}{1:s}#{2:<30s}\n".format(cr, lay_space, self._name)
        else:
            cr = self._get_fixed_cr(0, value=value)
        return cr

    def _get_fixed_cr(self, locat, value=None):
        fformat = self.format.fortran
        if value is None:
            value = self.cnstnt
        if self.format.binary:
            if locat is None:
                raise Exception(
                    "Util2d._get_fixed_cr(): locat is None but "
                    "format is binary"
                )
            if not self.format.array_free_format:
                locat = -1 * np.abs(locat)
        if locat is None:
            locat = 0
        if locat == 0:
            fformat = ""
        if self.dtype == np.int32:
            cr = "{0:>10.0f}{1:>10.0f}{2:>19s}{3:>10.0f} #{4}\n".format(
                locat, value, fformat, self.iprn, self._name
            )
        elif self._dtype == np.float32:
            cr = "{0:>10.0f}{1:>10.5G}{2:>19s}{3:>10.0f} #{4}\n".format(
                locat, value, fformat, self.iprn, self._name
            )
        else:
            raise Exception(
                "Util2d: error generating fixed-format control record, "
                "dtype must be np.int32 or np.float32"
            )
        return cr

    def get_internal_cr(self):
        if self.format.array_free_format:
            cr = "INTERNAL {0:15} {1:>10s} {2:2.0f} #{3:<30s}\n".format(
                self.cnstnt_str, self.format.fortran, self.iprn, self._name
            )
            return cr
        else:
            return self._get_fixed_cr(self.locat)

    @property
    def cnstnt_str(self):
        if isinstance(self.cnstnt, str):
            return self.cnstnt
        else:
            return "{0:15.6G}".format(self.cnstnt)

    def get_openclose_cr(self):
        cr = "OPEN/CLOSE  {0:>30s} {1:15} {2:>10s} {3:2.0f} {4:<30s}\n".format(
            self.model_file_path,
            self.cnstnt_str,
            self.format.fortran,
            self.iprn,
            self._name,
        )
        return cr

    def get_external_cr(self):
        locat = self._model.next_ext_unit()
        # if self.format.binary:
        #    locat = -1 * np.abs(locat)
        self._model.add_external(
            self.model_file_path, locat, self.format.binary
        )
        if self.format.array_free_format:
            cr = "EXTERNAL  {0:>30d} {1:15} {2:>10s} {3:2.0f} {4:<30s}\n".format(
                locat,
                self.cnstnt_str,
                self.format.fortran,
                self.iprn,
                self._name,
            )
            return cr
        else:
            return self._get_fixed_cr(locat)

    def get_file_entry(self, how=None):

        if how is not None:
            how = how.lower()
        else:
            how = self._how

        if not self.format.array_free_format and self.format.free:
            print(
                "Util2d {0}: can't be free format...resetting".format(
                    self._name
                )
            )
            self.format._isfree = False

        if (
            not self.format.array_free_format
            and self.how == "internal"
            and self.locat is None
        ):
            print(
                "Util2d {0}: locat is None, but model does not "
                "support free format and how is internal... "
                "resetting how = external".format(self._name)
            )
            how = "external"

        if (self.format.binary or self._model.external_path) and how in [
            "constant",
            "internal",
        ]:
            print("Util2d:{0}: resetting 'how' to external".format(self._name))
            if self.format.array_free_format:
                how = "openclose"
            else:
                how = "external"
        if how == "internal":
            assert (
                not self.format.binary
            ), "Util2d error: 'how' is internal, but format is binary"
            cr = self.get_internal_cr()
            return cr + self.string

        elif how == "external" or how == "openclose":
            if how == "openclose":
                assert self.format.array_free_format, (
                    "Util2d error: 'how' is openclose, "
                    "but model doesn't support free fmt"
                )

            # write a file if needed
            if self.vtype != str:
                if self.format.binary:
                    self.write_bin(
                        self.shape,
                        self.python_file_path,
                        self._array,
                        bintype="head",
                    )
                else:
                    self.write_txt(
                        self.shape,
                        self.python_file_path,
                        self._array,
                        fortran_format=self.format.fortran,
                    )

            elif self.__value != self.python_file_path:
                if os.path.exists(self.python_file_path):
                    # if the file already exists, remove it
                    if self._model.verbose:
                        print(
                            "Util2d warning: removing existing array "
                            "file {0}".format(self.model_file_path)
                        )
                    try:
                        os.remove(self.python_file_path)
                    except Exception as e:
                        raise Exception(
                            "Util2d: error removing existing file "
                            + self.python_file_path
                        )
                # copy the file to the new model location
                try:
                    shutil.copy2(self.__value, self.python_file_path)
                except Exception as e:
                    raise Exception(
                        "Util2d.get_file_array(): error copying "
                        "{0} to {1}:{2}".format(
                            self.__value, self.python_file_path, str(e)
                        )
                    )
            if how == "external":
                return self.get_external_cr()
            else:
                return self.get_openclose_cr()

        elif how == "constant":
            if self.vtype not in [np.int32, np.float32]:
                u = np.unique(self._array)
                assert (
                    u.shape[0] == 1
                ), "Util2d error: 'how' is constant, but array is not uniform"
                value = u[0]
            else:
                value = self.__value
            return self.get_constant_cr(value)

        else:
            raise Exception(
                "Util2d.get_file_entry() error: "
                "unrecognized 'how':{0}".format(how)
            )

    @property
    def string(self):
        """
        get the string representation of value attribute

        Note:
            the string representation DOES NOT include the effects of the control
            record multiplier - this method is used primarily for writing model input files

        """
        # convert array to sting with specified format
        a_string = self.array2string(
            self.shape, self._array, python_format=self.format.py
        )
        return a_string

    @property
    def array(self):
        """
        Get the COPY of array representation of value attribute with the
        effects of the control record multiplier applied.

        Returns
        -------
        array : numpy.ndarray
            Copy of the array with the multiplier applied.

        Note
        ----
            .array is a COPY of the array representation as seen by the
            model - with the effects of the control record multiplier applied.

        """
        if isinstance(self.cnstnt, str):
            print("WARNING: cnstnt is str for {0}".format(self.name))
            return self._array.astype(self.dtype)
        if isinstance(self.cnstnt, (int, np.int32)):
            cnstnt = self.cnstnt
        else:
            if self.cnstnt == 0.0:
                cnstnt = 1.0
            else:
                cnstnt = self.cnstnt
        # return a copy of self._array since it is being
        # multiplied
        return (self._array * cnstnt).astype(self._dtype)

    @property
    def _array(self):
        """
        get the array representation of value attribute
        if value is a string or a constant, the array is loaded/built only once

        Note:
            the return array representation DOES NOT include the effect of the multiplier
            in the control record.  To get the array as the model sees it (with the multiplier applied),
            use the Util2d.array method.
        """
        if self.vtype == str:
            if self.__value_built is None:
                file_in = open(self.__value, "r")

                if self.format.binary:
                    header, self.__value_built = Util2d.load_bin(
                        self.shape, file_in, self._dtype, bintype="head"
                    )
                else:
                    self.__value_built = Util2d.load_txt(
                        self.shape, file_in, self._dtype, self.format.fortran
                    ).astype(self._dtype)
                file_in.close()
            return self.__value_built
        elif self.vtype != np.ndarray:
            if self.__value_built is None:
                self.__value_built = (
                    np.ones(self.shape, dtype=self._dtype) * self.__value
                )
            return self.__value_built
        else:
            return self.__value

    @staticmethod
    def load_block(shape, file_in, dtype):
        """Load block format from a MT3D file to a 2-D array

        Parameters
        ----------
        shape : tuple of int
            Array dimensions (nrow, ncol)
        file_in : file or str
            Filename or file handle
        dtype : np.int32 or np.float32

        Returns
        -------
        2-D array
        """
        if len(shape) != 2:
            raise ValueError(
                "Util2d.load_block(): expected 2 dimensions, found shape {0}".format(
                    shape
                )
            )
        nrow, ncol = shape
        data = np.ma.zeros(shape, dtype=dtype)
        data.mask = True
        openfile = not hasattr(file_in, "read")
        if openfile:
            file_in = open(file_in, "r")
        line = file_in.readline().strip()
        nblock = int(line.split()[0])
        for n in range(nblock):
            line = file_in.readline().strip()
            raw = line.split()
            if len(raw) < 5:
                raise ValueError(
                    "Util2d.load_block(): expected 5 items, "
                    "found {0}: {1}".format(len(raw), line)
                )
            i1, i2 = int(raw[0]) - 1, int(raw[1])
            j1, j2 = int(raw[2]) - 1, int(raw[3])
            data[i1:i2, j1:j2] = raw[4]
        if openfile:
            file_in.close()
        if data.mask.any():
            warn("Util2d.load_block(): blocks do not cover full array")
        return data.data

    @staticmethod
    def load_txt(shape, file_in, dtype, fmtin):
        """Load formatted file to a 1-D or 2-D array

        Parameters
        ----------
        shape : tuple of int
            One or two array dimensions
        file_in : file or str
            Filename or file handle
        dtype : np.int32 or np.float32
        fmtin : str
            Fortran array format descriptor, '(FREE)' or e.g. '(10G11.4)'

        Notes
        -----
        This method is similar to MODFLOW's U1DREL, U1DINT, U2DREL and U2DINT
        subroutines, but only for formatted files.

        Returns
        -------
        1-D or 2-D array
        """
        if len(shape) == 1:
            num_items = shape[0]
        elif len(shape) == 2:
            nrow, ncol = shape
            num_items = nrow * ncol
        else:
            raise ValueError(
                "Util2d.load_txt(): expected 1 or 2 dimensions, found shape {0}".format(
                    shape
                )
            )
        openfile = not hasattr(file_in, "read")
        if openfile:
            file_in = open(file_in, "r")
        npl, fmt, width, decimal = ArrayFormat.decode_fortran_descriptor(fmtin)
        items = []
        while len(items) < num_items:
            line = file_in.readline()
            if len(line) == 0:
                raise ValueError("Util2d.load_txt(): no data found")
            if npl == "free":
                if "," in line:
                    line = line.replace(",", " ")
                if "*" in line:  # use slower method for these types of lines
                    for item in line.split():
                        if "*" in item:
                            num, val = item.split("*")
                            # repeat val num times
                            items += int(num) * [val]
                        else:
                            items.append(item)
                else:
                    items += line.split()
            else:  # fixed width
                pos = 0
                for i in range(npl):
                    try:
                        item = line[pos : pos + width].strip()
                        pos += width
                        if item:
                            items.append(item)
                    except IndexError:
                        break
        if openfile:
            file_in.close()
        data = np.fromiter(items, dtype=dtype, count=num_items)
        if data.size != num_items:
            raise ValueError(
                "Util2d.load_txt(): expected array size {0},"
                " but found size {1}".format(num_items, data.size)
            )
        return data.reshape(shape)

    @staticmethod
    def write_txt(
        shape, file_out, data, fortran_format="(FREE)", python_format=None
    ):
        if fortran_format.upper() == "(FREE)" and python_format is None:
            np.savetxt(
                file_out,
                np.atleast_2d(data),
                ArrayFormat.get_default_numpy_fmt(data.dtype),
                delimiter="",
            )
            return
        if not hasattr(file_out, "write"):
            file_out = open(file_out, "w")
        file_out.write(
            Util2d.array2string(
                shape,
                data,
                fortran_format=fortran_format,
                python_format=python_format,
            )
        )

    @staticmethod
    def array2string(shape, data, fortran_format="(FREE)", python_format=None):
        """
        return a string representation of
        a (possibly wrapped format) array from a file
        (self.__value) and casts to the proper type (self._dtype)
        made static to support the load functionality
        this routine now supports fixed format arrays where the numbers
        may touch.
        """
        if len(shape) == 2:
            nrow, ncol = shape
        else:
            nrow = 1
            ncol = shape[0]
        data = np.atleast_2d(data)
        if python_format is None:
            (
                column_length,
                fmt,
                width,
                decimal,
            ) = ArrayFormat.decode_fortran_descriptor(fortran_format)
            if decimal is None:
                output_fmt = "{0}0:{1}{2}{3}".format("{", width, "d", "}")
            else:
                output_fmt = "{0}0:{1}.{2}{3}{4}".format(
                    "{", width, decimal, fmt, "}"
                )
        else:
            try:
                column_length, output_fmt = (
                    int(python_format[0]),
                    python_format[1],
                )
            except:
                raise Exception(
                    "Util2d.write_txt: \nunable to parse "
                    "python_format:\n    {0}\n"
                    "  python_format should be a list with\n"
                    "   [column_length, fmt]\n"
                    "    e.g., [10, {0:10.2e}]".format(python_format)
                )
        # write the array to a string
        len_data = data.size
        str_fmt_data = [
            output_fmt.format(d) + "\n"
            if (((i + 1) % column_length == 0.0) and (i != 0 or ncol == 1))
            or ((i + 1 == ncol) and (ncol != 1))
            or (i + 1 == len_data)
            else output_fmt.format(d)
            for i, d in enumerate(data.flatten())
        ]
        s = "".join(str_fmt_data)
        return s

    @staticmethod
    def load_bin(shape, file_in, dtype, bintype=None):
        """Load unformatted file to a 2-D array

        Parameters
        ----------
        shape : tuple of int
            One or two array dimensions
        file_in : file or str
            Filename or file handle
        dtype : np.int32 or np.float32
            Data type of unformatted file and Numpy array; use np.int32 for
            Fortran's INTEGER, and np.float32 for Fortran's REAL data types.
        bintype : str
            Normally 'Head'

        Notes
        -----
        This method is similar to MODFLOW's U2DREL and U2DINT subroutines,
        but only for unformatted files.

        Returns
        -------
        2-D array
        """
        import flopy.utils.binaryfile as bf

        nrow, ncol = shape
        num_items = nrow * ncol
        if dtype != np.int32 and np.issubdtype(dtype, np.integer):
            # Modflow only uses 4-byte integers
            dtype = np.dtype(dtype)
            if dtype.itemsize != 4:
                # show warning for platforms where int is not 4-bytes
                warn(
                    "Util2d: setting integer dtype from {0} to int32".format(
                        dtype
                    )
                )
            dtype = np.int32
        openfile = not hasattr(file_in, "read")
        if openfile:
            file_in = open(file_in, "rb")
        header_data = None
        if bintype is not None and np.issubdtype(dtype, np.floating):
            header_dtype = bf.BinaryHeader.set_dtype(bintype=bintype)
            header_data = np.fromfile(file_in, dtype=header_dtype, count=1)
        data = np.fromfile(file_in, dtype=dtype, count=num_items)
        if openfile:
            file_in.close()
        if data.size != num_items:
            raise ValueError(
                "Util2d.load_bin(): expected array size {0},"
                " but found size {1}".format(num_items, data.size)
            )
        return header_data, data.reshape(shape)

    @staticmethod
    def write_bin(shape, file_out, data, bintype=None, header_data=None):
        if not hasattr(file_out, "write"):
            file_out = open(file_out, "wb")
        dtype = data.dtype
        if bintype is not None:
            if header_data is None:
                header_data = BinaryHeader.create(
                    bintype=bintype, nrow=shape[0], ncol=shape[1]
                )
        if header_data is not None:
            header_data.tofile(file_out)
        data.tofile(file_out)
        return

    def parse_value(self, value):
        """
        parses and casts the raw value into an acceptable format for __value
        lot of defense here, so we can make assumptions later
        """
        if isinstance(value, list):
            value = np.array(value)

        if isinstance(value, bool):
            if self._dtype == bool:
                try:
                    self.__value = bool(value)

                except:
                    raise Exception(
                        "Util2d:could not cast "
                        'boolean value to type "bool": {}'.format(value)
                    )
            else:
                raise Exception(
                    "Util2d:value type is bool, but dtype not set as bool"
                )
        elif isinstance(value, str):
            if os.path.exists(value):
                self.__value = value
                return
            elif self.dtype == np.int32:
                try:
                    self.__value = np.int32(value)
                except:
                    raise Exception(
                        "Util2d error: str not a file and "
                        "couldn't be cast to int: {0}".format(value)
                    )

            else:
                try:
                    self.__value = float(value)
                except:
                    raise Exception(
                        "Util2d error: str not a file and "
                        "couldn't be cast to float: {0}".format(value)
                    )

        elif np.isscalar(value):
            if self.dtype == np.int32:
                try:
                    self.__value = np.int32(value)
                except:
                    raise Exception(
                        "Util2d:could not cast scalar "
                        'value to type "int": {}'.format(value)
                    )
            elif self._dtype == np.float32:
                try:
                    self.__value = np.float32(value)
                except:
                    raise Exception(
                        "Util2d:could not cast "
                        'scalar value to type "float": {}'.format(value)
                    )

        elif isinstance(value, np.ndarray):
            # if value is 3d, but dimension 1 is only length 1,
            # then drop the first dimension
            if len(value.shape) == 3 and value.shape[0] == 1:
                value = value[0]
            if self.shape != value.shape:
                raise Exception(
                    "Util2d:self.shape: {} does not match value.shape: "
                    "{}".format(self.shape, value.shape)
                )
            if self._dtype != value.dtype:
                value = value.astype(self._dtype)
            self.__value = value

        else:
            raise Exception(
                "Util2d:unsupported type in util_array: " + str(type(value))
            )

    @classmethod
    def load(
        cls,
        f_handle,
        model,
        shape,
        dtype,
        name,
        ext_unit_dict=None,
        array_free_format=None,
        array_format="modflow",
    ):
        """
        functionality to load Util2d instance from an existing
        model input file.
        external and internal record types must be fully loaded
        if you are using fixed format record types,make sure
        ext_unit_dict has been initialized from the NAM file
        """
        if shape == (0, 0):
            raise IndexError(
                "No information on model grid dimensions. "
                "Need nrow, ncol to load a Util2d array."
            )
        curr_unit = None
        if ext_unit_dict is not None:
            # determine the current file's unit number
            cfile = f_handle.name
            for cunit in ext_unit_dict:
                if cfile == ext_unit_dict[cunit].filename:
                    curr_unit = cunit
                    break

        # Allows for special MT3D array reader
        # array_format = None
        # if hasattr(model, 'array_format'):
        #    array_format = model.array_format

        cr_dict = Util2d.parse_control_record(
            f_handle.readline(),
            current_unit=curr_unit,
            dtype=dtype,
            ext_unit_dict=ext_unit_dict,
            array_format=array_format,
        )

        if cr_dict["type"] == "constant":
            u2d = cls(
                model,
                shape,
                dtype,
                cr_dict["cnstnt"],
                name=name,
                iprn=cr_dict["iprn"],
                fmtin="(FREE)",
                array_free_format=array_free_format,
            )

        elif cr_dict["type"] == "open/close":
            # clean up the filename a little
            fname = cr_dict["fname"]
            fname = fname.replace("'", "")
            fname = fname.replace('"', "")
            fname = fname.replace("'", "")
            fname = fname.replace('"', "")
            fname = fname.replace("\\", os.path.sep)
            fname = os.path.join(model.model_ws, fname)
            # load_txt(shape, file_in, dtype, fmtin):
            assert os.path.exists(
                fname
            ), "Util2d.load() error: open/close file {} not found".format(
                fname
            )
            if str("binary") not in str(cr_dict["fmtin"].lower()):
                f = open(fname, "r")
                data = Util2d.load_txt(
                    shape=shape, file_in=f, dtype=dtype, fmtin=cr_dict["fmtin"]
                )
            else:
                f = open(fname, "rb")
                header_data, data = Util2d.load_bin(
                    shape, f, dtype, bintype="Head"
                )
            f.close()
            u2d = cls(
                model,
                shape,
                dtype,
                data,
                name=name,
                iprn=cr_dict["iprn"],
                fmtin="(FREE)",
                cnstnt=cr_dict["cnstnt"],
                array_free_format=array_free_format,
            )

        elif cr_dict["type"] == "internal":
            data = Util2d.load_txt(shape, f_handle, dtype, cr_dict["fmtin"])
            u2d = cls(
                model,
                shape,
                dtype,
                data,
                name=name,
                iprn=cr_dict["iprn"],
                fmtin="(FREE)",
                cnstnt=cr_dict["cnstnt"],
                locat=None,
                array_free_format=array_free_format,
            )

        elif cr_dict["type"] == "external":
            ext_unit = ext_unit_dict[cr_dict["nunit"]]
            if ext_unit.filehandle is None:
                raise IOError(
                    "cannot read unit {0}, filename: {1}".format(
                        cr_dict["nunit"], ext_unit.filename
                    )
                )
            elif "binary" not in str(cr_dict["fmtin"].lower()):
                assert cr_dict["nunit"] in list(ext_unit_dict.keys())
                data = Util2d.load_txt(
                    shape, ext_unit.filehandle, dtype, cr_dict["fmtin"]
                )
            else:
                if cr_dict["nunit"] not in list(ext_unit_dict.keys()):
                    cr_dict["nunit"] *= -1
                assert cr_dict["nunit"] in list(ext_unit_dict.keys())
                header_data, data = Util2d.load_bin(
                    shape, ext_unit.filehandle, dtype, bintype="Head"
                )
            u2d = cls(
                model,
                shape,
                dtype,
                data,
                name=name,
                iprn=cr_dict["iprn"],
                fmtin="(FREE)",
                cnstnt=cr_dict["cnstnt"],
                array_free_format=array_free_format,
            )
            # track this unit number so we can remove it from the external
            # file list later
            model.pop_key_list.append(cr_dict["nunit"])
        elif cr_dict["type"] == "block":
            data = Util2d.load_block(shape, f_handle, dtype)
            u2d = cls(
                model,
                shape,
                dtype,
                data,
                name=name,
                iprn=cr_dict["iprn"],
                fmtin="(FREE)",
                cnstnt=cr_dict["cnstnt"],
                locat=None,
                array_free_format=array_free_format,
            )

        return u2d

    @staticmethod
    def parse_control_record(
        line,
        current_unit=None,
        dtype=np.float32,
        ext_unit_dict=None,
        array_format=None,
    ):
        """
        parses a control record when reading an existing file
        rectifies fixed to free format
        current_unit (optional) indicates the unit number of the file being parsed
        """
        free_fmt = ["open/close", "internal", "external", "constant"]
        raw = line.strip().split()
        freefmt, cnstnt, fmtin, iprn, nunit = None, None, None, -1, None
        fname = None
        isfloat = False
        if dtype == float or dtype == np.float32:
            isfloat = True
            # if free format keywords
        if str(raw[0].lower()) in str(free_fmt):
            freefmt = raw[0].lower()
            if raw[0].lower() == "constant":
                if isfloat:
                    cnstnt = float(raw[1].lower().replace("d", "e"))
                else:
                    cnstnt = int(raw[1].lower())
            if raw[0].lower() == "internal":
                if isfloat:
                    cnstnt = float(raw[1].lower().replace("d", "e"))
                else:
                    cnstnt = int(raw[1].lower())
                fmtin = raw[2].strip()
                iprn = 0
                if len(raw) >= 4:
                    iprn = int(raw[3])
            elif raw[0].lower() == "external":
                if ext_unit_dict is not None:
                    try:
                        # td = ext_unit_dict[int(raw[1])]
                        fname = ext_unit_dict[int(raw[1])].filename.strip()
                    except:
                        print(
                            "   could not determine filename "
                            "for unit {}".format(raw[1])
                        )

                nunit = int(raw[1])
                if isfloat:
                    cnstnt = float(raw[2].lower().replace("d", "e"))
                else:
                    cnstnt = int(raw[2].lower())
                fmtin = raw[3].strip()
                iprn = 0
                if len(raw) >= 5:
                    iprn = int(raw[4])
            elif raw[0].lower() == "open/close":
                fname = raw[1].strip()
                if isfloat:
                    cnstnt = float(raw[2].lower().replace("d", "e"))
                else:
                    cnstnt = int(raw[2].lower())
                fmtin = raw[3].strip()
                iprn = 0
                if len(raw) >= 5:
                    iprn = int(raw[4])
                npl, fmt, width, decimal = None, None, None, None
        else:
            locat = int(line[0:10].strip())
            if isfloat:
                if len(line) >= 20:
                    cnstnt = float(
                        line[10:20].strip().lower().replace("d", "e")
                    )
                else:
                    cnstnt = 0.0
            else:
                if len(line) >= 20:
                    cnstnt = int(line[10:20].strip())
                else:
                    cnstnt = 0
                # if cnstnt == 0:
                #    cnstnt = 1
            if locat != 0:
                if len(line) >= 40:
                    fmtin = line[20:40].strip()
                else:
                    fmtin = ""
                try:
                    iprn = int(line[40:50].strip())
                except:
                    iprn = 0
            # locat = int(raw[0])
            # cnstnt = float(raw[1])
            # fmtin = raw[2].strip()
            # iprn = int(raw[3])
            if locat == 0:
                freefmt = "constant"
            elif locat < 0:
                freefmt = "external"
                nunit = int(locat) * -1
                fmtin = "(binary)"
            elif locat > 0:
                # if the unit number matches the current file, it's internal
                if locat == current_unit:
                    freefmt = "internal"
                else:
                    freefmt = "external"
                nunit = int(locat)

            # Reset for special MT3D control flags
            if array_format == "mt3d":
                if locat == 100:
                    freefmt = "internal"
                    nunit = current_unit
                elif locat == 101:
                    freefmt = "block"
                    nunit = current_unit
                elif locat == 102:
                    raise NotImplementedError(
                        "MT3D zonal format not supported..."
                    )
                elif locat == 103:
                    freefmt = "internal"
                    nunit = current_unit
                    fmtin = "(free)"

        cr_dict = {}
        cr_dict["type"] = freefmt
        cr_dict["cnstnt"] = cnstnt
        cr_dict["nunit"] = nunit
        cr_dict["iprn"] = iprn
        cr_dict["fmtin"] = fmtin
        cr_dict["fname"] = fname
        return cr_dict
