"""
Generic classes and utility functions
"""

from datetime import timedelta
from typing import Literal, get_args
from warnings import warn

import numpy as np

Precision = Literal["single", "double"]


class FlopyBinaryData:
    """
    Defines integer, floating point, and character data types for
    MODFLOW binary files.
    """

    def __init__(self):
        self.precision = "double"

    @property
    def integer(self) -> type:
        return np.int32

    @property
    def integerbyte(self) -> int:
        return self.integer(1).nbytes

    @property
    def character(self) -> type:
        return np.uint8

    @property
    def textbyte(self) -> int:
        return 1

    @property
    def precision(self) -> Precision:
        return self._precision

    @precision.setter
    def precision(self, value: Precision):
        if value not in get_args(Precision):
            raise ValueError(
                f"Invalid floating precision '{value}', expected 'single' or 'double'"
            )

        value = value.lower()
        self._precision = value
        if value == "double":
            self._real = np.float64
            self._floattype = "f8"
        else:
            self._real = np.float32
            self._floattype = "f4"
        self._realbyte = self.real(1).nbytes

    @property
    def real(self) -> type:
        return self._real

    @property
    def realbyte(self) -> int:
        return self._realbyte

    @property
    def floattype(self) -> str:
        return self._floattype

    def set_float(self, precision):
        """
        Set floating point precision.

            .. deprecated:: 3.5
                This method will be removed in Flopy 3.10+.
                Use ``precision`` property setter instead.
        """
        warn(
            "set_float() is deprecated, use precision property setter instead",
            PendingDeprecationWarning,
        )
        self.precision = precision

    def read_text(self, nchar=20):
        bytesvalue = self._read_values(self.character, nchar).tobytes()
        return bytesvalue.decode().strip()

    def read_integer(self):
        return self._read_values(self.integer, 1)[0]

    def read_real(self):
        return self._read_values(self.real, 1)[0]

    def read_record(self, count, dtype=None):
        if dtype is None:
            dtype = self.dtype
        return self._read_values(dtype, count)

    def _read_values(self, dtype, count):
        return np.fromfile(self.file, dtype, count)


def totim_to_datetime(totim, start="1-1-1970", timeunit="D"):
    """

    Parameters
    ----------
    totim : list or numpy array

    start : str
        Starting date for simulation. (default is 1-1-1970).
    timeunit : string
        time unit of the simulation time. Valid values are 'S'econds,
        'M'inutes, 'H'ours, 'D'ays, 'Y'ears. (default is 'D').

    Returns
    -------
    out : list
        datetime object calculated from start and totim values

    """
    key = None
    fact = 1.0
    if timeunit.upper() == "S":
        key = "seconds"
    elif timeunit.upper() == "M":
        key = "minutes"
    elif timeunit.upper() == "H":
        key = "hours"
    elif timeunit.upper() == "D":
        key = "days"
    elif timeunit.upper() == "Y":
        key = "days"
        fact = 365.25
    else:
        err = (
            "'S'econds, 'M'inutes, 'H'ours, 'D'ays, 'Y'ears are the "
            + "only timeunit values that can be passed to totim_"
            + "to_datetime() function"
        )
        raise Exception(err)
    out = []
    kwargs = {}
    for to in totim:
        kwargs[key] = to * fact
        t = timedelta(**kwargs)
        out.append(start + t)
    return out


def get_pak_vals_shape(model, vals):
    """Function to define shape of package input data for Util2d.

    Parameters
    ----------
    model : flopy model object
    vals : Package input values (dict of arrays or scalars, or ndarray, or
        single scalar).

    Returns
    -------
    shape: tuple
        shape of input data for Util2d

    """
    nrow, ncol, nlay, nper = model.nrow_ncol_nlay_nper
    if nrow is None:  # unstructured
        if isinstance(vals, dict):
            try:  # check for iterable
                _ = (v for v in next(iter(vals.values())))
            except:
                return (1, ncol[0])  # default to layer 1 node count
            return np.array(next(iter(vals.values())), ndmin=2).shape
        else:
            # check for single iterable
            try:
                _ = (v for v in vals)
            except:
                return (1, ncol[0])  # default to layer 1 node count
            return np.array(vals, ndmin=2).shape
    else:
        return (nrow, ncol)  # structured


def get_util2d_shape_for_layer(model, layer=0):
    """
    Define nrow and ncol for array (Util2d) shape of a given layer in
    structured and/or unstructured models.

    Parameters
    ----------
    model : model object
        model for which Util2d shape is sought.
    layer : int
        layer (base 0) for which Util2d shape is sought.

    Returns
    -------
    (nrow,ncol) : tuple of ints
        util2d shape for the given layer
    """
    nr, nc, _, _ = model.get_nrow_ncol_nlay_nper()
    if nr is None:  # unstructured
        nrow = 1
        ncol = nc[layer]
    else:  # structured
        nrow = nr
        ncol = nc

    return (nrow, ncol)


def get_unitnumber_from_ext_unit_dict(model, pak_class, ext_unit_dict=None, ipakcb=0):
    """
    For a given modflow package, defines input file unit number,
    plus package input and (optionally) output (budget) save file names.

    Parameters
    ----------
    model : model object
        model for which the unit number is sought.
    pak_class : modflow package class for which the unit number is sought.
    ext_unit_dict : external unit dictionary, optional.
        If not provided, unitnumber and filenames will be returned as None.
    ipakcb : int, optional
        Modflow package unit number on which budget is saved.
        Default is 0, in which case the returned output file is None.

    Returns
    -------
    unitnumber : int
        file unit number for the given modflow package (or None)
    filenames : list
        list of [package input file name, budget file name],
    """
    unitnumber = None
    filenames = [None, None]
    if ext_unit_dict is not None:
        unitnumber, filenames[0] = model.get_ext_dict_attr(
            ext_unit_dict, filetype=pak_class._ftype()
        )
        if ipakcb > 0:
            _, filenames[1] = model.get_ext_dict_attr(ext_unit_dict, unit=ipakcb)
            model.add_pop_key_list(ipakcb)

    return unitnumber, filenames


def type_from_iterable(_iter, index=0, _type=int, default_val=0):
    """Returns value of specified type from iterable.

    Parameters
    ----------
    _iter : iterable
    index : int
        Iterable index to try to convert
    _type : Python type
    default_val : default value (0)

    Returns
    -------
    val : value of type _type, or default_val
    """
    try:
        val = _type(_iter[index])
    except ValueError:
        val = default_val
    except IndexError:
        val = default_val

    return val


def get_open_file_object(fname_or_fobj, read_write="rw"):
    """Returns an open file object for either a file name or open file object."""
    openfile = not (hasattr(fname_or_fobj, "read") or hasattr(fname_or_fobj, "write"))
    if openfile:
        filename = fname_or_fobj
        f_obj = open(filename, read_write)
    else:
        f_obj = fname_or_fobj

    return f_obj


def get_dis(model):
    """Returns dis or disu object from a given model object."""
    dis = model.dis
    if not model.structured:
        dis = model.disu
    return dis
