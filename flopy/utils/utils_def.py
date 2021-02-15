"""
Generic classes and utility functions
"""

from datetime import timedelta
import numpy as np


class FlopyBinaryData(object):
    """
    The FlopyBinaryData class is a class to that defines the data types for
    integer, floating point, and character data in MODFLOW binary
    files. The FlopyBinaryData class is the super class from which the
    specific derived classes are formed.  This class should not be
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
        if precision.lower() == "double":
            self.real = np.float64
            self.floattype = "f8"
        else:
            self.real = np.float32
            self.floattype = "f4"
        self.realbyte = self.real(1).nbytes
        return

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
