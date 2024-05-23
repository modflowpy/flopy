import numpy as np
from numpy.lib.recfunctions import repack_fields


def create_empty_recarray(length, dtype, default_value=0):
    """
    Create a empty recarray with a defined default value for floats.

    Parameters
    ----------
    length : int
        Shape of the empty recarray.
    dtype : np.dtype
        dtype of the empty recarray.
    default_value : float
        default value to use for floats in recarray.

    Returns
    -------
    r : np.recarray
        Recarray of type dtype with shape length.

    Examples
    --------
    >>> import numpy as np
    >>> from flopy.utils import create_empty_recarray
    >>> dt = np.dtype([('x', np.float32), ('y', np.float32)])
    >>> create_empty_recarray(1, dt)
    rec.array([(0., 0.)],
              dtype=[('x', '<f4'), ('y', '<f4')])
    """
    r = np.zeros(length, dtype=dtype)
    msg = "dtype argument must be an instance of np.dtype, not list."
    assert isinstance(dtype, np.dtype), msg
    for name in dtype.names:
        dt = dtype.fields[name][0]
        if np.issubdtype(dt, np.float64):
            r[name] = default_value
    return r.view(np.recarray)


def ra_slice(ra, cols):
    """
    Create a slice of a recarray

    .. deprecated:: 3.5
        Use numpy.lib.recfunctions.repack_fields instead

    Parameters
    ----------
    ra : np.recarray
        recarray to extract a limited number of columns from.
    cols : list of str
        List of key names to extract from ra.

    Returns
    -------
    ra_slice : np.recarray
        Slice of ra

    Examples
    --------
    >>> import numpy as np
    >>> from flopy.utils import ra_slice
    >>> a = np.core.records.fromrecords([("a", 1, 1.1), ("b", 2, 2.1)])
    >>> ra_slice(a, ['f0', 'f1'])
    rec.array([('a', 1), ('b', 2)],
              dtype=[('f0', '<U1'), ('f1', '<i4')])
    """
    return repack_fields(ra[cols])


def recarray(array, dtype):
    """
    Convert a list of lists or tuples to a recarray.

    .. deprecated:: 3.5
        Use numpy.core.records.fromrecords instead

    Parameters
    ----------
    array : list of lists
        list of lists containing data to convert to a recarray. The number of
        entries in each list in the list must be the same.
    dtype : np.dtype
        dtype of the array data

    Returns
    -------
    r : np.recarray
        Recarray of type dtype with shape equal to the length of array.

    Examples
    --------
    >>> import numpy as np
    >>> import flopy
    >>> dt = np.dtype([('x', np.float32), ('y', np.float32)])
    >>> a = [(1., 2.), (10., 20.), (100., 200.)]
    >>> flopy.utils.recarray(a, dt)
    rec.array([(  1.,   2.), ( 10.,  20.), (100., 200.)],
              dtype=[('x', '<f4'), ('y', '<f4')])
    """
    array = np.atleast_2d(array)

    # convert each entry of the list to a tuple
    if not isinstance(array[0], tuple):
        array = list(map(tuple, array))
    return np.array(array, dtype=dtype).view(np.recarray)
