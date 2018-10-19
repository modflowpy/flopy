import numpy as np


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
    >>> import flopy
    >>> dtype = np.dtype([('x', np.float32), ('y', np.float32)])
    >>> ra = flopy.utils.create_empty_recarray(10, dtype)

    """
    r = np.zeros(length, dtype=dtype)
    msg = 'dtype argument must be an instance of np.dtype, not list.'
    assert isinstance(dtype, np.dtype), msg
    for name in dtype.names:
        dt = dtype.fields[name][0]
        if np.issubdtype(dt, np.float_):
            r[name] = default_value
    return r.view(np.recarray)


def ra_slice(ra, cols):
    """
    Create a slice of a recarray

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
    >>> import flopy
    >>> raslice = flopy.utils.ra_slice(ra, ['x', 'y'])


    """
    raslice = np.column_stack([ra[c] for c in cols])
    dtype = [(str(d[0]), str(d[1])) for d in ra.dtype.descr if d[0] in cols]
    return np.array([tuple(r) for r in raslice],
                    dtype=dtype).view(np.recarray)


def recarray(array, dtype):
    """
    Convert a list of lists or tuples to a recarray.

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
    >>> dtype = np.dtype([('x', np.float32), ('y', np.float32)])
    >>> arr = [(1., 2.), (10., 20.), (100., 200.)]
    >>> ra = flopy.utils.recarray(arr, dtype)

    """
    array = np.atleast_2d(array)
    # convert each entry of the list to a tuple
    if not isinstance(array[0], tuple):
        array = list(map(tuple, array))
    return np.array(array, dtype=dtype).view(np.recarray)
