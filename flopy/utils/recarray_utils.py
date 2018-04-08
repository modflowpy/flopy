import numpy as np

def create_empty_recarray(length, dtype, default_value=0):
    r = np.zeros(length, dtype=dtype)
    assert isinstance(dtype, np.dtype), "dtype argument must be an instance of np.dtype, not list."
    for name in dtype.names:
        dt = dtype.fields[name][0]
        if np.issubdtype(dt, np.float_):
            r[name] = default_value
    return r.view(np.recarray)

def ra_slice(ra, cols):
    raslice = np.column_stack([ra[c] for c in cols])
    dtype = [(str(d[0]), d[1]) for d in ra.dtype.descr if d[0] in cols]
    return np.array([tuple(r) for r in raslice],
                    dtype=dtype).view(np.recarray)

def recarray(array, dtype):
    # handle sequences of lists
    # (recarrays must be constructed from tuples)
    array = np.atleast_2d(array)
    if not isinstance(array[0], tuple):
        array = list(map(tuple, array))
    return np.array(array, dtype=dtype).view(np.recarray)