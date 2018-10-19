"""
Module for input/output utilities
"""
import sys
import numpy as np

def _fmt_string(array, float_format='{}'):
    """makes a formatting string for a rec-array; given a desired float_format."""
    fmt_string = ''
    for field in array.dtype.descr:
        vtype = field[1][1].lower()
        if (vtype == 'i'):
            fmt_string += '{:.0f} '
        elif (vtype == 'f'):
            fmt_string += '{} '.format(float_format)
        elif (vtype == 'o'):
            fmt_string += '{} '
        elif (vtype == 's'):
            raise Exception("MfList error: '\str\' type found it dtype." + \
                            " This gives unpredictable results when " + \
                            "recarray to file - change to \'object\' type")
        else:
            raise Exception("MfList.fmt_string error: unknown vtype " + \
                            "in dtype:" + vtype)
    return fmt_string

def line_parse(line):
    """
    Convert a line of text into to a list of values.  This handles the
    case where a free formatted MODFLOW input file may have commas in
    it.
    """
    for comment_flag in [';', '#', '!!']:
        line = line.split(comment_flag)[0]
    line = line.replace(',', ' ')
    return line.strip().split()

def pop_item(line, dtype=str):
    if len(line) > 0:
        if dtype == str:
            return line.pop(0)
        elif dtype == float:
            return float(line.pop(0))
        elif dtype == int:
            # handle strings like this:
            # '-10.'
            return int(float(line.pop(0)))
    return dtype(0)

def read_nwt_options(f):
    """convert options codeblock to single line."""
    options = []
    while True:
        options += line_parse(f.readline().lower())
        if 'end' in options:
            return ' '.join(options[:-1])




def write_fixed_var(v, length=10, ipos=None, free=False, comment=None):
    """

    Parameters
    ----------
    v : list, int, float, bool, or numpy array
        list, int, float, bool, or numpy array containing the data to be
        written to a string.
    length : int
        length of each column for fixed column widths. (default is 10)
    ipos : list, int, or numpy array
        user-provided column widths. (default is None)
    free : bool
        boolean indicating if a free format string should be generated.
        length and ipos are not used if free is True. (default is False)
    comment : str
        comment string to add to the end of the string

    Returns
    -------
    out : str
        fixed or free format string generated using user-provided data

    """
    if isinstance(v, np.ndarray):
        v = v.tolist()
    elif isinstance(v, int) or isinstance(v, float) or isinstance(v, bool):
        v = [v]
    ncol = len(v)
    # construct ipos if it was not passed
    if ipos is None:
        ipos = []
        for i in range(ncol):
            ipos.append(length)
    else:
        if isinstance(ipos, np.ndarray):
            ipos = ipos.flatten().tolist()
        elif isinstance(ipos, int):
            ipos = [ipos]
        if len(ipos) < ncol:
            err = 'user provided ipos length ({})'.format(len(ipos)) + \
                  'should be greater than or equal ' + \
                  'to the length of v ({})'.format(ncol)
            raise Exception(err)
    out = ''
    for n in range(ncol):
        if free:
            write_fmt = '{} '
        else:
            write_fmt = '{{:>{}}}'.format(ipos[n])
        out += write_fmt.format(v[n])
    if comment is not None:
        out += '  # {}'.format(comment)
    out += '\n'
    return out

def read_fixed_var(line, ncol=1, length=10, ipos=None, free=False):
    """
    Parse a fixed format line using user provided data

    Parameters
    ----------
    line : str
        text string to parse.
    ncol : int
        number of columns to parse from line. (default is 1)
    length : int
        length of each column for fixed column widths. (default is 10)
    ipos : list, int, or numpy array
        user-provided column widths. (default is None)
    free : bool
        boolean indicating if sting is free format. ncol, length, and
        ipos are not used if free is True. (default is False)

    Returns
    -------
    out : list
        padded list containing data parsed from the passed text string

    """
    if free:
        out = line.rstrip().split()
    else:
        # construct ipos if it was not passed
        if ipos is None:
            ipos = []
            for i in range(ncol):
                ipos.append(length)
        else:
            if isinstance(ipos, np.ndarray):
                ipos = ipos.flatten().tolist()
            elif isinstance(ipos, int):
                ipos = [ipos]
            ncol = len(ipos)
        line = line.rstrip()
        out = []
        istart = 0
        for ivar in range(ncol):
            istop = istart + ipos[ivar]
            try:
                txt = line[istart:istop]
                if len(txt.strip()) > 0:
                    out.append(txt)
                else:
                    out.append(0)
            except:
                break
            istart = istop
    return out

def flux_to_wel(cbc_file,text,precision="single",model=None,verbose=False):
    """
    Convert flux in a binary cell budget file to a wel instance

    Parameters
    ----------
    cbc_file : (str) cell budget file name
    text : (str) text string of the desired flux type (e.g. "drains")
    precision : (optional str) precision of the cell budget file
    model : (optional) BaseModel instance.  If passed, a new ModflowWel
        instance will be added to model
    verbose : bool flag passed to CellBudgetFile

    Returns
    -------
    flopy.modflow.ModflowWel instance

    """
    from . import CellBudgetFile as CBF
    from .util_list import MfList
    from ..modflow import Modflow, ModflowWel
    cbf = CBF(cbc_file,precision=precision,verbose=verbose)

    # create a empty numpy array of shape (time,layer,row,col)
    m4d = np.zeros((cbf.nper,cbf.nlay,cbf.nrow,cbf.ncol),dtype=np.float32)
    m4d[:] = np.NaN

    # process the records in the cell budget file
    iper = -1
    for kstpkper in cbf.kstpkper:

        kstpkper = (kstpkper[0]-1,kstpkper[1]-1)
        kper = kstpkper[1]
        #if we haven't visited this kper yet
        if kper != iper:
            arr = cbf.get_data(kstpkper=kstpkper,text=text,full3D=True)
            if len(arr) > 0:
                arr = arr[0]
                print(arr.max(),arr.min(),arr.sum())
                # masked where zero
                arr[np.where(arr==0.0)] = np.NaN
                m4d[iper+1] = arr
            iper += 1



    # model wasn't passed, then create a generic model
    if model is None:
        model = Modflow("test")
    # if model doesn't have a wel package, then make a generic one...
    # need this for the from_m4d method
    if model.wel is None:
        ModflowWel(model)

    # get the stress_period_data dict {kper:np recarray}
    sp_data = MfList.from_4d(model,"WEL",{"flux":m4d})

    wel = ModflowWel(model,stress_period_data=sp_data)
    return wel

def loadtxt(file, delimiter=' ', dtype=None, skiprows=0, use_pandas=True,
            **kwargs):
    """
    Use pandas if it is available to load a text file
    (significantly faster than n.loadtxt or genfromtxt see
    http://stackoverflow.com/questions/18259393/numpy-loading-csv-too-slow-compared-to-matlab)

    Parameters
    ----------
    file : file or str
        File, filename, or generator to read.
    delimiter : str, optional
        The string used to separate values. By default, this is any whitespace.
    dtype : data-type, optional
        Data-type of the resulting array
    skiprows : int, optional
        Skip the first skiprows lines; default: 0.
    use_pandas : bool
        If true, the much faster pandas.read_csv method is used.
    kwargs : dict
        Keyword arguments passed to numpy.loadtxt or pandas.read_csv.

    Returns
    -------
    ra : np.recarray
        Numpy record array of file contents.
    """
    try:
        if use_pandas:
            import pandas as pd
            if delimiter.isspace():
                kwargs['delim_whitespace'] = True
            if isinstance(dtype, np.dtype) and 'names' not in kwargs:
                kwargs['names'] = dtype.names
    except:
        if use_pandas:
            msg = 'loadtxt: pandas is not available'
            raise ImportError(msg)
        pd = False

    if use_pandas and pd:
        df = pd.read_csv(file, dtype=dtype, skiprows=skiprows, **kwargs)
        return df.to_records(index=False)
    else:
        return np.loadtxt(file, dtype=dtype, skiprows=skiprows, **kwargs)

def get_url_text(url, error_msg=None):
    """Get text from a url, using either python 3 or 2."""
    try:
        # For Python 3.0 and later
        from urllib.request import urlopen
    except ImportError:
        # Fall back to Python 2's urllib2
        from urllib2 import urlopen
    try:
        urlobj = urlopen(url)
        text = urlobj.read().decode()
        return text
    except:
        e = sys.exc_info()
        print(e)
        if error_msg is not None:
            print(error_msg)
        return
