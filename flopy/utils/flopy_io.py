"""
Module for input/output utilities
"""
import os
import sys
import numpy as np

try:
    import pandas as pd
except:
    pd = False


def _fmt_string(array, float_format="{}"):
    """
    makes a formatting string for a rec-array;
    given a desired float_format.

    Parameters
    ----------
    array : np.recarray
    float_format : str
        formatter for floating point variable

    Returns
    -------
    fmt_string : str
        formatting string for writing output
    """
    fmt_string = ""
    for field in array.dtype.descr:
        vtype = field[1][1].lower()
        if vtype == "i":
            fmt_string += "{:.0f} "
        elif vtype == "f":
            fmt_string += "{} ".format(float_format)
        elif vtype == "o":
            fmt_string += "{} "
        elif vtype == "s":
            raise Exception(
                "MfList error: 'str' type found in dtype. "
                "This gives unpredictable results when "
                "recarray to file - change to 'object' type"
            )
        else:
            raise Exception(
                "MfList.fmt_string error: unknown vtype in dtype:" + vtype
            )
    return fmt_string


def line_strip(line):
    """
    Remove comments and replace commas from input text
    for a free formatted modflow input file

    Parameters
    ----------
        line : str
            a line of text from a modflow input file

    Returns
    -------
        str : line with comments removed and commas replaced
    """
    for comment_flag in [";", "#", "!!"]:
        line = line.split(comment_flag)[0]
    line = line.strip()
    return line.replace(",", " ")


def multi_line_strip(fobj):
    """
    Get next line that is not blank or is not a comment line
    from a free formatted modflow input file

    Parameters
    ----------
        fobj : open file object
            a line of text from an input file

    Returns
    -------
        str : line with comments removed and commas replaced

    """
    while True:
        line = line_strip(fobj.readline())
        if line:
            return line.lower()


def get_next_line(f):
    """
    Get the next line from a file that is not a blank line

    Parameters
    ----------
    f : filehandle
        filehandle to a open file

    Returns
    -------
    line : string
        next non-empty line in a open file


    """
    while True:
        line = f.readline().rstrip()
        if len(line) > 0:
            break
    return line


def line_parse(line):
    """
    Convert a line of text into to a list of values.  This handles the
    case where a free formatted MODFLOW input file may have commas in
    it.
    """
    line = line_strip(line)
    return line.split()


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
            raise Exception(
                "user provided ipos length ({}) should be greater than or "
                "equal to the length of v ({})".format(len(ipos), ncol)
            )
    out = ""
    for n in range(ncol):
        if free:
            write_fmt = "{} "
        else:
            width = ipos[n]
            if isinstance(v[n], (float, np.float32, np.float64)):
                decimal = width - 6
                vmin, vmax = 10 ** -decimal, 10 ** decimal
                if abs(v[n]) < vmin or abs(v[n]) > vmax:
                    ctype = "g"  # default precision is 6 if not specified
                else:
                    ctype = ".{}f".format(decimal)
                    # evaluate if the fixed format value will exceed width
                    if (
                        len("{{:>{}{}}}".format(width, ctype).format(v[n]))
                        > width
                    ):
                        ctype = ".{}g".format(decimal)  # preserve precision
            elif isinstance(v[n], (int, np.int32, np.int64)):
                ctype = "d"
            else:
                ctype = ""
            write_fmt = "{{:>{}{}}}".format(width, ctype)
        out += write_fmt.format(v[n])
    if comment is not None:
        out += "  # {}".format(comment)
    out += "\n"
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


def flux_to_wel(cbc_file, text, precision="single", model=None, verbose=False):
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

    cbf = CBF(cbc_file, precision=precision, verbose=verbose)

    # create a empty numpy array of shape (time,layer,row,col)
    m4d = np.zeros((cbf.nper, cbf.nlay, cbf.nrow, cbf.ncol), dtype=np.float32)
    m4d[:] = np.NaN

    # process the records in the cell budget file
    iper = -1
    for kstpkper in cbf.kstpkper:

        kstpkper = (kstpkper[0] - 1, kstpkper[1] - 1)
        kper = kstpkper[1]
        # if we haven't visited this kper yet
        if kper != iper:
            arr = cbf.get_data(kstpkper=kstpkper, text=text, full3D=True)
            if len(arr) > 0:
                arr = arr[0]
                print(arr.max(), arr.min(), arr.sum())
                # masked where zero
                arr[np.where(arr == 0.0)] = np.NaN
                m4d[iper + 1] = arr
            iper += 1

    # model wasn't passed, then create a generic model
    if model is None:
        model = Modflow("test")
    # if model doesn't have a wel package, then make a generic one...
    # need this for the from_m4d method
    if model.wel is None:
        ModflowWel(model)

    # get the stress_period_data dict {kper:np recarray}
    sp_data = MfList.from_4d(model, "WEL", {"flux": m4d})

    wel = ModflowWel(model, stress_period_data=sp_data)
    return wel


def loadtxt(
    file, delimiter=" ", dtype=None, skiprows=0, use_pandas=True, **kwargs
):
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
    # test if pandas should be used, if available
    if use_pandas:
        if pd:
            if delimiter.isspace():
                kwargs["delim_whitespace"] = True
            if isinstance(dtype, np.dtype) and "names" not in kwargs:
                kwargs["names"] = dtype.names

    # if use_pandas and pd then use pandas
    if use_pandas and pd:
        df = pd.read_csv(file, dtype=dtype, skiprows=skiprows, **kwargs)
        return df.to_records(index=False)
    # default use of numpy
    else:
        return np.loadtxt(file, dtype=dtype, skiprows=skiprows, **kwargs)


def get_url_text(url, error_msg=None):
    """
    Get text from a url.
    """
    from urllib.request import urlopen

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


def ulstrd(f, nlist, ra, model, sfac_columns, ext_unit_dict):
    """
    Read a list and allow for open/close, binary, external, sfac, etc.

    Parameters
    ----------
    f : file handle
        file handle for where the list is being read from
    nlist : int
        size of the list (number of rows) to read
    ra : np.recarray
        A record array of the correct size that will be filled with the list
    model : model object
        The model object (of type :class:`flopy.modflow.mf.Modflow`) to
        which this package will be added.
    sfac_columns : list
        A list of strings containing the column names to scale by sfac
    ext_unit_dict : dictionary, optional
        If the list in the file is specified using EXTERNAL,
        then in this case ext_unit_dict is required, which can be
        constructed using the function
        :class:`flopy.utils.mfreadnam.parsenamefile`.

    Returns
    -------

    """

    # initialize variables
    line = f.readline()
    sfac = 1.0
    binary = False
    ncol = len(ra.dtype.names)
    line_list = line.strip().split()
    close_the_file = False
    file_handle = f
    mode = "r"

    # check for external
    if line.strip().lower().startswith("external"):
        inunit = int(line_list[1])
        errmsg = "Could not find a file for unit {}".format(inunit)
        if ext_unit_dict is not None:
            if inunit in ext_unit_dict:
                namdata = ext_unit_dict[inunit]
                file_handle = namdata.filehandle
            else:
                raise IOError(errmsg)
        else:
            raise IOError(errmsg)
        if namdata.filetype == "DATA(BINARY)":
            binary = True
        if not binary:
            line = file_handle.readline()

    # or check for open/close
    elif line.strip().lower().startswith("open/close"):
        raw = line.strip().split()
        fname = raw[1]
        if "/" in fname:
            raw = fname.split("/")
        elif "\\" in fname:
            raw = fname.split("\\")
        else:
            raw = [fname]
        fname = os.path.join(*raw)
        oc_filename = os.path.join(model.model_ws, fname)
        msg = (
            "Package.load() error: open/close filename "
            + oc_filename
            + " not found"
        )
        assert os.path.exists(oc_filename), msg
        if "(binary)" in line.lower():
            binary = True
            mode = "rb"
        file_handle = open(oc_filename, mode)
        close_the_file = True
        if not binary:
            line = file_handle.readline()

    # check for scaling factor
    if not binary:
        line_list = line.strip().split()
        if line.strip().lower().startswith("sfac"):
            sfac = float(line_list[1])
            line = file_handle.readline()

    # fast binary read fromfile
    if binary:
        dtype2 = []
        for name in ra.dtype.names:
            dtype2.append((name, np.float32))
        dtype2 = np.dtype(dtype2)
        d = np.fromfile(file_handle, dtype=dtype2, count=nlist)
        ra = np.array(d, dtype=ra.dtype)
        ra = ra.view(np.recarray)

    # else, read ascii
    else:

        for ii in range(nlist):

            # first line was already read
            if ii != 0:
                line = file_handle.readline()

            if model.free_format_input:
                # whitespace separated
                t = line.strip().split()
                if len(t) < ncol:
                    t = t + (ncol - len(t)) * [0.0]
                else:
                    t = t[:ncol]
                t = tuple(t)
                ra[ii] = t
            else:
                # fixed format
                t = read_fixed_var(line, ncol=ncol)
                t = tuple(t)
                ra[ii] = t

    # scale the data and check
    for column_name in sfac_columns:
        ra[column_name] *= sfac
        if "auxsfac" in ra.dtype.names:
            ra[column_name] *= ra["auxsfac"]

    if close_the_file:
        file_handle.close()

    return ra


def get_ts_sp(line):
    """
    Reader method to get time step and stress period numbers from
    list files and Modflow other output files

    Parameters
    ----------
    line : str
        line containing information about the stress period and time step.
        The line must contain "STRESS PERIOD   <x> TIME STEP   <y>"

    Returns
    -------
        tuple of stress period and time step numbers
    """
    # Get rid of nasty things
    line = line.replace(",", "").replace("*", "")

    searchstring = "TIME STEP"
    idx = line.index(searchstring) + len(searchstring)
    ll = line[idx:].strip().split()
    ts = int(ll[0])

    searchstring = "STRESS PERIOD"
    idx = line.index(searchstring) + len(searchstring)
    ll = line[idx:].strip().split()
    sp = int(ll[0])

    return ts, sp
