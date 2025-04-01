"""
Module to read MODFLOW binary output files.  The module contains four
important classes that can be accessed by the user.

*  HeadFile (Binary head file.  Can also be used for drawdown)
*  HeadUFile (Binary MODFLOW-USG unstructured head file)
*  UcnFile (Binary concentration file from MT3DMS)
*  CellBudgetFile (Binary cell-by-cell flow file)

"""

import os
import tempfile
import warnings
from pathlib import Path
from shutil import move
from typing import Optional, Union

import numpy as np
import pandas as pd

from flopy.discretization.modeltime import ModelTime

from ..datafile import Header, LayerFile
from ..gridutil import get_lni


class BinaryHeader(Header):
    """
    Represents data headers for binary output files.

    Parameters
    ----------
    bintype : str, default None
        Type of file being opened. Accepted values are 'head' and 'ucn'.
    precision : str, default 'single'
        Precision of floating point data in the file.

    """

    def __init__(self, bintype=None, precision="single"):
        super().__init__(bintype, precision)

    def set_values(self, **kwargs):
        """
        Set values using kwargs
        """
        ikey = [
            "ntrans",
            "kstp",
            "kper",
            "ncol",
            "nrow",
            "ilay",
            "ncpl",
            "nodes",
            "m1",
            "m2",
            "m3",
        ]
        fkey = ["pertim", "totim"]
        ckey = ["text"]
        for k in ikey:
            if k in kwargs.keys():
                try:
                    self.header[0][k] = int(kwargs[k])
                except:
                    print(f"{k} key not available in {self.header_type} header dtype")
        for k in fkey:
            if k in kwargs.keys():
                try:
                    self.header[0][k] = float(kwargs[k])
                except:
                    print(f"{k} key not available in {self.header_type} header dtype")
        for k in ckey:
            if k in kwargs.keys():
                # Convert to upper case to be consistent case used by MODFLOW
                # text strings. Necessary to work with HeadFile and UcnFile
                # routines
                ttext = kwargs[k].upper()
                # trim a long string
                if len(ttext) > 16:
                    text = ttext[0:16]
                # pad a short string
                elif len(ttext) < 16:
                    text = f"{ttext:<16}"
                # the string is just right
                else:
                    text = ttext
                self.header[0][k] = text
            else:
                self.header[0][k] = "DUMMY TEXT"

    @staticmethod
    def set_dtype(bintype=None, precision="single"):
        """
        Set the dtype

        """
        header = Header(filetype=bintype, precision=precision)
        return header.dtype

    @staticmethod
    def create(bintype=None, precision="single", **kwargs):
        """
        Create a binary header

        """
        header = BinaryHeader(bintype=bintype, precision=precision)
        if header.get_dtype() is not None:
            header.set_values(**kwargs)
        return header.get_values()


def binaryread_struct(file, vartype, shape=(1,), charlen=16):
    """
    Read text, a scalar value, or an array of values from a binary file.

        file : file object
            is an open file object
        vartype : type
            is the return variable type: str, numpy.int32, numpy.float32,
            or numpy.float64
        shape : tuple
            is the shape of the returned array (shape(1, ) returns a single
            value) for example, shape = (nlay, nrow, ncol)
        charlen : int
            is the length of the text string.  Note that string arrays
            cannot be returned, only multi-character strings.  Shape has no
            affect on strings.

    .. deprecated:: 3.8.0
       Use :meth:`binaryread` instead.

    """
    import struct

    warnings.warn(
        "binaryread_struct() is deprecated; use binaryread() instead.",
        DeprecationWarning,
    )

    # store the mapping from type to struct format (fmt)
    typefmtd = {np.int32: "i", np.float32: "f", np.float64: "d"}

    # read a string variable of length charlen
    if vartype == str:
        result = file.read(charlen * 1)

    # read other variable types
    else:
        fmt = typefmtd[vartype]
        # find the number of bytes for one value
        numbytes = vartype(1).nbytes
        # find the number of values
        nval = np.prod(shape)
        fmt = str(nval) + fmt
        s = file.read(numbytes * nval)
        result = struct.unpack(fmt, s)
        if nval == 1:
            result = vartype(result[0])
        else:
            result = np.array(result, dtype=vartype)
            result = np.reshape(result, shape)
    return result


def binaryread(file, vartype, shape=(1,), charlen=16):
    """
    Read character bytes, scalar or array values from a binary file.

    Parameters
    ----------
    file : file object
        is an open file object
    vartype : type
        is the return variable type: bytes, numpy.int32,
        numpy.float32, or numpy.float64. Using str is deprecated since
        bytes is preferred.
    shape : tuple, default (1,)
        is the shape of the returned array (shape(1, ) returns a single
        value) for example, shape = (nlay, nrow, ncol)
    charlen : int, default 16
        is the length character bytes.  Note that arrays of bytes
        cannot be returned, only multi-character bytes.  Shape has no
        affect on bytes.

    Raises
    ------
    EOFError
    """

    if vartype == str:
        # handle a hang-over from python2
        warnings.warn(
            "vartype=str is deprecated; use vartype=bytes instead.",
            DeprecationWarning,
        )
        vartype = bytes
    if vartype == bytes:
        # read character bytes of length charlen
        result = file.read(charlen)
        if len(result) < charlen:
            raise EOFError
    else:
        # find the number of values
        nval = np.prod(shape)
        result = np.fromfile(file, vartype, nval)
        if result.size < nval:
            raise EOFError
        if nval != 1:
            result = np.reshape(result, shape)
    return result


def join_struct_arrays(arrays):
    """
    Simple function that can join two numpy structured arrays.

    """
    newdtype = sum((a.dtype.descr for a in arrays), [])
    newrecarray = np.empty(len(arrays[0]), dtype=newdtype)
    for a in arrays:
        for name in a.dtype.names:
            newrecarray[name] = a[name]
    return newrecarray


def get_headfile_precision(filename: Union[str, os.PathLike]):
    """
    Determine precision of a MODFLOW head file.

    Parameters
    ----------
    filename : str or PathLike
        Path of binary MODFLOW file to determine precision.

    Returns
    -------
    str
        Result will be unknown, single, or double

    """

    # Set default result if neither single or double works
    result = "unknown"

    # Open file, and check filesize to ensure this is not an empty file
    f = open(filename, "rb")
    f.seek(0, 2)
    totalbytes = f.tell()
    f.seek(0, 0)  # reset to beginning
    assert f.tell() == 0
    if totalbytes == 0:
        raise ValueError(f"datafile error: file is empty: {filename}")

    # first try single
    vartype = [
        ("kstp", "<i4"),
        ("kper", "<i4"),
        ("pertim", "<f4"),
        ("totim", "<f4"),
        ("text", "S16"),
    ]
    hdr = binaryread(f, vartype)
    charbytes = list(hdr[0][4])
    if min(charbytes) >= 32 and max(charbytes) <= 126:
        # check if bytes are within conventional ASCII range
        result = "single"
        success = True
    else:
        success = False

    # next try double
    if not success:
        f.seek(0)
        vartype = [
            ("kstp", "<i4"),
            ("kper", "<i4"),
            ("pertim", "<f8"),
            ("totim", "<f8"),
            ("text", "S16"),
        ]
        hdr = binaryread(f, vartype)
        charbytes = list(hdr[0][4])
        if min(charbytes) >= 32 and max(charbytes) <= 126:
            result = "double"
        else:
            f.close()
            raise ValueError(
                f"Could not determine the precision of the headfile {filename}"
            )

    # close and return result
    f.close()
    return result


class BinaryLayerFile(LayerFile):
    """
    The BinaryLayerFile class is a parent class from which concrete
    classes inherit. This class should not be instantiated directly.

    Notes
    -----

    The BinaryLayerFile class is built on a record array consisting of
    headers, which are record arrays of the modflow header information
    (kstp, kper, pertim, totim, text, nrow, ncol, ilay), and long ints
    pointing to the 1st byte of data for the corresponding data arrays.
    """

    def __init__(self, filename: Union[str, os.PathLike], precision, verbose, **kwargs):
        super().__init__(filename, precision, verbose, **kwargs)

    def _build_index(self):
        """
        Build the recordarray and iposarray, which maps the header information
        to the position in the binary file.

        """
        header = self._get_header()
        self.nrow = header["nrow"]
        self.ncol = header["ncol"]
        if header["ilay"] > self.nlay:
            self.nlay = header["ilay"]

        if self.nrow < 0 or self.ncol < 0:
            raise Exception("negative nrow, ncol")

        warn_threshold = 10000000
        if self.nrow > 1 and self.nrow * self.ncol > warn_threshold:
            warnings.warn(
                f"Very large grid, ncol ({self.ncol}) * nrow ({self.nrow})"
                f" > {warn_threshold}"
            )
        self.file.seek(0, 2)
        self.totalbytes = self.file.tell()
        self.file.seek(0, 0)
        ipos = 0
        while ipos < self.totalbytes:
            header = self._get_header()
            self.recordarray.append(header)
            if self.text.upper() not in header["text"]:
                continue
            if ipos == 0:
                self.times.append(header["totim"])
                self.kstpkper.append((header["kstp"], header["kper"]))
            else:
                totim = header["totim"]
                if totim != self.times[-1]:
                    self.times.append(totim)
                    self.kstpkper.append((header["kstp"], header["kper"]))
            ipos = self.file.tell()
            self.iposarray.append(ipos)
            databytes = self.get_databytes(header)
            self.file.seek(databytes, 1)
            ipos = self.file.tell()

        # self.recordarray contains a recordarray of all the headers.
        self.recordarray = np.array(self.recordarray, dtype=self.header_dtype)
        self.iposarray = np.array(self.iposarray, dtype=np.int64)
        self.nlay = np.max(self.recordarray["ilay"])

        # provide headers as a pandas frame
        self.headers = pd.DataFrame(self.recordarray, index=self.iposarray)
        self.headers["text"] = (
            self.headers["text"].str.decode("ascii", "strict").str.strip()
        )

    def get_databytes(self, header):
        """

        Parameters
        ----------
        header : datafile.Header
            header object

        Returns
        -------
         databytes : int
            size of the data array, in bytes, following the header

        """
        return (
            np.int64(header["ncol"])
            * np.int64(header["nrow"])
            * np.int64(self.realtype(1).nbytes)
        )

    def _read_data(self, shp):
        return binaryread(self.file, self.realtype, shape=shp)

    def _get_header(self):
        """
        Read the file header

        """
        header = binaryread(self.file, self.header_dtype, (1,))
        return header[0]

    def get_ts(self, idx):
        """
        Get a time series from the binary file.

        Parameters
        ----------
        idx : tuple of ints, or a list of a tuple of ints
            idx can be (layer, row, column) or it can be a list in the form
            [(layer, row, column), (layer, row, column), ...].  The layer,
            row, and column values must be zero based.

        Returns
        -------
        out : numpy array
            Array has size (ntimes, ncells + 1).  The first column in the
            data array will contain time (totim).

        See Also
        --------

        Notes
        -----

        The layer, row, and column values must be zero-based, and must be
        within the following ranges: 0 <= k < nlay; 0 <= i < nrow; 0 <= j < ncol

        Examples
        --------

        """
        kijlist = self._build_kijlist(idx)
        nstation = self._get_nstation(idx, kijlist)

        # Initialize result array and put times in first column
        result = self._init_result(nstation)

        istat = 1
        for k, i, j in kijlist:
            ioffset = (i * self.ncol + j) * self.realtype(1).nbytes
            for irec, header in enumerate(self.recordarray):
                ilay = header["ilay"] - 1  # change ilay from header to zero-based
                if ilay != k:
                    continue
                ipos = self.iposarray[irec].item()

                # Calculate offset necessary to reach intended cell
                self.file.seek(ipos + int(ioffset), 0)

                # Find the time index and then put value into result in the
                # correct location.
                itim = np.asarray(result[:, 0] == header["totim"]).nonzero()[0]
                result[itim, istat] = binaryread(self.file, self.realtype)
            istat += 1
        return result


class HeadFile(BinaryLayerFile):
    """
    The HeadFile class provides simple ways to retrieve and manipulate
    2D or 3D head arrays, or time series arrays for one or more cells,
    from a binary head output file. A utility method is also provided
    to reverse the order of head data, for use with particle tracking
    simulations in which particles are tracked backwards in time from
    terminating to release locations (e.g., to compute capture zones).

    Parameters
    ----------
    filename : str or PathLike
        Path of the head file.
    text : string
        Name of the text string in the head file. Default is 'head'.
    precision : string
        Precision of floating point head data in the value. Accepted
        values are 'auto', 'single' or 'double'. Default is 'auto',
        which enables automatic detection of precision.
    verbose : bool
        Toggle logging output. Default is False.

    Examples
    --------

    >>> import flopy.utils.binaryfile as bf
    >>> hdobj = bf.HeadFile('model.hds', precision='single')
    >>> hdobj.headers
    >>> rec = hdobj.get_data(kstpkper=(0, 49))

    >>> ddnobj = bf.HeadFile('model.ddn', text='drawdown', precision='single')
    >>> ddnobj.headers
    >>> rec = ddnobj.get_data(totim=100.)

    """

    def __init__(
        self,
        filename: Union[str, os.PathLike],
        text="head",
        precision="auto",
        verbose=False,
        **kwargs,
    ):
        self.text = text.encode()
        if precision == "auto":
            precision = get_headfile_precision(filename)
            if precision == "unknown":
                s = f"Error. Precision could not be determined for {filename}"
                print(s)
                raise Exception()
        self.header_dtype = BinaryHeader.set_dtype(bintype="Head", precision=precision)
        super().__init__(filename, precision, verbose, **kwargs)

    def reverse(self, filename: Optional[os.PathLike] = None):
        """
        Reverse the time order of the currently loaded binary head file. If a head
        file name is not provided or the provided name is the same as the existing
        filename, the file will be overwritten and reloaded.

        Parameters
        ----------

        filename : str or PathLike
            Path of the reversed binary head file.
        """

        filename = (
            Path(filename).expanduser().absolute()
            if filename is not None
            else self.filename
        )

        time = ModelTime.from_headers(self.recordarray)
        time._set_totim_dict()
        trev = time.reverse()
        trev._set_totim_dict()
        nper = time.nper
        seen = set()

        def reverse_header(header):
            """Reverse period, step and time fields in the record header"""

            nonlocal seen
            kper = header["kper"] - 1
            kstp = header["kstp"] - 1
            header = header.copy()
            header["kper"] = nper - kper
            header["kstp"] = time.nstp[kper] - kstp
            kper = header["kper"] - 1
            kstp = header["kstp"] - 1
            seen.add((kper, kstp))
            header["pertim"] = trev._pertim_dict[(kper, kstp)]
            header["totim"] = trev._totim_dict[(kper, kstp)]
            return header

        target = filename

        # if rewriting the same file, write
        # temp file then copy it into place
        inplace = filename == self.filename
        if inplace:
            temp_dir_path = Path(tempfile.gettempdir())
            temp_file_path = temp_dir_path / filename.name
            target = temp_file_path

        # reverse record order
        with open(target, "wb") as f:
            for i in range(len(self) - 1, -1, -1):
                header = self.recordarray[i].copy()
                header = reverse_header(header)
                text = header["text"]
                ilay = header["ilay"]
                kstp = header["kstp"]
                kper = header["kper"]
                pertim = header["pertim"]
                totim = header["totim"]
                data = self.get_data(idx=i)[ilay - 1]
                dt = np.dtype(
                    [
                        ("kstp", np.int32),
                        ("kper", np.int32),
                        ("pertim", np.float64),
                        ("totim", np.float64),
                        ("text", "S16"),
                        ("ncol", np.int32),
                        ("nrow", np.int32),
                        ("ilay", np.int32),
                    ]
                )
                nrow = data.shape[0]
                ncol = data.shape[1]
                h = np.array(
                    (kstp, kper, pertim, totim, text, ncol, nrow, ilay), dtype=dt
                )
                h.tofile(f)
                data.tofile(f)

        # if we rewrote the original file, reinitialize
        if inplace:
            move(target, filename)
            super().__init__(filename, self.precision, self.verbose)


class UcnFile(BinaryLayerFile):
    """
    UcnFile Class.

    Parameters
    ----------
    filename : string
        Name of the concentration file
    text : string
        Name of the text string in the ucn file.  Default is 'CONCENTRATION'
    precision : string
        'auto', 'single' or 'double'.  Default is 'auto'.
    verbose : bool
        Write information to the screen.  Default is False.

    Attributes
    ----------

    Methods
    -------

    See Also
    --------

    Notes
    -----
    The UcnFile class provides simple ways to retrieve 2d and 3d
    concentration arrays from a MT3D binary head file and time series
    arrays for one or more cells.

    The BinaryLayerFile class is built on a record array consisting of
    headers, which are record arrays of the modflow header information
    (kstp, kper, pertim, totim, text, nrow, ncol, ilay)
    and long integers, which are pointers to first bytes of data for
    the corresponding data array.

    Examples
    --------

    >>> import flopy.utils.binaryfile as bf
    >>> ucnobj = bf.UcnFile('MT3D001.UCN', precision='single')
    >>> ucnobj.headers
    >>> rec = ucnobj.get_data(kstpkper=(0, 0))

    """

    def __init__(
        self,
        filename,
        text="concentration",
        precision="auto",
        verbose=False,
        **kwargs,
    ):
        self.text = text.encode()
        if precision == "auto":
            precision = get_headfile_precision(filename)
        if precision == "unknown":
            s = f"Error. Precision could not be determined for {filename}"
            print(s)
            raise Exception()
        self.header_dtype = BinaryHeader.set_dtype(bintype="Ucn", precision=precision)
        super().__init__(filename, precision, verbose, **kwargs)
        return


class HeadUFile(BinaryLayerFile):
    """
    The HeadUFile class provides simple ways to retrieve a list of
    head arrays from a MODFLOW-USG binary head file and time series
    arrays for one or more cells.

    Parameters
    ----------
    filename : str or PathLike
        Path of the head file
    text : string
        Name of the text string in the head file. Default is 'headu'.
    precision : string
        Precision of the floating point head data in the file. Accepted
        values are 'auto', 'single' or 'double'. Default is 'auto', which
        enables precision to be automatically detected.
    verbose : bool
        Toggle logging output. Default is False.

    Notes
    -----

    The BinaryLayerFile class is built on a record array consisting of
    headers, which are record arrays of the modflow header information
    (kstp, kper, pertim, totim, text, nrow, ncol, ilay), and long ints
    pointing to the 1st byte of data for the corresponding data arrays.
    This class overrides methods in the parent class so that the proper
    sized arrays are created: for unstructured grids, nrow and ncol are
    the starting and ending node numbers for layer, ilay.

    When the get_data method is called for this class, a list of
    one-dimensional arrays will be returned, where each array is the head
    array for a layer. If the heads for a layer were not saved, then
    None will be returned for that layer.

    Examples
    --------

    >>> import flopy.utils.binaryfile as bf
    >>> hdobj = bf.HeadUFile('model.hds')
    >>> hdobj.headers
    >>> usgheads = hdobj.get_data(kstpkper=(0, 49))

    """

    def __init__(
        self,
        filename: Union[str, os.PathLike],
        text="headu",
        precision="auto",
        verbose=False,
        **kwargs,
    ):
        """
        Class constructor
        """
        self.text = text.encode()
        if precision == "auto":
            precision = get_headfile_precision(filename)
            if precision == "unknown":
                s = f"Error. Precision could not be determined for {filename}"
                print(s)
                raise Exception()
        self.header_dtype = BinaryHeader.set_dtype(bintype="Head", precision=precision)
        super().__init__(filename, precision, verbose, **kwargs)

    def _get_data_array(self, totim=0.0):
        """
        Get a list of 1D arrays for the
        specified kstp and kper value or totim value.

        """

        if totim >= 0.0:
            keyindices = np.asarray(self.recordarray["totim"] == totim).nonzero()[0]
            if len(keyindices) == 0:
                msg = f"totim value ({totim}) not found in file..."
                raise Exception(msg)
        else:
            raise Exception("Data not found...")

        # fill a list of 1d arrays with heads from binary file
        data = self.nlay * [None]
        for idx in keyindices:
            ipos = self.iposarray[idx]
            ilay = self.recordarray["ilay"][idx]
            nstrt = self.recordarray["ncol"][idx]
            nend = self.recordarray["nrow"][idx]
            npl = nend - nstrt + 1
            if self.verbose:
                print(f"Byte position in file: {ipos} for layer {ilay}")
            self.file.seek(ipos, 0)
            data[ilay - 1] = binaryread(self.file, self.realtype, shape=(npl,))
        return data

    def get_databytes(self, header):
        """

        Parameters
        ----------
        header : datafile.Header
            header object

        Returns
        -------
         databytes : int
            size of the data array, in bytes, following the header

        """
        # unstructured head files contain node starting and ending indices
        # for each layer
        nstrt = np.int64(header["ncol"])
        nend = np.int64(header["nrow"])
        npl = nend - nstrt + 1
        return npl * np.int64(self.realtype(1).nbytes)

    def get_ts(self, idx):
        """
        Get a time series from the binary HeadUFile

        Parameters
        ----------
        idx : int or list of ints
            idx can be nodenumber or it can be a list in the form
            [nodenumber, nodenumber, ...].  The nodenumber,
            values must be zero based.

        Returns
        -------
        out : numpy array
            Array has size (ntimes, ncells + 1).  The first column in the
            data array will contain time (totim).

        """
        times = self.get_times()
        data = self.get_data(totim=times[0])
        layers = len(data)
        ncpl = [len(data[l]) for l in range(layers)]
        result = []

        if isinstance(idx, int):
            layer, nn = get_lni(ncpl, [idx])[0]
            for i, time in enumerate(times):
                data = self.get_data(totim=time)
                value = data[layer][nn]
                result.append([time, value])
        elif isinstance(idx, list) and all(isinstance(x, int) for x in idx):
            for i, time in enumerate(times):
                data = self.get_data(totim=time)
                row = [time]
                lni = get_lni(ncpl, idx)
                for layer, nn in lni:
                    value = data[layer][nn]
                    row += [value]
                result.append(row)
        else:
            raise ValueError("idx must be an integer or a list of integers")

        return np.array(result)


class BudgetIndexError(Exception):
    pass


class CellBudgetFile:
    """
    The CellBudgetFile class provides convenient ways to retrieve and
    manipulate budget data from a binary cell budget file. A utility
    method is also provided to reverse the budget records for particle
    tracking simulations in which particles are tracked backwards from
    terminating to release locations (e.g., to compute capture zones).

    Parameters
    ----------
    filename : str or PathLike
        Path of the cell budget file.
    precision : string
        Precision of floating point budget data in the file. Accepted
        values are 'single' or 'double'. Default is 'single'.
    verbose : bool
        Toggle logging output. Default is False.

    Examples
    --------

    >>> import flopy.utils.binaryfile as bf
    >>> cbb = bf.CellBudgetFile('mymodel.cbb')
    >>> cbb.headers
    >>> rec = cbb.get_data(kstpkper=(0,0), text='RIVER LEAKAGE')

    """

    def __init__(
        self,
        filename: Union[str, os.PathLike],
        precision="auto",
        verbose=False,
        **kwargs,
    ):
        self.filename = Path(filename).expanduser().absolute()
        self.precision = precision
        self.verbose = verbose
        self.file = open(self.filename, "rb")
        # Get filesize to ensure this is not an empty file
        self.file.seek(0, 2)
        totalbytes = self.file.tell()
        self.file.seek(0, 0)  # reset to beginning
        assert self.file.tell() == 0
        if totalbytes == 0:
            raise ValueError(f"datafile error: file is empty: {filename}")
        self.nrow = 0
        self.ncol = 0
        self.nlay = 0
        self.nper = 0
        self.times = []
        self.kstpkper = []
        self.recordarray = []
        self.iposheader = []
        self.iposarray = []
        self.textlist = []
        self.imethlist = []
        self.paknamlist_from = []
        self.paknamlist_to = []
        self.compact = True  # compact budget file flag
        self.dis = None
        self.modelgrid = None
        if "model" in kwargs.keys():
            self.model = kwargs.pop("model")
            self.modelgrid = self.model.modelgrid
            self.dis = self.model.dis
        if "dis" in kwargs.keys():
            self.dis = kwargs.pop("dis")
            self.modelgrid = self.dis.parent.modelgrid
        if "tdis" in kwargs.keys():
            self.tdis = kwargs.pop("tdis")
        if "modelgrid" in kwargs.keys():
            self.modelgrid = kwargs.pop("modelgrid")
        if len(kwargs.keys()) > 0:
            args = ",".join(kwargs.keys())
            raise Exception(f"LayerFile error: unrecognized kwargs: {args}")

        if precision == "auto":
            success = self._set_precision("single")
            if not success:
                success = self._set_precision("double")
            if not success:
                s = "Budget precision could not be auto determined"
                raise BudgetIndexError(s)
        elif precision == "single":
            success = self._set_precision(precision)
        elif precision == "double":
            success = self._set_precision(precision)
        else:
            raise Exception(f"Unknown precision specified: {precision}")

        # set shape for full3D option
        if self.modelgrid is None:
            self.shape = (self.nlay, self.nrow, self.ncol)
            self.nnodes = self.nlay * self.nrow * self.ncol
        else:
            self.shape = self.modelgrid.shape
            self.nnodes = self.modelgrid.nnodes

        if not success:
            raise Exception(
                f"Budget file could not be read using {precision} precision"
            )

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __len__(self):
        """
        Return the number of records (headers) in the file.
        """
        return len(self.recordarray)

    @property
    def nrecords(self):
        """
        Return the number of records (headers) in the file.

        .. deprecated:: 3.8.0
           Use :meth:`len` instead.
        """
        warnings.warn(
            "obj.nrecords is deprecated; use len(obj) instead.",
            DeprecationWarning,
        )
        return len(self)

    def __reset(self):
        """
        Reset indexing lists when determining precision
        """
        self.file.seek(0, 0)
        self.times = []
        self.kstpkper = []
        self.recordarray = []
        self.iposheader = []
        self.iposarray = []
        self.textlist = []
        self.imethlist = []
        self.paknamlist_from = []
        self.paknamlist_to = []

    def _set_precision(self, precision="single"):
        """
        Method to set the budget precision from a CBC file. Enables
        Auto precision code to work

        Parameters
        ----------
        precision : str
            budget file precision (accepts 'single' or 'double')
        """
        success = True
        h1dt = [
            ("kstp", "i4"),
            ("kper", "i4"),
            ("text", "S16"),
            ("ncol", "i4"),
            ("nrow", "i4"),
            ("nlay", "i4"),
        ]
        if precision == "single":
            self.realtype = np.float32
            ffmt = "f4"
        else:
            self.realtype = np.float64
            ffmt = "f8"

        h2dt0 = [
            ("imeth", "i4"),
            ("delt", ffmt),
            ("pertim", ffmt),
            ("totim", ffmt),
        ]
        h2dt = [
            ("imeth", "i4"),
            ("delt", ffmt),
            ("pertim", ffmt),
            ("totim", ffmt),
            ("modelnam", "S16"),
            ("paknam", "S16"),
            ("modelnam2", "S16"),
            ("paknam2", "S16"),
        ]
        self.header1_dtype = np.dtype(h1dt)
        self.header2_dtype0 = np.dtype(h2dt0)
        self.header2_dtype = np.dtype(h2dt)
        hdt = h1dt + h2dt
        self.header_dtype = np.dtype(hdt)

        try:
            self._build_index()
        except (BudgetIndexError, EOFError) as e:
            success = False
            self.__reset()

        return success

    def _totim_from_kstpkper(self, kstpkper):
        if self.dis is None:
            return -1.0
        kstp, kper = kstpkper
        perlen = self.dis.perlen.array
        nstp = self.dis.nstp.array[kper]
        tsmult = self.dis.tsmult.array[kper]
        kper_len = np.sum(perlen[:kper])
        this_perlen = perlen[kper]
        if tsmult == 1:
            dt1 = this_perlen / float(nstp)
        else:
            dt1 = this_perlen * (tsmult - 1.0) / ((tsmult**nstp) - 1.0)
        kstp_len = [dt1]
        for i in range(kstp + 1):
            kstp_len.append(kstp_len[-1] * tsmult)
        kstp_len = sum(kstp_len[: kstp + 1])
        return kper_len + kstp_len

    def _build_index(self):
        """
        Build the ordered dictionary, which maps the header information
        to the position in the binary file.
        """
        # read first record
        header = self._get_header()
        nrow = header["nrow"]
        ncol = header["ncol"]
        text = header["text"].decode("ascii").strip()
        if nrow < 0 or ncol < 0:
            raise Exception("negative nrow, ncol")
        if text != "FLOW-JA-FACE":
            self.nrow = nrow
            self.ncol = ncol
            self.nlay = np.abs(header["nlay"])
        self.file.seek(0, 2)
        self.totalbytes = self.file.tell()
        self.file.seek(0, 0)
        self.recorddict = {}
        # read the remaining records
        ipos = 0
        while ipos < self.totalbytes:
            self.iposheader.append(ipos)
            header = self._get_header()
            totim = header["totim"]
            # if old-style (non-compact) file,
            # compute totim from kstp and kper
            if not self.compact:
                totim = self._totim_from_kstpkper(
                    (header["kstp"] - 1, header["kper"] - 1)
                )
                header["totim"] = totim
            if totim >= 0 and totim not in self.times:
                self.times.append(totim)
            kstpkper = (header["kstp"], header["kper"])
            if kstpkper not in self.kstpkper:
                self.kstpkper.append(kstpkper)
            if header["text"] not in self.textlist:
                # check the precision of the file using text records
                tlist = [header["text"], header["modelnam"]]
                for text in tlist:
                    if len(text) == 0:
                        continue
                    charbytes = list(text)
                    if min(charbytes) < 32 or max(charbytes) > 126:
                        # not in conventional ASCII range
                        raise BudgetIndexError("Improper precision")
                self.textlist.append(header["text"])
                self.imethlist.append(header["imeth"])
            if header["paknam"] not in self.paknamlist_from:
                self.paknamlist_from.append(header["paknam"])
            if header["paknam2"] not in self.paknamlist_to:
                self.paknamlist_to.append(header["paknam2"])
            ipos = self.file.tell()

            if self.verbose:
                for itxt in [
                    "kstp",
                    "kper",
                    "text",
                    "ncol",
                    "nrow",
                    "nlay",
                    "imeth",
                    "delt",
                    "pertim",
                    "totim",
                    "modelnam",
                    "paknam",
                    "modelnam2",
                    "paknam2",
                ]:
                    s = header[itxt]
                    print(f"{itxt}: {s}")
                print("file position: ", ipos)
                if header["imeth"].item() not in {5, 6, 7}:
                    print("")

            # set the nrow, ncol, and nlay if they have not been set
            if self.nrow == 0:
                text = header["text"].decode("ascii").strip()
                if text != "FLOW-JA-FACE":
                    self.nrow = header["nrow"]
                    self.ncol = header["ncol"]
                    self.nlay = np.abs(header["nlay"])

            # store record and byte position mapping
            self.recorddict[tuple(header)] = (
                ipos  # store the position right after header2
            )
            self.recordarray.append(header)
            self.iposarray.append(ipos)  # store the position right after header2

            # skip over the data to the next record and set ipos
            self._skip_record(header)
            ipos = self.file.tell()

        # convert to numpy arrays
        self.recordarray = np.array(self.recordarray, dtype=self.header_dtype)
        self.iposheader = np.array(self.iposheader, dtype=np.int64)
        self.iposarray = np.array(self.iposarray, dtype=np.int64)
        self.nper = self.recordarray["kper"].max()

        # provide headers as a pandas frame
        self.headers = pd.DataFrame(self.recordarray, index=self.iposarray)
        # remove irrelevant columns
        cols = self.headers.columns.to_list()
        unique_imeth = self.headers["imeth"].unique()
        if unique_imeth.max() == 0:
            drop_cols = cols[cols.index("imeth") :]
        elif 6 not in unique_imeth:
            drop_cols = cols[cols.index("modelnam") :]
        else:
            drop_cols = []
        if drop_cols:
            self.headers.drop(columns=drop_cols, inplace=True)
        for name in self.headers.columns:
            dtype = self.header_dtype[name]
            if np.issubdtype(dtype, bytes):  # convert to str
                self.headers[name] = (
                    self.headers[name].str.decode("ascii", "strict").str.strip()
                )

    def _skip_record(self, header):
        """
        Skip over this record, not counting header and header2.

        """
        nlay = abs(header["nlay"])
        nrow = header["nrow"]
        ncol = header["ncol"]
        imeth = header["imeth"]
        realtype_nbytes = self.realtype(1).nbytes
        if imeth == 0:
            nbytes = nrow * ncol * nlay * realtype_nbytes
        elif imeth == 1:
            nbytes = nrow * ncol * nlay * realtype_nbytes
        elif imeth == 2:
            nlist = binaryread(self.file, np.int32)[0]
            nbytes = nlist * (4 + realtype_nbytes)
        elif imeth == 3:
            nbytes = nrow * ncol * realtype_nbytes + (nrow * ncol * 4)
        elif imeth == 4:
            nbytes = nrow * ncol * realtype_nbytes
        elif imeth == 5:
            nauxp1 = binaryread(self.file, np.int32)[0]
            naux = nauxp1 - 1
            naux_nbytes = naux * 16
            if naux_nbytes:
                check = self.file.seek(naux_nbytes, 1)
                if check < naux_nbytes:
                    raise EOFError
            nlist = binaryread(self.file, np.int32)[0]
            if self.verbose:
                print("naux: ", naux)
                print("nlist: ", nlist)
                print("")
            nbytes = nlist * (4 + realtype_nbytes + naux * realtype_nbytes)
        elif imeth == 6:
            # read rest of list data
            nauxp1 = binaryread(self.file, np.int32)[0]
            naux = nauxp1 - 1
            naux_nbytes = naux * 16
            if naux_nbytes:
                check = self.file.seek(naux_nbytes, 1)
                if check < naux_nbytes:
                    raise EOFError
            nlist = binaryread(self.file, np.int32)[0]
            if self.verbose:
                print("naux: ", naux)
                print("nlist: ", nlist)
                print("")
            nbytes = nlist * (4 * 2 + realtype_nbytes + naux * realtype_nbytes)
        else:
            raise Exception(f"invalid method code {imeth}")
        if nbytes != 0:
            self.file.seek(nbytes, 1)

    def _get_header(self):
        """
        Read the file header

        """
        header1 = binaryread(self.file, self.header1_dtype, (1,))
        nlay = header1["nlay"]
        self.compact = bool(nlay < 0)
        if self.compact:
            # fill header2 by first reading imeth, delt, pertim and totim
            # and then adding modelnames and paknames if imeth = 6
            temp = binaryread(self.file, self.header2_dtype0, (1,))
            header2 = np.array(
                [(0, 0.0, 0.0, 0.0, "", "", "", "")], dtype=self.header2_dtype
            )
            for name in temp.dtype.names:
                header2[name] = temp[name]
            if header2["imeth"].item() == 6:
                header2["modelnam"] = binaryread(self.file, bytes, charlen=16)
                header2["paknam"] = binaryread(self.file, bytes, charlen=16)
                header2["modelnam2"] = binaryread(self.file, bytes, charlen=16)
                header2["paknam2"] = binaryread(self.file, bytes, charlen=16)
        else:
            header2 = np.array(
                [(0, 0.0, 0.0, 0.0, "", "", "", "")], dtype=self.header2_dtype
            )
        fullheader = join_struct_arrays([header1, header2])
        return fullheader[0]

    def _find_text(self, text):
        """
        Determine if selected record name is in budget file

        """
        # check and make sure that text is in file
        text16 = None
        if text is not None:
            if isinstance(text, bytes):
                ttext = text.decode()
            else:
                ttext = text
            for t in self.textlist:
                if ttext.upper() in t.decode():
                    text16 = t
                    break
            if text16 is None:
                errmsg = "The specified text string is not in the budget file."
                raise Exception(errmsg)
        return text16

    def _find_paknam(self, paknam, to=False):
        """
        Determine if selected record name is in budget file

        """
        # check and make sure that text is in file
        paknam16 = None
        if paknam is not None:
            if isinstance(paknam, bytes):
                tpaknam = paknam.decode()
            else:
                tpaknam = paknam
            for t in self._unique_package_names(to):
                if tpaknam.upper() in t.decode():
                    paknam16 = t
                    break
            if paknam16 is None:
                raise Exception(
                    "The specified package name string is not in the budget file."
                )
        return paknam16

    def list_records(self):
        """
        Print a list of all of the records in the file

        .. deprecated:: 3.8.0
           Use :attr:`headers` instead.
        """
        warnings.warn(
            "list_records() is deprecated; use headers instead.",
            DeprecationWarning,
        )
        for rec in self.recordarray:
            if isinstance(rec, bytes):
                rec = rec.decode()
            print(rec)

    def list_unique_records(self):
        """
        Print a list of unique record names

        .. deprecated:: 3.8.0
           Use `headers[["text", "imeth"]].drop_duplicates()` instead.
        """
        warnings.warn(
            "list_unique_records() is deprecated; use "
            'headers[["text", "imeth"]].drop_duplicates() instead.',
            DeprecationWarning,
        )
        print("RECORD           IMETH")
        print(22 * "-")
        for rec, imeth in zip(self.textlist, self.imethlist):
            if isinstance(rec, bytes):
                rec = rec.decode()
            print(f"{rec.strip():16} {imeth:5d}")

    def list_unique_packages(self, to=False):
        """
        Print a list of unique package names

        .. deprecated:: 3.8.0
           Use `headers.paknam.drop_duplicates()` or
           `headers.paknam2.drop_duplicates()` instead.
        """
        warnings.warn(
            "list_unique_packages() is deprecated; use "
            "headers.paknam.drop_duplicates() or "
            "headers.paknam2.drop_duplicates() instead",
            DeprecationWarning,
        )
        for rec in self._unique_package_names(to):
            if isinstance(rec, bytes):
                rec = rec.decode()
            print(rec)

    def get_unique_record_names(self, decode=False):
        """
        Get a list of unique record names in the file

        Parameters
        ----------
        decode : bool
            Optional boolean used to decode byte strings (default is False).

        Returns
        -------
        names : list of strings
            List of unique text names in the binary file.

        """
        if decode:
            names = []
            for text in self.textlist:
                if isinstance(text, bytes):
                    text = text.decode()
                names.append(text)
        else:
            names = self.textlist
        return names

    def get_unique_package_names(self, decode=False, to=False):
        """
        Get a list of unique package names in the file

        Parameters
        ----------
        decode : bool
            Optional boolean used to decode byte strings (default is False).

        Returns
        -------
        names : list of strings
            List of unique package names in the binary file.

        """

        if decode:
            names = []
            for text in self._unique_package_names(to):
                if isinstance(text, bytes):
                    text = text.decode()
                names.append(text)
        else:
            names = self._unique_package_names(to)
        return names

    def _unique_package_names(self, to=False):
        """
        Get a list of unique package names in the file

        Returns
        -------
        out : list of strings
            List of unique package names in the binary file.

        """
        return self.paknamlist_to if to else self.paknamlist_from

    def get_kstpkper(self):
        """
        Get a list of unique tuples (stress period, time step) in the file.
        Indices are 0-based, use the `kstpkper` attribute for 1-based.

        Returns
        -------
        list of (kstp, kper) tuples
            List of unique combinations of stress period &
            time step indices (0-based) in the binary file
        """
        return [(kstp - 1, kper - 1) for kstp, kper in self.kstpkper]

    def get_indices(self, text=None):
        """
        Get a list of indices for a selected record name

        Parameters
        ----------
        text : str
            The text identifier for the record.  Examples include
            'RIVER LEAKAGE', 'STORAGE', 'FLOW RIGHT FACE', etc.

        Returns
        -------
        out : tuple
            indices of selected record name in budget file.

        """
        # check and make sure that text is in file
        if text is not None:
            text16 = self._find_text(text)
            select_indices = np.asarray(self.recordarray["text"] == text16).nonzero()
            if isinstance(select_indices, tuple):
                select_indices = select_indices[0]
        else:
            select_indices = None
        return select_indices

    def get_position(self, idx, header=False):
        """
        Get the starting position of the data or header for a specified record
        number in the binary budget file.

        Parameters
        ----------
        idx : int
            The zero-based record number.  The first record is record 0.
        header : bool
            If True, the position of the start of the header data is returned.
            If False, the position of the start of the data is returned
            (default is False).

        Returns
        -------
        ipos : int64
            The position of the start of the data in the cell budget file
            or the start of the header.

        """
        if header:
            ipos = self.iposheader[idx]
        else:
            ipos = self.iposarray[idx]
        return ipos

    def get_data(
        self,
        idx=None,
        kstpkper=None,
        totim=None,
        text=None,
        paknam=None,
        paknam2=None,
        full3D=False,
    ) -> Union[list, np.ndarray]:
        """
        Get data from the binary budget file.

        Parameters
        ----------
        idx : int or list
            The zero-based record number.  The first record is record 0.
        kstpkper : tuple of ints
            A tuple containing the time step and stress period (kstp, kper).
            The kstp and kper values are zero based.
        totim : float
            The simulation time.
        text : str
            The text identifier for the record.  Examples include
            'RIVER LEAKAGE', 'STORAGE', 'FLOW RIGHT FACE', etc.
        paknam : str
            The `from` package name for the record.
        paknam2 : str
            The `to` package name for the record.  This argument can be
            useful for MODFLOW 6 budget files if multiple packages of
            the same type are specified.  The paknam2 argument can be
            specified as the package name (not the package type) in
            order to retrieve budget data for a specific named package.
        full3D : boolean
            If true, then return the record as a three dimensional numpy
            array, even for those list-style records written as part of a
            'COMPACT BUDGET' MODFLOW budget file.  (Default is False.)

        Returns
        -------
        recordlist : list of records
            A list of budget objects.  The structure of the returned object
            depends on the structure of the data in the cbb file.

            If full3D is True, then this method will return a numpy masked
            array of size (nlay, nrow, ncol) for those list-style
            'COMPACT BUDGET' records written by MODFLOW.

        See Also
        --------

        Notes
        -----

        Examples
        --------

        """
        # trap for totim error
        if totim is not None:
            if len(self.times) == 0:
                errmsg = """This is an older style budget file that
                         does not have times in it.  Use the MODFLOW
                         compact budget format if you want to work with
                         times.  Or you may access this file using the
                         kstp and kper arguments or the idx argument."""
                raise Exception(errmsg)

        # check and make sure that text is in file
        text16 = None
        if text is not None:
            text16 = self._find_text(text)
        paknam16 = None
        if paknam is not None:
            paknam16 = self._find_paknam(paknam)
        paknam16_2 = None
        if paknam2 is not None:
            paknam16_2 = self._find_paknam(paknam2, to=True)

        # build the selection mask
        select_indices = np.array([True] * len(self.recordarray))
        selected = False
        if idx is not None:
            select_indices[idx] = False
            select_indices = ~select_indices
            selected = True
        if kstpkper is not None:
            kstp1 = kstpkper[0] + 1
            kper1 = kstpkper[1] + 1
            select_indices = select_indices & (self.recordarray["kstp"] == kstp1)
            select_indices = select_indices & (self.recordarray["kper"] == kper1)
            selected = True
        if text16 is not None:
            select_indices = select_indices & (self.recordarray["text"] == text16)
            selected = True
        if paknam16 is not None:
            select_indices = select_indices & (self.recordarray["paknam"] == paknam16)
            selected = True
        if paknam16_2 is not None:
            select_indices = select_indices & (
                self.recordarray["paknam2"] == paknam16_2
            )
            selected = True
        if totim is not None:
            select_indices = select_indices & np.isclose(
                self.recordarray["totim"], totim
            )
            selected = True

        if not selected:
            raise TypeError(
                "get_data() missing 1 required argument: 'kstpkper', 'totim', "
                "'idx', or 'text'"
            )
        return [
            self.get_record(idx, full3D=full3D)
            for idx, t in enumerate(select_indices)
            if t
        ]

    def get_ts(self, idx, text=None, times=None):
        """
        Get a time series from the binary budget file.

        Parameters
        ----------
        idx : tuple of ints, or a list of a tuple of ints
            idx can be (layer, row, column) or it can be a list in the form
            [(layer, row, column), (layer, row, column), ...].  The layer,
            row, and column values must be zero based.
        text : str
            The text identifier for the record.  Examples include
            'RIVER LEAKAGE', 'STORAGE', 'FLOW RIGHT FACE', etc.
        times : iterable of floats
            List of times to from which to get time series.

        Returns
        -------
        out : numpy array
            Array has size (ntimes, ncells + 1).  The first column in the
            data array will contain time (totim).

        See Also
        --------

        Notes
        -----

        The layer, row, and column values must be zero-based, and must be
        within the following ranges: 0 <= k < nlay; 0 <= i < nrow; 0 <= j < ncol

        Examples
        --------

        """
        # issue exception if text not provided
        if text is None:
            raise Exception(
                "text keyword must be provided to CellBudgetFile get_ts() method."
            )

        kijlist = self._build_kijlist(idx)
        nstation = self._get_nstation(idx, kijlist)

        # Initialize result array and put times in first column
        result = self._init_result(nstation)

        timesint = self.get_times()
        kstpkper = self.get_kstpkper()
        nsteps = len(kstpkper)
        if len(timesint) < 1:
            if times is None:
                timesint = [x + 1 for x in range(nsteps)]
            else:
                if isinstance(times, np.ndarray):
                    times = times.tolist()
                if len(times) != nsteps:
                    raise ValueError(
                        f"number of times provided ({len(times)}) must equal "
                        f"number of time steps in cell budget file ({nsteps})"
                    )
                timesint = times
        for idx, t in enumerate(timesint):
            result[idx, 0] = t

        for itim, k in enumerate(kstpkper):
            try:
                v = self.get_data(kstpkper=k, text=text, full3D=True)
                # skip missing data - required for storage
                if len(v) > 0:
                    v = v[0]
                    istat = 1
                    for k, i, j in kijlist:
                        result[itim, istat] = v[k, i, j].copy()
                        istat += 1
            except ValueError:
                v = self.get_data(kstpkper=k, text=text)
                # skip missing data - required for storage
                if len(v) > 0:
                    if self.modelgrid is None:
                        s = (
                            "A modelgrid instance must be provided during "
                            "instantiation to get IMETH=6 timeseries data"
                        )
                        raise AssertionError(s)

                    if self.modelgrid.grid_type == "structured":
                        ndx = [
                            lrc[0] * (self.modelgrid.nrow * self.modelgrid.ncol)
                            + lrc[1] * self.modelgrid.ncol
                            + (lrc[2] + 1)
                            for lrc in kijlist
                        ]
                    else:
                        ndx = [
                            lrc[0] * self.modelgrid.ncpl + (lrc[-1] + 1)
                            for lrc in kijlist
                        ]

                    for vv in v:
                        field = vv.dtype.names[2]
                        dix = np.asarray(np.isin(vv["node"], ndx)).nonzero()[0]
                        if len(dix) > 0:
                            result[itim, 1:] = vv[field][dix]

        return result

    def _build_kijlist(self, idx):
        if isinstance(idx, list):
            kijlist = idx
        elif isinstance(idx, tuple):
            kijlist = [idx]
        else:
            raise Exception("Could not build kijlist from ", idx)

        # Check to make sure that k, i, j are within range, otherwise
        # the seek approach won't work.  Can't use k = -1, for example.
        for k, i, j in kijlist:
            fail = False
            if k < 0 or k > self.nlay - 1:
                fail = True
            if i < 0 or i > self.nrow - 1:
                fail = True
            if j < 0 or j > self.ncol - 1:
                fail = True
            if fail:
                raise Exception(
                    "Invalid cell index. Cell {} not within model grid: {}".format(
                        (k, i, j), (self.nlay, self.nrow, self.ncol)
                    )
                )
        return kijlist

    def _get_nstation(self, idx, kijlist):
        if isinstance(idx, list):
            return len(kijlist)
        elif isinstance(idx, tuple):
            return 1

    def _init_result(self, nstation):
        # Initialize result array and put times in first column
        result = np.empty((len(self.kstpkper), nstation + 1), dtype=self.realtype)
        result[:, :] = np.nan
        if len(self.times) == result.shape[0]:
            result[:, 0] = np.array(self.times)
        return result

    def get_record(self, idx, full3D=False):
        """
        Get a single data record from the budget file.

        Parameters
        ----------
        idx : int
            The zero-based record number.  The first record is record 0.
        full3D : boolean
            If true, then return the record as a three dimensional numpy
            array, even for those list-style records written as part of a
            'COMPACT BUDGET' MODFLOW budget file.  (Default is False.)

        Returns
        -------
        record : a single data record
            The structure of the returned object depends on the structure of
            the data in the cbb file. Compact list data are returned as

            If full3D is True, then this method will return a numpy masked
            array of size (nlay, nrow, ncol) for those list-style
            'COMPACT BUDGET' records written by MODFLOW.

        See Also
        --------

        Notes
        -----

        Examples
        --------

        """
        # idx must be an ndarray, so if it comes in as an integer then convert
        if np.isscalar(idx):
            idx = np.array([idx])

        header = self.recordarray[idx]
        ipos = self.iposarray[idx].item()
        self.file.seek(ipos, 0)
        imeth = header["imeth"][0]

        t = header["text"][0].decode("ascii")
        s = f"Returning {t.strip()} as "

        nlay = abs(header["nlay"][0])
        nrow = header["nrow"][0]
        ncol = header["ncol"][0]

        # default method
        if imeth == 0:
            if self.verbose:
                s += f"an array of shape {(nlay, nrow, ncol)}"
                print(s)
            return binaryread(self.file, self.realtype(1), shape=(nlay, nrow, ncol))
        # imeth 1
        elif imeth == 1:
            if self.verbose:
                s += f"an array of shape {(nlay, nrow, ncol)}"
                print(s)
            return binaryread(self.file, self.realtype(1), shape=(nlay, nrow, ncol))

        # imeth 2
        elif imeth == 2:
            nlist = binaryread(self.file, np.int32)[0]
            dtype = np.dtype([("node", np.int32), ("q", self.realtype)])
            if self.verbose:
                if full3D:
                    s += f"a numpy masked array of size ({nlay}, {nrow}, {ncol})"
                else:
                    s += f"a numpy recarray of size ({nlist}, 2)"
                print(s)
            data = binaryread(self.file, dtype, shape=(nlist,))
            if full3D:
                return self.__create3D(data)
            else:
                return data.view(np.recarray)

        # imeth 3
        elif imeth == 3:
            ilayer = binaryread(self.file, np.int32, shape=(nrow, ncol))
            data = binaryread(self.file, self.realtype(1), shape=(nrow, ncol))
            if self.verbose:
                if full3D:
                    s += f"a numpy masked array of size ({nlay}, {nrow}, {ncol})"
                else:
                    s += (
                        "a list of two 2D numpy arrays. The first is an "
                        f"integer layer array of shape ({nrow}, {ncol}). The "
                        f"second is real data array of shape ({nrow}, {ncol})"
                    )
                print(s)
            if full3D:
                out = np.ma.zeros(self.nnodes, dtype=np.float32)
                out.mask = True
                vertical_layer = ilayer.flatten() - 1
                # create the 2D cell index and then move it to
                # the correct vertical location
                idx = np.arange(0, vertical_layer.shape[0])
                idx += vertical_layer * nrow * ncol
                out[idx] = data.flatten()
                return out.reshape(self.shape)
            else:
                return [ilayer, data]

        # imeth 4
        elif imeth == 4:
            if self.verbose:
                s += f"a 2d numpy array of size ({nrow}, {ncol})"
                print(s)
            return binaryread(self.file, self.realtype(1), shape=(nrow, ncol))

        # imeth 5
        elif imeth == 5:
            nauxp1 = binaryread(self.file, np.int32)[0]
            naux = nauxp1 - 1
            l = [("node", np.int32), ("q", self.realtype)]
            for i in range(naux):
                auxname = binaryread(self.file, bytes, charlen=16)
                l.append((auxname.decode("ascii").strip(), self.realtype))
            dtype = np.dtype(l)
            nlist = binaryread(self.file, np.int32)[0]
            data = binaryread(self.file, dtype, shape=(nlist,))
            if full3D:
                if self.verbose:
                    s += f"a list array of shape ({nlay}, {nrow}, {ncol})"
                    print(s)
                return self.__create3D(data)
            else:
                if self.verbose:
                    s += f"a numpy recarray of size ({nlist}, {2 + naux})"
                    print(s)
                return data.view(np.recarray)

        # imeth 6
        elif imeth == 6:
            # read rest of list data
            nauxp1 = binaryread(self.file, np.int32)[0]
            naux = nauxp1 - 1
            l = [("node", np.int32), ("node2", np.int32), ("q", self.realtype)]
            for i in range(naux):
                auxname = binaryread(self.file, bytes, charlen=16)
                l.append((auxname.decode("ascii").strip(), self.realtype))
            dtype = np.dtype(l)
            nlist = binaryread(self.file, np.int32)[0]
            data = binaryread(self.file, dtype, shape=(nlist,))
            if self.verbose:
                if full3D:
                    s += f"a list array of shape ({nlay}, {nrow}, {ncol})"
                else:
                    s += f"a numpy recarray of size ({nlist}, 2)"
                print(s)
            if full3D:
                data = self.__create3D(data)
                if self.modelgrid is not None:
                    return np.reshape(data, self.shape)
                else:
                    return data
            else:
                return data.view(np.recarray)
        else:
            raise ValueError(f"invalid imeth value - {imeth}")

        # should not reach this point
        return

    def __create3D(self, data):
        """
        Convert a dictionary of {node: q, ...} into a numpy masked array.
        Used to create full grid arrays when the full3D keyword is set
        to True in get_data.

        Parameters
        ----------
        data : dictionary
            Dictionary with node keywords and flows (q) items.

        Returns
        -------
        out : numpy masked array
            List contains unique simulation times (totim) in binary file.

        """
        out = np.ma.zeros(self.nnodes, dtype=np.float32)
        out.mask = True
        for [node, q] in zip(data["node"], data["q"]):
            idx = node - 1
            out.data[idx] += q
            out.mask[idx] = False
        return np.ma.reshape(out, self.shape)

    def get_times(self):
        """
        Get a list of unique times in the file

        Returns
        -------
        out : list of floats
            List contains unique simulation times (totim) in binary file.

        """
        return self.times

    def get_nrecords(self):
        """
        Return the number of records in the file

        Returns
        -------
        int
            Number of records in the file.

        .. deprecated:: 3.8.0
           Use :meth:`len` instead.
        """
        warnings.warn(
            "get_nrecords is deprecated; use len(obj) instead.",
            DeprecationWarning,
        )
        return len(self)

    def get_residual(self, totim, scaled=False):
        """
        Return an array the size of the model grid containing the flow residual
        calculated from the budget terms.  Residual will not be correct unless
        all flow terms are written to the budget file.

        Parameters
        ----------
        totim : float
            Simulation time for which to calculate the residual.  This value
            must be precise, so it is best to get it from the get_times
            method.

        scaled : bool
            If True, then divide the residual by the total cell inflow

        Returns
        -------
        residual : np.ndarray
            The flow residual for the cell of shape (nlay, nrow, ncol)

        """

        nlay = self.nlay
        nrow = self.nrow
        ncol = self.ncol
        residual = np.zeros((nlay, nrow, ncol), dtype=float)
        if scaled:
            inflow = np.zeros((nlay, nrow, ncol), dtype=float)
        select_indices = np.asarray(self.recordarray["totim"] == totim).nonzero()[0]

        for i in select_indices:
            text = self.recordarray[i]["text"].decode()
            if self.verbose:
                print(f"processing {text}")
            flow = self.get_record(idx=i, full3D=True)
            if ncol > 1 and "RIGHT FACE" in text:
                residual -= flow[:, :, :]
                residual[:, :, 1:] += flow[:, :, :-1]
                if scaled:
                    idx = np.asarray(flow < 0.0).nonzero()
                    inflow[idx] -= flow[idx]
                    idx = np.asarray(flow > 0.0).nonzero()
                    l, r, c = idx
                    idx = (l, r, c + 1)
                    inflow[idx] += flow[idx]
            elif nrow > 1 and "FRONT FACE" in text:
                residual -= flow[:, :, :]
                residual[:, 1:, :] += flow[:, :-1, :]
                if scaled:
                    idx = np.asarray(flow < 0.0).nonzero()
                    inflow[idx] -= flow[idx]
                    idx = np.asarray(flow > 0.0).nonzero()
                    l, r, c = idx
                    idx = (l, r + 1, c)
                    inflow[idx] += flow[idx]
            elif nlay > 1 and "LOWER FACE" in text:
                residual -= flow[:, :, :]
                residual[1:, :, :] += flow[:-1, :, :]
                if scaled:
                    idx = np.asarray(flow < 0.0).nonzero()
                    inflow[idx] -= flow[idx]
                    idx = np.asarray(flow > 0.0).nonzero()
                    l, r, c = idx
                    idx = (l + 1, r, c)
                    inflow[idx] += flow[idx]
            else:
                residual += flow
                if scaled:
                    idx = np.asarray(flow > 0.0).nonzero()
                    inflow[idx] += flow[idx]

        if scaled:
            residual_scaled = np.zeros((nlay, nrow, ncol), dtype=float)
            idx = inflow > 0.0
            residual_scaled[idx] = residual[idx] / inflow[idx]
            return residual_scaled

        return residual

    def close(self):
        """
        Close the file handle
        """
        self.file.close()

    def reverse(self, filename: Optional[os.PathLike] = None):
        """
        Reverse the time order and signs of the currently loaded binary cell budget
        file. If a file name is not provided or if the provided name is the same as
        the existing filename, the file will be overwritten and reloaded.

        Notes
        -----
        While `HeadFile.reverse()` reverses only the temporal order of head data,
        this method must reverse not only the order but also the sign (direction)
        of the model's intercell flows.

        filename : str or PathLike, optional
            Path of the reversed binary cell budget file.
        """

        filename = (
            Path(filename).expanduser().absolute()
            if filename is not None
            else self.filename
        )

        # header array formats
        dt1 = np.dtype(
            [
                ("kstp", np.int32),
                ("kper", np.int32),
                ("text", "S16"),
                ("ndim1", np.int32),
                ("ndim2", np.int32),
                ("ndim3", np.int32),
                ("imeth", np.int32),
                ("delt", np.float64),
                ("pertim", np.float64),
                ("totim", np.float64),
            ]
        )
        dt2 = np.dtype(
            [
                ("text1id1", "S16"),
                ("text1id2", "S16"),
                ("text2id1", "S16"),
                ("text2id2", "S16"),
            ]
        )

        nrecords = len(self)
        target = filename

        time = ModelTime.from_headers(self.recordarray)
        time._set_totim_dict()
        trev = time.reverse()
        trev._set_totim_dict()
        nper = time.nper
        seen = set()

        def reverse_header(header):
            """Reverse period, step and time fields in the record header"""

            nonlocal seen
            kper = header["kper"] - 1
            kstp = header["kstp"] - 1
            header = header.copy()
            header["kper"] = nper - kper
            header["kstp"] = time.nstp[kper] - kstp
            kper = header["kper"] - 1
            kstp = header["kstp"] - 1
            seen.add((kper, kstp))
            header["pertim"] = trev._pertim_dict[(kper, kstp)]
            header["totim"] = trev._totim_dict[(kper, kstp)]
            return header

        # if rewriting the same file, write
        # temp file then copy it into place
        inplace = filename == self.filename
        if inplace:
            temp_dir_path = Path(tempfile.gettempdir())
            temp_file_path = temp_dir_path / filename.name
            target = temp_file_path

        with open(target, "wb") as f:
            # loop over budget file records in reverse order
            for idx in range(nrecords - 1, -1, -1):
                # load header array
                header = self.recordarray[idx]
                header = reverse_header(header)

                # Write main header information to backward budget file
                h = header[
                    [
                        "kstp",
                        "kper",
                        "text",
                        "ncol",
                        "nrow",
                        "nlay",
                        "imeth",
                        "delt",
                        "pertim",
                        "totim",
                    ]
                ]
                # Note: much of the code below is based on binary_file_writer.py
                h = np.array(h, dtype=dt1)
                h.tofile(f)
                if header["imeth"] == 6:
                    # Write additional header information to the backward budget file
                    h = header[["modelnam", "paknam", "modelnam2", "paknam2"]]
                    h = np.array(h, dtype=dt2)
                    h.tofile(f)
                    # Load data
                    data = self.get_data(idx)[0]
                    data = np.array(data)
                    # Negate flows
                    data["q"] = -data["q"]
                    # Write ndat (number of floating point columns)
                    colnames = data.dtype.names
                    ndat = len(colnames) - 2
                    dt = np.dtype([("ndat", np.int32)])
                    h = np.array([(ndat,)], dtype=dt)
                    h.tofile(f)
                    # Write auxiliary column names
                    naux = ndat - 1
                    if naux > 0:
                        auxtxt = ["{:16}".format(colname) for colname in colnames[3:]]
                        auxtxt = tuple(auxtxt)
                        dt = np.dtype([(colname, "S16") for colname in colnames[3:]])
                        h = np.array(auxtxt, dtype=dt)
                        h.tofile(f)
                    # Write nlist
                    nlist = data.shape[0]
                    dt = np.dtype([("nlist", np.int32)])
                    h = np.array([(nlist,)], dtype=dt)
                    h.tofile(f)
                elif header["imeth"] == 1:
                    # Load data
                    data = self.get_data(idx)[0]
                    data = np.array(data, dtype=np.float64)
                    # Negate flows
                    data = -data
                else:
                    raise ValueError("not expecting imeth " + header["imeth"])
                # Write data
                data.tofile(f)

        # if we rewrote the original file, reinitialize
        if inplace:
            move(target, filename)
            self.__init__(filename, self.precision, self.verbose)
