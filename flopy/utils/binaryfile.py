"""
Module to read MODFLOW binary output files.  The module contains four
important classes that can be accessed by the user.

*  HeadFile (Binary head file.  Can also be used for drawdown)
*  HeadUFile (Binary MODFLOW-USG unstructured head file)
*  UcnFile (Binary concentration file from MT3DMS)
*  CellBudgetFile (Binary cell-by-cell flow file)

"""
from __future__ import print_function
import numpy as np
import warnings
from collections import OrderedDict
from ..utils.datafile import Header, LayerFile


class BinaryHeader(Header):
    """
    The binary_header class is a class to create headers for MODFLOW
    binary files.

    Parameters
    ----------
        bintype : str
            is the type of file being opened (head and ucn file currently
            supported)
        precision : str
            is the precision of the floating point data in the file

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
                    msg = "{0} key not available in {1} header "
                    "dtype".format(k, self.header_type)
                    print(msg)
        for k in fkey:
            if k in kwargs.keys():
                try:
                    self.header[0][k] = float(kwargs[k])
                except:
                    msg = "{} key not available ".format(
                        k
                    ) + "in {} header dtype".format(self.header_type)
                    print(msg)
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
                    text = "{:<16}".format(ttext)
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

    """
    import struct
    import numpy as np

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
        nval = np.core.fromnumeric.prod(shape)
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
    Uses numpy to read from binary file.  This was found to be faster than the
        struct approach and is used as the default.

    """

    # read a string variable of length charlen
    if vartype == str:
        result = file.read(charlen * 1)
    else:
        # find the number of values
        nval = np.prod(shape)
        result = np.fromfile(file, vartype, nval)
        if nval == 1:
            result = result  # [0]
        else:
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


def get_headfile_precision(filename):
    """
    Determine precision of a MODFLOW head file.

    Parameters
    ----------
    filename : str
    Name of binary MODFLOW file to determine precision.

    Returns
    -------
    result : str
    Result will be unknown, single, or double

    """

    # Set default result if neither single or double works
    result = "unknown"

    # Create string containing set of ascii characters
    asciiset = " "
    for i in range(33, 127):
        asciiset += chr(i)

    # Open file, and check filesize to ensure this is not an empty file
    f = open(filename, "rb")
    f.seek(0, 2)
    totalbytes = f.tell()
    f.seek(0, 0)  # reset to beginning
    assert f.tell() == 0
    if totalbytes == 0:
        raise IOError("datafile error: file is empty: " + str(filename))

    # first try single
    vartype = [
        ("kstp", "<i4"),
        ("kper", "<i4"),
        ("pertim", "<f4"),
        ("totim", "<f4"),
        ("text", "S16"),
    ]
    hdr = binaryread(f, vartype)
    text = hdr[0][4]
    try:
        text = text.decode()
        for t in text:
            if t.upper() not in asciiset:
                raise Exception()
        result = "single"
        success = True
    except:
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
        text = hdr[0][4]
        try:
            text = text.decode()
            for t in text:
                if t.upper() not in asciiset:
                    raise Exception()
            result = "double"
        except:
            f.close()
            raise IOError(
                "Could not determine the precision of "
                "the headfile {}".format(filename)
            )

    # close and return result
    f.close()
    return result


class BinaryLayerFile(LayerFile):
    """
    The BinaryLayerFile class is the super class from which specific derived
    classes are formed.  This class should not be instantiated directly

    """

    def __init__(self, filename, precision, verbose, kwargs):
        super().__init__(filename, precision, verbose, kwargs)
        return

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

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
        if self.nrow > 1 and self.nrow * self.ncol > 10000000:
            s = "Possible error. ncol ({}) * nrow ({}) > 10,000,000 "
            s = s.format(self.ncol, self.nrow)
            warnings.warn(s)
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
                kstpkper = (header["kstp"], header["kper"])
                self.kstpkper.append(kstpkper)
            else:
                totim = header["totim"]
                if totim != self.times[-1]:
                    self.times.append(totim)
                    kstpkper = (header["kstp"], header["kper"])
                    self.kstpkper.append(kstpkper)
            ipos = self.file.tell()
            self.iposarray.append(ipos)
            databytes = self.get_databytes(header)
            self.file.seek(databytes, 1)
            ipos = self.file.tell()

        # self.recordarray contains a recordarray of all the headers.
        self.recordarray = np.array(self.recordarray, dtype=self.header_dtype)
        self.iposarray = np.array(self.iposarray)
        self.nlay = np.max(self.recordarray["ilay"])
        return

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
        ----------
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
                ilay = (
                    header["ilay"] - 1
                )  # change ilay from header to zero-based
                if ilay != k:
                    continue
                ipos = int(self.iposarray[irec])

                # Calculate offset necessary to reach intended cell
                self.file.seek(ipos + int(ioffset), 0)

                # Find the time index and then put value into result in the
                # correct location.
                itim = np.where(result[:, 0] == header["totim"])[0]
                result[itim, istat] = binaryread(self.file, self.realtype)
            istat += 1
        return result


class HeadFile(BinaryLayerFile):
    """
    HeadFile Class.

    Parameters
    ----------
    filename : string
        Name of the concentration file
    text : string
        Name of the text string in the head file.  Default is 'head'
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
    The HeadFile class provides simple ways to retrieve 2d and 3d
    head arrays from a MODFLOW binary head file and time series
    arrays for one or more cells.

    The BinaryLayerFile class is built on a record array consisting of
    headers, which are record arrays of the modflow header information
    (kstp, kper, pertim, totim, text, nrow, ncol, ilay)
    and long integers, which are pointers to first bytes of data for
    the corresponding data array.

    Examples
    --------

    >>> import flopy.utils.binaryfile as bf
    >>> hdobj = bf.HeadFile('model.hds', precision='single')
    >>> hdobj.list_records()
    >>> rec = hdobj.get_data(kstpkper=(1, 50))

    >>> ddnobj = bf.HeadFile('model.ddn', text='drawdown', precision='single')
    >>> ddnobj.list_records()
    >>> rec = ddnobj.get_data(totim=100.)


    """

    def __init__(
        self, filename, text="head", precision="auto", verbose=False, **kwargs
    ):
        self.text = text.encode()
        if precision == "auto":
            precision = get_headfile_precision(filename)
            if precision == "unknown":
                s = "Error. Precision could not be determined for {}".format(
                    filename
                )
                print(s)
                raise Exception()
        self.header_dtype = BinaryHeader.set_dtype(
            bintype="Head", precision=precision
        )
        super().__init__(filename, precision, verbose, kwargs)
        return


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
    >>> ucnobj.list_records()
    >>> rec = ucnobj.get_data(kstpkper=(1,1))

    """

    def __init__(
        self,
        filename,
        text="concentration",
        precision="auto",
        verbose=False,
        **kwargs
    ):
        self.text = text.encode()
        if precision == "auto":
            precision = get_headfile_precision(filename)
        if precision == "unknown":
            s = "Error. Precision could not be determined for {}".format(
                filename
            )
            print(s)
            raise Exception()
        self.header_dtype = BinaryHeader.set_dtype(
            bintype="Ucn", precision=precision
        )
        super().__init__(filename, precision, verbose, kwargs)
        return


class BudgetIndexError(Exception):
    pass


class CellBudgetFile:
    """
    CellBudgetFile Class.

    Parameters
    ----------
    filename : string
        Name of the cell budget file
    precision : string
        'single' or 'double'.  Default is 'single'.
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

    Examples
    --------

    >>> import flopy.utils.binaryfile as bf
    >>> cbb = bf.CellBudgetFile('mymodel.cbb')
    >>> cbb.list_records()
    >>> rec = cbb.get_data(kstpkper=(0,0), text='RIVER LEAKAGE')

    """

    def __init__(self, filename, precision="auto", verbose=False, **kwargs):
        self.filename = filename
        self.precision = precision
        self.verbose = verbose
        self.file = open(self.filename, "rb")
        # Get filesize to ensure this is not an empty file
        self.file.seek(0, 2)
        totalbytes = self.file.tell()
        self.file.seek(0, 0)  # reset to beginning
        assert self.file.tell() == 0
        if totalbytes == 0:
            raise IOError("datafile error: file is empty: " + str(filename))
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
        self.paknamlist = []
        self.nrecords = 0

        self.dis = None
        self.modelgrid = None
        if "model" in kwargs.keys():
            self.model = kwargs.pop("model")
            self.modelgrid = self.model.modelgrid
            self.dis = self.model.dis
        if "dis" in kwargs.keys():
            self.dis = kwargs.pop("dis")
            self.modelgrid = self.dis.parent.modelgrid
        if "sr" in kwargs.keys():
            from ..discretization import StructuredGrid, UnstructuredGrid

            sr = kwargs.pop("sr")
            if sr.__class__.__name__ == "SpatialReferenceUnstructured":
                self.modelgrid = UnstructuredGrid(
                    vertices=sr.verts,
                    iverts=sr.iverts,
                    xcenters=sr.xc,
                    ycenters=sr.yc,
                    ncpl=sr.ncpl,
                )
            elif sr.__class__.__name__ == "SpatialReference":
                self.modelgrid = StructuredGrid(
                    delc=sr.delc,
                    delr=sr.delr,
                    xoff=sr.xll,
                    yoff=sr.yll,
                    angrot=sr.rotation,
                )
        if "modelgrid" in kwargs.keys():
            self.modelgrid = kwargs.pop("modelgrid")
        if len(kwargs.keys()) > 0:
            args = ",".join(kwargs.keys())
            raise Exception("LayerFile error: unrecognized kwargs: " + args)

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
            raise Exception("Unknown precision specified: " + precision)

        if not success:
            raise Exception(
                "Budget file could not be read using "
                "{} precision".format(precision)
            )

        return

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

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
        self.paknamlist = []
        self.nrecords = 0

    def _set_precision(self, precision="single"):
        """
        Method to set the budget precsion from a CBC file. Enables
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
            ("text", "a16"),
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
            ("modelnam", "a16"),
            ("paknam", "a16"),
            ("modelnam2", "a16"),
            ("paknam2", "a16"),
        ]
        self.header1_dtype = np.dtype(h1dt)
        self.header2_dtype0 = np.dtype(h2dt0)
        self.header2_dtype = np.dtype(h2dt)
        hdt = h1dt + h2dt
        self.header_dtype = np.dtype(hdt)

        try:
            self._build_index()
        except BudgetIndexError:
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
            dt1 = this_perlen * (tsmult - 1.0) / ((tsmult ** nstp) - 1.0)
        kstp_len = [dt1]
        for i in range(kstp + 1):
            kstp_len.append(kstp_len[-1] * tsmult)
        # kstp_len = np.array(kstp_len)
        # kstp_len = kstp_len[:kstp].sum()
        kstp_len = sum(kstp_len[: kstp + 1])
        return kper_len + kstp_len

    def _build_index(self):
        """
        Build the ordered dictionary, which maps the header information
        to the position in the binary file.
        """
        asciiset = " "
        for i in range(33, 127):
            asciiset += chr(i)

        header = self._get_header()
        self.nrow = header["nrow"]
        self.ncol = header["ncol"]
        self.nlay = np.abs(header["nlay"])
        text = header["text"]
        if isinstance(text, bytes):
            text = text.decode()
        if self.nrow < 0 or self.ncol < 0:
            raise Exception("negative nrow, ncol")
        self.file.seek(0, 2)
        self.totalbytes = self.file.tell()
        self.file.seek(0, 0)
        self.recorddict = OrderedDict()
        ipos = 0
        while ipos < self.totalbytes:
            self.iposheader.append(ipos)
            header = self._get_header()
            self.nrecords += 1
            totim = header["totim"]
            if totim == 0:
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
                try:
                    tlist = [header["text"], header["modelnam"]]
                    for text in tlist:
                        if isinstance(text, bytes):
                            text = text.decode()
                        for t in text:
                            if t.upper() not in asciiset:
                                raise Exception()

                except:
                    raise BudgetIndexError("Improper precision")
                self.textlist.append(header["text"])
                self.imethlist.append(header["imeth"])
            if header["paknam"] not in self.paknamlist:
                self.paknamlist.append(header["paknam"])
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
                    if isinstance(s, bytes):
                        s = s.decode()
                    print(itxt + ": " + str(s))
                print("file position: ", ipos)
                if (
                    int(header["imeth"]) != 5
                    and int(header["imeth"]) != 6
                    and int(header["imeth"]) != 7
                ):
                    print("")

            # store record and byte position mapping
            self.recorddict[
                tuple(header)
            ] = ipos  # store the position right after header2
            self.recordarray.append(header)
            self.iposarray.append(
                ipos
            )  # store the position right after header2

            # skip over the data to the next record and set ipos
            self._skip_record(header)
            ipos = self.file.tell()

        # convert to numpy arrays
        self.recordarray = np.array(self.recordarray, dtype=self.header_dtype)
        self.iposheader = np.array(self.iposheader, dtype=np.int64)
        self.iposarray = np.array(self.iposarray, dtype=np.int64)
        self.nper = self.recordarray["kper"].max()
        return

    def _skip_record(self, header):
        """
        Skip over this record, not counting header and header2.

        """
        nlay = abs(header["nlay"])
        nrow = header["nrow"]
        ncol = header["ncol"]
        imeth = header["imeth"]
        if imeth == 0:
            nbytes = nrow * ncol * nlay * self.realtype(1).nbytes
        elif imeth == 1:
            nbytes = nrow * ncol * nlay * self.realtype(1).nbytes
        elif imeth == 2:
            nlist = binaryread(self.file, np.int32)[0]
            nbytes = nlist * (np.int32(1).nbytes + self.realtype(1).nbytes)
        elif imeth == 3:
            nbytes = nrow * ncol * self.realtype(1).nbytes
            nbytes += nrow * ncol * np.int32(1).nbytes
        elif imeth == 4:
            nbytes = nrow * ncol * self.realtype(1).nbytes
        elif imeth == 5:
            nauxp1 = binaryread(self.file, np.int32)[0]
            naux = nauxp1 - 1

            for i in range(naux):
                temp = binaryread(self.file, str, charlen=16)
            nlist = binaryread(self.file, np.int32)[0]
            if self.verbose:
                print("naux: ", naux)
                print("nlist: ", nlist)
                print("")
            nbytes = nlist * (
                np.int32(1).nbytes
                + self.realtype(1).nbytes
                + naux * self.realtype(1).nbytes
            )
        elif imeth == 6:
            # read rest of list data
            nauxp1 = binaryread(self.file, np.int32)[0]
            naux = nauxp1 - 1

            for i in range(naux):
                temp = binaryread(self.file, str, charlen=16)
            nlist = binaryread(self.file, np.int32)[0]
            if self.verbose:
                print("naux: ", naux)
                print("nlist: ", nlist)
                print("")
            nbytes = nlist * (
                np.int32(1).nbytes * 2
                + self.realtype(1).nbytes
                + naux * self.realtype(1).nbytes
            )
        else:
            raise Exception("invalid method code " + str(imeth))
        if nbytes != 0:
            self.file.seek(nbytes, 1)
        return

    def _get_header(self):
        """
        Read the file header

        """
        header1 = binaryread(self.file, self.header1_dtype, (1,))
        nlay = header1["nlay"]
        if nlay < 0:
            # fill header2 by first reading imeth, delt, pertim and totim
            # and then adding modelnames and paknames if imeth = 6
            temp = binaryread(self.file, self.header2_dtype0, (1,))
            header2 = np.array(
                [(0, 0.0, 0.0, 0.0, "", "", "", "")], dtype=self.header2_dtype
            )
            for name in temp.dtype.names:
                header2[name] = temp[name]
            if int(header2["imeth"]) == 6:
                header2["modelnam"] = binaryread(self.file, str, charlen=16)
                header2["paknam"] = binaryread(self.file, str, charlen=16)
                header2["modelnam2"] = binaryread(self.file, str, charlen=16)
                header2["paknam2"] = binaryread(self.file, str, charlen=16)
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

    def _find_paknam(self, paknam):
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
            for t in self._unique_package_names():
                if tpaknam.upper() in t.decode():
                    paknam16 = t
                    break
            if paknam16 is None:
                raise Exception(
                    "The specified package name string is not "
                    "in the budget file."
                )
        return paknam16

    def list_records(self):
        """
        Print a list of all of the records in the file
        """
        for rec in self.recordarray:
            if isinstance(rec, bytes):
                rec = rec.decode()
            print(rec)
        return

    def list_unique_records(self):
        """
        Print a list of unique record names
        """
        print("RECORD           IMETH")
        print(22 * "-")
        for rec, imeth in zip(self.textlist, self.imethlist):
            if isinstance(rec, bytes):
                rec = rec.decode()
            print("{:16} {:5d}".format(rec.strip(), imeth))
        return

    def list_unique_packages(self):
        """
        Print a list of unique package names
        """
        for rec in self._unique_package_names():
            if isinstance(rec, bytes):
                rec = rec.decode()
            print(rec)
        return

    def get_unique_record_names(self, decode=False):
        """
        Get a list of unique record names in the file

        Parameters
        ----------
        decode : bool
            Optional boolean used to decode byte strings (default is False).

        Returns
        ----------
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

    def get_unique_package_names(self, decode=False):
        """
        Get a list of unique package names in the file

        Parameters
        ----------
        decode : bool
            Optional boolean used to decode byte strings (default is False).

        Returns
        ----------
        names : list of strings
            List of unique package names in the binary file.

        """
        if decode:
            names = []
            for text in self.paknamlist:
                if isinstance(text, bytes):
                    text = text.decode()
                names.append(text)
        else:
            names = self.paknamlist
        return names

    def _unique_package_names(self):
        """
        Get a list of unique package names in the file

        Returns
        ----------
        out : list of strings
            List of unique package names in the binary file.

        """
        return self.paknamlist

    def get_kstpkper(self):
        """
        Get a list of unique stress periods and time steps in the file

        Returns
        ----------
        out : list of (kstp, kper) tuples
            List of unique kstp, kper combinations in binary file.  kstp and
            kper values are zero-based.

        """
        kstpkper = []
        for kstp, kper in self.kstpkper:
            kstpkper.append((kstp - 1, kper - 1))
        return kstpkper

    def get_indices(self, text=None):
        """
        Get a list of indices for a selected record name

        Parameters
        ----------
        text : str
            The text identifier for the record.  Examples include
            'RIVER LEAKAGE', 'STORAGE', 'FLOW RIGHT FACE', etc.

        Returns
        ----------
        out : tuple
            indices of selected record name in budget file.

        """
        # check and make sure that text is in file
        if text is not None:
            text16 = self._find_text(text)
            select_indices = np.where((self.recordarray["text"] == text16))
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
        full3D=False,
    ):
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
        full3D : boolean
            If true, then return the record as a three dimensional numpy
            array, even for those list-style records written as part of a
            'COMPACT BUDGET' MODFLOW budget file.  (Default is False.)

        Returns
        ----------
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

        if kstpkper is not None:
            kstp1 = kstpkper[0] + 1
            kper1 = kstpkper[1] + 1
            if text is None and paknam is None:
                select_indices = np.where(
                    (self.recordarray["kstp"] == kstp1)
                    & (self.recordarray["kper"] == kper1)
                )
            else:
                if paknam is None and text is not None:
                    select_indices = np.where(
                        (self.recordarray["kstp"] == kstp1)
                        & (self.recordarray["kper"] == kper1)
                        & (self.recordarray["text"] == text16)
                    )
                elif text is None and paknam is not None:
                    select_indices = np.where(
                        (self.recordarray["kstp"] == kstp1)
                        & (self.recordarray["kper"] == kper1)
                        & (self.recordarray["paknam"] == paknam16)
                    )
                else:
                    select_indices = np.where(
                        (self.recordarray["kstp"] == kstp1)
                        & (self.recordarray["kper"] == kper1)
                        & (self.recordarray["text"] == text16)
                        & (self.recordarray["paknam"] == paknam16)
                    )

        elif totim is not None:
            if text is None and paknam is None:
                select_indices = np.where((self.recordarray["totim"] == totim))
            else:
                if paknam is None and text is not None:
                    select_indices = np.where(
                        (self.recordarray["totim"] == totim)
                        & (self.recordarray["text"] == text16)
                    )
                elif text is None and paknam is not None:
                    select_indices = np.where(
                        (self.recordarray["totim"] == totim)
                        & (self.recordarray["paknam"] == paknam16)
                    )
                else:
                    select_indices = np.where(
                        (self.recordarray["totim"] == totim)
                        & (self.recordarray["text"] == text16)
                        & (self.recordarray["paknam"] == paknam16)
                    )

        # allow for idx to be a list or a scalar
        elif idx is not None:
            if isinstance(idx, list):
                select_indices = idx
            else:
                select_indices = [idx]

        # case where only text is entered
        elif text is not None:
            select_indices = np.where((self.recordarray["text"] == text16))

        else:
            raise TypeError(
                "get_data() missing 1 required argument: 'kstpkper', 'totim', "
                "'idx', or 'text'"
            )

        # build and return the record list
        if isinstance(select_indices, tuple):
            select_indices = select_indices[0]
        recordlist = []
        for idx in select_indices:
            rec = self.get_record(idx, full3D=full3D)
            recordlist.append(rec)

        return recordlist

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
        ----------
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
                "text keyword must be provided to CellBudgetFile "
                "get_ts() method."
            )

        kijlist = self._build_kijlist(idx)
        nstation = self._get_nstation(idx, kijlist)

        # Initialize result array and put times in first column
        result = self._init_result(nstation)

        kk = self.get_kstpkper()
        timesint = self.get_times()
        if len(timesint) < 1:
            if times is None:
                timesint = [x + 1 for x in range(len(kk))]
            else:
                if isinstance(times, np.ndarray):
                    times = times.tolist()
                if len(times) != len(kk):
                    raise Exception(
                        "times passed to CellBudgetFile get_ts() "
                        "method must be equal to {} "
                        "not {}".format(len(kk), len(times))
                    )
                timesint = times
        for idx, t in enumerate(timesint):
            result[idx, 0] = t

        for itim, k in enumerate(kk):
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
                            lrc[0]
                            * (self.modelgrid.nrow * self.modelgrid.ncol)
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
                        dix = np.where(np.isin(vv["node"], ndx))[0]
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
                    "Invalid cell index. Cell {} not within model grid: "
                    "{}".format((k, i, j), (self.nlay, self.nrow, self.ncol))
                )
        return kijlist

    def _get_nstation(self, idx, kijlist):
        if isinstance(idx, list):
            return len(kijlist)
        elif isinstance(idx, tuple):
            return 1

    def _init_result(self, nstation):
        # Initialize result array and put times in first column
        result = np.empty(
            (len(self.kstpkper), nstation + 1), dtype=self.realtype
        )
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
        ----------
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
        ipos = int(self.iposarray[idx])
        self.file.seek(ipos, 0)
        imeth = header["imeth"][0]

        t = header["text"][0]
        if isinstance(t, bytes):
            t = t.decode("utf-8")
        s = "Returning " + str(t).strip() + " as "

        nlay = abs(header["nlay"][0])
        nrow = header["nrow"][0]
        ncol = header["ncol"][0]

        # default method
        if imeth == 0:
            if self.verbose:
                s += "an array of shape " + str((nlay, nrow, ncol))
                print(s)
            return binaryread(
                self.file, self.realtype(1), shape=(nlay, nrow, ncol)
            )
        # imeth 1
        elif imeth == 1:
            if self.verbose:
                s += "an array of shape " + str((nlay, nrow, ncol))
                print(s)
            return binaryread(
                self.file, self.realtype(1), shape=(nlay, nrow, ncol)
            )

        # imeth 2
        elif imeth == 2:
            nlist = binaryread(self.file, np.int32)[0]
            dtype = np.dtype([("node", np.int32), ("q", self.realtype)])
            if self.verbose:
                if full3D:
                    s += "a numpy masked array of size ({}, {}, {})".format(
                        nlay, nrow, ncol
                    )
                else:
                    s += "a numpy recarray of size ({}, 2)".format(nlist)
                print(s)
            data = binaryread(self.file, dtype, shape=(nlist,))
            if full3D:
                return self.create3D(data, nlay, nrow, ncol)
            else:
                return data.view(np.recarray)

        # imeth 3
        elif imeth == 3:
            ilayer = binaryread(self.file, np.int32, shape=(nrow, ncol))
            data = binaryread(self.file, self.realtype(1), shape=(nrow, ncol))
            if self.verbose:
                if full3D:
                    s += "a numpy masked array of size ({}, {}, {})".format(
                        nlay, nrow, ncol
                    )
                else:
                    s += (
                        "a list of two 2D numpy arrays.  "
                        "The first is an integer layer array of shape {}.  "
                        "The second is real data array of shape {}".format(
                            (nrow, ncol),
                            (nrow, ncol),
                        )
                    )
                print(s)
            if full3D:
                out = np.ma.zeros((nlay, nrow, ncol), dtype=np.float32)
                out.mask = True
                vertical_layer = ilayer[0] - 1  # This is always the top layer
                out[vertical_layer, :, :] = data
                return out
            else:
                return [ilayer, data]

        # imeth 4
        elif imeth == 4:
            if self.verbose:
                s += "a 2d numpy array of size ({}, {})".format(nrow, ncol)
                print(s)
            return binaryread(self.file, self.realtype(1), shape=(nrow, ncol))

        # imeth 5
        elif imeth == 5:
            nauxp1 = binaryread(self.file, np.int32)[0]
            naux = nauxp1 - 1
            l = [("node", np.int32), ("q", self.realtype)]
            for i in range(naux):
                auxname = binaryread(self.file, str, charlen=16)
                if not isinstance(auxname, str):
                    auxname = auxname.decode()
                l.append((auxname, self.realtype))
            dtype = np.dtype(l)
            nlist = binaryread(self.file, np.int32)[0]
            data = binaryread(self.file, dtype, shape=(nlist,))
            if full3D:
                if self.verbose:
                    s += "a list array of shape ({}, {}, {})".format(
                        nlay, nrow, ncol
                    )
                    print(s)
                return self.create3D(data, nlay, nrow, ncol)
            else:
                if self.verbose:
                    s += "a numpy recarray of size ({}, {})".format(
                        nlist, 2 + naux
                    )
                    print(s)
                return data.view(np.recarray)

        # imeth 6
        elif imeth == 6:
            # read rest of list data
            nauxp1 = binaryread(self.file, np.int32)[0]
            naux = nauxp1 - 1
            l = [("node", np.int32), ("node2", np.int32), ("q", self.realtype)]
            for i in range(naux):
                auxname = binaryread(self.file, str, charlen=16)
                if not isinstance(auxname, str):
                    auxname = auxname.decode()
                l.append((auxname.strip(), self.realtype))
            dtype = np.dtype(l)
            nlist = binaryread(self.file, np.int32)[0]
            data = binaryread(self.file, dtype, shape=(nlist,))
            if self.verbose:
                if full3D:
                    s += (
                        "full 3D arrays not supported for "
                        "imeth = {}".format(imeth)
                    )
                else:
                    s += "a numpy recarray of size ({}, 2)".format(nlist)
                print(s)
            if full3D:
                s += "full 3D arrays not supported for imeth = {}".format(
                    imeth
                )
                raise ValueError(s)
            else:
                return data.view(np.recarray)
        else:
            raise ValueError("invalid imeth value - {}".format(imeth))

        # should not reach this point
        return

    def create3D(self, data, nlay, nrow, ncol):
        """
        Convert a dictionary of {node: q, ...} into a numpy masked array.
        In most cases this should not be called directly by the user unless
        you know what you're doing.  Instead, it is used as part of the
        full3D keyword for get_data.

        Parameters
        ----------
        data : dictionary
            Dictionary with node keywords and flows (q) items.

        nlay, nrow, ncol : int
            Number of layers, rows, and columns of the model grid.

        Returns
        ----------
        out : numpy masked array
            List contains unique simulation times (totim) in binary file.

        """
        out = np.ma.zeros((nlay * nrow * ncol), dtype=np.float32)
        out.mask = True
        for [node, q] in zip(data["node"], data["q"]):
            idx = node - 1
            out.data[idx] += q
            out.mask[idx] = False
        return np.ma.reshape(out, (nlay, nrow, ncol))

    def get_times(self):
        """
        Get a list of unique times in the file

        Returns
        ----------
        out : list of floats
            List contains unique simulation times (totim) in binary file.

        """
        return self.times

    def get_nrecords(self):
        """
        Return the number of records in the file

        Returns
        -------

        out : int
            Number of records in the file.

        """
        return self.recordarray.shape[0]

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
        select_indices = np.where((self.recordarray["totim"] == totim))[0]

        for i in select_indices:
            text = self.recordarray[i]["text"].decode()
            if self.verbose:
                print("processing {}".format(text))
            flow = self.get_record(idx=i, full3D=True)
            if ncol > 1 and "RIGHT FACE" in text:
                residual -= flow[:, :, :]
                residual[:, :, 1:] += flow[:, :, :-1]
                if scaled:
                    idx = np.where(flow < 0.0)
                    inflow[idx] -= flow[idx]
                    idx = np.where(flow > 0.0)
                    l, r, c = idx
                    idx = (l, r, c + 1)
                    inflow[idx] += flow[idx]
            elif nrow > 1 and "FRONT FACE" in text:
                residual -= flow[:, :, :]
                residual[:, 1:, :] += flow[:, :-1, :]
                if scaled:
                    idx = np.where(flow < 0.0)
                    inflow[idx] -= flow[idx]
                    idx = np.where(flow > 0.0)
                    l, r, c = idx
                    idx = (l, r + 1, c)
                    inflow[idx] += flow[idx]
            elif nlay > 1 and "LOWER FACE" in text:
                residual -= flow[:, :, :]
                residual[1:, :, :] += flow[:-1, :, :]
                if scaled:
                    idx = np.where(flow < 0.0)
                    inflow[idx] -= flow[idx]
                    idx = np.where(flow > 0.0)
                    l, r, c = idx
                    idx = (l + 1, r, c)
                    inflow[idx] += flow[idx]
            else:
                residual += flow
                if scaled:
                    idx = np.where(flow > 0.0)
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
        return


class HeadUFile(BinaryLayerFile):
    """
    Unstructured MODFLOW-USG HeadUFile Class.

    Parameters
    ----------
    filename : string
        Name of the concentration file
    text : string
        Name of the text string in the head file.  Default is 'headu'
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
    The HeadUFile class provides simple ways to retrieve a list of
    head arrays from a MODFLOW-USG binary head file and time series
    arrays for one or more cells.

    The BinaryLayerFile class is built on a record array consisting of
    headers, which are record arrays of the modflow header information
    (kstp, kper, pertim, totim, text, nrow, ncol, ilay)
    and long integers, which are pointers to first bytes of data for
    the corresponding data array.  For unstructured grids, nrow and ncol
    are the starting and ending node numbers for layer, ilay.  This class
    overrides methods in the parent class so that the proper sized arrays
    are created.

    When the get_data method is called for this class, a list of
    one-dimensional arrays will be returned, where each array is the head
    array for a layer.  If the heads for a layer were not saved, then
    None will be returned for that layer.

    Examples
    --------

    >>> import flopy.utils.binaryfile as bf
    >>> hdobj = bf.HeadUFile('model.hds')
    >>> hdobj.list_records()
    >>> usgheads = hdobj.get_data(kstpkper=(1, 50))


    """

    def __init__(
        self, filename, text="headu", precision="auto", verbose=False, **kwargs
    ):
        """
        Class constructor
        """
        self.text = text.encode()
        if precision == "auto":
            precision = get_headfile_precision(filename)
            if precision == "unknown":
                s = "Error. Precision could not be determined for {}".format(
                    filename
                )
                print(s)
                raise Exception()
        self.header_dtype = BinaryHeader.set_dtype(
            bintype="Head", precision=precision
        )
        super().__init__(filename, precision, verbose, kwargs)
        return

    def _get_data_array(self, totim=0.0):
        """
        Get a list of 1D arrays for the
        specified kstp and kper value or totim value.

        """

        if totim >= 0.0:
            keyindices = np.where((self.recordarray["totim"] == totim))[0]
            if len(keyindices) == 0:
                msg = "totim value ({}) not found in file...".format(totim)
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
                msg = "Byte position in file: {} for ".format(
                    ipos
                ) + "layer {}".format(ilay)
                print(msg)
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
        Get a time series from the binary HeadUFile (not implemented).

        Parameters
        ----------
        idx : tuple of ints, or a list of a tuple of ints
            idx can be (layer, row, column) or it can be a list in the form
            [(layer, row, column), (layer, row, column), ...].  The layer,
            row, and column values must be zero based.

        Returns
        ----------
        out : numpy array
            Array has size (ntimes, ncells + 1).  The first column in the
            data array will contain time (totim).

        See Also
        --------

        Notes
        -----

        Examples
        --------

        """
        msg = "HeadUFile: get_ts() is not implemented"
        raise NotImplementedError(msg)
