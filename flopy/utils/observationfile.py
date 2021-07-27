import numpy as np
import io
from ..utils.utils_def import FlopyBinaryData
from ..utils.flopy_io import get_ts_sp


class ObsFiles(FlopyBinaryData):
    def __init__(self):
        super().__init__()
        return

    def get_times(self):
        """
        Get a list of unique times in the file

        Returns
        ----------
        out : list of floats
            List contains unique simulation times (totim) in binary file.

        """
        return self.data["totim"].reshape(self.get_ntimes()).tolist()

    def get_ntimes(self):
        """
        Get the number of times in the file

        Returns
        ----------
        out : int
            The number of simulation times (totim) in binary file.

        """
        return self.data["totim"].shape[0]

    def get_nobs(self):
        """
        Get the number of observations in the file

        Returns
        ----------
        out : tuple of int
            A tupe with the number of records and number of flow items
            in the file. The number of flow items is non-zero only if
            swrtype='flow'.

        """
        return self.nobs

    def get_obsnames(self):
        """
        Get a list of observation names in the file

        Returns
        ----------
        out : list of strings
            List of observation names in the binary file. totim is not
            included in the list of observation names.

        """
        return list(self.data.dtype.names[1:])

    def get_data(self, idx=None, obsname=None, totim=None):
        """
        Get data from the observation file.

        Parameters
        ----------
        idx : int
            The zero-based record number.  The first record is record 0.
            If idx is None and totim are None, data for all simulation times
            are returned. (default is None)
        obsname : string
            The name of the observation to return. If obsname is None, all
            observation data are returned. (default is None)
        totim : float
            The simulation time to return. If idx is None and totim are None,
            data for all simulation times are returned. (default is None)

        Returns
        ----------
        data : numpy record array
            Array has size (ntimes, nitems). totim is always returned. nitems
            is 2 if idx or obsname is not None or nobs+1.

        See Also
        --------

        Notes
        -----
        If both idx and obsname are None, will return all of the observation
        data.

        Examples
        --------
        >>> hyd = HydmodObs("my_model.hyd")
        >>> ts = hyd.get_data()

        """
        i0 = 0
        i1 = self.data.shape[0]
        if totim is not None:
            idx = np.where(self.data["totim"] == totim)[0][0]
            i0 = idx
            i1 = idx + 1
        elif idx is not None:
            if idx < i1:
                i0 = idx
            i1 = i0 + 1
        r = None
        if obsname is None:
            obsname = self.get_obsnames()
        else:
            if obsname is not None:
                if obsname not in self.data.dtype.names:
                    obsname = None
                else:
                    if not isinstance(obsname, list):
                        obsname = [obsname]
        if obsname is not None:
            obsname.insert(0, "totim")
            r = get_selection(self.data, obsname)[i0:i1]
        return r

    def get_dataframe(
        self,
        start_datetime="1-1-1970",
        idx=None,
        obsname=None,
        totim=None,
        timeunit="D",
    ):
        """
        Get pandas dataframe with the incremental and cumulative water budget
        items in the hydmod file.

        Parameters
        ----------
        start_datetime : str
            If start_datetime is passed as None, the rows are indexed on totim.
            Otherwise, a DatetimeIndex is set. (default is 1-1-1970).
        idx : int
            The zero-based record number.  The first record is record 0.
            If idx is None and totim are None, a dataframe with all simulation
            times is  returned. (default is None)
        obsname : string
            The name of the observation to return. If obsname is None, all
            observation data are returned. (default is None)
        totim : float
            The simulation time to return. If idx is None and totim are None,
            a dataframe with all simulation times is returned.
            (default is None)
        timeunit : string
            time unit of the simulation time. Valid values are 'S'econds,
            'M'inutes, 'H'ours, 'D'ays, 'Y'ears. (default is 'D').

        Returns
        -------
        out : pandas dataframe
            Pandas dataframe of selected data.

        See Also
        --------

        Notes
        -----
        If both idx and obsname are None, will return all of the observation
        data as a dataframe.

        Examples
        --------
        >>> hyd = HydmodObs("my_model.hyd")
        >>> df = hyd.get_dataframes()

        """

        try:
            import pandas as pd
            from ..utils.utils_def import totim_to_datetime
        except Exception as e:
            msg = "ObsFiles.get_dataframe() error import pandas: " + str(e)
            raise ImportError(msg)

        i0 = 0
        i1 = self.data.shape[0]
        if totim is not None:
            idx = np.where(self.data["totim"] == totim)[0][0]
            i0 = idx
            i1 = idx + 1
        elif idx is not None:
            if idx < i1:
                i0 = idx
            i1 = i0 + 1

        if obsname is None:
            obsname = self.get_obsnames()
        else:
            if obsname is not None:
                if obsname not in self.data.dtype.names:
                    obsname = None
                else:
                    if not isinstance(obsname, list):
                        obsname = [obsname]
        if obsname is None:
            return None

        obsname.insert(0, "totim")

        dti = self.get_times()[i0:i1]
        if start_datetime is not None:
            dti = totim_to_datetime(
                dti, start=pd.to_datetime(start_datetime), timeunit=timeunit
            )

        df = pd.DataFrame(self.data[i0:i1], index=dti, columns=obsname)
        return df

    def _read_data(self):

        if self.data is not None:
            return

        while True:
            try:
                r = self.read_record(count=1)
                if self.data is None:
                    self.data = r.copy()
                elif r.size == 0:
                    break
                else:
                    self.data = np.hstack((self.data, r))
            except:
                break
        return

    def _build_dtype(self):
        """
        Build the recordarray and iposarray, which maps the header information
        to the position in the formatted file.
        """
        raise Exception(
            "Abstract method _build_dtype called in BinaryFiles. "
            "This method needs to be overridden."
        )

    def _build_index(self):
        """
        Build the recordarray and iposarray, which maps the header information
        to the position in the formatted file.
        """
        raise Exception(
            "Abstract method _build_index called in BinaryFiles. "
            "This method needs to be overridden."
        )


class Mf6Obs(ObsFiles):
    """
    Mf6Obs Class - used to read ascii and binary MODFLOW6 observation output

    Parameters
    ----------
    filename : str
        Name of the hydmod output file
    verbose : boolean
        If true, print additional information to to the screen during the
        extraction.  (default is False)
    isBinary : str, bool
        default is "auto", code will attempt to automatically check if
        file is binary. User can change this to True or False if the auto
        check fails to work

    Returns
    -------
    None

    """

    def __init__(self, filename, verbose=False, isBinary="auto"):
        """
        Class constructor.

        """
        super().__init__()
        # initialize class information
        self.verbose = verbose

        # check if this is a binary file
        if isBinary == "auto":
            with open(filename) as foo:
                if isinstance(foo, io.TextIOBase):
                    isBinary = False
                elif isinstance(foo, (io.RawIOBase, io.BufferedIOBase)):
                    isBinary = True
                else:
                    err = "Could not determine if file is binary or ascii"
                    raise IOError(err)
        if isBinary:
            # --open binary head file
            self.file = open(filename, "rb")

            # read control line
            cline = self.read_text(nchar=100)
            precision = "single"
            if "double" in cline[5:11].lower():
                precision = "double"
            self.set_float(precision)
            lenobsname = int(cline[11:])

            # get number of observations
            self.nobs = self.read_integer()

            # # continue reading the file
            # self.v = np.empty(self.nobs, dtype=float)
            # self.v.fill(1.0E+32)

            # read obsnames
            obsnames = []
            for idx in range(0, self.nobs):
                cid = self.read_text(lenobsname)
                obsnames.append(cid)
            self.obsnames = np.array(obsnames)

            # build dtype
            self.dtype = _build_dtype(self.obsnames, self.floattype)

            # build index
            self._build_index()

            self.data = None
            self._read_data()
        else:
            # read ascii data
            csv = CsvFile(filename)
            self.obsnames = csv.obsnames
            self.nobs = csv.nobs
            self.data = csv.data

    def _build_index(self):
        return


class HydmodObs(ObsFiles):
    """
    HydmodObs Class - used to read binary MODFLOW HYDMOD package output

    Parameters
    ----------
    filename : str
        Name of the hydmod output file
    verbose : boolean
        If true, print additional information to to the screen during the
        extraction.  (default is False)
    hydlbl_len : int
        Length of hydmod labels. (default is 20)

    Returns
    -------
    None

    """

    def __init__(self, filename, verbose=False, hydlbl_len=20):
        """
        Class constructor.

        """
        super().__init__()
        # initialize class information
        self.verbose = verbose
        # --open binary head file
        self.file = open(filename, "rb")
        # NHYDTOT,ITMUNI
        self.nobs = self.read_integer()
        precision = "single"
        if self.nobs < 0:
            self.nobs = abs(self.nobs)
            precision = "double"
        self.set_float(precision)

        # continue reading the file
        self.itmuni = self.read_integer()
        self.v = np.empty(self.nobs, dtype=float)
        self.v.fill(1.0e32)
        ctime = self.read_text(nchar=4)
        self.hydlbl_len = int(hydlbl_len)
        # read HYDLBL
        hydlbl = []
        for idx in range(0, self.nobs):
            cid = self.read_text(self.hydlbl_len)
            hydlbl.append(cid)
        self.hydlbl = np.array(hydlbl)

        # build dtype
        self.dtype = _build_dtype(self.hydlbl, self.floattype)

        # build index
        self._build_index()

        self.data = None
        self._read_data()

    def _build_index(self):
        return


class SwrObs(ObsFiles):
    """
    Read binary SWR observations output from MODFLOW SWR Process
    observation files

    Parameters
    ----------
    filename : string
        Name of the cell budget file
    precision : string
        'single' or 'double'.  Default is 'double'.
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

    >>> import flopy
    >>> so = flopy.utils.SwrObs('mymodel.swr.obs')

    """

    def __init__(self, filename, precision="double", verbose=False):
        """
        Class constructor.

        """
        super().__init__()
        self.set_float(precision=precision)
        # initialize class information
        self.verbose = verbose
        # open binary head file
        self.file = open(filename, "rb")

        # NOBS
        self.nobs = self.read_integer()
        # read obsnames
        obsnames = []
        for idx in range(0, self.nobs):
            cid = self.read_text()
            if isinstance(cid, bytes):
                cid = cid.decode()
            obsnames.append(cid.strip())
        self.obs = obsnames

        # read header information
        self._build_dtype()

        # build index
        self._build_index()

        # read data
        self.data = None
        self._read_data()

    def _build_dtype(self):
        vdata = [("totim", self.floattype)]
        for name in self.obs:
            vdata.append((str(name), self.floattype))
        self.dtype = np.dtype(vdata)
        return

    def _build_index(self):
        return


class CsvFile:
    """
    Class for reading csv based output files

    Parameters
    ----------
    csvfile : str
        csv file name
    delimiter : str
        optional delimiter for the csv or formatted text file,
        defaults to ","

    """

    def __init__(self, csvfile, delimiter=","):

        self.file = open(csvfile, "r")
        self.delimiter = delimiter

        # read header line
        line = self.file.readline()
        self._header = line.rstrip().split(delimiter)
        self.floattype = "f8"
        self.dtype = _build_dtype(self._header, self.floattype)

        self.data = self.read_csv(self.file, self.dtype, delimiter)

    @property
    def obsnames(self):
        """
        Method to get the observation names

        Returns
        -------
        list
        """
        return [i for i in self._header if i.lower() != "totim"]

    @property
    def nobs(self):
        """
        Method to get the number of observations

        Returns
        -------
        int
        """
        return len(self.obsnames)

    @staticmethod
    def read_csv(fobj, dtype, delimiter=","):
        """

        Parameters
        ----------
        fobj : file object
            open text file object to read
        dtype : np.dtype
        delimiter : str
            optional delimiter for the csv or formatted text file,
            defaults to ","

        Returns
        -------
        np.recarray
        """
        arr = np.genfromtxt(fobj, dtype=dtype, delimiter=delimiter)
        return arr.view(np.recarray)


def get_selection(data, names):
    """

    Parameters
    ----------
    data : numpy recarray
        recarray of data to make a selection from
    names : string or list of strings
        column names to return

    Returns
    -------
    out : numpy recarray
        recarray with selection

    """
    if not isinstance(names, list):
        names = [names]
    ierr = 0
    for name in names:
        if name not in data.dtype.names:
            ierr += 1
            print("Error: {} is not a valid column name".format(name))
    if ierr > 0:
        raise Exception("Error: {} names did not match".format(ierr))

    # Valid list of names so make a selection
    dtype2 = np.dtype({name: data.dtype.fields[name] for name in names})
    return np.ndarray(data.shape, dtype2, data, 0, data.strides)


def _build_dtype(obsnames, floattype="f4"):
    """
    Generic method to build observation file dtypes

    Parameters
    ----------
    obsnames : list
        observation names (column headers)
    floattype : str
        floating point type "f4" or "f8"

    Returns
    -------
    np.dtype object

    """
    dtype = []
    if "time" in obsnames or "TIME" in obsnames:
        try:
            idx = obsnames.index("time")
        except ValueError:
            idx = obsnames.index("TIME")
        obsnames[idx] = "totim"

    elif "totim" not in obsnames:
        dtype = [("totim", floattype)]

    for site in obsnames:
        if not isinstance(site, str):
            site_name = site.decode().strip()
        else:
            site_name = site.strip()

        if site_name in ("KPER", "KSTP", "NULL"):
            dtype.append((site_name, int))
        else:
            dtype.append((site_name, floattype))

    return np.dtype(dtype)


def get_reduced_pumping(f, structured=True):
    """
    Method to read reduced pumping from a list file or an external
    reduced pumping observation file

    Parameters
    ----------
    f : str
        file name
    structured : bool
        boolean flag to indicate if model is Structured or USG model. Defaults
        to True (structured grid).

    Returns
    -------
        np.recarray : recarray of reduced pumping records.

    """
    # Set dtypes for resulting data
    if structured:
        dtype = np.dtype(
            [
                ("SP", int),
                ("TS", int),
                ("LAY", int),
                ("ROW", int),
                ("COL", int),
                ("APPL.Q", float),
                ("ACT.Q", float),
                ("GW-HEAD", float),
                ("CELL-BOT", float),
            ]
        )

        key = "WELLS WITH REDUCED PUMPING FOR STRESS PERIOD"
    else:
        dtype = np.dtype(
            [
                ("SP", int),
                ("TS", int),
                ("WELL.NO", int),
                ("CLN NODE", int),
                ("APPL.Q", float),
                ("ACT.Q", float),
                ("GW_HEAD", float),
                ("CELL_BOT", float),
            ]
        )

        key = "WELLS WITH REDUCED PUMPING FOR STRESS PERIOD"

    with open(f) as foo:
        data = []
        while True:
            line = foo.readline()
            if line == "":
                break
            # If l is reduced ppg header row
            if key in line:
                # Extract sp and ts
                ts, sp = get_ts_sp(line)
                # Skip line of data column titles
                foo.readline()
                # Iterate through lines of reduced ppg data
                while True:
                    line = foo.readline()
                    # Condition to exit loop
                    if len(line.strip().split()) < 6:
                        break
                    # Create list of hold line of data
                    ls = [sp, ts]
                    # Add other data to list
                    ls.extend([float(x) for x in line.split()])
                    # Add list to overall list of data
                    data.append(tuple(ls))

    return np.rec.fromrecords(data, dtype=dtype)
