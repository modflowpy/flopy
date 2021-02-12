import numpy as np

from ..utils.utils_def import FlopyBinaryData


class ObsFiles(FlopyBinaryData):
    def __init__(self):
        super(ObsFiles, self).__init__()
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
                    # should be hstack based on (https://mail.scipy.org/pipermail/numpy-discussion/2010-June/051107.html)
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
            "Abstract method _build_dtype called in BinaryFiles.  This method needs to be overridden."
        )

    def _build_index(self):
        """
        Build the recordarray and iposarray, which maps the header information
        to the position in the formatted file.
        """
        raise Exception(
            "Abstract method _build_index called in BinaryFiles.  This method needs to be overridden."
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
    hydlbl_len : int
        Length of hydmod labels. (default is 20)

    Returns
    -------
    None

    """

    def __init__(self, filename, verbose=False, isBinary=True):
        """
        Class constructor.

        """
        super(Mf6Obs, self).__init__()
        # initialize class information
        self.verbose = verbose
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
            self._build_dtype()

            # build index
            self._build_index()

            self.data = None
            self._read_data()
        else:
            # --open binary head file
            self.file = open(filename, "r")

            # read header line
            line = self.file.readline()
            t = line.rstrip().split(",")
            self.set_float("double")

            # get number of observations
            self.nobs = len(t) - 1

            # set obsnames
            obsnames = []
            for idx in range(1, self.nobs + 1):
                obsnames.append(t[idx])
            self.obsnames = np.array(obsnames)

            # build dtype
            self._build_dtype()

            # build index
            self._build_index()

            # read ascii data
            self.data = np.loadtxt(
                self.file, dtype=self.dtype, delimiter=",", ndmin=1
            )
        return

    def _build_dtype(self):

        # create dtype
        dtype = [("totim", self.floattype)]
        for site in self.obsnames:
            if not isinstance(site, str):
                site_name = site.decode().strip()
            else:
                site_name = site.strip()
            dtype.append((site_name, self.floattype))
        self.dtype = np.dtype(dtype)
        return

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
        super(HydmodObs, self).__init__()
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
        self._build_dtype()

        # build index
        self._build_index()

        self.data = None
        self._read_data()

    def _build_dtype(self):

        # create dtype
        dtype = [("totim", self.floattype)]
        for site in self.hydlbl:
            if not isinstance(site, str):
                site_name = site.decode().strip()
            else:
                site_name = site.strip()
            dtype.append((site_name, self.floattype))
        self.dtype = np.dtype(dtype)
        return

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
        super(SwrObs, self).__init__()
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
