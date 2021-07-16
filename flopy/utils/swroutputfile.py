import sys
import numpy as np
from collections import OrderedDict

from ..utils.utils_def import FlopyBinaryData


class SwrFile(FlopyBinaryData):
    """
    Read binary SWR output from MODFLOW SWR Process binary output files
    The SwrFile class is the super class from which specific derived
    classes are formed.  This class should not be instantiated directly

    Parameters
    ----------
    filename : string
        Name of the swr output file
    swrtype : str
        swr data type. Valid data types are 'stage', 'budget',
        'flow', 'exchange', or 'structure'. (default is 'stage')
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
    >>> so = flopy.utils.SwrFile('mymodel.swr.stage.bin')

    """

    def __init__(
        self, filename, swrtype="stage", precision="double", verbose=False
    ):
        """
        Class constructor.

        """
        super().__init__()
        self.set_float(precision=precision)
        self.header_dtype = np.dtype(
            [
                ("totim", self.floattype),
                ("kswr", "i4"),
                ("kstp", "i4"),
                ("kper", "i4"),
            ]
        )
        self._recordarray = []

        self.file = open(filename, "rb")
        self.types = ("stage", "budget", "flow", "exchange", "structure")
        if swrtype.lower() in self.types:
            self.type = swrtype.lower()
        else:
            err = (
                "SWR type ({}) is not defined. ".format(type)
                + "Available types are:\n"
            )
            for t in self.types:
                err = "{}  {}\n".format(err, t)
            raise Exception(err)

        # set data dtypes
        self._build_dtypes()

        # debug
        self.verbose = verbose

        # Read the dimension data
        self.flowitems = 0
        if self.type == "flow":
            self.flowitems = self.read_integer()
        self.nrecord = self.read_integer()

        # set-up
        self.items = len(self.out_dtype) - 1

        # read connectivity for velocity data if necessary
        self.conn_dtype = None
        if self.type == "flow":
            self.connectivity = self._read_connectivity()
            if self.verbose:
                print("Connectivity: ")
                print(self.connectivity)

        # initialize itemlist and nentries for qaq data
        self.nentries = {}

        self.datastart = self.file.tell()

        # build index
        self._build_index()

    def get_connectivity(self):
        """
        Get connectivity data from the file.

        Parameters
        ----------

        Returns
        ----------
        data : numpy array
            Array has size (nrecord, 3). None is returned if swrtype is not
            'flow'

        See Also
        --------

        Notes
        -----

        Examples
        --------

        """
        if self.type == "flow":
            return self.connectivity
        else:
            return None

    def get_nrecords(self):
        """
        Get the number of records in the file

        Returns
        ----------
        out : tuple of int
            A tupe with the number of records and number of flow items
            in the file. The number of flow items is non-zero only if
            swrtype='flow'.

        """
        return self.nrecord, self.flowitems

    def get_kswrkstpkper(self):
        """
        Get a list of unique stress periods, time steps, and swr time steps
        in the file

        Returns
        ----------
        out : list of (kswr, kstp, kper) tuples
            List of unique kswr, kstp, kper combinations in binary file.
            kswr, kstp, and kper values are zero-based.

        """
        return self._kswrkstpkper

    def get_ntimes(self):
        """
        Get the number of times in the file

        Returns
        ----------
        out : int
            The number of simulation times (totim) in binary file.

        """
        return self._ntimes

    def get_times(self):
        """
        Get a list of unique times in the file

        Returns
        ----------
        out : list of floats
            List contains unique simulation times (totim) in binary file.

        """
        return self._times.tolist()

    def get_record_names(self):
        """
        Get a list of unique record names in the file

        Returns
        ----------
        out : list of strings
            List of unique text names in the binary file.

        """
        return self.out_dtype.names

    def get_data(self, idx=None, kswrkstpkper=None, totim=None):
        """
        Get data from the file for the specified conditions.

        Parameters
        ----------
        idx : int
            The zero-based record number.  The first record is record 0.
            (default is None)
        kswrkstpkper : tuple of ints
            A tuple containing the swr time step, time step, and stress period
            (kswr, kstp, kper). These are zero-based kswr, kstp, and kper
            values. (default is None)
        totim : float
            The simulation time. (default is None)

        Returns
        ----------
        data : numpy record array
            Array has size (nitems).

        See Also
        --------

        Notes
        -----
        if both kswrkstpkper and totim are None, will return the last entry

        Examples
        --------

        """
        if kswrkstpkper is not None:
            kswr1 = kswrkstpkper[0]
            kstp1 = kswrkstpkper[1]
            kper1 = kswrkstpkper[2]

            totim1 = self._recordarray[
                np.where(
                    (self._recordarray["kswr"] == kswr1)
                    & (self._recordarray["kstp"] == kstp1)
                    & (self._recordarray["kper"] == kper1)
                )
            ]["totim"][0]
        elif totim is not None:
            totim1 = totim
        elif idx is not None:
            totim1 = self._recordarray["totim"][idx]
        else:
            totim1 = self._times[-1]

        try:
            ipos = self.recorddict[totim1]
            self.file.seek(ipos)
            if self.type == "exchange":
                self.nitems, self.itemlist = self.nentries[totim1]
                r = self._read_qaq()
            elif self.type == "structure":
                self.nitems, self.itemlist = self.nentries[totim1]
                r = self._read_structure()
            else:
                r = self.read_record(count=self.nrecord)

            # add totim to data record array
            s = np.zeros(r.shape[0], dtype=self.out_dtype)
            s["totim"] = totim1
            for name in r.dtype.names:
                s[name] = r[name]
            return s.view(dtype=self.out_dtype)
        except:
            return None

    def get_ts(self, irec=0, iconn=0, klay=0, istr=0):
        """
        Get a time series from a swr binary file.

        Parameters
        ----------
        irec : int
            is the zero-based reach (stage, qm, qaq) or reach group number
            (budget) to retrieve. (default is 0)
        iconn : int
            is the zero-based connection number for reach (irch) to retrieve
            qm data. iconn is only used if qm data is being read.
            (default is 0)
        klay : int
            is the zero-based layer number for reach (irch) to retrieve
            qaq data . klay is only used if qaq data is being read.
            (default is 0)
        klay : int
            is the zero-based structure number for reach (irch) to retrieve
            structure data . isrt is only used if structure data is being read.
            (default is 0)

        Returns
        ----------
        out : numpy recarray
            Array has size (ntimes, nitems).  The first column in the
            data array will contain time (totim). nitems is 2 for stage
            data, 15 for budget data, 3 for qm data, and 11 for qaq
            data.

        See Also
        --------

        Notes
        -----

        The irec, iconn, and klay values must be zero-based.

        Examples
        --------

        """

        if irec + 1 > self.nrecord:
            err = "Error: specified irec ({}) ".format(
                irec
            ) + "exceeds the total number of records ()".format(self.nrecord)
            raise Exception(err)

        gage_record = None
        if self.type == "stage" or self.type == "budget":
            gage_record = self._get_ts(irec=irec)
        elif self.type == "flow":
            gage_record = self._get_ts_qm(irec=irec, iconn=iconn)
        elif self.type == "exchange":
            gage_record = self._get_ts_qaq(irec=irec, klay=klay)
        elif self.type == "structure":
            gage_record = self._get_ts_structure(irec=irec, istr=istr)

        return gage_record

    def _read_connectivity(self):
        self.conn_dtype = np.dtype(
            [("reach", "i4"), ("from", "i4"), ("to", "i4")]
        )
        conn = np.zeros((self.nrecord, 3), int)
        icount = 0
        for nrg in range(self.flowitems):
            flowitems = self.read_integer()
            for ic in range(flowitems):
                conn[icount, 0] = nrg
                conn[icount, 1] = self.read_integer() - 1
                conn[icount, 2] = self.read_integer() - 1
                icount += 1
        return conn

    def _build_dtypes(self):
        self.vtotim = ("totim", self.floattype)
        if self.type == "stage":
            vtype = [("stage", self.floattype)]
        elif self.type == "budget":
            vtype = [
                ("stage", self.floattype),
                ("qsflow", self.floattype),
                ("qlatflow", self.floattype),
                ("quzflow", self.floattype),
                ("rain", self.floattype),
                ("evap", self.floattype),
                ("qbflow", self.floattype),
                ("qeflow", self.floattype),
                ("qexflow", self.floattype),
                ("qbcflow", self.floattype),
                ("qcrflow", self.floattype),
                ("dv", self.floattype),
                ("inf-out", self.floattype),
                ("volume", self.floattype),
            ]
        elif self.type == "flow":
            vtype = [("flow", self.floattype), ("velocity", self.floattype)]
        elif self.type == "exchange":
            vtype = [
                ("layer", "i4"),
                ("bottom", "f8"),
                ("stage", "f8"),
                ("depth", "f8"),
                ("head", "f8"),
                ("wetper", "f8"),
                ("cond", "f8"),
                ("headdiff", "f8"),
                ("exchange", "f8"),
            ]
        elif self.type == "structure":
            vtype = [
                ("usstage", "f8"),
                ("dsstage", "f8"),
                ("gateelev", "f8"),
                ("opening", "f8"),
                ("strflow", "f8"),
            ]
        self.dtype = np.dtype(vtype)
        temp = list(vtype)
        if self.type == "exchange":
            temp.insert(0, ("reach", "i4"))
            self.qaq_dtype = np.dtype(temp)
        elif self.type == "structure":
            temp.insert(0, ("structure", "i4"))
            temp.insert(0, ("reach", "i4"))
            self.str_dtype = np.dtype(temp)
        temp.insert(0, self.vtotim)
        self.out_dtype = np.dtype(temp)
        return

    def _read_header(self):
        nitems = 0
        if self.type == "exchange" or self.type == "structure":
            itemlist = np.zeros(self.nrecord, int)
            try:
                for i in range(self.nrecord):
                    itemlist[i] = self.read_integer()
                    nitems += itemlist[i]
                self.nitems = nitems
            except:
                if self.verbose:
                    sys.stdout.write("\nCould not read itemlist")
                return 0.0, 0.0, 0, 0, 0, False
        try:
            totim = self.read_real()
            dt = self.read_real()
            kper = self.read_integer() - 1
            kstp = self.read_integer() - 1
            kswr = self.read_integer() - 1
            if self.type == "exchange" or self.type == "structure":
                self.nentries[totim] = (nitems, itemlist)
            return totim, dt, kper, kstp, kswr, True
        except:
            return 0.0, 0.0, 0, 0, 0, False

    def _get_ts(self, irec=0):

        # create array
        gage_record = np.zeros(self._ntimes, dtype=self.out_dtype)

        # iterate through the record dictionary
        idx = 0
        for key, value in self.recorddict.items():
            totim = np.array(key)
            gage_record["totim"][idx] = totim

            self.file.seek(value)
            r = self._get_data()
            for name in r.dtype.names:
                gage_record[name][idx] = r[name][irec]
            idx += 1

        return gage_record.view(dtype=self.out_dtype)

    def _get_ts_qm(self, irec=0, iconn=0):

        # create array
        gage_record = np.zeros(self._ntimes, dtype=self.out_dtype)

        # iterate through the record dictionary
        idx = 0
        for key, value in self.recorddict.items():
            totim = key
            gage_record["totim"][idx] = totim

            self.file.seek(value)
            r = self._get_data()

            # find correct entry for reach and connection
            for i in range(self.nrecord):
                inode = self.connectivity[i, 1]
                ic = self.connectivity[i, 2]
                if irec == inode and ic == iconn:
                    for name in r.dtype.names:
                        gage_record[name][idx] = r[name][i]
                    break
            idx += 1

        return gage_record.view(dtype=self.out_dtype)

    def _get_ts_qaq(self, irec=0, klay=0):

        # create array
        gage_record = np.zeros(self._ntimes, dtype=self.out_dtype)

        # iterate through the record dictionary
        idx = 0
        for key, value in self.recorddict.items():
            totim = key
            gage_record["totim"][idx] = totim

            self.nitems, self.itemlist = self.nentries[key]

            self.file.seek(value)
            r = self._get_data()

            # find correct entry for record and layer
            ilen = np.shape(r)[0]
            for i in range(ilen):
                ir = r["reach"][i]
                il = r["layer"][i]
                if ir == irec and il == klay:
                    for name in r.dtype.names:
                        gage_record[name][idx] = r[name][i]
                    break
            idx += 1

        return gage_record.view(dtype=self.out_dtype)

    def _get_ts_structure(self, irec=0, istr=0):

        # create array
        gage_record = np.zeros(self._ntimes, dtype=self.out_dtype)

        # iterate through the record dictionary
        idx = 0
        for key, value in self.recorddict.items():
            totim = key
            gage_record["totim"][idx] = totim

            self.nitems, self.itemlist = self.nentries[key]

            self.file.seek(value)
            r = self._get_data()

            # find correct entry for record and structure number
            ilen = np.shape(r)[0]
            for i in range(ilen):
                ir = r["reach"][i]
                il = r["structure"][i]
                if ir == irec and il == istr:
                    for name in r.dtype.names:
                        gage_record[name][idx] = r[name][i]
                    break
            idx += 1

        return gage_record.view(dtype=self.out_dtype)

    def _get_data(self):
        if self.type == "exchange":
            return self._read_qaq()
        elif self.type == "structure":
            return self._read_structure()
        else:
            return self.read_record(count=self.nrecord)

    def _read_qaq(self):

        # read qaq data using standard record reader
        bd = self.read_record(count=self.nitems)
        bd["layer"] -= 1

        # add reach number to qaq data
        r = np.zeros(self.nitems, dtype=self.qaq_dtype)

        # build array with reach numbers
        reaches = np.zeros(self.nitems, dtype=np.int32)
        idx = 0
        for irch in range(self.nrecord):
            klay = self.itemlist[irch]
            for k in range(klay):
                # r[idx, 0] = irch
                reaches[idx] = irch
                idx += 1

        # add reach to array returned
        r["reach"] = reaches.copy()

        # add read data to array returned
        for idx, k in enumerate(self.dtype.names):
            r[k] = bd[k]
        return r

    def _read_structure(self):

        # read qaq data using standard record reader
        bd = self.read_record(count=self.nitems)

        # add reach and structure number to structure data
        r = np.zeros(self.nitems, dtype=self.str_dtype)

        # build array with reach numbers
        reaches = np.zeros(self.nitems, dtype=np.int32)
        struct = np.zeros(self.nitems, dtype=np.int32)
        idx = 0
        for irch in range(self.nrecord):
            nstr = self.itemlist[irch]
            for n in range(nstr):
                reaches[idx] = irch
                struct[idx] = n
                idx += 1

        # add reach to array returned
        r["reach"] = reaches.copy()
        r["structure"] = struct.copy()

        # add read data to array returned
        for idx, k in enumerate(self.dtype.names):
            r[k] = bd[k]
        return r

    def _build_index(self):
        """
        Build the recordarray recarray and recorddict dictionary, which map
        the header information to the position in the binary file.
        """
        self.file.seek(self.datastart)
        if self.verbose:
            sys.stdout.write("Generating SWR binary data time list\n")
        self._ntimes = 0
        self._times = []
        self._kswrkstpkper = []
        self.recorddict = OrderedDict()

        idx = 0
        while True:
            # --output something to screen so it is possible to determine
            #  that the time list is being created
            idx += 1
            if self.verbose:
                v = divmod(float(idx), 72.0)
                if v[1] == 0.0:
                    sys.stdout.write(".")
            # read header
            totim, dt, kper, kstp, kswr, success = self._read_header()
            if success:
                if self.type == "exchange":
                    bytes = self.nitems * (
                        self.integerbyte + 8 * self.realbyte
                    )
                elif self.type == "structure":
                    bytes = self.nitems * (5 * self.realbyte)
                else:
                    bytes = self.nrecord * self.items * self.realbyte
                ipos = self.file.tell()
                self.file.seek(bytes, 1)
                # save data
                self._ntimes += 1
                self._times.append(totim)
                self._kswrkstpkper.append((kswr, kstp, kper))
                header = (totim, kswr, kstp, kper)
                self.recorddict[totim] = ipos
                self._recordarray.append(header)
            else:
                if self.verbose:
                    sys.stdout.write("\n")
                self._recordarray = np.array(
                    self._recordarray, dtype=self.header_dtype
                )
                self._times = np.array(self._times)
                self._kswrkstpkper = np.array(self._kswrkstpkper)
                return


class SwrStage(SwrFile):
    """
    Read binary SWR stage output from MODFLOW SWR Process binary output files

    Parameters
    ----------
    filename : string
        Name of the swr stage output file
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
    >>> stageobj = flopy.utils.SwrStage('mymodel.swr.stg')

    """

    def __init__(self, filename, precision="double", verbose=False):
        super().__init__(
            filename, swrtype="stage", precision=precision, verbose=verbose
        )
        return


class SwrBudget(SwrFile):
    """
    Read binary SWR budget output from MODFLOW SWR Process binary output files

    Parameters
    ----------
    filename : string
        Name of the swr budget output file
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
    >>> stageobj = flopy.utils.SwrStage('mymodel.swr.bud')

    """

    def __init__(self, filename, precision="double", verbose=False):
        super().__init__(
            filename, swrtype="budget", precision=precision, verbose=verbose
        )
        return


class SwrFlow(SwrFile):
    """
    Read binary SWR flow output from MODFLOW SWR Process binary output files

    Parameters
    ----------
    filename : string
        Name of the swr flow output file
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
    >>> stageobj = flopy.utils.SwrStage('mymodel.swr.flow')

    """

    def __init__(self, filename, precision="double", verbose=False):
        super().__init__(
            filename, swrtype="flow", precision=precision, verbose=verbose
        )
        return


class SwrExchange(SwrFile):
    """
    Read binary SWR surface-water groundwater exchange output from MODFLOW SWR Process binary output files

    Parameters
    ----------
    filename : string
        Name of the swr surface-water groundwater exchange output file
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
    >>> stageobj = flopy.utils.SwrStage('mymodel.swr.qaq')

    """

    def __init__(self, filename, precision="double", verbose=False):
        super().__init__(
            filename, swrtype="exchange", precision=precision, verbose=verbose
        )
        return


class SwrStructure(SwrFile):
    """
    Read binary SWR structure output from MODFLOW SWR Process binary output
    files

    Parameters
    ----------
    filename : string
        Name of the swr structure output file
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
    >>> stageobj = flopy.utils.SwrStage('mymodel.swr.str')

    """

    def __init__(self, filename, precision="double", verbose=False):
        super().__init__(
            filename, swrtype="structure", precision=precision, verbose=verbose
        )
        return
