import os
import copy
import numpy as np
from .binaryfile import CellBudgetFile
from itertools import groupby
from collections import OrderedDict
from ..utils.utils_def import totim_to_datetime


class ZoneBudget(object):
    """
    ZoneBudget class

    Parameters
    ----------
    cbc_file : str or CellBudgetFile object
        The file name or CellBudgetFile object for which budgets will be
        computed.
    z : ndarray
        The array containing to zones to be used.
    kstpkper : tuple of ints
        A tuple containing the time step and stress period (kstp, kper).
        The kstp and kper values are zero based.
    totim : float
        The simulation time.
    aliases : dict
        A dictionary with key, value pairs of zones and aliases. Replaces
        the corresponding record and field names with the aliases provided.
        When using this option in conjunction with a list of zones, the
        zone(s) passed may either be all strings (aliases), all integers,
        or mixed.

    Returns
    -------
    None

    Examples
    --------

    >>> from flopy.utils.zonbud import ZoneBudget, read_zbarray
    >>> zon = read_zbarray('zone_input_file')
    >>> zb = ZoneBudget('zonebudtest.cbc', zon, kstpkper=(0, 0))
    >>> zb.to_csv('zonebudtest.csv')
    >>> zb_mgd = zb * 7.48052 / 1000000
    """

    def __init__(
        self,
        cbc_file,
        z,
        kstpkper=None,
        totim=None,
        aliases=None,
        verbose=False,
        **kwargs
    ):

        if isinstance(cbc_file, CellBudgetFile):
            self.cbc = cbc_file
        elif isinstance(cbc_file, str) and os.path.isfile(cbc_file):
            self.cbc = CellBudgetFile(cbc_file)
        else:
            raise Exception(
                "Cannot load cell budget file: {}.".format(cbc_file)
            )

        if isinstance(z, np.ndarray):
            assert np.issubdtype(
                z.dtype, np.integer
            ), "Zones dtype must be integer"
        else:
            e = (
                "Please pass zones as a numpy ndarray of (positive)"
                " integers. {}".format(z.dtype)
            )
            raise Exception(e)

        # Check for negative zone values
        if np.any(z < 0):
            raise Exception(
                "Negative zone value(s) found:", np.unique(z[z < 0])
            )

        self.dis = None
        self.sr = None
        if "model" in kwargs.keys():
            self.model = kwargs.pop("model")
            self.sr = self.model.sr
            self.dis = self.model.dis
        if "dis" in kwargs.keys():
            self.dis = kwargs.pop("dis")
            self.sr = self.dis.parent.sr
        if "sr" in kwargs.keys():
            self.sr = kwargs.pop("sr")
        if len(kwargs.keys()) > 0:
            args = ",".join(kwargs.keys())
            raise Exception("LayerFile error: unrecognized kwargs: " + args)

        # Check the shape of the cbc budget file arrays
        self.cbc_shape = self.cbc.get_data(idx=0, full3D=True)[0].shape
        self.nlay, self.nrow, self.ncol = self.cbc_shape
        self.cbc_times = self.cbc.get_times()
        self.cbc_kstpkper = self.cbc.get_kstpkper()
        self.kstpkper = None
        self.totim = None

        if kstpkper is not None:
            if isinstance(kstpkper, tuple):
                kstpkper = [kstpkper]
            for kk in kstpkper:
                s = (
                    "The specified time step/stress period "
                    "does not exist {}".format(kk)
                )
                assert kk in self.cbc.get_kstpkper(), s
            self.kstpkper = kstpkper
        elif totim is not None:
            if isinstance(totim, float):
                totim = [totim]
            elif isinstance(totim, int):
                totim = [float(totim)]
            for t in totim:
                s = (
                    "The specified simulation time "
                    "does not exist {}".format(t)
                )
                assert t in self.cbc.get_times(), s
            self.totim = totim
        else:
            # No time step/stress period or simulation time pass
            self.kstpkper = self.cbc.get_kstpkper()

        # Set float and integer types
        self.float_type = np.float32
        self.int_type = np.int32

        # Check dimensions of input zone array
        s = (
            "Row/col dimensions of zone array {}"
            " do not match model row/col dimensions {}".format(
                z.shape, self.cbc_shape
            )
        )
        assert z.shape[-2] == self.nrow and z.shape[-1] == self.ncol, s

        if z.shape == self.cbc_shape:
            izone = z.copy()
        elif len(z.shape) == 2:
            izone = np.zeros(self.cbc_shape, self.int_type)
            izone[:] = z[:, :]
        elif len(z.shape) == 3 and z.shape[0] == 1:
            izone = np.zeros(self.cbc_shape, self.int_type)
            izone[:] = z[0, :, :]
        else:
            e = "Shape of the zone array is not recognized: {}".format(z.shape)
            raise Exception(e)

        self.izone = izone
        self.allzones = np.unique(izone)
        self._zonenamedict = OrderedDict(
            [(z, "ZONE_{}".format(z)) for z in self.allzones]
        )

        if aliases is not None:
            s = (
                "Input aliases not recognized. Please pass a dictionary "
                "with key,value pairs of zone/alias."
            )
            assert isinstance(aliases, dict), s
            # Replace the relevant field names (ignore zone 0)
            seen = []
            for z, a in iter(aliases.items()):
                if z != 0 and z in self._zonenamedict.keys():
                    if z in seen:
                        raise Exception(
                            "Zones may not have more than 1 alias."
                        )
                    self._zonenamedict[z] = "_".join(a.split())
                    seen.append(z)

        # self._iflow_recnames = self._get_internal_flow_record_names()

        # All record names in the cell-by-cell budget binary file
        self.record_names = [
            n.strip() for n in self.cbc.get_unique_record_names(decode=True)
        ]

        # Get imeth for each record in the CellBudgetFile record list
        self.imeth = {}
        for record in self.cbc.recordarray:
            self.imeth[record["text"].strip().decode("utf-8")] = record[
                "imeth"
            ]

        # INTERNAL FLOW TERMS ARE USED TO CALCULATE FLOW BETWEEN ZONES.
        # CONSTANT-HEAD TERMS ARE USED TO IDENTIFY WHERE CONSTANT-HEAD CELLS
        # ARE AND THEN USE FACE FLOWS TO DETERMINE THE AMOUNT OF FLOW.
        # SWIADDTO--- terms are used by the SWI2 groundwater flow process.
        internal_flow_terms = [
            "CONSTANT HEAD",
            "FLOW RIGHT FACE",
            "FLOW FRONT FACE",
            "FLOW LOWER FACE",
            "SWIADDTOCH",
            "SWIADDTOFRF",
            "SWIADDTOFFF",
            "SWIADDTOFLF",
        ]

        # Source/sink/storage term record names
        # These are all of the terms that are not related to constant
        # head cells or face flow terms
        self.ssst_record_names = [
            n for n in self.record_names if n not in internal_flow_terms
        ]

        # Initialize budget recordarray
        array_list = []
        if self.kstpkper is not None:
            for kk in self.kstpkper:
                recordarray = self._initialize_budget_recordarray(
                    kstpkper=kk, totim=None
                )
                array_list.append(recordarray)
        elif self.totim is not None:
            for t in self.totim:
                recordarray = self._initialize_budget_recordarray(
                    kstpkper=None, totim=t
                )
                array_list.append(recordarray)
        self._budget = np.concatenate(array_list, axis=0)

        # Update budget record array
        if self.kstpkper is not None:
            for kk in self.kstpkper:
                if verbose:
                    s = (
                        "Computing the budget for"
                        " time step {} in stress period {}".format(
                            kk[0] + 1, kk[1] + 1
                        )
                    )
                    print(s)
                self._compute_budget(kstpkper=kk)
        elif self.totim is not None:
            for t in self.totim:
                if verbose:
                    s = "Computing the budget for time {}".format(t)
                    print(s)
                self._compute_budget(totim=t)

        return

    def get_model_shape(self):
        """Get model shape

        Returns
        -------
        nlay : int
            Number of layers
        nrow : int
            Number of rows
        ncol : int
            Number of columns

        """
        return self.nlay, self.nrow, self.ncol

    def get_record_names(self, stripped=False):
        """
        Get a list of water budget record names in the file.

        Returns
        -------
        out : list of strings
            List of unique text names in the binary file.

        Examples
        --------

        >>> zb = ZoneBudget('zonebudtest.cbc', zon, kstpkper=(0, 0))
        >>> recnames = zb.get_record_names()

        """
        if not stripped:
            return np.unique(self._budget["name"])
        else:
            seen = []
            for recname in self.get_record_names():
                if recname in ["IN-OUT", "TOTAL_IN", "TOTAL_OUT"]:
                    continue
                if recname.endswith("_IN"):
                    recname = recname[:-3]
                elif recname.endswith("_OUT"):
                    recname = recname[:-4]
                if recname not in seen:
                    seen.append(recname)
            seen.extend(["IN-OUT", "TOTAL"])
            return np.array(seen)

    def get_budget(self, names=None, zones=None, net=False):
        """
        Get a list of zonebudget record arrays.

        Parameters
        ----------

        names : list of strings
            A list of strings containing the names of the records desired.
        zones : list of ints or strings
            A list of integer zone numbers or zone names desired.
        net : boolean
            If True, returns net IN-OUT for each record.

        Returns
        -------
        budget_list : list of record arrays
            A list of the zonebudget record arrays.

        Examples
        --------

        >>> names = ['FROM_CONSTANT_HEAD', 'RIVER_LEAKAGE_OUT']
        >>> zones = ['ZONE_1', 'ZONE_2']
        >>> zb = ZoneBudget('zonebudtest.cbc', zon, kstpkper=(0, 0))
        >>> bud = zb.get_budget(names=names, zones=zones)

        """
        if isinstance(names, str):
            names = [names]
        if isinstance(zones, str):
            zones = [zones]
        elif isinstance(zones, int):
            zones = [zones]
        select_fields = ["totim", "time_step", "stress_period", "name"] + list(
            self._zonenamedict.values()
        )
        select_records = np.where(
            (self._budget["name"] == self._budget["name"])
        )
        if zones is not None:
            for idx, z in enumerate(zones):
                if isinstance(z, int):
                    zones[idx] = self._zonenamedict[z]
            select_fields = [
                "totim",
                "time_step",
                "stress_period",
                "name",
            ] + zones
        if names is not None:
            names = self._clean_budget_names(names)
            select_records = np.in1d(self._budget["name"], names)
        if net:
            if names is None:
                names = self._clean_budget_names(self.get_record_names())
            net_budget = self._compute_net_budget()
            seen = []
            net_names = []
            for name in names:
                iname = "_".join(name.split("_")[1:])
                if iname not in seen:
                    seen.append(iname)
                else:
                    net_names.append(iname)
            select_records = np.in1d(net_budget["name"], net_names)
            return net_budget[select_fields][select_records]
        else:
            return self._budget[select_fields][select_records]

    def to_csv(self, fname):
        """
        Saves the budget record arrays to a formatted
        comma-separated values file.

        Parameters
        ----------
        fname : str
            The name of the output comma-separated values file.

        Returns
        -------
        None

        """
        # Needs updating to handle the new budget list structure. Write out
        # budgets for all kstpkper if kstpkper is None or pass list of
        # kstpkper/totim to save particular budgets.
        with open(fname, "w") as f:
            # Write header
            f.write(",".join(self._budget.dtype.names) + "\n")
            # Write rows
            for rowidx in range(self._budget.shape[0]):
                s = (
                    ",".join([str(i) for i in list(self._budget[:][rowidx])])
                    + "\n"
                )
                f.write(s)
        return

    def get_dataframes(
        self,
        start_datetime=None,
        timeunit="D",
        index_key="totim",
        names=None,
        zones=None,
        net=False,
    ):
        """
        Get pandas dataframes.

        Parameters
        ----------

        start_datetime : str
            Datetime string indicating the time at which the simulation starts.
        timeunit : str
            String that indicates the time units used in the model.
        index_key : str
            Indicates the fields to be used (in addition to "record") in the
            resulting DataFrame multi-index.
        names : list of strings
            A list of strings containing the names of the records desired.
        zones : list of ints or strings
            A list of integer zone numbers or zone names desired.
        net : boolean
            If True, returns net IN-OUT for each record.

        Returns
        -------
        df : Pandas DataFrame
            Pandas DataFrame with the budget information.

        Examples
        --------
        >>> from flopy.utils.zonbud import ZoneBudget, read_zbarray
        >>> zon = read_zbarray('zone_input_file')
        >>> zb = ZoneBudget('zonebudtest.cbc', zon, kstpkper=(0, 0))
        >>> df = zb.get_dataframes()

        """
        try:
            import pandas as pd
        except Exception as e:
            msg = "ZoneBudget.get_dataframes() error import pandas: " + str(e)
            raise ImportError(msg)

        valid_index_keys = ["totim", "kstpkper"]
        s = 'index_key "{}" is not valid.'.format(index_key)
        assert index_key in valid_index_keys, s

        valid_timeunit = ["S", "M", "H", "D", "Y"]

        if timeunit.upper() == "SECONDS":
            timeunit = "S"
        elif timeunit.upper() == "MINUTES":
            timeunit = "M"
        elif timeunit.upper() == "HOURS":
            timeunit = "H"
        elif timeunit.upper() == "DAYS":
            timeunit = "D"
        elif timeunit.upper() == "YEARS":
            timeunit = "Y"

        errmsg = (
            "Specified time units ({}) not recognized. "
            "Please use one of ".format(timeunit)
        )
        assert timeunit in valid_timeunit, (
            errmsg + ", ".join(valid_timeunit) + "."
        )

        df = pd.DataFrame().from_records(self.get_budget(names, zones, net))
        if start_datetime is not None:
            totim = totim_to_datetime(
                df.totim,
                start=pd.to_datetime(start_datetime),
                timeunit=timeunit,
            )
            df["datetime"] = totim
            index_cols = ["datetime", "name"]
        else:
            if index_key == "totim":
                index_cols = ["totim", "name"]
            elif index_key == "kstpkper":
                index_cols = ["time_step", "stress_period", "name"]
        df = df.set_index(index_cols)  # .sort_index(level=0)
        if zones is not None:
            keep_cols = zones
        else:
            keep_cols = self._zonenamedict.values()
        return df.loc[:, keep_cols]

    def copy(self):
        """
        Return a deepcopy of the object.
        """
        return copy.deepcopy(self)

    def __deepcopy__(self, memo):
        """
        Over-rides the default deepcopy behavior. Copy all attributes except
        the CellBudgetFile object which does not copy nicely.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        ignore_attrs = ["cbc"]
        for k, v in self.__dict__.items():
            if k not in ignore_attrs:
                setattr(result, k, copy.deepcopy(v, memo))

        # Set CellBudgetFile object attribute manually. This is object
        # read-only so should not be problems with pointers from
        # multiple objects.
        result.cbc = self.cbc
        return result

    def _compute_budget(self, kstpkper=None, totim=None):
        """
        Creates a budget for the specified zone array. This function only
        supports the use of a single time step/stress period or time.

        Parameters
        ----------
        kstpkper : tuple
            Tuple of kstp and kper to compute budget for (default is None).
        totim : float
            Totim to compute budget for (default is None).

        Returns
        -------
        None

        """
        # Initialize an array to track where the constant head cells
        # are located.
        ich = np.zeros(self.cbc_shape, self.int_type)
        swiich = np.zeros(self.cbc_shape, self.int_type)

        if "CONSTANT HEAD" in self.record_names:
            """
            C-----CONSTANT-HEAD FLOW -- DON'T ACCUMULATE THE CELL-BY-CELL VALUES FOR
            C-----CONSTANT-HEAD FLOW BECAUSE THEY MAY INCLUDE PARTIALLY CANCELING
            C-----INS AND OUTS.  USE CONSTANT-HEAD TERM TO IDENTIFY WHERE CONSTANT-
            C-----HEAD CELLS ARE AND THEN USE FACE FLOWS TO DETERMINE THE AMOUNT OF
            C-----FLOW.  STORE CONSTANT-HEAD LOCATIONS IN ICH ARRAY.
            """
            chd = self.cbc.get_data(
                text="CONSTANT HEAD",
                full3D=True,
                kstpkper=kstpkper,
                totim=totim,
            )[0]
            ich[np.ma.where(chd != 0.0)] = 1
        if "FLOW RIGHT FACE" in self.record_names:
            self._accumulate_flow_frf("FLOW RIGHT FACE", ich, kstpkper, totim)
        if "FLOW FRONT FACE" in self.record_names:
            self._accumulate_flow_fff("FLOW FRONT FACE", ich, kstpkper, totim)
        if "FLOW LOWER FACE" in self.record_names:
            self._accumulate_flow_flf("FLOW LOWER FACE", ich, kstpkper, totim)
        if "SWIADDTOCH" in self.record_names:
            swichd = self.cbc.get_data(
                text="SWIADDTOCH", full3D=True, kstpkper=kstpkper, totim=totim
            )[0]
            swiich[swichd != 0] = 1
        if "SWIADDTOFRF" in self.record_names:
            self._accumulate_flow_frf("SWIADDTOFRF", swiich, kstpkper, totim)
        if "SWIADDTOFFF" in self.record_names:
            self._accumulate_flow_fff("SWIADDTOFFF", swiich, kstpkper, totim)
        if "SWIADDTOFLF" in self.record_names:
            self._accumulate_flow_flf("SWIADDTOFLF", swiich, kstpkper, totim)

        # NOT AN INTERNAL FLOW TERM, SO MUST BE A SOURCE TERM OR STORAGE
        # ACCUMULATE THE FLOW BY ZONE
        # iterate over remaining items in the list
        for recname in self.ssst_record_names:
            self._accumulate_flow_ssst(recname, kstpkper, totim)

        # Compute mass balance terms
        self._compute_mass_balance(kstpkper, totim)

        return

    def _add_empty_record(
        self, recordarray, recname, kstpkper=None, totim=None
    ):
        """
        Build an empty records based on the specified flow direction and
        record name for the given list of zones.

        Parameters
        ----------
        recordarray :
        recname :
        kstpkper : tuple
            Tuple of kstp and kper to compute budget for (default is None).
        totim : float
            Totim to compute budget for (default is None).

        Returns
        -------
        recordarray : np.recarray

        """
        if kstpkper is not None:
            if len(self.cbc_times) > 0:
                totim = self.cbc_times[self.cbc_kstpkper.index(kstpkper)]
            else:
                totim = 0.0
        elif totim is not None:
            if len(self.cbc_times) > 0:
                kstpkper = self.cbc_kstpkper[self.cbc_times.index(totim)]
            else:
                kstpkper = (0, 0)

        row = [totim, kstpkper[0], kstpkper[1], recname]
        row += [0.0 for _ in self._zonenamedict.values()]
        recs = np.array(tuple(row), dtype=recordarray.dtype)
        recordarray = np.append(recordarray, recs)
        return recordarray

    def _initialize_budget_recordarray(self, kstpkper=None, totim=None):
        """
        Initialize the budget record array which will store all of the
        fluxes in the cell-budget file.

        Parameters
        ----------
        kstpkper : tuple
            Tuple of kstp and kper to compute budget for (default is None).
        totim : float
            Totim to compute budget for (default is None).

        Returns
        -------

        """

        # Create empty array for the budget terms.
        dtype_list = [
            ("totim", "<f4"),
            ("time_step", "<i4"),
            ("stress_period", "<i4"),
            ("name", (str, 50)),
        ]
        dtype_list += [
            (n, self.float_type) for n in self._zonenamedict.values()
        ]
        dtype = np.dtype(dtype_list)
        recordarray = np.array([], dtype=dtype)

        # Add "from" records
        if "STORAGE" in self.record_names:
            recordarray = self._add_empty_record(
                recordarray, "FROM_STORAGE", kstpkper, totim
            )
        if "CONSTANT HEAD" in self.record_names:
            recordarray = self._add_empty_record(
                recordarray, "FROM_CONSTANT_HEAD", kstpkper, totim
            )
        for recname in self.ssst_record_names:
            if recname != "STORAGE":
                recordarray = self._add_empty_record(
                    recordarray,
                    "FROM_" + "_".join(recname.split()),
                    kstpkper,
                    totim,
                )

        for z, n in self._zonenamedict.items():
            if z == 0 and 0 not in self.allzones:
                continue
            else:
                recordarray = self._add_empty_record(
                    recordarray, "FROM_" + "_".join(n.split()), kstpkper, totim
                )
        recordarray = self._add_empty_record(
            recordarray, "TOTAL_IN", kstpkper, totim
        )

        # Add "out" records
        if "STORAGE" in self.record_names:
            recordarray = self._add_empty_record(
                recordarray, "TO_STORAGE", kstpkper, totim
            )
        if "CONSTANT HEAD" in self.record_names:
            recordarray = self._add_empty_record(
                recordarray, "TO_CONSTANT_HEAD", kstpkper, totim
            )
        for recname in self.ssst_record_names:
            if recname != "STORAGE":
                recordarray = self._add_empty_record(
                    recordarray,
                    "TO_" + "_".join(recname.split()),
                    kstpkper,
                    totim,
                )

        for z, n in self._zonenamedict.items():
            if z == 0 and 0 not in self.allzones:
                continue
            else:
                recordarray = self._add_empty_record(
                    recordarray, "TO_" + "_".join(n.split()), kstpkper, totim
                )
        recordarray = self._add_empty_record(
            recordarray, "TOTAL_OUT", kstpkper, totim
        )

        recordarray = self._add_empty_record(
            recordarray, "IN-OUT", kstpkper, totim
        )
        recordarray = self._add_empty_record(
            recordarray, "PERCENT_DISCREPANCY", kstpkper, totim
        )
        return recordarray

    @staticmethod
    def _filter_circular_flow(fz, tz, f):
        """

        Parameters
        ----------
        fz
        tz
        f

        Returns
        -------

        """
        e = np.equal(fz, tz)
        fz = fz[np.logical_not(e)]
        tz = tz[np.logical_not(e)]
        f = f[np.logical_not(e)]
        return fz, tz, f

    def _update_budget_fromfaceflow(
        self, fz, tz, f, kstpkper=None, totim=None
    ):
        """

        Parameters
        ----------
        fz
        tz
        f
        kstpkper
        totim

        Returns
        -------

        """

        # No circular flow within zones
        fz, tz, f = self._filter_circular_flow(fz, tz, f)

        if len(f) == 0:
            return

        # Inflows
        idx = tz != 0
        fzi = fz[idx]
        tzi = tz[idx]
        rownames = ["FROM_" + self._zonenamedict[z] for z in fzi]
        colnames = [self._zonenamedict[z] for z in tzi]
        fluxes = f[idx]
        self._update_budget_recordarray(
            rownames, colnames, fluxes, kstpkper, totim
        )

        # Outflows
        idx = fz != 0
        fzi = fz[idx]
        tzi = tz[idx]
        rownames = ["TO_" + self._zonenamedict[z] for z in tzi]
        colnames = [self._zonenamedict[z] for z in fzi]
        fluxes = f[idx]
        self._update_budget_recordarray(
            rownames, colnames, fluxes, kstpkper, totim
        )
        return

    def _update_budget_fromssst(self, fz, tz, f, kstpkper=None, totim=None):
        """

        Parameters
        ----------
        fz
        tz
        f
        kstpkper
        totim

        Returns
        -------

        """
        if len(f) == 0:
            return
        self._update_budget_recordarray(fz, tz, f, kstpkper, totim)
        return

    def _update_budget_recordarray(
        self, rownames, colnames, fluxes, kstpkper=None, totim=None
    ):
        """
        Update the budget record array with the flux for the specified
        flow direction (in/out), record name, and column.

        Parameters
        ----------
        rownames
        colnames
        fluxes
        kstpkper
        totim

        Returns
        -------
        None

        """
        try:

            if kstpkper is not None:
                for rn, cn, flux in zip(rownames, colnames, fluxes):
                    rowidx = np.where(
                        (self._budget["time_step"] == kstpkper[0])
                        & (self._budget["stress_period"] == kstpkper[1])
                        & (self._budget["name"] == rn)
                    )
                    self._budget[cn][rowidx] += flux
            elif totim is not None:
                for rn, cn, flux in zip(rownames, colnames, fluxes):
                    rowidx = np.where(
                        (self._budget["totim"] == totim)
                        & (self._budget["name"] == rn)
                    )
                    self._budget[cn][rowidx] += flux

        except Exception as e:
            print(e)
            raise
        return

    def _accumulate_flow_frf(self, recname, ich, kstpkper, totim):
        """

        Parameters
        ----------
        recname
        ich
        kstpkper
        totim

        Returns
        -------

        """
        try:
            if self.ncol >= 2:
                data = self.cbc.get_data(
                    text=recname, kstpkper=kstpkper, totim=totim
                )[0]

                # "FLOW RIGHT FACE"  COMPUTE FLOW BETWEEN ZONES ACROSS COLUMNS.
                # COMPUTE FLOW ONLY BETWEEN A ZONE AND A HIGHER ZONE -- FLOW FROM
                # ZONE 4 TO 3 IS THE NEGATIVE OF FLOW FROM 3 TO 4.
                # 1ST, CALCULATE FLOW BETWEEN NODE J,I,K AND J-1,I,K

                k, i, j = np.where(
                    self.izone[:, :, 1:] > self.izone[:, :, :-1]
                )

                # Adjust column values to account for the starting position of "nz"
                j += 1

                # Define the zone to which flow is going
                nz = self.izone[k, i, j]

                # Define the zone from which flow is coming
                jl = j - 1
                nzl = self.izone[k, i, jl]

                # Get the face flow
                q = data[k, i, jl]

                # Get indices where flow face values are positive (flow out of higher zone)
                # Don't include CH to CH flow (can occur if CHTOCH option is used)
                # Create an iterable tuple of (from zone, to zone, flux)
                # Then group tuple by (from_zone, to_zone) and sum the flux values
                idx = np.where(
                    (q > 0) & ((ich[k, i, j] != 1) | (ich[k, i, jl] != 1))
                )
                fzi, tzi, fi = sum_flux_tuples(nzl[idx], nz[idx], q[idx])
                self._update_budget_fromfaceflow(
                    fzi, tzi, np.abs(fi), kstpkper, totim
                )

                # Get indices where flow face values are negative (flow into higher zone)
                # Don't include CH to CH flow (can occur if CHTOCH option is used)
                # Create an iterable tuple of (from zone, to zone, flux)
                # Then group tuple by (from_zone, to_zone) and sum the flux values
                idx = np.where(
                    (q < 0) & ((ich[k, i, j] != 1) | (ich[k, i, jl] != 1))
                )
                fzi, tzi, fi = sum_flux_tuples(nz[idx], nzl[idx], q[idx])
                self._update_budget_fromfaceflow(
                    fzi, tzi, np.abs(fi), kstpkper, totim
                )

                # FLOW BETWEEN NODE J,I,K AND J+1,I,K
                k, i, j = np.where(
                    self.izone[:, :, :-1] > self.izone[:, :, 1:]
                )

                # Define the zone from which flow is coming
                nz = self.izone[k, i, j]

                # Define the zone to which flow is going
                jr = j + 1
                nzr = self.izone[k, i, jr]

                # Get the face flow
                q = data[k, i, j]

                # Get indices where flow face values are positive (flow out of higher zone)
                # Don't include CH to CH flow (can occur if CHTOCH option is used)
                # Create an iterable tuple of (from zone, to zone, flux)
                # Then group tuple by (from_zone, to_zone) and sum the flux values
                idx = np.where(
                    (q > 0) & ((ich[k, i, j] != 1) | (ich[k, i, jr] != 1))
                )
                fzi, tzi, fi = sum_flux_tuples(nz[idx], nzr[idx], q[idx])
                self._update_budget_fromfaceflow(
                    fzi, tzi, np.abs(fi), kstpkper, totim
                )

                # Get indices where flow face values are negative (flow into higher zone)
                # Don't include CH to CH flow (can occur if CHTOCH option is used)
                # Create an iterable tuple of (from zone, to zone, flux)
                # Then group tuple by (from_zone, to_zone) and sum the flux values
                idx = np.where(
                    (q < 0) & ((ich[k, i, j] != 1) | (ich[k, i, jr] != 1))
                )
                fzi, tzi, fi = sum_flux_tuples(nzr[idx], nz[idx], q[idx])
                self._update_budget_fromfaceflow(
                    fzi, tzi, np.abs(fi), kstpkper, totim
                )

                # CALCULATE FLOW TO CONSTANT-HEAD CELLS IN THIS DIRECTION
                k, i, j = np.where(ich == 1)
                k, i, j = k[j > 0], i[j > 0], j[j > 0]
                jl = j - 1
                nzl = self.izone[k, i, jl]
                nz = self.izone[k, i, j]
                q = data[k, i, jl]
                idx = np.where(
                    (q > 0) & ((ich[k, i, j] != 1) | (ich[k, i, jl] != 1))
                )
                fzi, tzi, f = sum_flux_tuples(nzl[idx], nz[idx], q[idx])
                fz = ["TO_CONSTANT_HEAD"] * len(tzi)
                tz = [self._zonenamedict[z] for z in tzi]
                self._update_budget_fromssst(
                    fz, tz, np.abs(f), kstpkper, totim
                )

                idx = np.where(
                    (q < 0) & ((ich[k, i, j] != 1) | (ich[k, i, jl] != 1))
                )
                fzi, tzi, f = sum_flux_tuples(nzl[idx], nz[idx], q[idx])
                fz = ["FROM_CONSTANT_HEAD"] * len(fzi)
                tz = [self._zonenamedict[z] for z in tzi[tzi != 0]]
                self._update_budget_fromssst(
                    fz, tz, np.abs(f), kstpkper, totim
                )

                k, i, j = np.where(ich == 1)
                k, i, j = (
                    k[j < self.ncol - 1],
                    i[j < self.ncol - 1],
                    j[j < self.ncol - 1],
                )
                nz = self.izone[k, i, j]
                jr = j + 1
                nzr = self.izone[k, i, jr]
                q = data[k, i, j]
                idx = np.where(
                    (q > 0) & ((ich[k, i, j] != 1) | (ich[k, i, jr] != 1))
                )
                fzi, tzi, f = sum_flux_tuples(nzr[idx], nz[idx], q[idx])
                fz = ["FROM_CONSTANT_HEAD"] * len(tzi)
                tz = [self._zonenamedict[z] for z in tzi]
                self._update_budget_fromssst(
                    fz, tz, np.abs(f), kstpkper, totim
                )

                idx = np.where(
                    (q < 0) & ((ich[k, i, j] != 1) | (ich[k, i, jr] != 1))
                )
                fzi, tzi, f = sum_flux_tuples(nzr[idx], nz[idx], q[idx])
                fz = ["TO_CONSTANT_HEAD"] * len(fzi)
                tz = [self._zonenamedict[z] for z in tzi]
                self._update_budget_fromssst(
                    fz, tz, np.abs(f), kstpkper, totim
                )

        except Exception as e:
            print(e)
            raise
        return

    def _accumulate_flow_fff(self, recname, ich, kstpkper, totim):
        """

        Parameters
        ----------
        recname
        ich
        kstpkper
        totim

        Returns
        -------

        """
        try:
            if self.nrow >= 2:
                data = self.cbc.get_data(
                    text=recname, kstpkper=kstpkper, totim=totim
                )[0]

                # "FLOW FRONT FACE"
                # CALCULATE FLOW BETWEEN NODE J,I,K AND J,I-1,K
                k, i, j = np.where(
                    self.izone[:, 1:, :] < self.izone[:, :-1, :]
                )
                i += 1
                ia = i - 1
                nza = self.izone[k, ia, j]
                nz = self.izone[k, i, j]
                q = data[k, ia, j]
                idx = np.where(
                    (q > 0) & ((ich[k, i, j] != 1) | (ich[k, ia, j] != 1))
                )
                fzi, tzi, fi = sum_flux_tuples(nza[idx], nz[idx], q[idx])
                self._update_budget_fromfaceflow(
                    fzi, tzi, np.abs(fi), kstpkper, totim
                )

                idx = np.where(
                    (q < 0) & ((ich[k, i, j] != 1) | (ich[k, ia, j] != 1))
                )
                fzi, tzi, fi = sum_flux_tuples(nz[idx], nza[idx], q[idx])
                self._update_budget_fromfaceflow(
                    fzi, tzi, np.abs(fi), kstpkper, totim
                )

                # CALCULATE FLOW BETWEEN NODE J,I,K AND J,I+1,K.
                k, i, j = np.where(
                    self.izone[:, :-1, :] < self.izone[:, 1:, :]
                )
                nz = self.izone[k, i, j]
                ib = i + 1
                nzb = self.izone[k, ib, j]
                q = data[k, i, j]
                idx = np.where(
                    (q > 0) & ((ich[k, i, j] != 1) | (ich[k, ib, j] != 1))
                )
                fzi, tzi, fi = sum_flux_tuples(nz[idx], nzb[idx], q[idx])
                self._update_budget_fromfaceflow(
                    fzi, tzi, np.abs(fi), kstpkper, totim
                )

                idx = np.where(
                    (q < 0) & ((ich[k, i, j] != 1) | (ich[k, ib, j] != 1))
                )
                fzi, tzi, fi = sum_flux_tuples(nzb[idx], nz[idx], q[idx])
                self._update_budget_fromfaceflow(
                    fzi, tzi, np.abs(fi), kstpkper, totim
                )

                # CALCULATE FLOW TO CONSTANT-HEAD CELLS IN THIS DIRECTION
                k, i, j = np.where(ich == 1)
                k, i, j = k[i > 0], i[i > 0], j[i > 0]
                ia = i - 1
                nza = self.izone[k, ia, j]
                nz = self.izone[k, i, j]
                q = data[k, ia, j]
                idx = np.where(
                    (q > 0) & ((ich[k, i, j] != 1) | (ich[k, ia, j] != 1))
                )
                fzi, tzi, f = sum_flux_tuples(nza[idx], nz[idx], q[idx])
                fz = ["TO_CONSTANT_HEAD"] * len(tzi)
                tz = [self._zonenamedict[z] for z in tzi]
                self._update_budget_fromssst(
                    fz, tz, np.abs(f), kstpkper, totim
                )

                idx = np.where(
                    (q < 0) & ((ich[k, i, j] != 1) | (ich[k, ia, j] != 1))
                )
                fzi, tzi, f = sum_flux_tuples(nza[idx], nz[idx], q[idx])
                fz = ["FROM_CONSTANT_HEAD"] * len(fzi)
                tz = [self._zonenamedict[z] for z in tzi]
                self._update_budget_fromssst(
                    fz, tz, np.abs(f), kstpkper, totim
                )

                k, i, j = np.where(ich == 1)
                k, i, j = (
                    k[i < self.nrow - 1],
                    i[i < self.nrow - 1],
                    j[i < self.nrow - 1],
                )
                nz = self.izone[k, i, j]
                ib = i + 1
                nzb = self.izone[k, ib, j]
                q = data[k, i, j]
                idx = np.where(
                    (q > 0) & ((ich[k, i, j] != 1) | (ich[k, ib, j] != 1))
                )
                fzi, tzi, f = sum_flux_tuples(nzb[idx], nz[idx], q[idx])
                fz = ["FROM_CONSTANT_HEAD"] * len(tzi)
                tz = [self._zonenamedict[z] for z in tzi]
                self._update_budget_fromssst(
                    fz, tz, np.abs(f), kstpkper, totim
                )

                idx = np.where(
                    (q < 0) & ((ich[k, i, j] != 1) | (ich[k, ib, j] != 1))
                )
                fzi, tzi, f = sum_flux_tuples(nzb[idx], nz[idx], q[idx])
                fz = ["TO_CONSTANT_HEAD"] * len(fzi)
                tz = [self._zonenamedict[z] for z in tzi]
                self._update_budget_fromssst(
                    fz, tz, np.abs(f), kstpkper, totim
                )

        except Exception as e:
            print(e)
            raise
        return

    def _accumulate_flow_flf(self, recname, ich, kstpkper, totim):
        """

        Parameters
        ----------
        recname
        ich
        kstpkper
        totim

        Returns
        -------

        """
        try:
            if self.nlay >= 2:
                data = self.cbc.get_data(
                    text=recname, kstpkper=kstpkper, totim=totim
                )[0]

                # "FLOW LOWER FACE"
                # CALCULATE FLOW BETWEEN NODE J,I,K AND J,I,K-1
                k, i, j = np.where(
                    self.izone[1:, :, :] < self.izone[:-1, :, :]
                )
                k += 1
                ka = k - 1
                nza = self.izone[ka, i, j]
                nz = self.izone[k, i, j]
                q = data[ka, i, j]
                idx = np.where(
                    (q > 0) & ((ich[k, i, j] != 1) | (ich[ka, i, j] != 1))
                )
                fzi, tzi, fi = sum_flux_tuples(nza[idx], nz[idx], q[idx])
                self._update_budget_fromfaceflow(
                    fzi, tzi, np.abs(fi), kstpkper, totim
                )

                idx = np.where(
                    (q < 0) & ((ich[k, i, j] != 1) | (ich[ka, i, j] != 1))
                )
                fzi, tzi, fi = sum_flux_tuples(nz[idx], nza[idx], q[idx])
                self._update_budget_fromfaceflow(
                    fzi, tzi, np.abs(fi), kstpkper, totim
                )

                # CALCULATE FLOW BETWEEN NODE J,I,K AND J,I,K+1
                k, i, j = np.where(
                    self.izone[:-1, :, :] < self.izone[1:, :, :]
                )
                nz = self.izone[k, i, j]
                kb = k + 1
                nzb = self.izone[kb, i, j]
                q = data[k, i, j]
                idx = np.where(
                    (q > 0) & ((ich[k, i, j] != 1) | (ich[kb, i, j] != 1))
                )
                fzi, tzi, fi = sum_flux_tuples(nz[idx], nzb[idx], q[idx])
                self._update_budget_fromfaceflow(
                    fzi, tzi, np.abs(fi), kstpkper, totim
                )

                idx = np.where(
                    (q < 0) & ((ich[k, i, j] != 1) | (ich[kb, i, j] != 1))
                )
                fzi, tzi, fi = sum_flux_tuples(nzb[idx], nz[idx], q[idx])
                self._update_budget_fromfaceflow(
                    fzi, tzi, np.abs(fi), kstpkper, totim
                )

                # CALCULATE FLOW TO CONSTANT-HEAD CELLS IN THIS DIRECTION
                k, i, j = np.where(ich == 1)
                k, i, j = k[k > 0], i[k > 0], j[k > 0]
                ka = k - 1
                nza = self.izone[ka, i, j]
                nz = self.izone[k, i, j]
                q = data[ka, i, j]
                idx = np.where(
                    (q > 0) & ((ich[k, i, j] != 1) | (ich[ka, i, j] != 1))
                )
                fzi, tzi, f = sum_flux_tuples(nza[idx], nz[idx], q[idx])
                fz = ["TO_CONSTANT_HEAD"] * len(tzi)
                tz = [self._zonenamedict[z] for z in tzi]
                self._update_budget_fromssst(
                    fz, tz, np.abs(f), kstpkper, totim
                )

                idx = np.where(
                    (q < 0) & ((ich[k, i, j] != 1) | (ich[ka, i, j] != 1))
                )
                fzi, tzi, f = sum_flux_tuples(nza[idx], nz[idx], q[idx])
                fz = ["FROM_CONSTANT_HEAD"] * len(fzi)
                tz = [self._zonenamedict[z] for z in tzi]
                self._update_budget_fromssst(
                    fz, tz, np.abs(f), kstpkper, totim
                )

                k, i, j = np.where(ich == 1)
                k, i, j = (
                    k[k < self.nlay - 1],
                    i[k < self.nlay - 1],
                    j[k < self.nlay - 1],
                )
                nz = self.izone[k, i, j]
                kb = k + 1
                nzb = self.izone[kb, i, j]
                q = data[k, i, j]
                idx = np.where(
                    (q > 0) & ((ich[k, i, j] != 1) | (ich[kb, i, j] != 1))
                )
                fzi, tzi, f = sum_flux_tuples(nzb[idx], nz[idx], q[idx])
                fz = ["FROM_CONSTANT_HEAD"] * len(tzi)
                tz = [self._zonenamedict[z] for z in tzi]
                self._update_budget_fromssst(
                    fz, tz, np.abs(f), kstpkper, totim
                )

                idx = np.where(
                    (q < 0) & ((ich[k, i, j] != 1) | (ich[kb, i, j] != 1))
                )
                fzi, tzi, f = sum_flux_tuples(nzb[idx], nz[idx], q[idx])
                fz = ["TO_CONSTANT_HEAD"] * len(fzi)
                tz = [self._zonenamedict[z] for z in tzi]
                self._update_budget_fromssst(
                    fz, tz, np.abs(f), kstpkper, totim
                )

        except Exception as e:
            print(e)
            raise
        return

    def _accumulate_flow_ssst(self, recname, kstpkper, totim):

        # NOT AN INTERNAL FLOW TERM, SO MUST BE A SOURCE TERM OR STORAGE
        # ACCUMULATE THE FLOW BY ZONE

        imeth = self.imeth[recname]

        data = self.cbc.get_data(text=recname, kstpkper=kstpkper, totim=totim)
        if len(data) == 0:
            # Empty data, can occur during the first time step of a transient
            # model when storage terms are zero and not in the cell-budget
            # file.
            return
        else:
            data = data[0]

        if imeth == 2 or imeth == 5:
            # LIST
            qin = np.ma.zeros(
                (self.nlay * self.nrow * self.ncol), self.float_type
            )
            qout = np.ma.zeros(
                (self.nlay * self.nrow * self.ncol), self.float_type
            )
            for [node, q] in zip(data["node"], data["q"]):
                idx = node - 1
                if q > 0:
                    qin.data[idx] += q
                elif q < 0:
                    qout.data[idx] += q
            qin = np.ma.reshape(qin, (self.nlay, self.nrow, self.ncol))
            qout = np.ma.reshape(qout, (self.nlay, self.nrow, self.ncol))
        elif imeth == 0 or imeth == 1:
            # FULL 3-D ARRAY
            qin = np.ma.zeros(self.cbc_shape, self.float_type)
            qout = np.ma.zeros(self.cbc_shape, self.float_type)
            qin[data > 0] = data[data > 0]
            qout[data < 0] = data[data < 0]
        elif imeth == 3:
            # 1-LAYER ARRAY WITH LAYER INDICATOR ARRAY
            rlay, rdata = data[0], data[1]
            data = np.ma.zeros(self.cbc_shape, self.float_type)
            for (r, c), l in np.ndenumerate(rlay):
                data[l - 1, r, c] = rdata[r, c]
            qin = np.ma.zeros(self.cbc_shape, self.float_type)
            qout = np.ma.zeros(self.cbc_shape, self.float_type)
            qin[data > 0] = data[data > 0]
            qout[data < 0] = data[data < 0]
        elif imeth == 4:
            # 1-LAYER ARRAY THAT DEFINES LAYER 1
            qin = np.ma.zeros(self.cbc_shape, self.float_type)
            qout = np.ma.zeros(self.cbc_shape, self.float_type)
            r, c = np.where(data > 0)
            qin[0, r, c] = data[r, c]
            r, c = np.where(data < 0)
            qout[0, r, c] = data[r, c]
        else:
            # Should not happen
            raise Exception(
                'Unrecognized "imeth" for {} record: {}'.format(recname, imeth)
            )

        # Inflows
        fz = []
        tz = []
        f = []
        for z in self.allzones:
            if z != 0:
                flux = qin[(self.izone == z)].sum()
                if type(flux) == np.ma.core.MaskedConstant:
                    flux = 0.0
                fz.append("FROM_" + "_".join(recname.split()))
                tz.append(self._zonenamedict[z])
                f.append(flux)
        fz = np.array(fz)
        tz = np.array(tz)
        f = np.array(f)
        self._update_budget_fromssst(fz, tz, np.abs(f), kstpkper, totim)

        # Outflows
        fz = []
        tz = []
        f = []
        for z in self.allzones:
            if z != 0:
                flux = qout[(self.izone == z)].sum()
                if type(flux) == np.ma.core.MaskedConstant:
                    flux = 0.0
                fz.append("TO_" + "_".join(recname.split()))
                tz.append(self._zonenamedict[z])
                f.append(flux)
        fz = np.array(fz)
        tz = np.array(tz)
        f = np.array(f)
        self._update_budget_fromssst(fz, tz, np.abs(f), kstpkper, totim)

        return

    def _compute_mass_balance(self, kstpkper, totim):
        # Returns a record array with total inflow, total outflow,
        # and percent error summed by column.
        skipcols = ["time_step", "stress_period", "totim", "name"]

        # Compute inflows
        recnames = self.get_record_names()
        innames = [n for n in recnames if n.startswith("FROM_")]
        outnames = [n for n in recnames if n.startswith("TO_")]
        if kstpkper is not None:
            rowidx = np.where(
                (self._budget["time_step"] == kstpkper[0])
                & (self._budget["stress_period"] == kstpkper[1])
                & np.in1d(self._budget["name"], innames)
            )
        elif totim is not None:
            rowidx = np.where(
                (self._budget["totim"] == totim)
                & np.in1d(self._budget["name"], innames)
            )
        a = _numpyvoid2numeric(
            self._budget[list(self._zonenamedict.values())][rowidx]
        )
        intot = np.array(a.sum(axis=0))
        tz = np.array(
            list([n for n in self._budget.dtype.names if n not in skipcols])
        )
        fz = np.array(["TOTAL_IN"] * len(tz))
        self._update_budget_fromssst(fz, tz, intot, kstpkper, totim)

        # Compute outflows
        if kstpkper is not None:
            rowidx = np.where(
                (self._budget["time_step"] == kstpkper[0])
                & (self._budget["stress_period"] == kstpkper[1])
                & np.in1d(self._budget["name"], outnames)
            )
        elif totim is not None:
            rowidx = np.where(
                (self._budget["totim"] == totim)
                & np.in1d(self._budget["name"], outnames)
            )
        a = _numpyvoid2numeric(
            self._budget[list(self._zonenamedict.values())][rowidx]
        )
        outot = np.array(a.sum(axis=0))
        tz = np.array(
            list([n for n in self._budget.dtype.names if n not in skipcols])
        )
        fz = np.array(["TOTAL_OUT"] * len(tz))
        self._update_budget_fromssst(fz, tz, outot, kstpkper, totim)

        # Compute IN-OUT
        tz = np.array(
            list([n for n in self._budget.dtype.names if n not in skipcols])
        )
        f = intot - outot
        fz = np.array(["IN-OUT"] * len(tz))
        self._update_budget_fromssst(fz, tz, np.abs(f), kstpkper, totim)

        # Compute percent discrepancy
        tz = np.array(
            list([n for n in self._budget.dtype.names if n not in skipcols])
        )
        fz = np.array(["PERCENT_DISCREPANCY"] * len(tz))
        in_minus_out = intot - outot
        in_plus_out = intot + outot
        f = 100 * in_minus_out / (in_plus_out / 2.0)
        self._update_budget_fromssst(fz, tz, np.abs(f), kstpkper, totim)

        return

    def _clean_budget_names(self, names):
        newnames = []
        mbnames = ["TOTAL_IN", "TOTAL_OUT", "IN-OUT", "PERCENT_DISCREPANCY"]
        for name in names:
            if name in mbnames:
                newnames.append(name)
            elif not name.startswith("FROM_") and not name.startswith("TO_"):
                newname_in = "FROM_" + name.upper()
                newname_out = "TO_" + name.upper()
                if newname_in in self._budget["name"]:
                    newnames.append(newname_in)
                if newname_out in self._budget["name"]:
                    newnames.append(newname_out)
            else:
                if name in self._budget["name"]:
                    newnames.append(name)
        return newnames

    def _compute_net_budget(self):
        recnames = self.get_record_names()
        innames = [n for n in recnames if n.startswith("FROM_")]
        outnames = [n for n in recnames if n.startswith("TO_")]
        select_fields = ["totim", "time_step", "stress_period", "name"] + list(
            self._zonenamedict.values()
        )
        select_records_in = np.in1d(self._budget["name"], innames)
        select_records_out = np.in1d(self._budget["name"], outnames)
        in_budget = self._budget[select_fields][select_records_in]
        out_budget = self._budget[select_fields][select_records_out]
        net_budget = in_budget.copy()
        for f in [
            n for n in self._zonenamedict.values() if n in select_fields
        ]:
            net_budget[f] = np.array([r for r in in_budget[f]]) - np.array(
                [r for r in out_budget[f]]
            )
        newnames = ["_".join(n.split("_")[1:]) for n in net_budget["name"]]
        net_budget["name"] = newnames
        return net_budget

    def __mul__(self, other):
        newbud = self._budget.copy()
        for f in self._zonenamedict.values():
            newbud[f] = np.array([r for r in newbud[f]]) * other
        idx = np.in1d(self._budget["name"], "PERCENT_DISCREPANCY")
        newbud[:][idx] = self._budget[:][idx]
        newobj = self.copy()
        newobj._budget = newbud
        return newobj

    def __truediv__(self, other):
        newbud = self._budget.copy()
        for f in self._zonenamedict.values():
            newbud[f] = np.array([r for r in newbud[f]]) / float(other)
        idx = np.in1d(self._budget["name"], "PERCENT_DISCREPANCY")
        newbud[:][idx] = self._budget[:][idx]
        newobj = self.copy()
        newobj._budget = newbud
        return newobj

    def __div__(self, other):
        newbud = self._budget.copy()
        for f in self._zonenamedict.values():
            newbud[f] = np.array([r for r in newbud[f]]) / float(other)
        idx = np.in1d(self._budget["name"], "PERCENT_DISCREPANCY")
        newbud[:][idx] = self._budget[:][idx]
        newobj = self.copy()
        newobj._budget = newbud
        return newobj

    def __add__(self, other):
        newbud = self._budget.copy()
        for f in self._zonenamedict.values():
            newbud[f] = np.array([r for r in newbud[f]]) + other
        idx = np.in1d(self._budget["name"], "PERCENT_DISCREPANCY")
        newbud[:][idx] = self._budget[:][idx]
        newobj = self.copy()
        newobj._budget = newbud
        return newobj

    def __sub__(self, other):
        newbud = self._budget.copy()
        for f in self._zonenamedict.values():
            newbud[f] = np.array([r for r in newbud[f]]) - other
        idx = np.in1d(self._budget["name"], "PERCENT_DISCREPANCY")
        newbud[:][idx] = self._budget[:][idx]
        newobj = self.copy()
        newobj._budget = newbud
        return newobj


def _numpyvoid2numeric(a):
    # The budget record array has multiple dtypes and a slice returns
    # the flexible-type numpy.void which must be converted to a numeric
    # type prior to performing reducing functions such as sum() or
    # mean()
    return np.array([list(r) for r in a])


def write_zbarray(fname, X, fmtin=None, iprn=None):
    """
    Saves a numpy array in a format readable by the zonebudget program
    executable.

    File format:
    line 1: nlay, nrow, ncol
    line 2: INTERNAL (format)
    line 3: begin data
    .
    .
    .

    example from NACP:
    19 250 500
    INTERNAL      (10I8)
    199     199     199     199     199     199     199     199     199     199
    199     199     199     199     199     199     199     199     199     199
    ...
    INTERNAL      (10I8)
    199     199     199     199     199     199     199     199     199     199
    199     199     199     199     199     199     199     199     199     199
    ...

    Parameters
    ----------
    X : array
        The array of zones to be written.
    fname :  str
        The path and name of the file to be written.
    fmtin : int
        The number of values to write to each line.
    iprn : int
        Padding space to add between each value.

    Returns
    -------

    """
    if len(X.shape) == 2:
        b = np.zeros((1, X.shape[0], X.shape[1]), dtype=np.int32)
        b[0, :, :] = X[:, :]
        X = b.copy()
    elif len(X.shape) < 2 or len(X.shape) > 3:
        raise Exception(
            "Shape of the input array is not recognized: {}".format(X.shape)
        )
    if np.ma.is_masked(X):
        X = np.ma.filled(X, 0)

    nlay, nrow, ncol = X.shape

    if fmtin is not None:
        assert fmtin < ncol, (
            "The specified width is greater than the "
            "number of columns in the array."
        )
    else:
        fmtin = ncol

    iprnmin = len(str(X.max()))
    if iprn is None or iprn <= iprnmin:
        iprn = iprnmin + 1

    formatter_str = "{{:>{iprn}}}".format(iprn=iprn)
    formatter = formatter_str.format

    with open(fname, "w") as f:
        header = "{nlay} {nrow} {ncol}\n".format(
            nlay=nlay, nrow=nrow, ncol=ncol
        )
        f.write(header)
        for lay in range(nlay):
            record_2 = "INTERNAL\t({fmtin}I{iprn})\n".format(
                fmtin=fmtin, iprn=iprn
            )
            f.write(record_2)
            if fmtin < ncol:
                for row in range(nrow):
                    rowvals = X[lay, row, :].ravel()
                    start = 0
                    end = start + fmtin
                    vals = rowvals[start:end]
                    while len(vals) > 0:
                        s = (
                            "".join([formatter(int(val)) for val in vals])
                            + "\n"
                        )
                        f.write(s)
                        start = end
                        end = start + fmtin
                        vals = rowvals[start:end]

            elif fmtin == ncol:
                for row in range(nrow):
                    vals = X[lay, row, :].ravel()
                    f.write(
                        "".join([formatter(int(val)) for val in vals]) + "\n"
                    )
    return


def read_zbarray(fname):
    """
    Reads an ascii array in a format readable by the zonebudget program
    executable.

    Parameters
    ----------
    fname :  str
        The path and name of the file to be written.

    Returns
    -------
    zones : numpy ndarray
        An integer array of the zones.
    """
    with open(fname, "r") as f:
        lines = f.readlines()

    # Initialize layer
    lay = 0

    # Initialize data counter
    totlen = 0
    i = 0

    # First line contains array dimensions
    dimstring = lines.pop(0).strip().split()
    nlay, nrow, ncol = [int(v) for v in dimstring]
    zones = np.zeros((nlay, nrow, ncol), dtype=np.int32)

    # The number of values to read before placing
    # them into the zone array
    datalen = nrow * ncol

    # List of valid values for LOCAT
    locats = ["CONSTANT", "INTERNAL", "EXTERNAL"]

    # ITERATE OVER THE ROWS
    for line in lines:
        rowitems = line.strip().split()

        # Skip blank lines
        if len(rowitems) == 0:
            continue

        # HEADER
        if rowitems[0].upper() in locats:
            vals = []
            locat = rowitems[0].upper()

            if locat == "CONSTANT":
                iconst = int(rowitems[1])
            else:
                fmt = rowitems[1].strip("()")
                fmtin, iprn = [int(v) for v in fmt.split("I")]

        # ZONE DATA
        else:
            if locat == "CONSTANT":
                vals = np.ones((nrow, ncol), dtype=np.int32) * iconst
                lay += 1
            elif locat == "INTERNAL":
                # READ ZONES
                rowvals = [int(v) for v in rowitems]
                s = "Too many values encountered on this line."
                assert len(rowvals) <= fmtin, s
                vals.extend(rowvals)

            elif locat == "EXTERNAL":
                # READ EXTERNAL FILE
                fname = rowitems[0]
                if not os.path.isfile(fname):
                    errmsg = 'Could not find external file "{}"'.format(fname)
                    raise Exception(errmsg)
                with open(fname, "r") as ext_f:
                    ext_flines = ext_f.readlines()
                for ext_frow in ext_flines:
                    ext_frowitems = ext_frow.strip().split()
                    rowvals = [int(v) for v in ext_frowitems]
                    vals.extend(rowvals)
                if len(vals) != datalen:
                    errmsg = (
                        "The number of values read from external "
                        'file "{}" does not match the expected '
                        "number.".format(len(vals))
                    )
                    raise Exception(errmsg)
            else:
                # Should not get here
                raise Exception("Locat not recognized: {}".format(locat))

                # IGNORE COMPOSITE ZONES

            if len(vals) == datalen:
                # place values for the previous layer into the zone array
                vals = np.array(vals, dtype=np.int32).reshape((nrow, ncol))
                zones[lay, :, :] = vals[:, :]
                lay += 1
            totlen += len(rowitems)
        i += 1
    s = (
        "The number of values read ({:,.0f})"
        " does not match the number expected"
        " ({:,.0f})".format(totlen, nlay * nrow * ncol)
    )
    assert totlen == nlay * nrow * ncol, s
    return zones


def sum_flux_tuples(fromzones, tozones, fluxes):
    tup = zip(fromzones, tozones, fluxes)
    sorted_tups = sort_tuple(tup)

    # Group the sorted tuples by (from zone, to zone)
    # itertools.groupby() returns the index (from zone, to zone) and
    # a list of the tuples with that index
    from_zones = []
    to_zones = []
    fluxes = []
    for (fz, tz), ftup in groupby(sorted_tups, lambda tup: tup[:2]):
        f = np.sum([tup[-1] for tup in list(ftup)])
        from_zones.append(fz)
        to_zones.append(tz)
        fluxes.append(f)
    return np.array(from_zones), np.array(to_zones), np.array(fluxes)


def sort_tuple(tup, n=2):
    """Sort a tuple by the first n values

    tup: tuple
        input tuple
    n : int
        values to sort tuple by (default is 2)

    Returns
    -------
    tup : tuple
        tuple sorted by the first n values

    """
    return tuple(sorted(tup, key=lambda t: t[:n]))


def get_totim_modflow6(tdis):
    """Create a totim array from the tdis file in modflow 6

    Parameters
    ----------
    tdis : ModflowTdis object
        MODDFLOW 6 TDIS object

    Returns
    -------
    totim : np.ndarray
        total time vector for simulation


    """
    recarray = tdis.perioddata.array
    delt = []
    for record in recarray:
        perlen = record.perlen
        nstp = record.nstp
        tsmult = record.tsmult
        for stp in range(nstp):
            if stp == 0:
                if tsmult != 1.0:
                    dt = perlen * (tsmult - 1) / ((tsmult ** nstp) - 1)
                else:
                    dt = perlen / nstp
            else:
                dt = delt[-1] * tsmult

            delt.append(dt)

    totim = np.add.accumulate(delt)

    return totim


class ZBNetOutput(object):
    """
    Class that holds zonebudget netcdf output and allows export utilities
    to recognize the output data type.

    Parameters
    ----------
    zones : np.ndarray
        array of zone numbers
    time : np.ndarray
        array of totim
    arrays : dict
        dictionary of budget term arrays.
        axis 0 is totim,
        axis 1 is zones
    flux : bool
        boolean flag to indicate if budget data is a flux "L^3/T"(True,
        default) or if the data have been processed to
        volumetric values "L^3" (False)
    """

    def __init__(self, zones, time, arrays, zone_array, flux=True):
        self.zones = zones
        self.time = time
        self.arrays = arrays
        self.zone_array = zone_array
        self.flux = flux


class ZoneBudgetOutput(object):
    """
    Class method to process zonebudget output into volumetric budgets

    Parameters
    ----------
    f : str
        zonebudget output file path
    dis : flopy.modflow.ModflowDis object
    zones : np.ndarray
        numpy array of zones

    """

    def __init__(self, f, dis, zones):
        import pandas as pd
        from ..modflow import ModflowDis

        self._filename = f
        self._otype = None
        self._zones = zones
        self.__pd = pd

        if isinstance(dis, ModflowDis):
            self._totim = dis.get_totim()
            self._nstp = dis.nstp.array
            self._steady = dis.steady.array

        else:
            self._totim = get_totim_modflow6(dis)
            self._nstp = np.array(dis.perioddata.array.nstp)
            # self._steady is a placeholder, data not used for ZB6 read
            self._steady = [False for _ in dis.perioddata.array]

        self._tslen = None
        self._date_time = None
        self._data = None

        if self._otype is None:
            self._get_otype()

        self._calculate_tslen()
        self._read_file()

    def __repr__(self):
        """
        String representation of the ZoneBudgetOutput class

        """
        zones = ", ".join([str(i) for i in self.zones])
        l = [
            "ZoneBudgetOutput Class",
            "----------------------\n",
            "Number of zones: {}".format(len(self.zones)),
            "Unique zones: {}".format(zones),
            "Number of buget records: {}".format(len(self.dataframe)),
        ]

        return "\n".join(l)

    @property
    def zone_array(self):
        """
        Property method to get the zone array

        """
        return np.asarray(self._zones, dtype=int)

    @property
    def zones(self):
        """
        Get a unique list of zones

        """
        return np.unique(self.zone_array)

    @property
    def dataframe(self):
        """
        Returns a net flux dataframe of the zonebudget output

        """
        return self.__pd.DataFrame.from_dict(self._data)

    def _calculate_tslen(self):
        """
        Method to calculate each timestep length from totim
        and reset totim to a dictionary of {(kstp, kper): totim}

        """
        n = 0
        totim = {}
        for ix, stp in enumerate(self._nstp):
            for i in range(stp):
                if self._tslen is None:
                    tslen = self._totim[n]
                    self._tslen = {(i, ix): tslen}
                else:
                    tslen = self._totim[n] - self._totim[n - 1]
                    self._tslen[(i, ix)] = tslen

                totim[(i, ix)] = self._totim[n]
                n += 1

        self._totim = totim

    def _read_file(self):
        """
        Delegator method for reading zonebudget outputs

        """
        if self._otype == 1:
            self._read_file1()
        elif self._otype == 2:
            self._read_file2()
        elif self._otype == 3:
            self._read_file3()
        else:
            raise AssertionError(
                "Invalid otype supplied: {}".format(self._otype)
            )

    def _read_file1(self):
        """
        Read original style zonebudget output file

        """

        with open(self._filename) as foo:

            data_in = {}
            data_out = {}
            read_in = False
            read_out = False
            flow_budget = False
            empty = 0
            while True:
                line = foo.readline().strip().lower()

                if "flow budget for zone" in line:
                    flow_budget = True
                    read_in = False
                    read_out = False
                    empty = 0
                    t = line.split()
                    zone = int(t[4])
                    if len(t[7]) > 4:
                        t.insert(8, t[7][4:])
                    kstp = int(t[8]) - 1
                    if len(t[11]) > 6:
                        t.append(t[11][6:])
                    kper = int(t[12]) - 1
                    if "zone" not in data_in:
                        data_in["zone"] = [zone]
                        data_in["kstp"] = [kstp]
                        data_in["kper"] = [kper]
                    else:
                        data_in["zone"].append(zone)
                        data_in["kstp"].append(kstp)
                        data_in["kper"].append(kper)

                    if self._steady[kper]:
                        try:
                            data_in["storage"].append(0.0)
                            data_out["storage"].append(0.0)
                        except KeyError:
                            data_in["storage"] = [0.0]
                            data_out["storage"] = [0.0]

                elif line in ("", " "):
                    empty += 1

                elif read_in:
                    if "=" in line:
                        t = line.split("=")
                        label = t[0].strip()
                        if "zone" in line:
                            # currently we do not support zone to zone
                            # flow for option 1
                            pass
                        else:
                            if "total" in line:
                                label = "total"

                            if label in data_in:
                                data_in[label].append(float(t[1]))
                            else:
                                data_in[label] = [float(t[1])]

                    elif "out:" in line:
                        read_out = True
                        read_in = False

                    else:
                        pass

                elif read_out:
                    if "=" in line:
                        t = line.split("=")
                        label = t[0].strip()
                        if "zone" in line:
                            # currently we do not support zone to zone
                            # flow for option 1
                            pass

                        elif "in - out" in line:
                            pass

                        elif "percent discrepancy" in line:
                            pass

                        else:
                            if "total" in line:
                                label = "total"

                            if label in data_out:
                                data_out[label].append(float(t[1]))
                            else:
                                data_out[label] = [float(t[1])]
                    else:
                        pass

                elif flow_budget:
                    if "in:" in line:
                        read_in = True
                        flow_budget = False

                else:
                    pass

                if empty >= 30:
                    break

        data = self._net_flux(data_in, data_out)

        self._data = data

    def _read_file2(self):
        """
        Method to read csv output type 1

        """
        with open(self._filename) as foo:
            data_in = {}
            data_out = {}
            zone_header = False
            read_in = False
            read_out = False
            empty = 0
            while True:
                line = foo.readline().strip().lower()

                if "time step" in line:
                    t = line.split(",")
                    kstp = int(t[1]) - 1
                    kper = int(t[3]) - 1
                    if "kstp" not in data_in:
                        data_in["kstp"] = []
                        data_in["kper"] = []
                        data_in["zone"] = []

                    zone_header = True
                    empty = 0

                elif zone_header:
                    t = line.split(",")
                    zones = [
                        int(i.split()[-1]) for i in t[1:] if i not in ("",)
                    ]

                    for zone in zones:
                        data_in["kstp"].append(kstp)
                        data_in["kper"].append(kper)
                        data_in["zone"].append(zone)
                        if self._steady[kper]:
                            try:
                                data_in["storage"].append(0.0)
                                data_out["storage"].append(0.0)
                            except KeyError:
                                data_in["storage"] = [0.0]
                                data_out["storage"] = [0.0]

                    zone_header = False
                    read_in = True

                elif read_in:
                    t = line.split(",")
                    if "in" in t[1]:
                        pass

                    elif "out" in t[1]:
                        read_in = False
                        read_out = True

                    else:
                        if "zone" in t[0]:
                            label = " ".join(t[0].split()[1:])

                        elif "total" in t[0]:
                            label = "total"

                        else:
                            label = t[0]

                        if label not in data_in:
                            data_in[label] = []

                        for val in t[1:]:
                            if val in ("",):
                                continue

                            data_in[label].append(float(val))

                elif read_out:
                    t = line.split(",")

                    if "percent error" in line:
                        read_out = False

                    elif "in-out" in line:
                        pass

                    else:
                        if "zone" in t[0]:
                            label = " ".join(t[0].split()[1:])

                        elif "total" in t[0]:
                            label = "total"

                        else:
                            label = t[0]

                        if label not in data_out:
                            data_out[label] = []

                        for val in t[1:]:
                            if val in ("",):
                                continue

                            data_out[label].append(float(val))

                elif line in ("", " "):
                    empty += 1

                else:
                    pass

                if empty >= 25:
                    break

        data = self._net_flux(data_in, data_out)

        self._data = data

    def _read_file3(self):
        """
        Method to read CSV2 output from zonebudget and CSV output
        from Zonebudget6

        """
        with open(self._filename) as foo:
            data_in = {}
            data_out = {}
            read_in = True
            read_out = False
            # read the header
            header = foo.readline().lower().strip().split(",")
            header = [i.strip() for i in header]

            array = np.genfromtxt(foo, delimiter=",").T

        for ix, label in enumerate(header):
            if label in ("totim", "in-out", "percent error"):
                continue

            elif label == "percent error":
                continue

            elif label == "step":
                label = "kstp"

            elif label == "period":
                label = "kper"

            elif "other zones" in label:
                label = "other zones"

            elif "from zone" in label or "to zone" in label:
                if "from" in label:
                    read_in = True
                    read_out = False
                else:
                    read_out = True
                    read_in = False
                label = " ".join(label.split()[1:])

            elif "total" in label:
                label = "total"

            elif label.split("-")[-1] == "in":
                label = "-".join(label.split("-")[:-1])
                read_in = True
                read_out = False

            elif label.split("-")[-1] == "out":
                label = "-".join(label.split("-")[:-1])
                read_in = False
                read_out = True

            else:
                pass

            if read_in:

                if label in ("kstp", "kper"):
                    data_in[label] = np.asarray(array[ix], dtype=int) - 1

                elif label == "zone":
                    data_in[label] = np.asarray(array[ix], dtype=int)

                else:
                    data_in[label] = array[ix]

                if label == "total":
                    read_in = False
                    read_out = True

            elif read_out:
                data_out[label] = array[ix]

            else:
                pass

        data = self._net_flux(data_in, data_out)

        self._data = data

    def _net_flux(self, data_in, data_out):
        """
        Method to create a single dictionary of net flux data

        data_in : dict
            inputs to the zone
        data_out : dict
            outputs from the zone

        Returns
        -------
        dict : dictionary of netflux data to feed into a pandas dataframe
        """
        data = {}
        # calculate net storage flux (subroutine this?)
        for key, value in data_in.items():
            if key in ("zone", "kstp", "kper"):
                data[key] = np.asarray(value, dtype=int)
            else:
                arrayin = np.asarray(value)
                arrayout = np.asarray(data_out[key])

                data[key] = arrayin - arrayout

        kstp = data["kstp"]
        kper = data["kper"]
        tslen = np.array(
            [self._tslen[(stp, kper[ix])] for ix, stp in enumerate(kstp)]
        )
        totim = np.array(
            [self._totim[(stp, kper[ix])] for ix, stp in enumerate(kstp)]
        )

        data["tslen"] = tslen
        data["totim"] = totim

        return data

    def _get_otype(self):
        """
        Method to automatically distinguish output type based on the
        zonebudget header

        """
        with open(self._filename) as foo:
            line = foo.readline()
            if "zonebudget version" in line.lower():
                self._otype = 1
            elif "time step" in line.lower():
                self._otype = 2
            elif "totim" in line.lower():
                self._otype = 3
            else:
                raise AssertionError("Cant distinguish output type")

    def export(self, f, ml, **kwargs):
        """
        Method to export a netcdf file, or add zonebudget output to
        an open netcdf file instance

        Parameters
        ----------
        f : str or flopy.export.netcdf.NetCdf object
        ml : flopy.modflow.Modflow or flopy.mf6.ModflowGwf object
        **kwargs :
            logger : flopy.export.netcdf.Logger instance
            masked_vals : list
                list of values to mask

        Returns
        -------
            flopy.export.netcdf.NetCdf object

        """
        from flopy.export.utils import output_helper

        if isinstance(f, str):
            if not f.endswith(".nc"):
                raise AssertionError(
                    "File extension must end with .nc to "
                    "export a netcdf file"
                )

        zbncfobj = self.dataframe_to_netcdf_fmt(self.dataframe)
        oudic = {"zbud": zbncfobj}
        return output_helper(f, ml, oudic, **kwargs)

    def volumetric_flux(self, extrapolate_kper=False):
        """
        Method to generate a volumetric budget table based on flux information

        Parameters
        ----------
        extrapolate_kper : bool
            flag to determine if we fill in data gaps with other
            timestep information from the same stress period.
            if True, we assume that flux is constant throughout a stress period
            and the pandas dataframe returned contains a
            volumetric budget per stress period

            if False, calculates volumes from available flux data

        Returns
        -------
            pd.DataFrame

        """
        nper = len(self._nstp)
        volumetric_data = {}

        for key in self._data:
            volumetric_data[key] = []

        if extrapolate_kper:
            volumetric_data.pop("tslen")
            volumetric_data.pop("kstp")
            volumetric_data["perlen"] = []

            perlen = []
            for per in range(nper):
                tslen = 0
                for stp in range(self._nstp[per]):
                    tslen += self._tslen[(stp, per)]

                perlen.append(tslen)

            totim = np.add.accumulate(perlen)

            for per in range(nper):
                idx = np.where(self._data["kper"] == per)[0]

                if len(idx) == 0:
                    continue

                temp = self._data["zone"][idx]

                for zone in self.zones:
                    if zone == 0:
                        continue

                    zix = np.where(temp == zone)[0]

                    if len(zix) == 0:
                        raise Exception

                    for key, value in self._data.items():
                        if key == "totim":
                            volumetric_data[key].append(totim[per])

                        elif key == "tslen":
                            volumetric_data["perlen"].append(perlen[per])

                        elif key == "kstp":
                            continue

                        elif key == "kper":
                            volumetric_data[key].append(per)

                        elif key == "zone":
                            volumetric_data[key].append(zone)

                        else:
                            tv = value[idx]
                            zv = tv[zix]
                            for i in zv:
                                vol = i * perlen[per]
                                volumetric_data[key].append(vol)
                                break

        else:

            for key, value in self._data.items():
                if key in ("zone", "kstp", "kper", "tslen"):
                    volumetric_data[key] = value
                else:
                    volumetric_data[key] = value * self._data["tslen"]

        return self.__pd.DataFrame.from_dict(volumetric_data)

    def dataframe_to_netcdf_fmt(self, df, flux=True):
        """
        Method to transform a volumetric zonebudget dataframe into
        array format for netcdf.

        time is on axis 0
        zone is on axis 1

        Parameters
        ----------
        df : pd.DataFrame
        flux : bool
            boolean flag to indicate if budget data is a flux "L^3/T" (True,
            default) or if the data have been processed to
            volumetric values "L^3" (False)
        zone_array : np.ndarray
            zonebudget zones array

        Returns
        -------
            ZBNetOutput object

        """
        zones = np.sort(np.unique(df.zone.values))
        totim = np.sort(np.unique(df.totim.values))

        data = {}
        for col in df.columns:
            if col in ("totim", "zone", "kper", "perlen"):
                pass
            else:
                data[col] = np.zeros((totim.size, zones.size), dtype=float)

        for i, time in enumerate(totim):
            tdf = df.loc[
                df.totim.isin(
                    [
                        time,
                    ]
                )
            ]
            tdf = tdf.sort_values(by=["zone"])

            for col in df.columns:
                if col in ("totim", "zone", "kper", "perlen"):
                    pass
                else:
                    data[col][i, :] = tdf[col].values

        return ZBNetOutput(zones, totim, data, self.zone_array, flux=flux)
