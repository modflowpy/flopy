"""
This is a set of classes for reading budget information out of MODFLOW-style
listing files.  Cumulative and incremental budgets are returned as numpy
recarrays, which can then be easily plotted.

"""

import collections
import os
import re
import numpy as np
import errno

from ..utils.utils_def import totim_to_datetime
from ..utils.flopy_io import get_ts_sp


class ListBudget:
    """
    MODFLOW family list file handling

    Parameters
    ----------
    file_name : str
        the list file name
    budgetkey : str
        the text string identifying the budget table. (default is None)
    timeunit : str
        the time unit to return in the recarray. (default is 'days')

    Notes
    -----
    The ListBudget class should not be instantiated directly.  Access is
    through derived classes: MfListBudget (MODFLOW), SwtListBudget (SEAWAT)
    and SwrListBudget (MODFLOW with the SWR process)

    Examples
    --------
    >>> mf_list = MfListBudget("my_model.list")
    >>> incremental, cumulative = mf_list.get_budget()
    >>> df_in, df_out = mf_list.get_dataframes(start_datetime="10-21-2015")

    """

    def __init__(self, file_name, budgetkey=None, timeunit="days"):

        # Set up file reading
        assert os.path.exists(file_name), "file_name {0} not found".format(
            file_name
        )
        self.file_name = file_name
        self.f = open(file_name, "r", encoding="ascii", errors="replace")

        self.tssp_lines = 0

        # Assign the budgetkey, which should have been overridden
        if budgetkey is None:
            self.set_budget_key()
        else:
            self.budgetkey = budgetkey

        self.totim = []
        self.timeunit = timeunit
        self.idx_map = []
        self.entries = []
        self.null_entries = []

        self.time_line_idx = 20
        if timeunit.upper() == "SECONDS":
            self.timeunit = "S"
            self.time_idx = 0
        elif timeunit.upper() == "MINUTES":
            self.timeunit = "M"
            self.time_idx = 1
        elif timeunit.upper() == "HOURS":
            self.timeunit = "H"
            self.time_idx = 2
        elif timeunit.upper() == "DAYS":
            self.timeunit = "D"
            self.time_idx = 3
        elif timeunit.upper() == "YEARS":
            self.timeunit = "Y"
            self.time_idx = 4
        else:
            raise Exception(
                "need to reset time_idxs attribute to "
                "use units other than days and check usage of "
                "timedelta"
            )

        # Fill budget recarrays
        self._load()
        self._isvalid = False
        if len(self.idx_map) > 0:
            self._isvalid = True

        # Close the open file
        self.f.close()

        # return
        return

    def set_budget_key(self):
        raise Exception("Must be overridden...")

    def isvalid(self):
        """
        Get a boolean indicating if budget data are available in the file.

        Returns
        -------
        out : boolean
            Boolean indicating if budget data are available in the file.

        Examples
        --------
        >>> mf_list = MfListBudget('my_model.list')
        >>> valid = mf_list.isvalid()

        """
        return self._isvalid

    def get_record_names(self):
        """
        Get a list of water budget record names in the file.

        Returns
        -------
        out : list of strings
            List of unique text names in the binary file.

        Examples
        --------
        >>> mf_list = MfListBudget('my_model.list')
        >>> names = mf_list.get_record_names()

        """
        if not self._isvalid:
            return None
        return self.inc.dtype.names

    def get_times(self):
        """
        Get a list of unique water budget times in the list file.

        Returns
        -------
        out : list of floats
            List contains unique water budget simulation times (totim) in list
            file.

        Examples
        --------
        >>> mf_list = MfListBudget('my_model.list')
        >>> times = mf_list.get_times()

        """
        if not self._isvalid:
            return None
        return self.inc["totim"].tolist()

    def get_kstpkper(self):
        """
        Get a list of unique stress periods and time steps in the list file
        water budgets.

        Returns
        ----------
        out : list of (kstp, kper) tuples
            List of unique kstp, kper combinations in list file.  kstp and
            kper values are zero-based.

        Examples
        --------
        >>> mf_list = MfListBudget("my_model.list")
        >>> kstpkper = mf_list.get_kstpkper()

        """
        if not self._isvalid:
            return None
        kstpkper = []
        for kstp, kper in zip(
            self.inc["time_step"], self.inc["stress_period"]
        ):
            kstpkper.append((kstp, kper))
        return kstpkper

    def get_incremental(self, names=None):
        """
        Get a recarray with the incremental water budget items in the list
        file.

        Parameters
        ----------
        names : str or list of strings
            Selection of column names to return.  If names is not None then
            totim, time_step, stress_period, and selection(s) will be returned.
            (default is None).

        Returns
        -------
        out : recarray
            Numpy recarray with the water budget items in list file. The
            recarray also includes totim, time_step, and stress_period.

        Examples
        --------
        >>> mf_list = MfListBudget("my_model.list")
        >>> incremental = mf_list.get_incremental()

        """
        if not self._isvalid:
            return None
        if names is None:
            return self.inc
        else:
            if not isinstance(names, list):
                names = [names]
            names.insert(0, "stress_period")
            names.insert(0, "time_step")
            names.insert(0, "totim")
            return self.inc[names].view(np.recarray)

    def get_cumulative(self, names=None):
        """
        Get a recarray with the cumulative water budget items in the list file.

        Parameters
        ----------
        names : str or list of strings
            Selection of column names to return.  If names is not None then
            totim, time_step, stress_period, and selection(s) will be returned.
            (default is None).

        Returns
        -------
        out : recarray
            Numpy recarray with the water budget items in list file. The
            recarray also includes totim, time_step, and stress_period.

        Examples
        --------
        >>> mf_list = MfListBudget("my_model.list")
        >>> cumulative = mf_list.get_cumulative()

        """
        if not self._isvalid:
            return None
        if names is None:
            return self.cum
        else:
            if not isinstance(names, list):
                names = [names]
            names.insert(0, "stress_period")
            names.insert(0, "time_step")
            names.insert(0, "totim")
            return np.array(self.cum)[names].view(np.recarray)

    def get_model_runtime(self, units="seconds"):
        """
        Get the elapsed runtime of the model from the list file.

        Parameters
        ----------
        units : str
            Units in which to return the runtime. Acceptable values are
            'seconds', 'minutes', 'hours' (default is 'seconds')

        Returns
        -------
        out : float
            Floating point value with the runtime in requested units. Returns
            NaN if runtime not found in list file

        Examples
        --------
        >>> mf_list = MfListBudget("my_model.list")
        >>> budget = mf_list.get_model_runtime(units='hours')
        """
        if not self._isvalid:
            return None

        # reopen the file
        self.f = open(self.file_name, "r", encoding="ascii", errors="replace")
        units = units.lower()
        if (
            not units == "seconds"
            and not units == "minutes"
            and not units == "hours"
        ):
            err = (
                '"units" input variable must be "minutes", "hours", '
                'or "seconds": {0} was specified'.format(units)
            )
            raise AssertionError(err)
        try:
            seekpoint = self._seek_to_string("Elapsed run time:")
        except:
            print("Elapsed run time not included in list file. Returning NaN")
            return np.nan

        self.f.seek(seekpoint)
        line = self.f.readline()

        self.f.close()
        # yank out the floating point values from the Elapsed run time string
        times = list(map(float, re.findall(r"[+-]?[0-9.]+", line)))
        # pad an array with zeros and times with
        # [days, hours, minutes, seconds]
        times = np.array([0 for _ in range(4 - len(times))] + times)
        # convert all to seconds
        time2sec = np.array([24 * 60 * 60, 60 * 60, 60, 1])
        times_sec = np.sum(times * time2sec)
        # return in the requested units
        if units == "seconds":
            return times_sec
        elif units == "minutes":
            return times_sec / 60.0
        elif units == "hours":
            return times_sec / 60.0 / 60.0

    def get_budget(self, names=None):
        """
        Get the recarrays with the incremental and cumulative water budget
        items in the list file.

        Parameters
        ----------
        names : str or list of strings
            Selection of column names to return.  If names is not None then
            totim, time_step, stress_period, and selection(s) will be returned.
            (default is None).

        Returns
        -------
        out : recarrays
            Numpy recarrays with the water budget items in list file. The
            recarray also includes totim, time_step, and stress_period. A
            separate recarray is returned for the incremental and cumulative
            water budget entries.

        Examples
        --------
        >>> mf_list = MfListBudget("my_model.list")
        >>> budget = mf_list.get_budget()

        """
        if not self._isvalid:
            return None
        if names is None:
            return self.inc, self.cum
        else:
            if not isinstance(names, list):
                names = [names]
            names.insert(0, "stress_period")
            names.insert(0, "time_step")
            names.insert(0, "totim")
            return (
                self.inc[names].view(np.recarray),
                self.cum[names].view(np.recarray),
            )

    def get_data(self, kstpkper=None, idx=None, totim=None, incremental=False):
        """
        Get water budget data from the list file for the specified conditions.

        Parameters
        ----------
        idx : int
            The zero-based record number.  The first record is record 0.
            (default is None).
        kstpkper : tuple of ints
            A tuple containing the time step and stress period (kstp, kper).
            These are zero-based kstp and kper values. (default is None).
        totim : float
            The simulation time. (default is None).
        incremental : bool
            Boolean flag used to determine if incremental or cumulative water
            budget data for the specified conditions will be returned. If
            incremental=True, incremental water budget data will be returned.
            If incremental=False, cumulative water budget data will be
            returned. (default is False).

        Returns
        -------
        data : numpy recarray
            Array has size (number of budget items, 3). Recarray names are
            'index', 'value', 'name'.

        See Also
        --------

        Notes
        -----
        if both kstpkper and totim are None, will return the last entry

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> import flopy
        >>> mf_list = flopy.utils.MfListBudget("my_model.list")
        >>> data = mf_list.get_data(kstpkper=(0,0))
        >>> plt.bar(data['index'], data['value'])
        >>> plt.xticks(data['index'], data['name'], rotation=45, size=6)
        >>> plt.show()

        """
        if not self._isvalid:
            return None
        ipos = None
        if kstpkper is not None:
            try:
                ipos = self.get_kstpkper().index(kstpkper)
            except:
                print(
                    "   could not retrieve kstpkper "
                    "{} from the lst file".format(kstpkper)
                )
        elif totim is not None:
            try:
                ipos = self.get_times().index(totim)
            except:
                print(
                    "   could not retrieve totime "
                    "{} from the lst file".format(totim)
                )
        elif idx is not None:
            ipos = idx
        else:
            ipos = -1

        if ipos is None:
            print("Could not find specified condition.")
            print("  kstpkper = {}".format(kstpkper))
            print("  totim = {}".format(totim))
            # TODO: return zero-length array, or update docstring return type
            return None

        if incremental:
            t = self.inc[ipos]
        else:
            t = self.cum[ipos]

        dtype = np.dtype(
            [("index", np.int32), ("value", np.float32), ("name", "|S25")]
        )
        v = np.recarray(shape=(len(self.inc.dtype.names[3:])), dtype=dtype)
        for i, name in enumerate(self.inc.dtype.names[3:]):
            mult = 1.0
            if "_OUT" in name:
                mult = -1.0
            v[i]["index"] = i
            v[i]["value"] = mult * t[name]
            v[i]["name"] = name
        return v

    def get_dataframes(self, start_datetime="1-1-1970", diff=False):
        """
        Get pandas dataframes with the incremental and cumulative water budget
        items in the list file.

        Parameters
        ----------
        start_datetime : str
            If start_datetime is passed as None, the rows are indexed on totim.
            Otherwise, a DatetimeIndex is set. (default is 1-1-1970).

        Returns
        -------
        out : pandas dataframes
            Pandas dataframes with the incremental and cumulative water budget
            items in list file. A separate pandas dataframe is returned for the
            incremental and cumulative water budget entries.

        Examples
        --------
        >>> mf_list = MfListBudget("my_model.list")
        >>> incrementaldf, cumulativedf = mf_list.get_dataframes()

        """

        try:
            import pandas as pd
        except Exception as e:
            msg = "ListBudget.get_dataframe(): requires pandas: " + str(e)
            raise ImportError(msg)

        if not self._isvalid:
            return None
        totim = self.get_times()
        if start_datetime is not None:
            totim = totim_to_datetime(
                totim,
                start=pd.to_datetime(start_datetime),
                timeunit=self.timeunit,
            )

        df_flux = pd.DataFrame(self.inc, index=totim).loc[:, self.entries]
        df_vol = pd.DataFrame(self.cum, index=totim).loc[:, self.entries]

        if not diff:
            return df_flux, df_vol

        else:
            in_names = [col for col in df_flux.columns if col.endswith("_IN")]

            base_names = [name.replace("_IN", "") for name in in_names]
            for name in base_names:
                in_name = name + "_IN"
                out_name = name + "_OUT"
                df_flux.loc[:, name.lower()] = (
                    df_flux.loc[:, in_name] - df_flux.loc[:, out_name]
                )
                df_flux.pop(in_name)
                df_flux.pop(out_name)
                df_vol.loc[:, name.lower()] = (
                    df_vol.loc[:, in_name] - df_vol.loc[:, out_name]
                )
                df_vol.pop(in_name)
                df_vol.pop(out_name)
            cols = list(df_flux.columns)
            cols = [col.lower() for col in cols]
            df_flux.columns = cols
            df_vol.columns = cols
            df_flux.sort_index(axis=1, inplace=True)
            df_vol.sort_index(axis=1, inplace=True)
            return df_flux, df_vol

    def get_reduced_pumping(self):
        """
        Get numpy recarray of reduced pumping data from a list file.
        Reduced pumping data most have been written to the list file
        during the model run. Works with MfListBudget and MfusgListBudget.

        Returns
        -------
        numpy recarray
            A numpy recarray with the reduced pumping data from the list
            file.

        Example
        --------
        >>> objLST = MfListBudget("my_model.lst")
        >>> raryReducedPpg = objLST.get_reduced_pumping()
        >>> dfReducedPpg = pd.DataFrame.from_records(raryReducedPpg)

        """
        from ..utils.observationfile import get_reduced_pumping

        # Ensure list file exists
        if not os.path.isfile(self.f.name):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), self.f.name
            )

        # Eval based on model list type
        if isinstance(self, MfListBudget):
            structured = True
            # Check if reduced pumping data was set to be written
            # to list file
            check_str = (
                "WELLS WITH REDUCED PUMPING WILL BE REPORTED "
                "TO THE MAIN LISTING FILE"
            )

            check_str_ag = "AG WELLS WITH REDUCED PUMPING FOR STRESS PERIOD"

            if (
                not open(self.f.name).read().find(check_str) > 0
                and not open(self.f.name).read().find(check_str_ag) > 0
            ):
                err = (
                    "Pumping reductions not written to list file. "
                    'Try removing "noprint" keyword from well file.'
                    "External pumping reduction files can be read using: "
                    "flopy.utils.observationfile.get_pumping_reduction(<file>)"
                )
                raise AssertionError(err)

        elif isinstance(self, MfusgListBudget):
            structured = False
            # Check if reduced pumping data was written and if set to
            # be written to list file
            check_str = "WELL REDUCTION INFO WILL BE WRITTEN TO UNIT:"
            bool_list_unit = False
            pump_reduction_flag = False
            for line in open(self.f.name):
                # Assumes LST unit always first
                if "UNIT" in line and not bool_list_unit:
                    list_unit = int(line.strip().split()[-1])
                    bool_list_unit = True
                if check_str in line:
                    pump_reduction_flag = True

                    if int(line.strip().split()[-1]) != list_unit:
                        raise AssertionError(
                            "Pumping reductions not written to list file. "
                            "External pumping reduction files can be "
                            "read using: flopy.utils.observationfile."
                            "get_pumping_reduction(<file>, structured=False)"
                        )

            if not pump_reduction_flag:
                raise AssertionError("Auto pumping reductions not active.")

        else:
            raise NotImplementedError(
                "get_reduced_pumping() is only implemented for the "
                "MfListBudget or MfusgListBudget classes. Please "
                "feel free to expand the functionality to other "
                "ListBudget classes."
            )

        return get_reduced_pumping(self.f.name, structured)

    def _build_index(self, maxentries):
        self.idx_map = self._get_index(maxentries)
        return

    def _get_index(self, maxentries):
        # --parse through the file looking for matches and parsing ts and sp
        idxs = []
        l_count = 1
        while True:
            seekpoint = self.f.tell()
            line = self.f.readline()
            if line == "":
                break
            if self.budgetkey in line:
                for _ in range(self.tssp_lines):
                    line = self.f.readline()
                try:
                    ts, sp = get_ts_sp(line)
                except:
                    print(
                        "unable to cast ts,sp on line number",
                        l_count,
                        " line: ",
                        line,
                    )
                    break
                # print('info found for timestep stress period',ts,sp)

                idxs.append([ts, sp, seekpoint])

                if maxentries and len(idxs) >= maxentries:
                    break

        return idxs

    def _seek_to_string(self, s):
        """
        Parameters
        ----------
        s : str
            Seek through the file to the next occurrence of s.  Return the
            seek location when found.

        Returns
        -------
        seekpoint : int
            Next location of the string

        """
        while True:
            seekpoint = self.f.tell()
            line = self.f.readline()
            if line == "":
                break
            if s in line:
                break
        return seekpoint

    def _set_entries(self):
        if len(self.idx_map) < 1:
            return None, None
        if len(self.entries) > 0:
            raise Exception("entries already set:" + str(self.entries))
        if not self.idx_map:
            raise Exception("must call build_index before call set_entries")
        try:
            incdict, cumdict = self._get_sp(
                self.idx_map[0][0], self.idx_map[0][1], self.idx_map[0][2]
            )
        except:
            raise Exception(
                "unable to read budget information from first "
                "entry in list file"
            )
        self.entries = incdict.keys()
        null_entries = collections.OrderedDict()
        incdict = collections.OrderedDict()
        cumdict = collections.OrderedDict()
        for entry in self.entries:
            incdict[entry] = []
            cumdict[entry] = []
            null_entries[entry] = np.NaN
        self.null_entries = [null_entries, null_entries]
        return incdict, cumdict

    def _load(self, maxentries=None):
        self._build_index(maxentries)
        incdict, cumdict = self._set_entries()
        if incdict is None and cumdict is None:
            return
        totim = []
        for ts, sp, seekpoint in self.idx_map:
            tinc, tcum = self._get_sp(ts, sp, seekpoint)
            for entry in self.entries:
                incdict[entry].append(tinc[entry])
                cumdict[entry].append(tcum[entry])

            # Get the time for this record
            seekpoint = self._seek_to_string("TIME SUMMARY AT END")
            tslen, sptim, tt = self._get_totim(ts, sp, seekpoint)
            totim.append(tt)

        # get kstp and kper
        idx_array = np.array(self.idx_map)

        # build dtype for recarray
        dtype_tups = [
            ("totim", np.float32),
            ("time_step", np.int32),
            ("stress_period", np.int32),
        ]
        for entry in self.entries:
            dtype_tups.append((entry, np.float32))
        dtype = np.dtype(dtype_tups)

        # create recarray
        nentries = len(incdict[entry])
        self.inc = np.recarray(shape=(nentries,), dtype=dtype)
        self.cum = np.recarray(shape=(nentries,), dtype=dtype)

        # fill each column of the recarray
        for entry in self.entries:
            self.inc[entry] = incdict[entry]
            self.cum[entry] = cumdict[entry]

        # file the totim, time_step, and stress_period columns for the
        # incremental and cumulative recarrays (zero-based kstp,kper)
        self.inc["totim"] = np.array(totim)[:]
        self.inc["time_step"] = idx_array[:, 0] - 1
        self.inc["stress_period"] = idx_array[:, 1] - 1

        self.cum["totim"] = np.array(totim)[:]
        self.cum["time_step"] = idx_array[:, 0] - 1
        self.cum["stress_period"] = idx_array[:, 1] - 1

        return

    def _get_sp(self, ts, sp, seekpoint):
        self.f.seek(seekpoint)
        # --read to the start of the "in" budget information
        while True:
            line = self.f.readline()
            if line == "":
                print(
                    "end of file found while seeking budget "
                    "information for ts,sp: {} {}".format(ts, sp)
                )
                return self.null_entries

            # --if there are two '=' in this line, then it is a budget line
            if len(re.findall("=", line)) == 2:
                break

        tag = "IN"
        incdict = collections.OrderedDict()
        cumdict = collections.OrderedDict()
        entrydict = {}
        while True:

            if line == "":
                print(
                    "end of file found while seeking budget "
                    "information for ts,sp: {} {}".format(ts, sp)
                )
                return self.null_entries
            if len(re.findall("=", line)) == 2:
                try:
                    entry, flux, cumu = self._parse_budget_line(line)
                except Exception:
                    print("error parsing budget line in ts,sp", ts, sp)
                    return self.null_entries
                if flux is None:
                    print(
                        "error casting in flux for",
                        entry,
                        " to float in ts,sp",
                        ts,
                        sp,
                    )
                    return self.null_entries
                if cumu is None:
                    print(
                        "error casting in cumu for",
                        entry,
                        " to float in ts,sp",
                        ts,
                        sp,
                    )
                    return self.null_entries
                if entry.endswith(tag.upper()):
                    if " - " in entry.upper():
                        key = entry.replace(" ", "")
                    else:
                        key = entry.replace(" ", "_")
                elif "PERCENT DISCREPANCY" in entry.upper():
                    key = entry.replace(" ", "_")
                else:
                    entry = entry.replace(" ", "_")
                    if entry in entrydict:
                        entrydict[entry] += 1
                        inum = entrydict[entry]
                        entry = "{}{}".format(entry, inum + 1)
                    else:
                        entrydict[entry] = 0
                    key = "{}_{}".format(entry, tag)
                incdict[key] = flux
                cumdict[key] = cumu
            else:
                if "OUT:" in line.upper():
                    tag = "OUT"
                    entrydict = {}
            line = self.f.readline()
            if entry.upper() == "PERCENT DISCREPANCY":
                break

        return incdict, cumdict

    def _parse_budget_line(self, line):

        # get the budget item name
        entry = line.strip().split("=")[0].strip()

        # get the cumulative string
        idx = line.index("=") + 1
        line2 = line[idx:]
        ll = line2.strip().split()
        cu_str = ll[0]

        idx = line2.index("=") + 1
        fx_str = line2[idx:].split()[0].strip()

        flux, cumu = None, None
        try:
            cumu = float(cu_str)
        except:
            if "NAN" in cu_str.strip().upper():
                cumu = np.NaN
        try:
            flux = float(fx_str)
        except:
            if "NAN" in fx_str.strip().upper():
                flux = np.NaN
        return entry, flux, cumu

    def _get_totim(self, ts, sp, seekpoint):
        self.f.seek(seekpoint)
        # --read header lines
        ihead = 0
        while True:
            line = self.f.readline()
            ihead += 1
            if line == "":
                print(
                    "end of file found while seeking budget "
                    "information for ts,sp: {} {}".format(ts, sp)
                )
                return np.NaN, np.NaN, np.NaN
            elif (
                ihead == 2
                and "SECONDS     MINUTES      HOURS       DAYS        YEARS"
                not in line
            ):
                break
            elif (
                "-----------------------------------------------------------"
                in line
            ):
                line = self.f.readline()
                break

        if isinstance(self, SwtListBudget):
            translen = self._parse_time_line(line)
            line = self.f.readline()
            if translen is None:
                print("error parsing translen for ts,sp", ts, sp)
                return np.NaN, np.NaN, np.NaN

        tslen = self._parse_time_line(line)
        if tslen is None:
            print("error parsing tslen for ts,sp", ts, sp)
            return np.NaN, np.NaN, np.NaN

        sptim = self._parse_time_line(self.f.readline())
        if sptim is None:
            print("error parsing sptim for ts,sp", ts, sp)
            return np.NaN, np.NaN, np.NaN

        totim = self._parse_time_line(self.f.readline())
        if totim is None:
            print("error parsing totim for ts,sp", ts, sp)
            return np.NaN, np.NaN, np.NaN
        return tslen, sptim, totim

    def _parse_time_line(self, line):
        if line == "":
            print("end of file found while parsing time information")
            return None
        try:
            time_str = line[self.time_line_idx :]
            raw = time_str.split()
            idx = self.time_idx
            # catch case where itmuni is undefined
            # in this case, the table format is different
            try:
                v = float(raw[0])
            except:
                time_str = line[45:]
                raw = time_str.split()
                idx = 0
            tval = float(raw[idx])
        except:
            print("error parsing tslen information: ", time_str)
            return None
        return tval


class SwtListBudget(ListBudget):
    """ """

    def set_budget_key(self):
        self.budgetkey = "MASS BUDGET FOR ENTIRE MODEL"
        return


class MfListBudget(ListBudget):
    """ """

    def set_budget_key(self):
        self.budgetkey = "VOLUMETRIC BUDGET FOR ENTIRE MODEL"
        return


class Mf6ListBudget(ListBudget):
    """ """

    def set_budget_key(self):
        self.budgetkey = "VOLUME BUDGET FOR ENTIRE MODEL"
        return


class MfusgListBudget(ListBudget):
    """ """

    def set_budget_key(self):
        self.budgetkey = "VOLUMETRIC BUDGET FOR ENTIRE MODEL"
        return


class SwrListBudget(ListBudget):
    """ """

    def set_budget_key(self):
        self.budgetkey = "VOLUMETRIC SURFACE WATER BUDGET FOR ENTIRE MODEL"
        self.tssp_lines = 1
        return
