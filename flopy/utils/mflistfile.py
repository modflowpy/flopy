import collections
import os
import re
import sys
from datetime import timedelta

import numpy as np


class ListBudget(object):
    """
    MODFLOW family list file handling

    Parameters:
    ----------
        file_name : (str) the list file name

    Methods:
    ----------
        get_record_names : returns a list of water budget items in the list file.
        The names also include totim, stress period, and time step.

        get_times : returns a list of unique water budget times in the list file.

        get_kstpkper : returns a list of unique stress periods and time steps in
        the list file water budgets.

        get_incremental : returns a numpy.recarray for all cumulative water budget
        entries in the list file budget.  The columns include totim, stress period,
        and time step.

        get_cumulative : returns a numpy.recarray for all cumulative water budget
        entries in the list file budget.  The columns include totim, stress period,
        and time step.

        get_budget : returns incremental, cumulative numpy.recarrays for all entries
        in the list file budget.  The columns include totim, stress period, and time step.

        get_data : returns a numpy.recarray with water budget data from the list file
        for the specified conditions. The numpy.recarray includes index, value, and
        name columns.

        get_dataframes(start_datetime='1-1-1970') : returns incremental and cumulative
        water budget dateframes.  If start_datetime is passed as none, the rows are indexed
        on totim.  Otherwise, a DatetimeIndex is set.

    Note:
    ----
        The ListBudget class should not be instantiated directly.  Access is
        through derived classes: MfListBudget (MODFLOW), SwtListBudget (SEAWAT)
        and SwrListBudget (MODFLOW with the SWR process)

    Example:
    -------
        >>> mf_list = MfListBudget("my_model.list")
        >>> incremental, cumulative = mf_list.get_budget()
        >>> df_in, df_out = mf_list.get_dataframes(start_datetime="10-21-2015")

    """

    def __init__(self, file_name):
        raise Exception('base class lstbudget does not have a " +\
        "constructor - must call a derived class')

    def get_record_names(self):
        """
        Get a list of water budget record names in the file

        Returns
        ----------
        out : list of strings
            List of unique text names in the binary file.

        Example:
        -------
            >>> mf_list = MfListBudget("my_model.list")
            >>> names = mf_list.get_record_names()

        """
        return self.inc.dtype.names

    def get_times(self):
        """
        Get a list of unique water budget times in the list file

        Returns
        ----------
        out : list of floats
            List contains unique water budget simulation times (totim) in list file.


        Example:
        -------
            >>> mf_list = MfListBudget("my_model.list")
            >>> times = mf_list.get_times()

        """
        return self.inc['totim'].tolist()

    def get_kstpkper(self):
        """
        Get a list of unique stress periods and time steps in the list file
        water budgets.

        Returns
        ----------
        out : list of (kstp, kper) tuples
            List of unique kstp, kper combinations in list file.  kstp and
            kper values are zero-based.


        Example:
        -------
            >>> mf_list = MfListBudget("my_model.list")
            >>> kstpkper = mf_list.get_kstpkper()

        """
        kstpkper = []
        for kstp, kper in zip(self.inc['time_step'], self.inc['stress_period']):
            kstpkper.append((kstp, kper))
        return kstpkper

    def get_incremental(self, names=None):
        """
        Get a recarray with the incremental water budget items in the list file

        Parameters
        ----------
        names : str or list of strings
            Selection of column names to return.  If names is not None then
            totim, time_step, stress_period, and selection(s) will be returned.
            (default is None).

        Returns
        ----------
        out : recarray
            Numpy recarray with the water budget items in list file. The
            recarray also includes totim, time_step, and stress_period.


        Example:
        -------
            >>> mf_list = MfListBudget("my_model.list")
            >>> incremental = mf_list.get_incremental()
        """
        if names is None:
            return self.inc
        else:
            if not isinstance(names, list):
                names = [names]
            names.insert(0, 'stress_period')
            names.insert(0, 'time_step')
            names.insert(0, 'totim')
            return self.inc[names].view(np.recarray)

    def get_cumulative(self, names=None):
        """
        Get a recarray with the cumulative water budget items in the list file

        Parameters
        ----------
        names : str or list of strings
            Selection of column names to return.  If names is not None then
            totim, time_step, stress_period, and selection(s) will be returned.
            (default is None).

        Returns
        ----------
        out : recarray
            Numpy recarray with the water budget items in list file. The
            recarray also includes totim, time_step, and stress_period.


        Example:
        -------
            >>> mf_list = MfListBudget("my_model.list")
            >>> cumulative = mf_list.get_cumulative()
       """
        if names is None:
            return self.cum
        else:
            if not isinstance(names, list):
                names = [names]
            names.insert(0, 'stress_period')
            names.insert(0, 'time_step')
            names.insert(0, 'totim')
            return self.cum[names].view(np.recarray)

    def get_budget(self, names=None):
        """
        Get the recarrays with the incremental and cumulative water budget items
        in the list file

        Parameters
        ----------
        names : str or list of strings
            Selection of column names to return.  If names is not None then
            totim, time_step, stress_period, and selection(s) will be returned.
            (default is None).

        Returns
        ----------
        out : recarrays
            Numpy recarrays with the water budget items in list file. The
            recarray also includes totim, time_step, and stress_period. A
            separate recarray is returned for the incremental and cumulative
            water budget entries.


        Example:
        -------
            >>> mf_list = MfListBudget("my_model.list")
            >>> budget = mf_list.get_budget()
        """
        if names is None:
            return self.inc, self.cum
        else:
            if not isinstance(names, list):
                names = [names]
            names.insert(0, 'stress_period')
            names.insert(0, 'time_step')
            names.insert(0, 'totim')
            return self.inc[names].view(np.recarray), self.cum[names].view(np.recarray)

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
        ----------
        data : numpy recarray
            Array has size (number of budget items, 3). Recarray names are 'index',
            'value', 'name'.

        See Also
        --------

        Notes
        -----
        if both kstpkper and totim are None, will return the last entry

        Example:
        -------
            >>> import matplotlib.pyplot as plt
            >>> import flopy
            >>> mf_list = flopy.utils.MfListBudget("my_model.list")
            >>> data = mf_list.get_data(kstpkper=(0,0))
            >>> plt.bar(data['index'], data['value'])
            >>> plt.xticks(data['index'], data['name'], rotation=45, size=6)
            >>> plt.show()

        """
        ipos = None
        if kstpkper is not None:
            try:
                ipos = self.get_kstpkper().index(kstpkper)
            except:
                pass
        elif totim is not None:
            try:
                ipos = self.get_times().index(totim)
            except:
                pass
        elif idx is not None:
            ipos = idx
        else:
            ipos = -1

        if ipos is None:
            print('Could not find specified condition.')
            print('  kstpkper = {}'.format(kstpkper))
            print('  totim = {}'.format(totim))
            return None

        if incremental:
            t = self.inc[ipos]
        else:
            t = self.cum[ipos]

        dtype = np.dtype([('index', np.int32), ('value', np.float32), ('name', '|S25')])
        v = np.recarray(shape=(len(self.inc.dtype.names[3:])), dtype=dtype)
        for i, name in enumerate(self.inc.dtype.names[3:]):
            mult = 1.
            if '_OUT' in name:
                mult = -1.
            v[i]['index'] = i
            v[i]['value'] = mult * t[name]
            v[i]['name'] = name
        return v

    def get_dataframes(self, start_datetime='1-1-1970'):
        """
        Get pandas dataframes with the incremental and cumulative water budget
        items in the list file

        Parameters
        ----------
        start_datetime : str
            If start_datetime is passed as None, the rows are indexed on totim.
            Otherwise, a DatetimeIndex is set. (default is 1-1-1970).

        Returns
        ----------
        out : panda dataframes
            Pandas dataframes with the incremental and cumulative water budget
            items in list file. A separate pandas dataframe is returned for the
            incremental and cumulative water budget entries.


        Example:
        -------
            >>> mf_list = MfListBudget("my_model.list")
            >>> incrementaldf, cumulativedf = mf_list.get_dataframes()
        """

        try:
            import pandas as pd
        except Exception as e:
            raise Exception("ListBudget.get_dataframe() error import pandas: " + \
                            str(e))

            # so we can get a datetime index for the dataframe
        if start_datetime is not None:
            lt = ListTime(self.file_name, start=pd.to_datetime(start_datetime))
        else:
            # idx = pd.MultiIndex.from_tuples(list(zip(fin["stress_period"], fin["time_step"])))
            lt = ListTime(self.file_name, start=pd.to_datetime(start_datetime))
        idx = lt.get_times()

        df_flux = pd.DataFrame(self.inc, index=idx).loc[:, self.entries]

        df_vol = pd.DataFrame(self.cum, index=idx).loc[:, self.entries]
        return df_flux, df_vol


    def _build_index(self, maxentries):
        # print('building index...')
        self.idx_map = self._get_index(maxentries)
        # print('\ndone - found',len(self.idx_map),'entries')


    def _get_index(self, maxentries):
        # --parse through the file looking for matches and parsing ts and sp
        idxs = []
        l_count = 1
        while True:
            seekpoint = self.f.tell()
            line = self.f.readline()
            if line == '':
                break
            if self.lstkey in line:
                for l in range(self.tssp_lines):
                    line = self.f.readline()
                try:
                    ts, sp = self._get_ts_sp(line)
                except:
                    print('unable to cast ts,sp on line number', l_count, ' line: ', line)
                    break
                # print('info found for timestep stress period',ts,sp)

                idxs.append([ts, sp, seekpoint])

                if maxentries and len(idxs) >= maxentries:
                    break

        return idxs


    def _get_ts_sp(self, line):
        ts = int(line[self.ts_idxs[0]:self.ts_idxs[1]])
        sp = int(line[self.sp_idxs[0]:self.sp_idxs[1]])
        return ts, sp


    def _set_entries(self):
        if len(self.entries) > 0:
            raise Exception('entries already set:' + str(self.entries))
        if not self.idx_map:
            raise Exception('must call build_index before call set_entries')
        try:
            incdict, cumdict = self._get_sp(self.idx_map[0][0],
                                            self.idx_map[0][1],
                                            self.idx_map[0][2])
        except:
            raise Exception('unable to read budget information from first entry in list file')
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
        for ts, sp, seekpoint in self.idx_map:
            tinc, tcum = self._get_sp(ts, sp, seekpoint)
            for entry in self.entries:
                incdict[entry].append(tinc[entry])
                cumdict[entry].append(tcum[entry])

        # get kstp and kper
        idx_array = np.array(self.idx_map)

        # get totime
        lt = ListTime(self.file_name, start=None, timeunit=self.timeunit)
        totim = lt.get_times()

        # build rec arrays
        dtype_tups = [('totim', np.float32), ("time_step", np.int32), ("stress_period", np.int32)]
        for entry in self.entries:
            dtype_tups.append((entry, np.float32))
        dtype = np.dtype(dtype_tups)
        nentries = len(incdict[entry])
        self.inc = np.recarray(shape=(nentries,), dtype=dtype)
        self.cum = np.recarray(shape=(nentries,), dtype=dtype)
        for entry in self.entries:
            self.inc[entry] = incdict[entry]
            self.cum[entry] = cumdict[entry]
        self.inc['totim'], self.cum['totim'] = np.array(totim)[:], np.array(totim)[:]
        # zero based time_step (kstp) and stress_period (kper)
        self.inc["time_step"], self.inc["stress_period"] = idx_array[:, 0] - 1, idx_array[:, 1] - 1
        self.cum["time_step"], self.cum["stress_period"] = idx_array[:, 0] - 1, idx_array[:, 1] - 1

        return


    def _get_sp(self, ts, sp, seekpoint):
        self.f.seek(seekpoint)
        # --read to the start of the "in" budget information
        while True:
            line = self.f.readline()
            if line == '':
                # raise Exception('end of file found while seeking budget information')
                print('end of file found while seeking budget information for ts,sp', ts, sp)
                return self.null_entries

            # --if there are two '=' in this line, then it is a budget line
            if len(re.findall('=', line)) == 2:
                break

        tag = 'IN'
        incdict, cumdict = collections.OrderedDict(), collections.OrderedDict()
        while True:

            if line == '':
                # raise Exception('end of file found while seeking budget information')
                print('end of file found while seeking budget information for ts,sp', ts, sp)
                return self.null_entries
            if len(re.findall('=', line)) == 2:
                try:
                    entry, flux, cumu = self._parse_budget_line(line)
                except e:
                    print('error parsing budget line in ts,sp', ts, sp)
                    return self.null_entries
                if flux is None:
                    print('error casting in flux for', entry, ' to float in ts,sp', ts, sp)
                    return self.null_entries
                if cumu is None:
                    print('error casting in cumu for', entry, ' to float in ts,sp', ts, sp)
                    return self.null_entries
                if tag.upper() in entry:
                    if ' - ' in entry.upper():
                        key = entry.replace(' ', '')
                    else:
                        key = entry.replace(' ', '_')
                elif 'PERCENT DISCREPANCY' in entry.upper():
                    key = entry.replace(' ', '_')
                else:
                    key = '{}_{}'.format(entry.replace(' ', '_'), tag)
                incdict[key] = flux
                cumdict[key] = cumu
            else:
                if 'OUT:' in line.upper():
                    tag = 'OUT'
            line = self.f.readline()
            if entry.upper() == 'PERCENT DISCREPANCY':
                break

        return incdict, cumdict


    def _parse_budget_line(self, line):
        raw = line.strip().split()
        entry = line.strip().split('=')[0].strip()
        cu_str = line[self.cumu_idxs[0]:self.cumu_idxs[1]]
        fx_str = line[self.flux_idxs[0]:self.flux_idxs[1]]
        flux, cumu = None, None
        try:
            cumu = float(cu_str)
        except:
            if 'NAN' in cu_str.strip().upper():
                cumu = np.NaN
        try:
            flux = float(fx_str)
        except:
            if 'NAN' in fx_str.strip().upper():
                flux = np.NaN
        return entry, flux, cumu


class SwtListBudget(ListBudget):
    """

    """
    def __init__(self, file_name, key_string='MASS BUDGET FOR ENTIRE MODEL',
                 timeunit='days'):
        assert os.path.exists(file_name)
        self.file_name = file_name
        if sys.version_info[0] == 2:
            self.f = open(file_name, 'r')
        elif sys.version_info[0] == 3:
            self.f = open(file_name, 'r', encoding='ascii', errors='replace')
        self.timeunit = timeunit
        self.lstkey = key_string
        self.idx_map = []
        self.entries = []
        self.null_entries = []
        self.cumu_idxs = [22, 40]
        self.flux_idxs = [63, 80]
        self.ts_idxs = [50, 54]
        self.sp_idxs = [70, 75]
        self.tssp_lines = 0
        # set budget recarrays
        self._load()


class MfListBudget(ListBudget):
    """

    """
    def __init__(self, file_name, key_string='VOLUMETRIC BUDGET FOR ENTIRE MODEL',
                 timeunit='days'):
        assert os.path.exists(file_name)
        self.file_name = file_name
        if sys.version_info[0] == 2:
            self.f = open(file_name, 'r')
        elif sys.version_info[0] == 3:
            self.f = open(file_name, 'r', encoding='ascii', errors='replace')
        self.timeunit = timeunit
        self.lstkey = key_string
        self.idx_map = []
        self.entries = []
        self.null_entries = []
        self.cumu_idxs = [22, 40]
        self.flux_idxs = [63, 80]
        self.ts_idxs = [56, 61]
        self.sp_idxs = [76, 80]
        self.tssp_lines = 0
        # set budget recarrays
        self._load()


class SwrListBudget(ListBudget):
    """

    """
    def __init__(self, file_name, key_string='VOLUMETRIC SURFACE WATER BUDGET FOR ENTIRE MODEL',
                 timeunit='days'):
        assert os.path.exists(file_name)
        self.file_name = file_name
        if sys.version_info[0] == 2:
            self.f = open(file_name, 'r')
        elif sys.version_info[0] == 3:
            self.f = open(file_name, 'r', encoding='ascii', errors='replace')
        self.timeunit = timeunit
        self.lstkey = key_string
        self.idx_map = []
        self.entries = []
        self.null_entries = []
        self.cumu_idxs = [25, 43]
        self.flux_idxs = [66, 84]
        self.ts_idxs = [39, 46]
        self.sp_idxs = [62, 68]
        self.tssp_lines = 1
        # set budget recarrays
        self._load()


class ListTime(ListBudget):
    """
    Class to extract time information from lst file
    passing a start datetime results in casting the totim to dts from start

    Parameters:
    ----------


    Methods:
    ----------
        get_times : returns a list of unique water budget times in the list file.

    Note:
    ----
        The ListBudget class should not be instantiated directly.  Access is
        through derived classes: MfListBudget (MODFLOW), SwtListBudget (SEAWAT)
        and SwrListBudget (MODFLOW with the SWR process)

    Example:
    -------
        >>> mf_listtime = ListTime("my_model.list")
        >>> times = mf_listtime.get_times()
    """

    def __init__(self, file_name, timeunit='days', key_str='TIME SUMMARY AT END',
                 start=None, flow=True):

        assert os.path.exists(file_name)
        self.file_name = file_name
        if sys.version_info[0] == 2:
            self.f = open(file_name, 'r')
        elif sys.version_info[0] == 3:
            self.f = open(file_name, 'r', encoding='ascii', errors='replace')
        self.idx_map = []
        self.tslen = []
        self.sptim = []
        self.totim = []
        # self.lstkey = re.compile(key_str)
        self.lstkey = key_str
        self.tssp_lines = 0
        if flow:
            self.ts_idxs = [42, 47]
            self.sp_idxs = [63, 69]
        else:
            self.ts_idxs = [65, 71]
            self.sp_idxs = [87, 92]
        self.time_line_idx = 20
        if timeunit.upper() == 'SECONDS':
            self.timeunit = 'S'
            self.time_idx = 0
        elif timeunit.upper() == 'MINUTES':
            self.timeunit = 'M'
            self.time_idx = 1
        elif timeunit.upper() == 'HOURS':
            self.timeunit = 'H'
            self.time_idx = 2
        elif timeunit.upper() == 'DAYS':
            self.timeunit = 'D'
            self.time_idx = 3
        elif timeunit.upper() == 'YEARS':
            self.timeunit = 'Y'
            self.time_idx = 4
        else:
            raise Exception('need to reset time_idxs attribute to ' + \
                            'use units other than days and check usage of timedelta')
        self.null_entries = [np.NaN, np.NaN, np.NaN]
        self.start = start
        if start:
            self.dt = []

        # load the data
        self._load()

    def get_times(self):
        """
        Get a list of unique water budget times in the list file

        Returns
        ----------
        out : list of floats
            List contains unique water budget simulation times (totim) in list file.

        """
        return self.totim

    def _load(self, maxentries=None):
        self._build_index(maxentries)

        for i, [ts, sp, seekpoint] in enumerate(self.idx_map):
            # print 'loading stress period, timestep',sp,ts,

            tslen, sptim, totim = self._get_sp(ts, sp, seekpoint)
            self.tslen.append(tslen)
            self.sptim.append(sptim)
            self.totim.append(totim)
        if self.start is not None:
            self.dt = self.cast_totim()
        return

    def _cast_totim(self):
        if self.timeunit == 'S':
            totim = []
            for to in self.totim:
                t = timedelta(seconds=to)
                totim.append(self.start + t)
        elif self.timeunit == 'M':
            totim = []
            for to in self.totim:
                t = timedelta(minutes=to)
                totim.append(self.start + t)
        elif self.timeunit == 'H':
            totim = []
            for to in self.totim:
                t = timedelta(hours=to)
                totim.append(self.start + t)
        elif self.timeunit == 'D':
            totim = []
            for to in self.totim:
                t = timedelta(days=to)
                totim.append(self.start + t)
        elif self.timeunit == 'Y':
            totim = []
            for to in self.totim:
                t = timedelta(days=to * 365.25)
                totim.append(self.start + t)
        return totim

    def _get_sp(self, ts, sp, seekpoint):
        self.f.seek(seekpoint)
        # --read header lines
        ihead = 0
        while True:
            line = self.f.readline()
            ihead += 1
            if line == '':
                # raise Exception('end of file found while seeking budget information')
                print('end of file found while seeking time information for ts,sp', ts, sp)
                return self.null_entries
            elif ihead == 2 and 'SECONDS     MINUTES      HOURS       DAYS        YEARS' not in line:
                break
            elif '-----------------------------------------------------------' in line:
                line = self.f.readline()
                break
        tslen = self._parse_time_line(line)
        if tslen == None:
            print('error parsing tslen for ts,sp', ts, sp)
            return self.null_entries

        sptim = self._parse_time_line(self.f.readline())
        if sptim == None:
            print('error parsing sptim for ts,sp', ts, sp)
            return self.null_entries

        totim = self._parse_time_line(self.f.readline())
        if totim == None:
            print('error parsing totim for ts,sp', ts, sp)
            return self.null_entries
        return tslen, sptim, totim

    def _parse_time_line(self, line):
        if line == '':
            print('end of file found while parsing time information')
            return None
        try:
            time_str = line[self.time_line_idx:]
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
            print('error parsing tslen information', time_str)
            return None
        return tval
