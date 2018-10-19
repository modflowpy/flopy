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

    def __init__(self, cbc_file, z, kstpkper=None, totim=None, aliases=None,
                 verbose=False, **kwargs):

        if isinstance(cbc_file, CellBudgetFile):
            self.cbc = cbc_file
        elif isinstance(cbc_file, str) and os.path.isfile(cbc_file):
            self.cbc = CellBudgetFile(cbc_file)
        else:
            raise Exception(
                'Cannot load cell budget file: {}.'.format(cbc_file))

        if isinstance(z, np.ndarray):
            assert np.issubdtype(z.dtype,
                                 np.integer), 'Zones dtype must be integer'
        else:
            raise Exception(
                'Please pass zones as a numpy ndarray of (positive) integers. {}'.format(
                    z.dtype))

        # Check for negative zone values
        for zi in np.unique(z):
            if zi < 0:
                raise Exception('Negative zone value(s) found:', zi)

        self.dis = None
        self.sr = None
        if 'model' in kwargs.keys():
            self.model = kwargs.pop('model')
            self.sr = self.model.sr
            self.dis = self.model.dis
        if 'dis' in kwargs.keys():
            self.dis = kwargs.pop('dis')
            self.sr = self.dis.parent.sr
        if 'sr' in kwargs.keys():
            self.sr = kwargs.pop('sr')
        if len(kwargs.keys()) > 0:
            args = ','.join(kwargs.keys())
            raise Exception('LayerFile error: unrecognized kwargs: ' + args)

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
                s = 'The specified time step/stress period ' \
                    'does not exist {}'.format(kk)
                assert kk in self.cbc.get_kstpkper(), s
            self.kstpkper = kstpkper
        elif totim is not None:
            if isinstance(totim, float):
                totim = [totim]
            elif isinstance(totim, int):
                totim = [float(totim)]
            for t in totim:
                s = 'The specified simulation time ' \
                    'does not exist {}'.format(t)
                assert t in self.cbc.get_times(), s
            self.totim = totim
        else:
            # No time step/stress period or simulation time pass
            self.kstpkper = self.cbc.get_kstpkper()

        # Set float and integer types
        self.float_type = np.float32
        self.int_type = np.int32

        # Check dimensions of input zone array
        s = 'Row/col dimensions of zone array {}' \
            ' do not match model row/col dimensions {}'.format(z.shape,
                                                               self.cbc_shape)
        assert z.shape[-2] == self.nrow and \
               z.shape[-1] == self.ncol, s

        if z.shape == self.cbc_shape:
            izone = z.copy()
        elif len(z.shape) == 2:
            izone = np.zeros(self.cbc_shape, self.int_type)
            izone[:] = z[:, :]
        elif len(z.shape) == 3 and z.shape[0] == 1:
            izone = np.zeros(self.cbc_shape, self.int_type)
            izone[:] = z[0, :, :]
        else:
            raise Exception(
                'Shape of the zone array is not recognized: {}'.format(
                    z.shape))

        self.izone = izone
        self.allzones = [z for z in np.unique(self.izone)]
        self._zonenamedict = OrderedDict([(z, 'ZONE_{}'.format(z))
                                          for z in self.allzones if
                                          z != 0])

        if aliases is not None:
            assert isinstance(aliases,
                              dict), 'Input aliases not recognized. Please pass a dictionary ' \
                                     'with key,value pairs of zone/alias.'
            # Replace the relevant field names (ignore zone 0)
            seen = []
            for z, a in iter(aliases.items()):
                if z != 0 and z in self._zonenamedict.keys():
                    if z in seen:
                        raise Exception(
                            'Zones may not have more than 1 alias.')
                    self._zonenamedict[z] = '_'.join(a.split())
                    seen.append(z)

        self._iflow_recnames = self._get_internal_flow_record_names()

        # All record names in the cell-by-cell budget binary file
        self.record_names = [n.strip().decode("utf-8") for n in
                             self.cbc.get_unique_record_names()]

        # Get imeth for each record in the CellBudgetFile record list
        self.imeth = {}
        for record in self.cbc.recordarray:
            self.imeth[record['text'].strip().decode("utf-8")] = record[
                'imeth']

        # INTERNAL FLOW TERMS ARE USED TO CALCULATE FLOW BETWEEN ZONES.
        # CONSTANT-HEAD TERMS ARE USED TO IDENTIFY WHERE CONSTANT-HEAD CELLS ARE AND THEN USE
        # FACE FLOWS TO DETERMINE THE AMOUNT OF FLOW.
        # SWIADDTO--- terms are used by the SWI2 groundwater flow process.
        internal_flow_terms = ['CONSTANT HEAD', 'FLOW RIGHT FACE',
                               'FLOW FRONT FACE', 'FLOW LOWER FACE',
                               'SWIADDTOCH', 'SWIADDTOFRF', 'SWIADDTOFFF',
                               'SWIADDTOFLF']

        # Source/sink/storage term record names
        # These are all of the terms that are not related to constant
        # head cells or face flow terms
        self.ssst_record_names = [n for n in self.record_names
                                  if n not in internal_flow_terms]

        # Initialize budget recordarray
        array_list = []
        if self.kstpkper is not None:
            for kk in self.kstpkper:
                recordarray = self._initialize_budget_recordarray(kstpkper=kk,
                                                                  totim=None)
                array_list.append(recordarray)
        elif self.totim is not None:
            for t in self.totim:
                recordarray = self._initialize_budget_recordarray(
                    kstpkper=None, totim=t)
                array_list.append(recordarray)
        self._budget = np.concatenate(array_list, axis=0)

        # Update budget record array
        if self.kstpkper is not None:
            for kk in self.kstpkper:
                if verbose:
                    s = 'Computing the budget for' \
                        ' time step {} in stress period {}'.format(kk[0] + 1,
                                                                   kk[1] + 1)
                    print(s)
                self._compute_budget(kstpkper=kk)
        elif self.totim is not None:
            for t in self.totim:
                if verbose:
                    s = 'Computing the budget for time {}'.format(t)
                    print(s)
                self._compute_budget(totim=t)

        return

    def get_model_shape(self):
        """

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
            return np.unique(self._budget['name'])
        else:
            seen = []
            for recname in self.get_record_names():
                if recname in ['IN-OUT', 'TOTAL_IN', 'TOTAL_OUT']:
                    continue
                if recname.endswith('_IN'):
                    recname = recname[:-3]
                elif recname.endswith('_OUT'):
                    recname = recname[:-4]
                if recname not in seen:
                    seen.append(recname)
            seen.extend(['IN-OUT', 'TOTAL'])
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
        budget_list : list of reecord arrays
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
        select_fields = ['totim', 'time_step', 'stress_period',
                         'name'] + list(self._zonenamedict.values())
        select_records = np.where(
            (self._budget['name'] == self._budget['name']))
        if zones is not None:
            for idx, z in enumerate(zones):
                if isinstance(z, int):
                    zones[idx] = self._zonenamedict[z]
            select_fields = ['totim', 'time_step', 'stress_period',
                             'name'] + zones
        if names is not None:
            names = self._clean_budget_names(names)
            select_records = np.in1d(self._budget['name'], names)
        if net:
            if names is None:
                names = self._clean_budget_names(self.get_record_names())
            net_budget = self._compute_net_budget()
            seen = []
            net_names = []
            for name in names:
                iname = '_'.join(name.split('_')[1:])
                if iname not in seen:
                    seen.append(iname)
                else:
                    net_names.append(iname)
            select_records = np.in1d(net_budget['name'], net_names)
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
        # Needs updating to handle the new budget list structure. Write out budgets for all kstpkper
        # if kstpkper is None or pass list of kstpkper/totim to save particular budgets.
        with open(fname, 'w') as f:
            # Write header
            f.write(','.join(self._budget.dtype.names) + '\n')
            # Write rows
            for rowidx in range(self._budget.shape[0]):
                s = ','.join(
                    [str(i) for i in list(self._budget[:][rowidx])]) + '\n'
                f.write(s)
        return

    def get_dataframes(self, start_datetime=None, timeunit='D',
                       index_key='totim', names=None, zones=None, net=False):
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

        valid_index_keys = ['totim', 'kstpkper']
        assert index_key in valid_index_keys, 'index_key "{}" is not valid.'.format(
            index_key)

        valid_timeunit = ['S', 'M', 'H', 'D', 'Y']

        if timeunit.upper() == 'SECONDS':
            timeunit = 'S'
        elif timeunit.upper() == 'MINUTES':
            timeunit = 'M'
        elif timeunit.upper() == 'HOURS':
            timeunit = 'H'
        elif timeunit.upper() == 'DAYS':
            timeunit = 'D'
        elif timeunit.upper() == 'YEARS':
            timeunit = 'Y'

        errmsg = 'Specified time units ({}) not recognized. ' \
                 'Please use one of '.format(timeunit)
        assert timeunit in valid_timeunit, errmsg + ', '.join(
            valid_timeunit) + '.'

        df = pd.DataFrame().from_records(self.get_budget(names, zones, net))
        if start_datetime is not None:
            totim = totim_to_datetime(df.totim,
                                      start=pd.to_datetime(start_datetime),
                                      timeunit=timeunit)
            df['datetime'] = totim
            index_cols = ['datetime', 'name']
        else:
            if index_key == 'totim':
                index_cols = ['totim', 'name']
            elif index_key == 'kstpkper':
                index_cols = ['time_step', 'stress_period', 'name']
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
        ignore_attrs = ['cbc']
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
        Creates a budget for the specified zone array. This function only supports the
        use of a single time step/stress period or time.

        Parameters
        ----------
        kstpkper : tuple
            Tuple of kstp and kper to compute budget for (default is None).
        totim : float
            Totim to compute budget for (default is None).

        Returns
        -------
        Noen

        """
        # Initialize an array to track where the constant head cells
        # are located.
        ich = np.zeros(self.cbc_shape, self.int_type)
        swiich = np.zeros(self.cbc_shape, self.int_type)

        if 'CONSTANT HEAD' in self.record_names:
            """
            C-----CONSTANT-HEAD FLOW -- DON'T ACCUMULATE THE CELL-BY-CELL VALUES FOR
            C-----CONSTANT-HEAD FLOW BECAUSE THEY MAY INCLUDE PARTIALLY CANCELING
            C-----INS AND OUTS.  USE CONSTANT-HEAD TERM TO IDENTIFY WHERE CONSTANT-
            C-----HEAD CELLS ARE AND THEN USE FACE FLOWS TO DETERMINE THE AMOUNT OF
            C-----FLOW.  STORE CONSTANT-HEAD LOCATIONS IN ICH ARRAY.
            """
            chd = self.cbc.get_data(text='CONSTANT HEAD', full3D=True,
                                    kstpkper=kstpkper, totim=totim)[0]
            ich[np.ma.where(chd != 0.)] = 1
        if 'FLOW RIGHT FACE' in self.record_names:
            self._accumulate_flow_frf('FLOW RIGHT FACE', ich, kstpkper, totim)
        if 'FLOW FRONT FACE' in self.record_names:
            self._accumulate_flow_fff('FLOW FRONT FACE', ich, kstpkper, totim)
        if 'FLOW LOWER FACE' in self.record_names:
            self._accumulate_flow_flf('FLOW LOWER FACE', ich, kstpkper, totim)
        if 'SWIADDTOCH' in self.record_names:
            swichd = self.cbc.get_data(text='SWIADDTOCH', full3D=True,
                                       kstpkper=kstpkper, totim=totim)[0]
            swiich[swichd != 0] = 1
        if 'SWIADDTOFRF' in self.record_names:
            self._accumulate_flow_frf('SWIADDTOFRF', swiich, kstpkper, totim)
        if 'SWIADDTOFFF' in self.record_names:
            self._accumulate_flow_fff('SWIADDTOFFF', swiich, kstpkper, totim)
        if 'SWIADDTOFLF' in self.record_names:
            self._accumulate_flow_flf('SWIADDTOFLF', swiich, kstpkper, totim)

        # NOT AN INTERNAL FLOW TERM, SO MUST BE A SOURCE TERM OR STORAGE
        # ACCUMULATE THE FLOW BY ZONE
        # iterate over remaining items in the list
        for recname in self.ssst_record_names:
            self._accumulate_flow_ssst(recname, kstpkper, totim)

        # Compute mass balance terms
        self._compute_mass_balance(kstpkper, totim)

        return

    def _get_internal_flow_record_names(self):
        """
        Get internal flow record names

        Returns
        -------
        iflow_recnames : np.recarray
            recarray of internal flow terms

        """
        iflow_recnames = OrderedDict([(0, 'ZONE_0')])
        for z, a in iter(self._zonenamedict.items()):
            iflow_recnames[z] = '{}'.format(a)
        dtype = np.dtype([('zone', '<i4'), ('name', (str, 50))])
        iflow_recnames = np.array(list(iflow_recnames.items()), dtype=dtype)
        return iflow_recnames

    def _add_empty_record(self, recordarray, recname, kstpkper=None,
                          totim=None):
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
                totim = 0.
        elif totim is not None:
            if len(self.cbc_times) > 0:
                kstpkper = self.cbc_kstpkper[self.cbc_times.index(totim)]
            else:
                kstpkper = (0, 0)

        row = [totim, kstpkper[0], kstpkper[1], recname]
        row += [0. for _ in self._zonenamedict.values()]
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
        dtype_list = [('totim', '<f4'), ('time_step', '<i4'),
                      ('stress_period', '<i4'), ('name', (str, 50))]
        dtype_list += [(n, self.float_type) for n in
                       self._zonenamedict.values()]
        dtype = np.dtype(dtype_list)
        recordarray = np.array([], dtype=dtype)

        # Add "from" records
        if 'STORAGE' in self.record_names:
            recordarray = self._add_empty_record(recordarray, 'FROM_STORAGE',
                                                 kstpkper, totim)
        if 'CONSTANT HEAD' in self.record_names:
            recordarray = self._add_empty_record(recordarray,
                                                 'FROM_CONSTANT_HEAD',
                                                 kstpkper, totim)
        for recname in self.ssst_record_names:
            if recname != 'STORAGE':
                recordarray = self._add_empty_record(recordarray,
                                                     'FROM_' + '_'.join(
                                                         recname.split()),
                                                     kstpkper, totim)

        for z, n in self._iflow_recnames:
            if n == 0 and 0 not in self.allzones:
                continue
            else:
                recordarray = self._add_empty_record(recordarray,
                                                     'FROM_' + '_'.join(
                                                         n.split()),
                                                     kstpkper, totim)
        recordarray = self._add_empty_record(recordarray, 'TOTAL_IN',
                                             kstpkper, totim)

        # Add "out" records
        if 'STORAGE' in self.record_names:
            recordarray = self._add_empty_record(recordarray, 'TO_STORAGE',
                                                 kstpkper, totim)
        if 'CONSTANT HEAD' in self.record_names:
            recordarray = self._add_empty_record(recordarray,
                                                 'TO_CONSTANT_HEAD',
                                                 kstpkper, totim)
        for recname in self.ssst_record_names:
            if recname != 'STORAGE':
                recordarray = self._add_empty_record(recordarray,
                                                     'TO_' + '_'.join(
                                                         recname.split()),
                                                     kstpkper, totim)

        for n in self._iflow_recnames['name']:
            if n == 0 and 0 not in self.allzones:
                continue
            else:
                recordarray = self._add_empty_record(recordarray,
                                                     'TO_' + '_'.join(
                                                         n.split()), kstpkper,
                                                     totim)
        recordarray = self._add_empty_record(recordarray, 'TOTAL_OUT',
                                             kstpkper, totim)

        recordarray = self._add_empty_record(recordarray, 'IN-OUT', kstpkper,
                                             totim)
        recordarray = self._add_empty_record(recordarray,
                                             'PERCENT_DISCREPANCY', kstpkper,
                                             totim)
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

    def _update_budget_fromfaceflow(self, fz, tz, f,
                                    kstpkper=None, totim=None):
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
        idx = np.logical_not(np.array([item in tz for item in [0] * len(tz)]))
        fzi = fz[idx]
        tzi = tz[idx]
        rownames = np.array(list(['FROM_' +
                                  self._iflow_recnames[
                                      self._iflow_recnames['zone'] == z][
                                      'name'][0]
                                  for z in fzi]))
        colnames = np.array(list(
            [self._iflow_recnames[self._iflow_recnames['zone'] == z]['name'][0]
             for z in tzi]))
        fluxes = f[idx]
        self._update_budget_recordarray(rownames, colnames, fluxes, kstpkper,
                                        totim)

        # Outflows
        idx = np.logical_not(np.array([item in fz for item in [0] * len(fz)]))
        fzi = fz[idx]
        tzi = tz[idx]
        rownames = np.array(list(['TO_' +
                                  self._iflow_recnames[
                                      self._iflow_recnames['zone'] == z][
                                      'name'][0]
                                  for z in tzi]))
        colnames = np.array(list(
            [self._iflow_recnames[self._iflow_recnames['zone'] == z]['name'][0]
             for z in fzi]))
        fluxes = f[idx]
        self._update_budget_recordarray(rownames, colnames, fluxes, kstpkper,
                                        totim)
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

    def _update_budget_recordarray(self, rownames, colnames, fluxes,
                                   kstpkper=None, totim=None):
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
                for rn, cn, flux in list(zip(rownames, colnames, fluxes)):
                    rowidx = np.where(
                        (self._budget['time_step'] == kstpkper[0]) &
                        (self._budget['stress_period'] == kstpkper[1]) &
                        (self._budget['name'] == rn))
                    self._budget[cn][rowidx] += flux
            elif totim is not None:
                for rn, cn, flux in list(zip(rownames, colnames, fluxes)):
                    rowidx = np.where((self._budget['totim'] == totim) &
                                      (self._budget['name'] == rn))
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
                data = \
                    self.cbc.get_data(text=recname, kstpkper=kstpkper,
                                      totim=totim)[0]

                # "FLOW RIGHT FACE"  COMPUTE FLOW BETWEEN ZONES ACROSS COLUMNS.
                # COMPUTE FLOW ONLY BETWEEN A ZONE AND A HIGHER ZONE -- FLOW FROM
                # ZONE 4 TO 3 IS THE NEGATIVE OF FLOW FROM 3 TO 4.
                # 1ST, CALCULATE FLOW BETWEEN NODE J,I,K AND J-1,I,K

                k, i, j = np.where(
                    self.izone[:, :, 1:] > self.izone[:, :, :-1])

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
                    (q > 0) & ((ich[k, i, j] != 1) | (ich[k, i, jl] != 1)))
                fzi, tzi, fi = sum_flux_tuples(nzl[idx],
                                               nz[idx],
                                               q[idx])
                self._update_budget_fromfaceflow(fzi, tzi, np.abs(fi),
                                                 kstpkper, totim)

                # Get indices where flow face values are negative (flow into higher zone)
                # Don't include CH to CH flow (can occur if CHTOCH option is used)
                # Create an interable tuple of (from zone, to zone, flux)
                # Then group tuple by (from_zone, to_zone) and sum the flux values
                idx = np.where(
                    (q < 0) & ((ich[k, i, j] != 1) | (ich[k, i, jl] != 1)))
                fzi, tzi, fi = sum_flux_tuples(nz[idx],
                                               nzl[idx],
                                               q[idx])
                self._update_budget_fromfaceflow(fzi, tzi, np.abs(fi),
                                                 kstpkper, totim)

                # FLOW BETWEEN NODE J,I,K AND J+1,I,K
                k, i, j = np.where(
                    self.izone[:, :, :-1] > self.izone[:, :, 1:])

                # Define the zone from which flow is coming
                nz = self.izone[k, i, j]

                # Define the zone to which flow is going
                jr = j + 1
                nzr = self.izone[k, i, jr]

                # Get the face flow
                q = data[k, i, j]

                # Get indices where flow face values are positive (flow out of higher zone)
                # Don't include CH to CH flow (can occur if CHTOCH option is used)
                # Create an interable tuple of (from zone, to zone, flux)
                # Then group tuple by (from_zone, to_zone) and sum the flux values
                idx = np.where(
                    (q > 0) & ((ich[k, i, j] != 1) | (ich[k, i, jr] != 1)))
                fzi, tzi, fi = sum_flux_tuples(nz[idx],
                                               nzr[idx],
                                               q[idx])
                self._update_budget_fromfaceflow(fzi, tzi, np.abs(fi),
                                                 kstpkper, totim)

                # Get indices where flow face values are negative (flow into higher zone)
                # Don't include CH to CH flow (can occur if CHTOCH option is used)
                # Create an iterable tuple of (from zone, to zone, flux)
                # Then group tuple by (from_zone, to_zone) and sum the flux values
                idx = np.where(
                    (q < 0) & ((ich[k, i, j] != 1) | (ich[k, i, jr] != 1)))
                fzi, tzi, fi = sum_flux_tuples(nzr[idx],
                                               nz[idx],
                                               q[idx])
                self._update_budget_fromfaceflow(fzi, tzi, np.abs(fi),
                                                 kstpkper, totim)

                # CALCULATE FLOW TO CONSTANT-HEAD CELLS IN THIS DIRECTION
                k, i, j = np.where(ich == 1)
                k, i, j = k[j > 0], i[j > 0], j[j > 0]
                jl = j - 1
                nzl = self.izone[k, i, jl]
                nz = self.izone[k, i, j]
                q = data[k, i, jl]
                idx = np.where(
                    (q > 0) & ((ich[k, i, j] != 1) | (ich[k, i, jl] != 1)))
                fzi, tzi, fi = sum_flux_tuples(nzl[idx],
                                               nz[idx],
                                               q[idx])
                tz = tzi[tzi != 0]
                f = fi[tzi != 0]
                fz = np.array(['TO_CONSTANT_HEAD'] * len(tz))
                tz = np.array([self._zonenamedict[z] for z in tzi[tzi != 0]])
                self._update_budget_fromssst(fz, tz, np.abs(f), kstpkper,
                                             totim)

                idx = np.where(
                    (q < 0) & ((ich[k, i, j] != 1) | (ich[k, i, jl] != 1)))
                fzi, tzi, fi = sum_flux_tuples(nzl[idx],
                                               nz[idx],
                                               q[idx])
                fz = fzi[fzi != 0]
                f = fi[fzi != 0]
                fz = np.array(['FROM_CONSTANT_HEAD'] * len(fz))
                tz = np.array([self._zonenamedict[z] for z in tzi[tzi != 0]])
                self._update_budget_fromssst(fz, tz, np.abs(f), kstpkper,
                                             totim)

                k, i, j = np.where(ich == 1)
                k, i, j = k[j < self.ncol - 1], i[j < self.ncol - 1], j[
                    j < self.ncol - 1]
                nz = self.izone[k, i, j]
                jr = j + 1
                nzr = self.izone[k, i, jr]
                q = data[k, i, j]
                idx = np.where(
                    (q > 0) & ((ich[k, i, j] != 1) | (ich[k, i, jr] != 1)))
                fzi, tzi, fi = sum_flux_tuples(nzr[idx],
                                               nz[idx],
                                               q[idx])
                tz = tzi[tzi != 0]
                f = fi[tzi != 0]
                fz = np.array(['FROM_CONSTANT_HEAD'] * len(tz))
                tz = np.array([self._zonenamedict[z] for z in tzi[tzi != 0]])
                self._update_budget_fromssst(fz, tz, np.abs(f), kstpkper,
                                             totim)

                idx = np.where(
                    (q < 0) & ((ich[k, i, j] != 1) | (ich[k, i, jr] != 1)))
                fzi, tzi, fi = sum_flux_tuples(nzr[idx],
                                               nz[idx],
                                               q[idx])
                fz = fzi[fzi != 0]
                f = fi[fzi != 0]
                fz = np.array(['TO_CONSTANT_HEAD'] * len(fz))
                tz = np.array([self._zonenamedict[z] for z in tzi[tzi != 0]])
                self._update_budget_fromssst(fz, tz, np.abs(f), kstpkper,
                                             totim)

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
                data = \
                    self.cbc.get_data(text=recname, kstpkper=kstpkper,
                                      totim=totim)[0]

                # "FLOW FRONT FACE"
                # CALCULATE FLOW BETWEEN NODE J,I,K AND J,I-1,K
                k, i, j = np.where(
                    self.izone[:, 1:, :] < self.izone[:, :-1, :])
                i += 1
                ia = i - 1
                nza = self.izone[k, ia, j]
                nz = self.izone[k, i, j]
                q = data[k, ia, j]
                idx = np.where(
                    (q > 0) & ((ich[k, i, j] != 1) | (ich[k, ia, j] != 1)))
                fzi, tzi, fi = sum_flux_tuples(nza[idx],
                                               nz[idx],
                                               q[idx])
                self._update_budget_fromfaceflow(fzi, tzi, np.abs(fi),
                                                 kstpkper, totim)

                idx = np.where(
                    (q < 0) & ((ich[k, i, j] != 1) | (ich[k, ia, j] != 1)))
                fzi, tzi, fi = sum_flux_tuples(nz[idx],
                                               nza[idx],
                                               q[idx])
                self._update_budget_fromfaceflow(fzi, tzi, np.abs(fi),
                                                 kstpkper, totim)

                # CALCULATE FLOW BETWEEN NODE J,I,K AND J,I+1,K.
                k, i, j = np.where(
                    self.izone[:, :-1, :] < self.izone[:, 1:, :])
                nz = self.izone[k, i, j]
                ib = i + 1
                nzb = self.izone[k, ib, j]
                q = data[k, i, j]
                idx = np.where(
                    (q > 0) & ((ich[k, i, j] != 1) | (ich[k, ib, j] != 1)))
                fzi, tzi, fi = sum_flux_tuples(nz[idx],
                                               nzb[idx],
                                               q[idx])
                self._update_budget_fromfaceflow(fzi, tzi, np.abs(fi),
                                                 kstpkper, totim)

                idx = np.where(
                    (q < 0) & ((ich[k, i, j] != 1) | (ich[k, ib, j] != 1)))
                fzi, tzi, fi = sum_flux_tuples(nzb[idx],
                                               nz[idx],
                                               q[idx])
                self._update_budget_fromfaceflow(fzi, tzi, np.abs(fi),
                                                 kstpkper, totim)

                # CALCULATE FLOW TO CONSTANT-HEAD CELLS IN THIS DIRECTION
                k, i, j = np.where(ich == 1)
                k, i, j = k[i > 0], i[i > 0], j[i > 0]
                ia = i - 1
                nza = self.izone[k, ia, j]
                nz = self.izone[k, i, j]
                q = data[k, ia, j]
                idx = np.where(
                    (q > 0) & ((ich[k, i, j] != 1) | (ich[k, ia, j] != 1)))
                fzi, tzi, fi = sum_flux_tuples(nza[idx],
                                               nz[idx],
                                               q[idx])
                tz = tzi[tzi != 0]
                f = fi[tzi != 0]
                fz = np.array(['TO_CONSTANT_HEAD'] * len(tz))
                tz = np.array([self._zonenamedict[z] for z in tzi[tzi != 0]])
                self._update_budget_fromssst(fz, tz, np.abs(f), kstpkper,
                                             totim)

                idx = np.where(
                    (q < 0) & ((ich[k, i, j] != 1) | (ich[k, ia, j] != 1)))
                fzi, tzi, fi = sum_flux_tuples(nza[idx],
                                               nz[idx],
                                               q[idx])
                fz = fzi[fzi != 0]
                f = fi[fzi != 0]
                fz = np.array(['FROM_CONSTANT_HEAD'] * len(fz))
                tz = np.array([self._zonenamedict[z] for z in tzi[tzi != 0]])
                self._update_budget_fromssst(fz, tz, np.abs(f), kstpkper,
                                             totim)

                k, i, j = np.where(ich == 1)
                k, i, j = k[i < self.nrow - 1], i[i < self.nrow - 1], j[
                    i < self.nrow - 1]
                nz = self.izone[k, i, j]
                ib = i + 1
                nzb = self.izone[k, ib, j]
                q = data[k, i, j]
                idx = np.where(
                    (q > 0) & ((ich[k, i, j] != 1) | (ich[k, ib, j] != 1)))
                fzi, tzi, fi = sum_flux_tuples(nzb[idx],
                                               nz[idx],
                                               q[idx])
                tz = tzi[tzi != 0]
                f = fi[tzi != 0]
                fz = np.array(['FROM_CONSTANT_HEAD'] * len(tz))
                tz = np.array([self._zonenamedict[z] for z in tzi[tzi != 0]])
                self._update_budget_fromssst(fz, tz, np.abs(f), kstpkper,
                                             totim)

                idx = np.where(
                    (q < 0) & ((ich[k, i, j] != 1) | (ich[k, ib, j] != 1)))
                fzi, tzi, fi = sum_flux_tuples(nzb[idx],
                                               nz[idx],
                                               q[idx])
                fz = fzi[fzi != 0]
                f = fi[fzi != 0]
                fz = np.array(['TO_CONSTANT_HEAD'] * len(fz))
                tz = np.array([self._zonenamedict[z] for z in tzi[tzi != 0]])
                self._update_budget_fromssst(fz, tz, np.abs(f), kstpkper,
                                             totim)

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
                data = \
                    self.cbc.get_data(text=recname, kstpkper=kstpkper,
                                      totim=totim)[0]

                # "FLOW LOWER FACE"
                # CALCULATE FLOW BETWEEN NODE J,I,K AND J,I,K-1
                k, i, j = np.where(
                    self.izone[1:, :, :] < self.izone[:-1, :, :])
                k += 1
                ka = k - 1
                nza = self.izone[ka, i, j]
                nz = self.izone[k, i, j]
                q = data[ka, i, j]
                idx = np.where(
                    (q > 0) & ((ich[k, i, j] != 1) | (ich[ka, i, j] != 1)))
                fzi, tzi, fi = sum_flux_tuples(nza[idx],
                                               nz[idx],
                                               q[idx])
                self._update_budget_fromfaceflow(fzi, tzi, np.abs(fi),
                                                 kstpkper, totim)

                idx = np.where(
                    (q < 0) & ((ich[k, i, j] != 1) | (ich[ka, i, j] != 1)))
                fzi, tzi, fi = sum_flux_tuples(nz[idx],
                                               nza[idx],
                                               q[idx])
                self._update_budget_fromfaceflow(fzi, tzi, np.abs(fi),
                                                 kstpkper, totim)

                # CALCULATE FLOW BETWEEN NODE J,I,K AND J,I,K+1
                k, i, j = np.where(
                    self.izone[:-1, :, :] < self.izone[1:, :, :])
                nz = self.izone[k, i, j]
                kb = k + 1
                nzb = self.izone[kb, i, j]
                q = data[k, i, j]
                idx = np.where(
                    (q > 0) & ((ich[k, i, j] != 1) | (ich[kb, i, j] != 1)))
                fzi, tzi, fi = sum_flux_tuples(nz[idx],
                                               nzb[idx],
                                               q[idx])
                self._update_budget_fromfaceflow(fzi, tzi, np.abs(fi),
                                                 kstpkper, totim)

                idx = np.where(
                    (q < 0) & ((ich[k, i, j] != 1) | (ich[kb, i, j] != 1)))
                fzi, tzi, fi = sum_flux_tuples(nzb[idx],
                                               nz[idx],
                                               q[idx])
                self._update_budget_fromfaceflow(fzi, tzi, np.abs(fi),
                                                 kstpkper, totim)

                # CALCULATE FLOW TO CONSTANT-HEAD CELLS IN THIS DIRECTION
                k, i, j = np.where(ich == 1)
                k, i, j = k[k > 0], i[k > 0], j[k > 0]
                ka = k - 1
                nza = self.izone[ka, i, j]
                nz = self.izone[k, i, j]
                q = data[ka, i, j]
                idx = np.where(
                    (q > 0) & ((ich[k, i, j] != 1) | (ich[ka, i, j] != 1)))
                fzi, tzi, fi = sum_flux_tuples(nza[idx],
                                               nz[idx],
                                               q[idx])
                tz = tzi[tzi != 0]
                f = fi[tzi != 0]
                fz = np.array(['TO_CONSTANT_HEAD'] * len(tz))
                tz = np.array([self._zonenamedict[z] for z in tzi[tzi != 0]])
                self._update_budget_fromssst(fz, tz, np.abs(f), kstpkper,
                                             totim)

                idx = np.where(
                    (q < 0) & ((ich[k, i, j] != 1) | (ich[ka, i, j] != 1)))
                fzi, tzi, fi = sum_flux_tuples(nza[idx],
                                               nz[idx],
                                               q[idx])
                fz = fzi[fzi != 0]
                f = fi[fzi != 0]
                fz = np.array(['FROM_CONSTANT_HEAD'] * len(fz))
                tz = np.array([self._zonenamedict[z] for z in tzi[tzi != 0]])
                self._update_budget_fromssst(fz, tz, np.abs(f), kstpkper,
                                             totim)

                k, i, j = np.where(ich == 1)
                k, i, j = k[k < self.nlay - 1], i[k < self.nlay - 1], j[
                    k < self.nlay - 1]
                nz = self.izone[k, i, j]
                kb = k + 1
                nzb = self.izone[kb, i, j]
                q = data[k, i, j]
                idx = np.where(
                    (q > 0) & ((ich[k, i, j] != 1) | (ich[kb, i, j] != 1)))
                fzi, tzi, fi = sum_flux_tuples(nzb[idx],
                                               nz[idx],
                                               q[idx])
                tz = tzi[tzi != 0]
                f = fi[tzi != 0]
                fz = np.array(['FROM_CONSTANT_HEAD'] * len(tz))
                tz = np.array([self._zonenamedict[z] for z in tzi[tzi != 0]])
                self._update_budget_fromssst(fz, tz, np.abs(f), kstpkper,
                                             totim)

                idx = np.where(
                    (q < 0) & ((ich[k, i, j] != 1) | (ich[kb, i, j] != 1)))
                fzi, tzi, fi = sum_flux_tuples(nzb[idx],
                                               nz[idx],
                                               q[idx])
                fz = fzi[fzi != 0]
                f = fi[fzi != 0]
                fz = np.array(['TO_CONSTANT_HEAD'] * len(fz))
                tz = np.array([self._zonenamedict[z] for z in tzi[tzi != 0]])
                self._update_budget_fromssst(fz, tz, np.abs(f), kstpkper,
                                             totim)

        except Exception as e:
            print(e)
            raise
        return

    def _accumulate_flow_ssst(self, recname, kstpkper, totim):

        # NOT AN INTERNAL FLOW TERM, SO MUST BE A SOURCE TERM OR STORAGE
        # ACCUMULATE THE FLOW BY ZONE

        imeth = self.imeth[recname]

        data = self.cbc.get_data(text=recname, kstpkper=kstpkper,
                                 totim=totim)
        if len(data) == 0:
            # Empty data, can occur during the first time step of a transient model when
            # storage terms are zero and not in the cell-budget file.
            return
        else:
            data = data[0]

        if imeth == 2 or imeth == 5:
            # LIST
            qin = np.ma.zeros((self.nlay * self.nrow * self.ncol),
                              self.float_type)
            qout = np.ma.zeros((self.nlay * self.nrow * self.ncol),
                               self.float_type)
            for [node, q] in zip(data['node'], data['q']):
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
                'Unrecognized "imeth" for {} record: {}'.format(recname,
                                                                imeth))

        # Inflows
        fz = []
        tz = []
        f = []
        for z in self.allzones:
            if z != 0:
                flux = qin[(self.izone == z)].sum()
                if type(flux) == np.ma.core.MaskedConstant:
                    flux = 0.
                fz.append('FROM_' + '_'.join(recname.split()))
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
                    flux = 0.
                fz.append('TO_' + '_'.join(recname.split()))
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
        skipcols = ['time_step', 'stress_period', 'totim', 'name']

        # Compute inflows
        recnames = self.get_record_names()
        innames = [n for n in recnames if n.startswith('FROM_')]
        outnames = [n for n in recnames if n.startswith('TO_')]
        if kstpkper is not None:
            rowidx = np.where((self._budget['time_step'] == kstpkper[0]) &
                              (self._budget['stress_period'] == kstpkper[1]) &
                              np.in1d(self._budget['name'], innames))
        elif totim is not None:
            rowidx = np.where((self._budget['totim'] == totim) &
                              np.in1d(self._budget['name'], innames))
        a = _numpyvoid2numeric(
            self._budget[list(self._zonenamedict.values())][rowidx])
        intot = np.array(a.sum(axis=0))
        tz = np.array(
            list([n for n in self._budget.dtype.names if n not in skipcols]))
        fz = np.array(['TOTAL_IN'] * len(tz))
        self._update_budget_fromssst(fz, tz, intot, kstpkper, totim)

        # Compute outflows
        if kstpkper is not None:
            rowidx = np.where((self._budget['time_step'] == kstpkper[0]) &
                              (self._budget['stress_period'] == kstpkper[1]) &
                              np.in1d(self._budget['name'], outnames))
        elif totim is not None:
            rowidx = np.where((self._budget['totim'] == totim) &
                              np.in1d(self._budget['name'], outnames))
        a = _numpyvoid2numeric(
            self._budget[list(self._zonenamedict.values())][rowidx])
        outot = np.array(a.sum(axis=0))
        tz = np.array(
            list([n for n in self._budget.dtype.names if n not in skipcols]))
        fz = np.array(['TOTAL_OUT'] * len(tz))
        self._update_budget_fromssst(fz, tz, outot, kstpkper, totim)

        # Compute IN-OUT
        tz = np.array(
            list([n for n in self._budget.dtype.names if n not in skipcols]))
        f = intot - outot
        fz = np.array(['IN-OUT'] * len(tz))
        self._update_budget_fromssst(fz, tz, np.abs(f), kstpkper, totim)

        # Compute percent discrepancy
        tz = np.array(
            list([n for n in self._budget.dtype.names if n not in skipcols]))
        fz = np.array(['PERCENT_DISCREPANCY'] * len(tz))
        in_minus_out = intot - outot
        in_plus_out = intot + outot
        f = 100 * in_minus_out / (in_plus_out / 2.)
        self._update_budget_fromssst(fz, tz, np.abs(f), kstpkper, totim)

        return

    def _clean_budget_names(self, names):
        newnames = []
        mbnames = ['TOTAL_IN', 'TOTAL_OUT',
                   'IN-OUT', 'PERCENT_DISCREPANCY']
        for name in names:
            if name in mbnames:
                newnames.append(name)
            elif not name.startswith('FROM_') and not name.startswith('TO_'):
                newname_in = 'FROM_' + name.upper()
                newname_out = 'TO_' + name.upper()
                if newname_in in self._budget['name']:
                    newnames.append(newname_in)
                if newname_out in self._budget['name']:
                    newnames.append(newname_out)
            else:
                if name in self._budget['name']:
                    newnames.append(name)
        return newnames

    def _compute_net_budget(self):
        recnames = self.get_record_names()
        innames = [n for n in recnames if n.startswith('FROM_')]
        outnames = [n for n in recnames if n.startswith('TO_')]
        select_fields = ['totim', 'time_step', 'stress_period',
                         'name'] + list(self._zonenamedict.values())
        select_records_in = np.in1d(self._budget['name'], innames)
        select_records_out = np.in1d(self._budget['name'], outnames)
        in_budget = self._budget[select_fields][select_records_in]
        out_budget = self._budget[select_fields][select_records_out]
        net_budget = in_budget.copy()
        for f in [n for n in self._zonenamedict.values() if
                  n in select_fields]:
            net_budget[f] = np.array([r for r in in_budget[f]]) - np.array(
                [r for r in out_budget[f]])
        newnames = ['_'.join(n.split('_')[1:]) for n in net_budget['name']]
        net_budget['name'] = newnames
        return net_budget

    def __mul__(self, other):
        newbud = self._budget.copy()
        for f in self._zonenamedict.values():
            newbud[f] = np.array([r for r in newbud[f]]) * other
        idx = np.in1d(self._budget['name'], 'PERCENT_DISCREPANCY')
        newbud[:][idx] = self._budget[:][idx]
        newobj = self.copy()
        newobj._budget = newbud
        return newobj

    def __truediv__(self, other):
        newbud = self._budget.copy()
        for f in self._zonenamedict.values():
            newbud[f] = np.array([r for r in newbud[f]]) / float(other)
        idx = np.in1d(self._budget['name'], 'PERCENT_DISCREPANCY')
        newbud[:][idx] = self._budget[:][idx]
        newobj = self.copy()
        newobj._budget = newbud
        return newobj

    def __div__(self, other):
        newbud = self._budget.copy()
        for f in self._zonenamedict.values():
            newbud[f] = np.array([r for r in newbud[f]]) / float(other)
        idx = np.in1d(self._budget['name'], 'PERCENT_DISCREPANCY')
        newbud[:][idx] = self._budget[:][idx]
        newobj = self.copy()
        newobj._budget = newbud
        return newobj

    def __add__(self, other):
        newbud = self._budget.copy()
        for f in self._zonenamedict.values():
            newbud[f] = np.array([r for r in newbud[f]]) + other
        idx = np.in1d(self._budget['name'], 'PERCENT_DISCREPANCY')
        newbud[:][idx] = self._budget[:][idx]
        newobj = self.copy()
        newobj._budget = newbud
        return newobj

    def __sub__(self, other):
        newbud = self._budget.copy()
        for f in self._zonenamedict.values():
            newbud[f] = np.array([r for r in newbud[f]]) - other
        idx = np.in1d(self._budget['name'], 'PERCENT_DISCREPANCY')
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
    Saves a numpy array in a format readable by the zonebudget program executable.

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
            'Shape of the input array is not recognized: {}'.format(X.shape))
    if np.ma.is_masked(X):
        X = np.ma.filled(X, 0)

    nlay, nrow, ncol = X.shape

    if fmtin is not None:
        assert fmtin < ncol, 'The specified width is greater than the ' \
                             'number of columns in the array.'
    else:
        fmtin = ncol

    iprnmin = len(str(X.max()))
    if iprn is None or iprn <= iprnmin:
        iprn = iprnmin + 1

    formatter_str = '{{:>{iprn}}}'.format(iprn=iprn)
    formatter = formatter_str.format

    with open(fname, 'w') as f:
        header = '{nlay} {nrow} {ncol}\n'.format(nlay=nlay,
                                                 nrow=nrow,
                                                 ncol=ncol)
        f.write(header)
        for lay in range(nlay):
            record_2 = 'INTERNAL\t({fmtin}I{iprn})\n'.format(fmtin=fmtin,
                                                             iprn=iprn)
            f.write(record_2)
            if fmtin < ncol:
                for row in range(nrow):
                    rowvals = X[lay, row, :].ravel()
                    start = 0
                    end = start + fmtin
                    vals = rowvals[start:end]
                    while len(vals) > 0:
                        s = ''.join(
                            [formatter(int(val)) for val in vals]) + '\n'
                        f.write(s)
                        start = end
                        end = start + fmtin
                        vals = rowvals[start:end]
                        # vals = rowvals[start:end]
                        # if len(vals) > 0:
                        #     s = ''.join([formatter(int(val)) for val in vals]) + '\n'
                        #     f.write(s)
            elif fmtin == ncol:
                for row in range(nrow):
                    vals = X[lay, row, :].ravel()
                    f.write(
                        ''.join([formatter(int(val)) for val in vals]) + '\n')
    return


def read_zbarray(fname):
    """
    Reads an ascii array in a format readable by the zonebudget program executable.

    Parameters
    ----------
    fname :  str
        The path and name of the file to be written.

    Returns
    -------
    zones : numpy ndarray
        An integer array of the zones.
    """
    with open(fname, 'r') as f:
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
    locats = ['CONSTANT', 'INTERNAL', 'EXTERNAL']

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

            if locat == 'CONSTANT':
                iconst = int(rowitems[1])
            else:
                fmt = rowitems[1].strip('()')
                fmtin, iprn = [int(v) for v in fmt.split('I')]

        # ZONE DATA
        else:
            if locat == 'CONSTANT':
                vals = np.ones((nrow, ncol), dtype=np.int32) * iconst
                lay += 1
            elif locat == 'INTERNAL':
                # READ ZONES
                rowvals = [int(v) for v in rowitems]
                s = 'Too many values encountered on this line.'
                assert len(rowvals) <= fmtin, s
                vals.extend(rowvals)

            elif locat == 'EXTERNAL':
                # READ EXTERNAL FILE
                fname = rowitems[0]
                if not os.path.isfile(fname):
                    errmsg = 'Could not find external file "{}"'.format(fname)
                    raise Exception(errmsg)
                with open(fname, 'r') as ext_f:
                    ext_flines = ext_f.readlines()
                for ext_frow in ext_flines:
                    ext_frowitems = ext_frow.strip().split()
                    rowvals = [int(v) for v in ext_frowitems]
                    vals.extend(rowvals)
                if len(vals) != datalen:
                    errmsg = 'The number of values read from external ' \
                             'file "{}" does not match the expected ' \
                             'number.'.format(len(vals))
                    raise Exception(errmsg)
            else:
                # Should not get here
                raise Exception('Locat not recognized: {}'.format(locat))

                # IGNORE COMPOSITE ZONES

            if len(vals) == datalen:
                # place values for the previous layer into the zone array
                vals = np.array(vals, dtype=np.int32).reshape((nrow, ncol))
                zones[lay, :, :] = vals[:, :]
                lay += 1
            totlen += len(rowitems)
        i += 1
    s = 'The number of values read ({:,.0f})' \
        ' does not match the number expected' \
        ' ({:,.0f})'.format(totlen,
                            nlay * nrow * ncol)
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
    """
    Sort a tuple by the first n values
    :param tup:
    :param n:
    :return:
    """
    return tuple(sorted(tup, key=lambda t: t[:n]))
