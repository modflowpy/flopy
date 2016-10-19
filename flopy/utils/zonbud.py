from __future__ import print_function
import os
import numpy as np
from .binaryfile import CellBudgetFile
from copy import copy
from itertools import groupby


class Budget(object):
    """
    ZoneBudget Budget class. This is a wrapper around a numpy record array to allow users
    to save the record array to a formatted csv file.
    """

    def __init__(self, recordarray, kstpkper=None, totim=None):
        if kstpkper is None and totim is None:
            errmsg = 'Please specify a time step/stress period (kstpkper) ' \
                     'or simulation time (totim) for which the budget is ' \
                     'desired.'
            raise Exception(errmsg)
        self._recordarray = recordarray
        self.kstpkper = kstpkper
        self.totim = totim
        self._zonefields = [name for name in self._recordarray.dtype.names if 'ZONE' in name]
        self._massbalance = self._compute_mass_balance()
        return

    def get_records(self, recordlist=None, zones=None, aliases=None):
        """
        Returns the budget record array. Optionally, pass a list of
        (flow_dir, recname) tuples to get a subset of records. Pass
        a list of zones to get the desired records for just those
        zones.

        Parameters
        ----------
        recordlist : tuple or list of tuples
            A tuple or list of tuples containing flow direction and the name of
            the record desired [('IN', 'STORAGE'), ('OUT', 'TO ZONE 1')].
        zones : int or list of ints
            The zone(s) for which budget records are desired.
        aliases : dictionary
            A dictionary with key, value pairs of zones and aliases. Replaces
            the corresponding record and field names with the aliases provided.
            NOTE: When using this option in conjunction with a list of zones,
            the zone(s) passed may either be all strings (aliases) or
            integers, but may not not mixed.

        Returns
        -------
        records : numpy record array
            An array of the budget records

        """
        # Return a copy of the record array
        recordarray = self._recordarray.copy()

        # If the user has passed a dictionary of aliases, convert relevant field
        # names and records to comply with te desired names.
        if aliases is not None:
            assert isinstance(aliases, dict), 'Input aliases not recognized. Please pass a dictionary ' \
                                              'with key,value pairs of zone/alias.'
            newfieldnames = list(recordarray.dtype.names)
            for idx, name in enumerate(newfieldnames):
                if 'ZONE' in name:
                    zone = int(name.split()[-1])
                    if zone in aliases.keys():
                        newfieldnames[idx] = aliases[zone]
            recordarray.dtype.names = newfieldnames

            for idx, r in np.ndenumerate(recordarray):
                pieces = r['record'].split()
                if 'ZONE' in pieces:
                    zone = int(pieces[-1])
                    if zone in aliases.keys():
                        newrecname = '{} {}'.format(pieces[0], aliases[zone])
                        recordarray[idx]['record'] = newrecname

        if zones is not None:
            if isinstance(zones, int):
                zones = [zones]
            elif isinstance(zones, list) or isinstance(zones, tuple):
                zones = zones
            else:
                errmsg = 'Input zones are not recognized. Please ' \
                         'pass an integer or list of integers.'
                raise Exception(errmsg)
            if aliases is None:
                for zone in zones:
                    errmsg = 'Zone {} is not in the record array.'.format(zone)
                    assert 'ZONE {}'.format(zone) in self._zonefields, errmsg
                select_fields = ['ZONE {}'.format(z) for z in zones]
            else:
                # for alias in aliases.values():
                #     errmsg = 'Zone {} is not in the record array.'.format(alias)
                #     assert '{}'.format(alias) in self._zonefields, errmsg
                if isinstance(zones[0], str):
                    select_fields = [z for z in zones]
                else:
                    select_fields = [aliases[z] for z in zones]
        else:
            if aliases is None:
                select_fields = self._zonefields
            else:
                select_fields = [name for name in newfieldnames if name not in ['flow_dir', 'record']]
        select_fields = ['flow_dir', 'record'] + select_fields

        if recordlist is not None:
            if isinstance(recordlist, tuple):
                recordlist = [recordlist]
            elif isinstance(recordlist, list):
                recordlist = recordlist
            else:
                errmsg = 'Input records are not recognized. Please ' \
                         'pass a tuple of (flow_dir, recordname) or list of tuples.'
                raise Exception(errmsg)
            select_records = np.array([], dtype=np.int64)
            for flowdir, recname in recordlist:
                r = np.where((recordarray['flow_dir'] == flowdir) &
                             (recordarray['record'] == recname))
                select_records = np.append(select_records, r[0])
        else:
            flowdirs = recordarray['flow_dir']
            recnames = recordarray['record']
            select_records = np.where((recordarray['flow_dir'] == flowdirs) &
                                      (recordarray['record'] == recnames))

        records = recordarray[select_fields][select_records]
        return records

    def get_total_inflow(self, zones=None):
        """
        Returns the total inflow, summed by column. Optionally, pass a
        list of integer zones to get the total inflow for just those zones.

        Parameters
        ----------
        zones : int, list of ints
            The zone(s) for which total inflow is desired.

        Returns
        -------
        array : numpy array
            An array of the total inflow values

        """
        if zones is not None:
            if isinstance(zones, int):
                zones = [zones]
            elif isinstance(zones, list) or isinstance(zones, tuple):
                zones = zones
            else:
                errmsg = 'Input zones are not recognized. Please ' \
                         'pass an integer or list of integers.'
                raise Exception(errmsg)
            for zone in zones:
                errmsg = 'Zone {} is not in the record array.'.format(zone)
                assert 'ZONE {}'.format(zone) in self._zonefields, errmsg
            select_fields = ['ZONE {}'.format(z) for z in zones]
        else:
            select_fields = self._zonefields
        select_indices = np.where(self._massbalance['record'] == 'INFLOW')
        records = self._massbalance[select_fields][select_indices]
        array = np.array([r for r in records[0]])
        return array

    def get_total_outflow(self, zones=None):
        """
        Returns the total outflow, summed by column. Optionally, pass a
        list of integer zones to get the total outflow for just those zones.

        Parameters
        ----------
        zones : int, list of ints
            The zone(s) for which total outflow is desired.

        Returns
        -------
        array : numpy array
            An array of the total outflow values

        """
        if zones is not None:
            if isinstance(zones, int):
                zones = [zones]
            elif isinstance(zones, list) or isinstance(zones, tuple):
                zones = zones
            else:
                errmsg = 'Input zones are not recognized. Please ' \
                         'pass an integer or list of integers.'
                raise Exception(errmsg)
            for zone in zones:
                errmsg = 'Zone {} is not in the record array.'.format(zone)
                assert 'ZONE {}'.format(zone) in self._zonefields, errmsg
            select_fields = ['ZONE {}'.format(z) for z in zones]
        else:
            select_fields = self._zonefields
        select_indices = np.where(self._massbalance['record'] == 'OUTFLOW')
        records = self._massbalance[select_fields][select_indices]
        array = np.array([r for r in records[0]])
        return array

    def get_percent_error(self, zones=None):
        """
        Returns the percent error, summed by column. Optionally, pass a
        list of integer zones to get the percent error for just those zones.

        Parameters
        ----------
        zones : int, list of ints
            The zone(s) for which percent error is desired.

        Returns
        -------
        array : numpy array
            An array of the percent error values

        """
        if zones is not None:
            if isinstance(zones, int):
                zones = [zones]
            elif isinstance(zones, list) or isinstance(zones, tuple):
                zones = zones
            else:
                errmsg = 'Input zones are not recognized. Please ' \
                         'pass an integer or list of integers.'
                raise Exception(errmsg)
            for zone in zones:
                errmsg = 'Zone {} is not in the record array.'.format(zone)
                assert 'ZONE {}'.format(zone) in self._zonefields, errmsg
            select_fields = ['ZONE {}'.format(z) for z in zones]
        else:
            select_fields = self._zonefields
        select_indices = np.where(self._massbalance['record'] == 'ERROR')
        records = self._massbalance[select_fields][select_indices]
        array = np.array([r for r in records[0]])
        return array

    def get_mass_balance(self, zones=None):
        """
        Returns the mass-balance records. Optionally, pass a
        list of integer zones to get the mass-balance records for just those zones.

        Parameters
        ----------
        zones : int, list of ints
            The zone(s) for which percent error is desired.

        Returns
        -------
        records : numpy record array
            An array of the mass-balance records

        """
        if zones is not None:
            if isinstance(zones, int):
                zones = [zones]
            elif isinstance(zones, list) or isinstance(zones, tuple):
                zones = zones
            else:
                errmsg = 'Input zones are not recognized. Please ' \
                         'pass an integer or list of integers.'
                raise Exception(errmsg)
            for zone in zones:
                errmsg = 'Zone {} is not in the record array.'.format(zone)
                assert 'ZONE {}'.format(zone) in self._zonefields, errmsg
            select_fields = ['ZONE {}'.format(z) for z in zones]
        else:
            select_fields = self._zonefields
        select_fields = ['record'] + select_fields
        records = self._massbalance[select_fields]
        return records

    def _compute_mass_balance(self):
        # Returns a record array with total inflow, total outflow,
        # and percent error summed by column.

        # Compute inflows
        idx = np.where(self._recordarray['flow_dir'] == 'IN')[0]
        a = _numpyvoid2numeric(self._recordarray[self._zonefields][idx])
        intot = np.array(a.sum(axis=0))

        # Compute outflows
        idx = np.where(self._recordarray['flow_dir'] == 'OUT')[0]
        a = _numpyvoid2numeric(self._recordarray[self._zonefields][idx])
        outot = np.array(a.sum(axis=0))

        # Compute percent error
        ins_minus_out = intot - outot
        ins_plus_out = intot + outot
        pcterr = 100 * ins_minus_out / (ins_plus_out / 2.)
        pcterr = np.nan_to_num(pcterr)

        # Create the mass-balance record array
        dtype_list = [('record', (str, 7))] + [('{}'.format(f), np.float64) for f in self._zonefields]
        dtype = np.dtype(dtype_list)
        mb = np.array([], dtype=dtype)
        mb = np.append(mb, np.array(tuple(['INFLOW'] + list(intot)), dtype=dtype))
        mb = np.append(mb, np.array(tuple(['OUTFLOW'] + list(outot)), dtype=dtype))
        mb = np.append(mb, np.array(tuple(['ERROR'] + list(pcterr)), dtype=dtype))
        return mb

    def to_csv(self, fname, write_format='zonbud', formatter=None):
        """
        Saves the Budget object record array to a formatted
        comma-separated values file.

        Parameters
        ----------
        fname : str
            The name of the output comma-separated values file.
        write_format : str
            A write option for output comma-separated values file.
        formatter : function
            A string-formatter function for formatting floats.

        Returns
        -------
        None

        """
        assert write_format.lower() in ['pandas', 'zonbud'], 'Format must be one of "pandas" or "zonbud".'

        if formatter is None:
            formatter = '{:.16e}'.format

        if write_format.lower() == 'pandas':
            with open(fname, 'w') as f:

                # Write header
                f.write(','.join(self._recordarray.dtype.names)+'\n')

                # Write IN terms
                select_indices = np.where(self._recordarray['flow_dir'] == 'IN')
                for rec in self._recordarray[select_indices[0]]:
                    items = []
                    for i in rec:
                        if isinstance(i, str):
                            items.append(i)
                        else:
                            items.append(formatter(i))
                    f.write(','.join(items)+'\n')
                ins_sum = self.get_total_inflow()
                f.write(','.join([' ', 'Total IN'] + [formatter(i) for i in ins_sum])+'\n')

                # Write OUT terms
                select_indices = np.where(self._recordarray['flow_dir'] == 'OUT')
                for rec in self._recordarray[select_indices[0]]:
                    items = []
                    for i in rec:
                        if isinstance(i, str):
                            items.append(i)
                        else:
                            items.append(formatter(i))
                    f.write(','.join(items) + '\n')
                out_sum = self.get_total_outflow()
                f.write(','.join([' ', 'Total OUT'] + [formatter(i) for i in out_sum])+'\n')

                # Write mass balance terms
                ins_minus_out = self.get_total_inflow() - self.get_total_outflow()
                pcterr = self.get_percent_error()
                f.write(','.join([' ', 'IN-OUT'] + [formatter(i) for i in ins_minus_out])+'\n')
                f.write(','.join([' ', 'Percent Error'] + [formatter(i) for i in pcterr])+'\n')

        elif write_format.lower() == 'zonbud':
            with open(fname, 'w') as f:

                # Write header
                header = ''
                if self.kstpkper is not None:
                    kstp1 = self.kstpkper[0]+1
                    kper1 = self.kstpkper[1]+1
                    header = 'Time Step, {kstp}, Stress Period, {kper}\n'.format(kstp=kstp1, kper=kper1)
                elif self.totim is not None:
                    header = 'Sim. Time, {totim}\n'.format(totim=self.totim)
                f.write(header)
                f.write(','.join([' '] + [field for field in self._recordarray.dtype.names[2:]])+'\n')

                # Write IN terms
                f.write(','.join([' '] + ['IN']*(len(self._recordarray.dtype.names[1:])-1))+'\n')
                select_indices = np.where(self._recordarray['flow_dir'] == 'IN')
                for rec in self._recordarray[select_indices[0]]:
                    items = []
                    for i in list(rec)[1:]:
                        if isinstance(i, str):
                            items.append(i)
                        else:
                            items.append(formatter(i))
                    f.write(','.join(items)+'\n')
                ins_sum = self.get_total_inflow()
                f.write(','.join(['Total IN'] + [formatter(i) for i in ins_sum])+'\n')

                # Write OUT terms
                f.write(','.join([' '] + ['OUT']*(len(self._recordarray.dtype.names[1:])-1))+'\n')
                select_indices = np.where(self._recordarray['flow_dir'] == 'OUT')
                for rec in self._recordarray[select_indices[0]]:
                    items = []
                    for i in list(rec)[1:]:
                        if isinstance(i, str):
                            items.append(i)
                        else:
                            items.append(formatter(i))
                    f.write(','.join(items) + '\n')
                out_sum = self.get_total_outflow()
                f.write(','.join(['Total OUT'] + [formatter(i) for i in out_sum])+'\n')

                # Write mass balance terms
                ins_minus_out = self.get_total_inflow() - self.get_total_outflow()
                pcterr = self.get_percent_error()
                f.write(','.join(['IN-OUT'] + [formatter(i) for i in ins_minus_out])+'\n')
                f.write(','.join(['Percent Error'] + [formatter(i) for i in pcterr])+'\n')
        return


class ZoneBudget(object):
    """
    ZoneBudget class

    Example usage:

    >>>from flopy.utils import ZoneBudget
    >>>zb = ZoneBudget('zonebudtest.cbc')
    >>>zon = np.loadtxt('zones.txt')
    >>>bud = zb.get_budget(zon, kstpkper=(0, 0))
    >>>bud.to_csv('zonebudtest.csv')
    """
    def __init__(self, cbc_file):

        if isinstance(cbc_file, CellBudgetFile):
            self.cbc = cbc_file
        elif isinstance(cbc_file, str) and os.path.isfile(cbc_file):
            self.cbc = CellBudgetFile(cbc_file)
        else:
            raise Exception('Cannot load cell budget file: {}.'.format(cbc_file))

        # All record names in the cell-by-cell budget binary file
        self.record_names = [n.strip().decode("utf-8") for n in self.cbc.unique_record_names()]

        # Get imeth for each record in the CellBudgetFile record list
        self.imeth = {}
        for record in self.cbc.recordarray:
            self.imeth[record['text'].strip().decode("utf-8")] = record['imeth']

        # INTERNAL FLOW TERMS ARE USED TO CALCULATE FLOW BETWEEN ZONES.
        # CONSTANT-HEAD TERMS ARE USED TO IDENTIFY WHERE CONSTANT-HEAD CELLS ARE AND THEN USE
        # FACE FLOWS TO DETERMINE THE AMOUNT OF FLOW.
        # SWIADDTO--- terms are used by the SWI2 groundwater flow process.
        internal_flow_terms = ['CONSTANT HEAD', 'FLOW RIGHT FACE', 'FLOW FRONT FACE', 'FLOW LOWER FACE',
                               'SWIADDTOCH', 'SWIADDTOFRF', 'SWIADDTOFFF', 'SWIADDTOFLF']

        # Source/sink/storage term record names
        # These are all of the terms that are not related to constant
        # head cells or face flow terms
        self.ssst_record_names = [n for n in self.record_names
                                  if n not in internal_flow_terms]

        # Check the shape of the cbc budget file arrays
        self.cbc_shape = self.get_model_shape()
        self.nlay, self.nrow, self.ncol = self.cbc_shape

        self.float_type = np.float64
        return

    def get_model_shape(self):
        return self.cbc.get_data(idx=0, full3D=True)[0].shape

    def get_budget(self, z, kstpkper=None, totim=None):
        """
        Creates a budget for the specified zone array. This function only supports the
        use of a single time step/stress period or time.

        Parameters
        ----------
        z : ndarray
            The array containing to zones to be used.
        kstpkper : tuple of ints
            A tuple containing the time step and stress period (kstp, kper).
            The kstp and kper values are zero based.
        totim : float
            The simulation time.

        Returns
        -------
        A Budget object

        """
        # Check the keyword arguments
        if kstpkper is not None:
            s = 'The specified time step/stress period ' \
                'does not exist {}'.format(kstpkper)
            assert kstpkper in self.cbc.get_kstpkper(), s
        elif totim is not None:
            s = 'The specified simulation time ' \
                'does not exist {}'.format(totim)
            assert totim in self.cbc.get_times(), s
        else:
            # No time step/stress period or simulation time pass
            errmsg = 'Please specify a time step/stress period (kstpkper) ' \
                     'or simulation time (totim) for which the budget is ' \
                     'desired.'
            raise Exception(errmsg)

        # Zones must be passed as an array
        assert isinstance(z, np.ndarray), 'Please pass zones as type {}'.format(np.ndarray)

        # Check for negative zone values
        for zi in np.unique(z):
            if zi < 0:
                raise Exception('Negative zone value(s) found:', zi)

        # Make sure the input zone array has the same shape as the cell budget file
        if len(z.shape) == 2 and self.nlay == 1:
            # Reshape a 2-D array to 3-D to match output from
            # the CellBudgetFile object.
            izone = np.zeros(self.cbc_shape, np.int32)
            izone[0, :, :] = z[:, :]
        elif len(z.shape) == 2 and self.nlay > 1:
            # 2-D array specified, but model is more than 1 layer. Don't assume
            # user wants same zones for all layers.
            raise Exception('Zone array and CellBudgetFile shapes '
                            'do not match {} {}'.format(z.shape, self.cbc_shape))
        elif len(z.shape) == 3:
            izone = z.copy()
        else:
            raise Exception('Shape of the zone array is not recognized: {}'.format(z.shape))

        assert izone.shape == self.cbc_shape, \
            'Shape of input zone array {} does not ' \
            'match the cell by cell ' \
            'budget file {}'.format(izone.shape, self.cbc_shape)

        # List of unique zones numbers
        lstzon = [z for z in np.unique(izone)]

        # Initialize an array to track where the constant head cells
        # are located.
        ich = np.zeros(self.cbc_shape, np.int32)

        # Create empty array for the budget terms.
        # This array has the structure: ('flow direction', 'record name', value zone 1, value zone 2, etc.)
        self._initialize_records(lstzon)

        # Create a throwaway list of all record names
        reclist = list(self.record_names)

        if 'CONSTANT HEAD' in reclist:
            reclist.remove('CONSTANT HEAD')
            chd = self.cbc.get_data(text='CONSTANT HEAD', full3D=True, kstpkper=kstpkper, totim=totim)[0]
            ich = np.zeros(self.cbc_shape, np.int32)
            ich[chd != 0] = 1
        if 'FLOW RIGHT FACE' in reclist:
            reclist.remove('FLOW RIGHT FACE')
            self._accumulate_flow_frf('FLOW RIGHT FACE', izone, ich, kstpkper=kstpkper, totim=totim)
        if 'FLOW FRONT FACE' in reclist:
            reclist.remove('FLOW FRONT FACE')
            self._accumulate_flow_fff('FLOW FRONT FACE', izone, ich, kstpkper=kstpkper, totim=totim)
        if 'FLOW LOWER FACE' in reclist:
            reclist.remove('FLOW LOWER FACE')
            self._accumulate_flow_flf('FLOW LOWER FACE', izone, ich, kstpkper=kstpkper, totim=totim)
        if 'SWIADDTOCH' in reclist:
            reclist.remove('SWIADDTOCH')
            swichd = self.cbc.get_data(text='SWIADDTOCH', full3D=True, kstpkper=kstpkper, totim=totim)[0]
            swiich = np.zeros(self.cbc_shape, np.int32)
            swiich[swichd != 0] = 1
        if 'SWIADDTOFRF' in reclist:
            reclist.remove('SWIADDTOFRF')
            self._accumulate_flow_frf('SWIADDTOFRF', izone, swiich, kstpkper=kstpkper, totim=totim)
        if 'SWIADDTOFFF' in reclist:
            reclist.remove('SWIADDTOFFF')
            self._accumulate_flow_fff('SWIADDTOFFF', izone, swiich, kstpkper=kstpkper, totim=totim)
        if 'SWIADDTOFLF' in reclist:
            reclist.remove('SWIADDTOFLF')
            self._accumulate_flow_flf('SWIADDTOFLF', izone, swiich, kstpkper=kstpkper, totim=totim)

        # NOT AN INTERNAL FLOW TERM, SO MUST BE A SOURCE TERM OR STORAGE
        # ACCUMULATE THE FLOW BY ZONE
        # iterate over remaining items in the list
        for recname in reclist:
            imeth = self.imeth[recname]

            data = self.cbc.get_data(text=recname, kstpkper=kstpkper, totim=totim)
            if len(data) == 0:
                # Empty data, can occur during the first time step of a transient model when
                # storage terms are zero and not in the cell-budget file.
                continue
            else:
                data = data[0]

            if imeth == 2 or imeth == 5:
                # LIST
                qin = np.ma.zeros((self.nlay * self.nrow * self.ncol), self.float_type)
                qout = np.ma.zeros((self.nlay * self.nrow * self.ncol), self.float_type)
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
                raise Exception('Unrecognized "imeth" for {} record: {}'.format(recname, imeth))
            self._accumulate_flow_ssst(recname, qin, qout, izone, lstzon)

        # Create the budget object, which is primarily a wrapper around the
        # budget record array that allows the user to write out the budget
        # to a csv file. Pass along the kwargs which hold the desired time
        # step/stress period or totim so we can print it to the header of
        # the output file.
        return Budget(self.zonbudrecords, kstpkper=kstpkper, totim=totim)

    def _build_empty_record(self, flow_dir, recname, lstzon):
        # Builds empty records based on the specified flow direction and
        # record name for the given list of zones.
        recs = np.array(tuple([flow_dir, recname] + [0. for _ in lstzon if _ != 0]),
                        dtype=self.zonbudrecords.dtype)
        self.zonbudrecords = np.append(self.zonbudrecords, recs)
        return

    def _initialize_records(self, lstzon):
        # Initialize the budget record array which will store all of the
        # fluxes in the cell-budget file.
        dtype_list = [('flow_dir', (str, 3)), ('record', (str, 20))]
        dtype_list += [('ZONE {:d}'.format(z), self.float_type) for z in lstzon if z != 0]
        dtype = np.dtype(dtype_list)
        self.zonbudrecords = np.array([], dtype=dtype)

        # Add "in" records
        if 'STORAGE' in self.record_names:
            self._build_empty_record('IN', 'STORAGE', lstzon)
        if 'CONSTANT HEAD' in self.record_names:
            self._build_empty_record('IN', 'CONSTANT HEAD', lstzon)
        for recname in self.ssst_record_names:
            if recname != 'STORAGE':
                self._build_empty_record('IN', recname, lstzon)
        for z in lstzon:
            self._build_empty_record('IN', 'FROM ZONE {}'.format(z), lstzon)

        # Add "out" records
        if 'STORAGE' in self.record_names:
            self._build_empty_record('OUT', 'STORAGE', lstzon)
        if 'CONSTANT HEAD' in self.record_names:
            self._build_empty_record('OUT', 'CONSTANT HEAD', lstzon)
        for recname in self.ssst_record_names:
            if recname != 'STORAGE':
                self._build_empty_record('OUT', recname, lstzon)
        for z in lstzon:
            self._build_empty_record('OUT', 'TO ZONE {}'.format(z), lstzon)
        return

    def _update_record(self, flow_dir, recname, colname, flux):
        # Update the budget record array with the flux for the specified
        # flow direction (in/out), record name, and column (exclusive of
        # ZONE 0).
        if colname != 'ZONE 0':

            if 'ZONE' in recname:
                # Make sure the flux is between different zones
                recname_z = int(recname.split()[-1])
                colname_z = int(colname.split()[-1])
                if recname_z == colname_z:
                    errmsg = 'Circular flow detected: {}\t{}\t{}\t{}'.format(flow_dir,
                                                                             recname,
                                                                             colname,
                                                                             flux)
                    raise Exception(errmsg)

            rowidx = np.where((self.zonbudrecords['flow_dir'] == flow_dir) &
                              (self.zonbudrecords['record'] == recname))
            self.zonbudrecords[colname][rowidx] += flux
        return

    def _accumulate_flow_frf(self, recname, izone, ich, **kwargs):
        """
        C
        C-----"FLOW RIGHT FACE"  COMPUTE FLOW BETWEEN ZONES ACROSS COLUMNS.
        C-----COMPUTE FLOW ONLY BETWEEN A ZONE AND A HIGHER ZONE -- FLOW FROM
        C-----ZONE 4 TO 3 IS THE NEGATIVE OF FLOW FROM 3 TO 4.
        C-----1ST, CALCULATE FLOW BETWEEN NODE J,I,K AND J-1,I,K
        300   IF(NCOL.LT.2) RETURN
              DO 340 K=1,NLAY
              DO 340 I=1,NROW
              DO 340 J=2,NCOL
              NZ=IZONE(J,I,K)
              JL=J-1
              NZL=IZONE(JL,I,K)
              IF(NZL.LE.NZ) GO TO 340
        C  Don't include CH to CH flow (can occur if CHTOCH option is used)
              IF(ICH(J,I,K).EQ.1 .AND. ICH(J-1,I,K).EQ.1) GO TO 340
              DBUFF=BUFFD(JL,I,K)
              IF(DBUFF.LT.DZERO) THEN
                 VBZNFL(2,NZ,NZL)=VBZNFL(2,NZ,NZL)-DBUFF
              ELSE
                 VBZNFL(1,NZ,NZL)=VBZNFL(1,NZ,NZL)+DBUFF
              END IF
          340 CONTINUE
        C
        C-----FLOW BETWEEN NODE J,I,K AND J+1,I,K
              DO 370 K=1,NLAY
              DO 370 I=1,NROW
              DO 370 J=1,NCOL-1
              NZ=IZONE(J,I,K)
              JR=J+1
              NZR=IZONE(JR,I,K)
              IF(NZR.LE.NZ) GO TO 370
        C  Don't include CH to CH flow (can occur if CHTOCH option is used)
              IF(ICH(J,I,K).EQ.1 .AND. ICH(J+1,I,K).EQ.1) GO TO 370
              DBUFF=BUFFD(J,I,K)
              IF(DBUFF.LT.DZERO) THEN
                 VBZNFL(1,NZ,NZR)=VBZNFL(1,NZ,NZR)-DBUFF
              ELSE
                 VBZNFL(2,NZ,NZR)=VBZNFL(2,NZ,NZR)+DBUFF
              END IF
          370 CONTINUE
        C
        C-----CALCULATE FLOW TO CONSTANT-HEAD CELLS IN THIS DIRECTION
              DO 395 K=1,NLAY
              DO 395 I=1,NROW
              DO 395 J=1,NCOL
              IF(ICH(J,I,K).EQ.0) GO TO 395
              NZ=IZONE(J,I,K)
              IF(NZ.EQ.0) GO TO 395
              IF(J.EQ.NCOL) GO TO 380
              IF(ICH(J+1,I,K).EQ.1) GO TO 380
              DBUFF=BUFFD(J,I,K)
              IF(DBUFF.EQ.DZERO) THEN
              ELSE IF(DBUFF.LT.DZERO) THEN
                 VBVL(2,MSUMCH,NZ)=VBVL(2,MSUMCH,NZ)-DBUFF
              ELSE
                 VBVL(1,MSUMCH,NZ)=VBVL(1,MSUMCH,NZ)+DBUFF
              END IF
        380   IF(J.EQ.1) GO TO 395
              IF(ICH(J-1,I,K).EQ.1) GO TO 395
              DBUFF=BUFFD(J-1,I,K)
              IF(DBUFF.EQ.DZERO) THEN
              ELSE IF(DBUFF.LT.DZERO) THEN
                 VBVL(1,MSUMCH,NZ)=VBVL(1,MSUMCH,NZ)-DBUFF
              ELSE
                 VBVL(2,MSUMCH,NZ)=VBVL(2,MSUMCH,NZ)+DBUFF
              END IF
        395   CONTINUE
              RETURN
        """
        if self.ncol >= 2:
            data = self.cbc.get_data(text=recname, **kwargs)[0]

            # "FLOW RIGHT FACE"  COMPUTE FLOW BETWEEN ZONES ACROSS COLUMNS.
            # COMPUTE FLOW ONLY BETWEEN A ZONE AND A HIGHER ZONE -- FLOW FROM
            # ZONE 4 TO 3 IS THE NEGATIVE OF FLOW FROM 3 TO 4.
            # 1ST, CALCULATE FLOW BETWEEN NODE J,I,K AND J-1,I,K

            l, r, c = np.where(izone[:, :, 1:] > izone[:, :, :-1])

            # Adjust column values to account for the starting position of "nz"
            c = np.copy(c) + 1

            # Define the zone from which flow is coming
            cl = c-1
            nzl = izone[l, r, cl]

            # Define the zone to which flow is going
            nz = izone[l, r, c]

            # Get the face flow
            q = data[l, r, cl]

            # Get indices where flow face values are positive (flow out of higher zone)
            # Don't include CH to CH flow (can occur if CHTOCH option is used)
            # Create an interable tuple of (from zone, to zone, flux)
            # Then group tuple by (from_zone, to_zone) and sum the flux values
            idx = np.where((q > 0) & ((ich[l, r, c] != 1) | (ich[l, r, cl] != 1)))
            fluxes = sum_flux_tuples(nzl[idx],
                                     nz[idx],
                                     np.abs(q[idx]))
            for (fz, tz, flux) in fluxes:
                self._update_record('IN', 'FROM ZONE {}'.format(fz), 'ZONE {}'.format(tz), flux)
                self._update_record('OUT', 'TO ZONE {}'.format(tz), 'ZONE {}'.format(fz), flux)

            # Get indices where flow face values are negative (flow into higher zone)
            # Don't include CH to CH flow (can occur if CHTOCH option is used)
            # Create an interable tuple of (from zone, to zone, flux)
            # Then group tuple by (from_zone, to_zone) and sum the flux values
            idx = np.where((q < 0) & ((ich[l, r, c] != 1) | (ich[l, r, cl] != 1)))
            fluxes = sum_flux_tuples(nz[idx],
                                     nzl[idx],
                                     np.abs(q[idx]))
            for (fz, tz, flux) in fluxes:
                self._update_record('IN', 'FROM ZONE {}'.format(fz), 'ZONE {}'.format(tz), flux)
                self._update_record('OUT', 'TO ZONE {}'.format(tz), 'ZONE {}'.format(fz), flux)

            # FLOW BETWEEN NODE J,I,K AND J+1,I,K
            l, r, c = np.where(izone[:, :, :-1] > izone[:, :, 1:])

            # Define the zone from which flow is coming
            nz = izone[l, r, c]

            # Define the zone to which flow is going
            cr = c+1
            nzr = izone[l, r, cr]

            # Get the face flow
            q = data[l, r, c]

            # Get indices where flow face values are positive (flow out of higher zone)
            # Don't include CH to CH flow (can occur if CHTOCH option is used)
            # Create an interable tuple of (from zone, to zone, flux)
            # Then group tuple by (from_zone, to_zone) and sum the flux values
            idx = np.where((q > 0) & ((ich[l, r, c] != 1) | (ich[l, r, cr] != 1)))
            fluxes = sum_flux_tuples(nz[idx],
                                     nzr[idx],
                                     np.abs(q[idx]))
            for (fz, tz, flux) in fluxes:
                self._update_record('IN', 'FROM ZONE {}'.format(fz), 'ZONE {}'.format(tz), flux)
                self._update_record('OUT', 'TO ZONE {}'.format(tz), 'ZONE {}'.format(fz), flux)

            # Get indices where flow face values are negative (flow into higher zone)
            # Don't include CH to CH flow (can occur if CHTOCH option is used)
            # Create an interable tuple of (from zone, to zone, flux)
            # Then group tuple by (from_zone, to_zone) and sum the flux values
            idx = np.where((q < 0) & ((ich[l, r, c] != 1) | (ich[l, r, cr] != 1)))
            fluxes = sum_flux_tuples(nzr[idx],
                                     nz[idx],
                                     np.abs(q[idx]))
            for (fz, tz, flux) in fluxes:
                self._update_record('IN', 'FROM ZONE {}'.format(fz), 'ZONE {}'.format(tz), flux)
                self._update_record('OUT', 'TO ZONE {}'.format(tz), 'ZONE {}'.format(fz), flux)

            # CALCULATE FLOW TO CONSTANT-HEAD CELLS IN THIS DIRECTION
            l, r, c = np.where(ich == 1)
            l, r, c = l[c > 0], r[c > 0], c[c > 0]
            cl = c - 1
            nzl = izone[l, r, cl]
            nz = izone[l, r, c]
            q = data[l, r, cl]
            idx = np.where((q > 0) & ((ich[l, r, c] != 1) | (ich[l, r, cl] != 1)))
            fluxes = sum_flux_tuples(nzl[idx],
                                     nz[idx],
                                     np.abs(q[idx]))
            for (fz, tz, flux) in fluxes:
                self._update_record('OUT', 'CONSTANT HEAD', 'ZONE {}'.format(tz), flux)
            idx = np.where((q < 0) & ((ich[l, r, c] != 1) | (ich[l, r, cl] != 1)))
            fluxes = sum_flux_tuples(nz[idx],
                                     nzl[idx],
                                     np.abs(q[idx]))
            for (fz, tz, flux) in fluxes:
                self._update_record('IN', 'CONSTANT HEAD', 'ZONE {}'.format(fz), flux)
            l, r, c = np.where(ich == 1)
            l, r, c = l[c < self.ncol-1], r[c < self.ncol-1], c[c < self.ncol-1]
            nz = izone[l, r, c]
            cr = c+1
            nzr = izone[l, r, cr]
            q = data[l, r, c]
            idx = np.where((q > 0) & ((ich[l, r, c] != 1) | (ich[l, r, cr] != 1)))
            fluxes = sum_flux_tuples(nz[idx],
                                     nzr[idx],
                                     np.abs(q[idx]))
            for (fz, tz, flux) in fluxes:
                self._update_record('OUT', 'CONSTANT HEAD', 'ZONE {}'.format(tz), flux)
            idx = np.where((q < 0) & ((ich[l, r, c] != 1) | (ich[l, r, cr] != 1)))
            fluxes = sum_flux_tuples(nzr[idx],
                                     nz[idx],
                                     np.abs(q[idx]))
            for (fz, tz, flux) in fluxes:
                self._update_record('IN', 'CONSTANT HEAD', 'ZONE {}'.format(fz), flux)
        return

    def _accumulate_flow_fff(self, recname, izone, ich, **kwargs):
        """
        C
        C-----"FLOW FRONT FACE"
        C-----CALCULATE FLOW BETWEEN NODE J,I,K AND J,I-1,K
        400   IF(NROW.LT.2) RETURN
              DO 440 K=1,NLAY
              DO 440 I=2,NROW
              DO 440 J=1,NCOL
              NZ=IZONE(J,I,K)
              IA=I-1
              NZA=IZONE(J,IA,K)
              IF(NZA.LE.NZ) GO TO 440
        C  Don't include CH to CH flow (can occur if CHTOCH option is used)
              IF(ICH(J,I,K).EQ.1 .AND. ICH(J,I-1,K).EQ.1) GO TO 440
              DBUFF=BUFFD(J,IA,K)
              IF(DBUFF.LT.DZERO) THEN
                 VBZNFL(2,NZ,NZA)=VBZNFL(2,NZ,NZA)-DBUFF
              ELSE
                 VBZNFL(1,NZ,NZA)=VBZNFL(1,NZ,NZA)+DBUFF
              END IF
          440 CONTINUE
        C
        C-----CALCULATE FLOW BETWEEN NODE J,I,K AND J,I+1,K
              DO 470 K=1,NLAY
              DO 470 I=1,NROW-1
              DO 470 J=1,NCOL
              NZ=IZONE(J,I,K)
              IB=I+1
              NZB=IZONE(J,IB,K)
              IF(NZB.LE.NZ) GO TO 470
        C  Don't include CH to CH flow (can occur if CHTOCH option is used)
              IF(ICH(J,I,K).EQ.1 .AND. ICH(J,I+1,K).EQ.1) GO TO 470
              DBUFF=BUFFD(J,I,K)
              IF(DBUFF.LT.DZERO) THEN
                 VBZNFL(1,NZ,NZB)=VBZNFL(1,NZ,NZB)-DBUFF
              ELSE
                 VBZNFL(2,NZ,NZB)=VBZNFL(2,NZ,NZB)+DBUFF
              END IF
          470 CONTINUE
        C
        C-----CALCULATE FLOW TO CONSTANT-HEAD CELLS IN THIS DIRECTION
              DO 495 K=1,NLAY
              DO 495 I=1,NROW
              DO 495 J=1,NCOL
              IF(ICH(J,I,K).EQ.0) GO TO 495
              NZ=IZONE(J,I,K)
              IF(NZ.EQ.0) GO TO 495
              IF(I.EQ.NROW) GO TO 480
              IF(ICH(J,I+1,K).EQ.1) GO TO 480
              DBUFF=BUFFD(J,I,K)
              IF(DBUFF.EQ.DZERO) THEN
              ELSE IF(DBUFF.LT.DZERO) THEN
                 VBVL(2,MSUMCH,NZ)=VBVL(2,MSUMCH,NZ)-DBUFF
              ELSE
                 VBVL(1,MSUMCH,NZ)=VBVL(1,MSUMCH,NZ)+DBUFF
              END IF
        480   IF(I.EQ.1) GO TO 495
              IF(ICH(J,I-1,K).EQ.1) GO TO 495
              DBUFF=BUFFD(J,I-1,K)
              IF(DBUFF.EQ.DZERO) THEN
              ELSE IF(DBUFF.LT.DZERO) THEN
                 VBVL(1,MSUMCH,NZ)=VBVL(1,MSUMCH,NZ)-DBUFF
              ELSE
                 VBVL(2,MSUMCH,NZ)=VBVL(2,MSUMCH,NZ)+DBUFF
              END IF
        495   CONTINUE
              RETURN
        """
        if self.nrow >= 2:
            data = self.cbc.get_data(text=recname, **kwargs)[0]

            # "FLOW FRONT FACE"
            # CALCULATE FLOW BETWEEN NODE J,I,K AND J,I-1,K
            l, r, c = np.where(izone[:, 1:, :] < izone[:, :-1, :])
            r = np.copy(r) + 1
            ra = r-1
            nza = izone[l, ra, c]
            nz = izone[l, r, c]
            q = data[l, ra, c]
            idx = np.where((q > 0) & ((ich[l, r, c] != 1) | (ich[l, ra, c] != 1)))
            fluxes = sum_flux_tuples(nza[idx],
                                     nz[idx],
                                     np.abs(q[idx]))
            for (fz, tz, flux) in fluxes:
                self._update_record('IN', 'FROM ZONE {}'.format(fz), 'ZONE {}'.format(tz), flux)
                self._update_record('OUT', 'TO ZONE {}'.format(tz), 'ZONE {}'.format(fz), flux)
            idx = np.where((q < 0) & ((ich[l, r, c] != 1) | (ich[l, ra, c] != 1)))
            fluxes = sum_flux_tuples(nz[idx],
                                     nza[idx],
                                     np.abs(q[idx]))
            for (fz, tz, flux) in fluxes:
                self._update_record('IN', 'FROM ZONE {}'.format(fz), 'ZONE {}'.format(tz), flux)
                self._update_record('OUT', 'TO ZONE {}'.format(tz), 'ZONE {}'.format(fz), flux)

            # CALCULATE FLOW BETWEEN NODE J,I,K AND J,I+1,K.
            l, r, c = np.where(izone[:, :-1, :] < izone[:, 1:, :])
            nz = izone[l, r, c]
            rb = r + 1
            nzb = izone[l, rb, c]
            q = data[l, r, c]
            idx = np.where((q > 0) & ((ich[l, r, c] != 1) | (ich[l, rb, c] != 1)))
            fluxes = sum_flux_tuples(nz[idx],
                                     nzb[idx],
                                     np.abs(q[idx]))
            for (fz, tz, flux) in fluxes:
                self._update_record('IN', 'FROM ZONE {}'.format(fz), 'ZONE {}'.format(tz), flux)
                self._update_record('OUT', 'TO ZONE {}'.format(tz), 'ZONE {}'.format(fz), flux)
            idx = np.where((q < 0) & ((ich[l, r, c] != 1) | (ich[l, rb, c] != 1)))
            fluxes = sum_flux_tuples(nzb[idx],
                                     nz[idx],
                                     np.abs(q[idx]))
            for (fz, tz, flux) in fluxes:
                self._update_record('IN', 'FROM ZONE {}'.format(fz), 'ZONE {}'.format(tz), flux)
                self._update_record('OUT', 'TO ZONE {}'.format(tz), 'ZONE {}'.format(fz), flux)

            # CALCULATE FLOW TO CONSTANT-HEAD CELLS IN THIS DIRECTION
            l, r, c = np.where(ich == 1)
            l, r, c = l[r > 0], r[r > 0], c[r > 0]
            ra = r-1
            nza = izone[l, ra, c]
            nz = izone[l, r, c]
            q = data[l, ra, c]
            idx = np.where((q > 0) & ((ich[l, r, c] != 1) | (ich[l, ra, c] != 1)))
            fluxes = sum_flux_tuples(nza[idx],
                                     nz[idx],
                                     np.abs(q[idx]))
            for (fz, tz, flux) in fluxes:
                self._update_record('OUT', 'CONSTANT HEAD', 'ZONE {}'.format(tz), flux)
            idx = np.where((q < 0) & ((ich[l, r, c] != 1) | (ich[l, ra, c] != 1)))
            fluxes = sum_flux_tuples(nz[idx],
                                     nza[idx],
                                     np.abs(q[idx]))
            for (fz, tz, flux) in fluxes:
                self._update_record('IN', 'CONSTANT HEAD', 'ZONE {}'.format(fz), flux)
            l, r, c = np.where(ich == 1)
            l, r, c = l[r < self.nrow-1], r[r < self.nrow-1], c[r < self.nrow-1]
            nz = izone[l, r, c]
            rb = r+1
            nzb = izone[l, rb, c]
            q = data[l, r, c]
            idx = np.where((q > 0) & ((ich[l, r, c] != 1) | (ich[l, rb, c] != 1)))
            fluxes = sum_flux_tuples(nz[idx],
                                     nzb[idx],
                                     np.abs(q[idx]))
            for (fz, tz, flux) in fluxes:
                self._update_record('OUT', 'CONSTANT HEAD', 'ZONE {}'.format(tz), flux)
            idx = np.where((q < 0) & ((ich[l, r, c] != 1) | (ich[l, rb, c] != 1)))
            fluxes = sum_flux_tuples(nzb[idx],
                                     nz[idx],
                                     np.abs(q[idx]))
            for (fz, tz, flux) in fluxes:
                self._update_record('IN', 'CONSTANT HEAD', 'ZONE {}'.format(fz), flux)
        return

    def _accumulate_flow_flf(self, recname, izone, ich, **kwargs):
        """
        C
        C-----"FLOW LOWER FACE"
        C-----CALCULATE FLOW BETWEEN NODE J,I,K AND J,I,K-1
        500   IF(NLAY.LT.2) RETURN
              DO 540 K=2,NLAY
              DO 540 I=1,NROW
              DO 540 J=1,NCOL
              NZ=IZONE(J,I,K)
              KA=K-1
              NZA=IZONE(J,I,KA)
              IF(NZA.LE.NZ) GO TO 540
        C  Don't include CH to CH flow (can occur if CHTOCH option is used)
              IF(ICH(J,I,K).EQ.1 .AND. ICH(J,I,K-1).EQ.1) GO TO 540
              DBUFF=BUFFD(J,I,KA)
              IF(DBUFF.LT.DZERO) THEN
                 VBZNFL(2,NZ,NZA)=VBZNFL(2,NZ,NZA)-DBUFF
              ELSE
                 VBZNFL(1,NZ,NZA)=VBZNFL(1,NZ,NZA)+DBUFF
              END IF
          540 CONTINUE
        C
        C-----CALCULATE FLOW BETWEEN NODE J,I,K AND J,I,K+1
              DO 570 K=1,NLAY-1
              DO 570 I=1,NROW
              DO 570 J=1,NCOL
              NZ=IZONE(J,I,K)
              KB=K+1
              NZB=IZONE(J,I,KB)
              IF(NZB.LE.NZ) GO TO 570
        C  Don't include CH to CH flow (can occur if CHTOCH option is used)
              IF(ICH(J,I,K).EQ.1 .AND. ICH(J,I,K+1).EQ.1) GO TO 570
              DBUFF=BUFFD(J,I,K)
              IF(DBUFF.LT.DZERO) THEN
                 VBZNFL(1,NZ,NZB)=VBZNFL(1,NZ,NZB)-DBUFF
              ELSE
                 VBZNFL(2,NZ,NZB)=VBZNFL(2,NZ,NZB)+DBUFF
              END IF
          570 CONTINUE
        C
        C-----CALCULATE FLOW TO CONSTANT-HEAD CELLS IN THIS DIRECTION
              DO 595 K=1,NLAY
              DO 595 I=1,NROW
              DO 595 J=1,NCOL
              IF(ICH(J,I,K).EQ.0) GO TO 595
              NZ=IZONE(J,I,K)
              IF(NZ.EQ.0) GO TO 595
              IF(K.EQ.NLAY) GO TO 580
              IF(ICH(J,I,K+1).EQ.1) GO TO 580
              DBUFF=BUFFD(J,I,K)
              IF(DBUFF.EQ.DZERO) THEN
              ELSE IF(DBUFF.LT.DZERO) THEN
                 VBVL(2,MSUMCH,NZ)=VBVL(2,MSUMCH,NZ)-DBUFF
              ELSE
                 VBVL(1,MSUMCH,NZ)=VBVL(1,MSUMCH,NZ)+DBUFF
              END IF
        580   IF(K.EQ.1) GO TO 595
              IF(ICH(J,I,K-1).EQ.1) GO TO 595
              DBUFF=BUFFD(J,I,K-1)
              IF(DBUFF.EQ.DZERO) THEN
              ELSE IF(DBUFF.LT.DZERO) THEN
                 VBVL(1,MSUMCH,NZ)=VBVL(1,MSUMCH,NZ)-DBUFF
              ELSE
                 VBVL(2,MSUMCH,NZ)=VBVL(2,MSUMCH,NZ)+DBUFF
              END IF
        595   CONTINUE
              RETURN
        """
        if self.nlay >= 2:
            data = self.cbc.get_data(text=recname, **kwargs)[0]

            # "FLOW LOWER FACE"
            # CALCULATE FLOW BETWEEN NODE J,I,K AND J,I,K-1
            l, r, c = np.where(izone[1:, :, :] > izone[:-1, :, :])
            l = np.copy(l) + 1
            la = l - 1
            nza = izone[la, r, c]
            nz = izone[l, r, c]
            q = data[la, r, c]
            idx = np.where((q > 0) & ((ich[l, r, c] != 1) | (ich[la, r, c] != 1)))
            fluxes = sum_flux_tuples(nza[idx],
                                     nz[idx],
                                     np.abs(q[idx]))
            for (fz, tz, flux) in fluxes:
                print(1, fz, tz, flux)
                self._update_record('IN', 'FROM ZONE {}'.format(fz), 'ZONE {}'.format(tz), flux)
                self._update_record('OUT', 'TO ZONE {}'.format(tz), 'ZONE {}'.format(fz), flux)
            idx = np.where((q < 0) & ((ich[l, r, c] != 1) | (ich[la, r, c] != 1)))
            fluxes = sum_flux_tuples(nz[idx],
                                     nza[idx],
                                     np.abs(q[idx]))
            for (fz, tz, flux) in fluxes:
                print(2, fz, tz, flux)
                self._update_record('IN', 'FROM ZONE {}'.format(fz), 'ZONE {}'.format(tz), flux)
                self._update_record('OUT', 'TO ZONE {}'.format(tz), 'ZONE {}'.format(fz), flux)

            # CALCULATE FLOW BETWEEN NODE J,I,K AND J,I,K+1
            l, r, c = np.where(izone[:-1, :, :] < izone[1:, :, :])
            nz = izone[l, r, c]
            lb = l + 1
            nzb = izone[lb, r, c]
            q = data[l, r, c]
            idx = np.where((q > 0) & ((ich[l, r, c] != 1) | (ich[lb, r, c] != 1)))
            fluxes = sum_flux_tuples(nz[idx],
                                     nzb[idx],
                                     np.abs(q[idx]))
            for (fz, tz, flux) in fluxes:
                print(3, fz, tz, flux)
                self._update_record('IN', 'FROM ZONE {}'.format(fz), 'ZONE {}'.format(tz), flux)
                self._update_record('OUT', 'TO ZONE {}'.format(tz), 'ZONE {}'.format(fz), flux)
            idx = np.where((q < 0) & ((ich[l, r, c] != 1) | (ich[lb, r, c] != 1)))
            fluxes = sum_flux_tuples(nzb[idx],
                                     nz[idx],
                                     np.abs(q[idx]))
            for (fz, tz, flux) in fluxes:
                print(4, fz, tz, flux)
                self._update_record('IN', 'FROM ZONE {}'.format(fz), 'ZONE {}'.format(tz), flux)
                self._update_record('OUT', 'TO ZONE {}'.format(tz), 'ZONE {}'.format(fz), flux)

            # CALCULATE FLOW TO CONSTANT-HEAD CELLS IN THIS DIRECTION
            l, r, c = np.where(ich == 1)
            l, r, c = l[l > 0], r[l > 0], c[l > 0]
            la = l - 1
            nza = izone[la, r, c]
            nz = izone[l, r, c]
            q = data[la, r, c]
            idx = np.where((q > 0) & ((ich[l, r, c] != 1) | (ich[la, r, c] != 1)))
            fluxes = sum_flux_tuples(nza[idx],
                                     nz[idx],
                                     np.abs(q[idx]))
            for (fz, tz, flux) in fluxes:
                self._update_record('OUT', 'CONSTANT HEAD', 'ZONE {}'.format(tz), flux)
            idx = np.where((q < 0) & ((ich[l, r, c] != 1) | (ich[la, r, c] != 1)))
            fluxes = sum_flux_tuples(nz[idx],
                                     nza[idx],
                                     np.abs(q[idx]))
            for (fz, tz, flux) in fluxes:
                self._update_record('IN', 'CONSTANT HEAD', 'ZONE {}'.format(fz), flux)
            l, r, c = np.where(ich == 1)
            l, r, c = l[l < self.nlay - 1], r[l < self.nlay - 1], c[l < self.nlay - 1]
            nz = izone[l, r, c]
            lb = l + 1
            nzb = izone[lb, r, c]
            q = data[l, r, c]
            idx = np.where((q > 0) & ((ich[l, r, c] != 1) | (ich[lb, r, c] != 1)))
            fluxes = sum_flux_tuples(nz[idx],
                                     nzb[idx],
                                     np.abs(q[idx]))
            for (fz, tz, flux) in fluxes:
                self._update_record('OUT', 'CONSTANT HEAD', 'ZONE {}'.format(tz), flux)
            idx = np.where((q < 0) & ((ich[l, r, c] != 1) | (ich[lb, r, c] != 1)))
            fluxes = sum_flux_tuples(nzb[idx],
                                     nz[idx],
                                     np.abs(q[idx]))
            for (fz, tz, flux) in fluxes:
                self._update_record('IN', 'CONSTANT HEAD', 'ZONE {}'.format(fz), flux)
        return

    def _accumulate_flow_ssst(self, recname, qin, qout, izone, lstzon):
        # Source/sink/storage terms are accumulated by zone
        recin = [np.abs(qin[(izone == z)].sum()) for z in lstzon]
        recout = [np.abs(qout[(izone == z)].sum()) for z in lstzon]
        for idx, flux in enumerate(recin):
            if type(flux) == np.ma.core.MaskedConstant:
                flux = 0.
            self._update_record('IN', recname, 'ZONE {}'.format(lstzon[idx]), flux)
        for idx, flux in enumerate(recout):
            if type(flux) == np.ma.core.MaskedConstant:
                flux = 0.
            self._update_record('OUT', recname, 'ZONE {}'.format(lstzon[idx]), flux)
        return

    def get_kstpkper(self):
        # Courtesy access to the CellBudgetFile method
        return self.cbc.get_kstpkper()

    def get_times(self):
        # Courtesy access to the CellBudgetFile method
        return self.cbc.get_times()

    def get_indices(self):
        # Courtesy access to the CellBudgetFile method
        return self.cbc.get_indices()


def _numpyvoid2numeric(a):
    # The budget record array has multiple dtypes and a slice returns
    # the flexible-type numpy.void which must be converted to a numeric
    # type prior to performing reducing functions such as sum() or
    # mean()
    return np.array([list(r) for r in a])


def arr2ascii(fname, X, width=None):
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
    width : int
        The number of values to write to each line.

    Returns
    -------

    """
    if len(X.shape) == 2:
        nlay = 1
        nrow, ncol = X.shape
        b = np.zeros((nlay, nrow, ncol), dtype=np.int64)
        b[0, :, :] = X[:, :]
        X = b.copy()
    elif len(X.shape) == 3:
        nlay, nrow, ncol = X.shape
    else:
        raise Exception('Shape of the input array is not recognized: {}'.format(a.shape))

    with open(fname, 'w') as f:
        f.write('{nlay} {nrow} {ncol}\n'.format(nlay=nlay,
                                                nrow=nrow,
                                                ncol=ncol))
        for lay in range(nlay):
            if width is not None:
                assert width < ncol, 'The specified width is greater than the ' \
                                     'number of columns in the array.'
                f.write('INTERNAL\t({nvals}I8)\n'.format(nvals=width))
                for row in range(nrow):
                    vals = X[lay, row, :].ravel()
                    i = 1
                    while i <= round(ncol/width):
                        chunk = vals[i * width - 1:i * width - 1 + width]
                        f.write(''.join(['{:>8}'.format(int(val)) for val in chunk]) + '\n')
                        i += 1
                    chunk = vals[i * width - 1:]
                    f.write(''.join(['{:>8}'.format(int(val)) for val in chunk]) + '\n')
            else:
                f.write('INTERNAL\t({nvals}I8)\n'.format(nvals=ncol))
                for row in range(nrow):
                    vals = X[lay, row, :].ravel()
                    f.write(''.join(['{:>8}'.format(int(val)) for val in vals]) + '\n')
    return


def sum_flux_tuples(fromzones, tozones, fluxes):
    tup = zip(fromzones, tozones, fluxes)
    sorted_tups = sort_tuple(tup)

    # Group the sorted tuples by (from zone, to zone)
    # itertools.groupby() returns the index (from zone, to zone) and
    # a list of the tuples with that index
    fluxes = []
    for (fz, tz), ftup in groupby(sorted_tups, lambda tup: tup[:2]):
        f = np.sum([tup[-1] for tup in list(ftup)])
        fluxes.append((fz, tz, f))
    return fluxes


def sort_tuple(tup, n=2):
    """
    Sort a tuple by the first n values
    :param tup:
    :param n:
    :return:
    """
    return tuple(sorted(tup, key=lambda t: t[:n]))
