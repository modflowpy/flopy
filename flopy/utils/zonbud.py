from __future__ import print_function
import os
import sys
import numpy as np
import subprocess as sp
import warnings
import threading
if sys.version_info > (3,0):
    import queue as Queue
else:
    import Queue
from datetime import datetime
from itertools import groupby
from collections import OrderedDict
from .binaryfile import CellBudgetFile


class Budget(object):
    """
    ZoneBudget Budget class. This is a wrapper around a numpy record array to allow users
    to save the record array to a formatted csv file.
    """
    def __init__(self, records, kstpkper=None, totim=None):
        self.records = records
        self.kstpkper = kstpkper
        self.totim = totim
        assert(self.kstpkper is not None or self.totim is not None), 'Budget object requires either kstpkper ' \
                                                                     'or totim be be specified.'

        # List the field names to be used to slice the recarray
        # fields = ['ZONE{: >4d}'.format(z) for z in self.zones]
        fields = [name for name in self.records.dtype.names if 'ZONE' in name]

        self.ins_idx = np.where(self.records['flow_dir'] == 'in')[0]
        self.out_idx = np.where(self.records['flow_dir'] == 'out')[0]

        ins = self._fields_view(self.records[self.ins_idx], fields)
        out = self._fields_view(self.records[self.out_idx], fields)

        self.ins_sum = ins.sum(axis=0)
        self.out_sum = out.sum(axis=0)

        self.ins_minus_out = self.ins_sum - self.out_sum
        self.ins_plus_out = self.ins_sum + self.out_sum

        pcterr = 100 * (self.ins_minus_out) / ((self.ins_plus_out) / 2.)
        self.pcterr = [i if not np.isnan(i) else 0 for i in pcterr]

    def get_total_inflow(self):
        return self.ins_sum

    def get_total_outflow(self):
        return self.out_sum

    def get_percent_error(self):
        return self.pcterr

    def to_csv(self, fname='zbud.csv', write_format='pandas', formatter=None):
        """
        Saves the Budget object record array to a formatted csv file.

        Parameters
        ----------
        fname
        write_format
        formatter

        Returns
        -------

        """
        assert write_format.lower() in ['pandas', 'zonbud'], 'Format must be one of "pandas" or "zonbud".'

        if formatter is None:
            formatter = '{:.16e}'.format

        if write_format.lower() == 'pandas':
            with open(fname, 'w') as f:

                # Write header
                f.write(','.join(self.records.dtype.names)+'\n')

                # Write IN terms
                for rec in self.records[self.ins_idx]:
                    items = []
                    for i in rec:
                        if isinstance(i, str):
                            items.append(i)
                        elif isinstance(i, float):
                            items.append(formatter(i))
                    f.write(','.join(items)+'\n')
                f.write(','.join([' ', 'Total IN'] + [formatter(i) for i in self.ins_sum])+'\n')

                # Write OUT terms
                for rec in self.records[self.out_idx]:
                    items = []
                    for i in rec:
                        if isinstance(i, str):
                            items.append(i)
                        elif isinstance(i, float):
                            items.append(formatter(i))
                    f.write(','.join(items) + '\n')
                f.write(','.join([' ', 'Total OUT'] + [formatter(i) for i in self.out_sum])+'\n')

                # Write mass balance terms
                f.write(','.join([' ', 'IN-OUT'] + [formatter(i) for i in self.ins_minus_out])+'\n')
                f.write(','.join([' ', 'Percent Error'] + [formatter(i) for i in self.pcterr])+'\n')

        elif write_format.lower() == 'zonbud':
            with open(fname, 'w') as f:

                # Write header
                if self.kstpkper is not None:
                    header = 'Time Step, {kstp}, Stress Period, {kper}\n'.format(kstp=self.kstpkper[0]+1,
                                                                                 kper=self.kstpkper[1]+1)

                elif self.totim is not None:
                    header = 'Sim. Time, {totim}\n'.format(totim=self.totim)

                f.write(header)
                f.write(','.join([' '] + [field for field in self.records.dtype.names[2:]])+'\n')

                # Write IN terms
                f.write(','.join([' '] + ['IN']*(len(self.records.dtype.names[1:])-1))+'\n')
                for rec in self.records[self.ins_idx]:
                    items = []
                    for i in rec:
                        if isinstance(i, str):
                            items.append(i)
                        elif isinstance(i, float):
                            items.append(formatter(i))
                    f.write(','.join(items)+'\n')
                f.write(','.join(['Total IN'] + [formatter(i) for i in self.ins_sum])+'\n')

                # Write OUT terms
                f.write(','.join([' '] + ['OUT']*(len(self.records.dtype.names[1:])-1))+'\n')
                for rec in self.records[self.out_idx]:
                    items = []
                    for i in rec:
                        if isinstance(i, str):
                            items.append(i)
                        elif isinstance(i, float):
                            items.append(formatter(i))
                    f.write(','.join(items) + '\n')
                f.write(','.join(['Total OUT'] + [formatter(i) for i in self.out_sum])+'\n')

                # Write mass balance terms
                f.write(','.join(['IN-OUT'] + [formatter(i) for i in self.ins_minus_out])+'\n')
                f.write(','.join(['Percent Error'] + [formatter(i) for i in self.pcterr])+'\n')

    def _fields_view(self, a, fields):
        new = a[fields].view(np.float32).reshape(a.shape + (-1,))
        return new


class ZoneBudget(object):
    """
    ZoneBudget class

    Example usage:

    >>>from flopy.utils import ZoneBudget
    >>>zb = ZoneBudget('zonebudtest.cbc')
    >>>zbud = zb.get_budget('GWBasins.zon')
    >>>zbud.to_csv()
    """
    def __init__(self, cbc_file):

        # INTERNAL FLOW TERMS ARE USED TO CALCULATE FLOW BETWEEN ZONES.
        # CONSTANT-HEAD TERMS ARE USED TO IDENTIFY WHERE CONSTANT-HEAD CELLS ARE AND THEN USE
        # FACE FLOWS TO DETERMINE THE AMOUNT OF FLOW.
        # SWIADDTO* terms are used by the SWI2 package.
        internal_flow_terms = ['FLOW RIGHT FACE', 'FLOW FRONT FACE', 'FLOW LOWER FACE',
                               'SWIADDTOFRF', 'SWIADDTOFFF', 'SWIADDTOFLF']

        # OPEN THE CELL-BY-CELL BUDGET BINARY FILE
        self.cbc = CellBudgetFile(cbc_file)

        # All record names in the cell-by-cell budget binary file
        self.record_names = self.cbc.unique_record_names()

        # Check for SWI budget terms
        self.is_swi = False
        for recname in self.record_names:
            if 'SWI' in recname:
                self.is_swi = True
                break

        # Source/sink/storage term record names
        self.ssst_record_names = [rec.strip() for rec in self.record_names if rec.strip() not in internal_flow_terms]

        # Face flow record names
        self.ff_record_names = [r.strip() for r in self.record_names if r.strip() in internal_flow_terms]

        # Check the shape of the cbc budget file arrays
        self.cbc_shape = self.get_model_shape()
        self.nlay, self.nrow, self.ncol = self.cbc_shape

        self.float_type = np.float32

    def get_model_shape(self):
        l, r, c = self.cbc.get_data(idx=0, full3D=True)[0].shape
        return l, r, c

    def get_budget(self, zon, kstpkper=None, totim=None, **kwargs):
        """

        Parameters
        ----------
        zon
        kstpkper
        totim
        kwargs

        Returns
        -------
        Budget object
        """
        assert kstpkper in self.get_kstpkper(), 'The specified time step/stress period' \
                                                ' does not exist {}'.format(kstpkper)

        # Get budget data from cell budget file
        if kstpkper is not None:
            self.cbc_data = OrderedDict([(recname.strip(), self.cbc.get_data(text=recname, kstpkper=kstpkper)[0])
                                         for recname in self.record_names if recname.strip()])
        elif totim is not None:
            self.cbc_data = OrderedDict([(recname.strip(), self.cbc.get_data(text=recname, totim=totim)[0])
                                         for recname in self.record_names if recname.strip()])
        else:
            print('Reading budget for last timestep/stress period.')
            kstpkper = self.cbc.get_kstpkper()[-1]
            self.cbc_data = OrderedDict([(recname.strip(), self.cbc.get_data(text=recname, kstpkper=kstpkper)[0])
                                         for recname in self.record_names if recname.strip()])

        # OPEN THE ZONE FILE (OR ARRAY) AND FIND THE UNIQUE SET OF ZONES CONTAINED THEREIN
        izone = np.zeros(self.cbc_shape, np.int32)
        if isinstance(zon, str):
            if os.path.isfile(zon):
                if 'skiprows' in kwargs:
                    skiprows = kwargs.pop('skiprows')
                else:
                    skiprows = None
                try:
                    z = np.loadtxt(zon, dtype=np.int32, skiprows=skiprows)
                except Exception as e:
                    print(e)
        elif isinstance(zon, np.ndarray):
            z = zon
        else:
            s = 'Input zones format is not recognized.'
            raise Exception(s)

        # Make sure the input zone array has the same shape as the cell budget file
        if len(z.shape) == 2:
            for i in range(izone.shape[0]):
                izone[i, :, :] = z
        else:
            izone = z.copy()

        zones = self._find_unique_zones(izone.ravel())

        assert izone.shape == self.cbc_shape, \
            'Shape of input zone array {} does not' \
            ' match the cell by cell' \
            ' budget file {}'.format(izone.shape, self.cbc_shape)

        # CONSTANT-HEAD FLOW -- DON'T ACCUMULATE THE CELL-BY-CELL VALUES FOR
        # CONSTANT-HEAD FLOW BECAUSE THEY MAY INCLUDE PARTIALLY CANCELING
        # INS AND OUTS.  USE CONSTANT-HEAD TERM TO IDENTIFY WHERE CONSTANT-
        # HEAD CELLS ARE AND THEN USE FACE FLOWS TO DETERMINE THE AMOUNT OF
        # FLOW.  STORE CONSTANT-HEAD LOCATIONS IN ICH ARRAY.
        ich = np.ma.zeros(self.cbc_shape, np.int32)
        ich.mask = True
        # chd = self.cbc_data['CONSTANT HEAD']
        chd = self.cbc.get_data(text='CONSTANT HEAD', kstpkper=kstpkper, full3D=True)[0]
        for l, r, c in zip(*np.where(chd != 0.)):
            ich[l, r, c] = 1
            ich.mask[l, r, c] = False

        # TEMPORARY WARNINGS
        if ich.count() > 0 and self.is_swi:
            chwarn = 'Budget information for CONSTANT HEAD cells is not yet supported' \
                     'for SWI2 models. Any non-zero results for CONSTANT HEAD and' \
                     'SWIADDTOCH should be considered erroneous.'
            warnings.warn(chwarn, UserWarning)
        # /TEMPORARY WARNINGS

        # Create containers for budget term tuples
        inflows = []
        outflows = []

        # ACCUMULATE SOURCE/SINK/STORAGE TERMS
        for recname in self.ssst_record_names:

            if recname == 'RECHARGE':
                data = self.cbc.get_data(text=recname, kstpkper=kstpkper, full3D=True)[0]
                budin = np.ma.zeros(self.cbc_shape)
                budout = np.ma.zeros(self.cbc_shape)
                budin[data > 0] = data[data > 0]
                budout[data < 0] = data[data < 0]
                in_tup, out_tup = self._get_source_sink_storage_terms_tuple(recname, budin, budout, zones, izone)
                inflows.append(in_tup)
                outflows.append(out_tup)

            elif recname == 'CONSTANT HEAD':
                inflow, outflow = self._get_constant_head_flow_term_tuple(zones, izone, ich)
                inflows.append(tuple(['in', recname] + [val for val in inflow]))
                outflows.append(tuple(['out', recname] + [val for val in outflow]))

            elif recname == 'SWIADDTOCH':
                inflow, outflow = self._get_constant_head_flow_term_tuple(zones, izone, ich)
                inflows.append(tuple(['in', recname] + [val for val in inflow]))
                outflows.append(tuple(['out', recname] + [val for val in outflow]))

            else:
                data = self.cbc_data[recname]
                budin = np.ma.zeros((self.nlay*self.nrow*self.ncol), self.float_type)
                budout = np.ma.zeros((self.nlay*self.nrow*self.ncol), self.float_type)
                for [node, q] in zip(data['node'], data['q']):
                    idx = node - 1
                    if q > 0:
                        budin.data[idx] += q
                    elif q < 0:
                        budout.data[idx] += q
                budin = np.ma.reshape(budin, (self.nlay, self.nrow, self.ncol))
                budout = np.ma.reshape(budout, (self.nlay, self.nrow, self.ncol))
                in_tup, out_tup = self._get_source_sink_storage_terms_tuple(recname, budin, budout, zones, izone)
                inflows.append(in_tup)
                outflows.append(out_tup)

        # ACCUMULATE INTERNAL FLOW TERMS
        # Each flow term is a tuple of (from zone, to zone, flux)
        frf, fff, flf, swifrf, swifff, swiflf = self.get_internal_flow_terms(izone, ich)

        # Combine and sort flux tuples
        q_tups = sorted(frf + fff + flf + swifrf + swifff + swiflf)

        # Set starting values--INFLOW
        q_in = {z: OrderedDict([('flow_dir', 'in'),
                                ('record', 'FROM ZONE {:d}'.format(z))])
                for z in zones}
        for od in q_in.values():
            for zone in zones:
                od[zone] = 0.

        # Set starting values--OUTFLOW
        q_out = {z: OrderedDict([('flow_dir', 'out'),
                                 ('record', 'TO ZONE {:d}'.format(z))])
                 for z in zones}
        for od in q_out.values():
            for zone in zones:
                od[zone] = 0.

        # Group the flux tuples by from/to zone then assign the flux
        for (from_zone, to_zone), flux_tups in groupby(q_tups, lambda tup: tup[:2]):
            val = np.sum([tup[-1] for tup in list(flux_tups)])
            q_in[from_zone][to_zone] = val
            q_out[to_zone][from_zone] = val

        # Pull the from/to zone/flux tuples back out and append them to a list
        for v in q_in.values():
            inflows.append(tuple(v.values()))
        for v in q_out.values():
            outflows.append(tuple(v.values()))

        q = inflows + outflows

        # Define dtype for the recarray
        dtype = np.dtype([('flow_dir', '|S3'),
                          ('record', '|S20')] +
                         [('ZONE {:d}'.format(z), self.float_type) for z in zones])

        q = Budget(np.array(q, dtype=dtype), kstpkper=kstpkper, totim=totim)
        return q

    def get_internal_flow_terms(self, izone, ich):

        # PROCESS EACH INTERNAL FLOW RECORD IN THE CELL-BY-CELL BUDGET FILE
        frf, fff, flf, swifrf, swifff, swiflf = [], [], [], [], [], []
        for recname in self.ff_record_names:
            if recname.strip() == 'FLOW RIGHT FACE':
                if self.ncol >= 2:
                    bud = self.cbc_data[recname]
                    frf = self._get_internal_flow_terms_tuple_frf(bud, izone, ich)
            elif recname.strip() == 'FLOW FRONT FACE':
                if self.nrow >= 2:
                    bud = self.cbc_data[recname]
                    fff = self._get_internal_flow_terms_tuple_fff(bud, izone, ich)
            elif recname.strip() == 'FLOW LOWER FACE':
                if self.nlay >= 2:
                    bud = self.cbc_data[recname]
                    flf = self._get_internal_flow_terms_tuple_flf(bud, izone, ich)
            elif recname.strip() == 'SWIADDTOFRF':
                if self.ncol >= 2:
                    bud = self.cbc_data[recname]
                    swifrf = self._get_internal_flow_terms_tuple_frf(bud, izone, ich)
            elif recname.strip() == 'SWIADDTOFFF':
                if self.nrow >= 2:
                    bud = self.cbc_data[recname]
                    swifff = self._get_internal_flow_terms_tuple_fff(bud, izone, ich)
            elif recname.strip() == 'SWIADDTOFLF':
                if self.nlay >= 2:
                    bud = self.cbc_data[recname]
                    swiflf = self._get_internal_flow_terms_tuple_flf(bud, izone, ich)
        return frf, fff, flf, swifrf, swifff, swiflf

    def get_kstpkper(self):
        return self.cbc.get_kstpkper()

    def get_times(self):
        return self.cbc.get_times()

    def get_indices(self):
        return self.cbc.get_indices()

    def get_ssst_names(self):
        return self.ssst_record_names

    def get_ssst_cbc_array(self, recname, kstpkper=None, totim=None):
        try:
            recname = [n for n in self.ssst_record_names if n.strip() == recname][0]
        except IndexError:
            s = 'Please enter a valid source/sink/storage record name. Use ' \
                                                   'get_ssst_names() to view a list.'
            print(s)
            return
        if kstpkper is not None:
            a = self.cbc.get_data(text=recname, kstpkper=kstpkper, full3D=True)[0]
        elif totim is not None:
            a = self.cbc.get_data(text=recname, totim=totim, full3D=True)[0]
        else:
            print('Reading budget for last timestep/stress period.')
            kstpkper = self.cbc.get_kstpkper()[-1]
            a = self.cbc.get_data(text=recname, kstpkper=kstpkper, full3D=True)[0]
        return a

    def _fields_view(self, a, fields):
        new = a[fields].view(self.float_type).reshape(a.shape + (-1,))
        return new

    @staticmethod
    def _get_source_sink_storage_terms_tuple(recname, budin, budout, zones, izone):
        recin = ['in', recname.strip()] + [np.abs(budin[(izone == z)].sum()) for z in zones]
        recout = ['out', recname.strip()] + [np.abs(budout[(izone == z)].sum()) for z in zones]
        recin = [val if not type(val) == np.ma.core.MaskedConstant else 0. for val in recin]
        recout = [val if not type(val) == np.ma.core.MaskedConstant else 0. for val in recout]
        return tuple(recin), tuple(recout)

    @staticmethod
    def _get_internal_flow_terms_tuple_frf(bud, izone, ich):
        # ACCUMULATE FLOW BETWEEN ZONES ACROSS COLUMNS. COMPUTE FLOW ONLY BETWEEN A ZONE
        # AND A HIGHER ZONE -- FLOW FROM ZONE 4 TO 3 IS THE NEGATIVE OF FLOW FROM 3 TO 4.
        # FIRST, CALCULATE FLOW BETWEEN NODE J,I,K AND J-1,I,K.
        # Accumulate flow from lower zones to higher zones from "left" to "right".
        # Flow into the higher zone will be <0 Flow Right Face from the adjacent cell to the "left".
        # Returns a tuple of "to zone", "from zone", and the flux
        nz = izone[:, :, 1:]
        nzl = izone[:, :, :-1]
        l, r, c = np.where(nz > nzl)

        # Adjust column values to account for the starting position of "nz"
        c += 1

        # Define the zone from which flow is coming
        from_zones = izone[l, r, c-1]

        # Define the zone to which flow is going
        to_zones = izone[l, r, c]

        # Get the face flow
        q = bud[l, r, c-1]

        # Don't include CH to CH flow (can occur if CHTOCH option is used)
        q[(ich[l, r, c] == 1) & (ich[l, r, c-1] == 1)] = 0.

        # Get indices where flow face values are negative (flow into higher zone)
        idx_neg = np.where(q < 0)

        # Get indices where flow face values are positive (flow out of higher zone)
        idx_pos = np.where(q >= 0)

        neg = zip(to_zones[idx_neg], from_zones[idx_neg], np.abs(q[idx_neg]))
        pos = zip(from_zones[idx_pos], to_zones[idx_pos], np.abs(q[idx_pos]))
        nzgt_l2r = neg + pos

        # CALCULATE FLOW BETWEEN NODE J,I,K AND J+1,I,K.
        # Accumulate flow from lower zones to higher zones from "right" to "left".
        # Flow into the higher zone will be <0 Flow Right Face from the adjacent cell to the "left".
        nz = izone[:, :, :-1]
        nzr = izone[:, :, 1:]
        l, r, c = np.where(nz > nzr)

        # Define the zone from which flow is coming
        from_zones = izone[l, r, c]

        # Define the zone to which flow is going
        to_zones = izone[l, r, c+1]

        # Get the face flow
        q = bud[l, r, c]

        # Don't include CH to CH flow (can occur if CHTOCH option is used)
        q[(ich[l, r, c] == 1) & (ich[l, r, c-1] == 1)] = 0.

        # Get indices where flow face values are negative (flow into higher zone)
        idx_neg = np.where(q < 0)

        # Get indices where flow face values are positive (flow out of higher zone)
        idx_pos = np.where(q >= 0)

        neg = zip(to_zones[idx_neg], from_zones[idx_neg], np.abs(q[idx_neg]))
        pos = zip(from_zones[idx_pos], to_zones[idx_pos], np.abs(q[idx_pos]))
        nzgt_r2l = neg + pos

        # Sum all flow face and constant head terms
        nzgt = sorted(nzgt_l2r + nzgt_r2l, key=lambda tup: tup[:2])
        return nzgt

    @staticmethod
    def _get_internal_flow_terms_tuple_fff(bud, izone, ich):
        # ACCUMULATE FLOW BETWEEN ZONES ACROSS ROWS. COMPUTE FLOW ONLY BETWEEN A ZONE
        #  AND A HIGHER ZONE -- FLOW FROM ZONE 4 TO 3 IS THE NEGATIVE OF FLOW FROM 3 TO 4.
        # FIRST, CALCULATE FLOW BETWEEN NODE J,I,K AND J,I-1,K.
        # Accumulate flow from lower zones to higher zones from "up" to "down".
        nz = izone[:, 1:, :]
        nzu = izone[:, :-1, :]
        l, r, c = np.where(nz < nzu)
        # Adjust column values by +1 to account for the starting position of "nz"
        r += 1

        # Define the zone from which flow is coming
        from_zones = izone[l, r-1, c]

        # Define the zone to which flow is going
        to_zones = izone[l, r, c]

        # Get the face flow
        q = bud[l, r-1, c]

        # Don't include CH to CH flow (can occur if CHTOCH option is used)
        q[(ich[l, r, c] == 1) & (ich[l, r-1, c] == 1)] = 0.

        # Get indices where flow face values are negative (flow into higher zone)
        idx_neg = np.where(q < 0)

        # Get indices where flow face values are positive (flow out of higher zone)
        idx_pos = np.where(q >= 0)

        neg = zip(to_zones[idx_neg], from_zones[idx_neg], np.abs(q[idx_neg]))
        pos = zip(from_zones[idx_pos], to_zones[idx_pos], np.abs(q[idx_pos]))
        nzgt_u2d = neg + pos

        # CALCULATE FLOW BETWEEN NODE J,I,K AND J,I+1,K.
        # Accumulate flow from lower zones to higher zones from "down" to "up".
        nz = izone[:, :-1, :]
        nzd = izone[:, 1:, :]
        l, r, c = np.where(nz < nzd)

        # Define the zone from which flow is coming
        from_zones = izone[l, r, c]

        # Define the zone to which flow is going
        to_zones = izone[l, r+1, c]

        # Get the face flow
        q = bud[l, r, c]

        # Don't include CH to CH flow (can occur if CHTOCH option is used)
        q[(ich[l, r, c] == 1) & (ich[l, r-1, c] == 1)] = 0.

        # Get indices where flow face values are negative (flow into higher zone)
        idx_neg = np.where(q < 0)

        # Get indices where flow face values are positive (flow out of higher zone)
        idx_pos = np.where(q >= 0)

        neg = zip(to_zones[idx_neg], from_zones[idx_neg], np.abs(q[idx_neg]))
        pos = zip(from_zones[idx_pos], to_zones[idx_pos], np.abs(q[idx_pos]))
        nzgt_d2u = neg + pos

        # Returns a tuple of "to zone", "from zone", and the flux
        nzgt = sorted(nzgt_u2d + nzgt_d2u, key=lambda tup: tup[:2])
        return nzgt

    @staticmethod
    def _get_internal_flow_terms_tuple_flf(bud, izone, ich):
        # ACCUMULATE FLOW BETWEEN ZONES ACROSS LAYERS. COMPUTE FLOW ONLY BETWEEN A ZONE
        #  AND A HIGHER ZONE -- FLOW FROM ZONE 4 TO 3 IS THE NEGATIVE OF FLOW FROM 3 TO 4.
        # FIRST, CALCULATE FLOW BETWEEN NODE J,I,K AND J,I,K-1.
        # Accumulate flow from lower zones to higher zones from "top" to "bottom".
        nz = izone[1:, :, :]
        nzt = izone[:-1, :, :]
        l, r, c = np.where(nz > nzt)
        # Adjust column values by +1 to account for the starting position of "nz"
        l += 1

        # Define the zone from which flow is coming
        from_zones = izone[l-1, r, c]

        # Define the zone to which flow is going
        to_zones = izone[l, r, c]

        # Get the face flow
        q = bud[l-1, r, c]

        # Don't include CH to CH flow (can occur if CHTOCH option is used)
        q[(ich[l, r, c] == 1) & (ich[l-1, r, c] == 1)] = 0.

        # Get indices where flow face values are negative (flow into higher zone)
        idx_neg = np.where(q < 0)

        # Get indices where flow face values are positive (flow out of higher zone)
        idx_pos = np.where(q >= 0)

        neg = zip(to_zones[idx_neg], from_zones[idx_neg], np.abs(q[idx_neg]))
        pos = zip(from_zones[idx_pos], to_zones[idx_pos], np.abs(q[idx_pos]))
        nzgt_t2b = neg + pos

        # CALCULATE FLOW BETWEEN NODE J,I,K AND J+1,I,K.
        # Accumulate flow from lower zones to higher zones from "right" to "left".
        # Flow into the higher zone will be <0 Flow Right Face from the adjacent cell to the "left".
        nz = izone[:-1, :, :]
        nzb = izone[1:, :, :]
        l, r, c = np.where(nz < nzb)

        # Define the zone from which flow is coming
        from_zones = izone[l, r, c]

        # Define the zone to which flow is going
        to_zones = izone[l+1, r, c]

        # Get the face flow
        q = bud[l, r, c]

        # Don't include CH to CH flow (can occur if CHTOCH option is used)
        q[(ich[l, r, c] == 1) & (ich[l-1, r, c] == 1)] = 0.

        # Get indices where flow face values are negative (flow into higher zone)
        idx_neg = np.where(q < 0)

        # Get indices where flow face values are positive (flow out of higher zone)
        idx_pos = np.where(q >= 0)

        neg = zip(to_zones[idx_neg], from_zones[idx_neg], np.abs(q[idx_neg]))
        pos = zip(from_zones[idx_pos], to_zones[idx_pos], np.abs(q[idx_pos]))
        nzgt_b2t = neg + pos

        # Returns a tuple of "to zone", "from zone", and the flux
        nzgt = sorted(nzgt_t2b + nzgt_b2t, key=lambda tup: tup[:2])
        return nzgt

    def _get_constant_head_flow_term_tuple(self, zones, izone, ich):

        q_chd_in = np.zeros(self.cbc_shape, dtype=np.float32)
        q_chd_out = np.zeros(self.cbc_shape, dtype=np.float32)

        # CALCULATE FLOW TO CONSTANT-HEAD CELLS--FLOW RIGHT FACE
        if self.ncol > 1:
            q = self.cbc_data['FLOW RIGHT FACE']
            # q_chd = np.zeros_like(q)
            # q_chd[(ich != 0)] = q[(ich != 0)]
            # q_chd_in[q_chd > 0] += q_chd[q_chd > 0]
            # q_chd_out[q_chd < 0] += q_chd[q_chd < 0]
            q_chd_in[(ich == 1) & (q > 0)] += q[(ich == 1) & (q > 0)]
            q_chd_out[(ich == 1) & (q < 0)] += q[(ich == 1) & (q < 0)]

        # CALCULATE FLOW TO CONSTANT-HEAD CELLS--FLOW FRONT FACE
        if self.nrow > 1:
            q = self.cbc_data['FLOW FRONT FACE']
            # q_chd = np.zeros_like(q)
            # q_chd[(ich != 0)] = q[(ich != 0)]
            # q_chd_in[q_chd > 0] += q_chd[q_chd > 0]
            # q_chd_out[q_chd < 0] += q_chd[q_chd < 0]
            q_chd_in[(ich == 1) & (q > 0)] += q[(ich == 1) & (q > 0)]
            q_chd_out[(ich == 1) & (q < 0)] += q[(ich == 1) & (q < 0)]

        # CALCULATE FLOW TO CONSTANT-HEAD CELLS--FLOW LOWER FACE
        if self.nlay > 1:
            q = self.cbc_data['FLOW LOWER FACE']
            # q_chd = np.zeros_like(q)
            # q_chd[(ich != 0)] = q[(ich != 0)]
            # q_chd_in[q_chd > 0] += q_chd[q_chd > 0]
            # q_chd_out[q_chd < 0] += q_chd[q_chd < 0]
            q_chd_in[(ich == 1) & (q > 0)] += q[(ich == 1) & (q > 0)]
            q_chd_out[(ich == 1) & (q < 0)] += q[(ich == 1) & (q < 0)]

        chd_inflow = [np.abs(q_chd_in[(izone == z)].sum()) for z in zones]
        chd_outflow = [np.abs(q_chd_out[(izone == z)].sum()) for z in zones]
        chd_inflow = [val if not type(val) == np.ma.core.MaskedConstant else 0. for val in chd_inflow]
        chd_outflow = [val if not type(val) == np.ma.core.MaskedConstant else 0. for val in chd_outflow]
        return tuple(chd_inflow), tuple(chd_outflow)

    @staticmethod
    def _find_unique_zones(a):
        z = [int(i) for i in np.unique(a)]
        return z


# def run_zonbud(zonarray, cbcfile='modflowtest.cbc', listingfile_prefix='zbud', zonbud_ws='.',
#                zonbud_exe='zonbud.exe', title='ZoneBudget Test', budget_option='A', kstpkper=None,
#                iprn=-1, silent=True):
#     """
#
#     Parameters
#     ----------
#     zonarray : array of ints (nlay, nrow, ncol)
#         integer-array of zone numbers
#     cbcfile : str
#         name of the cell-by-cell budget file
#     listingfile_prefix : str
#         name of the listingfile
#     zonbud_ws : str
#         directory where ZoneBudget output will be stored
#     zonbud_exe : str
#         name of the ZoneBudget executable
#     title : str
#         title to be printed in the listing file
#     budget_option : str
#         must be one of "A" (for all timesteps) or "L" for a user-specified list of timesteps
#     kstpkper : list
#         list of zero-based timestep/stress periods for which budgets will be calculated
#     iprn : integer
#         specifies whether or not the zone values are printed in the output file
#         if less than zero, zone values will not be printed
#
#     Returns
#     -------
#     zbud, an ordered dictionary of recarrays.
#
#     Examples
#     -------
#     >>> import flopy
#     >>> zbud = flopy.utils.run_zonbud(zonarray, cbcfile='modflowtest.cbc')
#     """
#
#     # Need to catch some errors early on, ZoneBudget likes to crash without any feedback
#     # Locked output files, non-existent input, etc.
#     #
#     if not os.path.isfile(cbcfile):
#         s = 'The specified cell by cell budget file does not exist: {}'.format(cbcfile)
#         raise Exception(s)
#     assert budget_option.upper() in ['A', 'L'], 'Please enter a valid budget option ("A" for all or "L" for' \
#                                                 ' a list of times).'
#     listingfile_prefix = listingfile_prefix.split('.')[0]
#     zonfile = os.path.join(zonbud_ws, listingfile_prefix + '.zon')
#     listingfile = os.path.join(zonbud_ws, listingfile_prefix + ' csv')
#     outfile = os.path.join(zonbud_ws, listingfile_prefix + '.csv')
#     args = [listingfile, cbcfile, title, zonfile, budget_option]
#     if budget_option == 'L':
#         assert kstpkper is not None, 'You have chosen budget option "L", please enter a ' \
#                                      'list of times. For example (0, 0) for timestep 1 ' \
#                                      'of stress period 1.'
#         kstpkper_args = ['{kstp},{kper}'.format(kstp=kk[0]+1, kper=kk[1]+1) for kk in kstpkper]
#         kstpkper_args.append('0,0')
#         args += kstpkper_args
#
#     try:
#         with open(outfile, 'w') as f:
#             pass
#     except IOError:
#         s = 'Output file is not writable. Please check to make sure you have access to the file location ' \
#             'and that the file is not currently locked by another process.'
#         raise Exception(s)
#
#     _write_zonfile(zonarray, zonfile, iprn)
#     _call(zonbud_exe, args, zonbud_ws, silent)
#     zbud = _parse_zbud_file(outfile)
#     return zbud
#
#
# def _parse_zbud_file(zf):
#     assert os.path.isfile(zf), 'Output zonebudget file {} does not exist or cannot be read.'.format(zf)
#     kstpkper = []
#     ins = OrderedDict()
#     outs = OrderedDict()
#     ins_flag = False
#     outs_flag = False
#     with open(zf) as f:
#         for line in f:
#             line_items = [i.strip() for i in line.split(',')]
#             if line_items[0] == 'Time Step':
#                 kk = (int(line_items[1])-1, int(line_items[3])-1)
#                 kstpkper.append(kk)
#                 ins[kk] = []
#                 outs[kk] = []
#             elif 'ZONE' in line_items[1]:
#                 zones = [z for z in line_items if z != '']
#                 col_header = ['Record Name'] + zones
#                 dtype = [('flow_dir', '|S3'), ('record', '|S20')] + \
#                         [(col_name, np.float32) for col_name in col_header[1:]]
#             elif line_items[1] == 'IN':
#                 ins_flag = True
#                 continue
#             elif line_items[0] == 'Total IN':
#                 ins_flag = False
#             elif line_items[1] == 'OUT':
#                 outs_flag = True
#                 continue
#             elif line_items[0] == 'Total OUT':
#                 outs_flag = False
#             if ins_flag:
#                 z = [x for x in line_items if x != '']
#                 z.insert(0, 'in')
#                 z[2:] = [float(zz) for zz in z[2:]]
#                 ins[kk].append(tuple(z))
#             elif outs_flag:
#                 z = [x for x in line_items if x != '']
#                 z.insert(0, 'out')
#                 z[2:] = [float(zz) for zz in z[2:]]
#                 outs[kk].append(tuple(z))
#     zbud = OrderedDict()
#     for kk in kstpkper:
#         try:
#             dat = ins[kk] + outs[kk]
#             zbud[kk] = np.array(dat, dtype=dtype)
#         except Exception as e:
#             print(e)
#             return None
#     return zbud
#
#
# def _write_zonfile(izone, zonfile, iprn):
#     assert 'int' in str(izone.dtype), 'Input zone array (dtype={}) must be an integer array.'.format(izone.dtype)
#     if len(izone.shape) == 2:
#         nlay = 1
#         nrow, ncol = izone.shape
#         z = np.zeros((nlay, nrow, ncol))
#         z[0, :, :] = izone
#         izone = z
#     elif len(izone.shape) == 3:
#         nlay, nrow, ncol = izone.shape
#
#     with open(zonfile, 'w') as f:
#         f.write('{} {} {}\n'.format(nlay, nrow, ncol))
#
#         for lay in range(nlay):
#             f.write('INTERNAL ({ncol}I4) {iprn}\n'.format(ncol=ncol, iprn=iprn))
#             for row in range(nrow):
#                 f.write(''.join(['{:4d}'.format(int(val)) for val in izone[lay, row, :]])+'\n')
#
#         #     f.write('INTERNAL (free) {iprn}\n'.format(iprn=iprn))
#         #     for row in range(nrow):
#         #         f.write(' '.join(['{:d}'.format(int(val)) for val in izone[lay, row, :]])+'\n')
#         # f.write('ALLZONES ' + ' '.join([str(int(z)) for z in np.unique(izone)])+'\n')
#     return
#
#
# def is_exe(fpath):
#     """
#     Taken from flopy.mbase
#
#     """
#     return os.path.isfile(fpath) and os.access(fpath, os.X_OK)
#
#
# def which(program):
#     """
#     Taken from flopy.mbase
#
#     """
#     fpath, fname = os.path.split(program)
#     if fpath:
#         if is_exe(program):
#             return program
#     else:
#         # test for exe in current working directory
#         if is_exe(program):
#             return program
#         # test for exe in path statement
#         for path in os.environ["PATH"].split(os.pathsep):
#             path = path.strip('"')
#             exe_file = os.path.join(path, program)
#             if is_exe(exe_file):
#                 return exe_file
#     return None
#
#
# def _call(exe_name, args, zonbud_ws='./', silent=True):
#
#     # success = False
#     # buff = []
#
#     # Check to make sure that program and namefile exist
#     exe = which(exe_name)
#     if exe is None:
#         import platform
#
#         if platform.system() in 'Windows':
#             if not exe_name.lower().endswith('.exe'):
#                 exe = which(exe_name + '.exe')
#     if exe is None:
#         s = 'The program {} does not exist or is not executable.'.format(
#             exe_name)
#         raise Exception(s)
#     else:
#         if not silent:
#             s = 'FloPy is using the following executable to run ZoneBudget: {}'.format(
#                 exe)
#             print(s)
#
#     # simple little function for the thread to target
#     def q_output(output,q):
#             for line in iter(output.readline,b''):
#                 q.put(line)
#             #time.sleep(1)
#             #output.close()
#     argsstr = ''.join([arg+os.linesep for arg in args])
#     proc = sp.Popen([exe_name], stdin=sp.PIPE, stdout=sp.PIPE, cwd=zonbud_ws)
#     stdout = proc.communicate(input=argsstr)[0]
#     if not silent:
#         print(stdout)
#     while True:
#         line = proc.stdout.readline()
#         c = line.decode('utf-8')
#         if c != '':
#             c = c.rstrip('\r\n')
#             # if report == True:
#                 # buff.append(c)
#             if not silent:
#                 print(c)
#         else:
#             # success = True
#             break
#     return
