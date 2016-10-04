from __future__ import print_function
import os
import numpy as np
import warnings
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

        pcterr = 100 * self.ins_minus_out / (self.ins_plus_out / 2.)
        self.pcterr = np.array([i if not np.isnan(i) else 0 for i in pcterr])

    def get_total_inflow(self):
        return self.ins_sum

    def get_total_outflow(self):
        return self.out_sum

    def get_percent_error(self):
        return self.pcterr

    def to_csv(self, fname, write_format='pandas', formatter=None):
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
                        else:
                            items.append(formatter(i))
                    f.write(','.join(items)+'\n')
                f.write(','.join([' ', 'Total IN'] + [formatter(i) for i in self.ins_sum])+'\n')

                # Write OUT terms
                for rec in self.records[self.out_idx]:
                    items = []
                    for i in rec:
                        if isinstance(i, str):
                            items.append(i)
                        else:
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
                    for i in list(rec)[1:]:
                        if isinstance(i, str):
                            items.append(i)
                        else:
                            items.append(formatter(i))
                    f.write(','.join(items)+'\n')
                f.write(','.join(['Total IN'] + [formatter(i) for i in self.ins_sum])+'\n')

                # Write OUT terms
                f.write(','.join([' '] + ['OUT']*(len(self.records.dtype.names[1:])-1))+'\n')
                for rec in self.records[self.out_idx]:
                    items = []
                    for i in list(rec)[1:]:
                        if isinstance(i, str):
                            items.append(i)
                        else:
                            items.append(formatter(i))
                    f.write(','.join(items) + '\n')
                f.write(','.join(['Total OUT'] + [formatter(i) for i in self.out_sum])+'\n')

                # Write mass balance terms
                f.write(','.join(['IN-OUT'] + [formatter(i) for i in self.ins_minus_out])+'\n')
                f.write(','.join(['Percent Error'] + [formatter(i) for i in self.pcterr])+'\n')

    def _fields_view(self, a, fields):
        new = a[fields].view(np.float64).reshape(a.shape + (-1,))
        return new


class ZoneBudget(object):
    """
    ZoneBudget class

    Example usage:

    >>>from flopy.utils import ZoneBudget
    >>>zb = ZoneBudget('zonebudtest.cbc')
    >>>bud = zb.get_budget('GWBasins.zon')
    >>>bud.to_csv('zonebudtest.csv')
    """
    def __init__(self, cbc_file):

        # INTERNAL FLOW TERMS ARE USED TO CALCULATE FLOW BETWEEN ZONES.
        # CONSTANT-HEAD TERMS ARE USED TO IDENTIFY WHERE CONSTANT-HEAD CELLS ARE AND THEN USE
        # FACE FLOWS TO DETERMINE THE AMOUNT OF FLOW.
        # SWIADDTO* terms are used by the SWI2 package.
        internal_flow_terms = ['CONSTANT HEAD', 'FLOW RIGHT FACE', 'FLOW FRONT FACE', 'FLOW LOWER FACE',
                               'SWIADDTOCH', 'SWIADDTOFRF', 'SWIADDTOFFF', 'SWIADDTOFLF']

        if isinstance(cbc_file, CellBudgetFile):
            self.cbc = cbc_file
        elif isinstance(cbc_file, str) and os.path.isfile(cbc_file):
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

        # Internal flow record names
        self.ift_record_names = [r.strip() for r in self.record_names if r.strip() in internal_flow_terms]

        # Check the shape of the cbc budget file arrays
        self.cbc_shape = self.get_model_shape()
        self.nlay, self.nrow, self.ncol = self.cbc_shape

        # Determine if there are constant head cells to track
        self.ich = np.ma.zeros(self.cbc_shape, np.int32)
        self.ich.mask = True
        if 'CONSTANT HEAD' in self.ift_record_names:
            chd = self.cbc.get_data(text='CONSTANT HEAD', kstpkper=self.cbc.get_kstpkper()[0], full3D=True)[0]
            self.ich[np.nonzero(chd)] = 1
            self.ich.mask[np.nonzero(chd)] = False
            # for l, r, c in zip(*np.where(chd != 0.)):
            #     self.ich[l, r, c] = 1
            #     self.ich.mask[l, r, c] = False

        self.float_type = np.float64

    def get_model_shape(self):
        l, r, c = self.cbc.get_data(idx=0, full3D=True)[0].shape
        return l, r, c

    def get_budget(self, z, kstpkper=None):
        """

        Parameters
        ----------
        z
        kstpkper
        totim
        kwargs

        Returns
        -------
        Budget object
        """
        if kstpkper is None:
            print('Getting budget for last time step/stress period.')
            kstpkper = self.cbc.get_kstpkper()[-1]
        assert kstpkper in self.get_kstpkper(), 'The specified time step/stress period' \
                                                ' does not exist {}'.format(kstpkper)

        # Make sure the input zone array has the same shape as the cell budget file
        if len(z.shape) == 2:
            izone = np.zeros(self.cbc_shape, np.int32)
            for i in range(izone.shape[0]):
                izone[i, :, :] = z
        else:
            izone = z.copy()

        # FIND THE UNIQUE SET OF ZONES CONTAINED THEREIN
        zones = self._find_unique_zones(izone.ravel())

        assert izone.shape == self.cbc_shape, \
            'Shape of input zone array {} does not' \
            ' match the cell by cell' \
            ' budget file {}'.format(izone.shape, self.cbc_shape)

        # Create containers for budget term tuples
        inflows = []
        outflows = []

        # ACCUMULATE SOURCE/SINK/STORAGE TERMS
        for recname in self.ssst_record_names:

            if recname in ['RECHARGE', 'DRAINS']:
                data = self.cbc.get_data(text=recname, kstpkper=kstpkper, full3D=True)[0]
                budin = np.ma.zeros(self.cbc_shape)
                budout = np.ma.zeros(self.cbc_shape)
                budin[data > 0] = data[data > 0]
                budout[data < 0] = data[data < 0]

            else:
                data = self.cbc.get_data(text=recname, kstpkper=kstpkper, full3D=False)[0]
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

            in_tup, out_tup = self._get_source_sink_storage_terms_tuple(recname, budin, budout, izone)
            inflows.append(in_tup)
            outflows.append(out_tup)

        # ACCUMULATE CONSTANT HEAD TERMS
        if 'CONSTANT HEAD' in self.ift_record_names:
            # CONSTANT-HEAD FLOW -- DON'T ACCUMULATE THE CELL-BY-CELL VALUES FOR
            # CONSTANT-HEAD FLOW BECAUSE THEY MAY INCLUDE PARTIALLY CANCELING
            # INS AND OUTS.  USE CONSTANT-HEAD TERM TO IDENTIFY WHERE CONSTANT-
            # HEAD CELLS ARE AND THEN USE FACE FLOWS TO DETERMINE THE AMOUNT OF
            # FLOW.  STORE CONSTANT-HEAD LOCATIONS IN ICH ARRAY.
            inflow, outflow = self._get_constant_head_flow_term_tuple(kstpkper, izone)
            inflows.append(tuple(['in', 'CONSTANT HEAD'] + [val for val in inflow]))
            outflows.append(tuple(['out', 'CONSTANT HEAD'] + [val for val in outflow]))
            # TEMPORARY WARNINGS
            if np.count_nonzero(self.ich):
                chwarn = 'Budget information for CONSTANT HEAD cells is not yet supported.' \
                         'Any non-zero results for CONSTANT HEAD and' \
                         'SWIADDTOCH should be considered erroneous.'
                warnings.warn(chwarn, UserWarning)
                # /TEMPORARY WARNINGS

        if 'SWIADDTOCH' in self.ift_record_names:
            # CONSTANT-HEAD FLOW -- DON'T ACCUMULATE THE CELL-BY-CELL VALUES FOR
            # CONSTANT-HEAD FLOW BECAUSE THEY MAY INCLUDE PARTIALLY CANCELING
            # INS AND OUTS.  USE CONSTANT-HEAD TERM TO IDENTIFY WHERE CONSTANT-
            # HEAD CELLS ARE AND THEN USE FACE FLOWS TO DETERMINE THE AMOUNT OF
            # FLOW.  STORE CONSTANT-HEAD LOCATIONS IN ICH ARRAY.
            inflow, outflow = self._get_constant_head_flow_term_tuple(kstpkper, izone)
            inflows.append(tuple(['in', 'SWIADDTOCH'] + [val for val in inflow]))
            outflows.append(tuple(['out', 'SWIADDTOCH'] + [val for val in outflow]))
            # TEMPORARY WARNINGS
            if np.count_nonzero(self.ich) and self.is_swi:
                chwarn = 'Budget information for CONSTANT HEAD cells is not yet supported' \
                         'for SWI2 models. Any non-zero results for CONSTANT HEAD and' \
                         'SWIADDTOCH should be considered erroneous.'
                warnings.warn(chwarn, UserWarning)
            # /TEMPORARY WARNINGS

        # ACCUMULATE INTERNAL FLOW TERMS
        in_tups, out_tups = self.accumulate_internal_flow(kstpkper, izone)
        inflows.extend(in_tups)
        outflows.extend(out_tups)

        # Combine inflows and outflows and return a Budget object
        q = inflows + outflows
        dtype = np.dtype([('flow_dir', '|S3'),
                          ('record', '|S20')] +
                         [('ZONE {:d}'.format(z), self.float_type) for z in zones])
        return Budget(np.array(q, dtype=dtype), kstpkper=kstpkper)

    def accumulate_internal_flow(self, kstpkper, izone):
        # Each flow term is a tuple of (from zone, to zone, flux)
        frf, fff, flf, swifrf, swifff, swiflf = self._get_internal_flow_terms(kstpkper, izone)

        # Combine and sort flux tuples
        q_tups = sorted(frf + fff + flf + swifrf + swifff + swiflf)

        zones = self._find_unique_zones(izone.ravel())

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

        in_tups, out_tups = [], []
        # Pull the from/to zone/flux tuples back out and append them to a list
        for v in q_in.values():
            in_tups.append(tuple(v.values()))
        for v in q_out.values():
            out_tups.append(tuple(v.values()))

        return in_tups, out_tups

    def _get_internal_flow_terms(self, kstpkper, izone):
        # PROCESS EACH INTERNAL FLOW RECORD IN THE CELL-BY-CELL BUDGET FILE
        frf, fff, flf, swifrf, swifff, swiflf = [], [], [], [], [], []
        for recname in self.ift_record_names:
            if recname == 'FLOW RIGHT FACE':
                if self.ncol >= 2:
                    bud = self.cbc.get_data(text=recname, kstpkper=kstpkper, full3D=True)[0]
                    frf = self._get_internal_flow_terms_tuple_frf(bud, izone)
            elif recname == 'FLOW FRONT FACE':
                if self.nrow >= 2:
                    bud = self.cbc.get_data(text=recname, kstpkper=kstpkper, full3D=True)[0]
                    fff = self._get_internal_flow_terms_tuple_fff(bud, izone)
            elif recname == 'FLOW LOWER FACE':
                if self.nlay >= 2:
                    bud = self.cbc.get_data(text=recname, kstpkper=kstpkper, full3D=True)[0]
                    flf = self._get_internal_flow_terms_tuple_flf(bud, izone)
            elif recname == 'SWIADDTOFRF':
                if self.ncol >= 2:
                    bud = self.cbc.get_data(text=recname, kstpkper=kstpkper, full3D=True)[0]
                    swifrf = self._get_internal_flow_terms_tuple_frf(bud, izone)
            elif recname == 'SWIADDTOFFF':
                if self.nrow >= 2:
                    bud = self.cbc.get_data(text=recname, kstpkper=kstpkper, full3D=True)[0]
                    swifff = self._get_internal_flow_terms_tuple_fff(bud, izone)
            elif recname == 'SWIADDTOFLF':
                if self.nlay >= 2:
                    bud = self.cbc.get_data(text=recname, kstpkper=kstpkper, full3D=True)[0]
                    swiflf = self._get_internal_flow_terms_tuple_flf(bud, izone)
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
    def _get_source_sink_storage_terms_tuple(recname, budin, budout, izone):
        recin = ['in', recname.strip()] + [np.abs(budin[(izone == z)].sum()) for z in np.unique(izone.ravel())]
        recout = ['out', recname.strip()] + [np.abs(budout[(izone == z)].sum()) for z in np.unique(izone.ravel())]
        recin = tuple([val if not type(val) == np.ma.core.MaskedConstant else 0. for val in recin])
        recout = tuple([val if not type(val) == np.ma.core.MaskedConstant else 0. for val in recout])
        return recin, recout

    def _get_internal_flow_terms_tuple_frf(self, bud, izone):
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
        if np.count_nonzero(self.ich):
            q[(self.ich[l, r, c] == 1) & (self.ich[l, r, c-1] == 1)] = 0.

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
        if np.count_nonzero(self.ich):
            q[(self.ich[l, r, c] == 1) & (self.ich[l, r, c-1] == 1)] = 0.

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

    def _get_internal_flow_terms_tuple_fff(self, bud, izone):
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
        if np.count_nonzero(self.ich):
            q[(self.ich[l, r, c] == 1) & (self.ich[l, r-1, c] == 1)] = 0.

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
        if np.count_nonzero(self.ich):
            q[(self.ich[l, r, c] == 1) & (self.ich[l, r-1, c] == 1)] = 0.

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

    def _get_internal_flow_terms_tuple_flf(self, bud, izone):
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
        if np.count_nonzero(self.ich):
            q[(self.ich[l, r, c] == 1) & (self.ich[l-1, r, c] == 1)] = 0.

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
        if np.count_nonzero(self.ich):
            q[(self.ich[l, r, c] == 1) & (self.ich[l-1, r, c] == 1)] = 0.

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

    def _get_constant_head_flow_term_tuple(self, kstpkper, izone):

        q_chd_in = np.zeros(self.cbc_shape, dtype=np.float64)
        q_chd_out = np.zeros(self.cbc_shape, dtype=np.float64)

        # CALCULATE FLOW TO CONSTANT-HEAD CELLS
        for recname in self.ift_record_names:
            q = self.cbc.get_data(text=recname, kstpkper=kstpkper, full3D=True)[0]
            q_chd_in[(self.ich == 1) & (q > 0)] += q[(self.ich == 1) & (q > 0)]
            q_chd_out[(self.ich == 1) & (q < 0)] += q[(self.ich == 1) & (q < 0)]

        chd_inflow = [np.abs(q_chd_in[(izone == z)].sum()) for z in np.unique(izone.ravel())]
        chd_outflow = [np.abs(q_chd_out[(izone == z)].sum()) for z in np.unique(izone.ravel())]
        chd_inflow = tuple([val if not type(val) == np.ma.core.MaskedConstant else 0. for val in chd_inflow])
        chd_outflow = tuple([val if not type(val) == np.ma.core.MaskedConstant else 0. for val in chd_outflow])
        return chd_inflow, chd_outflow

    @staticmethod
    def _find_unique_zones(a):
        z = [int(i) for i in np.unique(a)]
        return z
