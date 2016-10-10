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
    def __init__(self, records, **kwargs):
        self.records = records
        # if 'kstpkper' in kwargs.keys():
        #     self.kstpkper =
        # self.kstpkper = kstpkper
        # self.totim = totim
        # assert(self.kstpkper is not None or self.totim is not None), 'Budget object requires either kstpkper ' \
        #                                                              'or totim be be specified.'
        self.kwargs = kwargs
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
                if 'kstpkper' in self.kwargs.keys():
                    header = 'Time Step, {kstp}, Stress Period, {kper}\n'.format(kstp=self.kwargs['kstpkper'][0]+1,
                                                                                 kper=self.kwargs['kstpkper'][1]+1)
                elif 'totim' in self.kwargs.keys():
                    header = 'Sim. Time, {totim}\n'.format(totim=self.kwargs['totim'])
                else:
                    raise Exception('No stress period/time step or time specified.')

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

    @staticmethod
    def _fields_view(a, fields):
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
        internal_flow_terms = ['FLOW RIGHT FACE', 'FLOW FRONT FACE', 'FLOW LOWER FACE',
                               'SWIADDTOFRF', 'SWIADDTOFFF', 'SWIADDTOFLF']
        chd_terms = ['CONSTANT HEAD', 'SWIADDTOCH']

        if isinstance(cbc_file, CellBudgetFile):
            self.cbc = cbc_file
        elif isinstance(cbc_file, str) and os.path.isfile(cbc_file):
            self.cbc = CellBudgetFile(cbc_file)

        # All record names in the cell-by-cell budget binary file
        self.record_names = [n.strip() for n in self.cbc.unique_record_names()]

        # Constant head term record names
        self.chd_record_names = [n.strip() for n in self.cbc.unique_record_names()
                                 if n.strip() in chd_terms]

        # Source/sink/storage term record names
        self.ssst_record_names = [n.strip() for n in self.cbc.unique_record_names()
                                  if n.strip() not in internal_flow_terms
                                  and n.strip() not in chd_terms]

        # Internal flow record names
        self.ift_record_names = [n.strip() for n in self.cbc.unique_record_names()
                                 if n.strip() in internal_flow_terms]

        # Check the shape of the cbc budget file arrays
        self.cbc_shape = self.get_model_shape()
        self.nlay, self.nrow, self.ncol = self.cbc_shape

        # Determine if there are constant head cells to track
        self.ich = np.ma.zeros(self.cbc_shape, np.int32)
        self.ich.mask = True
        self.ich_swi = np.ma.zeros(self.cbc_shape, np.int32)
        self.ich_swi.mask = True
        if 'CONSTANT HEAD' in self.record_names:
            recname = 'CONSTANT HEAD'
            chd = self.cbc.get_data(text=recname, kstpkper=self.cbc.get_kstpkper()[0], full3D=True)[0]
            self.ich[np.nonzero(chd)] = 1
            self.ich.mask[np.nonzero(chd)] = False
        elif 'SWIADDTOCH' in self.record_names:
            recname = 'SWIADDTOCH'
            chd_swi = self.cbc.get_data(text=recname, kstpkper=self.cbc.get_kstpkper()[0], full3D=True)[0]
            self.ich_swi[np.nonzero(chd_swi)] = 1
            self.ich_swi.mask[np.nonzero(chd_swi)] = False

        self.float_type = np.float64

    def get_model_shape(self):
        l, r, c = self.cbc.get_data(idx=0, full3D=True)[0].shape
        return l, r, c

    def get_budget(self, z, **kwargs):
        """
        Creates a budget for the specified zones and time step/stress period.

        Parameters
        ----------
        z:
        kstpkper:

        Returns
        -------
        Budget object
        """
        if 'kstpkper' in kwargs.keys():
            s = 'The specified time step/stress period' \
                ' does not exist {}'.format(kwargs['kstpkper'])
            assert kwargs['kstpkper'] in self.cbc.get_kstpkper(), print(s)
        elif 'totim' in kwargs.keys():
            s = 'The time ' \
                ' does not exist {}'.format(kwargs['totim'])
            assert kwargs['kstpkper'] in self.cbc.get_times(), print(s)
        else:
            raise Exception('No stress period/time step or time specified.')

        # Make sure the input zone array has the same shape as the cell budget file
        if len(z.shape) == 2:
            izone = np.zeros(self.cbc_shape, np.int32)
            for i in range(izone.shape[0]):
                izone[i, :, :] = z
        else:
            izone = z.copy()

        # Get the unique set of zones
        zones = [int(i) for i in np.unique(izone)]

        assert izone.shape == self.cbc_shape, \
            'Shape of input zone array {} does not' \
            ' match the cell by cell' \
            ' budget file {}'.format(izone.shape, self.cbc_shape)

        # Create containers for budget term tuples
        # The first two items in each tuple are flow direction and budget record name
        # The remainder of the tuple items are the fluxes to/from each zone
        # Example, inflow from river leakage aggregated over 4 zones:
        # ('in', 'RIVER LEAKAGE', 0.0, 0.00419, 0.0, 0.0)
        inflows = []
        outflows = []

        # ACCUMULATE CONSTANT HEAD TERM
        # (do this first to match output from the zonbud program executable)
        if 'CONSTANT HEAD' in self.chd_record_names:
            recname = 'CONSTANT HEAD'
            inflow, outflow = self._get_constant_head_flow_term_tuple(recname, izone, **kwargs)
            inflows.append(tuple(['in', recname] + [val for val in inflow]))
            outflows.append(tuple(['out', recname] + [val for val in outflow]))
            if np.count_nonzero(self.ich) > 0:
                # TEMPORARY WARNINGS
                chwarn = 'Budget information for CONSTANT HEAD cells is not yet supported.\n' \
                         'Any non-zero results for CONSTANT HEAD and ' \
                         'SWIADDTOCH should be considered erroneous.'
                warnings.warn(chwarn, UserWarning)
                # /TEMPORARY WARNINGS

        # ACCUMULATE SOURCE/SINK/STORAGE TERMS
        # (e.g. recharge, drains, wells, mnw, riv, etc.)
        for recname in self.ssst_record_names:

            if recname in ['RECHARGE']:
                """
                Update this piece--use full3D=False and grab it as a list. First
                item is a layer array and the other is a data array.
                """
                rlay, rdata = self.cbc.get_data(text=recname, full3D=False, **kwargs)[0]
                data = np.ma.zeros(self.cbc_shape, self.float_type)

                data = self.cbc.get_data(text=recname, full3D=True, **kwargs)[0]
                budin = np.ma.zeros(self.cbc_shape, self.float_type)
                budout = np.ma.zeros(self.cbc_shape, self.float_type)
                budin[data > 0] = data[data > 0]
                budout[data < 0] = data[data < 0]

            else:
                data = self.cbc.get_data(text=recname, full3D=False, **kwargs)[0]

                if not isinstance(data, np.recarray):
                    # Not a recarray, probably due to using old MF88-style budget file
                    mf88warn = 'Use of a MODFLOW-88 style budget files with the {recname}\n'\
                               'record may result in erroneous budget results for cells\n' \
                               'containing multiple boundaries of the same type. \n' \
                               'Please use the "COMPACT BUDGET" option of the Output ' \
                               'Control package.'.format(recname=recname)
                    warnings.warn(mf88warn, UserWarning)
                    budin = np.ma.zeros(self.cbc_shape, self.float_type)
                    budout = np.ma.zeros(self.cbc_shape, self.float_type)
                    budin[data > 0] = data[data > 0]
                    budout[data < 0] = data[data < 0]
                else:
                    budin = np.ma.zeros((self.nlay * self.nrow * self.ncol), self.float_type)
                    budout = np.ma.zeros((self.nlay * self.nrow * self.ncol), self.float_type)
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

        # ACCUMULATE SWIADDTOCH
        # (do this last to match output from the zonbud program executable)
        if 'SWIADDTOCH' in self.chd_record_names:
            recname = 'SWIADDTOCH'
            inflow, outflow = self._get_constant_head_flow_term_tuple(recname, izone, **kwargs)
            inflows.append(tuple(['in', recname] + [val for val in inflow]))
            outflows.append(tuple(['out', recname] + [val for val in outflow]))
            if np.count_nonzero(self.ich) > 0:
                # TEMPORARY WARNINGS
                chwarn = 'Budget information for CONSTANT HEAD cells is not yet supported ' \
                         'for SWI2 models. \nAny non-zero results for CONSTANT HEAD and ' \
                         'SWIADDTOCH should be considered erroneous.'
                warnings.warn(chwarn, UserWarning)
                # /TEMPORARY WARNINGS

        # ACCUMULATE INTERNAL FACE FLOW TERMS
        in_tups, out_tups = self.accumulate_internal_flow(izone, **kwargs)
        inflows.extend(in_tups)
        outflows.extend(out_tups)

        # Combine all inflows and outflows and return a Budget object
        q = inflows + outflows
        dtype_list = [('flow_dir', '|S3'), ('record', '|S20')]
        dtype_list += [('ZONE {:d}'.format(z), self.float_type) for z in zones]
        dtype = np.dtype(dtype_list)
        return Budget(np.array(q, dtype=dtype), **kwargs)

    def accumulate_internal_flow(self, izone, **kwargs):
        """
        Accumulate, by zone, fluxes for the face flow records.

        :param kstpkper:
        :param izone:
        :return:
        """
        # Each flow term is a tuple of ("from zone", "to zone", "absolute flux")
        frf, fff, flf, swiadd2frf, swiadd2fff, swiadd2flf = self._get_internal_flow_terms(izone, **kwargs)

        # Combine and sort flux tuples
        q_tups = sorted(frf + fff + flf + swiadd2frf + swiadd2fff + swiadd2flf)

        zones = [int(i) for i in np.unique(izone)]

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

        # Pull the from zone/to zone/flux tuples back out and append them to a list
        in_tups, out_tups = [], []
        for v in q_in.values():
            in_tups.append(tuple(v.values()))
        for v in q_out.values():
            out_tups.append(tuple(v.values()))

        return in_tups, out_tups

    def _get_internal_flow_terms(self, izone, **kwargs):
        """
        Process each internal flow record in the cell-by-cell budget file

        :param kstpkper:
        :param izone:
        :return:
        """
        frf, fff, flf, swiadd2frf, swiadd2fff, swiadd2flf = [], [], [], [], [], []
        for recname in self.ift_record_names:
            if recname == 'FLOW RIGHT FACE':
                frf = self._get_internal_flow_terms_tuple_frf(recname, izone, **kwargs)
            elif recname == 'FLOW FRONT FACE':
                fff = self._get_internal_flow_terms_tuple_fff(recname, izone, **kwargs)
            elif recname == 'FLOW LOWER FACE':
                flf = self._get_internal_flow_terms_tuple_flf(recname, izone, **kwargs)
            elif recname == 'SWIADDTOFRF':
                swiadd2frf = self._get_internal_flow_terms_tuple_frf(recname, izone, **kwargs)
            elif recname == 'SWIADDTOFFF':
                swiadd2fff = self._get_internal_flow_terms_tuple_fff(recname, izone, **kwargs)
            elif recname == 'SWIADDTOFLF':
                swiadd2flf = self._get_internal_flow_terms_tuple_flf(recname, izone, **kwargs)
        return frf, fff, flf, swiadd2frf, swiadd2fff, swiadd2flf

    def _get_internal_flow_terms_tuple_frf(self, recname, izone, **kwargs):
        # ACCUMULATE FLOW BETWEEN ZONES ACROSS COLUMNS. COMPUTE FLOW ONLY BETWEEN A ZONE
        # AND A HIGHER ZONE -- FLOW FROM ZONE 4 TO 3 IS THE NEGATIVE OF FLOW FROM 3 TO 4.
        # FIRST, CALCULATE FLOW BETWEEN NODE J,I,K AND J-1,I,K.
        # Accumulate flow from lower zones to higher zones from "left" to "right".
        # Flow into the higher zone will be <0 Flow Right Face from the adjacent cell to the "left".
        # Returns a tuple of ("to zone", "from zone", "absolute flux")
        bud = self.cbc.get_data(text=recname, **kwargs)[0]
        # if not isinstance(bud, np.recarray):
        #     mf88warn = 'Use of a MODFLOW-88 style budget file may result in \n' \
        #                'the partial cancellation of fluxes in {recname} ' \
        #                'cells where bi-directional flow occurs. \n' \
        #                'If using MODFLOW-2000 or later, please use the ' \
        #                '"COMPACT BUDGET" option of the Output Control package.'.format(recname=recname)
        #     warnings.warn(mf88warn, UserWarning)

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

        # Create tuples of ("to zone", "from zone", "absolute flux")
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

        # Create tuples of ("to zone", "from zone", "absolute flux")
        neg = zip(to_zones[idx_neg], from_zones[idx_neg], np.abs(q[idx_neg]))
        pos = zip(from_zones[idx_pos], to_zones[idx_pos], np.abs(q[idx_pos]))
        nzgt_r2l = neg + pos

        # Returns a tuple of ("to zone", "from zone", "absolute flux")
        nzgt = sorted(nzgt_l2r + nzgt_r2l, key=lambda tup: tup[:2])
        return nzgt

    def _get_internal_flow_terms_tuple_fff(self, recname, izone, **kwargs):
        # ACCUMULATE FLOW BETWEEN ZONES ACROSS ROWS. COMPUTE FLOW ONLY BETWEEN A ZONE
        #  AND A HIGHER ZONE -- FLOW FROM ZONE 4 TO 3 IS THE NEGATIVE OF FLOW FROM 3 TO 4.
        # FIRST, CALCULATE FLOW BETWEEN NODE J,I,K AND J,I-1,K.
        # Accumulate flow from lower zones to higher zones from "up" to "down".
        # Returns a tuple of ("to zone", "from zone", "absolute flux")
        bud = self.cbc.get_data(text=recname, **kwargs)[0]
        # if not isinstance(bud, np.recarray):
        #     mf88warn = 'Use of a MODFLOW-88 style budget file may result in \n' \
        #                'the partial cancellation of fluxes in {recname} ' \
        #                'cells where bi-directional flow occurs. \n' \
        #                'If using MODFLOW-2000 or later, please use the ' \
        #                '"COMPACT BUDGET" option of the Output Control package.'.format(recname=recname)
        #     warnings.warn(mf88warn, UserWarning)

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

        # Create tuples of ("to zone", "from zone", "absolute flux")
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

        # Create tuples of ("to zone", "from zone", "absolute flux")
        neg = zip(to_zones[idx_neg], from_zones[idx_neg], np.abs(q[idx_neg]))
        pos = zip(from_zones[idx_pos], to_zones[idx_pos], np.abs(q[idx_pos]))
        nzgt_d2u = neg + pos

        # Returns a tuple of ("to zone", "from zone", "absolute flux")
        nzgt = sorted(nzgt_u2d + nzgt_d2u, key=lambda tup: tup[:2])
        return nzgt

    def _get_internal_flow_terms_tuple_flf(self, recname, izone, **kwargs):
        # ACCUMULATE FLOW BETWEEN ZONES ACROSS LAYERS. COMPUTE FLOW ONLY BETWEEN A ZONE
        #  AND A HIGHER ZONE -- FLOW FROM ZONE 4 TO 3 IS THE NEGATIVE OF FLOW FROM 3 TO 4.
        # FIRST, CALCULATE FLOW BETWEEN NODE J,I,K AND J,I,K-1.
        # Accumulate flow from lower zones to higher zones from "top" to "bottom".
        # Returns a tuple of ("to zone", "from zone", "absolute flux")
        bud = self.cbc.get_data(text=recname, **kwargs)[0]
        # if not isinstance(bud, np.recarray):
        #     mf88warn = 'Use of a MODFLOW-88 style budget file may result in \n' \
        #                'the partial cancellation of fluxes in {recname} ' \
        #                'cells where bi-directional flow occurs. \n' \
        #                'If using MODFLOW-2000 or later, please use the ' \
        #                '"COMPACT BUDGET" option of the Output Control package.'.format(recname=recname)
        #     warnings.warn(mf88warn, UserWarning)

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

        # Create tuples of ("to zone", "from zone", "absolute flux")
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

        # Create tuples of ("to zone", "from zone", "absolute flux")
        neg = zip(to_zones[idx_neg], from_zones[idx_neg], np.abs(q[idx_neg]))
        pos = zip(from_zones[idx_pos], to_zones[idx_pos], np.abs(q[idx_pos]))
        nzgt_b2t = neg + pos

        # Returns a tuple of ("to zone", "from zone", "absolute flux")
        nzgt = sorted(nzgt_t2b + nzgt_b2t, key=lambda tup: tup[:2])
        return nzgt

    def _get_constant_head_flow_term_tuple(self, recname, izone, **kwargs):
        # CONSTANT-HEAD FLOW -- DON'T ACCUMULATE THE CELL-BY-CELL VALUES FOR
        # CONSTANT-HEAD FLOW BECAUSE THEY MAY INCLUDE PARTIALLY CANCELING
        # INS AND OUTS.  USE CONSTANT-HEAD TERM TO IDENTIFY WHERE CONSTANT-
        # HEAD CELLS ARE AND THEN USE FACE FLOWS TO DETERMINE THE AMOUNT OF
        # FLOW.
        q_chd_in = np.zeros(self.cbc_shape, dtype=np.float64)
        q_chd_out = np.zeros(self.cbc_shape, dtype=np.float64)

        if recname == 'CONSTANT HEAD':
            ich = self.ich
            ff_records = [n for n in self.ift_record_names if 'SWI' not in n]
        elif recname == 'SWIADDTOCH':
            ich = self.ich_swi
            ff_records = [n for n in self.ift_record_names if 'SWI' in n]

        for ff in ff_records:
            q = self.cbc.get_data(text=ff, **kwargs)[0]
            # print(ff, q[(ich == 1)])

            q_chd_in[(ich == 1) & (q > 0)] += q[(ich == 1) & (q > 0)]
            q_chd_out[(ich == 1) & (q < 0)] += q[(ich == 1) & (q < 0)]

        chd_inflow = [np.abs(q_chd_in[(izone == z)].sum()) for z in np.unique(izone.ravel())]
        chd_outflow = [np.abs(q_chd_out[(izone == z)].sum()) for z in np.unique(izone.ravel())]
        chd_inflow = tuple([val if not type(val) == np.ma.core.MaskedConstant else 0. for val in chd_inflow])
        chd_outflow = tuple([val if not type(val) == np.ma.core.MaskedConstant else 0. for val in chd_outflow])
        return chd_inflow, chd_outflow

    @staticmethod
    def _get_source_sink_storage_terms_tuple(recname, budin, budout, izone):
        recin = ['in', recname.strip()] + [np.abs(budin[(izone == z)].sum()) for z in np.unique(izone.ravel())]
        recout = ['out', recname.strip()] + [np.abs(budout[(izone == z)].sum()) for z in np.unique(izone.ravel())]
        recin = tuple([val if not type(val) == np.ma.core.MaskedConstant else 0. for val in recin])
        recout = tuple([val if not type(val) == np.ma.core.MaskedConstant else 0. for val in recout])
        return recin, recout

    def get_kstpkper(self):
        return self.cbc.get_kstpkper()

    def get_times(self):
        return self.cbc.get_times()

    def get_indices(self):
        return self.cbc.get_indices()

    # def get_ssst_names(self):
    #     return self.ssst_record_names
    #
    # def get_ssst_cbc_array(self, recname, kstpkper=None, totim=None):
    #     try:
    #         recname = [n for n in self.ssst_record_names if n.strip() == recname][0]
    #     except IndexError:
    #         s = 'Please enter a valid source/sink/storage record name. Use ' \
    #                                                'get_ssst_names() to view a list.'
    #         print(s)
    #         return
    #     if kstpkper is not None:
    #         a = self.cbc.get_data(text=recname, kstpkper=kstpkper, full3D=True)[0]
    #     elif totim is not None:
    #         a = self.cbc.get_data(text=recname, totim=totim, full3D=True)[0]
    #     else:
    #         print('Reading budget for last timestep/stress period.')
    #         kstpkper = self.cbc.get_kstpkper()[-1]
    #         a = self.cbc.get_data(text=recname, kstpkper=kstpkper, full3D=True)[0]
    #     return a

    # def _fields_view(self, a, fields):
    #     new = a[fields].view(self.float_type).reshape(a.shape + (-1,))
    #     return new
