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
    ZoneBudget Budget class.
    """
    # def __new__(cls, input_array, kstpkper, totim):
    #     # cls.budget = input_array
    #     # assert(type(cls.budget) == np.ndarray), 'Input budget is not {}.'.format(np.ndarray)
    #     # cls.kstpkper = kstpkper
    #     # cls.totim = totim
    #     # cls.float_type = np.float64
    #     # assert(cls.kstpkper is not None or cls.totim is not None), 'Budget object requires either kstpkper ' \
    #     #                                                              'or totim be be specified.'
    #     return input_array

    def __init__(self, input_budget, kstpkper=None, totim=None):
        self.array = input_budget
        self.kstpkper = kstpkper
        self.totim = totim
        self.float_type = np.float64
        assert(self.kstpkper is not None or self.totim is not None), 'Budget object requires either kstpkper ' \
                                                                     'or totim be be specified.'

    def to_csv(self, fname='zbud.csv', format='pandas'):

        assert format.lower() in ['pandas', 'zonbud'], 'Format must be one of "pandas" or "zonbud".'
        # List the field names to be used to slice the recarray
        # fields = ['ZONE{: >4d}'.format(z) for z in self.zones]
        fields = [name for name in self.array.dtype.names if 'ZONE' in name]

        ins_idx = np.where(self.array['flow_dir'] == 'in')[0]
        out_idx = np.where(self.array['flow_dir'] == 'out')[0]

        ins = self._fields_view(self.array[ins_idx], fields)
        out = self._fields_view(self.array[out_idx], fields)

        ins_sum = ins.sum(axis=0)
        out_sum = out.sum(axis=0)

        ins_minus_out = ins_sum-out_sum
        ins_plus_out = ins_sum+out_sum

        pcterr = 100*(ins_minus_out)/((ins_plus_out)/2.)

        if format.lower() == 'pandas':
            with open(fname, 'w') as f:

                # Write header
                f.write(','.join([field for field in self.array.dtype.names])+'\n')

                # Write IN terms
                for rec in self.array[ins_idx]:
                    f.write(','.join([str(i) for i in rec])+'\n')
                f.write(','.join([' ', 'Total IN'] + [str(i) for i in ins_sum])+'\n')

                # Write OUT terms
                for rec in self.array[out_idx]:
                    f.write(','.join([str(i) for i in rec])+'\n')
                f.write(','.join([' ', 'Total OUT'] + [str(i) for i in out_sum])+'\n')

                # Write mass balance terms
                dif = ins_sum-out_sum
                f.write(','.join([' ', 'IN-OUT'] + [str(i) for i in dif])+'\n')
                # pcterr = 100*(ins_sum-out_sum)/((ins_sum+out_sum)/2.)
                f.write(','.join([' ', 'Percent Error'] + [str(i) for i in pcterr])+'\n')

        if format.lower() == 'zonbud':
            with open(fname, 'w') as f:

                # Write header
                f.write(','.join(['Time Step', str(self.kstpkper[0]+1),
                                  'Stress Period', str(self.kstpkper[1]+1)])+'\n')
                f.write(','.join([' '] + [field for field in self.array.dtype.names[2:]])+'\n')

                # Write IN terms
                f.write(','.join([' '] + ['IN']*(len(self.array.dtype.names[1:])-1))+'\n')
                for rec in self.array[ins_idx]:
                    f.write(','.join([str(rec[i+1]) for i in range(len(self.array.dtype.names[1:]))])+'\n')
                f.write(','.join(['Total IN'] + [str(i) for i in ins_sum])+'\n')

                # Write OUT terms
                f.write(','.join([' '] + ['OUT']*(len(self.array.dtype.names[1:])-1))+'\n')
                for rec in self.array[out_idx]:
                    f.write(','.join([str(rec[i+1]) for i in range(len(self.array.dtype.names[1:]))])+'\n')
                f.write(','.join(['Total OUT'] + [str(i) for i in out_sum])+'\n')

                # Write mass balance terms
                dif = ins_sum-out_sum
                f.write(','.join(['IN-OUT'] + [str(i) for i in dif])+'\n')
                # pcterr = 100*(ins_sum-out_sum)/((ins_sum+out_sum)/2.)
                f.write(','.join(['Percent Error'] + [str(i) for i in pcterr])+'\n')

        # print('Total IN', ins_sum)
        # print('Total OUT', out_sum)
        # print('IN-OUT', ins_sum-out_sum)
        # print('Percent Error', 100*(ins_sum-out_sum)/((ins_sum+out_sum)/2.))

    def _fields_view(self, a, fields):
        new = a[fields].view(self.float_type).reshape(a.shape + (-1,))
        return new


class ZoneBudget(object):
    """
    ZoneBudget package class

    Example usage:

    >>>import flopy
    >>>from flopy.zonebud import ZoneBudget
    >>>zb = ZoneBudget('zonebudtest.cbc', 'GWBasins.zon')
    >>>zb.to_csv()
    """
    def __init__(self, cbc_file):

        # INTERNAL FLOW TERMS ARE USED TO CALCULATE FLOW BETWEEN ZONES.
        # CONSTANT-HEAD TERMS ARE USED TO IDENTIFY WHERE CONSTANT-HEAD CELLS ARE AND THEN USE
        #  FACE FLOWS TO DETERMINE THE AMOUNT OF FLOW.
        # SWIADDTO* terms are used by the SWI2 package.
        internal_flow_terms = ['CONSTANT HEAD', 'FLOW RIGHT FACE', 'FLOW FRONT FACE', 'FLOW LOWER FACE',
                                    'SWIADDTOCH', 'SWIADDTOFRF', 'SWIADDTOFFF', 'SWIADDTOFLF']


        # OPEN THE CELL-BY-CELL BUDGET BINARY FILE
        self.cbc = CellBudgetFile(cbc_file)

        # GET A LISTING OF THE UNIQUE RECORDS CONTAINED IN THE BUDGET FILE
        self.record_names = self.cbc.unique_record_names()
        self.ssst_record_names = [rec.strip() for rec in self.record_names if rec.strip() not in internal_flow_terms]
        self.ff_record_names = [r.strip() for r in self.record_names if r.strip() in internal_flow_terms]

    def get_budget(self, zon, kstpkper=None, totim=None):

        # Get budget data
        # Placeholder assertion, until multiple kstpkper/totims are supported. "None" will cause the entire
        # simulation to be pulled.
        if kstpkper is not None:
            self.cbc_data = OrderedDict([(recname.strip(), self.cbc.get_data(text=recname, kstpkper=kstpkper, full3D=True)[0])
                                         for recname in self.record_names])
        elif totim is not None:
            self.cbc_data = OrderedDict([(recname.strip(), self.cbc.get_data(text=recname, totim=totim, full3D=True)[0])
                                         for recname in self.record_names])
        else:
            print('Reading budget for last timestep/stress period.')
            kstpkper = self.cbc.get_kstpkper()[-1]
            self.cbc_data = OrderedDict([(recname.strip(), self.cbc.get_data(text=recname, kstpkper=kstpkper, full3D=True)[0])
                                         for recname in self.record_names])

        # OPEN THE ZONE FILE (OR ARRAY) AND FIND THE UNIQUE SET OF ZONES CONTAINED THEREIN
        if isinstance(zon, str):
            if os.path.isfile(zon):
                try:
                    self.izone = np.loadtxt(zon)
                except Exception as e:
                    print(e)
        else:
            self.izone = zon
        self.zones = self._find_unique_zones(self.izone.ravel())

        # Define dtype for the recarray
        self.float_type = np.float64
        self.dtype = np.dtype([('flow_dir', '|S3'), ('record', '|S20')] +
                               [('ZONE{: >4d}'.format(z), self.float_type) for z in self.zones])

        cbc_bud_shape = self.cbc_data[self.cbc_data.keys()[0]].shape
        self.nlay, self.nrow, self.ncol = cbc_bud_shape
        assert self.izone.shape == cbc_bud_shape, \
            'Shape of input zone array {} does not' \
            ' match the cell by cell' \
            'budget file {}'.format(self.izone.shape, cbc_bud_shape)

        # Accumulate source/sink/storage terms by zone
        inflows = []
        outflows = []
        for recname in self.ssst_record_names:
            # bud = cbc.get_data(text=recname, kstpkper=kstpkper, full3D=True)[0]
            bud = self.cbc_data[recname]
            in_tup, out_tup = self._get_source_sink_storage_terms_tuple(recname, bud)
            inflows.append(in_tup)
            outflows.append(out_tup)

        # IF RECORD IS A CONSTANT-HEAD INTERNAL FLOW TERM, ACCUMULATE FACE FLOWS ONLY FOR
        #  CONSTANT-HEAD CELLS
        # ----not yet tested----#
        # bud = cbc.get_data(text='   CONSTANT HEAD', kstpkper=kstpkper, full3D=True)[0]
        bud = self.cbc_data['CONSTANT HEAD']
        ich_lrc = bud[bud != 0]
        if len(ich_lrc) > 0:
            chwarn = 'WARNING: CONSTANT HEAD cells were detected, but will not be included in the zonebudget results.'
            warnings.warn(chwarn, UserWarning)

        ichswi_lrc = []
        if 'SWIADDTOCH' in [r.strip() for r in self.record_names]:
            # bud = cbc.get_data(text='      SWIADDTOCH', kstpkper=kstpkper, full3D=True)[0]
            bud = self.cbc_data['SWIADDTOCH']
            ichswi_lrc += bud[bud != 0]


        # PROCESS EACH INTERNAL FLOW RECORD IN THE CELL-BY-CELL BUDGET FILE
        frf, fff, flf, swifrf, swifff, swiflf = [], [], [], [], [], []
        for recname in self.ff_record_names:

            if recname.strip() == 'CONSTANT HEAD':
                continue
            elif recname.strip() == 'SWIADDTOCH':
                continue

            elif recname.strip() == 'FLOW RIGHT FACE':
                bud = self.cbc_data[recname]
                frf = self._get_internal_flow_terms_tuple_frf(bud)

            elif recname.strip() == 'FLOW FRONT FACE':
                bud = self.cbc_data[recname]
                fff = self._get_internal_flow_terms_tuple_fff(bud)

            elif recname.strip() == 'FLOW LOWER FACE':
                bud = self.cbc_data[recname]
                flf = self._get_internal_flow_terms_tuple_flf(bud)

            elif recname.strip() == 'SWIADDTOFRF':
                bud = self.cbc_data[recname]
                swifrf = self._get_internal_flow_terms_tuple_frf(bud)

            elif recname.strip() == 'SWIADDTOFFF':
                bud = self.cbc_data[recname]
                swifff = self._get_internal_flow_terms_tuple_fff(bud)

            elif recname.strip() == 'SWIADDTOFLF':
                bud = self.cbc_data[recname]
                swiflf = self._get_internal_flow_terms_tuple_flf(bud)

            else:
                print('Budget item', recname, 'not recognized.')
                # break

        # Format internal flow output
        q_in = {z: OrderedDict([('flow_dir', 'in'), ('record', 'FROM ZONE{: >4d}'.format(z))]) for z in self.zones}
        for k, v in q_in.iteritems():
            for z in self.zones:
                v[z] = 0.

        q_tups = sorted(frf + fff + flf + swifrf + swifff + swiflf)
        for f2z, gp in groupby(q_tups, lambda tup: tup[:2]):
            gpq = [i[-1] for i in list(gp)]
            q_in[f2z[0]][f2z[1]] = np.sum(gpq)

        for k, v in q_in.iteritems():
            inflows.append(tuple(v.values()))

        q_out = {z: OrderedDict([('flow_dir', 'out'), ('record', 'TO ZONE{: >4d}'.format(z))]) for z in self.zones}
        for k, v in q_out.iteritems():
            for z in self.zones:
                v[z] = 0.

        q_tups = sorted(frf + fff + flf + swifrf + swifff + swiflf)
        for f2z, gp in groupby(q_tups, lambda tup: tup[:2]):
            gpq = [i[-1] for i in list(gp)]
            q_out[f2z[1]][f2z[0]] = np.sum(gpq)

        for k, v in q_out.iteritems():
            outflows.append(tuple(v.values()))
        q = inflows + outflows
        q = Budget(np.array(q, dtype=self.dtype), kstpkper=kstpkper, totim=totim)
        return q

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

    def _get_source_sink_storage_terms_tuple(self, recname, bud):
        rec_inflow = ['in', recname.strip()] + [bud[(bud >= 0.) & (self.izone == z)].sum() for z in self.zones]
        rec_outflow = ['out', recname.strip()] + [bud[(bud < 0.) & (self.izone == z)].sum()*-1 for z in self.zones]
        rec_inflow = [val if not type(val) == np.ma.core.MaskedConstant else 0. for val in rec_inflow]
        rec_outflow = [val if not type(val) == np.ma.core.MaskedConstant else 0. for val in rec_outflow]
        return tuple(rec_inflow), tuple(rec_outflow)

    def _get_internal_flow_terms_tuple_frf(self, bud):

        assert self.ncol >= 2, 'Must have more than 2 columns to accumulate FLOW RIGHT FACE record'
        # ACCUMULATE FLOW BETWEEN ZONES ACROSS COLUMNS. COMPUTE FLOW ONLY BETWEEN A ZONE
        # AND A HIGHER ZONE -- FLOW FROM ZONE 4 TO 3 IS THE NEGATIVE OF FLOW FROM 3 TO 4.
        # FIRST, CALCULATE FLOW BETWEEN NODE J,I,K AND J-1,I,K.
        # Accumulate flow from lower zones to higher zones from "left" to "right".
        # Flow into the higher zone will be <0 Flow Right Face from the adjacent cell to the "left".
        nz = self.izone[:, :, 1:]
        nzl = self.izone[:, :, :-1]
        l, r, c = np.where(nz > nzl)

        # Adjust column values to account for the starting position of "nz"
        c += 1

        # Define the zone from which flow is coming
        from_zones = self.izone[l, r, c-1]

        # Define the zone to which flow is going
        to_zones = self.izone[l, r, c]

        # Get the face flow
        q = bud[l, r, c-1]

        # Get indices where flow face values are negative (flow into higher zone)
        idx_neg = np.where(q < 0)

        # Get indices where flow face values are positive (flow out of higher zone)
        idx_pos = np.where(q >= 0)

        neg = zip(to_zones[idx_neg], from_zones[idx_neg], q[idx_neg]*-1)
        pos = zip(from_zones[idx_pos], to_zones[idx_pos], q[idx_pos])
        nzgt_l2r = neg + pos

        # CALCULATE FLOW BETWEEN NODE J,I,K AND J+1,I,K.
        # Accumulate flow from lower zones to higher zones from "right" to "left".
        # Flow into the higher zone will be <0 Flow Right Face from the adjacent cell to the "left".
        nz = self.izone[:, :, :-1]
        nzr = self.izone[:, :, 1:]
        l, r, c = np.where(nz > nzr)

        # Define the zone from which flow is coming
        from_zones = self.izone[l, r, c]

        # Define the zone to which flow is going
        to_zones = self.izone[l, r, c+1]

        # Get the face flow
        q = bud[l, r, c]

        # Get indices where flow face values are negative (flow into higher zone)
        idx_neg = np.where(q < 0)

        # Get indices where flow face values are positive (flow out of higher zone)
        idx_pos = np.where(q >= 0)

        neg = zip(to_zones[idx_neg], from_zones[idx_neg], q[idx_neg]*-1)
        pos = zip(from_zones[idx_pos], to_zones[idx_pos], q[idx_pos])
        nzgt_r2l = neg + pos

        # Accumulate flow for constant head cells

        nzgt = sorted(nzgt_l2r + nzgt_r2l, key=lambda tup: tup[:2])
        return nzgt

    def _get_internal_flow_terms_tuple_fff(self, bud):

        assert self.nrow >= 2, 'Must have more than 2 rows to accumulate FLOW FRONT FACE record'
        # ACCUMULATE FLOW BETWEEN ZONES ACROSS ROWS. COMPUTE FLOW ONLY BETWEEN A ZONE
        #  AND A HIGHER ZONE -- FLOW FROM ZONE 4 TO 3 IS THE NEGATIVE OF FLOW FROM 3 TO 4.
        # FIRST, CALCULATE FLOW BETWEEN NODE J,I,K AND J,I-1,K.
        # Accumulate flow from lower zones to higher zones from "up" to "down".
        nz = self.izone[:, 1:, :]
        nzu = self.izone[:, :-1, :]
        l, r, c = np.where(nz < nzu)
        # Adjust column values by +1 to account for the starting position of "nz"
        r += 1

        # Define the zone from which flow is coming
        from_zones = self.izone[l, r-1, c]

        # Define the zone to which flow is going
        to_zones = self.izone[l, r, c]

        # Get the face flow
        q = bud[l, r-1, c]

        # Get indices where flow face values are negative (flow into higher zone)
        idx_neg = np.where(q < 0)

        # Get indices where flow face values are positive (flow out of higher zone)
        idx_pos = np.where(q >= 0)

        neg = zip(to_zones[idx_neg], from_zones[idx_neg], q[idx_neg]*-1)
        pos = zip(from_zones[idx_pos], to_zones[idx_pos], q[idx_pos])
        nzgt_u2d = neg + pos

        # CALCULATE FLOW BETWEEN NODE J,I,K AND J,I+1,K.
        # Accumulate flow from lower zones to higher zones from "down" to "up".
        nz = self.izone[:, :-1, :]
        nzd = self.izone[:, 1:, :]
        l, r, c = np.where(nz < nzd)

        # Define the zone from which flow is coming
        from_zones = self.izone[l, r, c]

        # Define the zone to which flow is going
        to_zones = self.izone[l, r+1, c]

        # Get the face flow
        q = bud[l, r, c]

        # Get indices where flow face values are negative (flow into higher zone)
        idx_neg = np.where(q < 0)

        # Get indices where flow face values are positive (flow out of higher zone)
        idx_pos = np.where(q >= 0)

        neg = zip(to_zones[idx_neg], from_zones[idx_neg], q[idx_neg]*-1)
        pos = zip(from_zones[idx_pos], to_zones[idx_pos], q[idx_pos])
        nzgt_d2u = neg + pos
        nzgt = sorted(nzgt_u2d + nzgt_d2u, key=lambda tup: tup[:2])
        return nzgt

    def _get_internal_flow_terms_tuple_flf(self, bud):

        assert self.nlay >= 2, 'Must have more than 2 layers to accumulate FLOW LOWER FACE record'
        # ACCUMULATE FLOW BETWEEN ZONES ACROSS LAYERS. COMPUTE FLOW ONLY BETWEEN A ZONE
        #  AND A HIGHER ZONE -- FLOW FROM ZONE 4 TO 3 IS THE NEGATIVE OF FLOW FROM 3 TO 4.
        # FIRST, CALCULATE FLOW BETWEEN NODE J,I,K AND J,I,K-1.
        # Accumulate flow from lower zones to higher zones from "top" to "bottom".
        nz = self.izone[1:, :, :]
        nzt = self.izone[:-1, :, :]
        l, r, c = np.where(nz > nzt)
        # Adjust column values by +1 to account for the starting position of "nz"
        l += 1

        # Define the zone from which flow is coming
        from_zones = self.izone[l-1, r, c]

        # Define the zone to which flow is going
        to_zones = self.izone[l, r, c]

        # Get the face flow
        q = bud[l-1, r, c]

        # Get indices where flow face values are negative (flow into higher zone)
        idx_neg = np.where(q < 0)

        # Get indices where flow face values are positive (flow out of higher zone)
        idx_pos = np.where(q >= 0)

        neg = zip(to_zones[idx_neg], from_zones[idx_neg], q[idx_neg]*-1)
        pos = zip(from_zones[idx_pos], to_zones[idx_pos], q[idx_pos])
        nzgt_t2b = neg + pos

        # CALCULATE FLOW BETWEEN NODE J,I,K AND J+1,I,K.
        # Accumulate flow from lower zones to higher zones from "right" to "left".
        # Flow into the higher zone will be <0 Flow Right Face from the adjacent cell to the "left".
        nz = self.izone[:-1, :, :]
        nzb = self.izone[1:, :, :]
        l, r, c = np.where(nz < nzb)

        # Define the zone from which flow is coming
        from_zones = self.izone[l, r, c]

        # Define the zone to which flow is going
        to_zones = self.izone[l+1, r, c]

        # Get the face flow
        q = bud[l, r, c]

        # Get indices where flow face values are negative (flow into higher zone)
        idx_neg = np.where(q < 0)

        # Get indices where flow face values are positive (flow out of higher zone)
        idx_pos = np.where(q >= 0)

        neg = zip(to_zones[idx_neg], from_zones[idx_neg], q[idx_neg]*-1)
        pos = zip(from_zones[idx_pos], to_zones[idx_pos], q[idx_pos])
        nzgt_b2t = neg + pos
        nzgt = sorted(nzgt_t2b + nzgt_b2t, key=lambda tup: tup[:2])
        return nzgt

    @staticmethod
    def _find_unique_zones(a):
        z = [int(i) for i in np.unique(a)]
        return z


def run_zonbud(zonarray, cbcfile='modflowtest.cbc', listingfile_prefix='zbud', zonbud_ws='.',
               zonbud_exe='zonbud.exe', title='ZoneBudget Test', budget_option='A', kstpkper=None,
               iprn=-1, silent=True):
    """

    Parameters
    ----------
    zonarray : array of ints (nlay, nrow, ncol)
        integer-array of zone numbers
    cbcfile : str
        name of the cell-by-cell budget file
    listingfile_prefix : str
        name of the listingfile
    zonbud_ws : str
        directory where ZoneBudget output will be stored
    zonbud_exe : str
        name of the ZoneBudget executable
    title : str
        title to be printed in the listing file
    budget_option : str
        must be one of "A" (for all timesteps) or "L" for a user-specified list of timesteps
    kstpkper : list
        list of zero-based timestep/stress periods for which budgets will be calculated
    iprn : integer
        specifies whether or not the zone values are printed in the output file
        if less than zero, zone values will not be printed

    Returns
    -------
    zbud, an ordered dictionary of recarrays.

    Examples
    -------
    >>> import flopy
    >>> zbud = flopy.utils.run_zonbud(zonarray, cbcfile='modflowtest.cbc')
    """

    # Need to catch some errors early on, ZoneBudget likes to crash without any feedback
    # Locked output files, non-existent input, etc.
    #
    assert os.path.isfile(cbcfile), 'Cell by cell budget file is not a file {}'.format(cbcfile)
    assert budget_option.upper() in ['A', 'L'], 'Please enter a valid budget option ("A" for all or "L" for' \
                                                ' a list of times).'
    listingfile_prefix = listingfile_prefix.split('.')[0]
    zonfile = os.path.join(zonbud_ws, listingfile_prefix + '.zon')
    listingfile = os.path.join(zonbud_ws, listingfile_prefix + ' csv')
    zbud_file = os.path.join(zonbud_ws, listingfile_prefix + '.csv')
    args = [listingfile, cbcfile, title, zonfile, budget_option]
    if budget_option == 'L':
        assert kstpkper is not None, 'You have chosen budget option "L", please enter a ' \
                                     'list of times. For example (0, 0) for timestep 1 ' \
                                     'of stress period 1.'
        kstpkper_args = ['{kstp},{kper}'.format(kstp=kk[0]+1, kper=kk[1]+1) for kk in kstpkper]
        kstpkper_args.append('0,0')
        args += kstpkper_args

    if not os.path.isfile(cbcfile):
        s = 'The cell by cell budget file for this model does not exists: {}'.format(cbcfile)
        raise Exception(s)

    _write_zonfile(zonarray, zonfile, iprn)
    _call(zonbud_exe, args, zonbud_ws, silent)
    zbud = _parse_zbud_file(zbud_file)
    return zbud


def _parse_zbud_file(zf):
    assert os.path.isfile(zf), 'Output zonebudget file {} does not exist or cannot be read.'.format(zf)
    kstpkper = []
    ins = OrderedDict()
    outs = OrderedDict()
    ins_flag = False
    outs_flag = False
    with open(zf) as f:
        for line in f:
            line_items = [i.strip() for i in line.split(',')]
            if line_items[0] == 'Time Step':
                kk = (int(line_items[1])-1, int(line_items[3])-1)
                kstpkper.append(kk)
                ins[kk] = []
                outs[kk] = []
            elif 'ZONE' in line_items[1]:
                zones = [z for z in line_items if z != '']
                col_header = ['Record Name'] + zones
                dtype = [('flow_dir', '|S3'), ('record', '|S20')] + \
                        [(col_name, np.float32) for col_name in col_header[1:]]
            elif line_items[1] == 'IN':
                ins_flag = True
                continue
            elif line_items[0] == 'Total IN':
                ins_flag = False
            elif line_items[1] == 'OUT':
                outs_flag = True
                continue
            elif line_items[0] == 'Total OUT':
                outs_flag = False
            if ins_flag:
                z = [x for x in line_items if x != '']
                z.insert(0, 'in')
                z[2:] = [float(zz) for zz in z[2:]]
                ins[kk].append(tuple(z))
            elif outs_flag:
                z = [x for x in line_items if x != '']
                z.insert(0, 'out')
                z[2:] = [float(zz) for zz in z[2:]]
                outs[kk].append(tuple(z))
    zbud = OrderedDict()
    for kk in kstpkper:
        try:
            dat = ins[kk] + outs[kk]
            zbud[kk] = np.array(dat, dtype=dtype)
        except Exception as e:
            print(e)
            return None
    return zbud


def _write_zonfile(izone, zonfile, iprn):
    assert 'int' in str(izone.dtype), 'Input zone array (dtype={}) must be an integer array.'.format(izone.dtype)
    if len(izone.shape) == 2:
        nlay = 1
        nrow, ncol = izone.shape
        z = np.zeros((nlay, nrow, ncol))
        z[0, :, :] = izone
        izone = z
    elif len(izone.shape) == 3:
        nlay, nrow, ncol = izone.shape

    with open(zonfile, 'w') as f:
        f.write('{} {} {}\n'.format(nlay, nrow, ncol))

        for lay in range(nlay):
            f.write('INTERNAL ({ncol}I4) {iprn}\n'.format(ncol=ncol, iprn=iprn))
            for row in range(nrow):
                f.write(''.join(['{:4d}'.format(int(val)) for val in izone[lay, row, :]])+'\n')

        #     f.write('INTERNAL (free) {iprn}\n'.format(iprn=iprn))
        #     for row in range(nrow):
        #         f.write(' '.join(['{:d}'.format(int(val)) for val in izone[lay, row, :]])+'\n')
        # f.write('ALLZONES ' + ' '.join([str(int(z)) for z in np.unique(izone)])+'\n')
    return


def is_exe(fpath):
    """
    Taken from flopy.mbase

    """
    return os.path.isfile(fpath) and os.access(fpath, os.X_OK)


def which(program):
    """
    Taken from flopy.mbase

    """
    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        # test for exe in current working directory
        if is_exe(program):
            return program
        # test for exe in path statement
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file
    return None


def _call(exe_name, args, zonbud_ws='./', silent=True):

    # success = False
    # buff = []

    # Check to make sure that program and namefile exist
    exe = which(exe_name)
    if exe is None:
        import platform

        if platform.system() in 'Windows':
            if not exe_name.lower().endswith('.exe'):
                exe = which(exe_name + '.exe')
    if exe is None:
        s = 'The program {} does not exist or is not executable.'.format(
            exe_name)
        raise Exception(s)
    else:
        if not silent:
            s = 'FloPy is using the following executable to run ZoneBudget: {}'.format(
                exe)
            print(s)

    # simple little function for the thread to target
    def q_output(output,q):
            for line in iter(output.readline,b''):
                q.put(line)
            #time.sleep(1)
            #output.close()
    argsstr = ''.join([arg+os.linesep for arg in args])
    proc = sp.Popen([exe_name], stdin=sp.PIPE, stdout=sp.PIPE, cwd=zonbud_ws)
    stdout = proc.communicate(input=argsstr)[0]
    if not silent:
        print(stdout)
    while True:
        line = proc.stdout.readline()
        c = line.decode('utf-8')
        if c != '':
            c = c.rstrip('\r\n')
            # if report == True:
                # buff.append(c)
            if not silent:
                print(c)
        else:
            # success = True
            break
    return
