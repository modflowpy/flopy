from __future__ import print_function, division
import os
from collections import OrderedDict
from itertools import groupby
import numpy as np
from ..utils.binaryfile import CellBudgetFile


class ZoneBudget(object):
    """
    ZoneBudget package class

    Example usage:

    >>>import flopy
    >>>from flopy.zonebud import ZoneBudget
    >>>zb = ZoneBudget('zonebudtest.cbc', 'GWBasins.zon')
    >>>zbud = zb.get_bud()
    >>>zb.to_csv()
    """
    def __init__(self, cbc_file, zon, kstpkper=(0, 0), out_file='zonebudtest.txt'):

        # INTERNAL FLOW TERMS ARE USED TO CALCULATE FLOW BETWEEN ZONES.
        # CONSTANT-HEAD TERMS ARE USED TO IDENTIFY WHERE CONSTANT-HEAD CELLS ARE AND THEN USE
        #  FACE FLOWS TO DETERMINE THE AMOUNT OF FLOW.
        # SWIADDTO* terms are used by the SWI2 package.
        self.internal_flow_terms = ['CONSTANT HEAD', 'FLOW RIGHT FACE', 'FLOW FRONT FACE', 'FLOW LOWER FACE',
                                    'SWIADDTOCH', 'SWIADDTOFRF', 'SWIADDTOFFF', 'SWIADDTOFLF']

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

        # OPEN THE CELL-BY-CELL BUDGET BINARY FILE
        cbc = CellBudgetFile(cbc_file)

        # GET A LISTING OF THE UNIQUE RECORDS CONTAINED IN THE BUDGET FILE
        self.record_names = cbc.unique_record_names()

        # Get dimensions of budget file arrays
        self.kstpkper = kstpkper
        self.nlay, self.nrow, self.ncol = cbc.get_data(idx=0, kstpkper=self.kstpkper, full3D=True)[0].shape
        assert self.izone.shape == (self.nlay, self.nrow, self.ncol), \
            'Shape of input zone array {} does not' \
            ' match the cell by cell' \
            'budget file {}'.format(self.izone.shape, (self.nlay, self.nrow, self.ncol))


        # Lets interrogate the budget file in a more systematic way. Read each non-internal flow record individually.
        # Then lets read each internal flow record individually, starting with Constant Head so that we can be sure
        # that theses cells have been identified when we need to aggregate them with the face flows.

        # Accumulate source/sink/storage terms by zone
        inflows = []
        outflows = []
        ssst_records = [rec for rec in self.record_names if rec.strip() not in self.internal_flow_terms]
        for recname in ssst_records:
            bud = cbc.get_data(text=recname, kstpkper=kstpkper, full3D=True)[0]
            in_tup, out_tup = self._get_source_sink_storage_terms_tuple(recname, bud)
            inflows.append(in_tup)
            outflows.append(out_tup)

        # IF RECORD IS A CONSTANT-HEAD INTERNAL FLOW TERM, ACCUMULATE FACE FLOWS ONLY FOR
        #  CONSTANT-HEAD CELLS
        # ----not yet tested----#
        bud = cbc.get_data(text='   CONSTANT HEAD', kstpkper=kstpkper, full3D=True)[0]
        self.ich_lrc = bud[bud != 0]

        if 'SWIADDTOCH' in [r.strip() for r in self.record_names]:
            bud = cbc.get_data(text='      SWIADDTOCH', kstpkper=kstpkper, full3D=True)[0]
            self.ichswi_lrc += bud[bud != 0]
        else:
            self.ichswi_lrc = []

        # PROCESS EACH INTERNAL FLOW RECORD IN THE CELL-BY-CELL BUDGET FILE
        frf, fff, flf, swifrf, swifff, swiflf = [], [], [], [], [], []
        for recname in [r for r in self.record_names if r.strip() in self.internal_flow_terms]:

            if recname.strip() == 'CONSTANT HEAD':
                continue
            elif recname.strip() == 'SWIADDTOCH':
                continue

            elif recname.strip() == 'FLOW RIGHT FACE':
                bud = cbc.get_data(text=recname, kstpkper=kstpkper, full3D=True)[0]
                frf = self._get_internal_flow_terms_tuple_frf(bud)

            elif recname.strip() == 'FLOW FRONT FACE':
                bud = cbc.get_data(text=recname, kstpkper=kstpkper, full3D=True)[0]
                fff = self._get_internal_flow_terms_tuple_fff(bud)

            elif recname.strip() == 'FLOW LOWER FACE':
                bud = cbc.get_data(text=recname, kstpkper=kstpkper, full3D=True)[0]
                flf = self._get_internal_flow_terms_tuple_flf(bud)

            elif recname.strip() == 'SWIADDTOFRF':
                bud = cbc.get_data(text=recname, kstpkper=kstpkper, full3D=True)[0]
                swifrf = self._get_internal_flow_terms_tuple_frf(bud)

            elif recname.strip() == 'SWIADDTOFFF':
                bud = cbc.get_data(text=recname, kstpkper=kstpkper, full3D=True)[0]
                swifff = self._get_internal_flow_terms_tuple_fff(bud)

            elif recname.strip() == 'SWIADDTOFLF':
                bud = cbc.get_data(text=recname, kstpkper=kstpkper, full3D=True)[0]
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
            if 0 not in f2z:
                # Ignore zone 0
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
            if 0 not in f2z:
                # Ignore zone 0
                gpq = [i[-1] for i in list(gp)]
                q_out[f2z[1]][f2z[0]] = np.sum(gpq)

        for k, v in q_out.iteritems():
            outflows.append(tuple(v.values()))
        q = inflows + outflows
        self.q = np.array(q, dtype=self.dtype)

    def get_bud(self):
        return self.q

    def to_csv(self, fname='zbud.csv', format='pandas'):

        assert format.lower() in ['pandas', 'zondbud'], 'Format must be one of "pandas" or "zonbud".'
        # List the field names to be used to slice the recarray
        fields = ['ZONE{: >4d}'.format(z) for z in self.zones]

        ins_idx = np.where(self.q['flow_dir'] == 'in')[0]
        out_idx = np.where(self.q['flow_dir'] == 'out')[0]

        ins = self._fields_view(self.q[ins_idx], fields)
        out = self._fields_view(self.q[out_idx], fields)

        ins_sum = ins.sum(axis=0)
        out_sum = out.sum(axis=0)

        if format.lower() == 'pandas':
            with open(fname, 'w') as f:

                # Write header
                f.write(','.join([field for field in self.dtype.names])+'\n')

                # Write IN terms
                for rec in self.q[ins_idx]:
                    f.write(','.join([str(i) for i in rec])+'\n')
                f.write(','.join([' ', 'Total IN'] + [str(i) for i in ins_sum])+'\n')

                # Write OUT terms
                for rec in self.q[out_idx]:
                    f.write(','.join([str(i) for i in rec])+'\n')
                f.write(','.join([' ', 'Total OUT'] + [str(i) for i in out_sum])+'\n')

                # Write mass balance terms
                dif = ins_sum-out_sum
                f.write(','.join([' ', 'IN-OUT'] + [str(i) for i in dif])+'\n')
                pcterr = 100*(ins_sum-out_sum)/((ins_sum+out_sum)/2.)
                f.write(','.join([' ', 'Percent Error'] + [str(i) for i in pcterr])+'\n')

        if format.lower() == 'zonbud':
            with open(fname, 'w') as f:

                # Write header
                f.write(','.join(['Time Step', str(self.kstpkper[0]+1),
                                  'Stress Period', str(self.kstpkper[1]+1)])+'\n')
                f.write(','.join([' '] + [field for field in self.dtype.names[2:]])+'\n')

                # Write IN terms
                f.write(','.join([' '] + ['IN']*(len(self.dtype.names[1:])-1))+'\n')
                for rec in self.q[ins_idx]:
                    f.write(','.join([str(rec[i+1]) for i in range(len(self.dtype.names[1:]))])+'\n')
                f.write(','.join(['Total IN'] + [str(i) for i in ins_sum])+'\n')

                # Write OUT terms
                f.write(','.join([' '] + ['OUT']*(len(self.dtype.names[1:])-1))+'\n')
                for rec in self.q[out_idx]:
                    f.write(','.join([str(rec[i+1]) for i in range(len(self.dtype.names[1:]))])+'\n')
                f.write(','.join(['Total OUT'] + [str(i) for i in out_sum])+'\n')

                # Write mass balance terms
                dif = ins_sum-out_sum
                f.write(','.join(['IN-OUT'] + [str(i) for i in dif])+'\n')
                pcterr = 100*(ins_sum-out_sum)/((ins_sum+out_sum)/2.)
                f.write(','.join(['Percent Error'] + [str(i) for i in pcterr])+'\n')

        # print('Total IN', ins_sum)
        # print('Total OUT', out_sum)
        # print('IN-OUT', ins_sum-out_sum)
        # print('Percent Error', 100*(ins_sum-out_sum)/((ins_sum+out_sum)/2.))

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
        z = [int(i) for i in np.unique(a) if int(i) != 0]
        return z
