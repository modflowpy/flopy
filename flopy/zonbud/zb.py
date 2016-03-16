from __future__ import print_function
import os
from collections import OrderedDict
from itertools import groupby
import flopy
from flopy.pakbase import Package
from flopy.utils.binaryfile import CellBudgetFile
from flopy.utils import Util3d
# from ..pakbase import Package
# from ..utils.binaryfile import CellBudgetFile
# from ..utils import Util2d, Util3d
import numpy as np
import pandas as pd
from datetime import datetime
# from matplotlib import pyplot as plt


class ZoneBudget(Package):
    """
    ZoneBudget package class

    Example usage:

    import flopy
    from flopy.zonebud import ZoneBudget

    ml = flopy.modflow.Modflow('zonebudtest.nam')
    dis = flopy.modflow.ModflowDis.load('zonebudtest.dis'), ml, check=False)
    zb = ZoneBudget(ml, 'zonebudtest.cbc', 'GWBasins.zon', swi=True)
    """
    def __init__(self, model, cbc_file, z, out_file='zonebudtest.txt',
                 zone_to_ignore=None, kstpkper=(0, 0), swi=False, extension='zb', unitnumber=15):

        """
        Package constructor.

        """
        start = datetime.now()
        Package.__init__(self, model, extension, 'ZB',
                         unitnumber)  # Call ancestor's init to set self.parent, extension, name and unit number
        self.nrow, self.ncol, self.nlay, self.nper = self.parent.nrow_ncol_nlay_nper
        self.unit_number = unitnumber

        # INTERNAL FLOW TERMS ARE USED TO CALCULATE FLOW BETWEEN ZONES.
        # CONSTANT-HEAD TERMS ARE USED TO IDENTIFY WHERE CONSTANT-HEAD CELLS ARE AND THEN USE
        #  FACE FLOWS TO DETERMINE THE AMOUNT OF FLOW.
        # SWIADDTO* terms are used by the SWI2 package.
        self.internal_flow_terms = ['CONSTANT HEAD', 'FLOW RIGHT FACE', 'FLOW FRONT FACE', 'FLOW LOWER FACE',
                                    'SWIADDTOCH', 'SWIADDTOFRF', 'SWIADDTOFFF', 'SWIADDTOFLF']

        # OPEN THE CELL-BY-CELL BUDGET BINARY FILE
        start_open_cbc = datetime.now()
        cbc = CellBudgetFile(cbc_file)
        end_open_cbc = datetime.now()
        print('Time to open cbc file', end_open_cbc-start_open_cbc)

        # for rec in cbc.recordarray:
        #     kstp, kper, text, ncol, nrow, nlay, imeth, delt, pertim, totim = rec
            # print(kstp, text)

        # GET A LISTING OF THE UNIQUE RECORDS CONTAINED IN THE BUDGET FILE
        self.record_names = cbc.unique_record_names()

        # GET DATA FOR ALL RECORDS
        # self.cbc_data = cbc.get_data(idx=range(len(self.record_names)), kstpkper=kstpkper, full3D=True)
        start_read_cbc = datetime.now()


        # Lets interrogate the budget file in a more systematic way. Read each non-internal flow record individually.
        # Then lets read each internal flow record individually, starting with Constant Head so that we can be sure
        # that theses cells have been identified when we need to aggregate them with the face flows.


        self.ssst_data = OrderedDict([(r, cbc.get_data(text=r, kstpkper=kstpkper, full3D=True)[0])
                                      for idx, r in enumerate(self.record_names)
                                      if r.strip() not in self.internal_flow_terms])
        self.internal_flow_data = OrderedDict([(r, cbc.get_data(text=r, kstpkper=kstpkper, full3D=True)[0])
                                               for idx, r in enumerate(self.record_names)
                                               if r.strip() in self.internal_flow_terms])
        end_read_cbc = datetime.now()
        cbctime = end_read_cbc-start_read_cbc
        print('Time to read cbc file', cbctime)

        # OPEN THE ZONE FILE (OR ARRAY) AND FIND THE UNIQUE SET OF ZONES CONTAINED THEREIN
        start_read_izone = datetime.now()
        izone = Util3d(model, (self.nlay, self.nrow, self.ncol),
                       np.int32, z, 'zonebud', locat=self.unit_number)
        self.izone = izone.array
        self.zones = self._find_unique_zones(self.izone.ravel(), ignore_value=zone_to_ignore)
        end_read_izone = datetime.now()
        print('Time to read izone', end_read_izone-start_read_izone)

        # Define dtype for the recarray
        self.dtype = np.dtype([('flow_dir', '|S3'), ('record', '|S20')] +
                               [('ZONE{: >4d}'.format(z), np.float32) for z in self.zones])

        # Accumulte source/sink/storage terms by zone
        start_ssst_inflows = datetime.now()
        self.ssst_inflows, self.ssst_outflows = self._get_source_sink_storage_terms()
        end_ssst_inflows = datetime.now()
        print('Time to read ssst items', end_ssst_inflows-start_ssst_inflows)



        # PROCESS EACH INTERNAL FLOW RECORD IN THE CELL-BY-CELL BUDGET FILE
        for recname, bud in self.internal_flow_data.iteritems():

            # IF RECORD IS A CONSTANT-HEAD INTERNAL FLOW TERM, ACCUMULATE FACE FLOWS ONLY FOR
            #  CONSTANT-HEAD CELLS
            # ----not yet tested----#
            if recname.strip() == 'CONSTANT HEAD':
                self.ich_lrc = bud[bud != 0]

            elif recname.strip() == 'FLOW RIGHT FACE':
                start_frf = datetime.now()
                frf = self._get_internal_flow_terms_frf(bud)
                end_frf = datetime.now()
                print('Time to compute frf', end_frf-start_frf)

            elif recname.strip() == 'FLOW FRONT FACE':
                start_fff = datetime.now()
                fff = self._get_internal_flow_terms_fff(bud)
                end_fff = datetime.now()
                print('Time to compute fff', end_fff-start_fff)

            elif recname.strip() == 'FLOW LOWER FACE':
                start_flf = datetime.now()
                flf = self._get_internal_flow_terms_flf(bud)
                end_flf = datetime.now()
                print('Time to compute flf', end_flf-start_flf)

            else:
                print('Budget item', recname, 'not recognized.')
                # break

        q_tups = sorted(frf + fff + flf)
        for f2z, gp in groupby(q_tups, lambda tup: tup[:2]):
            q = np.array([i[-1] for i in list(gp)])
            # print(f2z, np.sum(q))

        totim = datetime.now()-start
        print('Total time elapsed', totim)

    def _get_source_sink_storage_terms(self):

        # IF RECORD IS NOT AN INTERNAL FLOW TERM, ACCUMULATE FLOW BY ZONE.
        inflows = []
        outflows = []
        for recname, bud in self.ssst_data.iteritems():
            rec_inflow = ['in', recname.strip()] + [bud[(bud >= 0.) & (self.izone == z)].sum() for z in self.zones]
            rec_outflow = ['out', recname.strip()] + [bud[(bud < 0.) & (self.izone == z)].sum() for z in self.zones]
            # for z in self.zones:
            #     rec_inflow.append(bud[(bud >= 0.) & (self.izone == z)].sum())
            #     rec_outflow.append(bud[(bud < 0.) & (self.izone == z)].sum())
            # rec_inflow = ['in', recname.strip()] + [bud[(bud >= 0.) & (self.izone == z)].sum() for z in self.zones]
            inflows.append(tuple(rec_inflow))
            outflows.append(tuple(rec_outflow))
        ssst_inflows = np.array(inflows, dtype=self.dtype)
        ssst_outflows = np.array(outflows, dtype=self.dtype)
        #     zone_inflows = pd.Series(data=totals_in, index=index, name='Zone {}'.format(z))
        #     zone_outflows = pd.Series(data=totals_out, index=index, name='Zone {}'.format(z))
        #     inflows.append(zone_inflows)
        #     outflows.append(zone_outflows*-1)
        #
        # inflows = pd.concat(inflows, axis=1)
        # outflows = pd.concat(outflows, axis=1)
        return ssst_inflows, ssst_outflows

    def _get_internal_flow_terms_frf(self, bud):

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
        nzgt = sorted(nzgt_l2r + nzgt_r2l, key=lambda tup: tup[:2])
        return nzgt

    def _get_internal_flow_terms_fff(self, bud):

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

    def _get_internal_flow_terms_flf(self, bud):

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
    def _find_unique_zones(a, ignore_value=None):
        z = [i for i in np.unique(a) if i != ignore_value]
        return z

    # def list_internal_flow_records(self):
    #     for rec, bud in self.records.iteritems():
    #         if 'From Zone' in rec:
    #             print(rec, np.nansum(bud))
    #
    # def list_inflows(self):
    #     for rec, buds in self.records.iteritems():
    #         if rec.strip() not in self.internal_flow_terms:
    #             for z in self.zones:
    #                 print(rec, 'Zone', z, 'IN', buds[z][buds[z] >= 0.].sum())
    #         else:
    #             print(rec, buds[z].sum())
    #
    # def get_inflows(self):
    #     flows = {}
    #     for rec, buds in self.records.iteritems():
    #         flows[rec] = {}
    #         for z in self.zones:
    #             flows[rec][z] = buds[z][buds[z] >= 0.]
    #     return flows
    #
    # def get_outflows(self):
    #     flows = {}
    #     for rec, buds in self.records.iteritems():
    #         flows[rec] = {}
    #         for z in self.zones:
    #             flows[rec][z] = buds[z][buds[z] < 0.]
    #     return flows

if __name__ == '__main__':
    loadpth = r'testing\model'
    # ml = flopy.modflow.Modflow.load('fas.nam', model_ws=loadpth, check=False)
    ml = flopy.modflow.Modflow('zonebudtest.nam', model_ws=loadpth)
    dis = flopy.modflow.ModflowDis.load(os.path.join(loadpth, 'fas.dis'), ml, check=False)
    zon = np.loadtxt(os.path.join('testing', 'GWBasins.zon'))
    zb = ZoneBudget(ml, os.path.join(loadpth, 'fas.cbc'), zon, swi=True)
    # zb.list_internal_flow_records()
    # outflows = zb.get_outflows()
    # for rec, buds in outflows.iteritems():
    #     for z, outflow in buds.iteritems():
    #         print(rec, 'Zone', z, 'OUT', outflow.sum())