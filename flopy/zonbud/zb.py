from __future__ import print_function
from collections import OrderedDict
from ..pakbase import Package
from ..utils.binaryfile import CellBudgetFile
from ..utils import Util2d, Util3d
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
        self.ssst_data = OrderedDict([(r, cbc.get_data(text=r, kstpkper=kstpkper, full3D=True)[0])
                                      for idx, r in enumerate(self.record_names)
                                      if r.strip() not in self.internal_flow_terms])
        self.internal_flow_data = OrderedDict([(r, cbc.get_data(text=r, kstpkper=kstpkper, full3D=True)[0])
                                               for idx, r in enumerate(self.record_names)
                                               if r.strip() in self.internal_flow_terms])
        end_read_cbc = datetime.now()

        # OPEN THE ZONE FILE (OR ARRAY) AND FIND THE UNIQUE SET OF ZONES CONTAINED THEREIN
        start_read_izone = datetime.now()
        izone = Util3d(model, (self.nlay, self.nrow, self.ncol),
                       np.int32, z, 'zonebud', locat=self.unit_number)
        self.izone = izone.array
        self.zones = self._find_unique_zones(self.izone.ravel(), ignore_value=zone_to_ignore)
        end_read_izone = datetime.now()
        print('Time to read izone', end_read_izone-start_read_izone)

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
                pass

            elif recname.strip() == 'FLOW RIGHT FACE':
                start_frf = datetime.now()
                frf_inflows, frf_outflows = self._get_internal_flow_terms_frf(bud)
                end_frf = datetime.now()
                print('Time to compute frf', end_frf-start_frf)

            elif recname.strip() == 'FLOW FRONT FACE':
                start_fff = datetime.now()
                fff_inflows, fff_outflows = self._get_internal_flow_terms_fff(bud)
                end_fff = datetime.now()
                print('Time to compute fff', end_fff-start_fff)

            elif recname.strip() == 'FLOW LOWER FACE':
                start_flf = datetime.now()
                flf_inflows, flf_outflows = self._get_internal_flow_terms_flf(bud)
                end_flf = datetime.now()
                print('Time to compute flf', end_flf-start_flf)

            else:
                print('Budget item', recname, 'not recognized.')
                break

        self.internal_inflows = frf_inflows + fff_inflows + flf_inflows
        self.internal_outflows = frf_outflows + fff_outflows + flf_outflows
        print(frf_inflows)
        print(frf_outflows)



        totim = datetime.now()-start
        cbctime = end_read_cbc-start_read_cbc
        print('Total time elapsed', totim)
        print('Time to read cbc file', cbctime)

    def _get_source_sink_storage_terms(self):

        # IF RECORD IS NOT AN INTERNAL FLOW TERM, ACCUMULATE FLOW BY ZONE.
        inflows = []
        outflows = []
        for z in self.zones:
            index = []
            totals_in = []
            totals_out = []

            for recname, bud in self.ssst_data.iteritems():
                index.append(recname.strip())
                totals_in.append(bud[(bud >= 0.) & (self.izone == z)].sum())
                totals_out.append(bud[(bud < 0.) & (self.izone == z)].sum())

            zone_inflows = pd.Series(data=totals_in, index=index, name='Zone {}'.format(z))
            zone_outflows = pd.Series(data=totals_out, index=index, name='Zone {}'.format(z))
            inflows.append(zone_inflows)
            outflows.append(zone_outflows*-1)

        inflows = pd.concat(inflows, axis=1)
        outflows = pd.concat(outflows, axis=1)
        return inflows, outflows

    def _get_internal_flow_terms_frf(self, bud):

        assert self.ncol >= 2, 'Must have more than 2 columns to accumulate FLOW RIGHT FACE record'
        inflows = []
        outflows = []

        # ACCUMULATE FLOW BETWEEN ZONES ACROSS COLUMNS. COMPUTE FLOW ONLY BETWEEN A ZONE
        # AND A HIGHER ZONE -- FLOW FROM ZONE 4 TO 3 IS THE NEGATIVE OF FLOW FROM 3 TO 4.
        # FIRST, CALCULATE FLOW BETWEEN NODE J,I,K AND J-1,I,K.
        # Look for zone transition boundaries where zone number increases to the "right".
        # Flow into the higher zone will be <0 Flow Right Face from the adjacent cell to the "left".
        nz = self.izone[:, :, 1:]
        nzl = self.izone[:, :, :-1]
        l, r, c = np.where(nz > nzl)
        # Adjust column values by +1 to account for the starting position of "nz"
        c += 1

        # Define cells where we will accumulate flow
        nzgt = np.zeros((self.nlay, self.nrow, self.ncol), np.int32)
        nzgt[l, r, c] = 1

        # Define the zone from which flow is coming
        from_zones = np.zeros((self.nlay, self.nrow, self.ncol), np.int32)
        from_zones[l, r, c] = self.izone[l, r, c-1]

        # Define the zone to which flow is going
        to_zones = np.zeros((self.nlay, self.nrow, self.ncol), np.int32)
        to_zones[l, r, c] = self.izone[l, r, c]

        x = zip(from_zones[nzgt==1], to_zones[nzgt==1], bud[nzgt==1])
        print(zip(np.where(nzgt==1))[:5])
        print(x[:5])


        ##################

        # for z in self.zones:
        #     zone_bud = np.zeros_like(bud)
        #     zone_bud[(nzgt == 1) & (self.izone == z)] = bud[(nzgt == 1) & (self.izone == z)]
        #     index_from = []
        #     index_to = []
        #     totals_in = []
        #     # totals_out = []
        #     for zi in self.zones:
        #         index_from.append('From Zone {:3d}'.format(zi))
        #         totals_in.append(zone_bud[(from_zones == zi) & (zone_bud < 0)].sum())
        #         # totals_out.append(zone_bud[(from_zones == zi) & (zone_bud < 0)].sum())
        #
        #     zone_inflows = pd.Series(data=totals_in, index=index, name='Zone {}'.format(z))
        #     # zone_outflows = pd.Series(data=totals_out, index=index, name='Zone {}'.format(z))
        #     inflows.append(zone_inflows)
        #     # outflows.append(zone_outflows*-1)

        ###################

        # FLOW BETWEEN NODE J,I,K AND J+1,I,K.
        nz = self.izone[:, :, :-1]
        nzr = self.izone[:, :, 1:]
        l, r, c = np.where(nz > nzr)
        # # Adjust column values by -1 to account for the starting position of "nz"
        # c -= 1

        # Define the zone to which flow is going
        from_zones = np.zeros((self.nlay, self.nrow, self.ncol), np.int32)
        from_zones[l, r, c] = self.izone[l, r, c+1]

        # Define cells where we will accumulate flow
        nzlt = np.zeros((self.nlay, self.nrow, self.ncol), np.int32)
        nzlt[l, r, c] = 1

        for z in self.zones:
            zone_bud = np.zeros_like(bud)
            zone_bud[(nzlt == 1) & (self.izone == z)] = bud[(nzlt == 1) & (self.izone == z)]
            index = []
            totals_in = []
            # totals_out = []
            for zi in self.zones:
                index.append('To Zone {:3d}'.format(zi))
                totals_in.append(zone_bud[(from_zones == zi) & (zone_bud >= 0)].sum())
                # totals_out.append(zone_bud[(from_zones == zi) & (zone_bud < 0)].sum())

            zone_inflows = pd.Series(data=totals_in, index=index, name='Zone {}'.format(z))
            # zone_outflows = pd.Series(data=totals_out, index=index, name='Zone {}'.format(z))
            inflows.append(zone_inflows)
            # outflows.append(zone_outflows*-1)

        # CALCULATE FLOW TO CONSTANT-HEAD CELLS IN THIS DIRECTION.
        # ----not yet implemented----#

        inflows = pd.concat(inflows, axis=1)
        outflows = pd.concat(outflows, axis=1)
        return inflows, outflows

    def _get_internal_flow_terms_fff(self, bud):

        assert self.nrow >= 2, 'Must have more than 2 rows to accumulate FLOW FRONT FACE record'
        inflows = []
        outflows = []

        # ACCUMULATE FLOW BETWEEN ZONES ACROSS ROWS. COMPUTE FLOW ONLY BETWEEN A ZONE
        #  AND A HIGHER ZONE -- FLOW FROM ZONE 4 TO 3 IS THE NEGATIVE OF FLOW FROM 3 TO 4.
        # FIRST, CALCULATE FLOW BETWEEN NODE J,I,K AND J,I-1,K.
        nz = self.izone[:, 1:, :]
        nzl = self.izone[:, :-1, :]
        l, r, c = np.where(nz > nzl)
        # Adjust column values by +1 to account for the starting position of "nz"
        r += 1

        # Define the zone to which flow is going
        to_zones = np.zeros((self.nlay, self.nrow, self.ncol), np.int32)
        to_zones[l, r, c] = self.izone[l, r, c-1]

        # Define cells where we will accumulate flow
        nzgt = np.zeros((self.nlay, self.nrow, self.ncol), np.int32)
        nzgt[l, r, c] = 1

        for z in self.zones:
            zone_bud = np.zeros_like(bud)
            zone_bud[(nzgt == 1) & (self.izone == z)] = bud[(nzgt == 1) & (self.izone == z)]
            index = []
            # totals_in = []
            totals_out = []
            for zi in self.zones:
                index.append('To Zone {:3d}'.format(zi))
                # totals_in.append(zone_bud[(to_zones == zi) & (zone_bud >= 0)].sum())
                totals_out.append(zone_bud[(to_zones == zi) & (zone_bud < 0)].sum())

            # zone_inflows = pd.Series(data=totals_in, index=index, name='Zone {}'.format(z))
            zone_outflows = pd.Series(data=totals_out, index=index, name='Zone {}'.format(z))
            # inflows.append(zone_inflows)
            outflows.append(zone_outflows*-1)
        #
        # FLOW BETWEEN NODE J,I,K AND J,I+1,K.
        nz = self.izone[:, :-1, :]
        nzr = self.izone[:, 1:, :]
        l, r, c = np.where(nz < nzr)
        r -= 1  # Adjust column values by -1 to account for the starting position of "nz"

        # Define the zone from which flow is coming
        from_zones = np.zeros((self.nlay, self.nrow, self.ncol), np.int32)
        from_zones[l, r, c] = self.izone[l, r, c-1]

        # Define cells where we will accumulate flow
        nzlt = np.zeros((self.nlay, self.nrow, self.ncol), np.int32)
        nzlt[l, r, c] = 1

        for z in self.zones:
            zone_bud = np.zeros_like(bud)
            zone_bud[(nzlt == 1) & (self.izone == z)] = bud[(nzlt == 1) & (self.izone == z)]
            index = []
            totals_in = []
            # totals_out = []
            for zi in self.zones:
                index.append('From Zone {:3d}'.format(zi))
                totals_in.append(zone_bud[(from_zones == zi) & (zone_bud >= 0)].sum())
                # totals_out.append(zone_bud[(from_zones == zi) & (zone_bud < 0)].sum())

            zone_inflows = pd.Series(data=totals_in, index=index, name='Zone {}'.format(z))
            # zone_outflows = pd.Series(data=totals_out, index=index, name='Zone {}'.format(z))
            inflows.append(zone_inflows)
            # outflows.append(zone_outflows*-1)

        # CALCULATE FLOW TO CONSTANT-HEAD CELLS IN THIS DIRECTION.
        # ----not yet tested----#

        inflows = pd.concat(inflows, axis=1)
        outflows = pd.concat(outflows, axis=1)
        return inflows, outflows

    def _get_internal_flow_terms_flf(self, bud):

        assert self.nlay >= 2, 'Must have more than 2 layers to accumulate FLOW LOWER FACE record'
        inflows = []
        outflows = []

        # ACCUMULATE FLOW BETWEEN ZONES ACROSS LAYERS. COMPUTE FLOW ONLY BETWEEN A ZONE
        #  AND A HIGHER ZONE -- FLOW FROM ZONE 4 TO 3 IS THE NEGATIVE OF FLOW FROM 3 TO 4.
        # FIRST, CALCULATE FLOW BETWEEN NODE J,I,K AND J,I,K-1.
        nz = self.izone[1:, :, :]
        nzl = self.izone[-1:, :, :]
        l, r, c = np.where(nz > nzl)
        # Adjust column values by +1 to account for the starting position of "nz"
        l += 1

        # Define the zone to which flow is going
        to_zones = np.zeros((self.nlay, self.nrow, self.ncol), np.int32)
        to_zones[l, r, c] = self.izone[l, r, c-1]

        # Define cells where we will accumulate flow
        nzgt = np.zeros((self.nlay, self.nrow, self.ncol), np.int32)
        nzgt[l, r, c] = 1

        for z in self.zones:
            zone_bud = np.zeros_like(bud)
            zone_bud[(nzgt == 1) & (self.izone == z)] = bud[(nzgt == 1) & (self.izone == z)]
            index = []
            # totals_in = []
            totals_out = []
            for zi in self.zones:
                index.append('To Zone {:3d}'.format(zi))
                # totals_in.append(zone_bud[(to_zones == zi) & (zone_bud >= 0)].sum())
                totals_out.append(zone_bud[(to_zones == zi) & (zone_bud < 0)].sum())

            # zone_inflows = pd.Series(data=totals_in, index=index, name='Zone {}'.format(z))
            zone_outflows = pd.Series(data=totals_out, index=index, name='Zone {}'.format(z))
            # inflows.append(zone_inflows)
            outflows.append(zone_outflows*-1)

        # FLOW BETWEEN NODE J,I,K AND J,I+1,K.
        nz = self.izone[-1:, :, :]
        nzr = self.izone[1:, :, :]
        l, r, c = np.where(nz < nzr)
        r -= 1  # Adjust column values by -1 to account for the starting position of "nz"

        # Define the zone from which flow is coming
        from_zones = np.zeros((self.nlay, self.nrow, self.ncol), np.int32)
        from_zones[l, r, c] = self.izone[l, r, c-1]

        # Define cells where we will accumulate flow
        nzlt = np.zeros((self.nlay, self.nrow, self.ncol), np.int32)
        nzlt[l, r, c] = 1

        for z in self.zones:
            zone_bud = np.zeros_like(bud)
            zone_bud[(nzlt == 1) & (self.izone == z)] = bud[(nzlt == 1) & (self.izone == z)]
            index = []
            totals_in = []
            # totals_out = []
            for zi in self.zones:
                index.append('From Zone {:3d}'.format(zi))
                totals_in.append(zone_bud[(from_zones == zi) & (zone_bud >= 0)].sum())
                # totals_out.append(zone_bud[(from_zones == zi) & (zone_bud < 0)].sum())

            zone_inflows = pd.Series(data=totals_in, index=index, name='Zone {}'.format(z))
            # zone_outflows = pd.Series(data=totals_out, index=index, name='Zone {}'.format(z))
            inflows.append(zone_inflows)
            # outflows.append(zone_outflows*-1)

        # CALCULATE FLOW TO CONSTANT-HEAD CELLS IN THIS DIRECTION.
        # ----not yet tested----#

        inflows = pd.concat(inflows, axis=1)
        outflows = pd.concat(outflows, axis=1)
        return inflows, outflows

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

