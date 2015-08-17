"""
Module spatial and temporal referencing for flopy model objects

"""

from datetime import datetime
import numpy as np
#import pandas as pd
#from flopy.utils.util_array import util_2d

# def temporalreference_from_binary_headers(recordarray, verbose=False):
#
#     ukper = np.unique(recordarray['kper'])
#     totim = []
#     nstp = []
#     tsmult = []
#     for uk in ukper:
#         uk_recarray = recordarray[recordarray['kper'] == uk]
#         # what is tsmult used for?? Is it necessary for anything??
#         #  no pertim in ucn file
#         tm = 1.0
#         try:
#             us = np.unique(uk_recarray['pertim'])
#             if us.shape[0] > 1:
#                 tm = (us[1] / us[0]) - 1.0
#         except:
#             pass
#         t = uk_recarray['totim'].max()
#         n = uk_recarray['kstp'].max()
#         totim.append(t)
#         nstp.append(n)
#         tsmult.append(tm)
#     totim = np.array(totim, dtype=np.float32)
#     nstp = np.array(nstp, dtype=np.int)
#     tsmults = np.array(tsmult, dtype=np.float32)
#     perlen = [totim[0]]
#     perlen.extend(list(totim[1:] - totim[:-1]))
#     perlen = np.array(perlen, dtype=np.float32)
#     if verbose:
#         print('LayerFile._build_tr(): assuming time units of days...')
#     #should this be tsmults instead of tsmult??
#     tr = TemporalReference(np.array(perlen), np.zeros_like(nstp),
#                            nstp, tsmult, 4)
#     return tr

def spatialreference_from_gridspc_file(filename, lenuni=0):
    f = open(filename,'r')
    lines = f.readlines()
    raw = f.readline().strip().split()
    nrow = int(raw[0])
    ncol = int(raw[1])
    raw = f.readline().strip().split()
    xul, yul, rot = float(raw[0]), float(raw[1]), float(raw[2])
    delr = []
    j = 0
    while j < ncol:
        raw = f.readline().strip().split()
        for r in raw:
            if '*' in r:
                rraw = r.split('*')
                for n in range(int(rraw[0])):
                    delr.append(int(rraw[1]))
                    j += 1
            else:
                delr.append(int(r))
                j += 1

    delc = []
    i = 0
    while i < nrow:
        raw = f.readline().strip().split()
        for r in raw:
            if '*' in r:
                rraw = r.split('*')
                for n in range(int(rraw[0])):
                    delc.append(int(rraw[1]))
                    i += 1
            else:
                delc.append(int(r))
                i += 1

    f.close()
    return SpatialReference(np.array(delr), np.array(delc),
                            lenuni, xul=xul, yul=yul, rotation=rot)

class SpatialReference(object):

    def __init__(self, delr, delc, lenuni, xul=None, yul=None, rotation=0.0):
        """
            delr: delr array
            delc: delc array
            lenuni: lenght unit code
            xul: x coord of upper left corner of grid
            yul: y coord of upper left corner of grid
            rotation_degrees: grid rotation
        """
        self.delc = delc
        self.delr = delr

        self.nrow = self.delc.shape[0]
        self.ncol = self.delr.shape[0]

        self.lenuni = lenuni
        # Set origin and rotation
        if xul is None:
            self.xul = 0.
        else:
            self.xul = xul
        if yul is None:
            self.yul = np.add.reduce(self.delc)
        else:
            self.yul = yul
        self.rotation = rotation

        self._xgrid = None
        self._ygrid = None
        self._ycentergrid = None
        self._xcentergrid = None


    def __repr__(self):
        s = "spatialReference:xul:{0:<G}, yul:{1:<G},rotation:{2:<G}\n".\
            format(self.xul,self.yul,self.rotation)
        s += "delr:" + str(self.delr) + "\n"
        s += "delc:" + str(self.delc) + "\n"
        return s


    @property
    def xedge(self):
        return self.get_xedge_array()

    @property
    def yedge(self):
        return self.get_yedge_array()

    @property
    def xgrid(self):
        self._set_xygrid()
        return self._xgrid

    @property
    def ygrid(self):
        self._set_xygrid()
        return self._ygrid

    @property
    def xcenter(self):
        return self.get_xcenter_array()

    @property
    def ycenter(self):
        return self.get_ycenter_array()

    @property
    def ycentergrid(self):
        self._set_xycentergrid()
        return self._ycentergrid

    @property
    def xcentergrid(self):
        self._set_xycentergrid()
        return self._xcentergrid

    def _set_xycentergrid(self):
        self._xcentergrid, self._ycentergrid = np.meshgrid(self.xcenter,
                                                          self.ycenter)
        self._xcentergrid, self._ycentergrid = self.rotate(self._xcentergrid,
                                                          self._ycentergrid,
                                                          self.rotation,
                                                          0, self.yedge[0])
        self._xcentergrid += self.xul
        self._ycentergrid += self.yul - self.yedge[0]

    def _set_xygrid(self):
        self._xgrid, self._ygrid = np.meshgrid(self.xedge, self.yedge)
        self._xgrid, self._ygrid = self.rotate(self._xgrid, self._ygrid, self.rotation,
                                               0, self.yedge[0])
        self._xgrid += self.xul
        self._ygrid += self.yul - self.yedge[0]


    @staticmethod
    def rotate(x, y, theta, xorigin=0., yorigin=0.):
        """
        Given x and y array-like values calculate the rotation about an
        arbitrary origin and then return the rotated coordinates.  theta is in
        degrees.

        """
        theta = -theta * np.pi / 180.
        xrot = xorigin + np.cos(theta) * (x - xorigin) - np.sin(theta) * \
                                                         (y - yorigin)
        yrot = yorigin + np.sin(theta) * (x - xorigin) + np.cos(theta) * \
                                                         (y - yorigin)
        return xrot, yrot


    def get_extent(self):
        """
        Get the extent of the rotated and offset grid

        Return (xmin, xmax, ymin, ymax)

        """
        x0 = self.xedge[0]
        x1 = self.xedge[-1]
        y0 = self.yedge[0]
        y1 = self.yedge[-1]

        # upper left point
        x0r, y0r = self.rotate(x0, y0, self.rotation, 0, self.yedge[0])
        x0r += self.xul
        y0r += self.yul - self.yedge[0]

        # upper right point
        x1r, y1r = self.rotate(x1, y0, self.rotation, 0, self.yedge[0])
        x1r += self.xul
        y1r += self.yul - self.yedge[0]

        # lower right point
        x2r, y2r = self.rotate(x1, y1, self.rotation, 0, self.yedge[0])
        x2r += self.xul
        y2r += self.yul - self.yedge[0]

        # lower left point
        x3r, y3r = self.rotate(x0, y1, self.rotation, 0, self.yedge[0])
        x3r += self.xul
        y3r += self.yul - self.yedge[0]

        xmin = min(x0r, x1r, x2r, x3r)
        xmax = max(x0r, x1r, x2r, x3r)
        ymin = min(y0r, y1r, y2r, y3r)
        ymax = max(y0r, y1r, y2r, y3r)

        return (xmin, xmax, ymin, ymax)


    def get_xcenter_array(self):
        """
        Return a numpy one-dimensional float array that has the cell center x
        coordinate for every column in the grid.

        """
        x = np.add.accumulate(self.delr) - 0.5 * self.delr
        return x

    def get_ycenter_array(self):
        """
        Return a numpy one-dimensional float array that has the cell center x
        coordinate for every row in the grid.

        """
        Ly = np.add.reduce(self.delc)
        y = Ly - (np.add.accumulate(self.delc) - 0.5 *
                   self.delc)
        return y

    def get_xedge_array(self):
        """
        Return a numpy one-dimensional float array that has the cell edge x
        coordinates for every column in the grid.  Array is of size (ncol + 1)

        """
        xedge = np.concatenate(([0.], np.add.accumulate(self.delr)))
        return xedge

    def get_yedge_array(self):
        """
        Return a numpy one-dimensional float array that has the cell edge y
        coordinates for every row in the grid.  Array is of size (nrow + 1)

        """
        length_y = np.add.reduce(self.delc)
        yedge = np.concatenate(([length_y], length_y -
                np.add.accumulate(self.delc)))
        return yedge


    def write_gridSpec(self, filename):
        f = open(filename,'w')
        f.write("{0:10d} {1:10d}\n".format(self.delc.shape[0], self.delr.shape[0]))
        f.write("{0:15.6E} {1:15.6E} {2:15.6E}\n".format(self.xul,self.yul,self.rotation))
        for c in self.delc:
            f.write("{0:15.6E} ".format(c))
        f.write('\n')
        for r in self.delr:
            f.write("{0:15.6E} ".format(c))
        f.write('\n')
        return

    def get_vertices(self, i, j):
        pts = []
        xgrid, ygrid = self.xgrid, self.ygrid
        pts.append([xgrid[i, j], ygrid[i, j]])
        pts.append([xgrid[i, j], ygrid[i+1, j]])
        pts.append([xgrid[i, j+1], ygrid[i+1, j]])
        pts.append([xgrid[i, j+1], ygrid[i, j]])
        pts.append([xgrid[i, j], ygrid[i, j]])
        return pts


# class TemporalReference(object):
#
#     def __init__(self, perlen, steady, nstp, tsmult, itmuni, start_datetime=None):
#         """
#         :param perlen: stress period length array
#         :param steady: array-like boolean steady-state flag array
#         :param nstp: array of number of time steps per stress period
#         :param tsmult: array of time step length multiplier per stress period
#         :param itmuni: time unit
#         :param start_datetime: datetime instance
#         :return: none
#
#         stressperiod_start: date_range for start of stress periods
#         stressperiod_end: date_range for end of stress periods
#         stressperiod_deltas: timeoffset range for stress periods
#
#         timestep_start: date_range for start of time steps
#         timestep_end: date_range for end of time steps
#         timestep_delta: timeoffset range for time steps
#
#         kperkstp_loc: dict keyed on (kper,kstp) stores the index pos in the timestep ranges
#
#         """
#         self.itmuni_daterange = {1: 's', 2: 'm', 3: 'h', 4: 'd', 5: 'y'}
#         if start_datetime is None:
#             self.start = datetime(1970, 1, 1)
#             self.assumed = True
#         else:
#             assert isinstance(start_datetime, datetime)
#             self.start = start_datetime
#             self.assumed = False
#         if itmuni == 0:
#             print("temporalReference warning: time units (itmuni) undefined, assuming days")
#         self.unit = self.itmuni_daterange[itmuni]
#         # work out stress period lengths,starts and ends
#         self.stressperiod_deltas = pd.to_timedelta(perlen, unit=self.unit)
#         self.stressperiod_end = self.start + np.cumsum(self.stressperiod_deltas)
#         self.stressperiod_start = self.stressperiod_end - self.stressperiod_deltas
#
#         # work out time step lengths - not very elegant
#         offsets = []
#         idt = 0
#         self.kperkstp_loc = {}
#         for kper, (plen, nts, tmult) in enumerate(zip(perlen, nstp, tsmult)):
#             if tmult != 1.0:
#                 dt1 = plen * ((tmult - 1.)/((tmult**nts) - 1.))
#             else:
#                 dt1 = float(plen) / float(nts)
#             offsets.append(dt1)
#             self.kperkstp_loc[(kper, 0)] = idt
#             idt += 1
#             for its in range(nts-1):
#                 offsets.append(offsets[-1] * tmult)
#                 self.kperkstp_loc[(kper, its + 1)] = idt
#                 idt += 1
#         self.timestep_deltas = pd.to_timedelta(offsets, unit=self.unit)
#         self.timestep_end = self.start + np.cumsum(self.timestep_deltas)
#         self.timestep_start = self.timestep_end - self.timestep_deltas
#
#         self.perlen = perlen
#         self.steady = steady
#         self.nstp = nstp
#         self.tsmult = tsmult
#
#         if True in steady:
#             #raise NotImplementedError("temporalReference: not dealing with steady state yet")
#             print("temporalReference warning: not dealing with steady state yet")
#         return
#
#
#     def totim_to_datetime(self,totim):
#         return self.start + pd.to_timedelta(totim,unit=self.unit)
#     def __repr__(self):
#         s = "TemporalReference:start_datetime:" + str(self.start)
#         s += ", nper:{0:G}\n".format(self.perlen.shape[0])
#         s += "perlen:" + str(self.perlen) + '\n'
#         s += "nstp:" + str(self.nstp) + '\n'
#         s += "steady:" + str(self.steady) + '\n'
#         s += "tsmult:" + str(self.tsmult) + '\n'
#
#         return s
#
#     def get_datetimes_from_oc(self,oc):
#         raise NotImplementedError()