"""
Module spatial and temporal referencing for flopy model objects

"""

from datetime import datetime
import numpy as np
import pandas as pd
from flopy.utils import util_2d

class SpatialReference(object):

    def __init__(self, delr, delc, lenuni, xul=None, yul=None, rotation_degrees=0.0,):

        assert isinstance(delc,util_2d),"spatialReference error: delc must be util_2d instance"
        assert isinstance(delr,util_2d),"spatialReference error: delr must be util_2d instance"
        self.delc = delc
        self.delr = delr
        self.lenuni = lenuni
        # Set origin and rotation
        if xul is None:
            self.xul = 0.
        else:
            self.xul = xul
        if yul is None:
            self.yul = np.add.reduce(self.delc.array)
        else:
            self.yul = yul
        self.rotation = -rotation_degrees * np.pi / 180.

        # Create edge arrays and meshgrid for pcolormesh
        self.xedge = self.get_xedge_array()
        self.yedge = self.get_yedge_array()
        self.xgrid, self.ygrid = np.meshgrid(self.xedge, self.yedge)
        self.xgrid, self.ygrid = self.rotate(self.xgrid, self.ygrid, self.rotation,
                                        0, self.yedge[0])
        self.xgrid += self.xul
        self.ygrid += self.yul - self.yedge[0]

        # Create x and y center arrays and meshgrid of centers
        self.xcenter = self.get_xcenter_array()
        self.ycenter = self.get_ycenter_array()
        self.xcentergrid, self.ycentergrid = np.meshgrid(self.xcenter,
                                                         self.ycenter)
        self.xcentergrid, self.ycentergrid = self.rotate(self.xcentergrid,
                                                    self.ycentergrid,
                                                    self.rotation,
                                                    0, self.yedge[0])
        self.xcentergrid += self.xul
        self.ycentergrid += self.yul - self.yedge[0]


    @staticmethod
    def rotate(x, y, theta, xorigin=0., yorigin=0.):
        """
        Given x and y array-like values calculate the rotation about an
        arbitrary origin and then return the rotated coordinates.  theta is in
        radians.

        """
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
        x = np.add.accumulate(self.delr.array) - 0.5 * self.delr.array
        return x

    def get_ycenter_array(self):
        """
        Return a numpy one-dimensional float array that has the cell center x
        coordinate for every row in the grid.

        """
        Ly = np.add.reduce(self.delc.array)
        y = Ly - (np.add.accumulate(self.delc.array) - 0.5 *
                   self.delc.array)
        return y

    def get_xedge_array(self):
        """
        Return a numpy one-dimensional float array that has the cell edge x
        coordinates for every column in the grid.  Array is of size (ncol + 1)

        """
        xedge = np.concatenate(([0.], np.add.accumulate(self.delr.array)))
        return xedge

    def get_yedge_array(self):
        """
        Return a numpy one-dimensional float array that has the cell edge y
        coordinates for every row in the grid.  Array is of size (nrow + 1)

        """
        length_y = np.add.reduce(self.delc.array)
        yedge = np.concatenate(([length_y], length_y -
                np.add.accumulate(self.delc.array)))
        return yedge


    def write_gridSpec(self, filename):
        raise NotImplementedError()
        return

    def get_vertices(self, i, j):
        pts = []
        pts.append([self.xgrid[i, j], self.ygrid[i, j]])
        pts.append([self.xgrid[i, j], self.ygrid[i+1, j]])
        pts.append([self.xgrid[i, j+1], self.ygrid[i+1, j]])
        pts.append([self.xgrid[i, j+1], self.ygrid[i, j]])
        pts.append([self.xgrid[i, j], self.ygrid[i, j]])


class TemporalReference(object):

    def __init__(self, perlen, steady, nstp, tsmult, itmuni, start_datetime=None):
        self.itmuni_daterange = {1: "s", 2: "m", 3: "h", 4: "d", 5: "y"}
        if start_datetime is None:
            self.start = datetime(2015,1,1)
        else:
            self.start = start_datetime
        if itmuni == 0:
            print("temporalReference warning: time units (itmuni) undefined, assuming days")
        self.unit = self.itmuni_daterange[itmuni]
        #--work out stress period lengths,starts and ends
        self.stressperiod_deltas = pd.to_timedelta(perlen.array,unit=self.unit)
        self.stressperiod_end = self.start + np.cumsum(self.stressperiod_deltas)
        self.stressperiod_start = self.stressperiod_end - self.stressperiod_deltas

        #--work out time step lengths - not very elegant
        offsets = []
        for plen,nts,tmult in zip(perlen.array,nstp.array,tsmult.array):
            if tmult != 1.0:
                dt1 = plen * ((tmult-1.)/((tmult**nts)-1.))
            else:
                dt1 = float(plen) / float(nts)
            offsets.append(dt1)
            for its in range(nts-1):
                offsets.append(offsets[-1]*tmult)
        self.timestep_deltas = pd.to_timedelta(offsets,unit=self.unit)
        self.timestep_end = self.start + np.cumsum(self.timestep_deltas)
        self.timestep_start = self.timestep_end - self.timestep_deltas

        self.perlen = perlen
        self.steady = steady

        if False in steady:
            raise NotImplementedError("temporalReference: not dealing wth steady state yet")
        return


    def get_datetimes_from_oc(self,oc):
        raise NotImplementedError()