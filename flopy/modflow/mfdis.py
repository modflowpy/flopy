"""
mfdis module.  Contains the ModflowDis class. Note that the user can access
the ModflowDis class as `flopy.modflow.ModflowDis`.

Additional information for this MODFLOW package can be found at the `Online
MODFLOW Guide
<http://water.usgs.gov/ogw/modflow/MODFLOW-2005-Guide/index.html?dis.htm>`_.

"""

import sys
import warnings

import numpy as np

from ..pakbase import Package
from ..utils import Util2d, Util3d, reference, check
from ..utils.flopy_io import line_parse

ITMUNI = {"u": 0, "s": 1, "m": 2, "h": 3, "d": 4, "y": 5}
LENUNI = {"u": 0, "f": 1, "m": 2, "c": 3}


class ModflowDis(Package):
    """
    MODFLOW Discretization Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.Modflow`) to which
        this package will be added.
    nlay : int
        Number of model layers (the default is 1).
    nrow : int
        Number of model rows (the default is 2).
    ncol : int
        Number of model columns (the default is 2).
    nper : int
        Number of model stress periods (the default is 1).
    delr : float or array of floats (ncol), optional
        An array of spacings along a row (the default is 1.0).
    delc : float or array of floats (nrow), optional
        An array of spacings along a column (the default is 0.0).
    laycbd : int or array of ints (nlay), optional
        An array of flags indicating whether or not a layer has a Quasi-3D
        confining bed below it. 0 indicates no confining bed, and not zero
        indicates a confining bed. LAYCBD for the bottom layer must be 0. (the
        default is 0)
    top : float or array of floats (nrow, ncol), optional
        An array of the top elevation of layer 1. For the common situation in
        which the top layer represents a water-table aquifer, it may be
        reasonable to set Top equal to land-surface elevation (the default is
        1.0)
    botm : float or array of floats (nlay, nrow, ncol), optional
        An array of the bottom elevation for each model cell (the default is
        0.)
    perlen : float or array of floats (nper)
        An array of the stress period lengths.
    nstp : int or array of ints (nper)
        Number of time steps in each stress period (default is 1).
    tsmult : float or array of floats (nper)
        Time step multiplier (default is 1.0).
    steady : boolean or array of boolean (nper)
        true or False indicating whether or not stress period is steady state
        (default is True).
    itmuni : int
        Time units, default is days (4)
    lenuni : int
        Length units, default is meters (2)
    extension : string
        Filename extension (default is 'dis')
    unitnumber : int
        File unit number (default is None).
    filenames : str or list of str
        Filenames to use for the package. If filenames=None the package name
        will be created using the model name and package extension. If a
        single string is passed the package will be set to the string.
        Default is None.
    xul : float
        x coordinate of upper left corner of the grid, default is None
    yul : float
        y coordinate of upper left corner of the grid, default is None
    rotation : float
        clockwise rotation (in degrees) of the grid about the upper left
        corner. default is 0.0
    proj4_str : str
        PROJ4 string that defines the xul-yul coordinate system
        (.e.g. '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs ').
        Can be an EPSG code (e.g. 'EPSG:4326'). Default is 'EPSG:4326'
    start_dateteim : str
        starting datetime of the simulation. default is '1/1/1970'

    Attributes
    ----------
    heading : str
        Text string written to top of package input file.

    Methods
    -------

    See Also
    --------

    Notes
    -----

    Examples
    --------

    >>> import flopy
    >>> m = flopy.modflow.Modflow()
    >>> dis = flopy.modflow.ModflowDis(m)

    """

    def __init__(self, model, nlay=1, nrow=2, ncol=2, nper=1, delr=1.0,
                 delc=1.0, laycbd=0, top=1, botm=0, perlen=1, nstp=1,
                 tsmult=1, steady=True, itmuni=4, lenuni=2, extension='dis',
                 unitnumber=None, filenames=None,
                 xul=None, yul=None, rotation=0.0,
                 proj4_str=None, start_datetime=None):

        # set default unit number of one is not specified
        if unitnumber is None:
            unitnumber = ModflowDis.defaultunit()

        # set filenames
        if filenames is None:
            filenames = [None]
        elif isinstance(filenames, str):
            filenames = [filenames]

        # Fill namefile items
        name = [ModflowDis.ftype()]
        units = [unitnumber]
        extra = ['']

        # set package name
        fname = [filenames[0]]

        # Call ancestor's init to set self.parent, extension, name and unit number
        Package.__init__(self, model, extension=extension, name=name,
                         unit_number=units, extra=extra, filenames=fname)

        self.url = 'dis.htm'
        self.nrow = nrow
        self.ncol = ncol
        self.nlay = nlay
        self.nper = nper

        # initialize botm to an appropriate sized
        if nlay > 1:
            if isinstance(botm, float) or isinstance(botm, int):
                botm = np.linspace(top, botm, nlay)

        # Set values of all parameters
        self.heading = '# {} package for '.format(self.name[0]) + \
                       ' {}, '.format(model.version_types[model.version]) + \
                       'generated by Flopy.'
        self.laycbd = Util2d(model, (self.nlay,), np.int32, laycbd,
                             name='laycbd')
        self.laycbd[-1] = 0  # bottom layer must be zero
        self.delr = Util2d(model, (self.ncol,), np.float32, delr, name='delr',
                           locat=self.unit_number[0])
        self.delc = Util2d(model, (self.nrow,), np.float32, delc, name='delc',
                           locat=self.unit_number[0])
        self.top = Util2d(model, (self.nrow, self.ncol), np.float32,
                          top, name='model_top', locat=self.unit_number[0])
        self.botm = Util3d(model, (self.nlay + sum(self.laycbd),
                                   self.nrow, self.ncol), np.float32, botm,
                           'botm', locat=self.unit_number[0])
        self.perlen = Util2d(model, (self.nper,), np.float32, perlen,
                             name='perlen')
        self.nstp = Util2d(model, (self.nper,), np.int32, nstp, name='nstp')
        self.tsmult = Util2d(model, (self.nper,), np.float32, tsmult,
                             name='tsmult')
        self.steady = Util2d(model, (self.nper,), np.bool,
                             steady, name='steady')

        try:
            self.itmuni = int(itmuni)
        except:
            self.itmuni = ITMUNI[itmuni.lower()[0]]
        try:
            self.lenuni = int(lenuni)
        except:
            self.lenuni = LENUNI[lenuni.lower()[0]]

        self.parent.add_package(self)
        self.itmuni_dict = {0: "undefined", 1: "seconds", 2: "minutes",
                            3: "hours", 4: "days", 5: "years"}

        if xul is None:
            xul = model._xul
        if yul is None:
            yul = model._yul
        if rotation is None:
            rotation = model._rotation
        if proj4_str is None:
            proj4_str = model._proj4_str
        if start_datetime is None:
            start_datetime = model._start_datetime

        self.sr = reference.SpatialReference(self.delr.array, self.delc.array,
                                             self.lenuni, xul=xul, yul=yul,
                                             rotation=rotation,
                                             proj4_str=proj4_str)
        self.tr = reference.TemporalReference(itmuni=self.itmuni,
                                              start_datetime=start_datetime)
        self.start_datetime = start_datetime
        # calculate layer thicknesses
        self.__calculate_thickness()

    def checklayerthickness(self):
        """
        Check layer thickness.

        """
        return (self.thickness > 0).all()

    def get_totim(self):
        """
        Get the totim at the end of each time step

        Returns
        -------
        totim: numpy array
            numpy array with simulation totim at the end of each time step

        """
        totim = []
        nstp = self.nstp.array
        perlen = self.perlen.array
        tsmult = self.tsmult.array
        t = 0.
        for kper in range(self.nper):
            m = tsmult[kper]
            p = float(nstp[kper])
            dt = perlen[kper]
            if m > 1:
                dt *= (m - 1.) / (m**p - 1.)
            else:
                dt = dt / p
            for kstp in range(nstp[kper]):
                t += dt
                totim.append(t)
                if m > 1:
                    dt *= m
        return np.array(totim, dtype=np.float)

    def get_final_totim(self):
        """
        Get the totim at the end of the simulation

        Returns
        -------
        totim: float
            maximum simulation totim

        """
        return self.get_totim()[-1]

    def get_kstp_kper_toffset(self, t=0.):
        """
        Get the stress period, time step, and time offset from passed time.

        Parameters
        ----------
        t : float
            totim to return the stress period, time step, and toffset for
            based on time discretization data. Default is 0.

        Returns
        -------
        kstp : int
            time step in stress period corresponding to passed totim
        kper : int
            stress period corresponding to passed totim
        toffset : float
            time offset of passed totim from the beginning of kper

        """

        if t < 0.:
            t = 0.
        totim = self.get_totim()
        nstp = self.nstp.array
        ipos = 0
        t0 = 0.
        kper = self.nper - 1
        kstp = nstp[-1] - 1
        toffset = self.perlen.array[-1]
        done = False
        for iper in range(self.nper):
            tp0 = t0
            for istp in range(nstp[iper]):
                t1 = totim[ipos]
                if t >= t0 and t < t1:
                    done = True
                    kper = iper
                    kstp = istp
                    toffset = t - tp0
                    break
                ipos += 1
                t0 = t1
            if done:
                break
        return kstp, kper, toffset

    def get_totim_from_kper_toffset(self, kper=0, toffset=0.):
        """
        Get totim from a passed kper and time offset from the beginning
        of a stress period

        Parameters
        ----------
        kper : int
            stress period. Default is 0
        toffset : float
            time offset relative to the beginning of kper

        Returns
        -------
        t : float
            totim to return the stress period, time step, and toffset for
            based on time discretization data. Default is 0.

        """

        if kper < 0:
            kper = 0.
        if kper >= self.nper:
            msg = 'kper ({}) '.format(kper) + 'must be less than ' + \
                  'to nper ({}).'.format(self.nper)
            raise ValueError()
        totim = self.get_totim()
        nstp = self.nstp.array
        ipos = 0
        t0 = 0.
        tp0 = 0.
        for iper in range(kper+1):
            tp0 = t0
            if iper == kper:
                break
            for istp in range(nstp[iper]):
                t1 = totim[ipos]
                ipos += 1
                t0 = t1
        t = tp0 + toffset
        return t


    def get_cell_volumes(self):
        """
        Get an array of cell volumes.

        Returns
        -------
        vol : array of floats (nlay, nrow, ncol)

        """
        vol = np.empty((self.nlay, self.nrow, self.ncol))
        for l in range(self.nlay):
            vol[l, :, :] = self.thickness.array[l]
        for r in range(self.nrow):
            vol[:, r, :] *= self.delc[r]
        for c in range(self.ncol):
            vol[:, :, c] *= self.delr[c]
        return vol

    @property
    def zcentroids(self):
        z = np.empty((self.nlay, self.nrow, self.ncol))
        z[0, :, :] = (self.top[:, :] + self.botm[0, :, :]) / 2.

        for l in range(1, self.nlay):
            z[l, :, :] = (self.botm[l - 1, :, :] + self.botm[l, :, :]) / 2.
        return z

    def get_node_coordinates(self):
        """
        Get y, x, and z cell centroids.

        Returns
        -------
        y : list of cell y-centroids

        x : list of cell x-centroids

        z : array of floats (nlay, nrow, ncol)
        """
        # In row direction
        y = np.empty((self.nrow))
        for r in range(self.nrow):
            if (r == 0):
                y[r] = self.delc[r] / 2.
            else:
                y[r] = y[r - 1] + (self.delc[r] + self.delc[r - 1]) / 2.
        # Invert y to convert to a cartesian coordiante system
        y = y[::-1]
        # In column direction
        x = np.empty((self.ncol))
        for c in range(self.ncol):
            if (c == 0):
                x[c] = self.delr[c] / 2.
            else:
                x[c] = x[c - 1] + (self.delr[c] + self.delr[c - 1]) / 2.
        # In layer direction
        z = np.empty((self.nlay, self.nrow, self.ncol))
        for l in range(self.nlay):
            if (l == 0):
                z[l, :, :] = (self.top[:, :] + self.botm[l, :, :]) / 2.
            else:
                z[l, :, :] = (self.botm[l - 1, :, :] + self.botm[l, :, :]) / 2.
        return y, x, z

    def get_rc_from_node_coordinates(self, x, y):
        """Return the row and column of a point or sequence of points
        in model coordinates.

        Parameters
        ----------
        x : scalar or sequence of x coordinates
        y : scalar or sequence of y coordinates

        Returns
        -------
        r : row or sequence of rows (zero-based)
        c : column or sequence of columns (zero-based)
        """
        yn, xn, zn = self.get_node_coordinates()
        if np.isscalar(x):
            c = (np.abs(xn - x)).argmin()
            r = (np.abs(yn - y)).argmin()
        else:
            xcp = np.array([xn] * (len(x)))
            ycp = np.array([yn] * (len(x)))
            c = (np.abs(xcp.transpose() - x)).argmin(axis=0)
            r = (np.abs(ycp.transpose() - y)).argmin(axis=0)
        return r, c

    def get_lrc(self, nodes):
        """
        Get layer, row, column from a list of MODFLOW node numbers.

        Returns
        -------
        v : list of tuples containing the layer (k), row (i), 
            and column (j) for each node in the input list
        """
        if not isinstance(nodes, list):
            nodes = [nodes]
        nrc = self.nrow * self.ncol
        v = []
        for node in nodes:
            k = int(node / nrc)
            if (k * nrc) < node:
                k += 1
            ij = int(node - (k - 1) * nrc)
            i = int(ij / self.ncol)
            if (i * self.ncol) < ij:
                i += 1
            j = ij - (i - 1) * self.ncol
            v.append((k, i, j))
        return v

    def get_node(self, lrc_list):
        """
        Get node number from a list of MODFLOW layer, row, column tuples.

        Returns
        -------
        v : list of MODFLOW nodes for each layer (k), row (i), 
            and column (j) tuple in the input list
        """
        if not isinstance(lrc_list, list):
            lrc_list = [lrc_list]
        nrc = self.nrow * self.ncol
        v = []
        for [k, i, j] in lrc_list:
            node = int(((k) * nrc) + ((i) * self.ncol) + j)
            v.append(node)
        return v

    def get_layer(self, i, j, elev):
        """Return the layer for an elevation at an i, j location.

            Parameters
            ----------
            i : row index (zero-based)
            j : column index
            elev : elevation (in same units as model)

            Returns
            -------
            k : zero-based layer index
            """
        return get_layer(self, i, j, elev)

    def read_from_cnf(self, cnf_file_name, n_per_line=0):
        """
        Read discretization information from an MT3D configuration file.

        """

        def getn(ii, jj):
            if (jj == 0):
                n = 1
            else:
                n = int(ii / jj)
                if (ii % jj != 0):
                    n = n + 1

            return n

        try:
            f_cnf = open(cnf_file_name, 'r')

            # nlay, nrow, ncol
            line = f_cnf.readline()
            s = line.split()
            cnf_nlay = int(s[0])
            cnf_nrow = int(s[1])
            cnf_ncol = int(s[2])

            # ncol column widths delr[c]
            line = ''
            for dummy in range(getn(cnf_ncol, n_per_line)):
                line = line + f_cnf.readline()
            cnf_delr = [float(s) for s in line.split()]

            # nrow row widths delc[r]
            line = ''
            for dummy in range(getn(cnf_nrow, n_per_line)):
                line = line + f_cnf.readline()
            cnf_delc = [float(s) for s in line.split()]

            # nrow * ncol htop[r, c]
            line = ''
            for dummy in range(getn(cnf_nrow * cnf_ncol, n_per_line)):
                line = line + f_cnf.readline()
            cnf_top = [float(s) for s in line.split()]
            cnf_top = np.reshape(cnf_top, (cnf_nrow, cnf_ncol))

            # nlay * nrow * ncol layer thickness dz[l, r, c]
            line = ''
            for dummy in range(
                    getn(cnf_nlay * cnf_nrow * cnf_ncol, n_per_line)):
                line = line + f_cnf.readline()
            cnf_dz = [float(s) for s in line.split()]
            cnf_dz = np.reshape(cnf_dz, (cnf_nlay, cnf_nrow, cnf_ncol))

            # cinact, cdry, not used here so commented
            '''line = f_cnf.readline()
            s = line.split()
            cinact = float(s[0])
            cdry = float(s[1])'''

            f_cnf.close()
        finally:
            self.nlay = cnf_nlay
            self.nrow = cnf_nrow
            self.ncol = cnf_ncol

            self.delr = Util2d(model, (self.ncol,), np.float32, cnf_delr,
                               name='delr', locat=self.unit_number[0])
            self.delc = Util2d(model, (self.nrow,), np.float32, cnf_delc,
                               name='delc', locat=self.unit_number[0])
            self.top = Util2d(model, (self.nrow, self.ncol), np.float32,
                              cnf_top, name='model_top',
                              locat=self.unit_number[0])

            cnf_botm = np.empty((self.nlay + sum(self.laycbd), self.nrow,
                                 self.ncol))

            # First model layer
            cnf_botm[0:, :, :] = cnf_top - cnf_dz[0, :, :]
            # All other layers
            for l in range(1, self.nlay):
                cnf_botm[l, :, :] = cnf_botm[l - 1, :, :] - cnf_dz[l, :, :]

            self.botm = Util3d(model, (self.nlay + sum(self.laycbd),
                                       self.nrow, self.ncol), np.float32,
                               cnf_botm, 'botm',
                               locat=self.unit_number[0])

    def gettop(self):
        """
        Get the top array.

        Returns
        -------
        top : array of floats (nrow, ncol)
        """
        return self.top.array

    def getbotm(self, k=None):
        """
        Get the bottom array.

        Returns
        -------
        botm : array of floats (nlay, nrow, ncol), or

        botm : array of floats (nrow, ncol) if k is not none
        """
        if k is None:
            return self.botm.array
        else:
            return self.botm.array[k, :, :]

    def __calculate_thickness(self):
        thk = []
        thk.append(self.top - self.botm[0])
        for k in range(1, self.nlay + sum(self.laycbd)):
            thk.append(self.botm[k - 1] - self.botm[k])
        self.__thickness = Util3d(self.parent, (self.nlay + sum(self.laycbd),
                                                self.nrow, self.ncol),
                                  np.float32, thk, name='thickness')

    @property
    def thickness(self):
        """
        Get a Util3d array of cell thicknesses.

        Returns
        -------
        thickness : util3d array of floats (nlay, nrow, ncol)

        """
        #return self.__thickness
        thk = []
        thk.append(self.top - self.botm[0])
        for k in range(1, self.nlay + sum(self.laycbd)):
            thk.append(self.botm[k - 1] - self.botm[k])
        return Util3d(self.parent, (self.nlay + sum(self.laycbd),
                      self.nrow, self.ncol), np.float32,
                      thk, name='thickness')

    def write_file(self, check=True):
        """
        Write the package file.

        Parameters
        ----------
        check : boolean
            Check package data for common errors. (default True)

        Returns
        -------
        None

        """
        if check:  # allows turning off package checks when writing files at model level
            self.check(f='{}.chk'.format(self.name[0]),
                       verbose=self.parent.verbose, level=1)
        # Open file for writing
        f_dis = open(self.fn_path, 'w')
        # Item 0: heading        
        f_dis.write('{0:s}\n'.format(self.heading))
        # f_dis.write('#{0:s}'.format(str(self.sr)))
        # f_dis.write(" ,{0:s}:{1:s}\n".format("start_datetime",
        #                                    self.start_datetime))
        # Item 1: NLAY, NROW, NCOL, NPER, ITMUNI, LENUNI
        f_dis.write('{0:10d}{1:10d}{2:10d}{3:10d}{4:10d}{5:10d}\n' \
                    .format(self.nlay, self.nrow, self.ncol, self.nper,
                            self.itmuni, self.lenuni))
        # Item 2: LAYCBD
        for l in range(0, self.nlay):
            f_dis.write('{0:3d}'.format(self.laycbd[l]))
        f_dis.write('\n')
        # Item 3: DELR
        f_dis.write(self.delr.get_file_entry())
        # Item 4: DELC       
        f_dis.write(self.delc.get_file_entry())
        # Item 5: Top(NCOL, NROW)
        f_dis.write(self.top.get_file_entry())
        # Item 5: BOTM(NCOL, NROW)        
        f_dis.write(self.botm.get_file_entry())

        # Item 6: NPER, NSTP, TSMULT, Ss/tr
        for t in range(self.nper):
            f_dis.write('{0:14f}{1:14d}{2:10f} '.format(self.perlen[t],
                                                        self.nstp[t],
                                                        self.tsmult[t]))
            if self.steady[t]:
                f_dis.write(' {0:3s}\n'.format('SS'))
            else:
                f_dis.write(' {0:3s}\n'.format('TR'))
        f_dis.close()

    def check(self, f=None, verbose=True, level=1):
        """
        Check dis package data for zero and negative thicknesses.

        Parameters
        ----------
        f : str or file handle
            String defining file name or file handle for summary file
            of check method output. If a sting is passed a file handle
            is created. If f is None, check method does not write
            results to a summary file. (default is None)
        verbose : bool
            Boolean flag used to determine if check method results are
            written to the screen
        level : int
            Check method analysis level. If level=0, summary checks are
            performed. If level=1, full checks are performed.

        Returns
        -------
        None

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow.load('model.nam')
        >>> m.dis.check()
        """
        chk = check(self, f=f, verbose=verbose, level=level)

        # make ibound of same shape as thicknesses/botm for quasi-3D models
        active = chk.get_active(include_cbd=True)

        # Use either a numpy array or masked array
        thickness = self.thickness.array
        non_finite = ~(np.isfinite(thickness))
        if non_finite.any():
            thickness[non_finite] = 0
            thickness = np.ma.array(thickness, mask=non_finite)

        chk.values(thickness, active & (thickness <= 0),
                   'zero or negative thickness', 'Error')
        thin_cells = (thickness < chk.thin_cell_threshold) & (thickness > 0)
        chk.values(thickness, active & thin_cells,
                   'thin cells (less than checker threshold of {:.1f})'
                   .format(chk.thin_cell_threshold), 'Error')
        chk.values(self.top.array,
                   active[0, :, :] & np.isnan(self.top.array),
                   'nan values in top array', 'Error')
        chk.values(self.botm.array,
                   active & np.isnan(self.botm.array),
                   'nan values in bottom array', 'Error')
        chk.summarize()
        return chk

        '''
        if f is not None:
            if isinstance(f, str):
                pth = os.path.join(self.parent.model_ws, f)
                f = open(pth, 'w', 0)

        errors = False
        txt = '\n{} PACKAGE DATA VALIDATION:\n'.format(self.name[0])
        t = ''
        t1 = ''
        inactive = self.parent.bas6.ibound.array == 0
        # thickness errors
        d = self.thickness.array
        d[inactive] = 1.
        if d.min() <= 0:
            errors = True
            t = '{}  ERROR: Negative or zero cell thickness specified.\n'.format(t)
            if level > 0:
                idx = np.column_stack(np.where(d <= 0.))
                t1 = self.level1_arraylist(idx, d, self.thickness.name, t1)
        else:
            t = '{}  Specified cell thickness is OK.\n'.format(t)

        # add header to level 0 text
        txt += t

        if level > 0:
            if errors:
                txt += '\n  DETAILED SUMMARY OF {} ERRORS:\n'.format(self.name[0])
                # add level 1 header to level 1 text
                txt += t1

        # write errors to summary file
        if f is not None:
            f.write('{}\n'.format(txt))

        # write errors to stdout
        if verbose:
            print(txt)
        '''

    @staticmethod
    def load(f, model, ext_unit_dict=None, check=True):
        """
        Load an existing package.

        Parameters
        ----------
        f : filename or file handle
            File to load.
        model : model object
            The model object (of type :class:`flopy.modflow.mf.Modflow`) to
            which this package will be added.
        ext_unit_dict : dictionary, optional
            If the arrays in the file are specified using EXTERNAL,
            or older style array control records, then `f` should be a file
            handle.  In this case ext_unit_dict is required, which can be
            constructed using the function
            :class:`flopy.utils.mfreadnam.parsenamefile`.
        check : boolean
            Check package data for common errors. (default True)

        Returns
        -------
        dis : ModflowDis object
            ModflowDis object.

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow()
        >>> dis = flopy.modflow.ModflowDis.load('test.dis', m)

        """

        if model.verbose:
            sys.stdout.write('loading dis package file...\n')

        if not hasattr(f, 'read'):
            filename = f
            f = open(filename, 'r')
        # dataset 0 -- header
        header = ''
        while True:
            line = f.readline()
            if line[0] != '#':
                break
            header += line.strip()

        header = header.replace('#', '')
        xul, yul = None, None
        rotation = 0.0
        proj4_str = "EPSG:4326"
        start_datetime = "1/1/1970"
        dep = False
        for item in header.split(','):
            if "xul" in item.lower():
                try:
                    xul = float(item.split(':')[1])
                except:
                    pass
                dep = True
            elif "yul" in item.lower():
                try:
                    yul = float(item.split(':')[1])
                except:
                    pass
                dep = True
            elif "rotation" in item.lower():
                try:
                    rotation = float(item.split(':')[1])
                except:
                    pass
                dep = True
            elif "proj4_str" in item.lower():
                try:
                    proj4_str = ':'.join(item.split(':')[1:]).strip()
                except:
                    pass
                dep = True
            elif "start" in item.lower():
                try:
                    start_datetime = item.split(':')[1].strip()
                except:
                    pass
                dep = True
        if dep:
            warnings.warn("SpatialReference information found in DIS header,"
                          "this information is being ignored.  "
                          "SpatialReference info is now stored in the namfile"
                          "header")
        # dataset 1
        nlay, nrow, ncol, nper, itmuni, lenuni = line.strip().split()[0:6]
        nlay = int(nlay)
        nrow = int(nrow)
        ncol = int(ncol)
        nper = int(nper)
        itmuni = int(itmuni)
        lenuni = int(lenuni)
        # dataset 2 -- laycbd
        if model.verbose:
            print('   Loading dis package with:\n      ' + \
                  '{0} layers, {1} rows, {2} columns, and {3} stress periods'.format(
                      nlay, nrow, ncol, nper))
            print('   loading laycbd...')
        laycbd = np.zeros(nlay, dtype=np.int)
        d = 0
        while True:
            line = f.readline()
            raw = line.strip('\n').split()
            for val in raw:
                if (np.int(val)) != 0:
                    laycbd[d] = 1
                d += 1
                if d == nlay:
                    break
            if d == nlay:
                break
        # dataset 3 -- delr
        if model.verbose:
            print('   loading delr...')
        delr = Util2d.load(f, model, (ncol,), np.float32, 'delr',
                           ext_unit_dict)
        # dataset 4 -- delc
        if model.verbose:
            print('   loading delc...')
        delc = Util2d.load(f, model, (nrow,), np.float32, 'delc',
                           ext_unit_dict)
        # dataset 5 -- top
        if model.verbose:
            print('   loading top...')
        top = Util2d.load(f, model, (nrow, ncol), np.float32, 'top',
                          ext_unit_dict)
        # dataset 6 -- botm
        ncbd = laycbd.sum()
        if model.verbose:
            print('   loading botm...')
            print('      for {} layers and '.format(nlay) +
                  '{} confining beds'.format(ncbd))
        if nlay > 1:
            botm = Util3d.load(f, model, (nlay + ncbd, nrow, ncol), np.float32,
                               'botm', ext_unit_dict)
        else:
            botm = Util3d.load(f, model, (nlay, nrow, ncol), np.float32,
                               'botm',
                               ext_unit_dict)
        # dataset 7 -- stress period info
        if model.verbose:
            print('   loading stress period data...')
            print('       for {} stress periods'.format(nper))
        perlen = []
        nstp = []
        tsmult = []
        steady = []
        for k in range(nper):
            line = f.readline()
            a1, a2, a3, a4 = line_parse(line)[0:4]
            a1 = float(a1)
            a2 = int(a2)
            a3 = float(a3)
            if a4.upper() == 'TR':
                a4 = False
            else:
                a4 = True
            perlen.append(a1)
            nstp.append(a2)
            tsmult.append(a3)
            steady.append(a4)

        # set package unit number
        unitnumber = None
        filenames = [None]
        if ext_unit_dict is not None:
            unitnumber, filenames[0] = \
                model.get_ext_dict_attr(ext_unit_dict,
                                        filetype=ModflowDis.ftype())

        # create dis object instance
        dis = ModflowDis(model, nlay=nlay, nrow=nrow, ncol=ncol, nper=nper,
                         delr=delr, delc=delc, laycbd=laycbd,
                         top=top, botm=botm,
                         perlen=perlen, nstp=nstp, tsmult=tsmult,
                         steady=steady, itmuni=itmuni, lenuni=lenuni,
                         xul=xul, yul=yul, rotation=rotation,
                         proj4_str=proj4_str, start_datetime=start_datetime,
                         unitnumber=unitnumber, filenames=filenames)
        if check:
            dis.check(f='{}.chk'.format(dis.name[0]),
                      verbose=dis.parent.verbose, level=0)
        # return dis object instance
        return dis

    @staticmethod
    def ftype():
        return 'DIS'

    @staticmethod
    def defaultunit():
        return 11


def get_layer(dis, i, j, elev):
    """Return the layers for elevations at i, j locations.

    Parameters
    ----------
    dis : flopy.modflow.ModflowDis object
    i : scaler or sequence
        row index (zero-based)
    j : scaler or sequence
        column index
    elev : scaler or sequence
        elevation (in same units as model)

    Returns
    -------
    k : np.ndarray (1-D) or scalar
        zero-based layer index
    """
    def to_array(arg):
        if not isinstance(arg, np.ndarray):
            return np.array([arg])
        else:
            return arg

    i = to_array(i)
    j = to_array(j)
    elev = to_array(elev)
    botms = dis.botm.array[:, i, j].tolist()
    layers = np.sum(((botms - elev) > 0), axis=0)
    # force elevations below model bottom into bottom layer
    layers[layers > dis.nlay - 1] = dis.nlay - 1
    layers = np.atleast_1d(np.squeeze(layers))
    if len(layers) == 1:
        layers = layers[0]
    return layers
