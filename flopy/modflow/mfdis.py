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
from ..utils import Util2d, Util3d
from ..utils.reference import TemporalReference
from ..utils.flopy_io import line_parse

ITMUNI = {"u": 0, "s": 1, "m": 2, "h": 3, "d": 4, "y": 5}
LENUNI = {"u": 0, "f": 1, "m": 2, "c": 3}

warnings.simplefilter("always", PendingDeprecationWarning)


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
    steady : bool or array of bool (nper)
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
        x coordinate of upper left corner of the grid, default is None, which
        means xul will be set to zero.
    yul : float
        y coordinate of upper-left corner of the grid, default is None, which
        means yul will be calculated as the sum of the delc array.  This
        default, combined with the xul and rotation defaults will place the
        lower-left corner of the grid at (0, 0).
    rotation : float
        counter-clockwise rotation (in degrees) of the grid about the lower-
        left corner. default is 0.0
    proj4_str : str
        PROJ4 string that defines the projected coordinate system
        (e.g. '+proj=utm +zone=14 +datum=WGS84 +units=m +no_defs ').
        Can be an EPSG code (e.g. 'EPSG:32614'). Default is None.
    start_datetime : str
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

    def __init__(
        self,
        model,
        nlay=1,
        nrow=2,
        ncol=2,
        nper=1,
        delr=1.0,
        delc=1.0,
        laycbd=0,
        top=1,
        botm=0,
        perlen=1,
        nstp=1,
        tsmult=1,
        steady=True,
        itmuni=4,
        lenuni=2,
        extension="dis",
        unitnumber=None,
        filenames=None,
        xul=None,
        yul=None,
        rotation=None,
        proj4_str=None,
        start_datetime=None,
    ):

        # set default unit number of one is not specified
        if unitnumber is None:
            unitnumber = ModflowDis._defaultunit()

        # set filenames
        if filenames is None:
            filenames = [None]
        elif isinstance(filenames, str):
            filenames = [filenames]

        # Fill namefile items
        name = [ModflowDis._ftype()]
        units = [unitnumber]
        extra = [""]

        # set package name
        fname = [filenames[0]]

        # Call ancestor's init to set self.parent, extension, name and unit number
        Package.__init__(
            self,
            model,
            extension=extension,
            name=name,
            unit_number=units,
            extra=extra,
            filenames=fname,
        )

        self.url = "dis.htm"
        self.nrow = nrow
        self.ncol = ncol
        self.nlay = nlay
        self.nper = nper

        # initialize botm to an appropriate sized
        if nlay > 1:
            if isinstance(botm, float) or isinstance(botm, int):
                botm = np.linspace(top, botm, nlay)

        # Set values of all parameters
        self.heading = (
            "# {} package for ".format(self.name[0])
            + " {}, ".format(model.version_types[model.version])
            + "generated by Flopy."
        )
        self.laycbd = Util2d(
            model, (self.nlay,), np.int32, laycbd, name="laycbd"
        )
        self.laycbd[-1] = 0  # bottom layer must be zero
        self.delr = Util2d(
            model,
            (self.ncol,),
            np.float32,
            delr,
            name="delr",
            locat=self.unit_number[0],
        )
        self.delc = Util2d(
            model,
            (self.nrow,),
            np.float32,
            delc,
            name="delc",
            locat=self.unit_number[0],
        )
        self.top = Util2d(
            model,
            (self.nrow, self.ncol),
            np.float32,
            top,
            name="model_top",
            locat=self.unit_number[0],
        )
        self.botm = Util3d(
            model,
            (self.nlay + sum(self.laycbd), self.nrow, self.ncol),
            np.float32,
            botm,
            "botm",
            locat=self.unit_number[0],
        )
        self.perlen = Util2d(
            model, (self.nper,), np.float32, perlen, name="perlen"
        )
        self.nstp = Util2d(model, (self.nper,), np.int32, nstp, name="nstp")
        self.tsmult = Util2d(
            model, (self.nper,), np.float32, tsmult, name="tsmult"
        )
        self.steady = Util2d(model, (self.nper,), bool, steady, name="steady")

        try:
            self.itmuni = int(itmuni)
        except:
            self.itmuni = ITMUNI[itmuni.lower()[0]]
        try:
            self.lenuni = int(lenuni)
        except:
            self.lenuni = LENUNI[lenuni.lower()[0]]

        self.parent.add_package(self)
        self.itmuni_dict = {
            0: "undefined",
            1: "seconds",
            2: "minutes",
            3: "hours",
            4: "days",
            5: "years",
        }

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

        # set the model grid coordinate info
        xll = None
        yll = None
        mg = model.modelgrid
        if rotation is not None:
            mg.set_coord_info(xoff=None, yoff=None, angrot=rotation)
        if xul is not None:
            xll = mg._xul_to_xll(xul)
        if yul is not None:
            yll = mg._yul_to_yll(yul)
        mg.set_coord_info(xoff=xll, yoff=yll, angrot=rotation, proj4=proj4_str)

        self.tr = TemporalReference(
            itmuni=self.itmuni, start_datetime=start_datetime
        )

        self.start_datetime = start_datetime
        # calculate layer thicknesses
        self.__calculate_thickness()
        self._totim = None

    @property
    def sr(self):
        from ..utils.reference import SpatialReference

        warnings.warn(
            "SpatialReference has been deprecated. Use Grid instead.",
            DeprecationWarning,
        )
        if not hasattr(self, "_sr"):
            mg = self.parent.modelgrid
            self._sr = SpatialReference(
                self.delr,
                self.delc,
                self.lenuni,
                xll=mg.xoffset,
                yll=mg.yoffset,
                rotation=mg.angrot or 0.0,
                proj4_str=mg.proj4,
            )
        return self._sr

    @sr.setter
    def sr(self, sr):
        warnings.warn(
            "SpatialReference has been deprecated. Use Grid instead.",
            DeprecationWarning,
        )
        self._sr = sr

    def checklayerthickness(self):
        """
        Check layer thickness.

        """
        return (self.parent.modelgrid.thick > 0).all()

    def get_totim(self, use_cached=False):
        """
        Get the totim at the end of each time step

        Parameters
        ----------
        use_cached : bool
            method to use cached totim values instead of calculating totim
            dynamically


        Returns
        -------
        totim: numpy array
            numpy array with simulation totim at the end of each time step

        """
        if not use_cached or self._totim is None:
            totim = []
            nstp = self.nstp.array
            perlen = self.perlen.array
            tsmult = self.tsmult.array
            t = 0.0
            for kper in range(self.nper):
                m = tsmult[kper]
                p = float(nstp[kper])
                dt = perlen[kper]
                if m > 1:
                    dt *= (m - 1.0) / (m ** p - 1.0)
                else:
                    dt = dt / p
                for kstp in range(nstp[kper]):
                    t += dt
                    totim.append(t)
                    if m > 1:
                        dt *= m
            self._totim = np.array(totim, dtype=float)

        return self._totim

    def get_final_totim(self):
        """
        Get the totim at the end of the simulation

        Returns
        -------
        totim: float
            maximum simulation totim

        """
        return self.get_totim()[-1]

    def get_kstp_kper_toffset(self, t=0.0, use_cached_totim=False):
        """
        Get the stress period, time step, and time offset from passed time.

        Parameters
        ----------
        t : float
            totim to return the stress period, time step, and toffset for
            based on time discretization data. Default is 0.
        use_cached_totim : bool
            optional flag to use a cached calculation of totim, vs. dynamically
            calculating totim. Setting to True significantly speeds up looped
            operations that call this function (default is False).

        Returns
        -------
        kstp : int
            time step in stress period corresponding to passed totim
        kper : int
            stress period corresponding to passed totim
        toffset : float
            time offset of passed totim from the beginning of kper

        """

        if t < 0.0:
            t = 0.0
        totim = self.get_totim(use_cached_totim)
        nstp = self.nstp.array
        ipos = 0
        t0 = 0.0
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

    def get_totim_from_kper_toffset(
        self, kper=0, toffset=0.0, use_cached_totim=False
    ):
        """
        Get totim from a passed kper and time offset from the beginning
        of a stress period

        Parameters
        ----------
        kper : int
            stress period. Default is 0
        toffset : float
            time offset relative to the beginning of kper
        use_cached_totim : bool
            optional flag to use a cached calculation of totim, vs. dynamically
            calculating totim. Setting to True significantly speeds up looped
            operations that call this function (default is False).

        Returns
        -------
        t : float
            totim to return the stress period, time step, and toffset for
            based on time discretization data. Default is 0.

        """

        if kper < 0:
            kper = 0.0
        if kper >= self.nper:
            raise ValueError(
                "kper ({}) must be less than "
                "to nper ({}).".format(kper, self.nper)
            )

        totim = self.get_totim(use_cached_totim)
        nstp = self.nstp.array
        ipos = 0
        t0 = 0.0
        tp0 = 0.0
        for iper in range(kper + 1):
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
            vol[l, :, :] = self.parent.modelgrid.thick[l]
        for r in range(self.nrow):
            vol[:, r, :] *= self.delc[r]
        for c in range(self.ncol):
            vol[:, :, c] *= self.delr[c]
        return vol

    @property
    def zcentroids(self):
        z = np.empty((self.nlay, self.nrow, self.ncol))
        z[0, :, :] = (self.top[:, :] + self.botm[0, :, :]) / 2.0

        for l in range(1, self.nlay):
            z[l, :, :] = (self.botm[l - 1, :, :] + self.botm[l, :, :]) / 2.0
        return z

    def get_node_coordinates(self):
        """
        Get y, x, and z cell centroids in local model coordinates.

        Returns
        -------
        y : list of cell y-centroids

        x : list of cell x-centroids

        z : array of floats (nlay, nrow, ncol)

        """

        delr = self.delr.array
        delc = self.delc.array

        # In row direction
        Ly = np.add.reduce(delc)
        y = Ly - (np.add.accumulate(self.delc) - 0.5 * delc)

        # In column direction
        x = np.add.accumulate(self.delr) - 0.5 * delr

        # In layer direction
        z = self.zcentroids

        return y, x, z

    def get_rc_from_node_coordinates(self, x, y, local=True):
        """
        Get the row and column of a point or sequence of points
        in model coordinates.

        Parameters
        ----------
        x : float or sequence of floats
            x coordinate(s) of points to find in model grid
        y : float or sequence floats
            y coordinate(s) of points to find in model grid
        local : bool
          x and y coordinates are in model local coordinates.  If false, then
          x and y are in world coordinates. (default is True)

        Returns
        -------
        r : row or sequence of rows (zero-based)
        c : column or sequence of columns (zero-based)

        """
        mg = self.parent.modelgrid
        if np.isscalar(x):
            r, c = mg.intersect(x, y, local=local)
        else:
            r = []
            c = []
            for xx, yy in zip(x, y):
                rr, cc = mg.intersect(xx, yy, local=local)
                r.append(rr)
                c.append(cc)
        return r, c

    def get_lrc(self, nodes):
        """
        Get zero-based layer, row, column from a list of zero-based
        MODFLOW node numbers.

        Returns
        -------
        v : list of tuples containing the layer (k), row (i),
            and column (j) for each node in the input list
        """
        return self.parent.modelgrid.get_lrc(nodes)

    def get_node(self, lrc_list):
        """
        Get zero-based node number from a list of zero-based MODFLOW
        layer, row, column tuples.

        Returns
        -------
        v : list of MODFLOW nodes for each layer (k), row (i),
            and column (j) tuple in the input list
        """
        return self.parent.modelgrid.get_node(lrc_list)

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
        # thk = []
        # thk.append(self.top - self.botm[0])
        # for k in range(1, self.nlay + sum(self.laycbd)):
        #     thk.append(self.botm[k - 1] - self.botm[k])
        self.__thickness = Util3d(
            self.parent,
            (self.nlay + sum(self.laycbd), self.nrow, self.ncol),
            np.float32,
            self.parent.modelgrid.thick,
            name="thickness",
        )

    @property
    def thickness(self):
        """
        Return cell thicknesses.

        Returns
        -------
        thickness : array of floats (nlay, nrow, ncol)

        """
        warnings.warn(
            "ModflowDis.thickness will be deprecated and removed "
            "in version 3.3.5.  Use grid.thick().",
            PendingDeprecationWarning,
        )
        return self.parent.modelgrid.thick

    def write_file(self, check=True):
        """
        Write the package file.

        Parameters
        ----------
        check : bool
            Check package data for common errors. (default True)

        Returns
        -------
        None

        """
        if (
            check
        ):  # allows turning off package checks when writing files at model level
            self.check(
                f="{}.chk".format(self.name[0]),
                verbose=self.parent.verbose,
                level=1,
            )
        # Open file for writing
        f_dis = open(self.fn_path, "w")
        # Item 0: heading
        f_dis.write("{0:s}\n".format(self.heading))
        # Item 1: NLAY, NROW, NCOL, NPER, ITMUNI, LENUNI
        f_dis.write(
            "{0:10d}{1:10d}{2:10d}{3:10d}{4:10d}{5:10d}\n".format(
                self.nlay,
                self.nrow,
                self.ncol,
                self.nper,
                self.itmuni,
                self.lenuni,
            )
        )
        # Item 2: LAYCBD
        for l in range(0, self.nlay):
            f_dis.write("{0:3d}".format(self.laycbd[l]))
        f_dis.write("\n")
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
            f_dis.write(
                "{0:14f}{1:14d}{2:10f} ".format(
                    self.perlen[t], self.nstp[t], self.tsmult[t]
                )
            )
            if self.steady[t]:
                f_dis.write(" {0:3s}\n".format("SS"))
            else:
                f_dis.write(" {0:3s}\n".format("TR"))
        f_dis.close()

    def check(self, f=None, verbose=True, level=1, checktype=None):
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
        chk = self._get_check(f, verbose, level, checktype)

        # make ibound of same shape as thicknesses/botm for quasi-3D models
        active = chk.get_active(include_cbd=True)

        # Use either a numpy array or masked array
        thickness = self.parent.modelgrid.thick
        non_finite = ~(np.isfinite(thickness))
        if non_finite.any():
            thickness[non_finite] = 0
            thickness = np.ma.array(thickness, mask=non_finite)

        chk.values(
            thickness,
            active & (thickness <= 0),
            "zero or negative thickness",
            "Error",
        )
        thin_cells = (thickness < chk.thin_cell_threshold) & (thickness > 0)
        chk.values(
            thickness,
            active & thin_cells,
            "thin cells (less than checker threshold of {:.1f})".format(
                chk.thin_cell_threshold
            ),
            "Error",
        )
        chk.values(
            self.top.array,
            active[0, :, :] & np.isnan(self.top.array),
            "nan values in top array",
            "Error",
        )
        chk.values(
            self.botm.array,
            active & np.isnan(self.botm.array),
            "nan values in bottom array",
            "Error",
        )
        chk.summarize()
        return chk

    @classmethod
    def load(cls, f, model, ext_unit_dict=None, check=True):
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
        check : bool
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
            sys.stdout.write("loading dis package file...\n")

        openfile = not hasattr(f, "read")
        if openfile:
            filename = f
            f = open(filename, "r")

        # dataset 0 -- header
        header = ""
        while True:
            line = f.readline()
            if line[0] != "#":
                break
            header += line.strip()

        header = header.replace("#", "")
        xul, yul = None, None
        rotation = None
        proj4_str = None
        start_datetime = "1/1/1970"
        dep = False
        for item in header.split(","):
            if "xul" in item.lower():
                try:
                    xul = float(item.split(":")[1])
                except:
                    if model.verbose:
                        print("   could not parse xul in {}".format(filename))
                dep = True
            elif "yul" in item.lower():
                try:
                    yul = float(item.split(":")[1])
                except:
                    if model.verbose:
                        print("   could not parse yul in {}".format(filename))
                dep = True
            elif "rotation" in item.lower():
                try:
                    rotation = float(item.split(":")[1])
                except:
                    if model.verbose:
                        print(
                            "   could not parse rotation "
                            "in {}".format(filename)
                        )
                dep = True
            elif "proj4_str" in item.lower():
                try:
                    proj4_str = ":".join(item.split(":")[1:]).strip()
                except:
                    if model.verbose:
                        print(
                            "   could not parse proj4_str "
                            "in {}".format(filename)
                        )
                dep = True
            elif "start" in item.lower():
                try:
                    start_datetime = item.split(":")[1].strip()
                except:
                    if model.verbose:
                        print(
                            "   could not parse start in {}".format(filename)
                        )
                dep = True
        if dep:
            warnings.warn(
                "SpatialReference information found in DIS header,"
                "this information is being ignored.  "
                "SpatialReference info is now stored in the namfile"
                "header"
            )
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
            print(
                "   Loading dis package with:\n      {} layers, {} rows, {} "
                "columns, and {} stress periods".format(nlay, nrow, ncol, nper)
            )
            print("   loading laycbd...")
        laycbd = np.zeros(nlay, dtype=int)
        d = 0
        while True:
            line = f.readline()
            raw = line.strip("\n").split()
            for val in raw:
                if int(val) != 0:
                    laycbd[d] = 1
                d += 1
                if d == nlay:
                    break
            if d == nlay:
                break
        # dataset 3 -- delr
        if model.verbose:
            print("   loading delr...")
        delr = Util2d.load(
            f, model, (ncol,), np.float32, "delr", ext_unit_dict
        )
        # dataset 4 -- delc
        if model.verbose:
            print("   loading delc...")
        delc = Util2d.load(
            f, model, (nrow,), np.float32, "delc", ext_unit_dict
        )
        # dataset 5 -- top
        if model.verbose:
            print("   loading top...")
        top = Util2d.load(
            f, model, (nrow, ncol), np.float32, "top", ext_unit_dict
        )
        # dataset 6 -- botm
        ncbd = laycbd.sum()
        if model.verbose:
            print("   loading botm...")
            print(
                "      for {} layers and {} confining beds".format(nlay, ncbd)
            )
        if nlay > 1:
            botm = Util3d.load(
                f,
                model,
                (nlay + ncbd, nrow, ncol),
                np.float32,
                "botm",
                ext_unit_dict,
            )
        else:
            botm = Util3d.load(
                f, model, (nlay, nrow, ncol), np.float32, "botm", ext_unit_dict
            )
        # dataset 7 -- stress period info
        if model.verbose:
            print("   loading stress period data...")
            print("       for {} stress periods".format(nper))
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
            if a4.upper() == "TR":
                a4 = False
            else:
                a4 = True
            perlen.append(a1)
            nstp.append(a2)
            tsmult.append(a3)
            steady.append(a4)

        if openfile:
            f.close()

        # set package unit number
        unitnumber = None
        filenames = [None]
        if ext_unit_dict is not None:
            unitnumber, filenames[0] = model.get_ext_dict_attr(
                ext_unit_dict, filetype=ModflowDis._ftype()
            )

        # create dis object instance
        dis = cls(
            model,
            nlay=nlay,
            nrow=nrow,
            ncol=ncol,
            nper=nper,
            delr=delr,
            delc=delc,
            laycbd=laycbd,
            top=top,
            botm=botm,
            perlen=perlen,
            nstp=nstp,
            tsmult=tsmult,
            steady=steady,
            itmuni=itmuni,
            lenuni=lenuni,
            xul=xul,
            yul=yul,
            rotation=rotation,
            proj4_str=proj4_str,
            start_datetime=start_datetime,
            unitnumber=unitnumber,
            filenames=filenames,
        )
        if check:
            dis.check(
                f="{}.chk".format(dis.name[0]),
                verbose=dis.parent.verbose,
                level=0,
            )
        # return dis object instance
        return dis

    @staticmethod
    def _ftype():
        return "DIS"

    @staticmethod
    def _defaultunit():
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
