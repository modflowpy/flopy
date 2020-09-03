import sys
import numpy as np
from ..pakbase import Package
from ..utils import Util2d, Util3d
from ..utils.util_array import Transient3d


class SeawatVdf(Package):
    """
    SEAWAT Variable-Density Flow Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.seawat.swt.Seawat`) to which
        this package will be added.
    mtdnconc (or mt3drhoflg) : int
        is the MT3DMS species number that will be used in the equation of
        state to compute fluid density. This input variable was formerly
        referred to as MTDNCONC (Langevin and others, 2003).
        If MT3DRHOFLG = 0, fluid density is specified using items 6 and 7,
        and flow will be uncoupled with transport if the IMT Process is active.
        If MT3DRHOFLG > 0, fluid density is calculated using the MT3DMS
        species number that corresponds with MT3DRHOFLG. A value for
        MT3DRHOFLG greater than zero indicates that flow will be coupled with
        transport.
        If MT3DRHOFLG = -1, fluid density is calculated using one or more
        MT3DMS species. Items 4a, 4b, and 4c will be read instead of item 4.
        The dependence of fluid density on pressure head can only be activated
        when MT3DRHOFLG = -1. A value for MT3DRHOFLG of -1 indicates that flow
        will be coupled with transport.
    mfnadvfd : int
        is a flag that determines the method for calculating the internodal
        density values used to conserve fluid mass.
        If MFNADVFD = 2, internodal conductance values used to conserve fluid
        mass are calculated using a central-in-space algorithm.
        If MFNADVFD <> 2, internodal conductance values used to conserve fluid
        mass are calculated using an upstream-weighted algorithm.
    nswtcpl : int
        is a flag used to determine the flow and transport coupling procedure.
        If NSWTCPL = 0 or 1, flow and transport will be explicitly coupled
        using a one-timestep lag. The explicit coupling option is normally
        much faster than the iterative option and is recommended for most
        applications.
        If NSWTCPL > 1, NSWTCPL is the maximum number of non-linear coupling
        iterations for the flow and transport solutions. SEAWAT-2000 will stop
        execution after NSWTCPL iterations if convergence between flow and
        transport has not occurred.
        If NSWTCPL = -1, the flow solution will be recalculated only for: The
        first transport step of the simulation, or
        The last transport step of the MODFLOW timestep, or
        The maximum density change at a cell is greater than DNSCRIT.
    iwtable : int
        is a flag used to activate the variable-density water-table corrections
        (Guo and Langevin, 2002, eq. 82). If IWTABLE = 0, the water-table
        correction will not be applied.
        If IWTABLE > 0, the water-table correction will be applied.
    densemin : float
        is the minimum fluid density. If the resulting density value
        calculated with the equation of state is less than DENSEMIN, the
        density value is set to DENSEMIN.
        If DENSEMIN = 0, the computed fluid density is not limited by
        DENSEMIN (this is the option to use for most simulations).
        If DENSEMIN > 0, a computed fluid density less than DENSEMIN is
        automatically reset to DENSEMIN.
    densemax : float
        is the maximum fluid density. If the resulting density value
        calculated with the equation of state is greater than DENSEMAX, the
        density value is set to DENSEMAX.
        If DENSEMAX = 0, the computed fluid density is not limited by
        DENSEMAX (this is the option to use for most simulations).
        If DENSEMAX > 0, a computed fluid density larger than DENSEMAX is
        automatically reset to DENSEMAX.
    dnscrit : float
        is a user-specified density value. If NSWTCPL is greater than 1,
        DNSCRIT is the convergence crite- rion, in units of fluid density,
        for convergence between flow and transport. If the maximum fluid
        density difference between two consecutive implicit coupling
        iterations is not less than DNSCRIT, the program will continue to
        iterate on the flow and transport equations, or will terminate if
        NSWTCPL is reached. If NSWTCPL is -1, DNSCRIT is the maximum density
        threshold, in units of fluid density. If the fluid density change
        (between the present transport timestep and the last flow solution) at
        one or more cells is greater than DNSCRIT, then SEAWAT_V4 will update
        the flow field (by solving the flow equation with the updated density
        field).
    denseref : float
        is the fluid density at the reference concentration, temperature, and
        pressure. For most simulations, DENSEREF is specified as the density
        of freshwater at 25 degrees C and at a reference pressure of zero.
    drhodc : float
        formerly referred to as DENSESLP (Langevin and others, 2003), is the
        slope of the linear equation of state that relates fluid density to
        solute concentration. In SEAWAT_V4, separate values for DRHODC can be
        entered for as many MT3DMS species as desired. If DRHODC is not
        specified for a species, then that species does not affect fluid
        density. Any measurement unit can be used for solute concentration,
        provided DENSEREF and DRHODC are set properly. DRHODC can be
        approximated by the user by dividing the density difference over the
        range of end- member fluids by the difference in concentration between
        the end-member fluids.
    drhodprhd : float
        is the slope of the linear equation of state that relates fluid
        density to the height of the pressure head (in terms of the reference
        density). Note that DRHODPRHD can be calculated from the volumetric
        expansion coefficient for pressure using equation 15. If the
        simulation is formulated in terms of kilograms and meters, DRHODPRHD
        has an approximate value of 4.46 x 10-3 kg/m4. A value of zero, which
        is typically used for most problems, inactivates the dependence of
        fluid density on pressure.
    prhdref : float
        is the reference pressure head. This value should normally be set to
        zero.
    nsrhoeos : int
        is the number of MT3DMS species to be used in the equation of state
        for fluid density. This value is read only if MT3DRHOFLG = -1.
    mtrhospec : int
        is the MT3DMS species number corresponding to the adjacent DRHODC and
        CRHOREF.
    crhoref : float
        is the reference concentration (C0) for species, MTRHOSPEC. For most
        simulations, CRHOREF should be specified as zero. If MT3DRHOFLG > 0,
        CRHOREF is assumed to equal zero (as was done in previous versions of
        SEAWAT).
    firstdt : float
        is the length of the first transport timestep used to start the
        simulation if both of the following two condi- tions are met:
        1. The IMT Process is active, and 2. transport timesteps are
        calculated as a function of the user-specified Courant number (the
        MT3DMS input variable, PERCEL, is greater than zero).
    indense : int
        is a flag. INDENSE is read only if MT3DRHOFLG is equal to zero.
        If INDENSE < 0, values for the DENSE array will be reused from the
        previous stress period. If it is the first stress period, values for
        the DENSE array will be set to DENSEREF.
        If INDENSE = 0, values for the DENSE array will be set to DENSEREF.
        If INDENSE >= 1, values for the DENSE array will be read from item 7.
        If INDENSE = 2, values read for the DENSE array are assumed to
        represent solute concentration, and will be converted to density
        values using the equation of state.
    dense : Transient3d
        A float or array of floats (nlay, nrow, ncol) should be assigned as
        values to a dictionary related to keys of period number.  dense
        is the fluid density array read for each layer using the MODFLOW-2000
        U2DREL array reader. The DENSE array is read only if MT3DRHOFLG is
        equal to zero. The DENSE array may also be entered in terms of solute
        concentration, or any other units, if INDENSE is set to 2 and the
        constants used in the density equation of state are specified
        appropriately.
    extension : string
        Filename extension (default is 'vdf')
    unitnumber : int
        File unit number (default is 37).

    Attributes
    ----------

    Methods
    -------

    See Also
    --------

    Notes
    -----
    In swt_4 mtdnconc became mt3drhoflg. If the latter one is defined in
    kwargs, it will overwrite mtdnconc. Same goes for denseslp, which has
    become drhodc.

    When loading an existing SEAWAT model that has DENSE specified as
    concentrations, the load process will convert those concentrations into
    density values using the equation of state.  This is only relevant when
    mtdnconc (or mt3drhoflg) is set to zero.

    Examples
    --------

    >>> import flopy
    >>> m = flopy.seawat.Seawat()
    >>> lpf = flopy.seawat.SeawatVdf(m)

    """

    unitnumber = 37

    def __init__(
        self,
        model,
        mtdnconc=1,
        mfnadvfd=1,
        nswtcpl=1,
        iwtable=1,
        densemin=0,
        densemax=0,
        dnscrit=1e-2,
        denseref=1.000,
        denseslp=0.025,
        crhoref=0,
        firstdt=0.001,
        indense=1,
        dense=1.000,
        nsrhoeos=1,
        drhodprhd=4.46e-3,
        prhdref=0.0,
        extension="vdf",
        unitnumber=None,
        filenames=None,
        **kwargs
    ):

        if unitnumber is None:
            unitnumber = SeawatVdf._defaultunit()

        # set filenames
        if filenames is None:
            filenames = [None]
        elif isinstance(filenames, str):
            filenames = [filenames]

        # Fill namefile items
        name = [SeawatVdf._ftype()]
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

        nrow, ncol, nlay, nper = self.parent.nrow_ncol_nlay_nper

        self.mtdnconc = kwargs.pop("mt3drhoflg", mtdnconc)
        self.mfnadvfd = mfnadvfd
        self.nswtcpl = nswtcpl
        self.iwtable = iwtable
        self.densemin = densemin
        self.densemax = densemax
        self.dnscrit = dnscrit
        self.nsrhoeos = nsrhoeos
        self.denseref = denseref
        self.denseslp = kwargs.pop("drhodc", denseslp)
        self.crhoref = crhoref
        self.drhodprhd = drhodprhd
        self.prhdref = prhdref
        self.firstdt = firstdt
        self.indense = indense
        if self.mtdnconc == 0:
            self.dense = Transient3d(
                model,
                (nlay, nrow, ncol),
                np.float32,
                dense,
                name="dense_",
                locat=self.unit_number[0],
            )
        else:
            # dense not needed for most cases so setting to None
            self.dense = None
        self.parent.add_package(self)
        return

    def write_file(self):
        """
        Write the package file

        Returns
        -------
        None

        """
        f_vdf = open(self.fn_path, "w")

        # item 1
        f_vdf.write(
            "%10i%10i%10i%10i\n"
            % (self.mtdnconc, self.mfnadvfd, self.nswtcpl, self.iwtable)
        )

        # item 2
        f_vdf.write("%10.4f%10.4f\n" % (self.densemin, self.densemax))

        # item 3
        if self.nswtcpl > 1 or self.nswtcpl == -1:
            f_vdf.write("%10f\n" % (self.dnscrit))

        # item 4
        if self.mtdnconc >= 0:
            if self.nsrhoeos == 1:
                f_vdf.write("%10.4f%10.4f\n" % (self.denseref, self.denseslp))
            else:
                f_vdf.write(
                    "%10.4f%10.4f\n" % (self.denseref, self.denseslp[0])
                )

        elif self.mtdnconc == -1:
            f_vdf.write(
                "%10.4f%10.4f%10.4f\n"
                % (self.denseref, self.drhodprhd, self.prhdref)
            )
            f_vdf.write("%10i\n" % self.nsrhoeos)
            if self.nsrhoeos == 1:
                f_vdf.write(
                    "%10i%10.4f%10.4f\n" % (1, self.denseslp, self.crhoref)
                )
            else:
                for i in range(self.nsrhoeos):
                    mtrhospec = 1 + i
                    f_vdf.write(
                        "%10i%10.4f%10.4f\n"
                        % (mtrhospec, self.denseslp[i], self.crhoref[i])
                    )

        # item 5
        f_vdf.write("%10f\n" % (self.firstdt))

        # Transient DENSE array
        if self.mtdnconc == 0:

            nrow, ncol, nlay, nper = self.parent.nrow_ncol_nlay_nper
            for kper in range(nper):

                itmp, file_entry_dense = self.dense.get_kper_entry(kper)

                # item 6 (and possibly 7)
                if itmp > 0:
                    f_vdf.write("%10i\n" % (self.indense))
                    f_vdf.write(file_entry_dense)

                else:
                    f_vdf.write("%10i\n" % (itmp))

        f_vdf.close()
        return

    @classmethod
    def load(cls, f, model, nper=None, ext_unit_dict=None):
        """
        Load an existing package.

        Parameters
        ----------
        f : filename or file handle
            File to load.
        model : model object
            The model object (of type :class:`flopy.seawat.swt.Seawat`) to
            which this package will be added.
        nper : int
            The number of stress periods.  If nper is None, then nper will be
            obtained from the model object. (default is None).
        ext_unit_dict : dictionary, optional
            If the arrays in the file are specified using EXTERNAL,
            or older style array control records, then `f` should be a file
            handle.  In this case ext_unit_dict is required, which can be
            constructed using the function
            :class:`flopy.utils.mfreadnam.parsenamefile`.

        Returns
        -------
        vdf : SeawatVdf object
            SeawatVdf object.

        Examples
        --------

        >>> import flopy
        >>> mf = flopy.modflow.Modflow()
        >>> dis = flopy.modflow.ModflowDis(mf)
        >>> mt = flopy.mt3d.Mt3dms()
        >>> swt = flopy.seawat.Seawat(modflowmodel=mf, mt3dmsmodel=mt)
        >>> vdf = flopy.seawat.SeawatVdf.load('test.vdf', m)

        """

        if model.verbose:
            sys.stdout.write("loading vdf package file...\n")

        # Open file, if necessary
        openfile = not hasattr(f, "read")
        if openfile:
            filename = f
            f = open(filename, "r")

        # Dataset 0 -- comment line
        while True:
            line = f.readline()
            if line[0] != "#":
                break

        # Determine problem dimensions
        nrow, ncol, nlay, npertemp = model.get_nrow_ncol_nlay_nper()
        if nper is None:
            nper = npertemp

        # Item 1: MT3DRHOFLG MFNADVFD NSWTCPL IWTABLE - line already read above
        if model.verbose:
            print("   loading MT3DRHOFLG MFNADVFD NSWTCPL IWTABLE...")
        t = line.strip().split()
        mt3drhoflg = int(t[0])
        mfnadvfd = int(t[1])
        nswtcpl = int(t[2])
        iwtable = int(t[3])
        if model.verbose:
            print("   MT3DRHOFLG {}".format(mt3drhoflg))
            print("   MFNADVFD {}".format(mfnadvfd))
            print("   NSWTCPL {}".format(nswtcpl))
            print("   IWTABLE {}".format(iwtable))

        # Item 2 -- DENSEMIN DENSEMAX
        if model.verbose:
            print("   loading DENSEMIN DENSEMAX...")
        line = f.readline()
        t = line.strip().split()
        densemin = float(t[0])
        densemax = float(t[1])

        # Item 3 -- DNSCRIT
        if model.verbose:
            print("   loading DNSCRIT...")
        dnscrit = None
        if nswtcpl > 1 or nswtcpl == -1:
            line = f.readline()
            t = line.strip().split()
            dnscrit = float(t[0])

        # Item 4 -- DENSEREF DRHODC
        drhodprhd = None
        prhdref = None
        nsrhoeos = None
        mtrhospec = None
        crhoref = None
        if mt3drhoflg >= 0:
            if model.verbose:
                print("   loading DENSEREF DRHODC(1)...")
            line = f.readline()
            t = line.strip().split()
            denseref = float(t[0])
            drhodc = float(t[1])
            nsrhoeos = 1
        else:
            if model.verbose:
                print("   loading DENSEREF DRHODPRHD PRHDREF...")
            line = f.readline()
            t = line.strip().split()
            denseref = float(t[0])
            drhodprhd = float(t[1])
            prhdref = float(t[2])

            if model.verbose:
                print("   loading NSRHOEOS...")
            line = f.readline()
            t = line.strip().split()
            nsrhoeos = int(t[0])

            if model.verbose:
                print("    loading MTRHOSPEC DRHODC CRHOREF...")
            mtrhospec = []
            drhodc = []
            crhoref = []
            for i in range(nsrhoeos):
                line = f.readline()
                t = line.strip().split()
                mtrhospec.append(int(t[0]))
                drhodc.append(float(t[1]))
                crhoref.append(float(t[2]))

        # Item 5 -- FIRSTDT
        if model.verbose:
            print("   loading FIRSTDT...")
        line = f.readline()
        t = line.strip().split()
        firstdt = float(t[0])

        # Items 6 and 7 -- INDENSE DENSE
        indense = None
        dense = None
        if mt3drhoflg == 0:

            # Create dense as a Transient3D record
            dense = {}

            for iper in range(nper):

                if model.verbose:
                    print(
                        "   loading INDENSE "
                        "for stress period {}...".format(iper + 1)
                    )
                line = f.readline()
                t = line.strip().split()
                indense = int(t[0])

                if indense > 0:
                    name = "DENSE_StressPeriod_{}".format(iper)
                    t = Util3d.load(
                        f,
                        model,
                        (nlay, nrow, ncol),
                        np.float32,
                        name,
                        ext_unit_dict,
                    )
                    if indense == 2:
                        t = t.array
                        t = denseref + drhodc * t
                        t = Util3d(
                            model,
                            (nlay, nrow, ncol),
                            np.float32,
                            t,
                            name,
                            ext_unit_dict=ext_unit_dict,
                        )
                    dense[iper] = t

            dense = Transient3d(
                model, (nlay, nrow, ncol), np.float32, dense, name="dense_"
            )

            # Set indense = 1 because all concentrations converted to density
            indense = 1

        if openfile:
            f.close()

        # set package unit number
        unitnumber = None
        filenames = [None]
        if ext_unit_dict is not None:
            unitnumber, filenames[0] = model.get_ext_dict_attr(
                ext_unit_dict, filetype=SeawatVdf._ftype()
            )

        # Construct and return vdf package
        return cls(
            model,
            mt3drhoflg=mt3drhoflg,
            mfnadvfd=mfnadvfd,
            nswtcpl=nswtcpl,
            iwtable=iwtable,
            densemin=densemin,
            densemax=densemax,
            dnscrit=dnscrit,
            denseref=denseref,
            drhodc=drhodc,
            drhodprhd=drhodprhd,
            prhdref=prhdref,
            nsrhoeos=nsrhoeos,
            mtrhospec=mtrhospec,
            crhoref=crhoref,
            firstdt=firstdt,
            indense=indense,
            dense=dense,
            unitnumber=unitnumber,
            filenames=filenames,
        )

    @staticmethod
    def _ftype():
        return "VDF"

    @staticmethod
    def _defaultunit():
        return 37
