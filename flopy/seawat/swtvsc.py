import sys
import numpy as np
from ..pakbase import Package
from ..utils import Util3d, Transient3d


class SeawatVsc(Package):
    """
    SEAWAT Viscosity Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.seawat.swt.Seawat`) to which
        this package will be added.
    mt3dmuflg (or mt3drhoflg) : int
        is the MT3DMS species number that will be used in the equation to
        compute fluid viscosity.
        If MT3DMUFLG >= 0, fluid density is calculated using the MT3DMS
        species number that corresponds with MT3DMUFLG.
        If MT3DMUFLG = -1, fluid viscosity is calculated using one or more
        MT3DMS species.
    viscmin : float
        is the minimum fluid viscosity. If the resulting viscosity value
        calculated with the equation is less than VISCMIN, the viscosity
        value is set to VISCMIN.
        If VISCMIN = 0, the computed fluid viscosity is not limited by
        VISCMIN (this is the option to use for most simulations).
        If VISCMIN > 0, a computed fluid viscosity less than VISCMIN is
        automatically reset to VISCMIN.
    viscmax : float
        is the maximum fluid viscosity. If the resulting viscosity value
        calculated with the equation is greater than VISCMAX, the viscosity
        value is set to VISCMAX.
        If VISCMAX = 0, the computed fluid viscosity is not limited by
        VISCMAX (this is the option to use for most simulations).
        If VISCMAX > 0, a computed fluid viscosity larger than VISCMAX is
        automatically reset to VISCMAX.
    viscref : float
        is the fluid viscosity at the reference concentration and reference
        temperature. For most simulations, VISCREF is specified as the
        viscosity of freshwater.
    dmudc : float
        is the slope of the linear equation that relates fluid viscosity to
        solute concentration.
    nmueos : int
        is the number of MT3DMS species to be used in the linear equation
        for fluid viscosity (this number does not include the temperature
        species if the nonlinear option is being used). This value is read
        only if MT3DMUFLG = -1. A value of zero indicates that none of the
        MT3DMS species have a linear effect on fluid viscosity (the nonlinear
        temperature dependence may still be activated); nothing should be
        entered for item 3c in this case.
    mutempopt : int
        is a flag that specifies the option for including the effect of
        temperature on fluid viscosity.
        If MUTEMPOPT = 0, the effect of temperature on fluid viscosity is not
        included or is a simple linear relation that is specified in item 3c.
        If MUTEMPOPT = 1, fluid viscosity is calculated using equation 18.
        The size of the AMUCOEFF array in item 3e is 4 (MUNCOEFF = 4).
        If MUTEMPOPT = 2, fluid viscosity is calculated using equation 19.
        The size of the AMUCOEFF array in item 3e is 5 (MUNCOEFF = 5).
        If MUTEMPOPT = 3, fluid viscosity is calculated using equation 20.
        The size of the AMUCOEFF array in item 3e is 2 (MUNCOEFF = 2).
        If NSMUEOS and MUTEMPOPT are both set to zero, all fluid viscosities
        are set to VISCREF.
    mtmuspec : int, or list of ints (of size nsmueos) if nsmueos > 1
        is the MT3DMS species number corresponding to the adjacent DMUDC and
        CMUREF.
    dmudc : float, or list of floats (of size nsmueos) if nsmueos > 1
        is the slope of the linear equation that relates fluid viscosity to
        solute concentration.
    cmuref : float, or list of floats (of size nsmueos) if nsmueos > 1
        is the reference concentration.
    mtmuspectemp : int
        is the MT3DMS species number that corresponds to temperature. This
        value must be between 1 and NCOMP and should not be listed in
        MTMUSPEC of item 3c.
    amucoeff : float
        is the coefficient array of size MUNCOEFF. AMUCOEFF is A in
        equations 18, 19, and 20.
    muncoeff : int
        is the size of the AMUCOEFF array.
    invisc : int
        is a flag. INVISC is read only if MT3DMUFLG is equal to zero.
        If INVISC < 0, values for the VISC array will be reused from the
        previous stress period. If it is the first stress period, values for
        the VISC array will be set to VISCREF.
        If INVISC = 0, values for the VISC array will be set to VISCREF. If
        INVISC >= 1, values for the VISC array will be read from item 5.
        If INVISC = 2, values read for the VISC array are assumed to
        represent solute concentration, and will be converted to viscosity
        values.
    visc : float or array of floats (nlay, nrow, ncol)
        is the fluid viscosity array read for each layer using the
        MODFLOW-2000 U2DREL array reader. The VISC array is read only if
        MT3DMUFLG is equal to zero. The VISC array may also be entered in
        terms of solute concentration (or any other units) if INVISC is set
        to 2, and the simple linear expression in item 3 can be used to
        represent the relation to viscosity.
    extension : string
        Filename extension (default is 'vsc')
    unitnumber : int
        File unit number (default is 38).

    Attributes
    ----------

    Methods
    -------

    See Also
    --------

    Notes
    -----

    Examples
    --------

    >>> import flopy
    >>> m = flopy.seawat.Seawat()
    >>> vsc = flopy.modflow.SeawatVsc(m)

    """

    unitnumber = 38

    def __init__(
        self,
        model,
        mt3dmuflg=-1,
        viscmin=0.0,
        viscmax=0.0,
        viscref=8.904e-4,
        nsmueos=0,
        mutempopt=2,
        mtmuspec=1,
        dmudc=1.923e-06,
        cmuref=0.0,
        mtmutempspec=1,
        amucoeff=None,
        invisc=-1,
        visc=-1,
        extension="vsc",
        unitnumber=None,
        filenames=None,
        **kwargs
    ):

        if len(list(kwargs.keys())) > 0:
            raise Exception(
                "VSC error: unrecognized kwargs: "
                + " ".join(list(kwargs.keys()))
            )

        if unitnumber is None:
            unitnumber = SeawatVsc._defaultunit()

        # set filenames
        if filenames is None:
            filenames = [None]
        elif isinstance(filenames, str):
            filenames = [filenames]

        # Fill namefile items
        name = [SeawatVsc._ftype()]
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

        self.mt3dmuflg = mt3dmuflg
        self.viscmin = viscmin
        self.viscmax = viscmax
        self.viscref = viscref
        self.nsmueos = nsmueos
        self.mutempopt = mutempopt
        if not isinstance(mtmuspec, list):
            mtmuspec = [mtmuspec]
        self.mtmuspec = mtmuspec
        if not isinstance(dmudc, list):
            dmudc = [dmudc]
        self.dmudc = dmudc
        if not isinstance(cmuref, list):
            cmuref = [cmuref]
        self.cmuref = cmuref
        self.mtmutempspec = mtmutempspec
        if amucoeff is None:
            amucoeff = [0.001, 1, 0.015512, -20.0, -1.572]
        self.amucoeff = amucoeff
        self.invisc = invisc
        if self.mt3dmuflg == 0:
            self.visc = Transient3d(
                model,
                (nlay, nrow, ncol),
                np.float32,
                visc,
                name="visc_",
                locat=self.unit_number[0],
            )
        else:
            # visc not needed for most cases so setting to None
            self.visc = None
        self.parent.add_package(self)
        return

    def write_file(self):
        """
        Write the package file

        Returns
        -------
        None

        """
        f_vsc = open(self.fn_path, "w")

        # item 1
        f_vsc.write("{}\n".format(self.mt3dmuflg))

        # item 2
        f_vsc.write("{} {}\n".format(self.viscmin, self.viscmax))

        # item 3
        if self.mt3dmuflg >= 0:
            f_vsc.write(
                "{} {} {}\n".format(self.viscref, self.dmudc, self.cmuref)
            )

        # item 3a-d
        if self.mt3dmuflg == -1:
            f_vsc.write("{}\n".format(self.viscref))
            f_vsc.write("{} {}\n".format(self.nsmueos, self.mutempopt))
            # if self.nsmueos == 1:
            #     f_vsc.write('{} {} {}\n'.format(self.mtmuspec, self.dmudc,
            #                                   self.cmuref))
            # else:
            #     for iwr in range(self.nsmueos):
            #         f_vsc.write('{} {} {}\n'.format(self.mtmuspec[iwr],
            #                                         self.dmudc[iwr],
            #                                         self.cmuref[iwr]))
            if self.nsmueos > 0:
                for iwr in range(self.nsmueos):
                    f_vsc.write(
                        "{} {} {}\n".format(
                            self.mtmuspec[iwr],
                            self.dmudc[iwr],
                            self.cmuref[iwr],
                        )
                    )

            # item 3d
            if self.mutempopt > 0:
                s = "{} ".format(self.mtmutempspec)
                for a in tuple(self.amucoeff):
                    s += "{} ".format(a)
                f_vsc.write(s + "\n")

        # items 4 and 5, transient visc array
        if self.mt3dmuflg == 0:

            nrow, ncol, nlay, nper = self.parent.nrow_ncol_nlay_nper
            for kper in range(nper):

                itmp, file_entry_visc = self.visc.get_kper_entry(kper)

                # item 4 (and possibly 5)
                if itmp > 0:
                    f_vsc.write("{}\n".format(self.invisc))
                    f_vsc.write(file_entry_visc)

                else:
                    f_vsc.write("{}\n".format(itmp))

        f_vsc.close()
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
        vsc : SeawatVsc object
            SeawatVsc object.

        Examples
        --------

        >>> import flopy
        >>> mf = flopy.modflow.Modflow()
        >>> dis = flopy.modflow.ModflowDis(mf)
        >>> mt = flopy.mt3d.Mt3dms()
        >>> swt = flopy.seawat.Seawat(modflowmodel=mf, mt3dmsmodel=mt)
        >>> vdf = flopy.seawat.SeawatVsc.load('test.vsc', m)

        """

        if model.verbose:
            sys.stdout.write("loading vsc package file...\n")

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
        nrow, ncol, nlay, nper = model.get_nrow_ncol_nlay_nper()

        # Item 1: MT3DMUFLG - line already read above
        if model.verbose:
            print("   loading MT3DMUFLG...")
        t = line.strip().split()
        mt3dmuflg = int(t[0])
        if model.verbose:
            print("   MT3DMUFLG {}".format(mt3dmuflg))

        # Item 2 -- VISCMIN VISCMAX
        if model.verbose:
            print("   loading VISCMIN VISCMAX...")
        line = f.readline()
        t = line.strip().split()
        viscmin = float(t[0])
        viscmax = float(t[1])
        if model.verbose:
            print("   VISCMIN {}".format(viscmin))
            print("   VISCMAX {}".format(viscmax))

        # Item 3 -- VISCREF NSMUEOS MUTEMPOPT MTMUSPEC DMUDC CMUREF
        nsmueos = None
        mtmuspec = None
        cmuref = None
        nsmueos = None
        mutempopt = None
        mtmutempspec = None
        amucoeff = None
        if mt3dmuflg >= 0:
            if model.verbose:
                print("   loading VISCREF DMUDC(1) CMUREF(1)...")
            line = f.readline()
            t = line.strip().split()
            viscref = float(t[0])
            dmudc = float(t[1])
            cmuref = float(t[2])
            nsmueos = 1
            if model.verbose:
                print("   VISCREF {}".format(viscref))
                print("   DMUDC {}".format(dmudc))
                print("   CMUREF {}".format(cmuref))
        else:
            # Item 3a
            if model.verbose:
                print("   loading VISCREF...")
            line = f.readline()
            t = line.strip().split()
            viscref = float(t[0])
            if model.verbose:
                print("   VISCREF {}".format(viscref))

            # Item 3b
            if model.verbose:
                print("   loading NSMUEOS MUTEMPOPT...")
            line = f.readline()
            t = line.strip().split()
            nsmueos = int(t[0])
            mutempopt = int(t[1])
            if mutempopt == 1:
                muncoeff = 4
            elif mutempopt == 2:
                muncoeff = 5
            elif mutempopt == 3:
                muncoeff = 2
            else:
                muncoeff = None
            if model.verbose:
                print("   NSMUEOS {}".format(nsmueos))
                print("   MUTEMPOPT {}".format(mutempopt))

            # Item 3c
            if model.verbose:
                print("    loading MTMUSPEC DMUDC CMUREF...")
            mtmuspec = []
            dmudc = []
            cmuref = []
            for i in range(nsmueos):
                line = f.readline()
                t = line.strip().split()
                mtmuspec.append(int(t[0]))
                dmudc.append(float(t[1]))
                cmuref.append(float(t[2]))
            if model.verbose:
                print("   MTMUSPEC {}".format(mtmuspec))
                print("   DMUDC {}".format(dmudc))
                print("   CMUREF {}".format(cmuref))

            # Item 3d
            if mutempopt > 0:
                if model.verbose:
                    print("    loading MTMUTEMPSPEC AMUCOEFF...")
                line = f.readline()
                t = line.strip().split()
                mtmutempspec = int(t[0])
                amucoeff = []
                for i in range(muncoeff):
                    amucoeff.append(float(t[i + 1]))
                if model.verbose:
                    print("   MTMUTEMSPEC {}".format(mtmutempspec))
                    print("   AMUCOEFF {}".format(amucoeff))

        # Items 4 and 5 -- INVISC VISC
        invisc = None
        visc = None
        if mt3dmuflg == 0:

            # Create visc as a Transient3D record
            visc = {}

            for iper in range(nper):

                if model.verbose:
                    print(
                        "   loading INVISC "
                        "for stress period {}...".format(iper + 1)
                    )
                line = f.readline()
                t = line.strip().split()
                invisc = int(t[0])

                if invisc > 0:
                    name = "VISC_StressPeriod_{}".format(iper)
                    t = Util3d.load(
                        f,
                        model,
                        (nlay, nrow, ncol),
                        np.float32,
                        name,
                        ext_unit_dict,
                    )
                    if invisc == 2:
                        t = t.array
                        t = viscref + dmudc * (t - cmuref)
                        t = Util3d(
                            model,
                            (nlay, nrow, ncol),
                            np.float32,
                            t,
                            name,
                            ext_unit_dict=ext_unit_dict,
                        )
                    visc[iper] = t

            visc = Transient3d(
                model, (nlay, nrow, ncol), np.float32, visc, name="visc_"
            )

            # Set invisc = 1 because all concentrations converted to density
            invisc = 1

        if openfile:
            f.close()

        # set package unit number
        unitnumber = None
        filenames = [None]
        if ext_unit_dict is not None:
            unitnumber, filenames[0] = model.get_ext_dict_attr(
                ext_unit_dict, filetype=SeawatVsc._ftype()
            )

        # Construct and return vsc package
        return cls(
            model,
            mt3dmuflg=mt3dmuflg,
            viscmin=viscmin,
            viscmax=viscmax,
            viscref=viscref,
            nsmueos=nsmueos,
            mutempopt=mutempopt,
            mtmuspec=mtmuspec,
            dmudc=dmudc,
            cmuref=cmuref,
            mtmutempspec=mtmutempspec,
            amucoeff=amucoeff,
            invisc=invisc,
            visc=visc,
            unitnumber=unitnumber,
            filenames=filenames,
        )

    @staticmethod
    def _ftype():
        return "VSC"

    @staticmethod
    def _defaultunit():
        return 38
