import sys
import numpy as np
from ..pakbase import Package
from ..utils import Util3d


class Mt3dRct(Package):
    """
    Chemical reaction package class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.mt3dms.mt.Mt3dms`) to which
        this package will be added.
    isothm : int
        isothm is a flag indicating which type of sorption (or dual-domain mass
        transfer) is simulated: isothm = 0, no sorption is simulated;
        isothm = 1, linear isotherm (equilibrium-controlled); isothm = 2,
        Freundlich isotherm (equilibrium-controlled); isothm = 3, Langmuir
        isotherm (equilibrium-controlled); isothm = 4, first-order kinetic
        sorption (nonequilibrium); isothm = 5, dual-domain mass transfer
        (without sorption); isothm = 6, dual-domain mass transfer
        (with sorption). (default is 0).
    ireact : int
        ireact is a flag indicating which type of kinetic rate reaction is
        simulated: ireact = 0, no kinetic rate reaction is simulated;
        ireact = 1, first-order irreversible reaction. Note that this reaction
        package is not intended for modeling chemical reactions between
        species. An add-on reaction package developed specifically for that
        purpose may be used. (default is 0).
    igetsc : int
        igetsc is an integer flag indicating whether the initial concentration
        for the nonequilibrium sorbed or immobile phase of all species should
        be read when nonequilibrium sorption (isothm = 4) or dual-domain mass
        transfer (isothm = 5 or 6) is simulated: igetsc = 0, the initial
        concentration for the sorbed or immobile phase is not read. By default,
        the sorbed phase is assumed to be in equilibrium with the dissolved
        phase (isothm = 4), and the immobile domain is assumed to have zero
        concentration (isothm = 5 or 6). igetsc > 0, the initial concentration
        for the sorbed phase or immobile liquid phase of all species will be
        read. (default is 1).
    rhob : float or array of floats (nlay, nrow, ncol)
        rhob is the bulk density of the aquifer medium (unit, ML-3). rhob is
        used if isothm = 1, 2, 3, 4, or 6. If rhob is not user-specified and
        isothem is not 5 then rhob is set to 1.8e3. (default is None)
    prsity2 : float or array of floats (nlay, nrow, ncol)
        prsity2 is the porosity of the immobile domain (the ratio of pore
        spaces filled with immobile fluids over the bulk volume of the aquifer
        medium) when the simulation is intended to represent a dual-domain
        system. prsity2 is used if isothm = 5 or 6. If prsity2 is not user-
        specified and isothm = 5 or 6 then prsity2 is set to 0.1.
        (default is None)
    srconc : float or array of floats (nlay, nrow, ncol)
        srconc is the user-specified initial concentration for the sorbed phase
        of the first species if isothm = 4 (unit, MM-1). Note that for
        equilibrium-controlled sorption, the initial concentration for the
        sorbed phase cannot be specified. srconc is the user-specified initial
        concentration of the first species for the immobile liquid phase if
        isothm = 5 or 6 (unit, ML-3). If srconc is not user-specified and
        isothm = 4, 5, or 6 then srconc is set to 0. (default is None).
    sp1 : float or array of floats (nlay, nrow, ncol)
        sp1 is the first sorption parameter for the first species. The use of
        sp1 depends on the type of sorption selected (the value of isothm).
        For linear sorption (isothm = 1) and nonequilibrium sorption (isothm =
        4), sp1 is the distribution coefficient (Kd) (unit, L3M-1). For
        Freundlich sorption (isothm = 2), sp1 is the Freundlich equilibrium
        constant (Kf) (the unit depends on the Freundlich exponent a). For
        Langmuir sorption (isothm = 3), sp1 is the Langmuir equilibrium
        constant (Kl) (unit, L3M-1 ). For dual-domain mass transfer without
        sorption (isothm = 5), sp1 is not used, but still must be entered. For
        dual-domain mass transfer with sorption (isothm = 6), sp1 is also the
        distribution coefficient (Kd) (unit, L3M-1). If sp1 is not specified
        and isothm > 0 then sp1 is set to 0. (default is None).
    sp2 : float or array of floats (nlay, nrow, ncol)
        sp2 is the second sorption or dual-domain model parameter for the first
        species. The use of sp2 depends on the type of sorption or dual-domain
        model selected. For linear sorption (isothm = 1), sp2 is read but not
        used. For Freundlich sorption (isothm = 2), sp2 is the Freundlich
        exponent a. For Langmuir sorption (isothm = 3), sp2 is the total
        concentration of the sorption sites available ( S ) (unit, MM-1). For
        nonequilibrium sorption (isothm = 4), sp2 is the first-order mass
        transfer rate between the dissolved and sorbed phases (unit, T-1). For
        dual-domain mass transfer (isothm = 5 or 6), sp2 is the first-order
        mass transfer rate between the two domains (unit, T-1). If sp2 is not
        specified and isothm > 0 then sp2 is set to 0. (default is None).
    rc1 : float or array of floats (nlay, nrow, ncol)
        rc1 is the first-order reaction rate for the dissolved (liquid) phase
        for the first species (unit, T-1). rc1 is not used ireact = 0. If a
        dual-domain system is simulated, the reaction rates for the liquid
        phase in the mobile and immobile domains are assumed to be equal. If
        rc1 is not specified and ireact > 0 then rc1 is set to 0.
        (default is None).
    rc2 : float or array of floats (nlay, nrow, ncol)
        rc2 is the first-order reaction rate for the sorbed phase for the first
        species (unit, T-1). rc2 is not used ireact = 0. If a dual-domain
        system is simulated, the reaction rates for the sorbed phase in the
        mobile and immobile domains are assumed to be equal. Generally, if the
        reaction is radioactive decay, rc2 should be set equal to rc1, while
        for biodegradation, rc2 may be different from rc1. Note that rc2 is
        read but not used, if no sorption is included in the simulation. If
        rc2 is not specified and ireact > 0 then rc2 is set to 0.
        (default is None).
    extension : string
        Filename extension (default is 'rct')
    unitnumber : int
        File unit number. If file unit number is None then an unused unit
         number if used. (default is None).

    Other Parameters
    ----------------
    srconcn : float or array of floats (nlay, nrow, ncol)
        srconcn is the user-specified initial concentration for the sorbed
        phase of species n. If srconcn is not passed as a **kwarg and
        isothm = 4, 5, or 6 then srconc for species n is set to 0.
        See description of srconc for a more complete description of srconcn.
    sp1n : float or array of floats (nlay, nrow, ncol)
        sp1n is the first sorption parameter for species n. If sp1n is not
        passed as a **kwarg and isothm > 0 then sp1 for species n is set to 0.
        See description of sp1 for a more complete description of sp1n.
    sp2n : float or array of floats (nlay, nrow, ncol)
        sp2n is the second sorption or dual-domain model parameter for species
        n. If sp2n is not passed as a **kwarg and isothm > 0 then sp2 for
        species n is set to 0. See description of sp2 for a more complete
        description of sp2n.
    rc1n : float or array of floats (nlay, nrow, ncol)
        rc1n is the first-order reaction rate for the dissolved (liquid) phase
        for species n. If rc1n is not passed as a **kwarg and ireact > 0 then
        rc1 for species n is set to 0. See description of rc1 for a more
        complete description of rc1n.
    rc2n : float or array of floats (nlay, nrow, ncol)
        rc2n is the first-order reaction rate for the sorbed phase for species
        n. If rc2n is not passed as a **kwarg and ireact > 0 then rc2 for
        species n is set to 0. See description of rc2 for a more complete
        description of rc2n.


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
    >>> mt = flopy.mt3dms.Mt3dms()
    >>> rct = flopy.mt3dms.Mt3dRct(mt)
    """

    def __init__(self, model, isothm=0, ireact=0, igetsc=1, rhob=None,
                 prsity2=None, srconc=None, sp1=None, sp2=None, rc1=None,
                 rc2=None, extension='rct', unitnumber=None,
                 filenames=None, **kwargs):
        """
        Package constructor.

        """

        if unitnumber is None:
            unitnumber = Mt3dRct.defaultunit()
        elif unitnumber == 0:
            unitnumber = Mt3dRct.reservedunit()

        # set filenames
        if filenames is None:
            filenames = [None]
        elif isinstance(filenames, str):
            filenames = [filenames]

        # Fill namefile items
        name = [Mt3dRct.ftype()]
        units = [unitnumber]
        extra = ['']

        # set package name
        fname = [filenames[0]]

        # Call ancestor's init to set self.parent, extension, name and unit number
        Package.__init__(self, model, extension=extension, name=name,
                         unit_number=units, extra=extra, filenames=fname)

        nrow = model.nrow
        ncol = model.ncol
        nlay = model.nlay
        ncomp = model.ncomp

        # Item E1: ISOTHM, IREACT, IRCTOP, IGETSC
        self.isothm = isothm
        self.ireact = ireact
        self.irctop = 2  # All RCT vars are specified as 3D arrays
        self.igetsc = igetsc

        # Item E2A: RHOB
        if rhob is None:
            rhob = 1.8e3
        self.rhob = Util3d(model, (nlay, nrow, ncol), np.float32, rhob,
                           name='rhob', locat=self.unit_number[0],
                           array_free_format=False)

        # Item E2B: PRSITY
        if prsity2 is None:
            prsity2 = 0.1
        self.prsity2 = Util3d(model, (nlay, nrow, ncol), np.float32, prsity2,
                              name='prsity2', locat=self.unit_number[0],
                              array_free_format=False)

        # Item E2C: SRCONC
        if srconc is None:
            srconc = 0.0
        self.srconc = []
        u3d = Util3d(model, (nlay, nrow, ncol), np.float32, srconc,
                     name='srconc1', locat=self.unit_number[0],
                     array_free_format=False)
        self.srconc.append(u3d)
        if ncomp > 1:
            for icomp in range(2, ncomp + 1):
                name = "srconc" + str(icomp)
                val = 0.0
                if name in kwargs:
                    val = kwargs.pop(name)
                else:
                    print("RCT: setting srconc for component " +
                          str(icomp) + " to zero, kwarg name " +
                          name)
                u3d = Util3d(model, (nlay, nrow, ncol), np.float32, val,
                             name=name, locat=self.unit_number[0],
                             array_free_format=False)
                self.srconc.append(u3d)

        # Item E3: SP1
        if sp1 is None:
            sp1 = 0.0
        self.sp1 = []
        u3d = Util3d(model, (nlay, nrow, ncol), np.float32, sp1, name='sp11',
                     locat=self.unit_number[0], array_free_format=False)
        self.sp1.append(u3d)
        if ncomp > 1:
            for icomp in range(2, ncomp + 1):
                name = "sp1" + str(icomp)
                val = 0.0
                if name in kwargs:
                    val = kwargs.pop(name)
                else:
                    print("RCT: setting sp1 for component " +
                          str(icomp) + " to zero, kwarg name " +
                          name)
                u3d = Util3d(model, (nlay, nrow, ncol), np.float32, val,
                             name=name, locat=self.unit_number[0],
                             array_free_format=False)
                self.sp1.append(u3d)

        # Item E4: SP2
        if sp2 is None:
            sp2 = 0.0
        self.sp2 = []
        u3d = Util3d(model, (nlay, nrow, ncol), np.float32, sp2, name='sp21',
                     locat=self.unit_number[0], array_free_format=False)
        self.sp2.append(u3d)
        if ncomp > 1:
            for icomp in range(2, ncomp + 1):
                name = "sp2" + str(icomp)
                val = 0.0
                if name in kwargs:
                    val = kwargs.pop(name)
                else:
                    print("RCT: setting sp2 for component " +
                          str(icomp) + " to zero, kwarg name " +
                          name)
                u3d = Util3d(model, (nlay, nrow, ncol), np.float32, val,
                             name=name, locat=self.unit_number[0],
                             array_free_format=False)
                self.sp2.append(u3d)

        # Item E5: RC1
        if rc1 is None:
            rc1 = 0.0
        self.rc1 = []
        u3d = Util3d(model, (nlay, nrow, ncol), np.float32, rc1, name='rc11',
                     locat=self.unit_number[0], array_free_format=False)
        self.rc1.append(u3d)
        if ncomp > 1:
            for icomp in range(2, ncomp + 1):
                name = "rc1" + str(icomp)
                val = 0.0
                if name in kwargs:
                    val = kwargs.pop(name)
                else:
                    print("RCT: setting rc1 for component " +
                          str(icomp) + " to zero, kwarg name " +
                          name)
                u3d = Util3d(model, (nlay, nrow, ncol), np.float32, val,
                             name=name, locat=self.unit_number[0],
                             array_free_format=False)
                self.rc1.append(u3d)

        # Item E4: RC2
        if rc2 is None:
            rc2 = 0.0
        self.rc2 = []
        u3d = Util3d(model, (nlay, nrow, ncol), np.float32, rc2, name='rc21',
                     locat=self.unit_number[0], array_free_format=False)
        self.rc2.append(u3d)
        if ncomp > 1:
            for icomp in range(2, ncomp + 1):
                name = "rc2" + str(icomp)
                val = 0.0
                if name in kwargs:
                    val = kwargs.pop(name)
                else:
                    print("RCT: setting rc2 for component " +
                          str(icomp) + " to zero, kwarg name " +
                          name)
                u3d = Util3d(model, (nlay, nrow, ncol), np.float32, val,
                             name=name, locat=self.unit_number[0],
                             array_free_format=False)
                self.rc2.append(u3d)

        # Check to make sure that all kwargs have been consumed
        if len(list(kwargs.keys())) > 0:
            raise Exception("RCT error: unrecognized kwargs: " +
                            ' '.join(list(kwargs.keys())))

        self.parent.add_package(self)
        return

    def __repr__(self):
        return 'Chemical reaction package class'

    def write_file(self):
        """
        Write the package file

        Returns
        -------
        None

        """
        # Open file for writing
        f_rct = open(self.fn_path, 'w')
        f_rct.write('%10i%10i%10i%10i\n' % (self.isothm, self.ireact,
                                            self.irctop, self.igetsc))
        if (self.isothm in [1, 2, 3, 4, 6]):
            f_rct.write(self.rhob.get_file_entry())
        if (self.isothm in [5, 6]):
            f_rct.write(self.prsity2.get_file_entry())
        if (self.igetsc > 0):
            for icomp in range(len(self.srconc)):
                f_rct.write(self.srconc[icomp].get_file_entry())
        if (self.isothm > 0):
            for icomp in range(len(self.sp1)):
                f_rct.write(self.sp1[icomp].get_file_entry())
        if (self.isothm > 0):
            for icomp in range(len(self.sp2)):
                f_rct.write(self.sp2[icomp].get_file_entry())
        if (self.ireact > 0):
            for icomp in range(len(self.rc1)):
                f_rct.write(self.rc1[icomp].get_file_entry())
        if (self.ireact > 0):
            for icomp in range(len(self.rc2)):
                f_rct.write(self.rc2[icomp].get_file_entry())
        f_rct.close()
        return

    @staticmethod
    def load(f, model, nlay=None, nrow=None, ncol=None, ncomp=None,
             ext_unit_dict=None):
        """
        Load an existing package.

        Parameters
        ----------
        f : filename or file handle
            File to load.
        model : model object
            The model object (of type :class:`flopy.mt3d.mt.Mt3dms`) to
            which this package will be added.
        nlay : int
            Number of model layers in the reaction package. If nlay is not
            specified, the number of layers in the passed model object is
            used. (default is None).
        nrow : int
            Number of model rows in the reaction package. If nrow is not
            specified, the number of rows in the passed model object is
            used. (default is None).
        ncol : int
            Number of model columns in the reaction package. If nlay is not
            specified, the number of columns in the passed model object is
            used. (default is None).
        ext_unit_dict : dictionary, optional
            If the arrays in the file are specified using EXTERNAL,
            or older style array control records, then `f` should be a file
            handle.  In this case ext_unit_dict is required, which can be
            constructed using the function
            :class:`flopy.utils.mfreadnam.parsenamefile`.

        Returns
        -------
        rct :  Mt3dRct object
            Mt3dRct object.

        Examples
        --------

        >>> import flopy
        >>> mt = flopy.mt3d.Mt3dms()
        >>> rct = flopy.mt3d.Mt3dRct.load('test.rct', mt)

        """

        if model.verbose:
            sys.stdout.write('loading rct package file...\n')

        # Open file, if necessary
        if not hasattr(f, 'read'):
            filename = f
            f = open(filename, 'r')

        # Set dimensions if necessary
        if nlay is None:
            nlay = model.nlay
        if nrow is None:
            nrow = model.nrow
        if ncol is None:
            ncol = model.ncol
        if ncomp is None:
            ncomp = model.ncomp

        # Setup kwargs to store multispecies information
        kwargs = {}

        # Item E1
        line = f.readline()
        if model.verbose:
            print('   loading ISOTHM, IREACT, IRCTOP, IGETSC...')
        isothm = int(line[0:10])
        ireact = int(line[10:20])
        try:
            irctop = int(line[20:30])
        except:
            irctop = 0
        try:
            igetsc = int(line[30:40])
        except:
            igetsc = 0
        if model.verbose:
            print('   ISOTHM {}'.format(isothm))
            print('   IREACT {}'.format(ireact))
            print('   IRCTOP {}'.format(irctop))
            print('   IGETSC {}'.format(igetsc))

        # Item E2A: RHOB
        rhob = None
        if model.verbose:
            print('   loading RHOB...')
        if isothm in [1, 2, 3, 4, 6]:
            rhob = Util3d.load(f, model, (nlay, nrow, ncol), np.float32,
                               'rhob', ext_unit_dict, array_format="mt3d")
            if model.verbose:
                print('   RHOB {}'.format(rhob))

        # Item E2A: PRSITY2
        prsity2 = None
        if model.verbose:
            print('   loading PRSITY2...')
        if isothm in [5, 6]:
            prsity2 = Util3d.load(f, model, (nlay, nrow, ncol), np.float32,
                                  'prsity2', ext_unit_dict,
                                  array_format="mt3d")
            if model.verbose:
                print('   PRSITY2 {}'.format(prsity2))

        # Item E2C: SRCONC
        srconc = None
        if model.verbose:
            print('   loading SRCONC...')
        if igetsc > 0:
            srconc = Util3d.load(f, model, (nlay, nrow, ncol), np.float32,
                                 'srconc1', ext_unit_dict, array_format="mt3d")
            if model.verbose:
                print('   SRCONC {}'.format(srconc))
            if ncomp > 1:
                for icomp in range(2, ncomp + 1):
                    name = "srconc" + str(icomp)
                    if model.verbose:
                        print('   loading {}...'.format(name))
                    u3d = Util3d.load(f, model, (nlay, nrow, ncol), np.float32,
                                      name, ext_unit_dict, array_format="mt3d")
                    kwargs[name] = u3d
                    if model.verbose:
                        print('   SRCONC{} {}'.format(icomp, u3d))

        # Item E3: SP1
        sp1 = None
        if model.verbose:
            print('   loading SP1...')
        if isothm > 0:
            sp1 = Util3d.load(f, model, (nlay, nrow, ncol), np.float32,
                              'sp11', ext_unit_dict, array_format="mt3d")
            if model.verbose:
                print('   SP1 {}'.format(sp1))
            if ncomp > 1:
                for icomp in range(2, ncomp + 1):
                    name = "sp1" + str(icomp)
                    if model.verbose:
                        print('   loading {}...'.format(name))
                    u3d = Util3d.load(f, model, (nlay, nrow, ncol), np.float32,
                                      name, ext_unit_dict, array_format="mt3d")
                    kwargs[name] = u3d
                    if model.verbose:
                        print('   SP1{} {}'.format(icomp, u3d))

        # Item E4: SP2
        sp2 = None
        if model.verbose:
            print('   loading SP2...')
        if isothm > 0:
            sp2 = Util3d.load(f, model, (nlay, nrow, ncol), np.float32,
                              'sp21', ext_unit_dict, array_format="mt3d")
            if model.verbose:
                print('   SP2 {}'.format(sp2))
            if ncomp > 1:
                for icomp in range(2, ncomp + 1):
                    name = "sp2" + str(icomp)
                    if model.verbose:
                        print('   loading {}...'.format(name))
                    u3d = Util3d.load(f, model, (nlay, nrow, ncol), np.float32,
                                      name, ext_unit_dict, array_format="mt3d")
                    kwargs[name] = u3d
                    if model.verbose:
                        print('   SP2{} {}'.format(icomp, u3d))

        # Item E5: RC1
        rc1 = None
        if model.verbose:
            print('   loading RC1...')
        if ireact > 0:
            rc1 = Util3d.load(f, model, (nlay, nrow, ncol), np.float32,
                              'rc11', ext_unit_dict,
                              array_format="mt3d")
            if model.verbose:
                print('   RC1 {}'.format(rc1))
            if ncomp > 1:
                for icomp in range(2, ncomp + 1):
                    name = "rc1" + str(icomp)
                    if model.verbose:
                        print('   loading {}...'.format(name))
                    u3d = Util3d.load(f, model, (nlay, nrow, ncol), np.float32,
                                      name, ext_unit_dict, array_format="mt3d")
                    kwargs[name] = u3d
                    if model.verbose:
                        print('   RC1{} {}'.format(icomp, u3d))

        # Item E6: RC2
        rc2 = None
        if model.verbose:
            print('   loading RC2...')
        if ireact > 0:
            rc2 = Util3d.load(f, model, (nlay, nrow, ncol), np.float32,
                              'rc21', ext_unit_dict, array_format="mt3d")
            if model.verbose:
                print('   RC2 {}'.format(rc2))
            if ncomp > 1:
                for icomp in range(2, ncomp + 1):
                    name = "rc2" + str(icomp)
                    if model.verbose:
                        print('   loading {}...'.format(name))
                    u3d = Util3d.load(f, model, (nlay, nrow, ncol), np.float32,
                                      name, ext_unit_dict, array_format="mt3d")
                    kwargs[name] = u3d
                    if model.verbose:
                        print('   RC2{} {}'.format(icomp, u3d))

        # Close the file
        f.close()

        # set package unit number
        unitnumber = None
        filenames = [None]
        if ext_unit_dict is not None:
            unitnumber, filenames[0] = \
                model.get_ext_dict_attr(ext_unit_dict,
                                        filetype=Mt3dRct.ftype())

        # Construct and return rct package
        rct = Mt3dRct(model, isothm=isothm, ireact=ireact, igetsc=igetsc,
                      rhob=rhob, prsity2=prsity2, srconc=srconc, sp1=sp1,
                      sp2=sp2, rc1=rc1, rc2=rc2, unitnumber=unitnumber,
                      filenames=filenames, **kwargs)
        return rct

    @staticmethod
    def ftype():
        return 'RCT'

    @staticmethod
    def defaultunit():
        return 36

    @staticmethod
    def reservedunit():
        return 8
