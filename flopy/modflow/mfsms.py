"""
mfsms module.  This is the solver for MODFLOW-USG.
Contains the ModflowSms class. Note that the user can access
the ModflowSms class as `flopy.modflow.ModflowSms`.


"""

import sys

from ..pakbase import Package
from ..utils.flopy_io import line_parse


class ModflowSms(Package):
    """
    MODFLOW Basic Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.mf.Modflow`) to which
        this package will be added.
    hclose : float
        is the head change criterion for convergence of the outer (nonlinear)
        iterations, in units of length. When the maximum absolute value of the
        head change at all nodes during an iteration is less than or equal to
        HCLOSE, iteration stops. Commonly, HCLOSE equals 0.01.
    hiclose : float
        is the head change criterion for convergence of the inner (linear)
        iterations, in units of length. When the maximum absolute value of the
        head change at all nodes during an iteration is less than or equal to
        HICLOSE, the matrix solver assumes convergence. Commonly, HICLOSE is
        set an order of magnitude less than HCLOSE.
    mxiter : int
        is the maximum number of outer (nonlinear) iterations -- that is,
        calls to the solution routine. For a linear problem MXITER should be 1.
    iter1 : int
        is the maximum number of inner (linear) iterations. The number
        typically depends on the characteristics of the matrix solution
        scheme being used. For nonlinear problems, ITER1 usually ranges
        from 60 to 600; a value of 100 will be sufficient for most linear
        problems.
    iprsms : int
        is a flag that controls printing of convergence information from the
        solver: 0 is print nothing; 1 is print only the total number of
        iterations and nonlinear residual reduction summaries;  2 is print
        matrix solver information in addition to above.
    nonlinmeth : int
        is a flag that controls the nonlinear solution method and under-
        relaxation schemes. 0 is Picard iteration scheme is used without any
        under-relaxation schemes involved. > 0 is Newton-Raphson iteration
        scheme is used with under-relaxation. Note that the Newton-Raphson
        linearization scheme is available only for the upstream weighted
        solution scheme of the BCF and LPF packages. < 0 is Picard iteration
        scheme is used with under-relaxation. The absolute value of NONLINMETH
        determines the underrelaxation scheme used. 1 or -1, then
        Delta-Bar-Delta under-relaxation is used. 2 or -2 then Cooley
        under-relaxation scheme is used.
        Note that the under-relaxation schemes are used in conjunction with
        gradient based methods, however, experience has indicated that the
        Cooley under-relaxation and damping work well also for the Picard
        scheme with the wet/dry options of MODFLOW.
    linmeth : int
        is a flag that controls the matrix solution method. 1 is the XMD
        solver of Ibaraki (2005). 2 is the unstructured pre-conditioned
        conjugate gradient solver of White and Hughes (2011).
    theta : float
        is the reduction factor for the learning rate (under-relaxation term)
        of the delta-bar-delta algorithm. The value of THETA is between zero
        and one. If the change in the variable (head) is of opposite sign to
        that of the previous iteration, the under-relaxation term is reduced
        by a factor of THETA. The value usually ranges from 0.3 to 0.9; a
        value of 0.7 works well for most problems.
    akappa : float
        is the increment for the learning rate (under-relaxation term) of the
        delta-bar-delta algorithm. The value of AKAPPA is between zero and
        one. If the change in the variable (head) is of the same sign to that
        of the previous iteration, the under-relaxation term is increased by
        an increment of AKAPPA. The value usually ranges from 0.03 to 0.3; a
        value of 0.1 works well for most problems.
    gamma : float
        is the history or memory term factor of the delta-bar-delta algorithm.
        Gamma is between zero and 1 but cannot be equal to one. When GAMMA is
        zero, only the most recent history (previous iteration value) is
        maintained. As GAMMA is increased, past history of iteration changes
        has greater influence on the memory term.  The memory term is
        maintained as an exponential average of past changes. Retaining some
        past history can overcome granular behavior in the calculated function
        surface and therefore helps to overcome cyclic patterns of
        non-convergence. The value usually ranges from 0.1 to 0.3; a value of
        0.2 works well for most problems.
    amomentum : float
        is the fraction of past history changes that is added as a momentum
        term to the step change for a nonlinear iteration. The value of
        AMOMENTUM is between zero and one. A large momentum term should only
        be used when small learning rates are expected. Small amounts of the
        momentum term help convergence. The value usually ranges from 0.0001
        to 0.1; a value of 0.001 works well for most problems.
    numtrack : int
        is the maximum number of backtracking iterations allowed for residual
        reduction computations. If NUMTRACK = 0 then the backtracking
        iterations are omitted. The value usually ranges from 2 to 20; a
        value of 10 works well for most problems.
    numtrack : int
        is the maximum number of backtracking iterations allowed for residual
        reduction computations. If NUMTRACK = 0 then the backtracking
        iterations are omitted. The value usually ranges from 2 to 20; a
        value of 10 works well for most problems.
    btol : float
        is the tolerance for residual change that is allowed for residual
        reduction computations. BTOL should not be less than one to avoid
        getting stuck in local minima. A large value serves to check for
        extreme residual increases, while a low value serves to control
        step size more severely. The value usually ranges from 1.0 to 1e6 ; a
        value of 1e4 works well for most problems but lower values like 1.1
        may be required for harder problems.
    breduce : float
        is the reduction in step size used for residual reduction
        computations. The value of BREDUC is between zero and one. The value
        usually ranges from 0.1 to 0.3; a value of 0.2 works well for most
        problems.
    reslim : float
        is the limit to which the residual is reduced with backtracking.
        If the residual is smaller than RESLIM, then further backtracking is
        not performed. A value of 100 is suitable for large problems and
        residual reduction to smaller values may only slow down computations.
    iacl : int
        is the flag for choosing the acceleration method. 0 is Conjugate
        Gradient; select this option if the matrix is symmetric. 1 is
        ORTHOMIN. 2 is BiCGSTAB.
    norder : int
        is the flag for choosing the ordering scheme.
        0 is original ordering
        1 is reverse Cuthill McKee ordering
        2 is Minimum degree ordering
    level : int
        is the level of fill for ILU decomposition. Higher levels of fill
        provide more robustness but also require more memory. For optimal
        performance, it is suggested that a large level of fill be applied
        (7 or 8) with use of drop tolerance.
    north : int
        is the number of orthogonalizations for the ORTHOMIN acceleration
        scheme. A number between 4 and 10 is appropriate. Small values require
        less storage but more iteration may be required. This number should
        equal 2 for the other acceleration methods.
    iredsys : int
        is the index for creating a reduced system of equations using the
        red-black ordering scheme.
        0 is do not create reduced system
        1 is create reduced system using red-black ordering
    rrctol : float
        is a residual tolerance criterion for convergence. The root mean
        squared residual of the matrix solution is evaluated against this
        number to determine convergence. The solver assumes convergence if
        either HICLOSE (the absolute head tolerance value for the solver) or
        RRCTOL is achieved. Note that a value of zero ignores residual
        tolerance in favor of the absolute tolerance (HICLOSE) for closure of
        the matrix solver.
    idroptol : int
        is the flag to perform drop tolerance.
        0 is do not perform drop tolerance
        1 is perform drop tolerance
    epsrn : float
        is the drop tolerance value. A value of 1e-3 works well for most
        problems.
    clin : string
        an option keyword that defines the linear acceleration method used by
        the PCGU solver.
        CLIN is "CG", then preconditioned conjugate gradient method.
        CLIN is "BCGS", then preconditioned bi-conjugate gradient stabilized
        method.
    ipc : int
        an integer value that defines the preconditioner.
        IPC = 0, No preconditioning.
        IPC = 1, Jacobi preconditioning.
        IPC = 2, ILU(0) preconditioning.
        IPC = 3, MILU(0) preconditioning (default).
    iscl : int
        is the flag for choosing the matrix scaling approach used.
        0 is no matrix scaling applied
        1 is symmetric matrix scaling using the scaling method by the POLCG
        preconditioner in Hill (1992).
        2 is symmetric matrix scaling using the l2 norm of each row of
        A (DR) and the l2 norm of each row of DRA.
    iord : int
        is the flag for choosing the matrix reordering approach used.
        0 = original ordering
        1 = reverse Cuthill McKee ordering
        2 = minimum degree ordering
    rclosepcgu : float
        a real value that defines the flow residual tolerance for convergence
        of the PCGU linear solver. This value represents the maximum allowable
        residual at any single node. Value is in units of length cubed per
        time, and must be consistent with MODFLOW-USG length and time units.
        Usually a value of 1.0x10-1 is sufficient for the flow-residual
        criteria when meters and seconds are the defined MODFLOW-USG length
        and time.
    relaxpcgu : float
        a real value that defines the relaxation factor used by the MILU(0)
        preconditioner. RELAXPCGU is unitless and should be greater than or
        equal to 0.0 and less than or equal to 1.0. RELAXPCGU values of about
        1.0 are commonly used, and experience suggests that convergence can
        be optimized in some cases with RELAXPCGU values of 0.97. A RELAXPCGU
        value of 0.0 will result in ILU(0) preconditioning. RELAXPCGU is only
        specified if IPC=3. If RELAXPCGU is not specified and IPC=3, then a
        default value of 0.97 will be assigned to RELAXPCGU.
    extension : str, optional
        File extension (default is 'sms'.
    unitnumber : int, optional
        FORTRAN unit number for this package (default is None).
    filenames : str or list of str
        Filenames to use for the package. If filenames=None the package name
        will be created using the model name and package extension. If a
        single string is passed the package will be set to the string.
        Default is None.

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
    >>> m = flopy.modflow.Modflow()
    >>> sms = flopy.modflow.ModflowSms(m)

    """

    def __init__(self, model, hclose=1E-4, hiclose=1E-4, mxiter=100,
                 iter1=20, iprsms=2, nonlinmeth=0, linmeth=2,
                 theta=0.7, akappa=0.1, gamma=0.2, amomentum=0.001,
                 numtrack=20, btol=1e4, breduc=0.2, reslim=100.,
                 iacl=2, norder=0, level=7, north=2, iredsys=0,
                 rrctol=0., idroptol=0, epsrn=1.e-3,
                 clin='bcgs', ipc=3, iscl=0, iord=0, rclosepcgu=.1,
                 relaxpcgu=1.0, extension='sms', options=None,
                 unitnumber=None, filenames=None):
        # set default unit number of one is not specified
        if unitnumber is None:
            unitnumber = ModflowSms.defaultunit()

        # set filenames
        if filenames is None:
            filenames = [None]
        elif isinstance(filenames, str):
            filenames = [filenames]

        # Fill namefile items
        name = [ModflowSms.ftype()]
        units = [unitnumber]
        extra = ['']

        # set package name
        fname = [filenames[0]]

        # Call ancestor's init to set self.parent, extension, name and unit number
        Package.__init__(self, model, extension=extension, name=name,
                         unit_number=units, extra=extra, filenames=fname)

        self.heading = '# {} package for '.format(self.name[0]) + \
                       ' {}, '.format(model.version_types[model.version]) + \
                       'generated by Flopy.'
        self.url = ' '
        self.hclose = hclose
        self.hiclose = hiclose
        self.mxiter = mxiter
        self.iter1 = iter1
        self.iprsms = iprsms
        self.nonlinmeth = nonlinmeth
        self.linmeth = linmeth
        self.theta = theta
        self.akappa = akappa
        self.gamma = gamma
        self.amomentum = amomentum
        self.numtrack = numtrack
        self.btol = btol
        self.breduc = breduc
        self.reslim = reslim
        self.iacl = iacl
        self.norder = norder
        self.level = level
        self.north = north
        self.iredsys = iredsys
        self.rrctol = rrctol
        self.idroptol = idroptol
        self.epsrn = epsrn
        self.clin = clin
        self.ipc = ipc
        self.iscl = iscl
        self.iord = iord
        self.rclosepcgu = rclosepcgu
        self.relaxpcgu = relaxpcgu
        if options is None:
            self.options = []
        else:
            self.options = options
        self.parent.add_package(self)
        return

    def write_file(self):
        """
        Write the package file.

        Returns
        -------
        None

        """
        f = open(self.fn_path, 'w')
        f.write('{}\n'.format(self.heading))
        f.write('{0} {1} {2} {3} {4} {5} {6}\n'.format(
            self.hclose, self.hiclose, self.mxiter, self.iter1,
            self.iprsms, self.nonlinmeth, self.linmeth))
        if self.nonlinmeth != 0:
            f.write('{0} {1} {2} {3} {4} {5} {6} {7}\n'.format(
                self.theta, self.akappa, self.gamma, self.amomentum,
                self.numtrack, self.btol, self.breduc, self.reslim))
        if self.linmeth == 1:
            f.write('{0} {1} {2} {3} {4} {5} {6} {7}\n'.format(
                self.iacl, self.norder, self.level, self.north,
                self.iredsys, self.rrctol, self.idroptol, self.epsrn))
        if self.linmeth == 2:
            f.write('{0} {1} {2} {3} {4} {5}\n'.format(
                self.clin, self.ipc, self.iscl, self.iord,
                self.rclosepcgu, self.relaxpcgu))
        f.close()

    @staticmethod
    def load(f, model, ext_unit_dict=None):
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

        Returns
        -------
        sms : ModflowSms object

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow()
        >>> sms = flopy.modflow.ModflowPcg.load('test.sms', m)

        """

        if model.verbose:
            sys.stdout.write('loading sms package file...\n')

        if model.version != 'mfusg':
            msg = "Warning: model version was reset from " + \
                  "'{}' to 'mfusg' in order to load a SMS file".format(
                      model.version)
            print(msg)
            model.version = 'mfusg'

        if not hasattr(f, 'read'):
            filename = f
            f = open(filename, 'r')

        # dataset 0 -- header
        while True:
            line = f.readline()
            if line[0] != '#':
                break

        # Record 1a
        nopt = 0
        opts = ['simple', 'moderate', 'complex']
        for o in opts:
            if o in line.lower():
                options.append(0)
                nopt += 1

        if nopt > 0:
            line = f.readline()

        # Record 1b -- line will have already been read
        if model.verbose:
            print(
                '   loading HCLOSE HICLOSE MXITER ITER1 IPRSMS NONLINMETH LINMETH...')
        ll = line_parse(line)
        hclose = float(ll.pop(0))
        hiclose = float(ll.pop(0))
        mxiter = int(ll.pop(0))
        iter1 = int(ll.pop(0))
        iprsms = int(ll.pop(0))
        nonlinmeth = int(ll.pop(0))
        linmeth = int(ll.pop(0))
        if model.verbose:
            print('   HCLOSE {}'.format(hclose))
            print('   HICLOSE {}'.format(hiclose))
            print('   MXITER {}'.format(mxiter))
            print('   ITER1 {}'.format(iter1))
            print('   IPRSMS {}'.format(iprsms))
            print('   NONLINMETH {}'.format(nonlinmeth))
            print('   LINMETH {}'.format(linmeth))

        # Record 2
        theta = None
        akappa = None
        gamma = None
        amomentum = None
        numtrack = None
        btol = None
        breduc = None
        reslim = None
        if nonlinmeth != 0 and nopt == 0:
            if model.verbose:
                print(
                    '   loading THETA AKAPPA GAMMA AMOMENTUM NUMTRACK BTOL BREDUC RESLIM...')
            while True:
                line = f.readline()
                if line[0] != '#':
                    break
            ll = line_parse(line)
            theta = float(ll.pop(0))
            akappa = float(ll.pop(0))
            gamma = float(ll.pop(0))
            amomentum = float(ll.pop(0))
            numtrack = int(ll.pop(0))
            btol = float(ll.pop(0))
            breduc = float(ll.pop(0))
            reslim = float(ll.pop(0))
            if model.verbose:
                print('   THETA {}'.format(theta))
                print('   AKAPPA {}'.format(akappa))
                print('   GAMMA {}'.format(gamma))
                print('   AMOMENTUM {}'.format(amomentum))
                print('   NUMTRACK {}'.format(numtrack))
                print('   BTOL {}'.format(btol))
                print('   BREDUC {}'.format(breduc))
                print('   RESLIM {}'.format(reslim))

        iacl = None
        norder = None
        level = None
        north = None
        iredsys = None
        rrctol = None
        idroptol = None
        epsrn = None
        if linmeth == 1 and nopt == 0:
            if model.verbose:
                print(
                    '    loading IACL NORDER LEVEL NORTH IREDSYS RRCTOL IDROPTOL EPSRN')
            while True:
                line = f.readline()
                if line[0] != '#':
                    break
            ll = line_parse(line)
            iacl = int(ll.pop(0))
            norder = int(ll.pop(0))
            level = int(ll.pop(0))
            north = int(ll.pop(0))
            iredsys = int(ll.pop(0))
            rrctol = float(ll.pop(0))
            idroptol = int(ll.pop(0))
            epsrn = float(ll.pop(0))
            if model.verbose:
                print('   IACL {}'.format(iacl))
                print('   NORDER {}'.format(norder))
                print('   LEVEL {}'.format(level))
                print('   NORTH {}'.format(north))
                print('   IREDSYS {}'.format(iredsys))
                print('   RRCTOL {}'.format(rrctol))
                print('   IDROPTOL {}'.format(idroptol))
                print('   EPSRN {}'.format(epsrn))

        clin = None
        ipc = None
        iscl = None
        iord = None
        rclosepcgu = None
        relaxpcgu = None
        if linmeth == 2 and nopt == 0:
            if model.verbose:
                print(
                    '    loading [CLIN] IPC ISCL IORD RCLOSEPCGU [RELAXPCGU]')
            while True:
                line = f.readline()
                if line[0] != '#':
                    break
            ll = line_parse(line)
            if 'cg' in line.lower():  # this will get cg or bcgs
                clin = ll.pop(0)
            ipc = int(ll.pop(0))
            iscl = int(ll.pop(0))
            iord = int(ll.pop(0))
            rclosepcgu = float(ll.pop(0))
            if len(ll) > 0:
                relaxpcgu = float(ll.pop(0))
            if model.verbose:
                print('   CLIN {}'.format(clin))
                print('   IPC {}'.format(ipc))
                print('   ISCL {}'.format(iscl))
                print('   IORD {}'.format(iord))
                print('   RCLOSEPCGU {}'.format(rclosepcgu))
                print('   RELAXPCGU {}'.format(relaxpcgu))


        # set package unit number
        unitnumber = None
        filenames = [None]
        if ext_unit_dict is not None:
            unitnumber, filenames[0] = \
                model.get_ext_dict_attr(ext_unit_dict,
                                        filetype=ModflowSms.ftype())

        sms = ModflowSms(model, hclose=hclose, hiclose=hiclose, mxiter=mxiter,
                         iter1=iter1, iprsms=iprsms, nonlinmeth=nonlinmeth,
                         linmeth=linmeth, theta=theta, akappa=akappa,
                         gamma=gamma, amomentum=amomentum, numtrack=numtrack,
                         btol=btol, breduc=breduc, reslim=reslim,
                         iacl=iacl, norder=norder, level=level, north=north,
                         iredsys=iredsys, rrctol=rrctol, idroptol=idroptol,
                         epsrn=epsrn, clin=clin, ipc=ipc, iscl=iscl,
                         iord=iord, rclosepcgu=rclosepcgu,
                         relaxpcgu=relaxpcgu, unitnumber=unitnumber,
                         filenames=filenames)
        return sms

    @staticmethod
    def ftype():
        return 'SMS'

    @staticmethod
    def defaultunit():
        return 32
