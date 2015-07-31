"""
mfnwt module.  Contains the ModflowNwt class. Note that the user can access
the ModflowNwt class as `flopy.modflow.ModflowNwt`.

Additional information for this MODFLOW package can be found at the `Online
MODFLOW Guide
<http://water.usgs.gov/ogw/modflow-nwt/MODFLOW-NWT-Guide/nwt_newton_solver.htm>`_.

"""

import sys
from flopy.mbase import Package

class ModflowNwt(Package):
    """
    MODFLOW Nwt Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.mf.Modflow`) to which
        this package will be added.
    headtol : float
        is the maximum head change between outer iterations for solution of the
        nonlinear problem. (default is 1e-4).
    fluxtol : float
        is the maximum l2 norm for solution of the nonlinear problem.
        (default is 500).
    maxiterout : int
        is the maximum number of iterations to be allowed for solution of the
        outer (nonlinear) problem. (default is 100).
    thickfact : float
        is the portion of the cell thickness (length) used for smoothly
        adjusting storage and conductance coefficients to zero.
        (default is 1e-5).
    linmeth : int
        is a flag that determines which matrix solver will be used.
        A value of 1 indicates GMRES will be used
        A value of 2 indicates XMD will be used.
        (default is 1).
    iprnwt : int
        is a flag that indicates whether additional information about solver
        convergence will be printed to the main listing file.
        (default is 0).
    ibotavg : int
        is a flag that indicates whether corrections will be made to groundwater
        head relative to the cell-bottom altitude if the cell is surrounded by
        dewatered cells (integer). A value of 1 indicates that a correction will
        be made and a value of 0 indicates no correction will be made.
        (default is 0).
    options : string
        SPECIFIED indicates that the optional solver input values listed for items 1
        and 2 will be specified in the NWT input file by the user.
        SIMPLE indicates that default solver input values will be defined that work
        well for nearly linear models. This would be used for models that do not
        include nonlinear stress packages, and models that are either confined or
        consist of a single unconfined layer that is thick enough to contain the
        water table within a single layer.
        MODERATE indicates that default solver input values will be defined that work
        well for moderately nonlinear models. This would be used for models that include
        nonlinear stress packages, and models that consist of one or more unconfined
        layers. The MODERATE option should be used when the SIMPLE option does not
        result in successful convergence.
        COMPLEX indicates that default solver input values will be defined that work
        well for highly nonlinear models. This would be used for models that include
        nonlinear stress packages, and models that consist of one or more unconfined
        layers representing complex geology and sw/gw interaction. The COMPLEX option
        should be used when the MODERATE option does not result in successful
        convergence.
        (default is COMPLEX).
    continue : bool
        if the model fails to converge during a time step then it will continue to
        solve the following time step. (default is False).
    dbtheta : float
        is a coefficient used to reduce the weight applied to the head change between
        nonlinear iterations. DBDTHETA is used to control oscillations in head.
        Values range between 0.0 and 1.0, and larger values increase the weight
        (decrease under-relaxation) applied to the head change. (default is 0.4).
    dbkappa : float
        is a coefficient used to increase the weight applied to the head change between
        nonlinear iterations. DBDKAPPA is used to control oscillations in head. Values
        range between 0.0 and 1.0, and larger values increase the weight applied to the
        head change. (default is 1.e-5).
    dbgamma : float
        is a factor (used to weight the head change for the previous and current
        iteration. Values range between 0.0 and 1.0, and greater values apply more weight
        to the head change calculated during the current iteration. (default is 0.)
    momfact : float
        is the momentum coefficient and ranges between 0.0 and 1.0. Greater values apply
        more weight to the head change for the current iteration. (default is 0.1).
    backflag : int
        s a flag used to specify whether residual control will be used. A value of 1
        indicates that residual control is active and a value of 0 indicates residual
        control is inactive. (default is 1).
    maxbackiter : int
        is the maximum number of reductions (backtracks) in the head change between
        nonlinear iterations (integer). A value between 10 and 50 works well.
        (default is 50).
    backtol : float
        is the proportional decrease in the root-mean-squared error of the groundwater-
        flow equation used to determine if residual control is required at the end of
        a nonlinear iteration. (default is 1.1).
    backreduce : float
        is a reduction factor used for residual control that reduces the head change
        between nonlinear iterations. Values should be between 0.0 and 1.0, where
        smaller values result in smaller head-change values. (default 0.7).
    maxitinner : int
        (GMRES) is the maximum number of iterations for the linear solution.
        (default is 50).
    ilumethod : int
        (GMRES) is the index for selection of the method for incomplete factorization
        (ILU) used as a preconditioner. (default is 2).
        ILUMETHOD=1 is ILU with drop tolerance and fill limit. Fill-in terms less
        than drop tolerance times the diagonal are discarded. The number of fill-in
        terms in each row of L and U is limited to the fill limit. The fill-limit
        largest elements are kept in the L and U factors.
        ILUMETHOD=2 is ILU(k) order k incomplete LU factorization. Fill-in terms of
        higher order than k in the factorization are discarded.
    levfill : int
        (GMRES) is the fill limit for ILUMETHOD = 1 and is the level of fill for
        ILUMETHOD = 2. Recommended values: 5-10 for method 1, 0-2 for method 2.
        (default is 5).
    stoptol : float
        (GMRES) is the tolerance for convergence of the linear solver. This is the
        residual of the linear equations scaled by the norm of the root mean squared
        error. Usually 1.e-8 to 1.e-12 works well. (default is 1.e-10).
    msdr : int
        (GMRES) is the number of iterations between restarts of the GMRES Solver.
        (default is 15).
    iacl : int
        (XMD) is a flag for the acceleration method: 0 is conjugate gradient, 1 is ORTHOMIN,
        2 is Bi-CGSTAB. (default is 2).
    norder : int
        (XMD) is a flag for the scheme of ordering the unknowns: 0 is original ordering,
        1 is RCM ordering, 2 is Minimum Degree ordering. (default is 1).
    level : int
        (XMD) is the level of fill for incomplete LU factorization. (default is 5).
    north : int
        (XMD) is the number of orthogonalization for the ORTHOMIN acceleration scheme.
        A number between 4 and 10 is appropriate. Small values require less storage
        but more iterations may be required. This number should equal 2 for the other
        acceleration methods. (default is 7).
    iredsys : int
        (XMD) is a flag for reduced system preconditioning (integer): 0-do not apply
        reduced system preconditioning, 1-apply reduced system preconditioning.
        (default is 0)
    rrctols : int
        (XMD) is the residual reduction-convergence criteria. (default is 0.).
    idroptol : int
        (XMD) is a flag for using drop tolerance in the preconditioning: 0-don't
        use drop tolerance, 1-use drop tolerance. (default is 1).
    epsrn : float
        (XMD) is the drop tolerance for preconditioning. (default is 1.e-4).
    hclosexmd : float
        (XMD) is the head closure criteria for inner (linear) iterations.
        (default is 1.e-4).
    mxiterxmd : int
        (XMD) is the maximum number of iterations for the linear solution.
        (default is 50).
    extension : list string
        Filename extension (default is 'nwt')
    unitnumber : int
        File unit number (default is 32).

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
    >>> nwt = flopy.modflow.ModflowNwt(m)

    """
    def __init__(self, model, headtol = 1E-4, fluxtol = 500, maxiterout = 100, \
                 thickfact = 1E-5, linmeth = 1, iprnwt = 0, ibotav = 0, \
                 options = 'COMPLEX', Continue=False, \
                 dbtheta=0.4, dbkappa=1.e-5, dbgamma=0., momfact=0.1, \
                 backflg=1, maxbackiter=50, backtol=1.1, backreduce=0.70, \
                 maxitinner=50, ilumethod=2, levfill=5, stoptol=1.e-10, msdr=15, \
                 iacl=2, norder=1, level=5, north=7, iredsys=0, rrctols=0.0, \
                 idroptol=1, epsrn=1.e-4, hclosexmd=1e-4, mxiterxmd=50, \
                 extension='nwt', unitnumber = 32):
        Package.__init__(self, model, extension, 'NWT', unitnumber) # Call ancestor's init to set self.parent, extension, name and unit number
        self.heading = '# NWT for MODFLOW-NWT, generated by Flopy.'
        self.url = 'nwt_newton_solver.htm'
        self.headtol = headtol
        self.fluxtol = fluxtol
        self.maxiterout = maxiterout
        self.thickfact = thickfact
        self.linmeth = linmeth
        self.iprnwt = iprnwt
        self.ibotav = ibotav
        if isinstance(options, list):
            self.options = options
        else:
            self.options = [options.upper()]
        if Continue:
            self.options.append('CONTINUE')
        self.dbtheta = dbtheta
        self.dbkappa = dbkappa
        self.dbgamma = dbgamma
        self.momfact = momfact
        self.backflg = backflg
        self.maxbackiter = maxbackiter
        self.backtol = backtol
        self.backreduce = backreduce
        self.maxitinner = maxitinner
        self.ilumethod = ilumethod
        self.levfill = levfill
        self.stoptol = stoptol
        self.msdr = msdr
        self.iacl = iacl
        self.norder = norder
        self.level = level
        self.north = north
        self.iredsys = iredsys
        self.rrctols = rrctols
        self.idroptol = idroptol
        self.epsrn = epsrn
        self.hclosexmd = hclosexmd
        self.mxiterxmd = mxiterxmd
        self.parent.add_package(self)

    def __repr__( self ):
        return 'Newton solver package class'


    def write_file(self):
        """
        Write the package input file.

        """
        # Open file for writing
        f = open(self.fn_path, 'w')
        f.write('%s\n' % self.heading)
        f.write('%10.1e%10.1e%10i%10.1e%10i%10i%10i ' % (self.headtol, self.fluxtol, self.maxiterout, self.thickfact, self.linmeth, self.iprnwt, self.ibotav))
        isspecified = False
        for option in self.options:
            f.write('{0} '.format(option.upper()))
            if option.lower() == 'specified':
                isspecified = True
        if isspecified:
            f.write('{0:10.4g}'.format(self.dbtheta))
            f.write('{0:10.4g}'.format(self.dbkappa))
            f.write('{0:10.4g}'.format(self.dbgamma))
            f.write('{0:10.4g}'.format(self.momfact))
            f.write('{0:10d}'.format(self.backflg))
            if self.backflg > 0:
                f.write('{0:10d}'.format(self.maxbackiter))
                f.write('{0:10.4g}'.format(self.backtol))
                f.write('{0:10.4g}'.format(self.backreduce))
            f.write('\n')
            if self.linmeth == 1:
                f.write('{0:10d}'.format(self.maxitinner))
                f.write('{0:10d}'.format(self.ilumethod))
                f.write('{0:10d}'.format(self.levfil))
                f.write('{0:10.4g}'.format(self.stoptol))
                f.write('{0:10d}'.format(self.msdr))
            elif self.linmeth == 2:
                f.write('{0:10d}'.format(self.iacl))
                f.write('{0:10d}'.format(self.norder))
                f.write('{0:10d}'.format(self.level))
                f.write('{0:10d}'.format(self.north))
                f.write('{0:10d}'.format(self.iredsys))
                f.write('{0:10.4g}'.format(self.rrctols))
                f.write('{0:10d}'.format(self.idroptol))
                f.write('{0:10.4g}'.format(self.epsrn))
                f.write('{0:10.4g}'.format(self.hclosexmd))
                f.write('{0:10d}'.format(self.mxiterxmd))

        f.write('\n')
                
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
        nwt : ModflowNwt object

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow()
        >>> nwt = flopy.modflow.ModflowPcg.load('test.nwt', m)

        """

        if model.verbose:
            sys.stdout.write('loading nwt package file...\n')

        if not hasattr(f, 'read'):
            filename = f
            f = open(filename, 'r')
        #dataset 0 -- header

        print('  ...load method not completed. default nwt file created.')

        # close the open file
        f.close()

        nwt = ModflowNwt(model)
        return nwt
