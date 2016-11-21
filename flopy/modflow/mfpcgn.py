"""
mfpcgn module.  Contains the ModflowPcgn class. Note that the user can access
the ModflowStr class as `flopy.modflow.ModflowPcgn`.

Additional information for this MODFLOW package can be found at the `Online
MODFLOW Guide
<http://water.usgs.gov/ogw/modflow/MODFLOW-2005-Guide/pcgn.htm>`_.

"""

import sys
from ..pakbase import Package


class ModflowPcgn(Package):
    """
    MODFLOW Pcgn Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.mf.Modflow`) to which
        this package will be added.
    iter_mo : int
        The maximum number of picard (outer) iterations allowed. For nonlinear
        problems, this variable must be set to some number greater than one,
        depending on the problem size and degree of nonlinearity. If iter_mo
        is set to 1, then the pcgn solver assumes that the problem is linear
        and the input requirements are greatly truncated. (default is 50)
    iter_mi : int
        maximum number of pcg (inner) iterations allowed. Generally,
        this variable is set to some number greater than one, depending on
        the matrix size, degree of convergence called for, and the nature of
        the problem. For a nonlinear problem, iter_mi should be set large
        enough that the pcg iteration converges freely with the relative
        convergence parameter epsilon described in the Parameters Related
        to Convergence of Inner Iteration: Line 4 subsection.
        (default is 30)
    close_r : float
        The residual-based stopping criterion for iteration. This parameter is
        used differently, depending on whether it is applied to a linear or
        nonlinear problem.

        If iter_mo = 1: For a linear problem, the variant of the conjugate
        gradient method outlined in algorithm 2 is employed, but uses the
        absolute convergence criterion in place of the relative convergence
        criterion. close_r is used as the value in the absolute convergence
        criterion for quitting the pcg iterative solver. close_r is compared
        to the square root of the weighted residual norm. In particular, if
        the square root of the weighted residual norm is less than close_r,
        then the linear Pcg iterative solve is said to have converged,
        causing the pcg iteration to cease and control of the program to
        pass out of the pcg solver.

        If iter_mo > 1: For a nonlinear problem, close_r is used as a criterion
        for quitting the picard (outer) iteration. close_r is compared to the
        square root of the inner product of the residuals (the residual norm)
        as calculated on entry to the pcg solver at the beginning of every
        picard iteration. if this norm is less than close_r, then the picard
        iteration is considered to have converged.
    close_h : float
        close_h is used as an alternate stopping criterion for the picard
        iteration needed to solve a nonlinear problem. The maximum value of
        the head change is obtained for each picard iteration, after completion
        of the inner, pcg iteration. If this maximum head change is less than
        close_h, then the picard iteration is considered tentatively to have
        converged. However, as nonlinear problems can demonstrate oscillation
        in the head solution, the picard iteration is not declared to have
        converged unless the maximum head change is less than close_h for
        three picard iterations. If these picard iterations are sequential,
        then a good solution is assumed to have been obtained. If the picard
        iterations are not sequential, then a warning is issued advising that
        the convergence is conditional and the user is urged to examine the
        mass balance of the solution.
    relax : float
        is the relaxation parameter used with npcond = 1. (default is 1.0)
    ifill : int
        is the fill level of the mic preconditioner. Preconditioners with
        fill levels of 0 and 1 are available (ifill = 0 and ifill = 1,
        respectively). (default is 0)
    unit_pc : int
        is the unit number of an optional output file where progress for the
        inner PCG iteration can be written. (default is 0)
    unit_ts : int
        is the unit number of an optional output file where the actual time in
        the PCG solver is accumulated. (default is 0)
    adamp : int
        defines the mode of damping applied to the linear solution. In general,
        damping determines how much of the head changes vector shall be applied
        to the hydraulic head vector hj in picard iteration j. If adamp = 0,
        Ordinary damping is employed and a constant value of damping parameter
        will be used throughout the picard iteration; this option requires a
        valid value for damp. If adamp = 1, Adaptive damping is employed. If
        adamp = 2: Enhanced damping algorithm in which the damping value is
        increased (but never decreased) provided the picard iteration is
        proceeding satisfactorily. (default is 0)
    damp : float
        is the damping factor. (default is 1.)
    damp_lb : float
        is the lower bound placed on the dampening; generally, 0 < damp_lb < damp.
        (default is 0.001)
    rate_d : float
        is a rate parameter; generally, 0 < rate_d < 1. (default is 0.1)
    chglimit : float
        this variable limits the maximum head change applicable to the updated
        hydraulic heads in a Picard iteration. If chglimit = 0.0, then adaptive
        damping proceeds without this feature. (default is 0.)
    acnvg : int
        defines the mode of convergence applied to the PCG solver. (default is 0)
    cnvg_lb : int
        is the minimum value that the relative convergence is allowed to take under
        the self-adjusting convergence option. cnvg_lb is used only in convergence
        mode acnvg = 1. (default is 0.001)
    mcnvg : float
        increases the relative PCG convergence criteria by a power equal to MCNVG.
        MCNVG is used only in convergence mode acnvg = 2. (default is 2)
    rate_c : float
        this option results in variable enhancement of epsilon. If 0 < rate_c < 1,
        then enhanced relative convergence is allowed to decrease by increasing
        epsilon(j) = epsilon(j-1) + rate_c epsilon(j-1), where j is the Picard
        iteration number; this change in epsilon occurs so long as the Picard
        iteration is progressing satisfactorily. If rate_c <= 0, then the value
        of epsilon set by mcnvg remains unchanged through the picard iteration.
        It should be emphasized that rate_c must have a value greater than 0
        for the variable enhancement to be effected; otherwise epsilon remains
        constant. rate_c is used only in convergence mode acnvg = 2.
        (default is -1.)
    ipunit : int
        enables progress reporting for the picard iteration. If ipunit >= 0,
        then a record of progress made by the picard iteration for each time
        step is printed in the MODFLOW Listing file (Harbaugh and others, 2000).
        This record consists of the total number of dry cells at the end of each
        time step as well as the total number of PCG iterations necessary to
        obtain convergence. In addition, if ipunit > 0, then extensive
        diagnostics for each Picard iteration is also written in comma-separated
        format to a file whose unit number corresponds to ipunit; the name for
        this file, along with its unit number and type 'data' should be entered
        in the modflow Name file. If ipunit < 0 then printing of all progress
        concerning the Picard iteration is suppressed, as well as information on
        the nature of the convergence of the picard iteration. (default is 0)
    extension : list string
        Filename extension (default is 'pcgn')
    unitnumber : int
        File unit number (default is 27).

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
    >>> pcgn = flopy.modflow.ModflowPcgn(m)

    """

    def __init__(self, model, iter_mo=50, iter_mi=30, close_r=1e-5, close_h=1e-5,
                 relax=1.0, ifill=0, unit_pc=0, unit_ts=0,
                 adamp=0, damp=1.0, damp_lb=0.001, rate_d=0.1, chglimit=0.,
                 acnvg=0, cnvg_lb=0.001, mcnvg=2, rate_c=-1.0, ipunit=0,
                 extension='pcgn', unitnumber=None):
        """
        Package constructor.

        """
        # set default unit number of one is not specified
        if unitnumber is None:
            unitnumber = ModflowDrn.defaultunit()

        if not isinstance(extension, list):
            extension = [extension]
        name = [ModflowPcgn.ftype()]
        units = [unitnumber]
        extra = ['']
        tu = (unit_pc, unit_ts, ipunit)
        ea = ('pcgni', 'pcgnt', 'pcgno')
        for [t, e] in zip(tu, ea):
            if t > 0:
                extension.append(e)
                name.append('DATA')
                units.append(t)
                extra.append('REPLACE')

        # Call ancestor's init to set self.parent, extension, name and unit number
        Package.__init__(self, model, extension=extension, name=name,
                         unit_number=units, extra=extra)

        # check if a valid model version has been specified
        if model.version == 'mfusg':
            err = 'Error: cannot use {} package with model version {}'.format(
                self.name, model.version)
            raise Exception(err)

        self.heading = '# PCGN for MODFLOW, generated by Flopy.'
        self.url = 'pcgn.htm'
        self.iter_mo = iter_mo
        self.iter_mi = iter_mi
        self.close_h = close_h
        self.close_r = close_r
        self.relax = relax
        self.ifill = ifill
        self.unit_pc = unit_pc
        self.unit_ts = unit_ts
        self.adamp = adamp
        self.damp = damp
        self.damp_lb = damp_lb
        self.rate_d = rate_d
        self.chglimit = chglimit
        self.acnvg = acnvg
        self.cnvg_lb = cnvg_lb
        self.mcnvg = mcnvg
        self.rate_c = rate_c
        self.ipunit = ipunit
        # error trapping
        if self.ifill < 0 or self.ifill > 1:
            raise TypeError('PCGN: ifill must be 0 or 1 - an ifill value of {0} was specified'.format(self.ifill))
        # add package
        self.parent.add_package(self)

    def write_file(self):
        """
        Write the package file.

        Returns
        -------
        None

        """
        # Open file for writing
        f_pcgn = open(self.fn_path, 'w')
        f_pcgn.write('{0:s}\n'.format(self.heading))
        f_pcgn.write(
            ' {0:9d} {1:9d} {2:9.3g} {3:9.3g}\n'.format(self.iter_mo, self.iter_mi, self.close_r, self.close_h))
        f_pcgn.write(' {0:9.3g} {1:9d} {2:9d} {3:9d}\n'.format(self.relax, self.ifill, self.unit_pc, self.unit_ts))
        f_pcgn.write(
            ' {0:9d} {1:9.3g} {2:9.3g} {3:9.3g} {4:9.3g}\n'.format(self.adamp, self.damp, self.damp_lb, self.rate_d,
                                                                   self.chglimit))
        f_pcgn.write(
            ' {0:9d} {1:9.3g} {2:9d} {3:9.3g} {4:9d}\n'.format(self.acnvg, self.cnvg_lb, self.mcnvg, self.rate_c,
                                                               self.ipunit))
        f_pcgn.close()

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
        pcgn : ModflowPcgn object

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow()
        >>> pcgn = flopy.modflow.ModflowPcgn.load('test.pcgn', m)

        """

        if model.verbose:
            sys.stdout.write('loading pcgn package file...\n')

        if not hasattr(f, 'read'):
            filename = f
            f = open(filename, 'r')
        # dataset 0 -- header

        print('   Warning: load method not completed. default pcgn object created.')

        # close the open file
        f.close()

        # determine specified unit number
        unitnumber = None
        if ext_unit_dict is not None:
            for key, value in ext_unit_dict.items():
                if value.filetype == ModflowPcgn.ftype():
                    unitnumber = key

        pcgn = ModflowPcgn(model, unitnumber=unitnumber)
        return pcgn


    @staticmethod
    def ftype():
        return 'PCGN'


    @staticmethod
    def defaultunit():
        return 27
