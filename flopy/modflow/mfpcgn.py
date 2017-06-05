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
        File unit number (default is None).
    filenames : str or list of str
        Filenames to use for the package and the output files. If
        filenames=None the package name will be created using the model name
        and package extension and the pcgn output names will be created using
        the model name and .pcgni, .pcgnt, and .pcgno extensions. If a single
        string is passed the package will be set to the string and pcgn output
        names will be created using the model name and pcgn output extensions.
        To define the names for all package files (input and output) the length
        of the list of strings should be 4. Default is None.

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

    def __init__(self, model, iter_mo=50, iter_mi=30, close_r=1e-5,
                 close_h=1e-5, relax=1.0, ifill=0, unit_pc=None, unit_ts=None,
                 adamp=0, damp=1.0, damp_lb=0.001, rate_d=0.1, chglimit=0.,
                 acnvg=0, cnvg_lb=0.001, mcnvg=2, rate_c=-1.0, ipunit=None,
                 extension='pcgn', unitnumber=None, filenames=None):
        """
        Package constructor.

        """
        # set default unit number of one is not specified
        if unitnumber is None:
            unitnumber = ModflowPcgn.defaultunit()

        # set filenames
        if filenames is None:
            filenames = [None, None, None, None]
        elif isinstance(filenames, str):
            filenames = [filenames, None, None, None]
        elif isinstance(filenames, list):
            if len(filenames) < 4:
                for idx in range(len(filenames), 4):
                    filenames.append(None)

        # update external file information with unit_pc output, if necessary
        if unit_pc is not None:
            fname = filenames[1]
            model.add_output_file(unit_pc, fname=fname, extension='pcgni',
                                  binflag=False,
                                  package=ModflowPcgn.ftype())
        else:
            unit_pc = 0

        # update external file information with unit_ts output, if necessary
        if unit_ts is not None:
            fname = filenames[2]
            model.add_output_file(unit_ts, fname=fname, extension='pcgnt',
                                  binflag=False,
                                  package=ModflowPcgn.ftype())
        else:
            unit_ts = 0

        # update external file information with ipunit output, if necessary
        if ipunit is not None:
            if ipunit > 0:
                fname = filenames[3]
                model.add_output_file(ipunit, fname=fname, extension='pcgno',
                                      binflag=False,
                                      package=ModflowPcgn.ftype())
        else:
            ipunit = -1

        name = [ModflowPcgn.ftype()]
        units = [unitnumber]
        extra = ['']

        # set package name
        fname = [filenames[0]]

        # Call ancestor's init to set self.parent, extension, name and unit number
        Package.__init__(self, model, extension=extension, name=name,
                         unit_number=units, extra=extra, filenames=fname)

        # check if a valid model version has been specified
        if model.version == 'mfusg':
            err = 'Error: cannot use {} package with model version {}'.format(
                self.name, model.version)
            raise Exception(err)

        self.heading = '# {} package for '.format(self.name[0]) + \
                       ' {}, '.format(model.version_types[model.version]) + \
                       'generated by Flopy.'
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
        f = open(self.fn_path, 'w')
        f.write('{0:s}\n'.format(self.heading))

        ifrfm = self.parent.get_ifrefm()
        if ifrfm:
            # dataset 1
            line = '{} '.format(self.iter_mo)
            line += '{} '.format(self.iter_mi)
            line += '{} '.format(self.close_r)
            line += '{}\n'.format(self.close_h)
            f.write(line)

            # dataset 2
            line = '{} '.format(self.relax)
            line += '{} '.format(self.ifill)
            line += '{} '.format(self.unit_pc)
            line += '{}\n'.format(self.unit_ts)
            f.write(line)

            # dataset 3
            line = '{} '.format(self.adamp)
            line += '{} '.format(self.damp)
            line += '{} '.format(self.damp_lb)
            line += '{} '.format(self.rate_d)
            line += '{}\n'.format(self.chglimit)
            f.write(line)

            # dataset 4
            line = '{} '.format(self.acnvg)
            line += '{} '.format(self.cnvg_lb)
            line += '{} '.format(self.mcnvg)
            line += '{} '.format(self.rate_c)
            line += '{}\n'.format(self.ipunit)
            f.write(line)

        else:
            # dataset 1
            sfmt = ' {0:9d} {1:9d} {2:9.3g} {3:9.3g}\n'
            line = sfmt.format(self.iter_mo, self.iter_mi, self.close_r,
                               self.close_h)
            f.write(line)

            # dataset 2
            sfmt = ' {0:9.3g} {1:9d} {2:9d} {3:9d}\n'
            line = sfmt.format(self.relax, self.ifill, self.unit_pc,
                               self.unit_ts)
            f.write(line)

            # dataset 3
            sfmt = ' {0:9d} {1:9.3g} {2:9.3g} {3:9.3g} {4:9.3g}\n'
            line = sfmt.format(self.adamp, self.damp, self.damp_lb,
                               self.rate_d, self.chglimit)
            f.write(line)

            # dataset 4
            sfmt = ' {0:9d} {1:9.3g} {2:9d} {3:9.3g} {4:9d}\n'
            line = sfmt.format(self.acnvg, self.cnvg_lb, self.mcnvg,
                               self.rate_c, self.ipunit)
            f.write(line)
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

        ifrefm = model.get_ifrefm()
        # dataset 0 -- header
        while True:
            line = f.readline()
            if line[0] != '#':
                break
        if ifrefm:
            # dataset 1
            t = line.strip().split()
            iter_mo = int(t[0])
            iter_mi = int(t[1])
            close_r = float(t[2])
            close_h = float(t[3])

            # dataset 2
            while True:
                line = f.readline()
                if line[0] != '#':
                    break
            t = line.strip().split()
            relax = float(t[0])
            ifill = int(t[1])
            unit_pc = int(t[2])
            unit_ts = int(t[3])

            # read datasets 3 and 4 for non-linear problems
            if (iter_mo) > 1:
                # dataset 3
                while True:
                    line = f.readline()
                    if line[0] != '#':
                        break
                t = line.strip().split()
                adamp = int(t[0])
                damp = float(t[1])
                damp_lb = float(t[2])
                rate_d = float(t[3])
                chglimit = float(t[4])

                # dataset 4
                while True:
                    line = f.readline()
                    if line[0] != '#':
                        break
                t = line.strip().split()
                acnvg = int(t[0])
                cnvg_lb = float(t[1])
                mcnvg = int(t[2])
                rate_c = float(t[3])
                ipunit = int(t[4])
        else:
            iter_mo = int(line[0:10].strip())
            iter_mi = int(line[10:20].strip())
            close_r = float(line[20:30].strip())
            close_h = float(line[30:40].strip())

            # dataset 2
            while True:
                line = f.readline()
                if line[0] != '#':
                    break
            relax = float(line[0:10].strip())
            ifill = int(line[10:20].strip())
            unit_pc = int(line[20:30].strip())
            unit_ts = int(line[30:40].strip())

            # read datasets 3 and 4 for non-linear problems
            if (iter_mo) > 1:
                # dataset 3
                while True:
                    line = f.readline()
                    if line[0] != '#':
                        break
                adamp = int(line[0:10].strip())
                damp = float(line[10:20].strip())
                damp_lb = float(line[20:30].strip())
                rate_d = float(line[30:40].strip())
                chglimit = float(line[40:50].strip())

                # dataset 4
                while True:
                    line = f.readline()
                    if line[0] != '#':
                        break
                acnvg = int(line[0:10].strip())
                cnvg_lb = float(line[10:20].strip())
                mcnvg = int(line[20:30].strip())
                rate_c = float(line[30:40].strip())
                ipunit = int(line[40:50].strip())

        if iter_mo == 1:
            adamp = None
            damp = None
            damp_lb = None
            rate_d = None
            chglimit = None
            acnvg = None
            cnvg_lb = None
            mcnvg = None
            rate_c = None
            ipunit = None

        # close the open file
        f.close()

        # determine specified unit number
        unitnumber = None
        filenames = [None, None, None, None]
        if ext_unit_dict is not None:
            unitnumber, filenames[0] = \
                model.get_ext_dict_attr(ext_unit_dict,
                                        filetype=ModflowPcgn.ftype())
            if unit_pc > 0:
                iu, filenames[1] = \
                    model.get_ext_dict_attr(ext_unit_dict, unit=unit_pc)
            if unit_ts > 0:
                iu, filenames[2] = \
                    model.get_ext_dict_attr(ext_unit_dict, unit=unit_ts)
            if ipunit > 0:
                iu, filenames[3] = \
                    model.get_ext_dict_attr(ext_unit_dict, unit=ipunit)

        pcgn = ModflowPcgn(model, iter_mo=iter_mo, iter_mi=iter_mi,
                           close_r=close_r, close_h=close_h, relax=relax,
                           ifill=ifill, unit_pc=unit_pc, unit_ts=unit_ts,
                           adamp=adamp, damp=damp, damp_lb=damp_lb,
                           rate_d=rate_d, chglimit=chglimit, acnvg=acnvg,
                           cnvg_lb=cnvg_lb, mcnvg=mcnvg, rate_c=rate_c,
                           ipunit=ipunit, unitnumber=unitnumber,
                           filenames=filenames)
        return pcgn


    @staticmethod
    def ftype():
        return 'PCGN'


    @staticmethod
    def defaultunit():
        return 27
