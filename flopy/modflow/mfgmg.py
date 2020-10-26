"""
mfgmg module.  Contains the ModflowGmg class. Note that the user can access
the ModflowGmg class as `flopy.modflow.ModflowGmg`.

Additional information for this MODFLOW package can be found at the `Online
MODFLOW Guide
<http://water.usgs.gov/ogw/modflow/MODFLOW-2005-Guide/gmg.htm>`_.

"""
import sys
from ..pakbase import Package


class ModflowGmg(Package):
    """
    MODFLOW GMG Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.mf.Modflow`) to which
        this package will be added.
    mxiter : int
        maximum number of outer iterations. (default is 50)
    iiter : int
        maximum number of inner iterations. (default is 30)
    iadamp : int
        is a flag that controls adaptive damping. The possible values
        of iadamp are.

        If iadamp = 0, then the value assigned to DAMP is used as a constant
        damping parameter.

        If iadamp = 1, the value of damp is used for the first nonlinear
        iteration. The damping parameter is adaptively varied on the basis
        of the head change, using Cooley's method as described in Mehl
        and Hill (2001), for subsequent iterations.

        If iadamp = 2, the relative reduced residual damping method documented
        in Mehl and Hill (2001) and modified by Banta (2006) is used.

        When iadamp is specified as 2 and the value specified for DAMP is less
        than 0.5, the closure criterion for the inner iterations (drclose) is
        assigned simply as rclose. When damp is between 0.5 and 1.0, inclusive,
        or when iadamp is specified as 0 or 1, drclose is calculated according
        to equation 20 on p. 9 of Wilson and Naff (2004).
    hclose : float
        is the head change criterion for convergence. (default is 1e-5).
    rclose : float
        is the residual criterion for convergence. (default is 1e-5)
    relax : float
        is a relaxation parameter for the ILU preconditioned conjugate
        gradient method. The relax parameter can be used to improve the
        spectral condition number of the ILU preconditioned system. The value
        of relax should be approximately one. However, the relaxation parameter
        can cause the factorization to break down. If this happens, then the
        gmg solver will report an assembly error and a value smaller than one
        for relax should be tried. This item is read only if isc = 4.
    ioutgmg : int
        is a flag that controls the output of the gmg solver. The
        possible values of ioutgmg are.

        If ioutgmg = 0, then only the solver inputs are printed.

        If ioutgmg = 1, then for each linear solve, the number of pcg
        iterations, the value of the damping parameter, the l2norm of
        the residual, and the maxnorm of the head change and its location
        (column, row, layer) are printed. At the end of a time/stress period,
        the total number of gmg calls, pcg iterations, and a running total
        of pcg iterations for all time/stress periods are printed.

        If ioutgmg = 2, then the convergence history of the pcg iteration is
        printed, showing the l2norm of the residual and the convergence factor
        for each iteration.

        ioutgmg = 3 is the same as ioutgmg = 1 except output is sent to the
        terminal instead of the modflow list output file.

        ioutgmg = 4 is the same as ioutgmg = 2 except output is sent to the
        terminal instead of the modflow list output file.

        (default is 0)
    iunitmhc : int
        is a flag and a unit number, which controls output of maximum
        head change values. If iunitmhc = 0, maximum head change values
        are not written to an output file. If iunitmhc > 0, maximum head
        change values are written to unit iunitmhc. Unit iunitmhc should
        be listed in the Name file with 'DATA' as the file type. If
        iunitmhc < 0 or is not present, iunitmhc defaults to 0.
        (default is 0)
    ism : int
        is a flag that controls the type of smoother used in the multigrid
        preconditioner. If ism = 0, then ilu(0) smoothing is implemented in
        the multigrid preconditioner; this smoothing requires an additional
        ector on each multigrid level to store the pivots in the ilu
        factorization. If ism = 1, then symmetric gaussseidel (sgs) smoothing
        is implemented in the multigrid preconditioner. No additional storage
        is required if ism = 1; users may want to use this option if available
        memory is exceeded or nearly exceeded when using ism = 0. Using sgs
        smoothing is not as robust as ilu smoothing; additional iterations are
        likely to be required in reducing the residuals. In extreme cases, the
        solver may fail to converge as the residuals cannot be reduced
        sufficiently. (default is 0)
    isc : int
        is a flag that controls semicoarsening in the multigrid
        preconditioner. If isc = 0, then the rows, columns and layers are
        all coarsened. If isc = 1, then the rows and columns are coarsened,
        but the layers are not. If isc = 2, then the columns and layers are
        coarsened, but the rows are not. If isc = 3, then the rows and layers
        are coarsened, but the columns are not. If isc = 4, then there is no
        coarsening. Typically, the value of isc should be 0 or 1. In the case
        that there are large vertical variations in the hydraulic
        conductivities, then a value of 1 should be used. If no coarsening is
        implemented (isc = 4), then the gmg solver is comparable to the pcg2
        ilu(0) solver described in Hill (1990) and uses the least amount of
        memory. (default is 0)
    damp : float
        is the value of the damping parameter. For linear problems, a value
        of 1.0 should be used. For nonlinear problems, a value less than 1.0
        but greater than 0.0 may be necessary to achieve convergence. A typical
        value for nonlinear problems is 0.5. Damping also helps control the
        convergence criterion of the linear solve to alleviate excessive pcg
        iterations. (default 1.)
    dup : float
        is the maximum damping value that should be applied at any iteration
        when the solver is not oscillating; it is dimensionless. An appropriate
        value for dup will be problem-dependent. For moderately nonlinear
        problems, reasonable values for dup would be in the range 0.5 to 1.0.
        For a highly nonlinear problem, a reasonable value for dup could be as
        small as 0.1. When the solver is oscillating, a damping value as large
        as 2.0 x DUP may be applied. (default is 0.75)
    dlow : float
        is the minimum damping value to be generated by the adaptive-damping
        procedure; it is dimensionless. An appropriate value for dlow will be
        problem-dependent and will be smaller than the value specified for dup.
        For a highly nonlinear problem, an appropriate value for dlow might be
        as small as 0.001. Note that the value specified for the variable,
        chglimit, could result in application of a damping value smaller than
        dlow. (default is 0.01)
    chglimit : float
        is the maximum allowed head change at any cell between outer
        iterations; it has units of length. The effect of chglimit is to
        determine a damping value that, when applied to all elements of the
        head-change vector, will produce an absolute maximum head change equal
        to chglimit. (default is 1.0)
    extension : list string
        Filename extension (default is 'gmg')
    unitnumber : int
        File unit number (default is None).
    filenames : str or list of str
        Filenames to use for the package and the output files. If
        filenames=None the package name will be created using the model name
        and package extension and the gmg output name will be created using
        the model name and .cbc extension (for example, modflowtest.gmg.out),
        if iunitmhc is a number greater than zero. If a single string is passed
        the package will be set to the string and gmg output names will be
        created using the model name and .gmg.out extension, if iunitmhc is a
        number greater than zero. To define the names for all package files
        (input and output) the length of the list of strings should be 2.
        Default is None.

    Returns
    -------
    None

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
    >>> gmg = flopy.modflow.ModflowGmg(m)


    """

    def __init__(
        self,
        model,
        mxiter=50,
        iiter=30,
        iadamp=0,
        hclose=1e-5,
        rclose=1e-5,
        relax=1.0,
        ioutgmg=0,
        iunitmhc=None,
        ism=0,
        isc=0,
        damp=1.0,
        dup=0.75,
        dlow=0.01,
        chglimit=1.0,
        extension="gmg",
        unitnumber=None,
        filenames=None,
    ):
        """
        Package constructor.

        """

        # set default unit number of one is not specified
        if unitnumber is None:
            unitnumber = ModflowGmg._defaultunit()

        # set filenames
        if filenames is None:
            filenames = [None, None]
        elif isinstance(filenames, str):
            filenames = [filenames, None]
        elif isinstance(filenames, list):
            if len(filenames) < 2:
                filenames.append(None)

        # update external file information with gmg output, if necessary
        if iunitmhc is not None:
            fname = filenames[1]
            model.add_output_file(
                iunitmhc,
                fname=fname,
                extension="gmg.out",
                binflag=False,
                package=ModflowGmg._ftype(),
            )
        else:
            iunitmhc = 0

        # Fill namefile items
        name = [ModflowGmg._ftype()]
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

        # check if a valid model version has been specified
        if model.version == "mfusg":
            err = "Error: cannot use {} package with model version {}".format(
                self.name, model.version
            )
            raise Exception(err)

        self.heading = (
            "# {} package for ".format(self.name[0])
            + " {}, ".format(model.version_types[model.version])
            + "generated by Flopy."
        )
        self.url = "gmg.htm"

        self.mxiter = mxiter
        self.iiter = iiter
        self.iadamp = iadamp
        self.hclose = hclose
        self.rclose = rclose
        self.relax = relax
        self.ism = ism
        self.isc = isc
        self.dup = dup
        self.dlow = dlow
        self.chglimit = chglimit
        self.damp = damp
        self.ioutgmg = ioutgmg
        self.iunitmhc = iunitmhc
        self.parent.add_package(self)

    def write_file(self):
        """
        Write the package file.

        Returns
        -------
        None

        """
        f_gmg = open(self.fn_path, "w")
        f_gmg.write("%s\n" % self.heading)
        # dataset 0
        f_gmg.write(
            "{} {} {} {}\n".format(
                self.rclose, self.iiter, self.hclose, self.mxiter
            )
        )
        # dataset 1
        f_gmg.write(
            "{} {} {} {}\n".format(
                self.damp, self.iadamp, self.ioutgmg, self.iunitmhc
            )
        )
        # dataset 2
        f_gmg.write("{} {} ".format(self.ism, self.isc))
        if self.iadamp == 2:
            f_gmg.write("{} {} {}".format(self.dup, self.dlow, self.chglimit))
        f_gmg.write("\n")
        # dataset 3
        f_gmg.write("{}\n".format(self.relax))
        f_gmg.close()

    @classmethod
    def load(cls, f, model, ext_unit_dict=None):
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
        gmg : ModflowGmg object

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow()
        >>> gmg = flopy.modflow.ModflowGmg.load('test.gmg', m)

        """

        if model.verbose:
            sys.stdout.write("loading gmg package file...\n")

        openfile = not hasattr(f, "read")
        if openfile:
            filename = f
            f = open(filename, "r")

        # dataset 0 -- header
        while True:
            line = f.readline()
            if line[0] != "#":
                break
        # dataset 0
        t = line.strip().split()
        rclose = float(t[0])
        iiter = int(t[1])
        hclose = float(t[2])
        mxiter = int(t[3])
        # dataset 1
        line = f.readline()
        t = line.strip().split()
        damp = float(t[0])
        iadamp = int(t[1])
        ioutgmg = int(t[2])
        try:
            iunitmhc = int(t[3])
        except:
            iunitmhc = 0
        # dataset 2
        line = f.readline()
        t = line.strip().split()
        ism = int(t[0])
        isc = int(t[1])
        dup, dlow, chglimit = 0.75, 0.01, 1.0
        if iadamp == 2:
            dup = float(t[2])
            dlow = float(t[3])
            chglimit = float(t[4])
        # dataset 3
        line = f.readline()
        t = line.strip().split()
        relax = float(t[0])

        if openfile:
            f.close()

        # determine specified unit number
        unitnumber = None
        filenames = [None, None]
        if ext_unit_dict is not None:
            unitnumber, filenames[0] = model.get_ext_dict_attr(
                ext_unit_dict, filetype=ModflowGmg._ftype()
            )
            if iunitmhc > 0:
                iu, filenames[1] = model.get_ext_dict_attr(
                    ext_unit_dict, unit=iunitmhc
                )
                model.add_pop_key_list(iunitmhc)

        return cls(
            model,
            mxiter=mxiter,
            iiter=iiter,
            iadamp=iadamp,
            hclose=hclose,
            rclose=rclose,
            relax=relax,
            ioutgmg=ioutgmg,
            iunitmhc=iunitmhc,
            ism=ism,
            isc=isc,
            damp=damp,
            dup=dup,
            dlow=dlow,
            chglimit=chglimit,
            unitnumber=unitnumber,
        )

    @staticmethod
    def _ftype():
        return "GMG"

    @staticmethod
    def _defaultunit():
        return 27
