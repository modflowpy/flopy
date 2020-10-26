import sys
from ..pakbase import Package


class Mt3dGcg(Package):
    """
    MT3DMS Generalized Conjugate Gradient Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.mt3d.mt.Mt3dms`) to which
        this package will be added.
    mxiter : int
        is the maximum number of outer iterations; it should be set to an
        integer greater than one only when a nonlinear sorption isotherm is
        included in simulation. (default is 1)
    iter1 : int
        is the maximum number of inner iterations; a value of 30-50 should be
        adequate for most problems. (default is 50)
    isolve : int
        is the type of preconditioners to be used with the Lanczos/ORTHOMIN
        acceleration scheme:
        = 1, Jacobi
        = 2, SSOR
        = 3, Modified Incomplete Cholesky (MIC) (MIC usually converges faster,
        but it needs significantly more memory)
        (default is 3)
    ncrs : int
        is an integer flag for treatment of dispersion tensor cross terms:
        = 0, lump all dispersion cross terms to the right-hand-side
        (approximate but highly efficient). = 1, include full dispersion
        tensor (memory intensive).
        (default is 0)
    accl : float
        is the relaxation factor for the SSOR option; a value of 1.0 is
        generally adequate.
        (default is 1)
    cclose : float
        is the convergence criterion in terms of relative concentration; a
        real value between 10-4 and 10-6 is generally adequate.
        (default is 1.E-5)
    iprgcg : int
        IPRGCG is the interval for printing the maximum concentration changes
        of each iteration. Set IPRGCG to zero as default for printing at the
        end of each stress period.
        (default is 0)
    extension : string
        Filename extension (default is 'gcg')
    unitnumber : int
        File unit number (default is None).
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
    >>> m = flopy.mt3d.Mt3dms()
    >>> gcg = flopy.mt3d.Mt3dGcg(m)

    """

    unitnumber = 35

    def __init__(
        self,
        model,
        mxiter=1,
        iter1=50,
        isolve=3,
        ncrs=0,
        accl=1,
        cclose=1e-5,
        iprgcg=0,
        extension="gcg",
        unitnumber=None,
        filenames=None,
    ):

        if unitnumber is None:
            unitnumber = Mt3dGcg._defaultunit()
        elif unitnumber == 0:
            unitnumber = Mt3dGcg._reservedunit()

        # set filenames
        if filenames is None:
            filenames = [None]
        elif isinstance(filenames, str):
            filenames = [filenames]

        # Fill namefile items
        name = [Mt3dGcg._ftype()]
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

        self.mxiter = mxiter
        self.iter1 = iter1
        self.isolve = isolve
        self.ncrs = ncrs
        self.accl = accl
        self.cclose = cclose
        self.iprgcg = iprgcg
        self.parent.add_package(self)
        return

    def write_file(self):
        """
        Write the package file

        Returns
        -------
        None

        """
        # Open file for writing
        f_gcg = open(self.fn_path, "w")
        f_gcg.write(
            "{} {} {} {}\n".format(
                self.mxiter, self.iter1, self.isolve, self.ncrs
            )
        )
        f_gcg.write("{} {} {}\n".format(self.accl, self.cclose, self.iprgcg))
        f_gcg.close()
        return

    @classmethod
    def load(cls, f, model, ext_unit_dict=None):
        """
        Load an existing package.

        Parameters
        ----------
        f : filename or file handle
            File to load.
        model : model object
            The model object (of type :class:`flopy.mt3d.mt.Mt3dms`) to
            which this package will be added.
        ext_unit_dict : dictionary, optional
            If the arrays in the file are specified using EXTERNAL,
            or older style array control records, then `f` should be a file
            handle.  In this case ext_unit_dict is required, which can be
            constructed using the function
            :class:`flopy.utils.mfreadnam.parsenamefile`.

        Returns
        -------
        gcg :  Mt3dGcg object
            Mt3dGcg object.

        Examples
        --------

        >>> import flopy
        >>> mt = flopy.mt3d.Mt3dms()
        >>> gcg = flopy.mt3d.Mt3dGcg.load('test.gcg', m)

        """

        if model.verbose:
            sys.stdout.write("loading gcg package file...\n")

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

        # Item F1: MIXELM, PERCEL, MXPART, NADVFD - line already read above
        if model.verbose:
            print("   loading MXITER, ITER1, ISOLVE, NCRS...")
        t = line.strip().split()
        mxiter = int(t[0])
        iter1 = int(t[1])
        isolve = int(t[2])
        ncrs = int(t[3])
        if model.verbose:
            print("   MXITER {}".format(mxiter))
            print("   ITER1 {}".format(iter1))
            print("   ISOLVE {}".format(isolve))
            print("   NCRS {}".format(ncrs))

        # Item F2: ACCL, CCLOSE, IPRGCG
        if model.verbose:
            print("   loading ACCL, CCLOSE, IPRGCG...")
        line = f.readline()
        t = line.strip().split()
        accl = float(t[0])
        cclose = float(t[1])
        iprgcg = int(t[2])
        if model.verbose:
            print("   ACCL {}".format(accl))
            print("   CCLOSE {}".format(cclose))
            print("   IPRGCG {}".format(iprgcg))

        if openfile:
            f.close()

        # set package unit number
        unitnumber = None
        filenames = [None]
        if ext_unit_dict is not None:
            unitnumber, filenames[0] = model.get_ext_dict_attr(
                ext_unit_dict, filetype=Mt3dGcg._ftype()
            )

        # Construct and return gcg package
        return cls(
            model,
            mxiter=mxiter,
            iter1=iter1,
            isolve=isolve,
            ncrs=ncrs,
            accl=accl,
            cclose=cclose,
            iprgcg=iprgcg,
            unitnumber=unitnumber,
            filenames=filenames,
        )

    @staticmethod
    def _ftype():
        return "GCG"

    @staticmethod
    def _defaultunit():
        return 35

    @staticmethod
    def _reservedunit():
        return 9
