"""
mfsor module.  Contains the ModflowSor class. Note that the user can access
the ModflowSor class as `flopy.modflow.ModflowSor`.

Additional information for this MODFLOW package can be found at the `Online
MODFLOW Guide
<http://water.usgs.gov/nrp/gwsoftware/modflow2000/Guide/sor.htm>`_.

"""
from ..pakbase import Package


class ModflowSor(Package):
    """
    MODFLOW Slice-successive overrelaxation Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:flopy.modflow.mf.Modflow) to which
        this package will be added.
    mxiter : integer
        The maximum number of iterations allowed in a time step.
        (default is 200)
    accl : float
        The acceleration variable, which must be greater than zero
        and is generally between 1. and 2. (default is 1)
    hclose : float > 0
        The head change criterion for convergence. When the maximum absolute
        value of head change from all nodes during an iteration is less than
        or equal to hclose, iteration stops. (default is 1e-5)
    iprsor : integer > 0
        the printout interval for sor. iprsor, if equal to zero, is changed to
        999. The maximum head change (positive or negative) is printed for each
        iteration of a time step whenever the time step is an even multiple of
        iprsor. This printout also occurs at the end of each stress period
        regardless of the value of iprsor. (default is 0)
    extension : string
        Filename extension (default is 'sor')
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
    >>> ml = flopy.modflow.Modflow()
    >>> sor = flopy.modflow.ModflowSor(ml)

    """

    def __init__(
        self,
        model,
        mxiter=200,
        accl=1,
        hclose=1e-5,
        iprsor=0,
        extension="sor",
        unitnumber=None,
        filenames=None,
    ):
        """
        Package constructor.

        """
        # set default unit number of one is not specified
        if unitnumber is None:
            unitnumber = ModflowSor._defaultunit()

        # set filenames
        if filenames is None:
            filenames = [None]
        elif isinstance(filenames, str):
            filenames = [filenames]

        # Fill namefile items
        name = [ModflowSor._ftype()]
        units = [unitnumber]
        extra = [""]

        # set package name
        fname = [filenames[0]]

        # Call ancestor's init to set self.parent, extension, name and
        # unit number
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
        if model.version != "mf2k":
            raise Exception(
                f"Error: cannot use {self.name} "
                f"package with model version {model.version}"
            )

        self._generate_heading()
        self.url = "sor.htm"
        self.mxiter = mxiter
        self.accl = accl
        self.hclose = hclose
        self.iprsor = iprsor
        self.parent.add_package(self)

    def write_file(self):
        """
        Write the package file.

        Returns
        -------
        None

        """
        # Open file for writing
        f = open(self.fn_path, "w")
        f.write(f"{self.heading}\n")
        f.write(f"{self.mxiter:10d}\n")
        line = f"{self.accl:10.4g}{self.hclose:10.4g}{self.iprsor:10d}\n"
        f.write(line)
        f.close()

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
        sor : ModflowSor object

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow()
        >>> sor = flopy.modflow.ModflowSor.load('test.sor', m)

        """

        if model.verbose:
            print("loading sor package file...")

        openfile = not hasattr(f, "read")
        if openfile:
            filename = f
            f = open(filename, "r")

        # dataset 0 -- header

        print(
            "   Warning: load method not completed. "
            "Default sor object created."
        )

        if openfile:
            f.close()

        # set package unit number
        unitnumber = None
        filenames = [None]
        if ext_unit_dict is not None:
            unitnumber, filenames[0] = model.get_ext_dict_attr(
                ext_unit_dict, filetype=ModflowSor._ftype()
            )

        # return sor object
        return cls(model, unitnumber=unitnumber, filenames=filenames)

    @staticmethod
    def _ftype():
        return "SOR"

    @staticmethod
    def _defaultunit():
        return 26
