"""
mfpval module.  Contains the ModflowPval class. Note that the user can access
the ModflowPval class as `flopy.modflow.ModflowPval`.

Additional information for this MODFLOW package can be found at the `Online
MODFLOW Guide
<https://water.usgs.gov/ogw/modflow-nwt/MODFLOW-NWT-Guide/parameter_value_file.html>`_.

"""
from ..pakbase import Package


class ModflowPval(Package):
    """
    MODFLOW Mult Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.mf.Modflow`) to which
        this package will be added.
    pval_dict : dict
        Dictionary with pval data for the model. pval_dict is typically
        instantiated using load method.
    extension : string
        Filename extension (default is 'pval')
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
    Parameters are supported in Flopy only when reading in existing models.
    Parameter values are converted to native values in Flopy and the
    connection to "parameters" is thus nonexistent.

    Examples
    --------

    >>> import flopy
    >>> m = flopy.modflow.Modflow()
    >>> pval_dict = flopy.modflow.ModflowZon(m, pval_dict=pval_dict)

    """

    def __init__(
        self,
        model,
        pval_dict=None,
        extension="pval",
        unitnumber=None,
        filenames=None,
    ):
        # set default unit number of one is not specified
        if unitnumber is None:
            unitnumber = ModflowPval._defaultunit()

        # call base package constructor
        super().__init__(
            model,
            extension=extension,
            name=self._ftype(),
            unit_number=unitnumber,
            filenames=self._prepare_filenames(filenames),
        )

        self._generate_heading()
        self.url = "parameter_value_file.html"

        self.npval = 0
        if pval_dict is not None:
            self.pval = len(pval_dict)
            self.pval_dict = pval_dict
        self.parent.add_package(self)

    def write_file(self):
        """
        Write the package file.

        Returns
        -------
        None

        Notes
        -----
        Not implemented because parameters are only supported on load

        """
        pass

    def __getitem__(self, item):
        """
        overload __getitem__ to return a value from the pval_dict

        """

        if item in list(self.pval_dict.keys()):
            return self.pval_dict[item]
        else:
            return None

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
        pval : ModflowPval dict

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow()
        >>> mlt = flopy.modflow.ModflowPval.load('test.pval', m)

        """

        if model.verbose:
            print("loading pval package file...")

        openfile = not hasattr(f, "read")
        if openfile:
            filename = f
            f = open(filename, "r")
        else:
            filename = f.name

        # dataset 0 -- header
        while True:
            line = f.readline()
            if line[0] != "#":
                break
        # dataset 1
        t = line.strip().split()
        npval = int(t[0])

        if model.verbose:
            print(f'   reading parameter values from "{filename}"')

        # read PVAL data
        pval_dict = dict()
        for n in range(npval):
            line = f.readline()
            t = line.strip().split()
            if len(t[0]) > 10:
                pvalnam = t[0][0:10].lower()
            else:
                pvalnam = t[0].lower()

            pval_dict[pvalnam] = float(t[1])

        if openfile:
            f.close()

        # set package unit number
        unitnumber = None
        filenames = [None]
        if ext_unit_dict is not None:
            unitnumber, filenames[0] = model.get_ext_dict_attr(
                ext_unit_dict, filetype=ModflowPval._ftype()
            )

        return cls(
            model,
            pval_dict=pval_dict,
            unitnumber=unitnumber,
            filenames=filenames,
        )

    @staticmethod
    def _ftype():
        return "PVAL"

    @staticmethod
    def _defaultunit():
        return 1005
