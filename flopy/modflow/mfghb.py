"""
mfghb module.  Contains the ModflowGhb class. Note that the user can access
the ModflowGhb class as `flopy.modflow.ModflowGhb`.

Additional information for this MODFLOW package can be found at the `Online
MODFLOW Guide
<http://water.usgs.gov/ogw/modflow/MODFLOW-2005-Guide/index.html?ghb.htm>`_.

"""
import sys
import numpy as np
from ..pakbase import Package
from ..utils import MfList
from ..utils.recarray_utils import create_empty_recarray


class ModflowGhb(Package):
    """
    MODFLOW General-Head Boundary Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.mf.Modflow`) to which
        this package will be added.
    ipakcb : int
        A flag that is used to determine if cell-by-cell budget data should be
        saved. If ipakcb is non-zero cell-by-cell budget data will be saved.
        (default is 0).
    stress_period_data : list of boundaries, recarray of boundaries or,
        dictionary of boundaries.

        Each ghb cell is defined through definition of
        layer(int), row(int), column(int), stage(float), conductance(float)
        The simplest form is a dictionary with a lists of boundaries for each
        stress period, where each list of boundaries itself is a list of
        boundaries. Indices of the dictionary are the numbers of the stress
        period. This gives the form of::

            stress_period_data =
            {0: [
                [lay, row, col, stage, cond],
                [lay, row, col, stage, cond],
                [lay, row, col, stage, cond],
                ],
            1:  [
                [lay, row, col, stage, cond],
                [lay, row, col, stage, cond],
                [lay, row, col, stage, cond],
                ], ...
            kper:
                [
                [lay, row, col, stage, cond],
                [lay, row, col, stage, cond],
                [lay, row, col, stage, cond],
                ]
            }

        Note that if no values are specified for a certain stress period, then
        the list of boundaries for the previous stress period for which values
        were defined is used. Full details of all options to specify
        stress_period_data can be found in the flopy3boundaries Notebook in
        the basic subdirectory of the examples directory
    dtype : dtype definition
        if data type is different from default
    options : list of strings
        Package options. (default is None).
    extension : string
        Filename extension (default is 'ghb')
    unitnumber : int
        File unit number (default is None).
    filenames : str or list of str
        Filenames to use for the package and the output files. If
        filenames=None the package name will be created using the model name
        and package extension and the cbc output name will be created using
        the model name and .cbc extension (for example, modflowtest.cbc),
        if ipakcbc is a number greater than zero. If a single string is passed
        the package will be set to the string and cbc output names will be
        created using the model name and .cbc extension, if ipakcbc is a
        number greater than zero. To define the names for all package files
        (input and output) the length of the list of strings should be 2.
        Default is None.

    Attributes
    ----------

    Methods
    -------

    See Also
    --------

    Notes
    -----
    Parameters are not supported in FloPy.

    Examples
    --------

    >>> import flopy
    >>> ml = flopy.modflow.Modflow()
    >>> lrcsc = {0:[2, 3, 4, 10., 100.]}  #this ghb will be applied to all
    >>>                                   #stress periods
    >>> ghb = flopy.modflow.ModflowGhb(ml, stress_period_data=lrcsc)

    """

    def __init__(
        self,
        model,
        ipakcb=None,
        stress_period_data=None,
        dtype=None,
        no_print=False,
        options=None,
        extension="ghb",
        unitnumber=None,
        filenames=None,
    ):
        """
        Package constructor.

        """

        # set default unit number of one is not specified
        if unitnumber is None:
            unitnumber = ModflowGhb._defaultunit()

        # set filenames
        if filenames is None:
            filenames = [None, None]
        elif isinstance(filenames, str):
            filenames = [filenames, None]
        elif isinstance(filenames, list):
            if len(filenames) < 2:
                filenames.append(None)

        # update external file information with cbc output, if necessary
        if ipakcb is not None:
            fname = filenames[1]
            model.add_output_file(
                ipakcb, fname=fname, package=ModflowGhb._ftype()
            )
        else:
            ipakcb = 0

        # Fill namefile items
        name = [ModflowGhb._ftype()]
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

        self.heading = (
            "# {} package for ".format(self.name[0])
            + " {}, ".format(model.version_types[model.version])
            + "generated by Flopy."
        )
        self.url = "ghb.htm"

        self.ipakcb = ipakcb
        self.no_print = no_print
        self.np = 0
        if options is None:
            options = []
        if self.no_print:
            options.append("NOPRINT")
        self.options = options
        self.parent.add_package(self)
        if dtype is not None:
            self.dtype = dtype
        else:
            self.dtype = self.get_default_dtype(
                structured=self.parent.structured
            )
        self.stress_period_data = MfList(self, stress_period_data)

    def _ncells(self):
        """Maximum number of cells that have general head boundaries
        (developed for MT3DMS SSM package).

        Returns
        -------
        ncells: int
            maximum number of ghb cells

        """
        return self.stress_period_data.mxact

    def write_file(self, check=True):
        """
        Write the package file.

        Parameters
        ----------
        check : boolean
            Check package data for common errors. (default True)

        Returns
        -------
        None

        """
        if (
            check
        ):  # allows turning off package checks when writing files at model level
            self.check(
                f="{}.chk".format(self.name[0]),
                verbose=self.parent.verbose,
                level=1,
            )
        f_ghb = open(self.fn_path, "w")
        f_ghb.write("{}\n".format(self.heading))
        f_ghb.write(
            "{:10d}{:10d}".format(self.stress_period_data.mxact, self.ipakcb)
        )
        for option in self.options:
            f_ghb.write("  {}".format(option))
        f_ghb.write("\n")
        self.stress_period_data.write_transient(f_ghb)
        f_ghb.close()

    def add_record(self, kper, index, values):
        try:
            self.stress_period_data.add_record(kper, index, values)
        except Exception as e:
            raise Exception("mfghb error adding record to list: " + str(e))

    @staticmethod
    def get_empty(ncells=0, aux_names=None, structured=True):
        # get an empty recarray that corresponds to dtype
        dtype = ModflowGhb.get_default_dtype(structured=structured)
        if aux_names is not None:
            dtype = Package.add_to_dtype(dtype, aux_names, np.float32)
        return create_empty_recarray(ncells, dtype, default_value=-1.0e10)

    @staticmethod
    def get_default_dtype(structured=True):
        if structured:
            dtype = np.dtype(
                [
                    ("k", int),
                    ("i", int),
                    ("j", int),
                    ("bhead", np.float32),
                    ("cond", np.float32),
                ]
            )
        else:
            dtype = np.dtype(
                [("node", int), ("bhead", np.float32), ("cond", np.float32)]
            )
        return dtype

    @staticmethod
    def _get_sfac_columns():
        return ["cond"]

    @classmethod
    def load(cls, f, model, nper=None, ext_unit_dict=None, check=True):
        """
        Load an existing package.

        Parameters
        ----------
        f : filename or file handle
            File to load.
        model : model object
            The model object (of type :class:`flopy.modflow.mf.Modflow`) to
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
        check : boolean
            Check package data for common errors. (default True)

        Returns
        -------
        ghb : ModflowGhb object
            ModflowGhb object.

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow()
        >>> ghb = flopy.modflow.ModflowGhb.load('test.ghb', m)

        """

        if model.verbose:
            sys.stdout.write("loading ghb package file...\n")

        return Package.load(
            f,
            model,
            cls,
            nper=nper,
            check=check,
            ext_unit_dict=ext_unit_dict,
        )

    @staticmethod
    def _ftype():
        return "GHB"

    @staticmethod
    def _defaultunit():
        return 23
