"""
mfdrn module.  Contains the ModflowDrn class. Note that the user can access
the ModflowDrn class as `flopy.modflow.ModflowDrn`.

Additional information for this MODFLOW package can be found at the `Online
MODFLOW Guide
<http://water.usgs.gov/ogw/modflow/MODFLOW-2005-Guide/index.html?drn.htm>`_.

"""
import sys
import numpy as np
from ..pakbase import Package
from ..utils.util_list import MfList
from ..utils.recarray_utils import create_empty_recarray


class ModflowDrn(Package):
    """
    MODFLOW Drain Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.mf.Modflow`) to which
        this package will be added.
    ipakcb : int
        A flag that is used to determine if cell-by-cell budget data should be
        saved. If ipakcb is non-zero cell-by-cell budget data will be saved.
        (default is None).
    stress_period_data : list of boundaries, recarrays, or dictionary of
        boundaries.
        Each drain cell is defined through definition of
        layer(int), row(int), column(int), elevation(float),
        conductance(float).
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
        the basic subdirectory of the examples directory.
    dtype : dtype definition
        if data type is different from default
    options : list of strings
        Package options. (default is None).
    extension : string
        Filename extension (default is 'drn')
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
    If "RETURNFLOW" in passed in options, the drain return package (DRT) activated, which expects
    a different (longer) dtype for stress_period_data

    Examples
    --------

    >>> import flopy
    >>> ml = flopy.modflow.Modflow()
    >>> lrcec = {0:[2, 3, 4, 10., 100.]}  #this drain will be applied to all
    >>>                                   #stress periods
    >>> drn = flopy.modflow.ModflowDrn(ml, stress_period_data=lrcec)

    """

    def __init__(
        self,
        model,
        ipakcb=None,
        stress_period_data=None,
        dtype=None,
        extension="drn",
        unitnumber=None,
        options=None,
        filenames=None,
        **kwargs
    ):

        # set default unit number of one is not specified
        if unitnumber is None:
            unitnumber = ModflowDrn._defaultunit()

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
                ipakcb, fname=fname, package=ModflowDrn._ftype()
            )
        else:
            ipakcb = 0

        if options is None:
            options = []
        self.is_drt = False
        for opt in options:
            if opt.upper() == "RETURNFLOW":
                self.is_drt = True
                break
        if self.is_drt:
            name = ["DRT"]
        else:
            name = [ModflowDrn._ftype()]
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
        self.url = "drn.htm"

        self.ipakcb = ipakcb

        self.np = 0

        self.options = options
        if dtype is not None:
            self.dtype = dtype
        else:
            self.dtype = self.get_default_dtype(
                structured=self.parent.structured, is_drt=self.is_drt
            )
        self.stress_period_data = MfList(self, stress_period_data)
        self.parent.add_package(self)

    @staticmethod
    def get_default_dtype(structured=True, is_drt=False):
        if structured:
            if not is_drt:
                dtype = np.dtype(
                    [
                        ("k", int),
                        ("i", int),
                        ("j", int),
                        ("elev", np.float32),
                        ("cond", np.float32),
                    ]
                )
            else:
                dtype = np.dtype(
                    [
                        ("k", int),
                        ("i", int),
                        ("j", int),
                        ("elev", np.float32),
                        ("cond", np.float32),
                        ("layr", int),
                        ("rowr", int),
                        ("colr", int),
                        ("rfprop", np.float32),
                    ]
                )
        else:
            dtype = np.dtype(
                [("node", int), ("elev", np.float32), ("cond", np.float32)]
            )
        return dtype

    def _ncells(self):
        """Maximum number of cells that have drains (developed for MT3DMS
        SSM package).

        Returns
        -------
        ncells: int
            maximum number of drain cells

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
        f_drn = open(self.fn_path, "w")
        f_drn.write("{0}\n".format(self.heading))
        # f_drn.write('%10i%10i\n' % (self.mxactd, self.idrncb))
        line = "{0:10d}{1:10d}".format(
            self.stress_period_data.mxact, self.ipakcb
        )

        if self.is_drt:
            line += "{0:10d}{0:10d}".format(0)
        for opt in self.options:
            line += " " + str(opt)
        line += "\n"
        f_drn.write(line)
        self.stress_period_data.write_transient(f_drn)
        f_drn.close()

    def add_record(self, kper, index, values):
        try:
            self.stress_period_data.add_record(kper, index, values)
        except Exception as e:
            raise Exception("mfdrn error adding record to list: " + str(e))

    @staticmethod
    def get_empty(ncells=0, aux_names=None, structured=True, is_drt=False):
        # get an empty recarray that corresponds to dtype
        dtype = ModflowDrn.get_default_dtype(
            structured=structured, is_drt=is_drt
        )
        if aux_names is not None:
            dtype = Package.add_to_dtype(dtype, aux_names, np.float32)
        return create_empty_recarray(ncells, dtype, default_value=-1.0e10)

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
        drn : ModflowDrn object
            ModflowDrn object.

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow()
        >>> drn = flopy.modflow.ModflowDrn.load('test.drn', m)

        """

        if model.verbose:
            sys.stdout.write("loading drn package file...\n")

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
        return "DRN"

    @staticmethod
    def _defaultunit():
        return 21
