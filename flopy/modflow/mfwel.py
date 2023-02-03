"""
mfwel module.  Contains the ModflowWel class. Note that the user can access
the ModflowWel class as `flopy.modflow.ModflowWel`.

Additional information for this MODFLOW package can be found at the `Online
MODFLOW Guide
<https://water.usgs.gov/ogw/modflow/MODFLOW-2005-Guide/wel.html>`_.

"""
import numpy as np

from ..pakbase import Package
from ..utils import MfList
from ..utils.optionblock import OptionBlock
from ..utils.recarray_utils import create_empty_recarray


class ModflowWel(Package):
    """
    MODFLOW Well Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.modflow.mf.Modflow`) to which
        this package will be added.
    ipakcb : int
        A flag that is used to determine if cell-by-cell budget data should be
        saved. If ipakcb is non-zero cell-by-cell budget data will be saved.
        (default is 0).
    stress_period_data : list of boundaries, or recarray of boundaries, or
        dictionary of boundaries
        Each well is defined through definition of
        layer (int), row (int), column (int), flux (float).
        The simplest form is a dictionary with a lists of boundaries for each
        stress period, where each list of boundaries itself is a list of
        boundaries. Indices of the dictionary are the numbers of the stress
        period. This gives the form of:

            stress_period_data =
            {0: [
                [lay, row, col, flux],
                [lay, row, col, flux],
                [lay, row, col, flux]
                ],
            1:  [
                [lay, row, col, flux],
                [lay, row, col, flux],
                [lay, row, col, flux]
                ], ...
            kper:
                [
                [lay, row, col, flux],
                [lay, row, col, flux],
                [lay, row, col, flux]
                ]
            }

        Note that if the number of lists is smaller than the number of stress
        periods, then the last list of wells will apply until the end of the
        simulation. Full details of all options to specify stress_period_data
        can be found in the flopy3 boundaries Notebook in the basic
        subdirectory of the examples directory
    dtype : custom datatype of stress_period_data.
        If None the default well datatype will be applied (default is None).
    extension : string
        Filename extension (default is 'wel')
    options : list of strings
        Package options (default is None).
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
    add_package : bool
        Flag to add the initialised package object to the parent model object.
        Default is True.

    Attributes
    ----------
    mxactw : int
        Maximum number of wells for a stress period.  This is calculated
        automatically by FloPy based on the information in
        stress_period_data.

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
    >>> m = flopy.modflow.Modflow()
    >>> lrcq = {0:[[2, 3, 4, -100.]], 1:[[2, 3, 4, -100.]]}
    >>> wel = flopy.modflow.ModflowWel(m, stress_period_data=lrcq)

    """

    _options = dict(
        [
            (
                "specify",
                {
                    OptionBlock.dtype: np.bool_,
                    OptionBlock.nested: True,
                    OptionBlock.n_nested: 2,
                    OptionBlock.vars: dict(
                        [
                            ("phiramp", OptionBlock.simple_float),
                            (
                                "iunitramp",
                                dict(
                                    [
                                        (OptionBlock.dtype, int),
                                        (OptionBlock.nested, False),
                                        (OptionBlock.optional, True),
                                    ]
                                ),
                            ),
                        ]
                    ),
                },
            ),
            ("tabfiles", OptionBlock.simple_tabfile),
        ]
    )

    def __init__(
        self,
        model,
        ipakcb=None,
        stress_period_data=None,
        dtype=None,
        extension="wel",
        options=None,
        binary=False,
        unitnumber=None,
        filenames=None,
        add_package=True,
    ):
        # set default unit number of one is not specified
        if unitnumber is None:
            unitnumber = ModflowWel._defaultunit()

        # set filenames
        filenames = self._prepare_filenames(filenames, 2)

        # update external file information with cbc output, if necessary
        if ipakcb is not None:
            model.add_output_file(
                ipakcb, fname=filenames[1], package=self._ftype()
            )
        else:
            ipakcb = 0

        # call base package constructor
        super().__init__(
            model,
            extension=extension,
            name=self._ftype(),
            unit_number=unitnumber,
            filenames=filenames[0],
        )

        self._generate_heading()
        self.url = "wel.html"

        self.ipakcb = ipakcb
        self.np = 0

        if options is None:
            options = []
        self.specify = False
        self.phiramp = None
        self.iunitramp = None
        self.options = options
        if isinstance(options, OptionBlock):
            if not self.options.specify:
                self.specify = self.options.specify
            else:
                self.specify = True

            self.phiramp = self.options.phiramp
            self.iunitramp = self.options.iunitramp
            # this is to grab the aux variables...
            options = []

        else:
            for idx, opt in enumerate(options):
                if "specify" in opt:
                    t = opt.strip().split()
                    self.specify = True
                    self.phiramp = float(t[1])
                    self.iunitramp = int(t[2])
                    self.options.pop(idx)
                    break

        if dtype is not None:
            self.dtype = dtype
        else:
            self.dtype = self.get_default_dtype(
                structured=self.parent.structured
            )

        # determine if any aux variables in dtype
        dt = self.get_default_dtype(structured=self.parent.structured)
        if len(self.dtype.names) > len(dt.names):
            for name in self.dtype.names[len(dt.names) :]:
                ladd = True
                for option in options:
                    if name.lower() in option.lower():
                        ladd = False
                        break
                if ladd:
                    options.append(f"aux {name} ")

        if isinstance(self.options, OptionBlock):
            if not self.options.auxillary:
                self.options.auxillary = options
        else:
            self.options = options

        # initialize MfList
        self.stress_period_data = MfList(
            self, stress_period_data, binary=binary
        )

        if add_package:
            self.parent.add_package(self)

    def _ncells(self):
        """Maximum number of cells that have wells (developed for
        MT3DMS SSM package).

        Returns
        -------
        ncells: int
            maximum number of wel cells

        """
        return self.stress_period_data.mxact

    def write_file(self, f=None):
        """
        Write the package file.

        Parameters:
            f: (str) optional file name

        Returns
        -------
        None

        """
        if f is not None:
            if isinstance(f, str):
                f_wel = open(f, "w")
            else:
                f_wel = f
        else:
            f_wel = open(self.fn_path, "w")

        f_wel.write(f"{self.heading}\n")

        if (
            isinstance(self.options, OptionBlock)
            and self.parent.version == "mfnwt"
        ):
            self.options.update_from_package(self)
            if self.options.block:
                self.options.write_options(f_wel)

        line = f" {self.stress_period_data.mxact:9d} {self.ipakcb:9d} "

        if isinstance(self.options, OptionBlock):
            if self.options.noprint:
                line += "NOPRINT "
            if self.options.auxillary:
                line += " ".join(
                    [str(aux).upper() for aux in self.options.auxillary]
                )

        else:
            for opt in self.options:
                line += " " + str(opt)

        line += "\n"
        f_wel.write(line)

        if (
            isinstance(self.options, OptionBlock)
            and self.parent.version == "mfnwt"
        ):
            if not self.options.block:
                if isinstance(self.options.specify, np.ndarray):
                    self.options.tabfiles = False
                    self.options.write_options(f_wel)

        else:
            if self.specify and self.parent.version == "mfnwt":
                f_wel.write(
                    f"SPECIFY {self.phiramp:10.5g} {self.iunitramp:10d}\n"
                )

        self.stress_period_data.write_transient(f_wel)
        f_wel.close()

    def add_record(self, kper, index, values):
        try:
            self.stress_period_data.add_record(kper, index, values)
        except Exception as e:
            raise Exception(f"mfwel error adding record to list: {e!s}")

    @staticmethod
    def get_default_dtype(structured=True):
        if structured:
            dtype = np.dtype(
                [
                    ("k", int),
                    ("i", int),
                    ("j", int),
                    ("flux", np.float32),
                ]
            )
        else:
            dtype = np.dtype([("node", int), ("flux", np.float32)])
        return dtype

    @staticmethod
    def get_empty(ncells=0, aux_names=None, structured=True):
        # get an empty recarray that corresponds to dtype
        dtype = ModflowWel.get_default_dtype(structured=structured)
        if aux_names is not None:
            dtype = Package.add_to_dtype(dtype, aux_names, np.float32)
        return create_empty_recarray(ncells, dtype, default_value=-1.0e10)

    @staticmethod
    def _get_sfac_columns():
        return ["flux"]

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

        Returns
        -------
        wel : ModflowWel object
            ModflowWel object.

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow()
        >>> wel = flopy.modflow.ModflowWel.load('test.wel', m)

        """

        if model.verbose:
            print("loading wel package file...")

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
        return "WEL"

    @staticmethod
    def _defaultunit():
        return 20
