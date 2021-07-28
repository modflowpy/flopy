import errno
import sys
import inspect
import collections
import os.path
from ...mbase import run_model
from ..mfbase import (
    PackageContainer,
    MFFileMgmt,
    ExtFileAction,
    PackageContainerType,
    MFDataException,
    FlopyException,
    VerbosityLevel,
)
from ..mfpackage import MFPackage
from ..data.mfstructure import DatumType
from ..data import mfstructure
from ..utils import binaryfile_utils
from ..utils import mfobservation
from ..modflow import mfnam, mfims, mftdis, mfgwfgnc, mfgwfmvr
from ..data.mfdatautil import MFComment


class SimulationDict(collections.OrderedDict):
    """
    Class containing custom dictionary for MODFLOW simulations.  Dictionary
    contains model data.  Dictionary keys are "paths" to the data that include
    the model and package containing the data.

    Behaves as an OrderedDict with some additional features described below.

    Parameters
    ----------
    path : MFFileMgmt
        Object containing path information for the simulation

    """

    def __init__(self, path=None):
        collections.OrderedDict.__init__(self)
        self._path = path

    def __getitem__(self, key):
        """Define the __getitem__ magic method.

        Parameters
        ----------
        key (string): Part or all of a dictionary key

        Returns:
            MFData or numpy.ndarray

        """
        if key == "_path" or not hasattr(self, "_path"):
            raise AttributeError(key)

        # FIX: Transport - Include transport output files
        if key[1] in ("CBC", "HDS", "DDN", "UCN"):
            val = binaryfile_utils.MFOutput(self, self._path, key)
            return val.data

        elif key[-1] == "Observations":
            val = mfobservation.MFObservation(self, self._path, key)
            return val.data

        if key in self:
            val = collections.OrderedDict.__getitem__(self, key)
            return val
        return AttributeError(key)

    def __setitem__(self, key, val):
        """Define the __setitem__ magic method.

        Parameters
        ----------
        key : str
            Dictionary key
        val : MFData
            MFData to store in dictionary

        """
        collections.OrderedDict.__setitem__(self, key, val)

    def find_in_path(self, key_path, key_leaf):
        """Attempt to find key_leaf in a partial key path key_path.

        Parameters
        ----------
        key_path : str
            partial path to the data
        key_leaf : str
            name of the data

        Returns
        -------
            Data: MFData,
            index: int

        """
        key_path_size = len(key_path)
        for key, item in self.items():
            if key[:key_path_size] == key_path:
                if key[-1] == key_leaf:
                    # found key_leaf as a key in the dictionary
                    return item, None
                if not isinstance(item, MFComment):
                    data_item_index = 0
                    data_item_structures = item.structure.data_item_structures
                    for data_item_struct in data_item_structures:
                        if data_item_struct.name == key_leaf:
                            # found key_leaf as a data item name in the data
                            # in the dictionary
                            return item, data_item_index
                        if data_item_struct.type != DatumType.keyword:
                            data_item_index += 1
        return None, None

    def output_keys(self, print_keys=True):
        """Return a list of output data keys supported by the dictionary.

        Parameters
        ----------
        print_keys : bool
            print keys to console

        Returns
        -------
        output keys : list

        """
        # get keys to request binary output
        x = binaryfile_utils.MFOutputRequester.getkeys(
            self, self._path, print_keys=print_keys
        )
        return [key for key in x.dataDict]

    def input_keys(self):
        """Return a list of input data keys.

        Returns
        -------
            input keys : list

        """
        # get keys to request input ie. package data
        for key in self:
            print(key)

    def observation_keys(self):
        """Return a list of observation keys.

        Returns
        -------
            observation keys : list

        """
        # get keys to request observation file output
        mfobservation.MFObservationRequester.getkeys(self, self._path)

    def keys(self):
        """Return a list of all keys.

        Returns
        -------
            all keys : list

        """
        # overrides the built in keys to print all keys, input and output
        self.input_keys()
        try:
            self.output_keys()
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
        try:
            self.observation_keys()
        except KeyError:
            pass


class MFSimulationData:
    """
    Class containing MODFLOW simulation data and file formatting data.  Use
    MFSimulationData to set simulation-wide settings which include data
    formatting and file location settings.

    Parameters
    ----------
    path : str
        path on disk to the simulation

    Attributes
    ----------
    indent_string : str
        String used to define how much indent to use (file formatting)
    internal_formatting : list
        List defining string to use for internal formatting
    external_formatting : list
        List defining string to use for external formatting
    open_close_formatting : list
        List defining string to use for open/close
    max_columns_of_data : int
        Maximum columns of data before line wraps.  For structured grids this
        is set to ncol by default.  For all other grids the default is 20.
    wrap_multidim_arrays : bool
        Whether to wrap line for multi-dimensional arrays at the end of a
        row/column/layer
    float_precision : int
        Number of decimal points to write for a floating point number
    float_characters : int
        Number of characters a floating point number takes up
    write_headers: bool
        When true flopy writes a header to each package file indicating that
        it was created by flopy
    sci_note_upper_thres : float
        Numbers greater than this threshold are written in scientific notation
    sci_note_lower_thres : float
        Numbers less than this threshold are written in scientific notation
    mfpath : MFFileMgmt
        File path location information for the simulation
    model_dimensions : OrderedDict
        Dictionary containing discretization information for each model
    mfdata : SimulationDict
        Custom dictionary containing all model data for the simulation

    """

    def __init__(self, path, mfsim):
        # --- formatting variables ---
        self.indent_string = "  "
        self.constant_formatting = ["constant", ""]
        self._max_columns_of_data = 20
        self.wrap_multidim_arrays = True
        self.float_precision = 8
        self.float_characters = 15
        self.write_headers = True
        self._sci_note_upper_thres = 100000
        self._sci_note_lower_thres = 0.001
        self.fast_write = True
        self.comments_on = False
        self.auto_set_sizes = True
        self.verify_data = True
        self.debug = False
        self.verbose = True
        self.verbosity_level = VerbosityLevel.normal
        self.max_columns_user_set = False
        self.max_columns_auto_set = False

        self._update_str_format()

        # --- file path ---
        self.mfpath = MFFileMgmt(path, mfsim)

        # --- ease of use variables to make working with modflow input and
        # output data easier --- model dimension class for each model
        self.model_dimensions = collections.OrderedDict()

        # --- model data ---
        self.mfdata = SimulationDict(self.mfpath)

        # --- temporary variables ---
        # other external files referenced
        self.referenced_files = collections.OrderedDict()

    @property
    def max_columns_of_data(self):
        return self._max_columns_of_data

    @max_columns_of_data.setter
    def max_columns_of_data(self, val):
        if not self.max_columns_user_set and (
            not self.max_columns_auto_set or val > self._max_columns_of_data
        ):
            self._max_columns_of_data = val
            self.max_columns_user_set = True

    def set_sci_note_upper_thres(self, value):
        """Sets threshold number where any number larger than threshold
        is represented in scientific notation.

        Parameters
        ----------
            value: float
                threshold value

        """
        self._sci_note_upper_thres = value
        self._update_str_format()

    def set_sci_note_lower_thres(self, value):
        """Sets threshold number where any number smaller than threshold
        is represented in scientific notation.

        Parameters
        ----------
            value: float
                threshold value

        """
        self._sci_note_lower_thres = value
        self._update_str_format()

    def _update_str_format(self):
        """Update floating point formatting strings."""
        self.reg_format_str = "{:.%dE}" % self.float_precision
        self.sci_format_str = "{:%d.%df" "}" % (
            self.float_characters,
            self.float_precision,
        )


class MFSimulation(PackageContainer):
    """
    Entry point into any MODFLOW simulation.

    MFSimulation is used to load, build, and/or save a MODFLOW 6 simulation.
    A MFSimulation object must be created before creating any of the MODFLOW 6
    model objects.

    Parameters
    ----------
    sim_name : str
        Name of the simulation.
    version : str
        Version of MODFLOW 6 executable
    exe_name : str
        Relative path to MODFLOW 6 executable from the simulation
        working folder.
    sim_ws : str
        Path to MODFLOW 6 simulation working folder.  This is the folder
        containing the simulation name file.
    verbosity_level : int
        Verbosity level of standard output from 0 to 2. When 0 is specified no
        standard output is written.  When 1 is specified standard
        error/warning messages with some informational messages are written.
        When 2 is specified full error/warning/informational messages are
        written (this is ideal for debugging).
    continue_ : bool
        Sets the continue option in the simulation name file. The continue
        option is a keyword flag to indicate that the simulation should
        continue even if one or more solutions do not converge.
    nocheck : bool
         Sets the nocheck option in the simulation name file. The nocheck
         option is a keyword flag to indicate that the model input check
         routines should not be called prior to each time step. Checks
         are performed by default.
    memory_print_option : str
         Sets memory_print_option in the simulation name file.
         Memory_print_option is a flag that controls printing of detailed
         memory manager usage to the end of the simulation list file.  NONE
         means do not print detailed information. SUMMARY means print only
         the total memory for each simulation component. ALL means print
         information for each variable stored in the memory manager. NONE is
         default if memory_print_option is not specified.
    write_headers: bool
        When true flopy writes a header to each package file indicating that
        it was created by flopy.

    Examples
    --------
    >>> s = MFSimulation.load('my simulation', 'simulation.nam')

    Attributes
    ----------
    sim_name : str
        Name of the simulation
    name_file : MFPackage
        Simulation name file package

    """

    def __init__(
        self,
        sim_name="sim",
        version="mf6",
        exe_name="mf6.exe",
        sim_ws=".",
        verbosity_level=1,
        continue_=None,
        nocheck=None,
        memory_print_option=None,
        write_headers=True,
    ):
        super().__init__(MFSimulationData(sim_ws, self), sim_name)
        self.simulation_data.verbosity_level = self._resolve_verbosity_level(
            verbosity_level
        )
        self.simulation_data.write_headers = write_headers
        # verify metadata
        fpdata = mfstructure.MFStructure()
        if not fpdata.valid:
            excpt_str = (
                "Invalid package metadata.  Unable to load MODFLOW "
                "file structure metadata."
            )
            raise FlopyException(excpt_str)

        # initialize
        self.dimensions = None
        self.type = "Simulation"

        self.version = version
        self.exe_name = exe_name
        self._models = collections.OrderedDict()
        self._tdis_file = None
        self._exchange_files = collections.OrderedDict()
        self._ims_files = collections.OrderedDict()
        self._ghost_node_files = {}
        self._mover_files = {}
        self._other_files = collections.OrderedDict()
        self.structure = fpdata.sim_struct
        self.model_type = None

        self._exg_file_num = {}
        self._gnc_file_num = 0
        self._mvr_file_num = 0

        self.simulation_data.mfpath.set_last_accessed_path()

        # build simulation name file
        self.name_file = mfnam.ModflowNam(
            self,
            filename="mfsim.nam",
            continue_=continue_,
            nocheck=nocheck,
            memory_print_option=memory_print_option,
        )

        # try to build directory structure
        sim_path = self.simulation_data.mfpath.get_sim_path()
        if not os.path.isdir(sim_path):
            try:
                os.makedirs(sim_path)
            except OSError as e:
                if (
                    self.simulation_data.verbosity_level.value
                    >= VerbosityLevel.quiet.value
                ):
                    print(
                        "An error occurred when trying to create the "
                        "directory {}: {}".format(sim_path, e.strerror)
                    )

        # set simulation validity initially to false since the user must first
        # add at least one model to the simulation and fill out the name and
        #  tdis files
        self.valid = False

    def __getattr__(self, item):
        """Override __getattr__ to allow retrieving models.

        __getattr__ is used to allow for getting models and packages as if
        they are attributes

        Parameters
        ----------
        item : str
            model or package name


        Returns
        -------
        md : Model or package object
            Model or package object of type :class:flopy6.mfmodel or
            :class:flopy6.mfpackage

        """
        if item == "valid" or not hasattr(self, "valid"):
            raise AttributeError(item)

        models = []
        if item in self.structure.model_types:
            # get all models of this type
            for model in self._models.values():
                if model.model_type == item:
                    models.append(model)

        if len(models) > 0:
            return models
        elif item in self._models:
            model = self.get_model(item)
            if model is not None:
                return model
            raise AttributeError(item)
        else:
            package = self.get_package(item)
            if package is not None:
                return package
            raise AttributeError(item)

    def __repr__(self):
        """Override __repr__ to print custom string.

        Returns
        --------
            repr string : str
                string describing object

        """
        return self._get_data_str(True)

    def __str__(self):
        """Override __str__ to print custom string.

        Returns
        --------
            str string : str
                string describing object

        """
        return self._get_data_str(False)

    def _get_data_str(self, formal):
        file_mgt = self.simulation_data.mfpath
        data_str = (
            "sim_name = {}\nsim_path = {}\nexe_name = "
            "{}\n"
            "\n".format(self.name, file_mgt.get_sim_path(), self.exe_name)
        )

        for package in self._packagelist:
            pk_str = package._get_data_str(formal, False)
            if formal:
                if len(pk_str.strip()) > 0:
                    data_str = (
                        "{}###################\nPackage {}\n"
                        "###################\n\n"
                        "{}\n".format(data_str, package._get_pname(), pk_str)
                    )
            else:
                if len(pk_str.strip()) > 0:
                    data_str = (
                        "{}###################\nPackage {}\n"
                        "###################\n\n"
                        "{}\n".format(data_str, package._get_pname(), pk_str)
                    )
        for model in self._models.values():
            if formal:
                mod_repr = repr(model)
                if len(mod_repr.strip()) > 0:
                    data_str = (
                        "{}@@@@@@@@@@@@@@@@@@@@\nModel {}\n"
                        "@@@@@@@@@@@@@@@@@@@@\n\n"
                        "{}\n".format(data_str, model.name, mod_repr)
                    )
            else:
                mod_str = str(model)
                if len(mod_str.strip()) > 0:
                    data_str = (
                        "{}@@@@@@@@@@@@@@@@@@@@\nModel {}\n"
                        "@@@@@@@@@@@@@@@@@@@@\n\n"
                        "{}\n".format(data_str, model.name, mod_str)
                    )
        return data_str

    @property
    def model_names(self):
        """Return a list of model names associated with this simulation.

        Returns
        --------
            list: list of model names

        """
        return self._models.keys()

    @classmethod
    def load(
        cls,
        sim_name="modflowsim",
        version="mf6",
        exe_name="mf6.exe",
        sim_ws=".",
        strict=True,
        verbosity_level=1,
        load_only=None,
        verify_data=False,
        write_headers=True,
    ):
        """Load an existing model.

        Parameters
        ----------
        sim_name : str
            Name of the simulation.
        version : str
            MODFLOW version
        exe_name : str
            Relative path to MODFLOW executable from the simulation working
            folder
        sim_ws : str
            Path to simulation working folder
        strict : bool
            Strict enforcement of file formatting
        verbosity_level : int
            Verbosity level of standard output
                0: No standard output
                1: Standard error/warning messages with some informational
                   messages
                2: Verbose mode with full error/warning/informational
                   messages.  This is ideal for debugging.
        load_only : list
            List of package abbreviations or package names corresponding to
            packages that flopy will load. default is None, which loads all
            packages. the discretization packages will load regardless of this
            setting. subpackages, like time series and observations, will also
            load regardless of this setting.
            example list: ['ic', 'maw', 'npf', 'oc', 'ims', 'gwf6-gwf6']
        verify_data : bool
            Verify data when it is loaded. this can slow down loading
        write_headers: bool
            When true flopy writes a header to each package file indicating
            that it was created by flopy

        Returns
        -------
        sim : MFSimulation object

        Examples
        --------
        >>> s = flopy.mf6.mfsimulation.load('my simulation')

        """
        # initialize
        instance = cls(
            sim_name,
            version,
            exe_name,
            sim_ws,
            verbosity_level,
            write_headers=write_headers,
        )
        verbosity_level = instance.simulation_data.verbosity_level
        instance.simulation_data.verify_data = verify_data

        if verbosity_level.value >= VerbosityLevel.normal.value:
            print("loading simulation...")

        # build case consistent load_only dictionary for quick lookups
        load_only = instance._load_only_dict(load_only)

        # load simulation name file
        if verbosity_level.value >= VerbosityLevel.normal.value:
            print("  loading simulation name file...")
        instance.name_file.load(strict)

        # load TDIS file
        tdis_pkg = "tdis{}".format(
            mfstructure.MFStructure().get_version_string()
        )
        tdis_attr = getattr(instance.name_file, tdis_pkg)
        instance._tdis_file = mftdis.ModflowTdis(
            instance, filename=tdis_attr.get_data()
        )

        instance._tdis_file._filename = instance.simulation_data.mfdata[
            ("nam", "timing", tdis_pkg)
        ].get_data()
        if verbosity_level.value >= VerbosityLevel.normal.value:
            print("  loading tdis package...")
        instance._tdis_file.load(strict)

        # load models
        try:
            model_recarray = instance.simulation_data.mfdata[
                ("nam", "models", "models")
            ]
            models = model_recarray.get_data()
        except MFDataException as mfde:
            message = (
                "Error occurred while loading model names from the "
                "simulation name file."
            )
            raise MFDataException(
                mfdata_except=mfde,
                model=instance.name,
                package="nam",
                message=message,
            )
        for item in models:
            # resolve model working folder and name file
            path, name_file = os.path.split(item[1])
            model_obj = PackageContainer.model_factory(item[0][:-1].lower())
            # load model
            if verbosity_level.value >= VerbosityLevel.normal.value:
                print("  loading model {}...".format(item[0].lower()))
            instance._models[item[2]] = model_obj.load(
                instance,
                instance.structure.model_struct_objs[item[0].lower()],
                item[2],
                name_file,
                version,
                exe_name,
                strict,
                path,
                load_only,
            )

        # load exchange packages and dependent packages
        try:
            exchange_recarray = instance.name_file.exchanges
            has_exch_data = exchange_recarray.has_data()
        except MFDataException as mfde:
            message = (
                "Error occurred while loading exchange names from the "
                "simulation name file."
            )
            raise MFDataException(
                mfdata_except=mfde,
                model=instance.name,
                package="nam",
                message=message,
            )
        if has_exch_data:
            try:
                exch_data = exchange_recarray.get_data()
            except MFDataException as mfde:
                message = (
                    "Error occurred while loading exchange names from "
                    "the simulation name file."
                )
                raise MFDataException(
                    mfdata_except=mfde,
                    model=instance.name,
                    package="nam",
                    message=message,
                )
            for exgfile in exch_data:
                if load_only is not None and not instance._in_pkg_list(
                    load_only, exgfile[0], exgfile[2]
                ):
                    if (
                        instance.simulation_data.verbosity_level.value
                        >= VerbosityLevel.normal.value
                    ):
                        print(
                            "    skipping package {}.."
                            ".".format(exgfile[0].lower())
                        )
                    continue
                # get exchange type by removing numbers from exgtype
                exchange_type = "".join(
                    [char for char in exgfile[0] if not char.isdigit()]
                ).upper()
                # get exchange number for this type
                if exchange_type not in instance._exg_file_num:
                    exchange_file_num = 0
                    instance._exg_file_num[exchange_type] = 1
                else:
                    exchange_file_num = instance._exg_file_num[exchange_type]
                    instance._exg_file_num[exchange_type] += 1

                exchange_name = "{}_EXG_{}".format(
                    exchange_type, exchange_file_num
                )
                # find package class the corresponds to this exchange type
                package_obj = instance.package_factory(
                    exchange_type.replace("-", "").lower(), ""
                )
                if not package_obj:
                    message = (
                        "An error occurred while loading the "
                        "simulation name file.  Invalid exchange type "
                        '"{}" specified.'.format(exchange_type)
                    )
                    type_, value_, traceback_ = sys.exc_info()
                    raise MFDataException(
                        instance.name,
                        "nam",
                        "nam",
                        "loading simulation name file",
                        exchange_recarray.structure.name,
                        inspect.stack()[0][3],
                        type_,
                        value_,
                        traceback_,
                        message,
                        instance._simulation_data.debug,
                    )

                # build and load exchange package object
                exchange_file = package_obj(
                    instance,
                    exgtype=exgfile[0],
                    exgmnamea=exgfile[2],
                    exgmnameb=exgfile[3],
                    filename=exgfile[1],
                    pname=exchange_name,
                    loading_package=True,
                )
                if verbosity_level.value >= VerbosityLevel.normal.value:
                    print(
                        "  loading exchange package {}.."
                        ".".format(exchange_file._get_pname())
                    )
                exchange_file.load(strict)
                instance._exchange_files[exgfile[1]] = exchange_file

        # load simulation packages
        solution_recarray = instance.simulation_data.mfdata[
            ("nam", "solutiongroup", "solutiongroup")
        ]

        try:
            solution_group_dict = solution_recarray.get_data()
        except MFDataException as mfde:
            message = (
                "Error occurred while loading solution groups from "
                "the simulation name file."
            )
            raise MFDataException(
                mfdata_except=mfde,
                model=instance.name,
                package="nam",
                message=message,
            )
        for solution_group in solution_group_dict.values():
            for solution_info in solution_group:
                if load_only is not None and not instance._in_pkg_list(
                    load_only, solution_info[0], solution_info[2]
                ):
                    if (
                        instance.simulation_data.verbosity_level.value
                        >= VerbosityLevel.normal.value
                    ):
                        print(
                            "    skipping package {}.."
                            ".".format(solution_info[0].lower())
                        )
                    continue
                ims_file = mfims.ModflowIms(
                    instance, filename=solution_info[1], pname=solution_info[2]
                )
                if verbosity_level.value >= VerbosityLevel.normal.value:
                    print(
                        "  loading ims package {}.."
                        ".".format(ims_file._get_pname())
                    )
                ims_file.load(strict)

        instance.simulation_data.mfpath.set_last_accessed_path()
        if verify_data:
            instance.check()
        return instance

    def check(self, f=None, verbose=True, level=1):
        """
        Check model data for common errors.

        Parameters
        ----------
        f : str or file handle
            String defining file name or file handle for summary file
            of check method output. If a string is passed a file handle
            is created. If f is None, check method does not write
            results to a summary file. (default is None)
        verbose : bool
            Boolean flag used to determine if check method results are
            written to the screen
        level : int
            Check method analysis level. If level=0, summary checks are
            performed. If level=1, full checks are performed.

        Returns
        -------
            check list: list
                Python list containing simulation check results

        Examples
        --------

        >>> import flopy
        >>> m = flopy.modflow.Modflow.load('model.nam')
        >>> m.check()
        """
        # check instance for simulation-level check
        chk_list = []

        # check models
        for model in self._models.values():
            print('Checking model "{}"...'.format(model.name))
            chk_list.append(model.check(f, verbose, level))

        print("Checking for missing simulation packages...")
        if self._tdis_file is None:
            if chk_list:
                chk_list[0]._add_to_summary(
                    "Error", desc="\r    No tdis package", package="model"
                )
            print("Error: no tdis package")
        if len(self._ims_files) == 0:
            if chk_list:
                chk_list[0]._add_to_summary(
                    "Error", desc="\r    No solver package", package="model"
                )
            print("Error: no ims package")
        return chk_list

    @property
    def sim_package_list(self):
        """List of all "simulation level" packages"""
        package_list = []
        if self._tdis_file is not None:
            package_list.append(self._tdis_file)
        for sim_package in self._ims_files.values():
            package_list.append(sim_package)
        for sim_package in self._exchange_files.values():
            package_list.append(sim_package)
        for sim_package in self._mover_files.values():
            package_list.append(sim_package)
        for sim_package in self._other_files.values():
            package_list.append(sim_package)
        return package_list

    def load_package(
        self,
        ftype,
        fname,
        pname,
        strict,
        ref_path,
        dict_package_name=None,
        parent_package=None,
    ):
        """Load a package from a file.

        Parameters
        ----------
        ftype : str
            the file type
        fname : str
            the name of the file containing the package input
        pname : str
            the user-defined name for the package
        strict : bool
            strict mode when loading the file
        ref_path : str
            path to the file. uses local path if set to None
        dict_package_name : str
            package name for dictionary lookup
        parent_package : MFPackage
            parent package

        """
        if ftype == "gnc":
            if fname not in self._ghost_node_files:
                # get package type from parent package
                if parent_package:
                    package_abbr = parent_package.package_abbr[0:3]
                else:
                    package_abbr = "GWF"
                # build package name and package
                gnc_name = "{}-GNC_{}".format(package_abbr, self._gnc_file_num)
                ghost_node_file = mfgwfgnc.ModflowGwfgnc(
                    self,
                    filename=fname,
                    pname=gnc_name,
                    parent_file=parent_package,
                    loading_package=True,
                )
                ghost_node_file.load(strict)
                self._ghost_node_files[fname] = ghost_node_file
                self._gnc_file_num += 1
                return ghost_node_file
        elif ftype == "mvr":
            if fname not in self._mover_files:
                # Get package type from parent package
                if parent_package:
                    package_abbr = parent_package.package_abbr[0:3]
                else:
                    package_abbr = "GWF"
                # build package name and package
                mvr_name = "{}-MVR_{}".format(package_abbr, self._mvr_file_num)
                mover_file = mfgwfmvr.ModflowGwfmvr(
                    self,
                    filename=fname,
                    pname=mvr_name,
                    parent_file=parent_package,
                    loading_package=True,
                )
                mover_file.load(strict)
                self._mover_files[fname] = mover_file
                self._mvr_file_num += 1
                return mover_file
        else:
            # create package
            package_obj = self.package_factory(ftype, "")
            package = package_obj(
                self,
                filename=fname,
                pname=dict_package_name,
                add_to_package_list=False,
                parent_file=parent_package,
                loading_package=True,
            )
            # verify that this is a utility package
            utl_struct = mfstructure.MFStructure().sim_struct.utl_struct_objs
            if package.package_type in utl_struct:
                package.load(strict)
                self._other_files[package.filename] = package
                # register child package with the simulation
                self._add_package(package, package.path)
                if parent_package is not None:
                    # register child package with the parent package
                    parent_package._add_package(package, package.path)
            else:
                if (
                    self.simulation_data.verbosity_level.value
                    >= VerbosityLevel.normal.value
                ):
                    print(
                        "WARNING: Unsupported file type {} for "
                        "simulation.".format(package.package_type)
                    )
            return package

    def register_ims_package(self, ims_file, model_list):
        """Register an ims package with the simulation.

        Parameters
            ims_file : MFPackage
                ims package to register
            model_list : list of strings
                list of models using the ims package to be registered

        """
        if isinstance(model_list, str):
            model_list = [model_list]

        if not isinstance(ims_file, mfims.ModflowIms):
            comment = (
                'Parameter "ims_file" is not a valid ims file.  '
                'Expected type ModflowIms, but type "{}" was given'
                ".".format(type(ims_file))
            )
            type_, value_, traceback_ = sys.exc_info()
            raise MFDataException(
                None,
                "ims",
                "",
                "registering ims package",
                "",
                inspect.stack()[0][3],
                type_,
                value_,
                traceback_,
                comment,
                self.simulation_data.debug,
            )

        in_simulation = False
        pkg_with_same_name = None
        for file in self._ims_files.values():
            if file is ims_file:
                in_simulation = True
            if file.package_name == ims_file.package_name and file != ims_file:
                pkg_with_same_name = file
                if (
                    self.simulation_data.verbosity_level.value
                    >= VerbosityLevel.normal.value
                ):
                    print(
                        "WARNING: ims package with name {} already exists. "
                        "New ims package will replace old package"
                        ".".format(file.package_name)
                    )
                self._remove_package(self._ims_files[file.filename])
                del self._ims_files[file.filename]
        # register ims package
        if not in_simulation:
            self._add_package(ims_file, self._get_package_path(ims_file))
        # do not allow an ims package to be registered twice with the
        # same simulation
        if not in_simulation:
            # create unique file/package name
            if ims_file.package_name is None:
                file_num = len(self._ims_files) - 1
                ims_file.package_name = "ims_{}".format(file_num)
            if ims_file.filename in self._ims_files:
                ims_file.filename = MFFileMgmt.unique_file_name(
                    ims_file.filename, self._ims_files
                )
            # add ims package to simulation
            self._ims_files[ims_file.filename] = ims_file

        # If ims file is being replaced, replace ims filename in
        # solution group
        if pkg_with_same_name is not None and self._is_in_solution_group(
            pkg_with_same_name.filename, 1
        ):
            # change existing solution group to reflect new ims file
            self._replace_ims_in_solution_group(
                pkg_with_same_name.filename, 1, ims_file.filename
            )
        # only allow an ims package to be registered to one solution group
        elif model_list is not None:
            ims_in_group = self._is_in_solution_group(ims_file.filename, 1)
            # add solution group to the simulation name file
            solution_recarray = self.name_file.solutiongroup
            solution_group_list = solution_recarray.get_active_key_list()
            if len(solution_group_list) == 0:
                solution_group_num = 0
            else:
                solution_group_num = solution_group_list[-1][0]

            if ims_in_group:
                self._append_to_ims_solution_group(
                    ims_file.filename, model_list
                )
            else:
                if self.name_file.mxiter.get_data(solution_group_num) is None:
                    self.name_file.mxiter.add_transient_key(solution_group_num)

                # associate any models in the model list to this
                # simulation file
                version_string = mfstructure.MFStructure().get_version_string()
                ims_pkg = "ims{}".format(version_string)
                new_record = [ims_pkg, ims_file.filename]
                for model in model_list:
                    new_record.append(model)
                try:
                    solution_recarray.append_list_as_record(
                        new_record, solution_group_num
                    )
                except MFDataException as mfde:
                    message = (
                        "Error occurred while updating the "
                        "simulation name file with the ims package "
                        'file "{}".'.format(ims_file.filename)
                    )
                    raise MFDataException(
                        mfdata_except=mfde, package="nam", message=message
                    )

    @staticmethod
    def _rename_package_group(group_dict, name):
        package_type_count = {}
        for package in group_dict.values():
            if package.package_type not in package_type_count:
                package.filename = "{}.{}".format(name, package.package_type)
                package_type_count[package.package_type] = 1
            else:
                package_type_count[package.package_type] += 1
                package.filename = "{}_{}.{}".format(
                    name,
                    package_type_count[package.package.package_type],
                    package.package_type,
                )

    def rename_all_packages(self, name):
        """Rename all packages with name as prefix.

        Parameters
        ----------
            name: str
                Prefix of package names

        """
        if self._tdis_file is not None:
            self._tdis_file.filename = "{}.{}".format(
                name, self._tdis_file.package_type
            )

        self._rename_package_group(self._exchange_files, name)
        self._rename_package_group(self._ims_files, name)
        self._rename_package_group(self._ghost_node_files, name)
        self._rename_package_group(self._mover_files, name)
        self._rename_package_group(self._other_files, name)
        for model in self._models.values():
            model.rename_all_packages(name)

    def set_all_data_external(
        self, check_data=True, external_data_folder=None
    ):
        """Sets the simulation's list and array data to be stored externally.

        Parameters
        ----------
            check_data: bool
                Determines if data error checking is enabled during this
                process.  Data error checking can be slow on large datasets.
            external_data_folder
                Folder, relative to the simulation path or model relative path
                (see use_model_relative_path parameter), where external data
                will be stored
        """
        # copy any files whose paths have changed
        self.simulation_data.mfpath.copy_files()
        # set data external for all packages in all models
        for model in self._models.values():
            model.set_all_data_external(check_data, external_data_folder)
        # set data external for ims packages
        for package in self._ims_files.values():
            package.set_all_data_external(check_data, external_data_folder)
        # set data external for ghost node packages
        for package in self._ghost_node_files.values():
            package.set_all_data_external(check_data, external_data_folder)
        # set data external for mover packages
        for package in self._mover_files.values():
            package.set_all_data_external(check_data, external_data_folder)
        for package in self._exchange_files.values():
            package.set_all_data_external(check_data, external_data_folder)

    def set_all_data_internal(self, check_data=True):
        # set data external for all packages in all models
        for model in self._models.values():
            model.set_all_data_internal(check_data)
        # set data external for ims packages
        for package in self._ims_files.values():
            package.set_all_data_internal(check_data)
        # set data external for ghost node packages
        for package in self._ghost_node_files.values():
            package.set_all_data_internal(check_data)
        # set data external for mover packages
        for package in self._mover_files.values():
            package.set_all_data_internal(check_data)
        for package in self._exchange_files.values():
            package.set_all_data_internal(check_data)

    def write_simulation(
        self, ext_file_action=ExtFileAction.copy_relative_paths, silent=False
    ):
        """Write the simulation to files.

        Parameters
            ext_file_action : ExtFileAction
                Defines what to do with external files when the simulation
                path has changed.  Defaults to copy_relative_paths which
                copies only files with relative paths, leaving files defined
                by absolute paths fixed.
            silent : bool
                Writes out the simulation in silent mode (verbosity_level = 0)

        """
        sim_data = self.simulation_data
        if not sim_data.max_columns_user_set:
            # search for dis packages
            for model in self._models.values():
                dis = model.get_package("dis")
                if dis is not None and hasattr(dis, "ncol"):
                    sim_data.max_columns_of_data = dis.ncol.get_data()
                    sim_data.max_columns_user_set = False
                    sim_data.max_columns_auto_set = True

        saved_verb_lvl = self.simulation_data.verbosity_level
        if silent:
            self.simulation_data.verbosity_level = VerbosityLevel.quiet

        # write simulation name file
        if (
            self.simulation_data.verbosity_level.value
            >= VerbosityLevel.normal.value
        ):
            print("writing simulation...")
            print("  writing simulation name file...")
        self.name_file.write(ext_file_action=ext_file_action)

        # write TDIS file
        if (
            self.simulation_data.verbosity_level.value
            >= VerbosityLevel.normal.value
        ):
            print("  writing simulation tdis package...")
        self._tdis_file.write(ext_file_action=ext_file_action)

        # write ims files
        for ims_file in self._ims_files.values():
            if (
                self.simulation_data.verbosity_level.value
                >= VerbosityLevel.normal.value
            ):
                print(
                    "  writing ims package {}.."
                    ".".format(ims_file._get_pname())
                )
            ims_file.write(ext_file_action=ext_file_action)

        # write exchange files
        for exchange_file in self._exchange_files.values():
            exchange_file.write()
            if (
                hasattr(exchange_file, "gnc_filerecord")
                and exchange_file.gnc_filerecord.has_data()
            ):
                try:
                    gnc_file = exchange_file.gnc_filerecord.get_data()[0][0]
                except MFDataException as mfde:
                    message = (
                        "An error occurred while retrieving the ghost "
                        "node file record from exchange file "
                        '"{}".'.format(exchange_file.filename)
                    )
                    raise MFDataException(
                        mfdata_except=mfde,
                        package=exchange_file._get_pname(),
                        message=message,
                    )
                if gnc_file in self._ghost_node_files:
                    if (
                        self.simulation_data.verbosity_level.value
                        >= VerbosityLevel.normal.value
                    ):
                        print(
                            "  writing gnc package {}...".format(
                                self._ghost_node_files[gnc_file]._get_pname()
                            )
                        )
                    self._ghost_node_files[gnc_file].write(
                        ext_file_action=ext_file_action
                    )
                else:
                    if (
                        self.simulation_data.verbosity_level.value
                        >= VerbosityLevel.normal.value
                    ):
                        print(
                            "WARNING: Ghost node file {} not loaded prior to"
                            " writing. File will not be written"
                            ".".format(gnc_file)
                        )
            if (
                hasattr(exchange_file, "mvr_filerecord")
                and exchange_file.mvr_filerecord.has_data()
            ):
                try:
                    mvr_file = exchange_file.mvr_filerecord.get_data()[0][0]
                except MFDataException as mfde:
                    message = (
                        "An error occurred while retrieving the mover "
                        "file record from exchange file "
                        '"{}".'.format(exchange_file.filename)
                    )
                    raise MFDataException(
                        mfdata_except=mfde,
                        package=exchange_file._get_pname(),
                        message=message,
                    )

                if mvr_file in self._mover_files:
                    if (
                        self.simulation_data.verbosity_level.value
                        >= VerbosityLevel.normal.value
                    ):
                        print(
                            "  writing mvr package {}...".format(
                                self._mover_files[mvr_file]._get_pname()
                            )
                        )
                    self._mover_files[mvr_file].write(
                        ext_file_action=ext_file_action
                    )
                else:
                    if (
                        self.simulation_data.verbosity_level.value
                        >= VerbosityLevel.normal.value
                    ):
                        print(
                            "WARNING: Mover file {} not loaded prior to "
                            "writing. File will not be "
                            "written.".format(mvr_file)
                        )

        # write other packages
        for pp in self._other_files.values():
            if (
                self.simulation_data.verbosity_level.value
                >= VerbosityLevel.normal.value
            ):
                print("  writing package {}...".format(pp._get_pname()))
            pp.write(ext_file_action=ext_file_action)

        # FIX: model working folder should be model name file folder

        # write models
        for model in self._models.values():
            if (
                self.simulation_data.verbosity_level.value
                >= VerbosityLevel.normal.value
            ):
                print("  writing model {}...".format(model.name))
            model.write(ext_file_action=ext_file_action)

        self.simulation_data.mfpath.set_last_accessed_path()

        if silent:
            self.simulation_data.verbosity_level = saved_verb_lvl

    def set_sim_path(self, path):
        """Return a list of output data keys.

        Parameters
        ----------
        path : str
            Relative or absolute path to simulation root folder.

        """
        # set all data internal
        self.set_all_data_internal()

        # set simulation path
        self.simulation_data.mfpath.set_sim_path(path, True)

        if not os.path.exists(path):
            # create new simulation folder
            os.makedirs(path)

    def run_simulation(
        self,
        silent=None,
        pause=False,
        report=False,
        normal_msg="normal termination",
        use_async=False,
        cargs=None,
    ):
        """Run the simulation.

        Parameters
        ----------
            silent: bool
                Run in silent mode
            pause: bool
                Pause at end of run
            report: bool
                Save stdout lines to a list (buff)
            normal_msg: str or list
                Normal termination message used to determine if the run
                terminated normally. More than one message can be provided
                using a list. (default is 'normal termination')
            use_async : bool
                Asynchronously read model stdout and report with timestamps.
                good for models that take long time to run.  not good for
                models that run really fast
            cargs : str or list of strings
                Additional command line arguments to pass to the executable.
                default is None

        Returns
        --------
            success : bool
            buff : list of lines of stdout

        """
        if silent is None:
            if (
                self.simulation_data.verbosity_level.value
                >= VerbosityLevel.normal.value
            ):
                silent = False
            else:
                silent = True
        return run_model(
            self.exe_name,
            None,
            self.simulation_data.mfpath.get_sim_path(),
            silent=silent,
            pause=pause,
            report=report,
            normal_msg=normal_msg,
            use_async=use_async,
            cargs=cargs,
        )

    def delete_output_files(self):
        """Deletes simulation output files."""
        output_req = binaryfile_utils.MFOutputRequester
        output_file_keys = output_req.getkeys(
            self.simulation_data.mfdata, self.simulation_data.mfpath, False
        )
        for path in output_file_keys.binarypathdict.values():
            if os.path.isfile(path):
                os.remove(path)

    def remove_package(self, package_name):
        """Removes package from the simulation.  `package_name` can be the
        package's name, type, or package object to be removed from the model.

        Parameters
        ----------
        package_name : str
            Name of package to be removed

        """
        if isinstance(package_name, MFPackage):
            packages = [package_name]
        else:
            packages = self.get_package(package_name)
            if not isinstance(packages, list):
                packages = [packages]
        for package in packages:
            if (
                self._tdis_file is not None
                and package.path == self._tdis_file.path
            ):
                self._tdis_file = None
            if package.filename in self._exchange_files:
                del self._exchange_files[package.filename]
            if package.filename in self._ims_files:
                del self._ims_files[package.filename]
                self._remove_ims_soultion_group(package.filename)
            if package.filename in self._ghost_node_files:
                del self._ghost_node_files[package.filename]
            if package.filename in self._mover_files:
                del self._mover_files[package.filename]
            if package.filename in self._other_files:
                del self._other_files[package.filename]

            self._remove_package(package)

    @property
    def model_dict(self):
        """Return a dictionary of models associated with this simulation.

        Returns
        --------
            model dict : dict
                dictionary of models

        """
        return self._models.copy()

    def get_model(self, model_name=None):
        """Returns the models in the simulation with a given model name, name
        file name, or model type.

        Parameters
        ----------
            model_name : str
                Name of the model to get.  Passing in None or an empty list
                will get the first model.

        Returns
        --------
            model : MFModel

        """
        if len(self._models) == 0:
            return None

        if model_name is None:
            for model in self._models.values():
                return model
        if model_name in self._models:
            return self._models[model_name]
        # do case-insensitive lookup
        for name, model in self._models.items():
            if model_name.lower() == name.lower():
                return model
        return None

    def get_exchange_file(self, filename):
        """Get a specified exchange file.

        Parameters
        ----------
            filename : str
                Name of exchange file to get

        Returns
        --------
            exchange package : MFPackage

        """
        if filename in self._exchange_files:
            return self._exchange_files[filename]
        else:
            excpt_str = 'Exchange file "{}" can not be found' ".".format(
                filename
            )
            raise FlopyException(excpt_str)

    def get_mvr_file(self, filename):
        """Get a specified mover file.

        Parameters
        ----------
            filename : str
                Name of mover file to get

        Returns
        --------
            mover package : MFPackage

        """
        if filename in self._mover_files:
            return self._mover_files[filename]
        else:
            excpt_str = 'MVR file "{}" can not be ' "found.".format(filename)
            raise FlopyException(excpt_str)

    def get_gnc_file(self, filename):
        """Get a specified gnc file.

        Parameters
        ----------
            filename : str
                Name of gnc file to get

        Returns
        --------
            gnc package : MFPackage

        """
        if filename in self._ghost_node_files:
            return self._ghost_node_files[filename]
        else:
            excpt_str = 'GNC file "{}" can not be ' "found.".format(filename)
            raise FlopyException(excpt_str)

    def register_exchange_file(self, package):
        """Register an exchange package file with the simulation.  This is a
        call-back method made from the package and should not be called
        directly.

        Parameters
        ----------
            package : MFPackage
                Exchange package object to register

        """
        if package.filename not in self._exchange_files:
            exgtype = package.exgtype
            exgmnamea = package.exgmnamea
            exgmnameb = package.exgmnameb

            if exgtype is None or exgmnamea is None or exgmnameb is None:
                excpt_str = (
                    "Exchange packages require that exgtype, "
                    "exgmnamea, and exgmnameb are specified."
                )
                raise FlopyException(excpt_str)

            self._exchange_files[package.filename] = package
            try:
                exchange_recarray_data = self.name_file.exchanges.get_data()
            except MFDataException as mfde:
                message = (
                    "An error occurred while retrieving exchange "
                    "data from the simulation name file.  The error "
                    "occurred while registering exchange file "
                    '"{}".'.format(package.filename)
                )
                raise MFDataException(
                    mfdata_except=mfde,
                    package=package._get_pname(),
                    message=message,
                )
            if exchange_recarray_data is not None:
                for index, exchange in zip(
                    range(0, len(exchange_recarray_data)),
                    exchange_recarray_data,
                ):
                    if exchange[1] == package.filename:
                        # update existing exchange
                        exchange_recarray_data[index][0] = exgtype
                        exchange_recarray_data[index][2] = exgmnamea
                        exchange_recarray_data[index][3] = exgmnameb
                        ex_recarray = self.name_file.exchanges
                        try:
                            ex_recarray.set_data(exchange_recarray_data)
                        except MFDataException as mfde:
                            message = (
                                "An error occurred while setting "
                                "exchange data in the simulation name "
                                "file.  The error occurred while "
                                "registering the following "
                                "values (exgtype, filename, "
                                'exgmnamea, exgmnameb): "{} {} {}'
                                '{}".'.format(
                                    exgtype,
                                    package.filename,
                                    exgmnamea,
                                    exgmnameb,
                                )
                            )
                            raise MFDataException(
                                mfdata_except=mfde,
                                package=package._get_pname(),
                                message=message,
                            )
                        return
            try:
                # add new exchange
                self.name_file.exchanges.append_data(
                    [(exgtype, package.filename, exgmnamea, exgmnameb)]
                )
            except MFDataException as mfde:
                message = (
                    "An error occurred while setting exchange data "
                    "in the simulation name file.  The error occurred "
                    "while registering the following values (exgtype, "
                    'filename, exgmnamea, exgmnameb): "{} {} {}'
                    '{}".'.format(
                        exgtype, package.filename, exgmnamea, exgmnameb
                    )
                )
                raise MFDataException(
                    mfdata_except=mfde,
                    package=package._get_pname(),
                    message=message,
                )
            if (
                package.dimensions is None
                or package.dimensions.model_dim is None
            ):
                # resolve exchange package dimensions object
                package.dimensions = package.create_package_dimensions()

    def register_package(
        self,
        package,
        add_to_package_list=True,
        set_package_name=True,
        set_package_filename=True,
    ):
        """Register a package file with the simulation.  This is a
        call-back method made from the package and should not be called
        directly.

        Parameters
        ----------
            package : MFPackage
                Package to register
            add_to_package_list : bool
                Add package to lookup list
            set_package_name : bool
                Produce a package name for this package
            set_package_filename : bool
                Produce a filename for this package

        Returns
        --------
            (path : tuple, package structure : MFPackageStructure)

        """
        package.container_type = [PackageContainerType.simulation]
        path = self._get_package_path(package)
        if add_to_package_list and package.package_type.lower != "nam":
            pname = None
            if package.package_name is not None:
                pname = package.package_name.lower()
            if (
                package.package_type.lower() == "tdis"
                and self._tdis_file is not None
                and self._tdis_file in self._packagelist
            ):
                # tdis package already exists. there can be only one tdis
                # package.  remove existing tdis package
                if (
                    self.simulation_data.verbosity_level.value
                    >= VerbosityLevel.normal.value
                ):
                    print(
                        "WARNING: tdis package already exists. Replacing "
                        "existing tdis package."
                    )
                self._remove_package(self._tdis_file)
            elif (
                package.package_type.lower() == "gnc"
                and package.filename in self._ghost_node_files
                and self._ghost_node_files[package.filename]
                in self._packagelist
            ):
                # gnc package with same file name already exists.  remove old
                # gnc package
                if (
                    self.simulation_data.verbosity_level.value
                    >= VerbosityLevel.normal.value
                ):
                    print(
                        "WARNING: gnc package with name {} already exists. "
                        "Replacing existing gnc package"
                        ".".format(pname)
                    )
                self._remove_package(self._ghost_node_files[package.filename])
                del self._ghost_node_files[package.filename]
            elif (
                package.package_type.lower() == "mvr"
                and package.filename in self._mover_files
                and self._mover_files[package.filename] in self._packagelist
            ):
                # mvr package with same file name already exists.  remove old
                # mvr package
                if (
                    self.simulation_data.verbosity_level.value
                    >= VerbosityLevel.normal.value
                ):
                    print(
                        "WARNING: mvr package with name {} already exists. "
                        "Replacing existing mvr package"
                        ".".format(pname)
                    )
                self._remove_package(self._mover_files[package.filename])
                del self._mover_files[package.filename]
            elif (
                package.package_type.lower() != "ims"
                and pname in self.package_name_dict
            ):
                if (
                    self.simulation_data.verbosity_level.value
                    >= VerbosityLevel.normal.value
                ):
                    print(
                        "WARNING: Package with name {} already exists.  "
                        "Replacing existing package"
                        ".".format(package.package_name.lower())
                    )
                self._remove_package(self.package_name_dict[pname])
            if package.package_type.lower() != "ims":
                # all but ims packages get added here.  ims packages are
                # added during ims package registration
                self._add_package(package, path)
        if package.package_type.lower() == "nam":
            return path, self.structure.name_file_struct_obj
        elif package.package_type.lower() == "tdis":
            self._tdis_file = package
            struct_root = mfstructure.MFStructure()
            tdis_pkg = "tdis{}".format(struct_root.get_version_string())
            tdis_attr = getattr(self.name_file, tdis_pkg)
            try:
                tdis_attr.set_data(package.filename)
            except MFDataException as mfde:
                message = (
                    "An error occurred while setting the tdis package "
                    'file name "{}".  The error occurred while '
                    "registering the tdis package with the "
                    "simulation".format(package.filename)
                )
                raise MFDataException(
                    mfdata_except=mfde,
                    package=package._get_pname(),
                    message=message,
                )
            return (
                path,
                self.structure.package_struct_objs[
                    package.package_type.lower()
                ],
            )
        elif package.package_type.lower() == "gnc":
            if package.filename not in self._ghost_node_files:
                self._ghost_node_files[package.filename] = package
                self._gnc_file_num += 1
            elif self._ghost_node_files[package.filename] != package:
                # auto generate a unique file name and register it
                file_name = MFFileMgmt.unique_file_name(
                    package.filename, self._ghost_node_files
                )
                package.filename = file_name
                self._ghost_node_files[file_name] = package
        elif package.package_type.lower() == "mvr":
            if package.filename not in self._mover_files:
                self._mover_files[package.filename] = package
            else:
                # auto generate a unique file name and register it
                file_name = MFFileMgmt.unique_file_name(
                    package.filename, self._mover_files
                )
                package.filename = file_name
                self._mover_files[file_name] = package
        elif package.package_type.lower() == "ims":
            # default behavior is to register the ims package with the first
            # unregistered model
            unregistered_models = []
            for model in self._models:
                model_registered = self._is_in_solution_group(model, 2)
                if not model_registered:
                    unregistered_models.append(model)
            if unregistered_models:
                self.register_ims_package(package, unregistered_models)
            else:
                self.register_ims_package(package, None)
            return (
                path,
                self.structure.package_struct_objs[
                    package.package_type.lower()
                ],
            )
        else:
            self._other_files[package.filename] = package

        if package.package_type.lower() in self.structure.package_struct_objs:
            return (
                path,
                self.structure.package_struct_objs[
                    package.package_type.lower()
                ],
            )
        elif package.package_type.lower() in self.structure.utl_struct_objs:
            return (
                path,
                self.structure.utl_struct_objs[package.package_type.lower()],
            )
        else:
            excpt_str = (
                'Invalid package type "{}".  Unable to register '
                "package.".format(package.package_type)
            )
            print(excpt_str)
            raise FlopyException(excpt_str)

    def register_model(self, model, model_type, model_name, model_namefile):
        """Add a model to the simulation. This is a call-back method made
        from the package and should not be called directly.

        Parameters
        ----------
           model : MFModel
               Model object to add to simulation
           sln_group : str
               Solution group of model

        Returns
        --------
           model_structure_object : MFModelStructure
        """

        # get model structure from model type
        if model_type not in self.structure.model_struct_objs:
            message = 'Invalid model type: "{}".'.format(model_type)
            type_, value_, traceback_ = sys.exc_info()
            raise MFDataException(
                model.name,
                "",
                model.name,
                "registering model",
                "sim",
                inspect.stack()[0][3],
                type_,
                value_,
                traceback_,
                message,
                self.simulation_data.debug,
            )

        # add model
        self._models[model_name] = model

        # update simulation name file
        self.name_file.models.append_list_as_record(
            [model_type, model_namefile, model_name]
        )

        if len(self._ims_files) > 0:
            # register model with first ims file found
            first_ims_key = next(iter(self._ims_files))
            self.register_ims_package(
                self._ims_files[first_ims_key], model_name
            )

        return self.structure.model_struct_objs[model_type]

    def get_ims_package(self, key):
        """Get the ims package with the specified `key`.

        Parameters
        ----------
            key : str
                ims package key

        Returns
        --------
            ims_package : ModflowIms

        """
        if key in self._ims_files:
            return self._ims_files[key]
        return None

    def remove_model(self, model_name):
        """Remove model with name `model_name` from the simulation

        Parameters
        ----------
            model_name : str
                Model name to remove from simulation

        """
        # Remove model
        del self._models[model_name]

        # TODO: Fully implement this
        # Update simulation name file

    def is_valid(self):
        """Checks the validity of the solution and all of its models and
        packages.  Returns true if the solution is valid, false if it is not.


        Returns
        --------
            valid : bool
                Whether this is a valid simulation

        """
        # name file valid
        if not self.name_file.is_valid():
            return False

        # tdis file valid
        if not self._tdis_file.is_valid():
            return False

        # exchanges valid
        for exchange in self._exchange_files:
            if not exchange.is_valid():
                return False

        # ims files valid
        for imsfile in self._ims_files.values():
            if not imsfile.is_valid():
                return False

        # a model exists
        if not self._models:
            return False

        # models valid
        for key in self._models:
            if not self._models[key].is_valid():
                return False

        # each model has an imsfile

        return True

    @staticmethod
    def _resolve_verbosity_level(verbosity_level):
        if verbosity_level == 0:
            return VerbosityLevel.quiet
        elif verbosity_level == 1:
            return VerbosityLevel.normal
        elif verbosity_level == 2:
            return VerbosityLevel.verbose
        else:
            return verbosity_level

    @staticmethod
    def _get_package_path(package):
        if package.parent_file is not None:
            return (package.parent_file.path) + (package.package_type,)
        else:
            return (package.package_type,)

    def _remove_ims_soultion_group(self, ims_file):
        solution_recarray = self.name_file.solutiongroup
        for solution_group_num in solution_recarray.get_active_key_list():
            try:
                rec_array = solution_recarray.get_data(solution_group_num[0])
            except MFDataException as mfde:
                message = (
                    "An error occurred while getting solution group"
                    '"{}" from the simulation name file'
                    ".".format(solution_group_num[0])
                )
                raise MFDataException(
                    mfdata_except=mfde, package="nam", message=message
                )

            new_array = []
            for record in rec_array:
                if record.slnfname == ims_file:
                    continue
                else:
                    new_array.append(record)

            if not new_array:
                new_array = None

            solution_recarray.set_data(new_array, solution_group_num[0])

    def _append_to_ims_solution_group(self, ims_file, new_models):
        solution_recarray = self.name_file.solutiongroup
        for solution_group_num in solution_recarray.get_active_key_list():
            try:
                rec_array = solution_recarray.get_data(solution_group_num[0])
            except MFDataException as mfde:
                message = (
                    "An error occurred while getting solution group"
                    '"{}" from the simulation name file'
                    ".".format(solution_group_num[0])
                )
                raise MFDataException(
                    mfdata_except=mfde, package="nam", message=message
                )
            new_array = []
            for index, record in enumerate(rec_array):
                new_record = []
                rec_model_dict = {}
                for index, item in enumerate(record):
                    if record[1] == ims_file or item not in new_models:
                        new_record.append(item)
                        if index > 1 and item is not None:
                            rec_model_dict[item.lower()] = 1

                if record[1] == ims_file:
                    for model in new_models:
                        if model.lower() not in rec_model_dict:
                            new_record.append(model)

                new_array.append(tuple(new_record))
            solution_recarray.set_data(new_array, solution_group_num[0])

    def _replace_ims_in_solution_group(self, item, index, new_item):
        solution_recarray = self.name_file.solutiongroup
        for solution_group_num in solution_recarray.get_active_key_list():
            try:
                rec_array = solution_recarray.get_data(solution_group_num[0])
            except MFDataException as mfde:
                message = (
                    "An error occurred while getting solution group"
                    '"{}" from the simulation name file.  The error '
                    'occurred while replacing IMS file "{}" with "{}"'
                    'at index "{}"'.format(
                        solution_group_num[0], item, new_item, index
                    )
                )
                raise MFDataException(
                    mfdata_except=mfde, package="nam", message=message
                )
            if rec_array is not None:
                for rec_item in rec_array:
                    if rec_item[index] == item:
                        rec_item[index] = new_item

    def _is_in_solution_group(self, item, index):
        solution_recarray = self.name_file.solutiongroup
        for solution_group_num in solution_recarray.get_active_key_list():
            try:
                rec_array = solution_recarray.get_data(solution_group_num[0])
            except MFDataException as mfde:
                message = (
                    "An error occurred while getting solution group"
                    '"{}" from the simulation name file.  The error '
                    'occurred while verifying file "{}" at index "{}" '
                    "is in the simulation name file"
                    ".".format(solution_group_num[0], item, index)
                )
                raise MFDataException(
                    mfdata_except=mfde, package="nam", message=message
                )

            if rec_array is not None:
                for rec_item in rec_array:
                    if rec_item[index] == item:
                        return True
        return False

    def plot(self, model_list=None, SelPackList=None, **kwargs):
        """Plot simulation or models.

        Method to plot a whole simulation or a series of models
        that are part of a simulation.

        Parameters
        ----------
            model_list: (list)
                List of model names to plot, if none all models will be plotted
            SelPackList: (list)
                List of package names to plot, if none all packages will be
                plotted
            kwargs:
                filename_base : str
                    Base file name that will be used to automatically
                    generate file names for output image files. Plots will be
                    exported as image files if file_name_base is not None.
                    (default is None)
                file_extension : str
                    Valid matplotlib.pyplot file extension for savefig().
                    Only used if filename_base is not None. (default is 'png')
                mflay : int
                    MODFLOW zero-based layer number to return.  If None, then
                    all layers will be included. (default is None)
                kper : int
                    MODFLOW zero-based stress period number to return.
                    (default is zero)
                key : str
                    MFList dictionary key. (default is None)

        Returns
        --------
             axes: (list)
                matplotlib.pyplot.axes objects

        """
        from flopy.plot.plotutil import PlotUtilities

        axes = PlotUtilities._plot_simulation_helper(
            self, model_list=model_list, SelPackList=SelPackList, **kwargs
        )
        return axes
