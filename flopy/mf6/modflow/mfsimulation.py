import errno
import inspect
import os.path
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from ...mbase import run_model
from ..data import mfstructure
from ..data.mfdatautil import MFComment
from ..data.mfstructure import DatumType
from ..mfbase import (
    ExtFileAction,
    FlopyException,
    MFDataException,
    MFFileMgmt,
    PackageContainer,
    PackageContainerType,
    VerbosityLevel,
)
from ..mfpackage import MFPackage
from ..modflow import mfnam, mftdis
from ..utils import binaryfile_utils, mfobservation


class SimulationDict(dict):
    """
    Class containing custom dictionary for MODFLOW simulations.  Dictionary
    contains model data.  Dictionary keys are "paths" to the data that include
    the model and package containing the data.

    Behaves as an dict with some additional features described below.

    Parameters
    ----------
    path : MFFileMgmt
        Object containing path information for the simulation

    """

    def __init__(self, path=None):
        dict.__init__(self)
        self._path = path

    def __getitem__(self, key):
        """
        Define the __getitem__ magic method.

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
            val = dict.__getitem__(self, key)
            return val
        return AttributeError(key)

    def __setitem__(self, key, val):
        """
        Define the __setitem__ magic method.

        Parameters
        ----------
        key : str
            Dictionary key
        val : MFData
            MFData to store in dictionary

        """
        dict.__setitem__(self, key, val)

    def find_in_path(self, key_path, key_leaf):
        """
        Attempt to find key_leaf in a partial key path key_path.

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
        """
        Return a list of output data keys supported by the dictionary.

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
        """
        Return a list of input data keys.

        Returns
        -------
            input keys : list

        """
        # get keys to request input ie. package data
        for key in self:
            print(key)

    def observation_keys(self):
        """
        Return a list of observation keys.

        Returns
        -------
            observation keys : list

        """
        # get keys to request observation file output
        mfobservation.MFObservationRequester.getkeys(self, self._path)

    def keys(self):
        """
        Return a list of all keys.

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
    _float_precision : int
        Number of decimal points to write for a floating point number
    _float_characters : int
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
    model_dimensions : dict
        Dictionary containing discretization information for each model
    mfdata : SimulationDict
        Custom dictionary containing all model data for the simulation

    """

    def __init__(self, path: Union[str, os.PathLike], mfsim):
        # --- formatting variables ---
        self.indent_string = "  "
        self.constant_formatting = ["constant", ""]
        self._max_columns_of_data = 20
        self.wrap_multidim_arrays = True
        self._float_precision = 8
        self._float_characters = 15
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
        self.model_dimensions = {}

        # --- model data ---
        self.mfdata = SimulationDict(self.mfpath)

        # --- temporary variables ---
        # other external files referenced
        self.referenced_files = {}

    @property
    def lazy_io(self):
        if not self.auto_set_sizes and not self.verify_data:
            return True
        return False

    @lazy_io.setter
    def lazy_io(self, val):
        if val:
            self.auto_set_sizes = False
            self.verify_data = False
        else:
            self.auto_set_sizes = True
            self.verify_data = True

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

    @property
    def float_precision(self):
        """
        Gets precision of floating point numbers.
        """
        return self._float_precision

    @float_precision.setter
    def float_precision(self, value):
        """
        Sets precision of floating point numbers.

        Parameters
        ----------
            value: float
                floating point precision

        """
        self._float_precision = value
        self._update_str_format()

    @property
    def float_characters(self):
        """
        Gets max characters used in floating point numbers.
        """
        return self._float_characters

    @float_characters.setter
    def float_characters(self, value):
        """
        Sets max characters used in floating point numbers.

        Parameters
        ----------
            value: float
                floating point max characters

        """
        self._float_characters = value
        self._update_str_format()

    def set_sci_note_upper_thres(self, value):
        """
        Sets threshold number where any number larger than threshold
        is represented in scientific notation.

        Parameters
        ----------
            value: float
                threshold value

        """
        self._sci_note_upper_thres = value
        self._update_str_format()

    def set_sci_note_lower_thres(self, value):
        """
        Sets threshold number where any number smaller than threshold
        is represented in scientific notation.

        Parameters
        ----------
            value: float
                threshold value

        """
        self._sci_note_lower_thres = value
        self._update_str_format()

    def _update_str_format(self):
        """
        Update floating point formatting strings."""
        self.reg_format_str = f"{{:.{self._float_precision}E}}"
        self.sci_format_str = (
            f"{{:{self._float_characters}.{self._float_precision}f}}"
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
        Path to MODFLOW 6 executable
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
    lazy_io: bool
        When true flopy only reads external data when the data is requested
        and only writes external data if the data has changed.  This option
        automatically overrides the verify_data and auto_set_sizes, turning
        both off.
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
        exe_name: Union[str, os.PathLike] = "mf6",
        sim_ws: Union[str, os.PathLike] = os.curdir,
        verbosity_level=1,
        continue_=None,
        nocheck=None,
        memory_print_option=None,
        write_headers=True,
        lazy_io=False,
    ):
        super().__init__(MFSimulationData(sim_ws, self), sim_name)
        self.simulation_data.verbosity_level = self._resolve_verbosity_level(
            verbosity_level
        )
        self.simulation_data.write_headers = write_headers
        if lazy_io:
            self.simulation_data.lazy_io = True

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
        self._models = {}
        self._tdis_file = None
        self._exchange_files = {}
        self._solution_files = {}
        self._other_files = {}
        self.structure = fpdata.sim_struct
        self.model_type = None

        self._exg_file_num = {}

        self.simulation_data.mfpath.set_last_accessed_path()

        # build simulation name file
        self.name_file = mfnam.ModflowNam(
            self,
            filename="mfsim.nam",
            continue_=continue_,
            nocheck=nocheck,
            memory_print_option=memory_print_option,
            _internal_package=True,
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
        """
        Override __getattr__ to allow retrieving models.

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
                if model.model_type == item or model.model_type[:-1] == item:
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
        """
        Override __repr__ to print custom string.

        Returns
        --------
            repr string : str
                string describing object

        """
        return self._get_data_str(True)

    def __str__(self):
        """
        Override __str__ to print custom string.

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
        """
        Return a list of model names associated with this simulation.

        Returns
        --------
            list: list of model names

        """
        return list(self._models.keys())

    @property
    def exchange_files(self):
        """
        Return list of exchange files associated with this simulation.

        Returns
        --------
            list: list of exchange names

        """
        return self._exchange_files.values()

    @classmethod
    def load(
        cls,
        sim_name="modflowsim",
        version="mf6",
        exe_name: Union[str, os.PathLike] = "mf6",
        sim_ws: Union[str, os.PathLike] = os.curdir,
        strict=True,
        verbosity_level=1,
        load_only=None,
        verify_data=False,
        write_headers=True,
        lazy_io=False,
    ):
        """
        Load an existing model.

        Parameters
        ----------
        sim_name : str
            Name of the simulation.
        version : str
            MODFLOW version
        exe_name : str or PathLike
            Path to MODFLOW executable (relative to the simulation workspace or absolute)
        sim_ws : str or PathLike
            Path to simulation workspace
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
        lazy_io: bool
            When true flopy only reads external data when the data is requested
            and only writes external data if the data has changed.  This option
            automatically overrides the verify_data and auto_set_sizes, turning
            both off.
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
        if lazy_io:
            instance.simulation_data.lazy_io = True

        if verbosity_level.value >= VerbosityLevel.normal.value:
            print("loading simulation...")

        # build case consistent load_only dictionary for quick lookups
        load_only = instance._load_only_dict(load_only)

        # load simulation name file
        if verbosity_level.value >= VerbosityLevel.normal.value:
            print("  loading simulation name file...")
        instance.name_file.load(strict)

        # load TDIS file
        tdis_pkg = f"tdis{mfstructure.MFStructure().get_version_string()}"
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
                print(f"  loading model {item[0].lower()}...")
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
                        print(f"    skipping package {exgfile[0].lower()}...")
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

                exchange_name = f"{exchange_type}_EXG_{exchange_file_num}"
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
                        f"  loading exchange package {exchange_file._get_pname()}..."
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
                            f"    skipping package {solution_info[0].lower()}..."
                        )
                    continue
                # create solution package
                sln_package_obj = instance.package_factory(
                    solution_info[0][:-1].lower(), ""
                )
                sln_package = sln_package_obj(
                    instance,
                    filename=solution_info[1],
                    pname=solution_info[2],
                )

                if verbosity_level.value >= VerbosityLevel.normal.value:
                    print(
                        f"  loading solution package "
                        f"{sln_package._get_pname()}..."
                    )
                sln_package.load(strict)

        instance.simulation_data.mfpath.set_last_accessed_path()
        if verify_data:
            instance.check()
        return instance

    def check(
        self,
        f: Optional[Union[str, os.PathLike]] = None,
        verbose=True,
        level=1,
    ):
        """
        Check model data for common errors.

        Parameters
        ----------
        f : str or PathLike, optional
            String defining file name or file handle for summary file
            of check method output. If str or pathlike, a file handle
            is created. If None, the method does not write results to
            a summary file. (default is None)
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
            print(f'Checking model "{model.name}"...')
            chk_list.append(model.check(f, verbose, level))

        print("Checking for missing simulation packages...")
        if self._tdis_file is None:
            if chk_list:
                chk_list[0]._add_to_summary(
                    "Error", desc="\r    No tdis package", package="model"
                )
            print("Error: no tdis package")
        if len(self._solution_files) == 0:
            if chk_list:
                chk_list[0]._add_to_summary(
                    "Error", desc="\r    No solver package", package="model"
                )
            print("Error: no solution package")
        return chk_list

    @property
    def sim_path(self) -> Path:
        return Path(self.simulation_data.mfpath.get_sim_path())

    @property
    def sim_package_list(self):
        """List of all "simulation level" packages"""
        package_list = []
        if self._tdis_file is not None:
            package_list.append(self._tdis_file)
        for sim_package in self._solution_files.values():
            package_list.append(sim_package)
        for sim_package in self._exchange_files.values():
            package_list.append(sim_package)
        for sim_package in self._other_files.values():
            package_list.append(sim_package)
        return package_list

    def load_package(
        self,
        ftype,
        fname: Union[str, os.PathLike],
        pname,
        strict,
        ref_path: Union[str, os.PathLike],
        dict_package_name=None,
        parent_package=None,
    ):
        """
        Load a package from a file.

        Parameters
        ----------
        ftype : str
            the file type
        fname : str or PathLike
            the path of the file containing the package input
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
        if (
            ftype in self.structure.package_struct_objs
            and self.structure.package_struct_objs[ftype].multi_package_support
        ) or (
            ftype in self.structure.utl_struct_objs
            and self.structure.utl_struct_objs[ftype].multi_package_support
        ):
            # resolve dictionary name for package
            if dict_package_name is not None:
                if parent_package is not None:
                    dict_package_name = f"{parent_package.path[-1]}_{ftype}"
                else:
                    # use dict_package_name as the base name
                    if ftype in self._ftype_num_dict:
                        self._ftype_num_dict[dict_package_name] += 1
                    else:
                        self._ftype_num_dict[dict_package_name] = 0
                    dict_package_name = "{}_{}".format(
                        dict_package_name,
                        self._ftype_num_dict[dict_package_name],
                    )
            else:
                # use ftype as the base name
                if ftype in self._ftype_num_dict:
                    self._ftype_num_dict[ftype] += 1
                else:
                    self._ftype_num_dict[ftype] = 0
                if pname is not None:
                    dict_package_name = pname
                else:
                    dict_package_name = (
                        f"{ftype}_{self._ftype_num_dict[ftype]}"
                    )
        else:
            dict_package_name = ftype

        # get package object from file type
        package_obj = self.package_factory(ftype, "")
        # create package
        package = package_obj(
            self,
            filename=fname,
            pname=dict_package_name,
            parent_file=parent_package,
            loading_package=True,
        )
        package.load(strict)
        self._other_files[package.filename] = package
        # register child package with the simulation
        self._add_package(package, package.path)
        if parent_package is not None:
            # register child package with the parent package
            parent_package._add_package(package, package.path)
        return package

    def register_ims_package(
        self, solution_file: MFPackage, model_list: Union[str, List[str]]
    ):
        self.register_solution_package(solution_file, model_list)

    def register_solution_package(
        self, solution_file: MFPackage, model_list: Union[str, List[str]]
    ):
        """
        Register a solution package with the simulation.

        Parameters
            solution_file : MFPackage
                solution package to register
            model_list : list of strings
                list of models using the solution package to be registered

        """
        if isinstance(model_list, str):
            model_list = [model_list]

        if (
            solution_file.package_type
            not in mfstructure.MFStructure().flopy_dict["solution_packages"]
        ):
            comment = (
                'Parameter "solution_file" is not a valid solution file.  '
                'Expected solution file, but type "{}" was given'
                ".".format(type(solution_file))
            )
            type_, value_, traceback_ = sys.exc_info()
            raise MFDataException(
                None,
                "solution",
                "",
                "registering solution package",
                "",
                inspect.stack()[0][3],
                type_,
                value_,
                traceback_,
                comment,
                self.simulation_data.debug,
            )
        valid_model_types = mfstructure.MFStructure().flopy_dict[
            "solution_packages"
        ][solution_file.package_type]
        # remove models from existing solution groups
        if model_list is not None:
            for model in model_list:
                md = self.get_model(model)
                if md is not None and (
                    md.model_type not in valid_model_types
                    and "*" not in valid_model_types
                ):
                    comment = (
                        f"Model type {md.model_type} is not a valid type "
                        f"for solution file {solution_file.filename} solution "
                        f"file type {solution_file.package_type}. Valid model "
                        f"types are {valid_model_types}"
                    )
                    type_, value_, traceback_ = sys.exc_info()
                    raise MFDataException(
                        None,
                        "solution",
                        "",
                        "registering solution package",
                        "",
                        inspect.stack()[0][3],
                        type_,
                        value_,
                        traceback_,
                        comment,
                        self.simulation_data.debug,
                    )
                self._remove_from_all_solution_groups(model)

        # register solution package with model list
        in_simulation = False
        pkg_with_same_name = None
        for file in self._solution_files.values():
            if file is solution_file:
                in_simulation = True
            if (
                file.package_name == solution_file.package_name
                and file != solution_file
            ):
                pkg_with_same_name = file
                if (
                    self.simulation_data.verbosity_level.value
                    >= VerbosityLevel.normal.value
                ):
                    print(
                        "WARNING: solution package with name {} already exists. "
                        "New solution package will replace old package"
                        ".".format(file.package_name)
                    )
                self._remove_package(self._solution_files[file.filename])
                del self._solution_files[file.filename]
                break
        # register solution package
        if not in_simulation:
            self._add_package(
                solution_file, self._get_package_path(solution_file)
            )
        # do not allow a solution package to be registered twice with the
        # same simulation
        if not in_simulation:
            # create unique file/package name
            if solution_file.package_name is None:
                file_num = len(self._solution_files) - 1
                solution_file.package_name = (
                    f"{solution_file.package_type}_{file_num}"
                )
            if solution_file.filename in self._solution_files:
                solution_file.filename = MFFileMgmt.unique_file_name(
                    solution_file.filename, self._solution_files
                )
            # add solution package to simulation
            self._solution_files[solution_file.filename] = solution_file

        # If solution file is being replaced, replace solution filename in
        # solution group
        if pkg_with_same_name is not None and self._is_in_solution_group(
            pkg_with_same_name.filename, 1
        ):
            # change existing solution group to reflect new solution file
            self._replace_solution_in_solution_group(
                pkg_with_same_name.filename, 1, solution_file.filename
            )
        # only allow solution package to be registered to one solution group
        elif model_list is not None:
            sln_file_in_group = self._is_in_solution_group(
                solution_file.filename, 1
            )
            # add solution group to the simulation name file
            solution_recarray = self.name_file.solutiongroup
            solution_group_list = solution_recarray.get_active_key_list()
            if len(solution_group_list) == 0:
                solution_group_num = 0
            else:
                solution_group_num = solution_group_list[-1][0]

            if sln_file_in_group:
                self._append_to_solution_group(
                    solution_file.filename, model_list
                )
            else:
                if self.name_file.mxiter.get_data(solution_group_num) is None:
                    self.name_file.mxiter.add_transient_key(solution_group_num)

                # associate any models in the model list to this
                # simulation file
                version_string = mfstructure.MFStructure().get_version_string()
                solution_pkg = f"{solution_file.package_abbr}{version_string}"
                new_record = [solution_pkg, solution_file.filename]
                for model in model_list:
                    new_record.append(model)
                try:
                    solution_recarray.append_list_as_record(
                        new_record, solution_group_num
                    )
                except MFDataException as mfde:
                    message = (
                        "Error occurred while updating the "
                        "simulation name file with the solution package "
                        'file "{}".'.format(solution_file.filename)
                    )
                    raise MFDataException(
                        mfdata_except=mfde, package="nam", message=message
                    )

    @staticmethod
    def _rename_package_group(group_dict, name):
        package_type_count = {}
        # first build an array to avoid key modification errors
        package_array = []
        for package in group_dict.values():
            package_array.append(package)
        # update package file names and count
        for package in package_array:
            if package.package_type not in package_type_count:
                file_name = f"{name}.{package.package_type}"
                package_type_count[package.package_type] = 1
            else:
                package_type_count[package.package_type] += 1
                ptc = package_type_count[package.package_type]
                file_name = f"{name}_{ptc}.{package.package_type}"
            base_filepath = os.path.split(package.filename)[0]
            if base_filepath != "":
                # include existing relative path in new file name
                file_name = os.path.join(base_filepath, file_name)
            package.filename = file_name

    def _rename_exchange_file(self, package, new_filename):
        self._exchange_files[package.filename] = package
        try:
            exchange_recarray_data = self.name_file.exchanges.get_data()
        except MFDataException as mfde:
            message = (
                "An error occurred while retrieving exchange "
                "data from the simulation name file.  The error "
                "occurred while registering exchange file "
                f'"{package.filename}".'
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
                    exchange_recarray_data[index][1] = new_filename
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
                            f'exgmnamea, exgmnameb): "{package.exgtype} '
                            f"{package.filename} {package.exgmnamea}"
                            f'{package.exgmnameb}".'
                        )
                        raise MFDataException(
                            mfdata_except=mfde,
                            package=package._get_pname(),
                            message=message,
                        )
                    return

    def _set_timing_block(self, file_name):
        struct_root = mfstructure.MFStructure()
        tdis_pkg = f"tdis{struct_root.get_version_string()}"
        tdis_attr = getattr(self.name_file, tdis_pkg)
        try:
            tdis_attr.set_data(file_name)
        except MFDataException as mfde:
            message = (
                "An error occurred while setting the tdis package "
                f'file name "{file_name}".  The error occurred while '
                "registering the tdis package with the "
                "simulation"
            )
            raise MFDataException(
                mfdata_except=mfde,
                package=file_name,
                message=message,
            )

    def update_package_filename(self, package, new_name):
        """
        Updates internal arrays to be consistent with a new file name.
        This is for internal flopy library use only.

        Parameters
        ----------
            package: MFPackage
                Package with new name
            new_name: str
                Package's new name

        """
        if (
            self._tdis_file is not None
            and package.filename == self._tdis_file.filename
        ):
            self._set_timing_block(new_name)
        elif package.filename in self._exchange_files:
            self._exchange_files[new_name] = self._exchange_files.pop(
                package.filename
            )
            self._rename_exchange_file(package, new_name)
        elif package.filename in self._solution_files:
            self._solution_files[new_name] = self._solution_files.pop(
                package.filename
            )
            self._update_solution_group(package.filename, new_name)
        else:
            self._other_files[new_name] = self._other_files.pop(
                package.filename
            )

    def rename_all_packages(self, name):
        """
        Rename all packages with name as prefix.

        Parameters
        ----------
            name: str
                Prefix of package names

        """
        if self._tdis_file is not None:
            self._tdis_file.filename = f"{name}.{self._tdis_file.package_type}"

        self._rename_package_group(self._exchange_files, name)
        self._rename_package_group(self._solution_files, name)
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
            external_data_folder: str or PathLike
                Path relative to the simulation path or model relative path
                (see use_model_relative_path parameter), where external data
                will be stored
        """
        # copy any files whose paths have changed
        self.simulation_data.mfpath.copy_files()
        # set data external for all packages in all models
        for model in self._models.values():
            model.set_all_data_external(check_data, external_data_folder)
        # set data external for solution packages
        for package in self._solution_files.values():
            package.set_all_data_external(check_data, external_data_folder)
        # set data external for other packages
        for package in self._other_files.values():
            package.set_all_data_external(check_data, external_data_folder)
        for package in self._exchange_files.values():
            package.set_all_data_external(check_data, external_data_folder)

    def set_all_data_internal(self, check_data=True):
        # set data external for all packages in all models
        for model in self._models.values():
            model.set_all_data_internal(check_data)
        # set data external for solution packages
        for package in self._solution_files.values():
            package.set_all_data_internal(check_data)
        # set data external for other packages
        for package in self._other_files.values():
            package.set_all_data_internal(check_data)
        # set data external for exchange packages
        for package in self._exchange_files.values():
            package.set_all_data_internal(check_data)

    def write_simulation(
        self, ext_file_action=ExtFileAction.copy_relative_paths, silent=False
    ):
        """
        Write the simulation to files.

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

        # write solution files
        for solution_file in self._solution_files.values():
            if (
                self.simulation_data.verbosity_level.value
                >= VerbosityLevel.normal.value
            ):
                print(
                    f"  writing solution package "
                    f"{solution_file._get_pname()}..."
                )
            solution_file.write(ext_file_action=ext_file_action)

        # write exchange files
        for exchange_file in self._exchange_files.values():
            exchange_file.write()

        # write other packages
        for pp in self._other_files.values():
            if (
                self.simulation_data.verbosity_level.value
                >= VerbosityLevel.normal.value
            ):
                print(f"  writing package {pp._get_pname()}...")
            pp.write(ext_file_action=ext_file_action)

        # FIX: model working folder should be model name file folder

        # write models
        for model in self._models.values():
            if (
                self.simulation_data.verbosity_level.value
                >= VerbosityLevel.normal.value
            ):
                print(f"  writing model {model.name}...")
            model.write(ext_file_action=ext_file_action)

        self.simulation_data.mfpath.set_last_accessed_path()

        if silent:
            self.simulation_data.verbosity_level = saved_verb_lvl

    def set_sim_path(self, path: Union[str, os.PathLike]):
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
        processors=None,
        normal_msg="normal termination",
        use_async=False,
        cargs=None,
    ):
        """
        Run the simulation.

        Parameters
        ----------
            silent: bool
                Run in silent mode
            pause: bool
                Pause at end of run
            report: bool
                Save stdout lines to a list (buff)
            processors: int
                Number of processors. Parallel simulations are only supported
                for MODFLOW 6 simulations. (default is None)
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
            processors=processors,
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
        """
        Removes package from the simulation.  `package_name` can be the
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
            if package.filename in self._solution_files:
                del self._solution_files[package.filename]
                self._update_solution_group(package.filename)
            if package.filename in self._other_files:
                del self._other_files[package.filename]

            self._remove_package(package)

    @property
    def model_dict(self):
        """
        Return a dictionary of models associated with this simulation.

        Returns
        --------
            model dict : dict
                dictionary of models

        """
        return self._models.copy()

    def get_model(self, model_name=None):
        """
        Returns the models in the simulation with a given model name, name
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
        """
        Get a specified exchange file.

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
            excpt_str = f'Exchange file "{filename}" can not be found.'
            raise FlopyException(excpt_str)

    def get_file(self, filename):
        """
        Get a specified file.

        Parameters
        ----------
            filename : str
                Name of mover file to get

        Returns
        --------
            mover package : MFPackage

        """
        if filename in self._other_files:
            return self._other_files[filename]
        else:
            excpt_str = f'file "{filename}" can not be found.'
            raise FlopyException(excpt_str)

    def get_mvr_file(self, filename):
        """
        Get a specified mover file.

        Parameters
        ----------
            filename : str
                Name of mover file to get

        Returns
        --------
            mover package : MFPackage

        """
        warnings.warn(
            "get_mvr_file will be deprecated and will be removed in version "
            "3.3.6. Use get_file",
            PendingDeprecationWarning,
        )
        if filename in self._other_files:
            return self._other_files[filename]
        else:
            excpt_str = f'MVR file "{filename}" can not be found.'
            raise FlopyException(excpt_str)

    def get_mvt_file(self, filename):
        """
        Get a specified mvt file.

        Parameters
        ----------
            filename : str
                Name of mover transport file to get

        Returns
        --------
            mover transport package : MFPackage

        """
        warnings.warn(
            "get_mvt_file will be deprecated and will be removed in version "
            "3.3.6. Use get_file",
            PendingDeprecationWarning,
        )
        if filename in self._other_files:
            return self._other_files[filename]
        else:
            excpt_str = f'MVT file "{filename}" can not be found.'
            raise FlopyException(excpt_str)

    def get_gnc_file(self, filename):
        """
        Get a specified gnc file.

        Parameters
        ----------
            filename : str
                Name of gnc file to get

        Returns
        --------
            gnc package : MFPackage

        """
        warnings.warn(
            "get_gnc_file will be deprecated and will be removed in version "
            "3.3.6. Use get_file",
            PendingDeprecationWarning,
        )
        if filename in self._other_files:
            return self._other_files[filename]
        else:
            excpt_str = f'GNC file "{filename}" can not be found.'
            raise FlopyException(excpt_str)

    def remove_exchange_file(self, package):
        """
        Removes the exchange file "package". This is for internal flopy
        library use only.

        Parameters
        ----------
            package: MFPackage
                Exchange package to be removed

        """
        self._exchange_files[package.filename] = package
        try:
            exchange_recarray_data = self.name_file.exchanges.get_data()
        except MFDataException as mfde:
            message = (
                "An error occurred while retrieving exchange "
                "data from the simulation name file.  The error "
                "occurred while registering exchange file "
                f'"{package.filename}".'
            )
            raise MFDataException(
                mfdata_except=mfde,
                package=package._get_pname(),
                message=message,
            )
        remove_indices = []
        if exchange_recarray_data is not None:
            for index, exchange in zip(
                range(0, len(exchange_recarray_data)),
                exchange_recarray_data,
            ):
                if (
                    package.filename is not None
                    and exchange[1] == package.filename
                ):
                    remove_indices.append(index)
        if len(remove_indices) > 0:
            self.name_file.exchanges.set_data(
                np.delete(exchange_recarray_data, remove_indices)
            )

    def register_exchange_file(self, package):
        """
        Register an exchange package file with the simulation.  This is a
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
                    f'"{package.filename}".'
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
                                f'exgmnamea, exgmnameb): "{exgtype} '
                                f"{package.filename} {exgmnamea}"
                                f'{exgmnameb}".'
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
                    f'filename, exgmnamea, exgmnameb): "{exgtype} '
                    f'{package.filename} {exgmnamea} {exgmnameb}".'
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

    def _remove_package_by_type(self, package):
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
            package.package_type.lower()
            in mfstructure.MFStructure().flopy_dict["solution_packages"]
            and pname in self.package_name_dict
        ):
            if (
                self.simulation_data.verbosity_level.value
                >= VerbosityLevel.normal.value
            ):
                print(
                    "WARNING: Package with name "
                    f"{package.package_name.lower()} already exists.  "
                    "Replacing existing package."
                )
            self._remove_package(self.package_name_dict[pname])
        else:
            if (
                package.filename in self._other_files
                and self._other_files[package.filename] in self._packagelist
            ):
                # other package with same file name already exists.  remove old
                # package
                if (
                    self.simulation_data.verbosity_level.value
                    >= VerbosityLevel.normal.value
                ):
                    print(
                        f"WARNING: package with name {pname} already exists. "
                        "Replacing existing package."
                    )
                self._remove_package(self._other_files[package.filename])
                del self._other_files[package.filename]

    def register_package(
        self,
        package,
        add_to_package_list=True,
        set_package_name=True,
        set_package_filename=True,
    ):
        """
        Register a package file with the simulation.  This is a
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
        if set_package_filename:
            # set initial package filename
            base_name = os.path.basename(os.path.normpath(self.name))
            package._filename = f"{base_name}.{package.package_type}"

        package.container_type = [PackageContainerType.simulation]
        path = self._get_package_path(package)
        if add_to_package_list and package.package_type.lower != "nam":
            if (
                package.package_type.lower()
                not in mfstructure.MFStructure().flopy_dict[
                    "solution_packages"
                ]
            ):
                # all but solution packages get added here.  solution packages
                # are added during solution package registration
                self._remove_package_by_type(package)
                self._add_package(package, path)
        sln_dict = mfstructure.MFStructure().flopy_dict["solution_packages"]
        if package.package_type.lower() == "nam":
            if not package.internal_package:
                excpt_str = (
                    "Unable to register nam file.  Do not create your own nam "
                    "files.  Nam files are automatically created and managed "
                    "for you by FloPy."
                )
                print(excpt_str)
                raise FlopyException(excpt_str)
            return path, self.structure.name_file_struct_obj
        elif package.package_type.lower() == "tdis":
            self._tdis_file = package
            self._set_timing_block(package.quoted_filename)
            return (
                path,
                self.structure.package_struct_objs[
                    package.package_type.lower()
                ],
            )
        elif package.package_type.lower() in sln_dict:
            supported_packages = sln_dict[package.package_type.lower()]
            # default behavior is to register the solution package with the
            # first unregistered model
            unregistered_models = []
            for model_name, model in self._models.items():
                model_registered = self._is_in_solution_group(
                    model_name, 2, True
                )
                if not model_registered and (
                    model.model_type in supported_packages
                    or "*" in supported_packages
                ):
                    unregistered_models.append(model_name)
            if unregistered_models:
                self.register_solution_package(package, unregistered_models)
            else:
                self.register_solution_package(package, None)
            return (
                path,
                self.structure.package_struct_objs[
                    package.package_type.lower()
                ],
            )
        else:
            if package.filename not in self._other_files:
                self._other_files[package.filename] = package
            else:
                # auto generate a unique file name and register it
                file_name = MFFileMgmt.unique_file_name(
                    package.filename, self._other_files
                )
                package.filename = file_name
                self._other_files[file_name] = package

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

    def rename_model_namefile(self, model, new_namefile):
        """
        Rename a model's namefile.  For internal flopy library use only.

        Parameters
        ----------
           model : MFModel
               Model object whose namefile to rename
           new_namefile : str
               Name of the new namefile

        """
        # update simulation name file
        models = self.name_file.models.get_data()
        for mdl in models:
            path, name_file_name = os.path.split(mdl[1])
            if name_file_name == model.name_file.filename:
                mdl[1] = os.path.join(path, new_namefile)
        self.name_file.models.set_data(models)

    def register_model(self, model, model_type, model_name, model_namefile):
        """
        Add a model to the simulation. This is a call-back method made
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
            message = f'Invalid model type: "{model_type}".'
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

        if len(self._solution_files) > 0:
            # register model with first solution file found
            first_solution_key = next(iter(self._solution_files))
            self.register_solution_package(
                self._solution_files[first_solution_key], model_name
            )

        return self.structure.model_struct_objs[model_type]

    def get_ims_package(self, key):
        warnings.warn(
            "get_ims_package() has been deprecated and will be "
            "removed in version 3.3.7. Use "
            "get_solution_package() instead.",
            DeprecationWarning,
        )
        return self.get_solution_package(key)

    def get_solution_package(self, key):
        """
        Get the solution package with the specified `key`.

        Parameters
        ----------
            key : str
                solution package file name

        Returns
        --------
            solution_package : MFPackage

        """
        if key in self._solution_files:
            return self._solution_files[key]
        return None

    def remove_model(self, model_name):
        """
        Remove model with name `model_name` from the simulation

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
        """
        Checks the validity of the solution and all of its models and
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

        # solution files valid
        for solution_file in self._solution_files.values():
            if not solution_file.is_valid():
                return False

        # a model exists
        if not self._models:
            return False

        # models valid
        for key in self._models:
            if not self._models[key].is_valid():
                return False

        # each model has a solution file

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

    def _update_solution_group(self, solution_file, new_name=None):
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
                if record.slnfname == solution_file:
                    if new_name is not None:
                        record.slnfname = new_name
                        new_array.append(tuple(record))
                    else:
                        continue
                else:
                    new_array.append(record)

            if not new_array:
                new_array = None

            solution_recarray.set_data(new_array, solution_group_num[0])

    def _remove_from_all_solution_groups(self, modelname):
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
            new_array = ["no_check"]
            for index, record in enumerate(rec_array):
                new_record = []
                new_record.append(record[0])
                new_record.append(record[1])
                for item in list(record)[2:]:
                    if item is not None and item.lower() != modelname.lower():
                        new_record.append(item)
                new_array.append(tuple(new_record))
            solution_recarray.set_data(new_array, solution_group_num[0])

    def _append_to_solution_group(self, solution_file, new_models):
        # clear models out of solution groups
        if new_models is not None:
            for model in new_models:
                self._remove_from_all_solution_groups(model)

        # append models to solution_file
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
                    if (
                        record[1] == solution_file or item not in new_models
                    ) and item is not None:
                        new_record.append(item)
                        if index > 1 and item is not None:
                            rec_model_dict[item.lower()] = 1

                if record[1] == solution_file:
                    for model in new_models:
                        if model.lower() not in rec_model_dict:
                            new_record.append(model)

                new_array.append(tuple(new_record))
            solution_recarray.set_data(new_array, solution_group_num[0])

    def _replace_solution_in_solution_group(self, item, index, new_item):
        solution_recarray = self.name_file.solutiongroup
        for solution_group_num in solution_recarray.get_active_key_list():
            try:
                rec_array = solution_recarray.get_data(solution_group_num[0])
            except MFDataException as mfde:
                message = (
                    "An error occurred while getting solution group"
                    '"{}" from the simulation name file.  The error '
                    'occurred while replacing solution file "{}" with "{}"'
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

    def _is_in_solution_group(self, item, index, any_idx_after=False):
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
                    if any_idx_after:
                        for idx in range(index, len(rec_item)):
                            if rec_item[idx] == item:
                                return True
                    else:
                        if rec_item[index] == item:
                            return True
        return False

    def plot(
        self,
        model_list: Optional[Union[str, List[str]]] = None,
        SelPackList=None,
        **kwargs,
    ):
        """
        Plot simulation or models.

        Method to plot a whole simulation or a series of models
        that are part of a simulation.

        Parameters
        ----------
            model_list: list, optional
                List of model names to plot, if none all models will be plotted
            SelPackList: list, optional
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
        from ...plot.plotutil import PlotUtilities

        axes = PlotUtilities._plot_simulation_helper(
            self, model_list=model_list, SelPackList=SelPackList, **kwargs
        )
        return axes
