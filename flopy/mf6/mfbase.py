import glob
import importlib
import inspect, sys, traceback
import os, copy
from collections import OrderedDict
from collections.abc import Iterable
from shutil import copyfile
from enum import Enum


# internal handled exceptions
class MFInvalidTransientBlockHeaderException(Exception):
    """
    Exception related to parsing a transient block header
    """

    def __init__(self, error):
        Exception.__init__(
            self, "MFInvalidTransientBlockHeaderException: {}".format(error)
        )


class ReadAsArraysException(Exception):
    """
    Attempted to load ReadAsArrays package as non-ReadAsArraysPackage
    """

    def __init__(self, error):
        Exception.__init__(self, "ReadAsArraysException: {}".format(error))


# external exceptions for users
class FlopyException(Exception):
    """
    General Flopy Exception
    """

    def __init__(self, error, location=""):
        self.message = error
        Exception.__init__(
            self, "FlopyException: {} ({})".format(error, location)
        )


class StructException(Exception):
    """
    Exception related to the package file structure
    """

    def __init__(self, error, location):
        self.message = error
        Exception.__init__(
            self, "StructException: {} ({})".format(error, location)
        )


class MFDataException(Exception):
    """
    Exception related to MODFLOW input/output data
    """

    def __init__(
        self,
        model=None,
        package=None,
        path=None,
        current_process=None,
        data_element=None,
        method_caught_in=None,
        org_type=None,
        org_value=None,
        org_traceback=None,
        message=None,
        debug=None,
        mfdata_except=None,
    ):
        if mfdata_except is not None and isinstance(
            mfdata_except, MFDataException
        ):
            # copy constructor - copying values from original exception
            self.model = mfdata_except.model
            self.package = mfdata_except.package
            self.current_process = mfdata_except.current_process
            self.data_element = mfdata_except.data_element
            self.path = mfdata_except.path
            self.messages = mfdata_except.messages
            self.debug = mfdata_except.debug
            self.method_caught_in = mfdata_except.method_caught_in
            self.org_type = mfdata_except.org_type
            self.org_value = mfdata_except.org_value
            self.org_traceback = mfdata_except.org_traceback
            self.org_tb_string = mfdata_except.org_tb_string
        else:
            self.messages = []
            if mfdata_except is not None and (
                isinstance(mfdata_except, StructException)
                or isinstance(mfdata_except, FlopyException)
            ):
                self.messages.append(mfdata_except.message)
            self.model = None
            self.package = None
            self.current_process = None
            self.data_element = None
            self.path = None
            self.debug = False
            self.method_caught_in = None
            self.org_type = None
            self.org_value = None
            self.org_traceback = None
            self.org_tb_string = None
        # override/assign any values that are not none
        if model is not None:
            self.model = model
        if package is not None:
            self.package = package
        if current_process is not None:
            self.current_process = current_process
        if data_element is not None:
            self.data_element = data_element
        if path is not None:
            self.path = path
        if message is not None:
            self.messages.append(message)
        if debug is not None:
            self.debug = debug
        if method_caught_in is not None:
            self.method_caught_in = method_caught_in
        if org_type is not None:
            self.org_type = org_type
        if org_value is not None:
            self.org_value = org_value
        if org_traceback is not None:
            self.org_traceback = org_traceback
        self.org_tb_string = traceback.format_exception(
            self.org_type, self.org_value, self.org_traceback
        )
        # build error string
        error_message_0 = "An error occurred in "
        if self.data_element is not None and self.data_element != "":
            error_message_1 = 'data element "{}"' " ".format(self.data_element)
        else:
            error_message_1 = ""
        if self.model is not None and self.model != "":
            error_message_2 = 'model "{}" '.format(self.model)
        else:
            error_message_2 = ""
        error_message_3 = 'package "{}".'.format(self.package)
        error_message_4 = (
            ' The error occurred while {} in the "{}" method'
            ".".format(self.current_process, self.method_caught_in)
        )
        if len(self.messages) > 0:
            error_message_5 = "\nAdditional Information:\n"
            for index, message in enumerate(self.messages):
                error_message_5 = "{}({}) {}\n".format(
                    error_message_5, index + 1, message
                )
        else:
            error_message_5 = ""
        error_message = "{}{}{}{}{}{}".format(
            error_message_0,
            error_message_1,
            error_message_2,
            error_message_3,
            error_message_4,
            error_message_5,
        )
        # if self.debug:
        #    tb_string = ''.join(self.org_tb_string)
        #    error_message = '{}\nCall Stack\n{}'.format(error_message,
        #                                                tb_string)
        Exception.__init__(self, error_message)


class VerbosityLevel(Enum):
    quiet = 1
    normal = 2
    verbose = 3


class PackageContainerType(Enum):
    simulation = 1
    model = 2
    package = 3


class ExtFileAction(Enum):
    copy_all = 1
    copy_none = 2
    copy_relative_paths = 3


class MFFilePath(object):
    def __init__(self, file_path, model_name):
        self.file_path = file_path
        self.model_name = {model_name: 0}

    def isabs(self):
        return os.path.isabs(self.file_path)


class MFFileMgmt(object):
    """
    Class containing MODFLOW path data

    Parameters
    ----------

    path : string
        path on disk to the simulation

    Attributes
    ----------

    sim_path : string
        path to the simulation
    model_relative_path : OrderedDict
        dictionary of relative paths to each model folder

    Methods
    -------

    get_model_path : (key : string) : string
        returns the model working path for the model key
    set_sim_path : string
        sets the simulation working path

    """

    def __init__(self, path):
        self._sim_path = ""
        self.set_sim_path(path)

        # keys:fully pathed filenames, vals:FilePath instances
        self.existing_file_dict = {}
        # keys:filenames,vals:instance name

        self.model_relative_path = OrderedDict()

        self._last_loaded_sim_path = None
        self._last_loaded_model_relative_path = OrderedDict()

    def copy_files(self, copy_relative_only=True):
        num_files_copied = 0
        if self._last_loaded_sim_path is not None:
            for mffile_path in self.existing_file_dict.values():
                # resolve previous simulation path.  if mf6 changes
                # so that paths are relative to the model folder, then
                # this call should have "model_name" instead of "None"
                path_old = self.resolve_path(mffile_path, None, True)
                if os.path.isfile(path_old) and (
                    not mffile_path.isabs() or not copy_relative_only
                ):
                    # change "None" to "model_name" as above if mf6
                    # supports model relative paths
                    path_new = self.resolve_path(mffile_path, None)
                    if path_old != path_new:
                        new_folders = os.path.split(path_new)[0]
                        if not os.path.exists(new_folders):
                            os.makedirs(new_folders)
                        try:
                            copyfile(path_old, path_new)
                        except:
                            type_, value_, traceback_ = sys.exc_info()
                            raise MFDataException(
                                self.structure.get_model(),
                                self.structure.get_package(),
                                self._path,
                                "appending data",
                                self.structure.name,
                                inspect.stack()[0][3],
                                type_,
                                value_,
                                traceback_,
                                None,
                                self._simulation_data.debug,
                            )

                        num_files_copied += 1
        return num_files_copied

    def get_updated_path(
        self, external_file_path, model_name, ext_file_action
    ):
        external_file_path = self.string_to_file_path(external_file_path)
        if ext_file_action == ExtFileAction.copy_all:
            if os.path.isabs(external_file_path):
                # move file path to local model or simulation path
                file_name = os.path.split(external_file_path)[1]
                if model_name:
                    return os.path.join(
                        self.get_model_path(model_name), file_name
                    )
                else:
                    return os.path.join(self.get_sim_path(), file_name)
            else:
                return external_file_path
        elif ext_file_action == ExtFileAction.copy_relative_paths:
            return external_file_path
        elif ext_file_action == ExtFileAction.copy_none:
            if os.path.isabs(external_file_path):
                return external_file_path
            else:
                return os.path.join(
                    self._build_relative_path(model_name), external_file_path
                )
        else:
            return None

    def _build_relative_path(self, model_name):
        old_abs_path = self.resolve_path("", model_name, True)
        current_abs_path = self.resolve_path("", model_name, False)
        return os.path.relpath(old_abs_path, current_abs_path)

    def strip_model_relative_path(self, model_name, path):
        if model_name in self.model_relative_path:
            model_rel_path = self.model_relative_path[model_name]
            new_path = None
            while path:
                path, leaf = os.path.split(path)
                if leaf != model_rel_path:
                    if new_path:
                        new_path = os.path.join(leaf, new_path)
                    else:
                        new_path = leaf
            return new_path

    @staticmethod
    def unique_file_name(file_name, lookup):
        num = 0
        while MFFileMgmt._build_file(file_name, num) in lookup:
            num += 1
        return MFFileMgmt._build_file(file_name, num)

    @staticmethod
    def _build_file(file_name, num):
        file, ext = os.path.splitext(file_name)
        if ext:
            return "{}_{}{}".format(file, num, ext)
        else:
            return "{}_{}".format(file, num)

    @staticmethod
    def string_to_file_path(fp_string):
        file_delimiters = ["/", "\\"]
        new_string = fp_string
        for delimiter in file_delimiters:
            arr_string = new_string.split(delimiter)
            if len(arr_string) > 1:
                if os.path.isabs(fp_string):
                    new_string = "{}{}{}".format(
                        arr_string[0], delimiter, arr_string[1]
                    )
                else:
                    new_string = os.path.join(arr_string[0], arr_string[1])
                if len(arr_string) > 2:
                    for path_piece in arr_string[2:]:
                        new_string = os.path.join(new_string, path_piece)
        return new_string

    def set_last_accessed_path(self):
        self._last_loaded_sim_path = self._sim_path
        self.set_last_accessed_model_path()

    def set_last_accessed_model_path(self):
        for key, item in self.model_relative_path.items():
            self._last_loaded_model_relative_path[key] = copy.deepcopy(item)

    def get_model_path(self, key, last_loaded_path=False):
        if last_loaded_path:
            return os.path.join(
                self._last_loaded_sim_path,
                self._last_loaded_model_relative_path[key],
            )
        else:
            if key in self.model_relative_path:
                return os.path.join(
                    self._sim_path, self.model_relative_path[key]
                )
            else:
                return self._sim_path

    def get_sim_path(self, last_loaded_path=False):
        if last_loaded_path:
            return self._last_loaded_sim_path
        else:
            return self._sim_path

    def add_ext_file(self, file_path, model_name):
        if file_path in self.existing_file_dict:
            if model_name not in self.existing_file_dict[file_path].model_name:
                self.existing_file_dict[file_path].model_name[model_name] = 0
        else:
            new_file_path = MFFilePath(file_path, model_name)
            self.existing_file_dict[file_path] = new_file_path

    def set_sim_path(self, path):
        """
        set the file path to the simulation files

        Parameters
        ----------
        path : string
            full path or relative path from working directory to
            simulation folder

        Returns
        -------

        Examples
        --------
        self.simulation_data.mfdata.set_sim_path()
        """

        # recalculate paths for everything
        # resolve path type
        path = self.string_to_file_path(path)
        if os.path.isabs(path):
            self._sim_path = path
        else:
            # assume path is relative to working directory
            self._sim_path = os.path.join(os.getcwd(), path)

    def resolve_path(
        self, path, model_name, last_loaded_path=False, move_abs_paths=False
    ):
        if isinstance(path, MFFilePath):
            file_path = path.file_path
        else:
            file_path = path

        if os.path.isabs(file_path):
            # path is an absolute path
            if move_abs_paths:
                if model_name is None:
                    return self.get_sim_path(last_loaded_path)
                else:
                    return self.get_model_path(model_name, last_loaded_path)
            else:
                return file_path
        else:
            # path is a relative path
            if model_name is None:
                return os.path.join(
                    self.get_sim_path(last_loaded_path), file_path
                )
            else:
                return os.path.join(
                    self.get_model_path(model_name, last_loaded_path),
                    file_path,
                )


class PackageContainer(object):
    """
    Base class for any class containing packages.

    Parameters
    ----------
    simulation_data : SimulationData
        the simulation's SimulationData object
    name : string
        name of the package container object

    Attributes
    ----------
    _packagelist : list
        packages contained in the package container
    package_type_dict : dictionary
        dictionary of packages by package type
    package_name_dict : dictionary
        dictionary of packages by package name
    package_key_dict : dictionary
        dictionary of packages by package key

    Methods
    -------
    package_factory : (package_type : string, model_type : string) :
      MFPackage subclass
        Static method that returns the appropriate package type object based
        on the package_type and model_type strings
    get_package : (name : string) : MFPackage or [MfPackage]
        finds a package by package name, package key, package type, or partial
        package name. returns either a single package, a list of packages,
        or None
    register_package : (package : MFPackage) : (tuple, PackageStructure)
        base class method for package registration
    """

    def __init__(self, simulation_data, name):
        self.type = "PackageContainer"
        self.simulation_data = simulation_data
        self.name = name
        self._packagelist = []
        self.package_type_dict = {}
        self.package_name_dict = {}
        self.package_key_dict = {}

    @staticmethod
    def package_factory(package_type, model_type):
        package_abbr = "{}{}".format(model_type, package_type)
        package_utl_abbr = "utl{}".format(package_type)
        package_list = []
        # iterate through python files
        package_file_paths = PackageContainer.get_package_file_paths()
        for package_file_path in package_file_paths:
            module = PackageContainer.get_module(package_file_path)
            if module is not None:
                # iterate imported items
                for item in dir(module):
                    value = PackageContainer.get_module_val(
                        module, item, "package_abbr"
                    )
                    if value is not None:
                        abbr = value.package_abbr
                        if package_type is None:
                            # don't store packages "group" classes
                            if len(abbr) <= 8 or abbr[-8:] != "packages":
                                package_list.append(value)
                        else:
                            # check package type
                            if (
                                value.package_abbr == package_abbr
                                or value.package_abbr == package_utl_abbr
                            ):
                                return value
        if package_type is None:
            return package_list
        else:
            return None

    @staticmethod
    def model_factory(model_type):
        package_file_paths = PackageContainer.get_package_file_paths()
        for package_file_path in package_file_paths:
            module = PackageContainer.get_module(package_file_path)
            if module is not None:
                # iterate imported items
                for item in dir(module):
                    value = PackageContainer.get_module_val(
                        module, item, "model_type"
                    )
                    if value is not None and value.model_type == model_type:
                        return value
        return None

    @staticmethod
    def get_module_val(module, item, attrb):
        value = getattr(module, item)
        # verify this is a class
        if (
            not value
            or not inspect.isclass(value)
            or not hasattr(value, attrb)
        ):
            return None
        return value

    @staticmethod
    def get_module(package_file_path):
        package_file_name = os.path.basename(package_file_path)
        module_path = os.path.splitext(package_file_name)[0]
        module_name = "{}{}{}".format(
            "Modflow", module_path[2].upper(), module_path[3:]
        )
        if module_name.startswith("__"):
            return None

        # import
        return importlib.import_module(
            "flopy.mf6.modflow.{}".format(module_path)
        )

    @staticmethod
    def get_package_file_paths():
        base_path = os.path.split(os.path.realpath(__file__))[0]
        package_path = os.path.join(base_path, "modflow")
        return glob.glob(os.path.join(package_path, "*.py"))

    @property
    def package_dict(self):
        return self.package_name_dict.copy()

    @property
    def package_names(self):
        return list(self.package_name_dict.keys())

    def _add_package(self, package, path):
        # put in packages list and update lookup dictionaries
        self._packagelist.append(package)
        if package.package_name is not None:
            self.package_name_dict[package.package_name.lower()] = package
        self.package_key_dict[path[-1].lower()] = package
        if package.package_type not in self.package_type_dict:
            self.package_type_dict[package.package_type.lower()] = []
        self.package_type_dict[package.package_type.lower()].append(package)

    def _remove_package(self, package):
        self._packagelist.remove(package)
        if (
            package.package_name is not None
            and package.package_name.lower() in self.package_name_dict
        ):
            del self.package_name_dict[package.package_name.lower()]
        del self.package_key_dict[package.path[-1].lower()]
        package_list = self.package_type_dict[package.package_type.lower()]
        package_list.remove(package)
        if len(package_list) == 0:
            del self.package_type_dict[package.package_type.lower()]

        # collect keys of items to be removed from main dictionary
        items_to_remove = []
        for key in self.simulation_data.mfdata:
            is_subkey = True
            for pitem, ditem in zip(package.path, key):
                if pitem != ditem:
                    is_subkey = False
                    break
            if is_subkey:
                items_to_remove.append(key)

        # remove items from main dictionary
        for key in items_to_remove:
            del self.simulation_data.mfdata[key]

    def get_package(self, name=None):
        """
        Get a package.

        Parameters
        ----------
        name : str
            Name of the package, 'RIV', 'LPF', etc.

        Returns
        -------
        pp : Package object

        """
        if name is None:
            return self._packagelist[:]

        # search for full package name
        if name.lower() in self.package_name_dict:
            return self.package_name_dict[name.lower()]

        # search for package type
        if name.lower() in self.package_type_dict:
            if len(self.package_type_dict[name.lower()]) == 0:
                return None
            elif len(self.package_type_dict[name.lower()]) == 1:
                return self.package_type_dict[name.lower()][0]
            else:
                return self.package_type_dict[name.lower()]

        # search for package key
        if name.lower() in self.package_key_dict:
            return self.package_key_dict[name.lower()]

        # search for partial and case-insensitive package name
        for pp in self._packagelist:
            if pp.package_name is not None:
                # get first package of the type requested
                package_name = pp.package_name.lower()
                if len(package_name) > len(name):
                    package_name = package_name[0 : len(name)]
                if package_name.lower() == name.lower():
                    return pp

        return None

    def register_package(self, package):
        path = (package.package_name,)
        return (path, None)

    @staticmethod
    def _load_only_dict(load_only):
        if load_only is None:
            return None
        if isinstance(load_only, dict):
            return load_only
        if not isinstance(load_only, Iterable):
            raise FlopyException(
                "load_only must be iterable or None. "
                'load_only value of "{}" is '
                "invalid".format(load_only)
            )
        load_only_dict = {}
        for item in load_only:
            load_only_dict[item.lower()] = True
        return load_only_dict

    @staticmethod
    def _in_pkg_list(pkg_list, pkg_type, pkg_name):
        if pkg_type is not None:
            pkg_type = pkg_type.lower()
        if pkg_name is not None:
            pkg_name = pkg_name.lower()
        if pkg_type in pkg_list or pkg_name in pkg_list:
            return True

        # split to make cases like "gwf6-gwf6" easier to process
        pkg_type = pkg_type.split("-")
        try:
            # if there is a number on the end of the package try
            # excluding it
            int(pkg_type[0][-1])
            for key in pkg_list.keys():
                key = key.split("-")
                if len(key) == len(pkg_type):
                    matches = True
                    for key_item, pkg_item in zip(key, pkg_type):
                        if pkg_item[0:-1] != key_item and pkg_item != key_item:
                            matches = False
                    if matches:
                        return True
        except ValueError:
            return False
        return False
