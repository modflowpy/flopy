""" Base classes for Modflow 6 """
import copy
import inspect
import os
import sys
import traceback
from collections.abc import Iterable
from enum import Enum
from shutil import copyfile


# internal handled exceptions
class MFInvalidTransientBlockHeaderException(Exception):
    """
    Exception occurs when parsing a transient block header
    """


class ReadAsArraysException(Exception):
    """
    Exception occurs when loading ReadAsArrays package as non-ReadAsArrays
    package.
    """


# external exceptions for users
class FlopyException(Exception):
    """
    General FloPy exception
    """

    def __init__(self, error, location=""):
        self.message = error
        super().__init__(f"{error} ({location})")


class StructException(Exception):
    """
    Exception with the package file structure
    """

    def __init__(self, error, location):
        self.message = error
        super().__init__(f"{error} ({location})")


class MFDataException(Exception):
    """
    Exception with MODFLOW data.  Exception includes detailed error
    information.
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
        error_message = "An error occurred in "
        if self.data_element is not None and self.data_element != "":
            error_message += f'data element "{self.data_element}" '
        if self.model is not None and self.model != "":
            error_message += f'model "{self.model}" '
        error_message += (
            f'package "{self.package}". The error occurred while '
            f'{self.current_process} in the "{self.method_caught_in}" method.'
        )
        if len(self.messages) > 0:
            error_message += "\nAdditional Information:\n"
            error_message += "\n".join(
                f"({idx}) {msg}" for (idx, msg) in enumerate(self.messages, 1)
            )
        super().__init__(error_message)


class VerbosityLevel(Enum):
    """Determines how much information FloPy writes to the console"""

    quiet = 1
    normal = 2
    verbose = 3


class PackageContainerType(Enum):
    """Determines whether a package container is a simulation, model, or
    package."""

    simulation = 1
    model = 2
    package = 3


class ExtFileAction(Enum):
    """Defines what to do with external files when the simulation or model's
    path change."""

    copy_all = 1
    copy_none = 2
    copy_relative_paths = 3


class MFFilePath:
    """Class that stores a single file path along with the associated model
    name."""

    def __init__(self, file_path, model_name):
        self.file_path = file_path
        self.model_name = {model_name: 0}

    def isabs(self):
        return os.path.isabs(self.file_path)


class MFFileMgmt:
    """
    Class containing MODFLOW path data

    Parameters
    ----------

    path : str
        Path on disk to the simulation

    Attributes
    ----------

    model_relative_path : dict
        Dictionary of relative paths to each model folder

    """

    def __init__(self, path, mfsim=None):
        self.simulation = mfsim
        self._sim_path = ""
        self.set_sim_path(path, True)

        # keys:fully pathed filenames, vals:FilePath instances
        self.existing_file_dict = {}
        # keys:filenames,vals:instance name

        self.model_relative_path = {}

        self._last_loaded_sim_path = None
        self._last_loaded_model_relative_path = {}

    def copy_files(self, copy_relative_only=True):
        """Copy files external to updated path.

        Parameters
        ----------
            copy_relative_only : bool
                Only copy files with relative paths.
        """
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
        """For internal FloPy use, not intended for end user."""
        return external_file_path

    def _build_relative_path(self, model_name):
        old_abs_path = self.resolve_path("", model_name, True)
        current_abs_path = self.resolve_path("", model_name, False)
        return os.path.relpath(old_abs_path, current_abs_path)

    def strip_model_relative_path(self, model_name, path):
        """Strip out the model relative path part of `path`.  For internal
        FloPy use, not intended for end user."""
        new_path = path
        if model_name in self.model_relative_path:
            model_rel_path = self.model_relative_path[model_name]
            if (
                model_rel_path is not None
                and len(model_rel_path) > 0
                and model_rel_path != "."
            ):
                model_rel_path_lst = model_rel_path.split(os.path.sep)
                path_lst = path.split(os.path.sep)
                new_path = ""
                for i, mrp in enumerate(model_rel_path_lst):
                    if i >= len(path_lst) or mrp != path_lst[i]:
                        new_path = os.path.join(new_path, path_lst[i])
                for rp in path_lst[len(model_rel_path_lst) :]:
                    new_path = os.path.join(new_path, rp)
        return new_path

    @staticmethod
    def unique_file_name(file_name, lookup):
        """Generate a unique file name.  For internal FloPy use, not intended
        for end user."""
        num = 0
        while MFFileMgmt._build_file(file_name, num) in lookup:
            num += 1
        return MFFileMgmt._build_file(file_name, num)

    @staticmethod
    def _build_file(file_name, num):
        file, ext = os.path.splitext(file_name)
        if ext:
            return f"{file}_{num}{ext}"
        else:
            return f"{file}_{num}"

    @staticmethod
    def string_to_file_path(fp_string):
        """Interpret string as a file path.  For internal FloPy use, not
        intended for end user."""
        file_delimiters = ["/", "\\"]
        new_string = fp_string
        for delimiter in file_delimiters:
            arr_string = new_string.split(delimiter)
            if len(arr_string) > 1:
                if os.path.isabs(fp_string):
                    if not arr_string[0] and not arr_string[1]:
                        new_string = f"{delimiter}{delimiter}"
                    else:
                        new_string = (
                            f"{arr_string[0]}{delimiter}{arr_string[1]}"
                        )
                else:
                    new_string = os.path.join(arr_string[0], arr_string[1])
                if len(arr_string) > 2:
                    for path_piece in arr_string[2:]:
                        new_string = os.path.join(new_string, path_piece)
        return new_string

    def set_last_accessed_path(self):
        """Set the last accessed simulation path to the current simulation
        path.  For internal FloPy use, not intended for end user."""
        self._last_loaded_sim_path = self._sim_path
        self.set_last_accessed_model_path()

    def set_last_accessed_model_path(self):
        """Set the last accessed model path to the current model path.
        For internal FloPy use, not intended for end user."""
        for key, item in self.model_relative_path.items():
            self._last_loaded_model_relative_path[key] = copy.deepcopy(item)

    def get_model_path(self, key, last_loaded_path=False):
        """Returns the model working path for the model `key`.

        Parameters
        ----------
        key : str
            Model name whose path flopy will retrieve
        last_loaded_path : bool
            Get the last path loaded by FloPy which may not be the most
            recent path.

        Returns
        -------
            model path : str

        """
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
        """Get the simulation path."""
        if last_loaded_path:
            return self._last_loaded_sim_path
        else:
            return self._sim_path

    def add_ext_file(self, file_path, model_name):
        """Add an external file to the path list.  For internal FloPy use, not
        intended for end user."""
        if file_path in self.existing_file_dict:
            if model_name not in self.existing_file_dict[file_path].model_name:
                self.existing_file_dict[file_path].model_name[model_name] = 0
        else:
            new_file_path = MFFilePath(file_path, model_name)
            self.existing_file_dict[file_path] = new_file_path

    def set_sim_path(self, path, internal_use=False):
        """
        Set the file path to the simulation files.  Internal use only,
        call MFSimulation's set_sim_path method instead.

        Parameters
        ----------
        path : str
            Full path or relative path from working directory to
            simulation folder

        Returns
        -------

        Examples
        --------
        self.simulation_data.mfdata.set_sim_path('sim_folder')
        """
        if not internal_use:
            print(
                "WARNING: MFFileMgt's set_sim_path has been deprecated.  "
                "Please use MFSimulation's set_sim_path in the future."
            )
            if self.simulation is not None:
                self.simulation.set_sim_path(path)
                return
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
        """Resolve a simulation or model path.  For internal FloPy use, not
        intended for end user."""
        if isinstance(path, MFFilePath):
            file_path = path.file_path
        else:
            file_path = path

        # remove quote characters from file path
        file_path = file_path.replace("'", "")
        file_path = file_path.replace('"', "")

        if os.path.isabs(file_path):
            # path is an absolute path
            if move_abs_paths:
                return self.get_sim_path(last_loaded_path)
            else:
                return file_path
        else:
            # path is a relative path
            return os.path.join(self.get_sim_path(last_loaded_path), file_path)


class PackageContainer:
    """
    Base class for any class containing packages.

    Parameters
    ----------
    simulation_data : SimulationData
        The simulation's SimulationData object
    name : str
        Name of the package container object

    Attributes
    ----------
    package_type_dict : dictionary
        Dictionary of packages by package type
    package_name_dict : dictionary
        Dictionary of packages by package name
    package_key_dict : dictionary
        Dictionary of packages by package key

    """

    modflow_packages = []
    packages_by_abbr = {}
    modflow_models = []
    models_by_type = {}

    def __init__(self, simulation_data, name):
        self.type = "PackageContainer"
        self.simulation_data = simulation_data
        self.name = name
        self._packagelist = []
        self.package_type_dict = {}
        self.package_name_dict = {}
        self.package_filename_dict = {}
        self.package_key_dict = {}

    @staticmethod
    def package_list():
        """Static method that returns the list of available packages.
        For internal FloPy use only, not intended for end users.

        Returns a list of MFPackage subclasses
        """
        # all packages except "group" classes
        package_list = []
        for abbr, package in sorted(PackageContainer.packages_by_abbr.items()):
            # don't store packages "group" classes
            if not abbr.endswith("packages"):
                package_list.append(package)
        return package_list

    @staticmethod
    def package_factory(package_type: str, model_type: str):
        """Static method that returns the appropriate package type object based
        on the package_type and model_type strings.  For internal FloPy use
        only, not intended for end users.

        Parameters
        ----------
            package_type : str
                Type of package to create
            model_type : str
                Type of model that package is a part of

        Returns
        -------
            package : MFPackage subclass

        """
        package_abbr = f"{model_type}{package_type}"
        factory = PackageContainer.packages_by_abbr.get(package_abbr)
        if factory is None:
            package_utl_abbr = "utl{}".format(package_type)
            factory = PackageContainer.packages_by_abbr.get(package_utl_abbr)
        return factory

    @staticmethod
    def model_factory(model_type):
        """Static method that returns the appropriate model type object based
        on the model_type string. For internal FloPy use only, not intended
        for end users.

        Parameters
        ----------
            model_type : str
                Type of model that package is a part of

        Returns
        -------
            model : MFModel subclass

        """
        return PackageContainer.models_by_type.get(model_type)

    @staticmethod
    def get_module_val(module, item, attrb):
        """Static method that returns a python class module value.  For
        internal FloPy use only, not intended for end users."""
        value = getattr(module, item)
        # verify this is a class
        if (
            not value
            or not inspect.isclass(value)
            or not hasattr(value, attrb)
        ):
            return None
        return value

    @property
    def package_dict(self):
        """Returns a copy of the package name dictionary."""
        return self.package_name_dict.copy()

    @property
    def package_names(self):
        """Returns a list of package names."""
        return list(self.package_name_dict.keys())

    def _add_package(self, package, path):
        # put in packages list and update lookup dictionaries
        self._packagelist.append(package)
        if package.package_name is not None:
            self.package_name_dict[package.package_name.lower()] = package
        if package.filename is not None:
            self.package_filename_dict[package.filename.lower()] = package
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
        if (
            package.filename is not None
            and package.filename.lower() in self.package_filename_dict
        ):
            del self.package_filename_dict[package.filename.lower()]
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

    def _rename_package(self, package, new_name):
        # fix package_name_dict key
        if (
            package.package_name is not None
            and package.package_name.lower() in self.package_name_dict
        ):
            del self.package_name_dict[package.package_name.lower()]
        self.package_name_dict[new_name.lower()] = package
        # fix package_key_dict key
        new_package_path = package.path[:-1] + (new_name,)
        del self.package_key_dict[package.path[-1].lower()]
        self.package_key_dict[new_package_path.lower()] = package
        # get keys to fix in main dictionary
        main_dict = self.simulation_data.mfdata
        items_to_fix = []
        for key in main_dict:
            is_subkey = True
            for pitem, ditem in zip(package.path, key):
                if pitem != ditem:
                    is_subkey = False
                    break
            if is_subkey:
                items_to_fix.append(key)

        # fix keys in main dictionary
        for key in items_to_fix:
            new_key = (
                package.path[:-1] + (new_name,) + key[len(package.path) - 1 :]
            )
            main_dict[new_key] = main_dict.pop(key)

    def get_package(self, name=None):
        """
        Finds a package by package name, package key, package type, or partial
        package name. returns either a single package, a list of packages,
        or None.

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

        # search for file name
        if name.lower() in self.package_filename_dict:
            return self.package_filename_dict[name.lower()]

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
        """Base method for registering a package.  Should be overridden."""
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
